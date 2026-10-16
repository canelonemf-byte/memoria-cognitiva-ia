#!/usr/bin/env python3
r"""Ingesta un documento en la memoria cognitiva, dividido en fragmentos.

Uso:
    python doc_chunker.py <archivo> <tema_documento> [--importance 0.6]
        [--max-length 1000] [--overlap 120] [--dry-run] [--force]

- .txt: párrafos (separados por líneas en blanco).
- .md:  además corta por encabezados (#).
- .tex: solo el cuerpo (tras \begin{document}); corta por \section/\subsection
        y mantiene enteros los entornos \begin{..}...\end{..}, aunque superen
        --max-length. Un entorno *balanceado* dentro de un párrafo largo no
        vuelve atómico el bloque: ese párrafo se parte por frases.
- Un párrafo mayor que --max-length se parte por frases; al partirlo, el
  fragmento siguiente repite la última frase si cabe en --overlap caracteres
  (solape por frase: el solape real está entre 0 y --overlap caracteres).
- Los fragmentos ya enviados (mismo tema + mismo texto) se omiten gracias a un
  manifiesto local; --force los reenvía. Si editas el documento, los fragmentos
  antiguos siguen en la memoria: este script solo envía ADD.
- Reintentos: hasta 3 intentos con espera creciente ante 429/5xx o errores de
  conexión; un 4xx definitivo falla el fragmento sin reintentar.
- Codificación: UTF-8; si falla, se intenta latin-1.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 15          # segundos por petición
MAX_ATTEMPTS = 3      # intentos por fragmento ante errores transitorios
RETRY_WAIT = 1.5      # segundos base de espera entre intentos (crece linealmente)
USER_AGENT = "doc-chunker/1.1 (Unified Cognitive Stack)"
CHUNKER_VERSION = "1.1"

# Fuente única de verdad: PIN_IMPORTANCE viene de memory_decay.py si está
# disponible en el mismo directorio/repo; si no, se usa el valor por defecto.
try:
    from memory_decay import PIN_IMPORTANCE as _PIN
except ImportError:  # memory_decay.py no accesible (p. ej., script copiado aislado)
    _PIN = 0.95
PIN_IMPORTANCE = _PIN

MANIFEST = Path(os.getenv("CHUNKER_MANIFEST",
                          Path(__file__).with_name(".doc_chunker_manifest.json")))

_KIND_BY_SUFFIX = {".md": "md", ".markdown": "md", ".tex": "tex"}
_HEADING = {
    "md": re.compile(r"(?m)^(?=#{1,6}\s)"),
    "tex": re.compile(r"(?m)^(?=\\(?:chapter|section|subsection|subsubsection)\*?[\[{])"),
}
_BEGIN = re.compile(r"\\begin\{(?!document\})[^}]+\}")
_END = re.compile(r"\\end\{(?!document\})[^}]+\}")
_COMMENT = re.compile(r"(?<!\\)%.*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_ENV_LIMIT = 6000  # si un entorno no se cierra antes de esto, se deja de agrupar
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


# --------------------------------------------------------------------------- #
# División en fragmentos
# --------------------------------------------------------------------------- #
def _tex_body(text: str) -> str:
    start = re.search(r"\\begin\{document\}", text)
    if start:
        text = text[start.end():]
    end = re.search(r"\\end\{document\}", text)
    if end:
        text = text[:end.start()]
    return text


def _env_depth(text: str) -> int:
    r"""\begin minus \end (ignorando comentarios %); > 0 => entorno abierto sin cerrar."""
    clean = _COMMENT.sub("", text)
    return len(_BEGIN.findall(clean)) - len(_END.findall(clean))


def _merge_environments(paragraphs: list[str]) -> list[str]:
    r"""Une párrafos mientras haya un \begin{..} sin su \end{..} (no parte tablas ni listas)."""
    merged, buf, depth = [], [], 0
    for p in paragraphs:
        depth += _env_depth(p)
        buf.append(p)
        if depth <= 0:
            merged.append("\n\n".join(buf))
            buf, depth = [], 0
        elif sum(map(len, buf)) > _ENV_LIMIT:  # entorno sin cerrar: no tragarse el documento
            merged.extend(buf)
            buf, depth = [], 0
    if buf:
        merged.append("\n\n".join(buf))
    return merged


def _split_oversized(text: str, max_length: int, overlap: int) -> list[str]:
    """Parte un bloque largo por frases; una frase gigante se corta en el último espacio.

    El fragmento siguiente repite la última frase completa del anterior si cabe
    en `overlap` caracteres (solape por frase, no por caracteres exactos).
    """
    pieces: list[str] = []
    cur, prev = "", ""
    for sent in _SENTENCE_END.split(text):
        sent = sent.strip()
        while len(sent) > max_length:
            if cur:
                pieces.append(cur)
                cur, prev = "", ""
            cut = sent.rfind(" ", 0, max_length)
            cut = cut if cut > 0 else max_length
            pieces.append(sent[:cut].rstrip())
            sent = sent[cut:].lstrip()
        if not sent:
            continue
        if cur and len(cur) + 1 + len(sent) > max_length:
            pieces.append(cur)
            carry = prev if prev and len(prev) <= overlap else ""
            cur = f"{carry} {sent}" if carry and len(carry) + 1 + len(sent) <= max_length else sent
        else:
            cur = f"{cur} {sent}" if cur else sent
        prev = sent
    if cur:
        pieces.append(cur)
    return pieces


def _pack(blocks: list[str], max_length: int, overlap: int, atomic_envs: bool) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for block in blocks:
        if len(block) > max_length:
            if cur:
                chunks.append(cur)
                cur = ""
            # Atómico solo si el bloque tiene un entorno ABIERTO sin cerrar (lo
            # agrupó _merge_environments). Un entorno balanceado dentro de un
            # párrafo largo no impide partirlo por frases.
            if atomic_envs and _env_depth(block) > 0:
                chunks.append(block)  # entorno LaTeX completo, aunque sea largo
            else:
                chunks.extend(_split_oversized(block, max_length, overlap))
        elif cur and len(cur) + 2 + len(block) > max_length:
            chunks.append(cur)
            cur = block
        else:
            cur = f"{cur}\n\n{block}" if cur else block
    if cur:
        chunks.append(cur)
    return chunks


def chunk_text(text: str, max_length: int = 1000, overlap: int = 120, kind: str = "") -> list[str]:
    """Divide `text` en fragmentos de hasta `max_length` caracteres. kind: "", "md" o "tex"."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if kind == "tex":
        text = _tex_body(text)
    sections = _HEADING[kind].split(text) if kind in _HEADING else [text]

    chunks: list[str] = []
    for section in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        if kind == "tex":
            paragraphs = _merge_environments(paragraphs)
        chunks.extend(_pack(paragraphs, max_length, overlap, atomic_envs=(kind == "tex")))
    return chunks


# --------------------------------------------------------------------------- #
# Manifiesto local (evita reenviar lo ya ingerido)
# --------------------------------------------------------------------------- #
def _fingerprint(subject_prefix: str, chunk: str) -> str:
    return hashlib.sha256(f"{subject_prefix}\n{chunk}".encode("utf-8")).hexdigest()[:16]


def _load_manifest() -> dict:
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_manifest(data: dict) -> None:
    try:
        MANIFEST.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")), encoding="utf-8")
    except OSError as e:
        print(f"⚠️  No se pudo guardar el manifiesto ({MANIFEST}): {e}")


# --------------------------------------------------------------------------- #
# Lectura tolerante de codificación
# --------------------------------------------------------------------------- #
def _read_document(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️  No es UTF-8; se intenta latin-1.")
        return path.read_text(encoding="latin-1")


# --------------------------------------------------------------------------- #
# Ingesta con reintentos
# --------------------------------------------------------------------------- #
def _post_with_retries(session: requests.Session, url: str, payload: dict,
                       attempts: int = MAX_ATTEMPTS) -> requests.Response:
    """POST con reintentos ante errores transitorios; los 4xx definitivos fallan enseguida."""
    last_exc: requests.RequestException | None = None
    for attempt in range(1, attempts + 1):
        try:
            res = session.post(url, json=payload, timeout=TIMEOUT)
            res.raise_for_status()
            return res
        except requests.ConnectionError as e:
            last_exc = e
            if attempt < attempts:
                time.sleep(RETRY_WAIT * attempt)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in _TRANSIENT_STATUS and attempt < attempts:
                time.sleep(RETRY_WAIT * attempt)
                continue
            raise  # 4xx definitivo u otro error HTTP: no reintentar
    assert last_exc is not None
    raise last_exc


def _reason(e: requests.RequestException) -> str:
    resp = getattr(e, "response", None)
    if resp is not None:
        return f"{resp.status_code} {resp.text[:200]}"
    return str(e)


def _keywords_for(subject_prefix: str, chunk: str, limit: int = 8) -> list[str]:
    """Palabras clave derivadas del tema y del inicio del fragmento (determinista)."""
    kws: list[str] = []
    for w in re.findall(r"\w+", subject_prefix, flags=re.UNICODE):
        if len(w) > 2 and w.lower() not in kws:
            kws.append(w.lower())
    for w in re.findall(r"\w+", chunk, flags=re.UNICODE):
        wl = w.lower()
        if len(wl) > 3 and wl not in kws:
            kws.append(wl)
        if len(kws) >= limit:
            break
    return kws[:limit]


def ingest_document(filepath: str, subject_prefix: str, importance: float = 0.6,
                    max_length: int = 1000, overlap: int = 120,
                    dry_run: bool = False, force: bool = False) -> int:
    """Ingesta el documento. Devuelve cuántos fragmentos fallaron o quedaron sin enviar."""
    path = Path(filepath)
    kind = _KIND_BY_SUFFIX.get(path.suffix.lower(), "")
    chunks = chunk_text(_read_document(path), max_length, overlap, kind)
    print(f"Documento dividido en {len(chunks)} fragmentos.")

    if dry_run:
        for i, chunk in enumerate(chunks, 1):
            vista = " ".join(chunk.split())[:70]
            aviso = "  (bloque indivisible)" if len(chunk) > max_length else ""
            print(f"{i:>4} | {len(chunk):>5} car. | {vista}{aviso}")
        return 0

    manifest = _load_manifest()
    ok = skipped = failed = 0
    try:
        with requests.Session() as session:
            session.headers.update({"User-Agent": USER_AGENT})
            for i, chunk in enumerate(chunks, 1):
                key = _fingerprint(subject_prefix, chunk)
                if key in manifest and not force:
                    skipped += 1
                    print(f"⏭️  Fragmento {i} ya ingerido, se omite.")
                    continue

                subject = f"{subject_prefix} - Parte {i}"
                payload = {
                    "action": "ADD",
                    "memory_note": {
                        "subject": subject,
                        "fact_contents": chunk,
                        "keywords": _keywords_for(subject_prefix, chunk),
                        "importance_score": importance,
                    },
                }
                try:
                    _post_with_retries(session, f"{API_URL}/ingest", payload)
                except requests.ConnectionError:
                    failed += len(chunks) - i + 1
                    print(f"❌ No se pudo conectar con la API en {API_URL}. "
                          "Arranca el servidor o ajusta API_URL; se aborta la ingesta.")
                    break
                except requests.RequestException as e:
                    failed += 1
                    print(f"❌ Error en fragmento {i}: {_reason(e)}")
                    continue

                manifest[key] = subject
                ok += 1
                print(f"✅ Fragmento {i} ingerido exitosamente.")
    finally:
        if ok:
            _save_manifest(manifest)

    print(f"Resumen: {ok} ingeridos, {skipped} omitidos, {failed} con error o sin enviar.")
    return failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Divide un documento (.txt, .md, .tex) e ingesta cada fragmento en la memoria.")
    ap.add_argument("archivo")
    ap.add_argument("tema_documento", help="prefijo del sujeto: '<tema> - Parte N'")
    ap.add_argument("--importance", type=float, default=0.6,
                    help=f"importancia 0-1 (desde {PIN_IMPORTANCE:g} la memoria queda fijada). "
                         "Por defecto 0.6")
    ap.add_argument("--max-length", type=int, default=1000, help="caracteres por fragmento")
    ap.add_argument("--overlap", type=int, default=120,
                    help="solape máximo (caracteres) al partir párrafos largos; el solape real "
                         "es por frase completa, entre 0 y este valor. 0 = sin solape")
    ap.add_argument("--dry-run", action="store_true", help="muestra los fragmentos sin enviar nada")
    ap.add_argument("--force", action="store_true", help="reenvía aunque ya consten como ingeridos")
    args = ap.parse_args(argv)

    if not 0.0 <= args.importance <= 1.0:
        ap.error("--importance debe estar entre 0 y 1")
    if args.max_length < 100:
        ap.error("--max-length debe ser al menos 100")
    if not 0 <= args.overlap < args.max_length:
        ap.error("--overlap debe estar entre 0 y --max-length")

    try:
        failed = ingest_document(args.archivo, args.tema_documento, args.importance,
                                 args.max_length, args.overlap, args.dry_run, args.force)
    except (OSError, UnicodeDecodeError) as e:
        print(f"❌ No se pudo leer '{args.archivo}' (prueba a convertirlo a UTF-8): {e}",
              file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
