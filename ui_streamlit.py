"""UI Streamlit del Unified Cognitive Stack.

Arranque:  streamlit run ui_streamlit.py
API:       variable de entorno API_URL (por defecto http://localhost:8000)
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TOP_K = 5
TIMEOUT_CHAT = 60      # segundos; el LLM puede tardar
TIMEOUT_INGEST = 15    # segundos
PIN_IMPORTANCE = 0.95  # igual que memory_decay.PIN_IMPORTANCE: desde aquí la memoria no se olvida

st.set_page_config(page_title="Cognitive Stack UI", layout="wide")
st.title("🧠 Unified Cognitive Stack")


class ApiError(Exception):
    """Error de la API con un mensaje listo para mostrar al usuario."""


def api_post(path: str, payload: dict, timeout: int) -> requests.Response:
    url = f"{API_URL}{path}"
    try:
        res = requests.post(url, json=payload, timeout=timeout)
        res.raise_for_status()
        return res
    except requests.Timeout:
        raise ApiError(f"La API no respondió en {timeout} s. "
                       "Inténtalo de nuevo o revisa el servidor.") from None
    except requests.ConnectionError:
        raise ApiError(f"No se pudo conectar con la API en {API_URL}. "
                       "Arranca el servidor o ajusta API_URL.") from None
    except requests.HTTPError as e:
        raise ApiError(f"La API rechazó la petición ({e.response.status_code}): "
                       f"{e.response.text[:300]}") from None


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("memorias"):
            st.caption(f"Memorias utilizadas: {msg['memorias']}")


tab_chat, tab_ingest = st.tabs(["Chat con Memoria", "Ingesta Manual"])

with tab_chat:
    st.session_state.setdefault("messages", [])
    for msg in st.session_state.messages:
        render_message(msg)

    if prompt := st.chat_input("Escribe tu mensaje..."):
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        render_message(user_msg)

        try:
            with st.spinner("Consultando memoria..."):
                data = api_post("/chat", {"message": prompt, "top_k": TOP_K}, TIMEOUT_CHAT).json()
            answer = data.get("answer") if isinstance(data, dict) else None
            if not answer:
                raise ApiError("La respuesta de la API no incluye 'answer'.")
        except ValueError:
            st.error("La API respondió, pero no con JSON válido.")
        except ApiError as e:
            st.error(str(e))
        else:
            reply = {"role": "assistant", "content": answer,
                     "memorias": len(data.get("used_memory_ids") or [])}
            st.session_state.messages.append(reply)
            render_message(reply)

with tab_ingest:
    with st.form("ingest_form"):
        subject = st.text_input("Sujeto (Ej. Proyecto CERO-A)")
        fact = st.text_area("Contenido / Hecho")
        keywords = st.text_input("Palabras clave (separadas por coma)")
        importance = st.slider("Importancia", 0.0, 1.0, 0.8, 0.05,
                               help=f"Desde {PIN_IMPORTANCE} la memoria queda fijada y no se olvida.")
        pin = st.checkbox("Fijar (no olvidar nunca)")
        submitted = st.form_submit_button("Guardar en Memoria")

    if submitted:
        if not subject.strip() or not fact.strip():
            st.warning("Rellena Sujeto y Contenido para guardar la memoria.")
        else:
            score = 1.0 if pin else round(importance, 2)
            payload = {
                "action": "ADD",
                "memory_note": {
                    "subject": subject.strip(),
                    "fact_contents": fact.strip(),
                    "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
                    "importance_score": score,
                },
            }
            try:
                api_post("/ingest", payload, TIMEOUT_INGEST)
            except ApiError as e:
                st.error(str(e))
            else:
                st.success("Memoria guardada y fijada." if score >= PIN_IMPORTANCE
                           else "Memoria guardada.")
