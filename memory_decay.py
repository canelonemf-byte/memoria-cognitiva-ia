r"""Modelo de olvido de memoria: fuente única para el servidor y la simulación.

    S = \exp(-\Delta t / \tau),    \tau = \tau_0 \,(1 + \alpha I)\,(1 + \beta \ln(1 + n))

    S          retención en (0, 1]               I   importancia en [0, 1]
    \Delta t   días desde el último acceso       n   número de accesos
    \tau_0     vida media base (días)

- Cada acceso reinicia \Delta t y aumenta n: lo que se usa dura más.
- Sin uso, S tiende a 0 y toda memoria acaba por debajo del umbral.
- Las memorias fijadas (pinned, o importancia >= PIN_IMPORTANCE) tienen S = 1.

Servidor: importar `_decay_score` / `should_forget` de este módulo (o pegar el código).
Simulación: python memory_decay.py
"""

import math
from datetime import datetime, timezone

TAU0_DAYS = 30.0          # vida media base (días) para I = 0 y n = 0
ALPHA_IMPORTANCE = 1.0    # cuánto alarga tau la importancia
BETA_USE = 1.0            # cuánto alarga tau el uso (accesos)
FORGET_THRESHOLD = 0.15   # S por debajo de este valor => se olvida
PIN_IMPORTANCE = 0.95     # importancia desde la que la memoria queda fijada


def _tau(importance: float, access_count: int) -> float:
    importance = min(max(importance, 0.0), 1.0)
    access_count = max(access_count, 0)
    return (TAU0_DAYS
            * (1 + ALPHA_IMPORTANCE * importance)
            * (1 + BETA_USE * math.log1p(access_count)))


def calculate_decay(importance: float, access_count: int, age_days: float,
                    *, pinned: bool = False) -> float:
    """Retención S de una memoria (1 = recién usada, 0 = olvidada)."""
    if pinned:
        return 1.0
    return math.exp(-max(age_days, 0.0) / _tau(importance, access_count))


def days_until_forgotten(importance: float, access_count: int, *,
                         pinned: bool = False,
                         threshold: float = FORGET_THRESHOLD) -> float:
    """Días sin acceso hasta que S < umbral (solución cerrada de S = umbral)."""
    if pinned:
        return math.inf
    return _tau(importance, access_count) * math.log(1 / threshold)


def is_pinned(mem) -> bool:
    """Fijada si tiene `pinned=True` (si el modelo lo define) o importancia >= PIN_IMPORTANCE."""
    return bool(getattr(mem, "pinned", False)) or mem.importance_score >= PIN_IMPORTANCE


def _decay_score(mem) -> float:
    """Sustituto directo de la versión anterior.

    Necesita mem.importance_score, mem.access_count y mem.last_access
    (datetime; si es naive se interpreta como UTC).
    """
    last = mem.last_access
    if last is None:
        age_days = 0.0
    else:
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
    return calculate_decay(mem.importance_score, mem.access_count, age_days,
                           pinned=is_pinned(mem))


def should_forget(mem, threshold: float = FORGET_THRESHOLD) -> bool:
    return not is_pinned(mem) and _decay_score(mem) < threshold


# --------------------------------------------------------------------------- #
# Simulación
# --------------------------------------------------------------------------- #
SCENARIOS = [
    {"label": "I=0.2 n=0",    "imp": 0.2, "acc": 0},
    {"label": "I=0.6 n=0",    "imp": 0.6, "acc": 0},   # ingesta por defecto de doc_chunker.py
    {"label": "I=0.8 n=0",    "imp": 0.8, "acc": 0},   # ingesta por defecto de la UI
    {"label": "I=0.9 n=0",    "imp": 0.9, "acc": 0},
    {"label": "I=0.5 n=15",   "imp": 0.5, "acc": 15},
    {"label": "I=1.0 fijada", "imp": 1.0, "acc": 0},
]


def run_simulation(threshold: float = FORGET_THRESHOLD, days: int = 365, step: int = 30) -> None:
    print(f"S = exp(-dt/tau),  tau = {TAU0_DAYS:g}*(1 + {ALPHA_IMPORTANCE:g}*I)"
          f"*(1 + {BETA_USE:g}*ln(1+n))  |  umbral = {threshold}")
    header = f"{'Día':<5}| " + " | ".join(f"{s['label']:<12}" for s in SCENARIOS)
    print(header)
    print("-" * len(header))

    for day in range(0, days + 1, step):
        cells = []
        for s in SCENARIOS:
            score = calculate_decay(s["imp"], s["acc"], day,
                                    pinned=s["imp"] >= PIN_IMPORTANCE)
            cells.append(f"{score:<11.3f}{'*' if score < threshold else ' '}")
        print(f"{day:<5}| " + " | ".join(cells))

    print("\n* = por debajo del umbral: se olvidaría\n")
    print("Días sin acceso hasta olvidar:")
    for s in SCENARIOS:
        d = days_until_forgotten(s["imp"], s["acc"],
                                 pinned=s["imp"] >= PIN_IMPORTANCE, threshold=threshold)
        texto = "nunca" if math.isinf(d) else f"{d:.0f} días"
        print(f"  {s['label']:<14}{texto}")


if __name__ == "__main__":
    run_simulation()
