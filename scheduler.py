"""
scheduler.py
────────────
Scheduler semanal que ejecuta el rebalanceo automático de todas las carteras
cada viernes a las 22:00 (hora local), usando los agentes SAC entrenados
y datos reales de mercado vía yfinance.

Uso:
    pip install schedule
    python scheduler.py

Dejar corriendo en segundo plano. Usa la misma BD SQLite que la app Streamlit.
Logs en scheduler.log.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
import schedule
import torch
import yfinance as yf

# ── Rutas ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH      = PROJECT_ROOT / "paper_trading.db"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agente_SAC import AgenteSAC

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "scheduler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("scheduler")

# ── Activos ─────────────────────────────────────────────────────────────────
ACTIVOS_RIESGO = [
    "^GSPC", "^NDX", "IWM", "XLF", "XLE",
    "EEM",   "EWJ",  "FEZ",
    "SHY",   "AGG",  "TLT", "TIP", "LQD",
    "GC=F",  "HG=F",
    "VNQ",
]
ACTIVOS_DESCARGA = ACTIVOS_RIESGO + ["^IRX"]

RIESGO_POR_PERFIL = {
    "muy_conservador": 0.10,
    "conservador":     0.30,
    "normal":          0.50,
    "arriesgado":      0.70,
    "muy_arriesgado":  0.90,
}

FACTOR_ANUALIZACION = 52.0
SEMANAS_HISTORIA    = 26
COSTE_TRANSACCION   = 0.001

# Cache de agentes en memoria para no recargar en cada usuario
_agentes_cache: dict[str, AgenteSAC] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Base de datos
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def obtener_todos_usuarios() -> list[dict]:
    """Devuelve todos los usuarios con cuestionario completado y perfil asignado."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, nombre, email, perfil_asignado "
            "FROM usuarios WHERE cuestionario_completado = 1 AND perfil_asignado IS NOT NULL"
        ).fetchall()
    return [{"id": r[0], "nombre": r[1], "email": r[2], "perfil": r[3]} for r in rows]


def obtener_ultima_entrada(usuario_id: str) -> Optional[dict]:
    """Devuelve el último registro del historial de un usuario."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT fecha, valor_cartera, pesos_json "
            "FROM historial_cartera WHERE usuario_id = ? "
            "ORDER BY fecha DESC LIMIT 1",
            (usuario_id,),
        ).fetchone()
    if not row:
        return None
    try:
        valor = float(row[1])
    except Exception:
        valor = 10_000.0
    return {
        "fecha":  row[0],
        "valor":  valor,
        "pesos":  json.loads(row[2]) if row[2] else None,
    }


def guardar_entrada_historial(
    usuario_id: str,
    valor: float,
    pesos: np.ndarray,
    retorno: float,
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO historial_cartera "
            "(usuario_id, fecha, valor_cartera, pesos_json, retorno_semana) "
            "VALUES (?, ?, ?, ?, ?)",
            (usuario_id, datetime.now().isoformat(),
             float(valor), json.dumps(pesos.tolist()), float(retorno)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Datos de mercado
# ══════════════════════════════════════════════════════════════════════════════

def descargar_precios() -> Optional[pd.DataFrame]:
    """Descarga precios semanales de los últimos SEMANAS_HISTORIA semanas."""
    from datetime import timedelta
    fecha_ini = (datetime.today() - timedelta(weeks=SEMANAS_HISTORIA + 4)).strftime("%Y-%m-%d")
    fecha_fin = datetime.today().strftime("%Y-%m-%d")

    log.info("Descargando precios de mercado...")
    try:
        raw = yf.download(
            ACTIVOS_DESCARGA,
            start=fecha_ini,
            end=fecha_fin,
            auto_adjust=False,
            progress=False,
        )
        col = "Adj Close" if "Adj Close" in raw else "Close"
        px  = raw[col].copy()
        px.index = pd.to_datetime(px.index).normalize()
        px  = px.sort_index()
        px  = px[~px.index.duplicated(keep="first")]
        px  = px.ffill().resample("W-FRI").last().dropna(how="all")
        precios = px[[c for c in ACTIVOS_RIESGO if c in px.columns]]
        log.info(f"Precios descargados: {len(precios)} semanas, {precios.shape[1]} activos")
        return precios
    except Exception as e:
        log.error(f"Error descargando precios: {e}")
        return None


def construir_features(ret: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    px, ret = px.sort_index().copy(), ret.sort_index().copy()
    f = pd.concat([
        px.pct_change(1).add_suffix("_r1w"),
        px.pct_change(4).add_suffix("_r4w"),
        px.pct_change(12).add_suffix("_r12w"),
        ret.rolling(4).std(ddof=1).add_suffix("_vol4w"),
        ret.rolling(12).std(ddof=1).add_suffix("_vol12w"),
        (px / px.rolling(12).max() - 1.0).add_suffix("_dd12w"),
        (px / px.rolling(4).mean()  - 1.0).add_suffix("_ma4w"),
        (px / px.rolling(12).mean() - 1.0).add_suffix("_ma12w"),
    ], axis=1).shift(1).dropna()
    return f


def construir_estado(
    precios: pd.DataFrame,
    pesos_previos: Optional[np.ndarray],
    riesgo: float,
    vol_ema: float = 0.0,
    drawdown_actual: float = 0.0,
) -> Optional[np.ndarray]:
    if len(precios) < 14:
        return None
    ret   = precios.pct_change(fill_method=None).dropna(how="any")
    feats = construir_features(ret, precios)
    if feats.empty:
        return None
    lag = ret.shift(1).loc[feats.index]
    df  = pd.concat([feats, lag], axis=1).dropna()
    if df.empty:
        return None

    estado_mercado = df.iloc[-1].values.astype(np.float32)

    n_act = len(ACTIVOS_RIESGO)
    if pesos_previos is None:
        pesos_previos = np.zeros(n_act + 1, dtype=np.float32)
        pesos_previos[-1] = 1.0
    pesos_previos = np.asarray(pesos_previos, dtype=np.float32)
    if len(pesos_previos) != n_act + 1:
        pesos_tmp = np.zeros(n_act + 1, dtype=np.float32)
        pesos_tmp[-1] = 1.0
        pesos_previos = pesos_tmp

    estado = np.concatenate([
        estado_mercado,
        pesos_previos,
        np.array([riesgo],          dtype=np.float32),
        np.array([vol_ema],         dtype=np.float32),
        np.array([drawdown_actual], dtype=np.float32),
    ])
    return np.nan_to_num(estado, nan=0.0, posinf=0.0, neginf=0.0)


def calcular_retorno_semana(
    pesos_previos: np.ndarray,
    precios: pd.DataFrame,
) -> float:
    if len(precios) < 2:
        return 0.0
    ret_ul  = precios.pct_change().iloc[-1]
    ret_act = np.array([ret_ul.get(a, 0.0) for a in ACTIVOS_RIESGO])
    ret_act = np.nan_to_num(ret_act, nan=0.0)
    return float(np.dot(pesos_previos[:len(ACTIVOS_RIESGO)], ret_act))


# ══════════════════════════════════════════════════════════════════════════════
# Carga de agentes
# ══════════════════════════════════════════════════════════════════════════════

def cargar_agente(perfil: str) -> Optional[AgenteSAC]:
    if perfil in _agentes_cache:
        return _agentes_cache[perfil]

    device = torch.device("cpu")
    rutas  = [
        PROJECT_ROOT / "agentes_finales"   / f"agente_final_{perfil}.pt",
        PROJECT_ROOT / "agentes_ganadores" / f"agente_{perfil}.pt",
        PROJECT_ROOT / "resultados"        / f"agente_final_{perfil}.pt",
    ]
    for ruta in rutas:
        if not ruta.exists():
            continue
        try:
            ck = torch.load(ruta, map_location=device, weights_only=False)
            de = ck["actor_state_dict"]["net.0.weight"].shape[1]
            da = ck["actor_state_dict"]["mu_head.weight"].shape[0]
            ag = AgenteSAC(
                dimension_estado=de,
                dimension_accion=da,
                dispositivo=device,
                reward_scale=float(ck.get("config", {}).get("reward_scale", 20.0)),
            )
            ag.actor.load_state_dict(ck["actor_state_dict"])
            ag.actor.eval()
            _agentes_cache[perfil] = ag
            log.info(f"Agente '{perfil}' cargado desde {ruta.name} "
                     f"(dim_estado={de}, dim_accion={da})")
            return ag
        except Exception as e:
            log.warning(f"Error cargando {ruta.name}: {e}")

    log.error(f"No se encontró agente para perfil '{perfil}'")
    return None


def decidir_pesos(agente: AgenteSAC, estado: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(estado, dtype=torch.float32).unsqueeze(0)
        pesos, _, _ = agente.actor.deterministic_action(t)
    return pesos.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Rebalanceo
# ══════════════════════════════════════════════════════════════════════════════

def rebalancear_usuario(
    usuario: dict,
    precios: pd.DataFrame,
) -> bool:
    """
    Ejecuta el ciclo completo para un usuario:
      1. Carga agente del perfil
      2. Recupera pesos y valor anteriores
      3. Calcula retorno de la semana pasada
      4. Construye estado actual y decide nuevos pesos
      5. Guarda en BD
    """
    uid    = usuario["id"]
    nombre = usuario["nombre"]
    perfil = usuario["perfil"]

    agente = cargar_agente(perfil)
    if agente is None:
        log.warning(f"[{nombre}] Sin agente para '{perfil}', se omite.")
        return False

    # Recuperar situación anterior
    ultima = obtener_ultima_entrada(uid)
    if ultima and ultima["pesos"]:
        pesos_prev  = np.array(ultima["pesos"], dtype=np.float32)
        valor_prev  = ultima["valor"]
    else:
        n_act       = len(ACTIVOS_RIESGO)
        pesos_prev  = np.zeros(n_act + 1, dtype=np.float32)
        pesos_prev[-1] = 1.0
        valor_prev  = 10_000.0

    # Retorno de la semana pasada con pesos anteriores
    ret_semana = calcular_retorno_semana(pesos_prev, precios)
    coste      = COSTE_TRANSACCION * np.sum(np.abs(pesos_prev - pesos_prev))  # 0 en primera iteración
    valor_nuevo = valor_prev * (1.0 + ret_semana - coste)

    # Drawdown actual
    dd_actual = float(np.clip(valor_nuevo / max(valor_prev, valor_nuevo) - 1.0, -1.0, 0.0))

    # Construir estado y decidir
    riesgo = RIESGO_POR_PERFIL.get(perfil, 0.50)
    estado = construir_estado(precios, pesos_prev, riesgo, 0.0, dd_actual)
    if estado is None:
        log.warning(f"[{nombre}] No se pudo construir el estado.")
        return False

    if len(estado) != agente.actor.obs_dim:
        log.error(f"[{nombre}] Dimensión incompatible: "
                  f"agente={agente.actor.obs_dim}, estado={len(estado)}")
        return False

    pesos_nuevos = decidir_pesos(agente, estado)

    # Coste de rebalanceo con pesos nuevos vs anteriores
    coste_real = COSTE_TRANSACCION * np.sum(np.abs(pesos_nuevos - pesos_prev))
    valor_nuevo -= valor_nuevo * coste_real

    # Guardar
    guardar_entrada_historial(uid, valor_nuevo, pesos_nuevos, ret_semana)

    p_cash = float(pesos_nuevos[-1]) if len(pesos_nuevos) > len(ACTIVOS_RIESGO) else 0.0
    log.info(
        f"[{nombre}] {perfil:20s} | "
        f"valor={valor_nuevo:>10,.2f} EUR | "
        f"ret={ret_semana*100:+.2f}% | "
        f"cash={p_cash*100:.1f}%"
    )
    return True


def ultimo_viernes_esperado() -> datetime:
    """Devuelve el último viernes a las 22:00 que debería haber ocurrido."""
    hoy = datetime.now()
    dias_desde_viernes = (hoy.weekday() - 4) % 7  # 4 = viernes
    ultimo_viernes = hoy - timedelta(days=dias_desde_viernes)
    return ultimo_viernes.replace(hour=22, minute=0, second=0, microsecond=0)


def rebalanceo_pendiente() -> bool:
    """
    Comprueba si hay un rebalanceo pendiente comparando la fecha del
    último rebalanceo en la BD con el último viernes esperado.
    """
    viernes_esperado = ultimo_viernes_esperado()

    # Si el viernes esperado es en el futuro (hoy ES viernes pero antes de las 22)
    if viernes_esperado > datetime.now():
        return False

    # Buscar la fecha del último rebalanceo en la BD (la entrada más reciente)
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(fecha) FROM historial_cartera"
            ).fetchone()
        if not row or not row[0]:
            log.info("No hay historial — primer rebalanceo pendiente.")
            return True

        ultimo_rebalanceo = datetime.fromisoformat(str(row[0]))
        if ultimo_rebalanceo < viernes_esperado:
            dias = (viernes_esperado - ultimo_rebalanceo).days
            log.info(
                f"Rebalanceo pendiente detectado: "
                f"ultimo={ultimo_rebalanceo.strftime('%Y-%m-%d %H:%M')} | "
                f"esperado={viernes_esperado.strftime('%Y-%m-%d %H:%M')} | "
                f"diferencia={dias} dias"
            )
            return True
        return False
    except Exception as e:
        log.warning(f"Error comprobando rebalanceo pendiente: {e}")
        return False


def job_rebalanceo() -> None:
    """Tarea principal del scheduler — se ejecuta cada viernes a las 22:00."""
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 60)
    log.info(f"REBALANCEO SEMANAL — {ahora}")
    log.info("=" * 60)

    # Descargar precios una sola vez para todos los usuarios
    precios = descargar_precios()
    if precios is None or precios.empty:
        log.error("No se pudo descargar datos de mercado. Rebalanceo cancelado.")
        return

    usuarios = obtener_todos_usuarios()
    if not usuarios:
        log.info("No hay usuarios con cuestionario completado.")
        return

    log.info(f"Procesando {len(usuarios)} usuario(s)...")
    ok = sum(rebalancear_usuario(u, precios) for u in usuarios)
    log.info(f"Rebalanceo completado: {ok}/{len(usuarios)} usuarios procesados.")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--forzar", action="store_true",
                        help="Forzar rebalanceo inmediato independientemente del dia")
    args = parser.parse_args()

    log.info("Scheduler iniciado.")
    log.info(f"BD: {DB_PATH}")
    log.info("Rebalanceo programado: cada viernes a las 22:00")

    hoy = datetime.now()
    es_viernes = hoy.weekday() == 4  # 4 = viernes

    if args.forzar:
        log.info("--forzar activado: ejecutando rebalanceo inmediato...")
        job_rebalanceo()
    elif rebalanceo_pendiente():
        log.info("Rebalanceo pendiente detectado — ejecutando ahora...")
        job_rebalanceo()
    elif es_viernes and hoy.hour >= 22:
        log.info("Es viernes despues de las 22:00 — ejecutando rebalanceo...")
        job_rebalanceo()
    else:
        dia = ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"][hoy.weekday()]
        dias_hasta_viernes = (4 - hoy.weekday()) % 7
        if dias_hasta_viernes == 0:
            dias_hasta_viernes = 7
        proximo = (hoy + timedelta(days=dias_hasta_viernes)).strftime("%d/%m")
        log.info(f"Hoy es {dia} — sin rebalanceos pendientes.")
        log.info(f"Proximo rebalanceo: viernes {proximo} a las 22:00")

    # Programar ejecucion semanal
    schedule.every().friday.at("22:00").do(job_rebalanceo)

    log.info("Scheduler en espera. Ctrl+C para detener.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Scheduler detenido manualmente.")