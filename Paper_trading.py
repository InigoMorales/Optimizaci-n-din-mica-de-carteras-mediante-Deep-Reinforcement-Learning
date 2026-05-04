"""
app_paper_trading.py
────────────────────
Paper trading en modo live con login, cuestionario de perfil y agentes SAC.

Instalación:
    pip install streamlit plotly yfinance torch bcrypt

Uso:
    streamlit run app_paper_trading.py
"""

from __future__ import annotations

import sys
import json
import uuid
import sqlite3
import warnings
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import bcrypt
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH      = PROJECT_ROOT / "paper_trading.db"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agente_SAC import AgenteSAC

# ── Activos ─────────────────────────────────────────────────────────────────
ACTIVOS_RIESGO = [
    "^GSPC", "^NDX", "IWM", "XLF", "XLE",
    "EEM",   "EWJ",  "FEZ",
    "SHY",   "AGG",  "TLT", "TIP", "LQD",
    "GC=F",  "HG=F",
    "VNQ",
]
ACTIVOS_DESCARGA = ACTIVOS_RIESGO + ["^IRX"]

NOMBRES_ACTIVOS = {
    "^GSPC": "S&P 500",     "^NDX":  "Nasdaq 100",  "IWM":  "Russell 2000",
    "XLF":   "Financieras", "XLE":   "Energia US",   "EEM":  "Em. Markets",
    "EWJ":   "Japon",       "FEZ":   "Euro Stoxx",   "SHY":  "Bonos C/P",
    "AGG":   "Bonos Int.",  "TLT":   "Bonos L/P",    "TIP":  "Inflacion",
    "LQD":   "High Yield",  "GC=F":  "Oro",          "HG=F": "Cobre",
    "VNQ":   "REITs",       "CASH":  "Cash",
}

PERFILES = {
    "muy_conservador": {"emoji": "shield",  "color": "#3b82f6", "label": "Muy Conservador"},
    "conservador":     {"emoji": "leaf",    "color": "#22c55e", "label": "Conservador"},
    "normal":          {"emoji": "balance", "color": "#f59e0b", "label": "Normal"},
    "arriesgado":      {"emoji": "fire",    "color": "#ef4444", "label": "Arriesgado"},
    "muy_arriesgado":  {"emoji": "rocket",  "color": "#a855f7", "label": "Muy Arriesgado"},
}

PERFIL_EMOJI = {
    "muy_conservador": "🛡️",
    "conservador":     "🌿",
    "normal":          "⚖️",
    "arriesgado":      "🔥",
    "muy_arriesgado":  "🚀",
}

FACTOR_ANUALIZACION = 52.0
SEMANAS_HISTORIA    = 26

# ── Cuestionario ─────────────────────────────────────────────────────────────
PREGUNTAS = [
    {
        "id": "edad",
        "texto": "Cuantos anos tienes?",
        "opciones": [
            ("Menos de 25 anos",   5),
            ("Entre 25 y 35 anos", 4),
            ("Entre 35 y 50 anos", 3),
            ("Entre 50 y 65 anos", 2),
            ("Mas de 65 anos",     1),
        ],
    },
    {
        "id": "caida",
        "texto": "Imagina que inviertes 10.000 euros y al mes siguiente valen 8.000 euros. Que harias?",
        "opciones": [
            ("Los saco todos, esto no es para mi",                        1),
            ("Saco una parte para no perder mas",                         2),
            ("No hago nada, espero a que se recupere",                    3),
            ("Depende de por que ha bajado",                              4),
            ("Meto mas dinero, esta mas barato - es una oportunidad",     5),
        ],
    },
    {
        "id": "objetivo",
        "texto": "Para que es este dinero?",
        "opciones": [
            ("Por si acaso, mi fondo de emergencia",                     1),
            ("Para algo concreto en los proximos anos (casa, coche...)", 2),
            ("Para dejar algo a mis hijos o familia",                    3),
            ("Para complementar mi pension algun dia",                   4),
            ("No tengo un plan especifico, quiero que crezca",           5),
        ],
    },
    {
        "id": "sueldo",
        "texto": (
            "Te ofrecen elegir entre dos opciones de sueldo:\n\n"
            "Opcion A: sueldo fijo de 1.900 euros al mes\n\n"
            "Opcion B: lanzas una moneda. Si sale cara cobras 1.000 euros, "
            "si sale cruz cobras 3.000 euros\n\n"
            "Cual elegirías?"
        ),
        "opciones": [
            ("El fijo de 1.900 euros, sin dudarlo", 1),
            ("Me lo tendria que pensar",             3),
            ("La moneda, prefiero arriesgar",        5),
        ],
    },
    {
        "id": "plazo",
        "texto": "Cuando crees que necesitaras este dinero?",
        "opciones": [
            ("Podria necesitarlo en cualquier momento", 1),
            ("En los proximos 2 anos",                  2),
            ("No lo se",                                3),
            ("En 5 o 10 anos",                          4),
            ("No lo necesitare en mucho tiempo",        5),
        ],
    },
    {
        "id": "importancia",
        "texto": "Como de importante es este dinero para ti respecto a todo lo que tienes?",
        "opciones": [
            ("Es casi todo lo que tengo",                          1),
            ("Es bastante, me afectaria mucho perderlo",           2),
            ("Es una parte, me afectaria pero lo superaria",       3),
            ("Es relativamente poco de todo lo que tengo",         4),
            ("Es una cantidad pequena, no me cambiaria la vida",   5),
        ],
    },
    {
        "id": "decision",
        "texto": "Cuando tomas decisiones importantes en tu vida, como sueles actuar?",
        "opciones": [
            ("Voy siempre a lo seguro aunque gane menos",              1),
            ("Prefiero seguridad pero acepto algo de riesgo",          2),
            ("Busco el equilibrio entre riesgo y recompensa",          3),
            ("Si la recompensa es buena, acepto el riesgo",            4),
            ("Me gusta arriesgar, las grandes recompensas lo exigen",  5),
        ],
    },
]


def calcular_perfil_desde_puntuacion(media: float) -> str:
    if media < 1.8:   return "muy_conservador"
    if media < 2.6:   return "conservador"
    if media < 3.4:   return "normal"
    if media < 4.2:   return "arriesgado"
    return "muy_arriesgado"


# ══════════════════════════════════════════════════════════════════════════════
# Base de datos SQLite
# ══════════════════════════════════════════════════════════════════════════════

_db_lock = threading.Lock()


@contextmanager
def get_conn():
    with _db_lock:
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


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id                      TEXT PRIMARY KEY,
            email                   TEXT UNIQUE NOT NULL,
            nombre                  TEXT NOT NULL,
            password_hash           TEXT NOT NULL,
            fecha_registro          TEXT NOT NULL,
            perfil_asignado         TEXT,
            cuestionario_completado INTEGER DEFAULT 0,
            tema                    TEXT DEFAULT 'dark',
            saldo                   REAL DEFAULT 10000.0
        );
        CREATE TABLE IF NOT EXISTS movimientos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id  TEXT NOT NULL REFERENCES usuarios(id),
            tipo        TEXT NOT NULL,
            importe     REAL NOT NULL,
            fecha       TEXT NOT NULL,
            nota        TEXT
        );

        CREATE TABLE IF NOT EXISTS respuestas_cuestionario (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id  TEXT NOT NULL REFERENCES usuarios(id),
            pregunta_id TEXT NOT NULL,
            puntuacion  INTEGER NOT NULL,
            fecha       TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS historial_cartera (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id     TEXT NOT NULL REFERENCES usuarios(id),
            fecha          TEXT NOT NULL,
            valor_cartera  REAL NOT NULL,
            pesos_json     TEXT NOT NULL,
            retorno_semana REAL DEFAULT 0,
            twr            REAL DEFAULT 1.0
        );
        """)


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


def registrar_usuario(email: str, nombre: str, pw: str) -> tuple[bool, str]:
    if len(pw) < 8:
        return False, "La contrasena debe tener al menos 8 caracteres."
    if "@" not in email or "." not in email:
        return False, "Email no valido."
    try:
        uid = str(uuid.uuid4())
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO usuarios (id,email,nombre,password_hash,fecha_registro) "
                "VALUES (?,?,?,?,?)",
                (uid, email.lower().strip(), nombre.strip(),
                 hash_password(pw), datetime.now().isoformat()),
            )
        return True, uid
    except sqlite3.IntegrityError:
        return False, "Ya existe una cuenta con ese email."
    except Exception as e:
        return False, str(e)


def login_usuario(email: str, pw: str) -> tuple[bool, str, Optional[dict]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id,nombre,password_hash,perfil_asignado,"
            "cuestionario_completado,tema FROM usuarios WHERE email=?",
            (email.lower().strip(),),
        ).fetchone()
    if not row:
        return False, "Email o contrasena incorrectos.", None
    uid, nombre, ph, perfil, cuest, tema = row
    if not verify_password(pw, ph):
        return False, "Email o contrasena incorrectos.", None
    return True, "OK", {
        "id": uid, "nombre": nombre, "email": email.lower().strip(),
        "perfil": perfil, "cuestionario_completado": bool(cuest),
        "tema": tema or "dark",
    }


def guardar_respuestas(usuario_id: str, resps: list[dict], perfil: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM respuestas_cuestionario WHERE usuario_id=?", (usuario_id,))
        for r in resps:
            conn.execute(
                "INSERT INTO respuestas_cuestionario (usuario_id,pregunta_id,puntuacion,fecha) "
                "VALUES (?,?,?,?)",
                (usuario_id, r["pregunta_id"], r["puntuacion"], datetime.now().isoformat()),
            )
        conn.execute(
            "UPDATE usuarios SET perfil_asignado=?,cuestionario_completado=1 WHERE id=?",
            (perfil, usuario_id),
        )


def actualizar_perfil_db(uid: str, perfil: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE usuarios SET perfil_asignado=? WHERE id=?", (perfil, uid))


def actualizar_tema_db(uid: str, tema: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE usuarios SET tema=? WHERE id=?", (tema, uid))


def obtener_saldo(uid: str) -> float:
    with get_conn() as conn:
        row = conn.execute("SELECT saldo FROM usuarios WHERE id=?", (uid,)).fetchone()
    if not row or row[0] is None:
        return 10_000.0
    try:
        return float(row[0])
    except Exception:
        return 10_000.0


def actualizar_saldo(uid: str, saldo_nuevo: float) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE usuarios SET saldo=? WHERE id=?", (float(saldo_nuevo), uid))


def registrar_movimiento(uid: str, tipo: str, importe: float, nota: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO movimientos (usuario_id, tipo, importe, fecha, nota) VALUES (?,?,?,?,?)",
            (uid, tipo, float(importe), datetime.now().isoformat(), nota),
        )


def obtener_movimientos(uid: str, limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT tipo, importe, fecha, nota FROM movimientos "
            "WHERE usuario_id=? ORDER BY fecha DESC LIMIT ?",
            (uid, limit),
        ).fetchall()
    return [{"tipo": r[0], "importe": float(r[1]), "fecha": r[2], "nota": r[3] or ""} for r in rows]


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
        raw = row[1]
        if isinstance(raw, (bytes, bytearray)):
            import struct
            try:    valor = struct.unpack("d", raw)[0]
            except: valor = float(raw.decode("utf-8", errors="ignore"))
        else:
            valor = float(raw)
    except Exception:
        valor = 10_000.0
    return {
        "fecha": str(row[0]),
        "valor": valor,
        "pesos": json.loads(row[2]) if row[2] else None,
    }


def guardar_historial_db(uid: str, valor: float, pesos: np.ndarray, ret: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO historial_cartera (usuario_id,fecha,valor_cartera,pesos_json,retorno_semana) "
            "VALUES (?,?,?,?,?)",
            (uid, datetime.now().isoformat(), float(valor), json.dumps(pesos.tolist()), float(ret)),
        )


def cargar_historial_db(uid: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT fecha,valor_cartera,pesos_json,retorno_semana "
            "FROM historial_cartera WHERE usuario_id=? ORDER BY fecha ASC",
            (uid,),
        ).fetchall()
    result = []
    for r in rows:
        try:
            fecha   = str(r[0])
            # r[1] puede ser float, int, str o bytes según versión de SQLite
            raw_val = r[1]
            if isinstance(raw_val, (bytes, bytearray)):
                import struct
                try:
                    valor = struct.unpack("d", raw_val)[0]
                except Exception:
                    valor = float(raw_val.decode("utf-8", errors="ignore"))
            else:
                valor = float(raw_val)
            pesos   = json.loads(r[2]) if r[2] else []
            retorno = float(r[3]) if r[3] is not None else 0.0
            result.append({"fecha": fecha, "valor": valor,
                           "pesos": pesos, "retorno": retorno})
        except Exception:
            continue  # ignorar filas corruptas
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Mercado y agente
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900)
def cargar_snapshots_bd(usuario_id: str, ventana_horas: int) -> pd.DataFrame:
    """
    Carga los snapshots guardados en la BD para el usuario
    dentro de la ventana temporal indicada.
    Devuelve DataFrame con columnas: fecha, valor, twr
    """
    from datetime import timedelta
    desde = (datetime.now() - timedelta(hours=ventana_horas)).isoformat()
    with get_conn() as conn:
        try:
            rows = conn.execute(
                "SELECT fecha, valor_cartera, twr FROM historial_cartera "
                "WHERE usuario_id=? AND fecha>=? ORDER BY fecha ASC",
                (usuario_id, desde),
            ).fetchall()
        except Exception:
            # Fallback si la columna twr no existe aún
            rows = conn.execute(
                "SELECT fecha, valor_cartera, 1.0 FROM historial_cartera "
                "WHERE usuario_id=? AND fecha>=? ORDER BY fecha ASC",
                (usuario_id, desde),
            ).fetchall()
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        try:
            data.append({
                "fecha": pd.to_datetime(r[0], format="mixed"),
                "valor": float(r[1]),
                "twr":   float(r[2]) if r[2] is not None else 1.0,
            })
        except Exception:
            continue
    return pd.DataFrame(data).set_index("fecha") if data else pd.DataFrame()


def descargar_precios_horarios() -> pd.DataFrame:
    """Precios con intervalo 1h de los ultimos 7 dias para zoom intradiario."""
    from datetime import timedelta
    fecha_ini = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    fecha_fin  = datetime.today().strftime("%Y-%m-%d")
    try:
        raw = yf.download(
            ACTIVOS_RIESGO,
            start=fecha_ini, end=fecha_fin,
            interval="1h", auto_adjust=True, progress=False,
        )
        if raw.empty:
            return pd.DataFrame()
        col = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.get_level_values(0)[0]
        px  = raw[col].copy()
        px.index = pd.to_datetime(px.index)
        return px.sort_index().ffill().dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)   # refresca cada 15 min
def descargar_precios_diarios() -> pd.DataFrame:
    """
    Descarga precios diarios de los últimos 5 días para calcular
    el valor actual de la cartera en tiempo real.
    Separado de los datos semanales que usa el agente.
    """
    from datetime import timedelta
    fecha_ini = (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    fecha_fin = datetime.today().strftime("%Y-%m-%d")
    try:
        raw = yf.download(
            ACTIVOS_RIESGO,
            start=fecha_ini,
            end=fecha_fin,
            auto_adjust=True,
            progress=False,
        )
        col = "Close" if "Close" in raw else raw.columns.get_level_values(0)[0]
        px  = raw[col].copy() if isinstance(raw.columns, pd.MultiIndex) else raw
        px.index = pd.to_datetime(px.index).normalize()
        px = px.sort_index().ffill()
        return px
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def descargar_precios() -> tuple[pd.DataFrame, pd.Series]:
    fi  = (datetime.today() - timedelta(weeks=SEMANAS_HISTORIA + 4)).strftime("%Y-%m-%d")
    ff  = datetime.today().strftime("%Y-%m-%d")
    raw = yf.download(ACTIVOS_DESCARGA, start=fi, end=ff,
                      auto_adjust=False, progress=False)
    col = "Adj Close" if "Adj Close" in raw else "Close"
    px  = raw[col].copy()
    px.index = pd.to_datetime(px.index).normalize()
    px  = px.sort_index()
    px  = px[~px.index.duplicated(keep="first")]
    px  = px.ffill().resample("W-FRI").last().dropna(how="all")
    irx = px["^IRX"].copy() if "^IRX" in px.columns else pd.Series(dtype=float)
    return px[[c for c in ACTIVOS_RIESGO if c in px.columns]], irx


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
    pesos_previos: Optional[np.ndarray] = None,
    riesgo: float = 0.5,
    vol_ema: float = 0.0,
    drawdown_actual: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Replica exactamente el estado que construye EntornoCartera:
      [features_mercado (128), retornos_lag1 (16), pesos_previos (17),
       riesgo (1), vol_ema (1), drawdown_actual (1)]
    Total: 164 features — coincide con dim_estado del agente entrenado.
    """
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

    estado_mercado = df.iloc[-1].values.astype(np.float32)  # 144

    # Pesos previos completos (16 activos + cash = 17)
    n_activos = len(ACTIVOS_RIESGO)
    if pesos_previos is None:
        # Cartera inicial: todo en cash
        pesos_previos = np.zeros(n_activos + 1, dtype=np.float32)
        pesos_previos[-1] = 1.0
    pesos_previos = pesos_previos.astype(np.float32)

    # Concatenar igual que EntornoCartera
    estado = np.concatenate([
        estado_mercado,                           # 144
        pesos_previos,                            # 17
        np.array([riesgo], dtype=np.float32),     # 1
        np.array([vol_ema], dtype=np.float32),    # 1
        np.array([drawdown_actual], dtype=np.float32),  # 1
    ])  # total = 164

    return np.nan_to_num(estado, nan=0.0, posinf=0.0, neginf=0.0)


@st.cache_resource
def cargar_agente(perfil: str) -> Optional[AgenteSAC]:
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
            ag = AgenteSAC(dimension_estado=de, dimension_accion=da, dispositivo=device,
                           reward_scale=float(ck.get("config", {}).get("reward_scale", 20.0)))
            ag.actor.load_state_dict(ck["actor_state_dict"])
            ag.actor.eval()
            return ag
        except Exception as e:
            st.warning(f"Error cargando {ruta.name}: {e}")
    return None


def decidir_pesos(agente: AgenteSAC, estado: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(estado, dtype=torch.float32).unsqueeze(0)
        pesos, _, _ = agente.actor.deterministic_action(t)
    return pesos.squeeze(0).cpu().numpy()


def calcular_metricas(hist: list[dict]) -> dict:
    if len(hist) < 2:
        return {"sharpe": 0.0, "cagr": 0.0, "mdd": 0.0}
    vals = np.array([float(h["valor"]) for h in hist if h.get("valor") is not None])
    rets = np.diff(vals) / vals[:-1]
    n    = len(rets)
    cagr = (vals[-1] / vals[0]) ** (FACTOR_ANUALIZACION / n) - 1.0
    sh   = (rets.mean() / rets.std() * np.sqrt(FACTOR_ANUALIZACION)
            if n > 1 and rets.std() > 0 else 0.0)
    acum = (1 + rets).cumprod()
    mdd  = float((acum / np.maximum.accumulate(acum) - 1).min()) if len(acum) else 0.0
    return {"sharpe": float(sh), "cagr": float(cagr), "mdd": float(mdd)}


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css(dark: bool) -> None:
    if dark:
        bg, bg2, border = "#07101f", "#0d1929", "#1a2744"
        txt, sub, card  = "#e2e8f0", "#64748b", "#0d1929"
        inp, live_bg    = "#111827", "#052e16"
        info_bg         = "#0c1a3a"
    else:
        bg, bg2, border = "#ffffff", "#f9fafb", "#f0f0f0"
        txt, sub, card  = "#111111", "#888888", "#ffffff"
        inp, live_bg    = "#f5f5f5", "#f0fdf4"
        info_bg         = "#f0f7ff"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html,body,[class*="css"],.stApp {{
    font-family:'Sora',sans-serif!important;
    background:{bg}!important;
    color:{txt}!important;
}}
.stApp {{ background:{bg}!important; }}

div[data-testid="stSidebar"] {{
    background:{bg2}!important;
    border-right:1px solid {border}!important;
}}

/* Cards */
.card {{
    background:{card};
    border:1px solid {border};
    border-radius:16px;
    padding:24px 28px;
    margin-bottom:12px;
    transition:border-color .2s, box-shadow .2s;
}}
{''.join([
    '.card:hover{border-color:#3b82f6;box-shadow:0 4px 24px rgba(59,130,246,.08);}' if not dark else
    '.card:hover{border-color:#3b82f6;box-shadow:0 4px 24px rgba(59,130,246,.15);}'
])}

/* Metric labels */
.ml {{
    font-family:'JetBrains Mono',monospace;
    font-size:10px;
    color:{sub};
    text-transform:uppercase;
    letter-spacing:.14em;
    margin-bottom:6px;
}}
.mv {{
    font-family:'JetBrains Mono',monospace;
    font-size:26px;
    font-weight:600;
    color:{txt};
}}
.mv.pos {{ color:#16a34a; }}
.mv.neg {{ color:#dc2626; }}

/* Hero value — pantalla principal */
.hero-label {{
    font-family:'Sora',sans-serif;
    font-size:13px;
    font-weight:400;
    color:{sub};
    letter-spacing:.04em;
    margin-bottom:4px;
}}
.hero-value {{
    font-family:'Sora',sans-serif;
    font-size:52px;
    font-weight:700;
    color:{txt};
    letter-spacing:-.02em;
    line-height:1;
    margin-bottom:8px;
}}
.hero-change {{
    font-family:'JetBrains Mono',monospace;
    font-size:16px;
    font-weight:500;
}}
.hero-change.pos {{ color:#16a34a; }}
.hero-change.neg {{ color:#dc2626; }}

/* Live badge */
.live-dot {{
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding:4px 12px;
    background:{live_bg};
    border:1px solid #16a34a;
    border-radius:20px;
    font-family:'JetBrains Mono',monospace;
    font-size:11px;
    color:#16a34a;
}}
.dot {{
    width:6px;height:6px;border-radius:50%;
    background:#16a34a;animation:blink 2s infinite;
}}
@keyframes blink {{0%,100%{{opacity:1;}}50%{{opacity:.2;}}}}

/* Cuestionario */
.q-text {{
    font-size:19px;font-weight:600;color:{txt};
    margin-bottom:24px;white-space:pre-line;line-height:1.6;
}}
.q-hdr {{
    font-family:'JetBrains Mono',monospace;font-size:10px;
    color:{sub};text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;
}}
.pb-out {{ background:{border};border-radius:4px;height:3px;margin-bottom:32px; }}
.pb-in  {{ background:linear-gradient(90deg,#3b82f6,#8b5cf6);border-radius:4px;height:3px;transition:width .4s ease; }}

/* Info box */
.info-box {{
    background:{info_bg};
    border-left:3px solid #3b82f6;
    border-radius:0 8px 8px 0;
    padding:14px 18px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:{sub};
    line-height:1.8;
}}

/* Perfil chip */
.chip {{
    display:inline-block;
    padding:4px 14px;
    border-radius:20px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;font-weight:600;
}}

/* Detail button */
.detail-btn {{
    display:inline-block;
    margin-top:8px;
    padding:10px 22px;
    border-radius:10px;
    font-family:'Sora',sans-serif;
    font-size:14px;font-weight:600;
    background:{'#1a2744' if dark else '#f1f5f9'};
    color:{'#94a3b8' if dark else '#475569'};
    border:1px solid {border};
    cursor:pointer;
    transition:all .2s;
    text-align:center;
}}

/* Inputs */
input,.stTextInput input,.stPasswordInput input {{
    background:{inp}!important;
    color:{txt}!important;
    border:1px solid {border}!important;
    border-radius:10px!important;
    font-family:'Sora',sans-serif!important;
}}

/* Botones */
.stButton>button {{
    font-family:'Sora',sans-serif!important;
    font-weight:600!important;
    border-radius:10px!important;
    transition:all .15s!important;
}}

/* Tabs */
.stTabs [data-baseweb="tab"] {{
    font-family:'Sora',sans-serif!important;
    font-size:14px!important;
}}

/* Radio */
.stRadio label {{
    font-family:'Sora',sans-serif!important;
    font-size:15px!important;
}}

/* Ajustes modo claro radio: forzar colores */
.stRadio [data-testid="stMarkdownContainer"] p {{
    color:{txt}!important;
}}
</style>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# Pantallas
# ══════════════════════════════════════════════════════════════════════════════

def pantalla_login() -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
<div style='text-align:center;margin-bottom:32px;'>
    <div style='font-size:52px;'>📊</div>
    <div style='font-family:Sora;font-size:30px;font-weight:700;margin-top:8px;'>Portfolio AI</div>
    <div style='font-size:13px;color:#64748b;margin-top:6px;'>Gestion inteligente de carteras con Deep RL</div>
</div>""", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Iniciar sesion", "Crear cuenta"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            em = st.text_input("Email", key="li_em", placeholder="tu@email.com")
            pw = st.text_input("Contrasena", type="password", key="li_pw", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Entrar", use_container_width=True, key="btn_li"):
                if not em or not pw:
                    st.error("Rellena todos los campos.")
                else:
                    ok, msg, usr = login_usuario(em, pw)
                    if ok:
                        st.session_state.usuario       = usr
                        st.session_state.token         = str(uuid.uuid4())
                        st.session_state.historico     = cargar_historial_db(usr["id"])
                        saldo_cargado = obtener_saldo(usr["id"])
                        st.session_state.capital_inicial = saldo_cargado
                        st.session_state.valor_cartera = (
                            st.session_state.historico[-1]["valor"]
                            if st.session_state.historico else saldo_cargado
                        )
                        st.session_state.pantalla = (
                            "cuestionario" if not usr["cuestionario_completado"] else "app"
                        )
                        st.rerun()
                    else:
                        st.error(msg)

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            nom  = st.text_input("Nombre", key="reg_nom", placeholder="Tu nombre")
            em2  = st.text_input("Email", key="reg_em", placeholder="tu@email.com")
            pw2  = st.text_input("Contrasena", type="password", key="reg_pw", placeholder="Minimo 8 caracteres")
            pw2b = st.text_input("Repite la contrasena", type="password", key="reg_pw2", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Crear cuenta", use_container_width=True, key="btn_reg"):
                if not all([nom, em2, pw2, pw2b]):
                    st.error("Rellena todos los campos.")
                elif pw2 != pw2b:
                    st.error("Las contrasenas no coinciden.")
                else:
                    ok, msg = registrar_usuario(em2, nom, pw2)
                    if ok:
                        _, _, usr = login_usuario(em2, pw2)
                        st.session_state.usuario       = usr
                        st.session_state.token         = str(uuid.uuid4())
                        st.session_state.historico     = []
                        st.session_state.valor_cartera = 10_000.0
                        st.session_state.pantalla      = "cuestionario"
                        st.rerun()
                    else:
                        st.error(msg)


def pantalla_cuestionario() -> None:
    usr    = st.session_state.usuario
    q_idx  = st.session_state.get("q_idx", 0)
    q_res  = st.session_state.get("q_resps", [])
    n      = len(PREGUNTAS)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 3, 1])
    with col:
        pct = int(q_idx / n * 100)
        st.markdown(f"""
<div style='font-family:Sora;font-size:22px;font-weight:700;margin-bottom:6px;'>Cuentanos un poco sobre ti</div>
<div style='font-size:13px;color:#64748b;margin-bottom:20px;'>
    Hola {usr['nombre']}, estas preguntas nos ayudan a elegir la estrategia mas adecuada para ti.
</div>
<div class='pb-out'><div class='pb-in' style='width:{pct}%;'></div></div>
<div class='q-hdr'>Pregunta {q_idx+1} de {n}</div>""", unsafe_allow_html=True)

        if q_idx < n:
            pregunta = PREGUNTAS[q_idx]
            st.markdown(f"<div class='q-text'>{pregunta['texto']}</div>", unsafe_allow_html=True)

            radio_key = f"q_resp_{q_idx}"
            sel = st.radio(
                "opcion", [o[0] for o in pregunta["opciones"]],
                key=radio_key, label_visibility="collapsed",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                if q_idx > 0 and st.button("Anterior", use_container_width=True, key=f"prev_{q_idx}"):
                    st.session_state.q_idx   = q_idx - 1
                    st.session_state.q_resps = q_res[:-1]
                    st.rerun()
            with cb:
                if st.button("Siguiente", use_container_width=True, key=f"next_{q_idx}"):
                    sel_actual = st.session_state.get(radio_key, pregunta["opciones"][0][0])
                    punt = next(
                        (o[1] for o in pregunta["opciones"] if o[0] == sel_actual),
                        pregunta["opciones"][0][1],
                    )
                    q_res.append({"pregunta_id": pregunta["id"], "puntuacion": punt})
                    st.session_state.q_resps = q_res
                    st.session_state.q_idx   = q_idx + 1
                    st.rerun()
        else:
            media  = np.mean([r["puntuacion"] for r in q_res])
            perfil = calcular_perfil_desde_puntuacion(media)
            info   = PERFILES[perfil]
            emoji  = PERFIL_EMOJI[perfil]
            guardar_respuestas(usr["id"], q_res, perfil)
            st.session_state.usuario["perfil"]                  = perfil
            st.session_state.usuario["cuestionario_completado"] = True

            st.markdown(f"""
<div style='text-align:center;padding:32px 0;'>
    <div style='font-size:64px;margin-bottom:16px;'>{emoji}</div>
    <div style='font-size:26px;font-weight:700;margin-bottom:10px;'>
        Tu perfil es <span style='color:{info["color"]};'>{info['label']}</span>
    </div>
    <div style='font-size:14px;color:#64748b;max-width:400px;margin:0 auto 32px;line-height:1.6;'>
        Hemos analizado tus respuestas y este perfil se adapta mejor
        a tu situacion y tolerancia al riesgo.
    </div>
</div>""", unsafe_allow_html=True)

            if st.button("Empezar a invertir", use_container_width=True):
                st.session_state.pantalla = "app"
                st.session_state.q_idx   = 0
                st.session_state.q_resps = []
                st.rerun()


def pantalla_ajustes() -> None:
    usr = st.session_state.usuario
    st.markdown("## Ajustes")
    st.markdown("---")

    st.markdown("### Apariencia")
    tema = usr.get("tema", "dark")
    t_nuevo = st.radio("Tema", ["dark", "light"],
                       index=0 if tema == "dark" else 1,
                       format_func=lambda x: "Modo oscuro" if x == "dark" else "Modo claro",
                       horizontal=True)
    if t_nuevo != tema:
        actualizar_tema_db(usr["id"], t_nuevo)
        st.session_state.usuario["tema"] = t_nuevo
        st.rerun()

    st.markdown("---")
    st.markdown("### Perfil de riesgo")
    ops = list(PERFILES.keys())
    p_actual = usr.get("perfil", "normal")
    p_nuevo  = st.selectbox("Cambiar perfil", ops,
                            index=ops.index(p_actual) if p_actual in ops else 2,
                            format_func=lambda p: f"{PERFIL_EMOJI[p]} {PERFILES[p]['label']}")
    if p_nuevo != p_actual:
        if st.button("Guardar perfil"):
            actualizar_perfil_db(usr["id"], p_nuevo)
            st.session_state.usuario["perfil"] = p_nuevo
            st.success(f"Perfil actualizado a {PERFILES[p_nuevo]['label']}")
            st.rerun()

    st.markdown("---")
    st.markdown("### Cuestionario")
    if st.button("Repetir cuestionario de perfil"):
        st.session_state.q_idx    = 0
        st.session_state.q_resps  = []
        st.session_state.pantalla = "cuestionario"
        st.rerun()

    st.markdown("---")
    st.markdown("### Cuenta")
    st.markdown(f"**Email:** {usr['email']}")
    st.markdown(f"**Nombre:** {usr['nombre']}")
    st.markdown("---")

    if st.button("Cerrar sesion", use_container_width=True):
        for k in ["usuario", "token", "pantalla", "historico",
                  "valor_cartera", "pesos_actuales", "agente_cargado",
                  "q_idx", "q_resps"]:
            st.session_state.pop(k, None)
        st.rerun()

    if st.button("Volver al dashboard", use_container_width=True):
        st.session_state.pantalla = "app"
        st.rerun()


def pantalla_app() -> None:
    usr    = st.session_state.usuario
    perfil = usr.get("perfil", "normal")
    info   = PERFILES[perfil]
    emoji  = PERFIL_EMOJI[perfil]
    dark   = usr.get("tema", "dark") == "dark"
    vista  = st.session_state.get("vista_app", "principal")

    bg_p   = "#07101f" if dark else "#ffffff"
    grid_c = "#1a2744" if dark else "#f0f0f0"
    txt_c  = "#e2e8f0" if dark else "#111111"
    border = "#1a2744" if dark else "#f0f0f0"
    sub_c  = "#64748b"

    RIESGO_VAL = {
        "muy_conservador": 0.10, "conservador": 0.30, "normal": 0.50,
        "arriesgado": 0.70, "muy_arriesgado": 0.90,
    }

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### Hola, {usr['nombre']}")
        st.markdown(
            f"<span class='chip' style='border:1px solid {info['color']};color:{info['color']};'>"
            f"{emoji} {info['label']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        # Saldo invertido
        saldo_actual = obtener_saldo(usr["id"])
        st.session_state.capital_inicial = saldo_actual
        color_perfil = info["color"]
        st.markdown(
            "<div style='font-family:JetBrains Mono;font-size:10px;color:#64748b;"
            "text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>Saldo invertido</div>"
            f"<div style='font-family:JetBrains Mono;font-size:24px;font-weight:700;"
            f"color:{color_perfil};'>EUR {saldo_actual:,.0f}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        cd1, cd2 = st.columns(2)
        with cd1:
            if st.button("+ Ingresar", use_container_width=True, key="btn_dep"):
                st.session_state.modal = "ingresar"
                st.rerun()
        with cd2:
            if st.button("- Retirar", use_container_width=True, key="btn_ret"):
                st.session_state.modal = "retirar"
                st.rerun()
        st.markdown("---")

        if vista == "principal":
            if st.button("Ver detalle completo", use_container_width=True):
                st.session_state.vista_app = "detalle"
                st.rerun()
        else:
            if st.button("Volver al inicio", use_container_width=True):
                st.session_state.vista_app = "principal"
                st.rerun()
        st.markdown("---")
        if st.button("Ajustes", use_container_width=True):
            st.session_state.pantalla = "ajustes"
            st.rerun()
        if st.button("Cerrar sesion", use_container_width=True):
            for k in ["usuario","token","pantalla","historico",
                      "valor_cartera","pesos_actuales","agente_cargado","vista_app"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.markdown("---")
        if st.button("Actualizar mercado", use_container_width=True):
            st.cache_data.clear()
            st.session_state["forzar_snapshot"] = True
            st.rerun()
        # Mostrar precio de referencia: si hay datos diarios, cuándo es
        try:
            px_d = descargar_precios_diarios()
            if not px_d.empty:
                ultima_hora = pd.to_datetime(px_d.index[-1]).strftime("%d/%m %H:%M")
                st.caption(f"Precio al: {ultima_hora} | Refresca en 15 min")
            else:
                st.caption(f"Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        except Exception:
            st.caption(f"Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    # ── Modales ingresar / retirar ────────────────────────────────────────
    modal = st.session_state.get("modal")
    if modal in ("ingresar", "retirar"):
        saldo_modal = obtener_saldo(usr["id"])
        titulo  = "Ingresar dinero" if modal == "ingresar" else "Retirar dinero"
        subtxt  = "El dinero se añade a tu cartera virtual" if modal == "ingresar"                   else f"Saldo disponible: EUR {saldo_modal:,.0f}"
        max_imp = 1_000_000 if modal == "ingresar" else max(100, int(saldo_modal))
        def_imp = 1_000 if modal == "ingresar" else min(1_000, int(saldo_modal))

        with st.container():
            st.markdown(f"### {titulo}")
            st.caption(subtxt)
            importe = st.number_input(
                "Importe (EUR)", min_value=100, max_value=max_imp,
                value=def_imp, step=100, key="modal_importe",
            )
            nota = st.text_input("Nota (opcional)", placeholder="Ej: ahorro mensual", key="modal_nota")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancelar", use_container_width=True, key="modal_cancel"):
                    st.session_state.modal = None
                    st.rerun()
            with c2:
                label_btn = "Confirmar ingreso" if modal == "ingresar" else "Confirmar retirada"
                if st.button(label_btn, use_container_width=True, key="modal_confirm"):
                    if modal == "retirar" and importe > saldo_modal:
                        st.error("No tienes suficiente saldo.")
                    else:
                        # Actualizar saldo Y valor de cartera — son la misma cosa
                        saldo_nuevo = saldo_modal + importe if modal == "ingresar" else saldo_modal - importe
                        saldo_nuevo = max(0.0, saldo_nuevo)

                        actualizar_saldo(usr["id"], saldo_nuevo)
                        registrar_movimiento(usr["id"], modal, importe, nota)

                        # Actualizar valor de cartera en sesión y en BD
                        st.session_state.capital_inicial = saldo_nuevo
                        st.session_state.valor_cartera   = saldo_nuevo

                        # Guardar en historial — el TWR NO cambia con ingresos/retiradas
                        pesos_act = st.session_state.get("pesos_actuales")
                        if pesos_act is None:
                            n_act = len(ACTIVOS_RIESGO)
                            pesos_act = np.zeros(n_act + 1, dtype=np.float32)
                            pesos_act[-1] = 1.0
                        hist_actual = st.session_state.get("historico", [])
                        twr_actual_dep = hist_actual[-1].get("twr", 1.0) if hist_actual else 1.0
                        guardar_historial_db(usr["id"], saldo_nuevo, pesos_act, 0.0, twr_actual_dep)

                        # Limpiar historial en memoria para forzar recarga
                        st.session_state.historico = []

                        st.session_state.modal = None
                        signo = "+" if modal == "ingresar" else "-"
                        st.success(f"{signo}EUR {importe:,.0f} — Valor cartera: EUR {saldo_nuevo:,.0f}")
                        st.rerun()
            st.markdown("---")

    # ── Autorefresh cada 15 minutos ────────────────────────────────────────
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=15 * 60 * 1000, key="autorefresh_mercado")
    except ImportError:
        pass  # si no está instalado, no pasa nada

    # ── Cargar historial desde BD si memoria está vacía ───────────────────
    if not st.session_state.get("historico"):
        hist_bd = cargar_historial_db(usr["id"])
        if hist_bd:
            st.session_state.historico = hist_bd
            st.session_state.valor_cartera = hist_bd[-1]["valor"]

    # ── Descargar datos ────────────────────────────────────────────────────
    with st.spinner("Descargando datos..."):
        try:
            precios, _ = descargar_precios()
            ok = not precios.empty and len(precios) >= 14
        except Exception as e:
            st.error(f"Error: {e}"); return
    if not ok:
        st.error("No se pudieron descargar datos."); return

    # ── Cargar agente ──────────────────────────────────────────────────────
    if st.session_state.get("agente_cargado") != perfil:
        with st.spinner(f"Cargando agente {perfil}..."):
            ag = cargar_agente(perfil)
        if ag is None:
            st.error(f"No se encontro el agente para {perfil}. "
                     f"Verifica agentes_ganadores/agente_{perfil}.pt"); return
        st.session_state.agente_cargado = perfil
    else:
        ag = cargar_agente(perfil)

    # ── Construir estado y decidir ─────────────────────────────────────────
    _pesos_prev = st.session_state.get("pesos_actuales")
    _riesgo_val = RIESGO_VAL.get(perfil, 0.50)
    _hist       = st.session_state.get("historico", [])
    _val_actual = st.session_state.valor_cartera
    _val_max    = max((float(h["valor"]) for h in _hist if h.get("valor") is not None), default=_val_actual)
    _dd_actual  = float(np.clip(_val_actual / _val_max - 1.0, -1.0, 0.0)) if _val_max > 0 else 0.0

    # ══════════════════════════════════════════════════════════════════════
    # LÓGICA SEMANAL
    # ─────────────────────────────────────────────────────────────────────
    # Durante la semana (Lunes-Jueves): los pesos NO cambian.
    #   La app carga los últimos pesos de la BD y actualiza el valor
    #   con los retornos reales de los activos desde el último rebalanceo.
    #
    # El viernes a las 22:00: el scheduler.py decide nuevos pesos y
    #   guarda la nueva entrada en la BD. La app los recoge aquí.
    #
    # La app NUNCA llama al agente para cambiar pesos — eso solo lo hace
    #   el scheduler. La app es solo lectura de la BD + cálculo de valor.
    # ══════════════════════════════════════════════════════════════════════

    ultima_entrada = obtener_ultima_entrada(usr["id"])

    if ultima_entrada and ultima_entrada["pesos"]:
        # Cargar pesos del último rebalanceo (fijos hasta el próximo viernes)
        pesos_vigentes = np.array(ultima_entrada["pesos"], dtype=np.float32)
        valor_base     = ultima_entrada["valor"]
    else:
        # Primera vez — todo en cash, valor inicial
        n_act          = len(ACTIVOS_RIESGO)
        pesos_vigentes = np.zeros(n_act + 1, dtype=np.float32)
        pesos_vigentes[-1] = 1.0
        valor_base     = st.session_state.get("capital_inicial", 10_000.0)
        # Si nunca ha rebalanceado, ejecutar el agente una primera vez para tener pesos
        estado = construir_estado(precios, None, _riesgo_val, 0.0, 0.0)
        if estado is not None and len(estado) == ag.actor.obs_dim:
            pesos_vigentes = decidir_pesos(ag, estado)
            guardar_historial_db(usr["id"], valor_base, pesos_vigentes, 0.0)
            ultima_entrada = {"valor": valor_base, "pesos": pesos_vigentes.tolist()}

    pr     = pesos_vigentes[:len(ACTIVOS_RIESGO)]
    p_cash = float(pesos_vigentes[len(ACTIVOS_RIESGO)]) if len(pesos_vigentes) > len(ACTIVOS_RIESGO) else 0.0

    # ── Calcular valor actual: unidades × precio actual ───────────────────
    # Lógica simple y correcta:
    # 1. En el momento del rebalanceo, guardamos el precio de cada activo
    # 2. Calculamos cuántas unidades compramos: unidades[i] = peso[i]*capital / precio_ref[i]
    # 3. Valor actual = sum(unidades[i] * precio_actual[i]) + cash
    # El precio de referencia se guarda en session_state al primer refresco
    # y solo cambia cuando el scheduler rebalancea (viernes).

    precios_hoy = descargar_precios_horarios()

    # Guardar precio de referencia la primera vez (o si cambia el perfil)
    ref_key = f"px_ref_{perfil}"
    if ref_key not in st.session_state and not precios_hoy.empty:
        st.session_state[ref_key] = precios_hoy.iloc[-1].to_dict()

    px_ref    = st.session_state.get(ref_key, {})
    px_actual = precios_hoy.iloc[-1].to_dict() if not precios_hoy.empty else {}

    if px_ref and px_actual:
        valor = 0.0
        for i, activo in enumerate(ACTIVOS_RIESGO):
            peso_i   = float(pr[i]) if i < len(pr) else 0.0
            px_r     = float(px_ref.get(activo, 0.0))
            px_a     = float(px_actual.get(activo, 0.0))
            if px_r > 1e-8 and px_a > 1e-8:
                # unidades * precio actual
                valor += peso_i * valor_base * (px_a / px_r)
            else:
                valor += peso_i * valor_base
        # Cash no varía
        valor += p_cash * valor_base
        valor = max(valor, 0.01)
    else:
        valor = valor_base


    cap   = st.session_state.get("capital_inicial", 10_000.0)
    fecha = precios.index[-1]

    # Actualizar session_state
    st.session_state.valor_cartera  = valor
    st.session_state.pesos_actuales = pesos_vigentes.copy()

    # TWR acumulado: encadenar retornos ignorando ingresos/retiradas
    hist_mem = st.session_state.get("historico", [])
    twr_prev   = hist_mem[-1].get("twr", 1.0) if hist_mem else 1.0
    ret_desde_rebalanceo = (valor / valor_base - 1.0) if valor_base > 0 else 0.0
    twr_actual = twr_prev * (1.0 + ret_desde_rebalanceo)

    # Guardar snapshot con timestamp actual (no solo viernes)
    # Así la gráfica muestra la evolución diaria/intradiaria real
    ahora_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    ultimo_str = hist_mem[-1].get("fecha", "") if hist_mem else ""

    # Guardar si: no hay historial, o han pasado más de 30 min desde el último
    def _min_desde_ultimo() -> float:
        try:
            from datetime import datetime as dt
            t_ult = dt.fromisoformat(ultimo_str[:16])
            t_now = dt.fromisoformat(ahora_str[:16])
            return (t_now - t_ult).total_seconds() / 60
        except Exception:
            return 999.0

    forzar = st.session_state.pop("forzar_snapshot", False)
    if not hist_mem or forzar or _min_desde_ultimo() >= 15:
        entrada = {
            "fecha":   ahora_str,
            "valor":   float(valor),
            "retorno": float(ret_desde_rebalanceo),
            "twr":     float(twr_actual),
        }
        hist_mem.append(entrada)
        st.session_state.historico = hist_mem
        # Guardar en BD para persistencia entre sesiones
        guardar_historial_db(usr["id"], valor, pesos_vigentes, ret_desde_rebalanceo)

    # Próximo rebalanceo
    from datetime import date
    hoy       = date.today()
    dias_hasta_viernes = (4 - hoy.weekday()) % 7  # 4 = viernes
    if dias_hasta_viernes == 0:
        prox_rebalanceo = "hoy (viernes)"
    elif dias_hasta_viernes == 1:
        prox_rebalanceo = "mañana"
    else:
        prox_rebalanceo = f"en {dias_hasta_viernes} dias (viernes)"

    st.session_state.proximo_rebalanceo = prox_rebalanceo

    met     = calcular_metricas(st.session_state.historico)
    ret_tot = valor / cap - 1.0

    # ══════════════════════════════════════════════════════════════════════
    # VISTA PRINCIPAL — minimalista tipo Trade Republic
    # ══════════════════════════════════════════════════════════════════════
    if vista == "principal":
        # Header mínimo
        ca, cb = st.columns([5, 1])
        with ca:
            st.markdown(
                f"<div style='font-family:Sora;font-size:15px;font-weight:600;"
                f"color:{sub_c};margin-bottom:2px;'>{emoji} {info['label']}</div>",
                unsafe_allow_html=True,
            )
        with cb:
            st.markdown("<div class='live-dot'><div class='dot'></div>EN VIVO</div>",
                        unsafe_allow_html=True)

        # Valor hero
        # TWR: retorno porcentual real independiente de ingresos/retiradas
        twr_actual_hero = hist_mem[-1].get("twr", 1.0) if hist_mem else 1.0
        ret_twr = (twr_actual_hero - 1.0) * 100.0
        cl_hero = "pos" if ret_twr >= 0 else "neg"
        signo   = "+" if ret_twr >= 0 else ""
        st.markdown(f"""
<div style='padding:32px 0 8px 0;'>
    <div class='hero-label'>Valor de tu cartera</div>
    <div class='hero-value'>€{valor:,.2f}</div>
    <div class='hero-change {cl_hero}'>{signo}{ret_twr:.2f}% rentabilidad</div>
</div>""", unsafe_allow_html=True)

        # Gráfica principal — línea fina, sin relleno agresivo, Trade Republic style
        st.markdown("<br>", unsafe_allow_html=True)
        if len(st.session_state.historico) > 1:
            df_h = pd.DataFrame(st.session_state.historico)
            df_h["fecha"] = pd.to_datetime(df_h["fecha"], format="mixed", dayfirst=False)

            color_line = info["color"]
            # Si está en pérdidas, línea roja
            if ret_tot < 0:
                color_line = "#dc2626"

            rgb = tuple(int(color_line.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

            # ── Gráfica con zoom dinámico ──────────────────────────────────────
            zoom_opts   = ["1h", "3h", "1d", "10d"]
            zoom_labels = {"1h": "1 hora", "3h": "3 horas", "1d": "Por días", "10d": "10 días"}
            zoom_sel    = st.session_state.get("zoom_sel", "1d")

            zc1, zc2, zc3, zc4, _ = st.columns([1, 1, 1, 1, 4])
            for col_z, opt in zip([zc1, zc2, zc3, zc4], zoom_opts):
                with col_z:
                    if st.button(zoom_labels[opt], key=f"zoom_{opt}",
                                 type="primary" if zoom_sel == opt else "secondary",
                                 use_container_width=True):
                        st.session_state.zoom_sel = opt
                        st.rerun()

            zoom_sel = st.session_state.get("zoom_sel", "1d")

            # Cargar snapshots reales de la BD según zoom
            horas_ventana = {"1h": 1, "3h": 3, "1d": 24*7, "10d": 24*30}[zoom_sel]
            df_snap = cargar_snapshots_bd(usr["id"], horas_ventana)

            if not df_snap.empty and len(df_snap) >= 2:
                # Calcular % de variación respecto al primer snapshot de la ventana
                valor_base_snap = df_snap["valor"].iloc[0]
                pct_snap = ((df_snap["valor"] - valor_base_snap) / valor_base_snap * 100.0)

                col_g = info["color"] if float(pct_snap.iloc[-1]) >= 0 else "#dc2626"
                rgb_g = tuple(int(col_g.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

                fmt_x = {"1h": "%H:%M", "3h": "%d %b %H:%M",
                         "1d": "%d %b", "10d": "%d %b"}[zoom_sel]
                nticks = {"1h": 6, "3h": 6, "1d": 7, "10d": 10}[zoom_sel]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_snap.index, y=pct_snap.values,
                    mode="lines",
                    line=dict(color=col_g, width=2, shape="spline", smoothing=0.6),
                    fill="tozeroy", fillcolor=f"rgba{rgb_g + (0.06,)}",
                    customdata=df_snap["valor"].values,
                    hovertemplate="€%{customdata:,.2f}<br>%{y:+.2f}%<br>%{x|%d %b %H:%M}<extra></extra>",
                ))
                fig.add_hline(y=0, line=dict(color=sub_c, width=1, dash="dot"))
                fig.update_layout(
                    template="plotly_dark" if dark else "plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=300, margin=dict(l=0, r=60, t=10, b=0), showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showline=False,
                               tickfont=dict(size=11, color=sub_c),
                               tickformat=fmt_x, nticks=nticks),
                    yaxis=dict(showgrid=True, gridcolor=border, zeroline=True,
                               zerolinecolor=sub_c, zerolinewidth=1,
                               showline=False, ticksuffix="%",
                               tickfont=dict(size=11, color=sub_c), side="right"),
                    hoverlabel=dict(bgcolor="#0d1929" if dark else "#ffffff",
                                   bordercolor=border,
                                   font=dict(family="JetBrains Mono", size=12, color=txt_c)),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                mins = 15
                st.markdown(
                    f"<div style='text-align:center;padding:60px 0;color:{sub_c};font-size:14px;'>"
                    f"Acumulando datos... La grafica aparece tras {mins} minutos de datos.</div>",
                    unsafe_allow_html=True,
                )

        # Stats rápidas debajo de la gráfica — 3 chips minimalistas
        st.markdown("<br>", unsafe_allow_html=True)
        cs1, cs2, cs3, cs4 = st.columns(4)
        def stat_chip(label, val, cls="", sub=""):
            return (f"<div style='padding:16px 20px;border-radius:12px;"
                    f"border:1px solid {border};background:{'#0d1929' if dark else '#f9fafb'};'>"
                    f"<div style='font-family:JetBrains Mono;font-size:10px;color:{sub_c};"
                    f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;'>{label}</div>"
                    f"<div class='mv {cls}' style='font-size:20px;'>{val}</div>"
                    f"{'<div style=\'font-size:10px;color:#64748b;margin-top:4px;\'>' + sub + '</div>' if sub else ''}"
                    f"</div>")

        # Métricas solo válidas con suficientes semanas de datos
        # Con menos de 4 semanas las métricas estadísticas no tienen sentido
        semanas_datos = len(hist_mem)
        metricas_validas = semanas_datos >= 4

        with cs1:
            if metricas_validas:
                cl = "pos" if met["sharpe"] >= 0 else "neg"
                st.markdown(stat_chip("Sharpe", f"{met['sharpe']:.2f}", cl), unsafe_allow_html=True)
            else:
                st.markdown(stat_chip("Sharpe", "—", "", f"disponible en {4-semanas_datos} sem."), unsafe_allow_html=True)
        with cs2:
            cl = "neg" if met["mdd"] < 0 else "pos"
            st.markdown(stat_chip("Max Drawdown", f"{met['mdd']*100:.1f}%", cl), unsafe_allow_html=True)
        with cs3:
            if metricas_validas:
                cl = "pos" if met["cagr"] >= 0 else "neg"
                st.markdown(stat_chip("CAGR", f"{met['cagr']*100:.1f}%", cl), unsafe_allow_html=True)
            else:
                st.markdown(stat_chip("CAGR", "—", "", f"disponible en {4-semanas_datos} sem."), unsafe_allow_html=True)
        with cs4:
            prox = st.session_state.get("proximo_rebalanceo", "viernes")
            st.markdown(stat_chip("Rebalanceo", prox, ""), unsafe_allow_html=True)

        # Call to action para ver detalle
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<hr style='border-color:{border};margin:0;'>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("Ver detalle de cartera →", use_container_width=True):
                st.session_state.vista_app = "detalle"
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # VISTA DETALLE — toda la info
    # ══════════════════════════════════════════════════════════════════════
    else:
        st.markdown(
            f"<h2 style='font-family:Sora;font-size:20px;font-weight:700;margin-bottom:0;'>"
            f"{emoji} Detalle de cartera — {info['label']}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<hr style='border-color:{border};'>", unsafe_allow_html=True)

        # Metricas
        c1, c2, c3, c4 = st.columns(4)
        def mk(label, val, cls=""):
            return (f"<div class='card'><div class='ml'>{label}</div>"
                    f"<div class='mv {cls}'>{val}</div></div>")
        with c1: st.markdown(mk("Valor", f"EUR {valor:,.0f}"), unsafe_allow_html=True)
        with c2:
            cl = "pos" if ret_tot >= 0 else "neg"
            st.markdown(mk("Retorno total", f"{ret_tot*100:+.2f}%", cl), unsafe_allow_html=True)
        with c3:
            cl = "pos" if met["sharpe"] >= 0 else "neg"
            st.markdown(mk("Sharpe", f"{met['sharpe']:.2f}", cl), unsafe_allow_html=True)
        with c4:
            cl = "neg" if met["mdd"] < 0 else "pos"
            st.markdown(mk("Max Drawdown", f"{met['mdd']*100:.1f}%", cl), unsafe_allow_html=True)

        # Graficas
        gi, gd = st.columns([3, 2])
        with gi:
            st.markdown("#### Evolucion de cartera")
            if len(st.session_state.historico) > 1:
                df_h = pd.DataFrame(st.session_state.historico)
                df_h["fecha"] = pd.to_datetime(df_h["fecha"], format="mixed", dayfirst=False)
                rgb  = tuple(int(info["color"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                fig  = go.Figure(go.Scatter(
                    x=df_h["fecha"], y=df_h["valor"],
                    mode="lines", line=dict(color=info["color"], width=2, shape="spline", smoothing=1.2),
                    fill="tozeroy", fillcolor=f"rgba{rgb + (0.07,)}",
                ))
                fig.update_layout(
                    template="plotly_dark" if dark else "plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                    xaxis=dict(showgrid=False, color=txt_c),
                    yaxis=dict(showgrid=True, gridcolor=grid_c,
                               tickprefix="EUR ", tickformat=",.0f", color=txt_c),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Historico insuficiente.")

        with gd:
            st.markdown("#### Asignacion actual")
            lbls = [NOMBRES_ACTIVOS.get(a,a) for a in ACTIVOS_RIESGO] + ["Cash"]
            vals = list(pr) + [p_cash]
            mask = np.array(vals) > 0.01
            fig2 = go.Figure(go.Pie(
                labels=[l for l,m in zip(lbls,mask) if m],
                values=[v for v,m in zip(vals,mask) if m],
                hole=0.58,
                marker=dict(colors=px.colors.qualitative.Set3,
                            line=dict(color=bg_p, width=2)),
                textfont=dict(family="JetBrains Mono", size=10),
                textinfo="label+percent",
            ))
            fig2.update_layout(
                template="plotly_dark" if dark else "plotly_white",
                paper_bgcolor="rgba(0,0,0,0)", height=260,
                margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # Tabla
        st.markdown("#### Pesos detallados")
        df_p = pd.DataFrame({
            "Ticker": ACTIVOS_RIESGO + ["CASH"],
            "Activo": [NOMBRES_ACTIVOS.get(a,a) for a in ACTIVOS_RIESGO] + ["Cash"],
            "Peso":   list(pr) + [p_cash],
        })
        df_p = df_p[df_p["Peso"] > 0.001].sort_values("Peso", ascending=False)
        df_p["Peso %"] = (df_p["Peso"] * 100).round(2).astype(str) + " %"
        df_p["Barra"]  = df_p["Peso"].apply(lambda w: "█"*int(w*28) + "░"*(28-int(w*28)))

        ct, ci = st.columns([2, 1])
        with ct:
            st.dataframe(df_p[["Ticker","Activo","Peso %","Barra"]],
                         hide_index=True, use_container_width=True,
                         height=min(50+len(df_p)*36, 400))
        with ci:
            rv  = float(sum(pr[:8]))
            rf  = float(sum(pr[8:13]))
            com = float(sum(pr[13:15]))
            rei = float(pr[15]) if len(pr) > 15 else 0.0
            st.markdown(f"""<div class='info-box'>
<b>Renta Variable</b> {rv*100:.1f}%<br>
<b>Renta Fija</b> {rf*100:.1f}%<br>
<b>Commodities</b> {com*100:.1f}%<br>
<b>REITs</b> {rei*100:.1f}%<br>
<b>Cash</b> {p_cash*100:.1f}%
</div>""", unsafe_allow_html=True)

        # Retornos mercado
        if len(precios) >= 2:
            st.markdown("#### Mercado — retornos ultima semana")
            rul  = precios.pct_change().iloc[-1].sort_values(ascending=False)
            df_r = pd.DataFrame({
                "Nombre":  [NOMBRES_ACTIVOS.get(t,t) for t in rul.index],
                "Retorno": (rul.values * 100).round(2),
            })
            fig3 = go.Figure(go.Bar(
                x=df_r["Nombre"], y=df_r["Retorno"],
                marker_color=[("#16a34a" if v >= 0 else "#dc2626") for v in df_r["Retorno"]],
                text=[f"{v:+.1f}%" for v in df_r["Retorno"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10, color=txt_c),
            ))
            fig3.update_layout(
                template="plotly_dark" if dark else "plotly_white",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(size=9), color=txt_c),
                yaxis=dict(showgrid=True, gridcolor=grid_c, ticksuffix="%", color=txt_c),
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"<hr style='border-color:{border};'>", unsafe_allow_html=True)
    # Historial de movimientos
    movs = obtener_movimientos(usr["id"])
    if movs:
        st.markdown("#### Movimientos recientes")
        df_mov = pd.DataFrame(movs)
        df_mov["fecha"] = pd.to_datetime(df_mov["fecha"], format="mixed").dt.strftime("%d/%m/%Y %H:%M")
        df_mov["importe"] = df_mov.apply(
            lambda r: f"+EUR {r['importe']:,.0f}" if r["tipo"] == "ingreso" else f"-EUR {r['importe']:,.0f}",
            axis=1,
        )
        df_mov["tipo"] = df_mov["tipo"].str.capitalize()
        st.dataframe(
            df_mov[["fecha", "tipo", "importe", "nota"]].rename(
                columns={"fecha": "Fecha", "tipo": "Tipo", "importe": "Importe", "nota": "Nota"}
            ),
            hide_index=True, use_container_width=True,
        )

    st.caption("Paper trading — dinero virtual. TFG: Optimizacion dinamica de carteras mediante Deep RL.")


def main() -> None:
    st.set_page_config(page_title="Portfolio AI", page_icon="📊",
                       layout="wide", initial_sidebar_state="auto")
    init_db()

    for k, v in [("pantalla","login"),("historico",[]),
                 ("valor_cartera",10_000.0),("pesos_actuales",None),
                 ("capital_inicial",10_000.0),("q_idx",0),("q_resps",[])]:
        if k not in st.session_state:
            st.session_state[k] = v

    dark = True
    if "usuario" in st.session_state:
        dark = st.session_state.usuario.get("tema", "dark") == "dark"
    inject_css(dark)

    p = st.session_state.pantalla
    if   p == "login":        pantalla_login()
    elif p == "cuestionario": pantalla_cuestionario()
    elif p == "ajustes":      pantalla_ajustes()
    elif p == "app":          pantalla_app()
    else:
        st.session_state.pantalla = "login"
        st.rerun()


if __name__ == "__main__":
    main()