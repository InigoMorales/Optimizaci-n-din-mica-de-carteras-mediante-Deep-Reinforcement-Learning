# src/Dataset/Dataset_TFG.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

ACTIVOS = [
    "^GSPC",  # S&P500
    "^NDX",   # Nasdaq 100
    "IWM",    # US small caps
    "EEM",    # Emerging Markets
    "EWP",    # España (ETF)
    "EWJ",    # Japón (ETF)
    "FEZ",    # Euro Stoxx 50 (ETF)
    "GC=F",   # Oro (futuro)
    "SI=F",   # Plata (futuro)
    "AGG",    # Bonds aggregate
    "TLT",    # Long bonds
    "SHY",    # Short bonds
    "^IRX",   # Risk-free proxy (yield anual en %)
]

FECHA_INICIO = "2004-01-01"
FECHA_FIN = "2026-01-01" 

FECHA_FIN_TRAIN = "2016-01-01"
FECHA_FIN_VALIDATION = "2020-01-01"

FACTOR_ANUALIZACION = 252.0

# ============================================================
# PATHS
# ============================================================

def _encontrar_raiz_proyecto() -> Path:
    """
    Busca hacia arriba hasta encontrar carpeta 'Datos' o 'src'.
    Si no encuentra, usa el directorio padre del archivo.
    """
    p = Path(__file__).resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "Datos").exists() or (parent / "src").exists():
            return parent
    return p.parent

RAIZ_PROYECTO = _encontrar_raiz_proyecto()
CARPETA_DATOS = RAIZ_PROYECTO / "Datos"

for sub in ["Raw", "Train", "Validation", "Test"]:
    (CARPETA_DATOS / sub).mkdir(parents=True, exist_ok=True)

# ============================================================
# FEATURES (sin leakage)
# ============================================================

def construir_features(retornos: pd.DataFrame) -> pd.DataFrame:
    """
    Features por activo:
      - momentum 20/60 (suma rolling)
      - volatilidad 20/60 (std rolling)

    Importante:
      - shift(1) para evitar leakage (usar info hasta t-1).
      - dropna al final.
    """
    ret = retornos.sort_index().copy()

    mom20 = ret.rolling(20).sum().add_suffix("_mom20")
    mom60 = ret.rolling(60).sum().add_suffix("_mom60")

    vol20 = ret.rolling(20).std(ddof=1).add_suffix("_vol20")
    vol60 = ret.rolling(60).std(ddof=1).add_suffix("_vol60")

    feats = pd.concat([mom20, mom60, vol20, vol60], axis=1)

    # Evitar leakage: las features del día t solo usan info hasta t-1
    feats = feats.shift(1)

    feats = feats.dropna()
    return feats


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    # =========================
    # 1) DESCARGA
    # =========================
    data = yf.download(
        ACTIVOS,
        start=FECHA_INICIO,
        end=FECHA_FIN,
        auto_adjust=False,
        progress=False,
    )

    if "Adj Close" not in data:
        raise ValueError("No se encontró 'Adj Close' en la descarga de yfinance.")

    precios = data["Adj Close"].copy()
    precios.index = pd.to_datetime(precios.index).normalize()
    precios = precios.sort_index()
    precios = precios[~precios.index.duplicated(keep="first")]

    # Forward fill para festivos desalineados y pequeños huecos (ya validaste que bloques max = 1)
    precios = precios.ffill()

    # Guardar precios raw (incluye IRX)
    precios.to_csv(CARPETA_DATOS / "Raw" / "precios_adjclose_full.csv")

    # =========================
    # 2) SEPARAR IRX y ACTIVOS RIESGO
    # =========================
    if "^IRX" not in precios.columns:
        raise ValueError("No se encontró '^IRX' en los datos descargados.")

    irx_yield = precios["^IRX"].copy()  # yield anual en %
    precios_activos = precios.drop(columns=["^IRX"])

    # =========================
    # 3) RETORNOS DIARIOS (riesgo)
    # =========================
    retornos = precios_activos.pct_change(fill_method=None)
    retornos = retornos.dropna(how="any")  # elimina primera fila y cualquier fila incompleta

    # Guardar retornos full (riesgo)
    retornos.to_csv(CARPETA_DATOS / "Raw" / "retornos_full.csv")

    # =========================
    # 4) RF ANUAL y RF DIARIO
    # =========================
    # IRX es yield anual en % -> anual decimal
    rf_anual = (irx_yield / 100.0).reindex(retornos.index).ffill()

    # Convertir a diario (compatible con reward diario del entorno)
    rf_diario = (1.0 + rf_anual) ** (1.0 / FACTOR_ANUALIZACION) - 1.0

    # Guardar rf full
    rf_anual.to_csv(CARPETA_DATOS / "Raw" / "rf_anual_full.csv")
    rf_diario.to_csv(CARPETA_DATOS / "Raw" / "rf_diario_full.csv")

    # =========================
    # 5) FEATURES sin leakage
    # =========================
    features = construir_features(retornos)
    features.to_csv(CARPETA_DATOS / "Raw" / "features_full.csv")

    # =========================
    # 6) ESTADO B: features + retornos_lag1
    # =========================
    retornos_lag1 = retornos.shift(1).loc[features.index]

    datos_estado = pd.concat([features, retornos_lag1], axis=1).dropna()

    # Alineación final obligatoria
    fechas = datos_estado.index.intersection(retornos.index).intersection(rf_diario.index)
    datos_estado = datos_estado.loc[fechas]
    retornos_ok = retornos.loc[fechas]
    rf_diario_ok = rf_diario.loc[fechas]
    rf_anual_ok = rf_anual.loc[fechas]

    # Guardar estado full
    datos_estado.to_csv(CARPETA_DATOS / "Raw" / "datos_estado_full.csv")

    # =========================
    # 7) SPLIT TEMPORAL
    # =========================
    datos_estado_train = datos_estado.loc[datos_estado.index < FECHA_FIN_TRAIN]
    datos_estado_validation = datos_estado.loc[
        (datos_estado.index >= FECHA_FIN_TRAIN) &
        (datos_estado.index < FECHA_FIN_VALIDATION)
    ]
    datos_estado_test = datos_estado.loc[datos_estado.index >= FECHA_FIN_VALIDATION]

    retornos_train = retornos_ok.loc[datos_estado_train.index]
    retornos_validation = retornos_ok.loc[datos_estado_validation.index]
    retornos_test = retornos_ok.loc[datos_estado_test.index]

    rf_diario_train = rf_diario_ok.loc[datos_estado_train.index]
    rf_diario_validation = rf_diario_ok.loc[datos_estado_validation.index]
    rf_diario_test = rf_diario_ok.loc[datos_estado_test.index]

    rf_anual_train = rf_anual_ok.loc[datos_estado_train.index]
    rf_anual_validation = rf_anual_ok.loc[datos_estado_validation.index]
    rf_anual_test = rf_anual_ok.loc[datos_estado_test.index]

    # =========================
    # 8) GUARDAR SPLITS
    # =========================
    # Estado
    datos_estado_train.to_csv(CARPETA_DATOS / "Train" / "datos_estado_train.csv")
    datos_estado_validation.to_csv(CARPETA_DATOS / "Validation" / "datos_estado_validation.csv")
    datos_estado_test.to_csv(CARPETA_DATOS / "Test" / "datos_estado_test.csv")

    # Retornos (reward)
    retornos_train.to_csv(CARPETA_DATOS / "Train" / "retornos_train.csv")
    retornos_validation.to_csv(CARPETA_DATOS / "Validation" / "retornos_validation.csv")
    retornos_test.to_csv(CARPETA_DATOS / "Test" / "retornos_test.csv")

    # rf diario (SAC)
    rf_diario_train.to_csv(CARPETA_DATOS / "Train" / "rf_diario_train.csv")
    rf_diario_validation.to_csv(CARPETA_DATOS / "Validation" / "rf_diario_validation.csv")
    rf_diario_test.to_csv(CARPETA_DATOS / "Test" / "rf_diario_test.csv")

    # rf anual (Markowitz)
    rf_anual_train.to_csv(CARPETA_DATOS / "Train" / "rf_anual_train.csv")
    rf_anual_validation.to_csv(CARPETA_DATOS / "Validation" / "rf_anual_validation.csv")
    rf_anual_test.to_csv(CARPETA_DATOS / "Test" / "rf_anual_test.csv")

    # Universo (reproducibilidad)
    with open(CARPETA_DATOS / "Raw" / "universo_tickers.json", "w", encoding="utf-8") as f:
        json.dump(ACTIVOS, f, ensure_ascii=False, indent=2)

    # =========================
    # 9) LOG FINAL
    # =========================
    print("✅ Dataset generado correctamente.")
    print("Ruta Datos:", CARPETA_DATOS.resolve())
    print("Full:", datos_estado.shape, retornos_ok.shape, rf_diario_ok.shape)
    print("Train:", datos_estado_train.shape, retornos_train.shape, rf_diario_train.shape)
    print("Validation:", datos_estado_validation.shape, retornos_validation.shape, rf_diario_validation.shape)
    print("Test:", datos_estado_test.shape, retornos_test.shape, rf_diario_test.shape)


if __name__ == "__main__":
    main()