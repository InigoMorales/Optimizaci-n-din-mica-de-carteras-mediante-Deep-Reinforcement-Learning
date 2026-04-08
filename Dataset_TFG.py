from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ACTIVOS = [
    # Renta variable US
    "^GSPC",   # S&P 500 - large cap blend
    "^NDX",    # Nasdaq 100 - growth/tech
    "IWM",     # Russell 2000 - small cap
    "XLF",     # Financieras US
    "XLE",     # Energía US

    # Renta variable internacional
    "EEM",     # Emerging Markets
    "EWJ",     # Japón
    "FEZ",     # Euro Stoxx 50

    # Renta fija
    "SHY",     # Bonos corto plazo (1-3 años)
    "AGG",     # Aggregate bonds (intermedio)
    "TLT",     # Bonos largo plazo (20+ años)
    "TIP",     # Bonos ligados a inflación
    "LQD",     # High yield bonds

    # Commodities
    "GC=F",    # Oro
    "HG=F",    # Cobre 

    # Real estate
    "VNQ",     # REITs US

    # Risk-free proxy
    "^IRX",    # T-Bill yield (no es activo invertible, solo para rf)
]

FECHA_INICIO = "2004-01-01"
FECHA_FIN = "2026-01-01"

FECHA_FIN_TRAIN = "2016-01-01"
FECHA_FIN_VALIDATION = "2020-01-01"

FACTOR_ANUALIZACION = 52.0


def _encontrar_raiz_proyecto() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent] + list(p.parents):
        if (parent / "Datos").exists() or (parent / "src").exists():
            return parent
    return p.parent


RAIZ_PROYECTO = _encontrar_raiz_proyecto()
CARPETA_DATOS = RAIZ_PROYECTO / "Datos"

for sub in ["Raw", "Train", "Validation", "Test"]:
    (CARPETA_DATOS / sub).mkdir(parents=True, exist_ok=True)


def construir_features(retornos: pd.DataFrame, precios: pd.DataFrame) -> pd.DataFrame:
    ret = retornos.sort_index().copy()
    px = precios.sort_index().copy()

    retorno_1w = px.pct_change(1).add_suffix("_r1w")
    retorno_4w = px.pct_change(4).add_suffix("_r4w")
    retorno_12w = px.pct_change(12).add_suffix("_r12w")

    vol_4w = ret.rolling(4).std(ddof=1).add_suffix("_vol4w")
    vol_12w = ret.rolling(12).std(ddof=1).add_suffix("_vol12w")

    dd_12w = (px / px.rolling(12).max() - 1.0).add_suffix("_dd12w")

    ma_4w = (px / px.rolling(4).mean() - 1.0).add_suffix("_ma4w")
    ma_12w = (px / px.rolling(12).mean() - 1.0).add_suffix("_ma12w")

    feats = pd.concat(
        [retorno_1w, retorno_4w, retorno_12w, vol_4w, vol_12w, dd_12w, ma_4w, ma_12w],
        axis=1
    )

    feats = feats.shift(1)
    feats = feats.dropna()
    return feats


def main() -> None:
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
    precios = precios.ffill()
    precios = precios.resample("W-FRI").last().dropna(how="all")
    precios.to_csv(CARPETA_DATOS / "Raw" / "precios_adjclose_full.csv")

    if "^IRX" not in precios.columns:
        raise ValueError("No se encontró '^IRX' en los datos descargados.")

    irx_yield = precios["^IRX"].copy()
    precios_activos = precios.drop(columns=["^IRX"])

    retornos = precios_activos.pct_change(fill_method=None)
    retornos = retornos.dropna(how="any")
    retornos.to_csv(CARPETA_DATOS / "Raw" / "retornos_full.csv")

    rf_anual = (irx_yield / 100.0).reindex(retornos.index).ffill()
    rf_semanal = (1.0 + rf_anual) ** (1.0 / FACTOR_ANUALIZACION) - 1.0

    rf_anual.to_csv(CARPETA_DATOS / "Raw" / "rf_anual_full.csv")
    rf_semanal.to_csv(CARPETA_DATOS / "Raw" / "rf_semanal_full.csv")

    features = construir_features(retornos, precios_activos)
    features.to_csv(CARPETA_DATOS / "Raw" / "features_full.csv")

    retornos_lag1 = retornos.shift(1).loc[features.index]
    datos_estado = pd.concat([features, retornos_lag1], axis=1).dropna()

    fechas = datos_estado.index.intersection(retornos.index).intersection(rf_semanal.index)
    datos_estado = datos_estado.loc[fechas]
    retornos_ok = retornos.loc[fechas]
    rf_semanal_ok = rf_semanal.loc[fechas]
    rf_anual_ok = rf_anual.loc[fechas]

    datos_estado.to_csv(CARPETA_DATOS / "Raw" / "datos_estado_full.csv")

    datos_estado_train = datos_estado.loc[datos_estado.index < FECHA_FIN_TRAIN]
    datos_estado_validation = datos_estado.loc[
        (datos_estado.index >= FECHA_FIN_TRAIN) &
        (datos_estado.index < FECHA_FIN_VALIDATION)
    ]
    datos_estado_test = datos_estado.loc[datos_estado.index >= FECHA_FIN_VALIDATION]

    retornos_train = retornos_ok.loc[datos_estado_train.index]
    retornos_validation = retornos_ok.loc[datos_estado_validation.index]
    retornos_test = retornos_ok.loc[datos_estado_test.index]

    rf_semanal_train = rf_semanal_ok.loc[datos_estado_train.index]
    rf_semanal_validation = rf_semanal_ok.loc[datos_estado_validation.index]
    rf_semanal_test = rf_semanal_ok.loc[datos_estado_test.index]

    rf_anual_train = rf_anual_ok.loc[datos_estado_train.index]
    rf_anual_validation = rf_anual_ok.loc[datos_estado_validation.index]
    rf_anual_test = rf_anual_ok.loc[datos_estado_test.index]

    datos_estado_train.to_csv(CARPETA_DATOS / "Train" / "datos_estado_train.csv")
    datos_estado_validation.to_csv(CARPETA_DATOS / "Validation" / "datos_estado_validation.csv")
    datos_estado_test.to_csv(CARPETA_DATOS / "Test" / "datos_estado_test.csv")

    retornos_train.to_csv(CARPETA_DATOS / "Train" / "retornos_train.csv")
    retornos_validation.to_csv(CARPETA_DATOS / "Validation" / "retornos_validation.csv")
    retornos_test.to_csv(CARPETA_DATOS / "Test" / "retornos_test.csv")

    rf_semanal_train.to_csv(CARPETA_DATOS / "Train" / "rf_semanal_train.csv")
    rf_semanal_validation.to_csv(CARPETA_DATOS / "Validation" / "rf_semanal_validation.csv")
    rf_semanal_test.to_csv(CARPETA_DATOS / "Test" / "rf_semanal_test.csv")

    rf_anual_train.to_csv(CARPETA_DATOS / "Train" / "rf_anual_train.csv")
    rf_anual_validation.to_csv(CARPETA_DATOS / "Validation" / "rf_anual_validation.csv")
    rf_anual_test.to_csv(CARPETA_DATOS / "Test" / "rf_anual_test.csv")

    # =========================================================
    # COVARIANZAS PRECALCULADAS POR SPLIT (sin leakage)
    # Train solo ve retornos de train, validation solo de validation, etc.
    # =========================================================
    cov_train = retornos_train.cov()
    cov_train.to_csv(CARPETA_DATOS / "Train" / "covarianzas_train.csv")

    cov_validation = retornos_validation.cov()
    cov_validation.to_csv(CARPETA_DATOS / "Validation" / "covarianzas_validation.csv")

    cov_test = retornos_test.cov()
    cov_test.to_csv(CARPETA_DATOS / "Test" / "covarianzas_test.csv")

    with open(CARPETA_DATOS / "Raw" / "universo_tickers.json", "w", encoding="utf-8") as f:
        json.dump(ACTIVOS, f, ensure_ascii=False, indent=2)

    print("✅ Dataset generado correctamente.")
    print("Ruta Datos:", CARPETA_DATOS.resolve())
    print("Train:", datos_estado_train.shape, retornos_train.shape)
    print("Validation:", datos_estado_validation.shape, retornos_validation.shape)
    print("Test:", datos_estado_test.shape, retornos_test.shape)
    print("Covarianzas train:", cov_train.shape)
    print("Covarianzas validation:", cov_validation.shape)
    print("Covarianzas test:", cov_test.shape)


if __name__ == "__main__":
    main()