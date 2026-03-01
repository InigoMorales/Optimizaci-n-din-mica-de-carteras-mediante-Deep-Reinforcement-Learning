import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# UNIVERSO DE ACTIVOS
# ============================================================

activos = [
    # USA SP500 Nasdaq y Small Caps
    "^GSPC", "^NDX", "IWM",
    # Emerging Markets
    "EEM",
    # Europa
    "^IBEX",
    # Japón
    "EWJ",
    # Commodities oro y plata
    "GC=F", "SI=F",
    # Renta Fija
    "AGG", "TLT", "SHY",
    # Alternativos
    "^RMZ",
    # Risk-free yield
    "^IRX"
]

start = "2004-01-01"
end   = "2026-01-01"

#data = yf.download(activos, start=start, end=end, auto_adjust=False, progress=False)
data_completa_2004_2025 = yf.download(activos, start=start, end=end, auto_adjust=False, progress=False)

# ============================================================
# PRECIOS AJUSTADOS
# ============================================================

precios = data_completa_2004_2025["Adj Close"]
precios.index = pd.to_datetime(precios.index).normalize()
precios = precios.sort_index()
precios = precios[~precios.index.duplicated(keep="first")]
precios = precios.ffill()

# Guardamos precios completos
precios.to_csv(BASE_DIR / "precios_2004_2025.csv")

# ============================================================
# SEPARAR IRX (yield anual en %)
# ============================================================

if "^IRX" not in precios.columns:
    raise ValueError("No se encontró '^IRX' en los datos descargados.")

irx_yield = precios["^IRX"].copy()

# Activos de riesgo (quitar IRX)
precios_activos_completos = precios.drop(columns=["^IRX"])

# ============================================================
# RETORNOS ACTIVOS DE RIESGO (diarios)
# ============================================================

#pct.change calcula la variación porcentual respecto a su valor anterior: (P_t - P_{t-1}) / P_{t-1}
#el iloc[1:] se usa para eliminar la primera fila ya que no podrás calcular la variación del 1er dato
retornos_activos_completos = precios_activos_completos.pct_change().iloc[1:]
retornos_activos_completos = retornos_activos_completos.fillna(0.0)
retornos_activos_completos.to_csv(BASE_DIR / "retornos_2004_2025.csv")

# ============================================================
# CONSTRUCCIÓN CORRECTA DEL RISK-FREE
# ============================================================

# IRX está en yield anual en porcentaje (ej: 4.25) por eso se divide entre 100

rf_anual_completo = irx_yield / 100.0
rf_anual_completo = rf_anual_completo.iloc[1:]

# Alinear índices con retornos activos
rf_anual_completo = rf_anual_completo.reindex(retornos_activos_completos.index).ffill()
rf_anual_completo.to_csv(BASE_DIR / "rf_2004_2025.csv")

# ============================================================
# SPLIT TEMPORAL
# ============================================================

fecha_fin_train = "2016-01-01"
fecha_fin_validation = "2020-01-01"

# Precios
precios_train = precios_activos_completos.loc[
    (precios_activos_completos.index < fecha_fin_train) & (precios_activos_completos.index >= start)]

precios_validation = precios_activos_completos.loc[
    (precios_activos_completos.index >= fecha_fin_train) & (precios_activos_completos.index < fecha_fin_validation)
]
precios_test = precios_activos_completos.loc[precios_activos_completos.index >= fecha_fin_validation]

# Retornos
retornos_train = retornos_activos_completos.loc[
    (retornos_activos_completos.index < fecha_fin_train) & (retornos_activos_completos.index >= start)]

retornos_validation = retornos_activos_completos.loc[
    (retornos_activos_completos.index >= fecha_fin_train) & (retornos_activos_completos.index < fecha_fin_validation)]

retornos_test = retornos_activos_completos.loc[retornos_activos_completos.index >= fecha_fin_validation]

# rf
rf_train = rf_anual_completo.loc[rf_anual_completo.index < fecha_fin_train]
rf_validation = rf_anual_completo.loc[
    (rf_anual_completo.index >= fecha_fin_train) &
    (rf_anual_completo.index < fecha_fin_validation)
]
rf_test = rf_anual_completo.loc[rf_anual_completo.index >= fecha_fin_validation]

rf_diario = (1.0 + rf_anual_completo) ** (1.0 / 252.0) - 1.0

# ============================================================
# GUARDAR TODO
# ============================================================

retornos_train.to_csv(BASE_DIR / "retornos_train_2004_2015.csv")
retornos_validation.to_csv(BASE_DIR / "retornos_validation_2016_2019.csv")
retornos_test.to_csv(BASE_DIR / "retornos_test_2020_2025.csv")

rf_train.to_csv(BASE_DIR / "rf_train_2004_2015.csv")
rf_validation.to_csv(BASE_DIR / "rf_validation_2016_2019.csv")
rf_test.to_csv(BASE_DIR / "rf_test_2020_2025.csv")

# ============================================================
# CONSTRUCCIÓN DE FEATURES (SIN LEAKAGE)
# ============================================================

def construir_features(ret: pd.DataFrame) -> pd.DataFrame:
    ret = ret.sort_index().copy()

    # Momentum
    momentum_20 = ret.rolling(20).sum().add_suffix("_momentum20")
    momentum_60 = ret.rolling(60).sum().add_suffix("_momentum60")

    # Volatilidad
    volatilidad_20 = ret.rolling(20).std(ddof=1).add_suffix("_volatilidad20")
    volatilidad_60 = ret.rolling(60).std(ddof=1).add_suffix("_volatilidad60")

    # Correlación media 60d
    def correlacion_media(window_df: pd.DataFrame) -> float:
        correlacion = window_df.corr().to_numpy()
        n = correlacion.shape[0]
        if n <= 1:
            return 0.0
        return float((correlacion.sum() - np.trace(correlacion)) / (n * (n - 1)))

    correlacion_media_vals = []
    idx = ret.index

    for t in range(len(ret)):
        if t < 60:
            correlacion_media_vals.append(np.nan)
        else:
            w = ret.iloc[t-60:t]
            correlacion_media_vals.append(correlacion_media(w))

    correlacion_media_60 = pd.Series(correlacion_media_vals, index=idx, name="mean_corr_60")

    feats = pd.concat([momentum_20, momentum_60, volatilidad_20, volatilidad_60, correlacion_media_60], axis=1)

    # IMPORTANTE: evitar leakage
    feats = feats.shift(1)

    feats = feats.dropna()
    return feats

# ============================================================
# GENERAR FEATURES
# ============================================================

features_full = construir_features(retornos_activos_completos)
features_full.to_csv(BASE_DIR / "features_2004_2025.csv")

features_train = features_full.loc[retornos_train.index.intersection(features_full.index)]
features_validation = features_full.loc[retornos_validation.index.intersection(features_full.index)]
features_test = features_full.loc[retornos_test.index.intersection(features_full.index)]

features_train.to_csv(BASE_DIR / "features_train_2004_2015.csv")
features_validation.to_csv(BASE_DIR / "features_validation_2016_2019.csv")
features_test.to_csv(BASE_DIR / "features_test_2020_2025.csv")

print("Dataset generado correctamente.")
print("Media rf anual (train):", rf_train.mean() * 100.0 , "%")
print("Media retorno S&P500 anual (train):", retornos_train["^GSPC"].mean() * 252 *100.0, "%")
print("Media retorno Nasdaq anual (train):", retornos_train["^NDX"].mean() * 252 *100.0, "%")
print("Media retorno oro anual (train):", retornos_train["GC=F"].mean() * 252 *100.0, "%")
print("Media retorno IBEX35 anual (train):", retornos_train["^IBEX"].mean() * 252 *100.0, "%")
print("Media retorno IBEX35 anual:", retornos_activos_completos["^IBEX"].mean() * 252 *100.0, "%")
print("Media retorno Plata anual (train):", retornos_train["SI=F"].mean() * 252 *100.0, "%")
print("Media retorno Japón anual (train):", retornos_train["EWJ"].mean() * 252 *100.0, "%")
print("Media retorno Renta Fija L/P anual (train):", retornos_train["TLT"].mean() * 252 *100.0, "%")
print("Media retorno Renta Fija S/P anual (train):", retornos_train["SHY"].mean() * 252 *100.0, "%")