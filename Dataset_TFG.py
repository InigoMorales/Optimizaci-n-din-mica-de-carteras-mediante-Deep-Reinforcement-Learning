import yfinance as yf
import pandas as pd
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
    "^STOXX50E", "^IBEX",
    # Japón
    "EWJ",
    # Commodities oro y plata
    "GLD", "SLV",
    # Renta Fija
    "AGG", "TLT", "SHY", "HYG",
    # Alternativos
    "VNQ", "BIL",
    # Risk-free yield
    "^IRX"
]

vista_histroica = "2005-01-01" #Lo que vería el inversor en 2010
start = "2010-01-01"
end   = "2020-01-01"

#data = yf.download(activos, start=start, end=end, auto_adjust=False, progress=False)
data_completa_2005_2019 = yf.download(activos, start=vista_histroica, end=end, auto_adjust=False, progress=False)


# ============================================================
# PRECIOS AJUSTADOS
# ============================================================

precios = data_completa_2005_2019["Adj Close"]
precios.index = pd.to_datetime(precios.index).normalize()
precios = precios.sort_index()
precios = precios[~precios.index.duplicated(keep="first")]
precios = precios.ffill()

# Guardamos precios completos
precios.to_csv(BASE_DIR / "precios_2005_2019.csv")

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
retornos_activos_completos = retornos_activos_completos.dropna(how="all")
retornos_activos_completos.to_csv(BASE_DIR / "retornos_2005_2019.csv")

# ============================================================
# CONSTRUCCIÓN CORRECTA DEL RISK-FREE
# ============================================================

# IRX está en yield anual en porcentaje (ej: 4.25) por eso se divide entre 100

rf_anual_completo = irx_yield / 100.0
rf_anual_completo = rf_anual_completo.iloc[1:]

# Alinear índices con retornos activos
rf_anual_completo = rf_anual_completo.reindex(retornos_activos_completos.index).ffill()
rf_anual_completo.to_csv(BASE_DIR / "rf_2005_2019.csv")

# ============================================================
# SPLIT TEMPORAL
# ============================================================

fecha_fin_train = "2016-01-01"
fecha_fin_validation = "2019-01-01"

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

# ============================================================
# GUARDAR TODO
# ============================================================

retornos_train.to_csv(BASE_DIR / "retornos_train_2010_2015.csv")
retornos_validation.to_csv(BASE_DIR / "retornos_validation_2016_2018.csv")
retornos_test.to_csv(BASE_DIR / "retornos_test_2019.csv")

rf_train.to_csv(BASE_DIR / "rf_train_2010_2015.csv")
rf_validation.to_csv(BASE_DIR / "rf_validation_2016_2018.csv")
rf_test.to_csv(BASE_DIR / "rf_test_2019.csv")

print("Dataset generado correctamente.")
print("Media rf anual (train):", rf_train.mean() * 100.0 , "%")
print("Media retorno S&P500 anual (train):", retornos_train["^GSPC"].mean() * 252 *100.0, "%")
print("Media retorno Nasdaq anual (train):", retornos_train["^NDX"].mean() * 252 *100.0, "%")
print("Media retorno oro anual (train):", retornos_train["GLD"].mean() * 252 *100.0, "%")
print("Media retorno IBEX35 anual (train):", retornos_train["^IBEX"].mean() * 252 *100.0, "%")
print("Media retorno IBEX35 anual:", retornos_activos_completos["^IBEX"].mean() * 252 *100.0, "%")
print("Media retorno Plata anual (train):", retornos_train["SLV"].mean() * 252 *100.0, "%")
print("Media retorno Japón anual (train):", retornos_train["EWJ"].mean() * 252 *100.0, "%")
print("Media retorno Renta Fija L/P anual (train):", retornos_train["TLT"].mean() * 252 *100.0, "%")
print("Media retorno Renta Fija S/P anual (train):", retornos_train["SHY"].mean() * 252 *100.0, "%")