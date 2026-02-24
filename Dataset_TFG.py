import yfinance as yf
import pandas as pd
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent

activos = [ #USA: S&P500, NASDAQ y Small Caps
            "^GSPC", "^NDX", "IWM",
            # Emerging Markets
            "EEM",
            #Europa (Core + ESPAÑA)
            "^STOXX50E", "^IBEX",
            # Japón 
            "EWJ",
            #Commodities ORO y PLATA
            "GLD", "SLV",
            #Renta Fija: Agregados, largo plazo, corto plazo, y de riesgo
            "AGG", "TLT", "SHY", "HYG",
            #Alternativos: Inmobiliario y deuda
            "VNQ", "BIL", "^IRX"]

start = "2010-01-01"
end   = "2020-01-01"  

data = yf.download(activos, start=start, end=end, auto_adjust=False, progress=False)

# ---------- Precios ajustados ----------
precios_ajustados = data["Adj Close"]
precios_ajustados.index = pd.to_datetime(precios_ajustados.index).normalize()
precios_ajustados = precios_ajustados.sort_index()
precios_ajustados = precios_ajustados.dropna(how="all")
precios_ajustados = precios_ajustados[~precios_ajustados.index.duplicated(keep="first")]
precios_ajustados = precios_ajustados.ffill()
precios_ajustados = precios_ajustados.dropna(how="all")

precios_ajustados.to_csv(BASE_DIR / "adj_close_2010_2019.csv")

#Retornos mide la variación diaria .pct_change hace p1/p0-1. 
#El .iloc[1:] es para eliminar la primera fila que queda vacía por el cálculo de retornos.
retornos = precios_ajustados.pct_change().iloc[1:]

# ---------- División temporal ----------
# Train: 2010-2015, Validación: 2016-2018, Test: 2019
fecha_fin_train = "2016-01-01"
fecha_fin_validation = "2019-01-01"

precios_train = precios_ajustados.loc[precios_ajustados.index < fecha_fin_train]
precios_validation = precios_ajustados.loc[(precios_ajustados.index >= fecha_fin_train) & (precios_ajustados.index < fecha_fin_validation)]
precios_test = precios_ajustados.loc[precios_ajustados.index >= fecha_fin_validation]

retornos_train = retornos.loc[retornos.index < fecha_fin_train]
retornos_validation = retornos.loc[(retornos.index >= fecha_fin_train) & (retornos.index < fecha_fin_validation)]
retornos_test = retornos.loc[retornos.index >= fecha_fin_validation]

# ---------- Guardar splits ----------
precios_train.to_csv(BASE_DIR / "precios_train_2010_2015.csv")
precios_validation.to_csv(BASE_DIR / "precios_validation_2016_2018.csv")
precios_test.to_csv(BASE_DIR / "precios_test_2019.csv")

retornos_train.to_csv(BASE_DIR / "returns_train_2010_2015.csv")
retornos_validation.to_csv(BASE_DIR / "returns_validation_2016_2018.csv")
retornos_test.to_csv(BASE_DIR / "returns_test_2019.csv")

print(retornos["^IRX"].mean() * 252/100)

print(retornos["^GSPC"].mean() * 252*100)


