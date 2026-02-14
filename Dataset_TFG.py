import yfinance as yf
import pandas as pd
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent

activos = [ #USA: S&P500 y NASDAQ
            "^GSPC", "^NDX",
            #Global: MSCI World y Emergentes
            "URTH", "EEM",
            #Europa
            "^STOXX50E", "^GDAXI", "^IBEX", "^FCHI",
            "FTSEMIB.MI", "^FTSE", "^AEX", "^SSMI",
            #Commodities ORO, PLATA PETRÓLEO
            "GLD", "SLV", "USO",
            #Crypto BITCOIN Y ETHEREUM
            "BTC-USD", "ETH-USD",
            #Renta Fija: Agregados, largo plazo, corto plazo, y de riesgo
            "AGG", "TLT", "SHY", "HYG", "BNDX",
            #Alternativos: Inmobiliario y deuda
            "VNQ", "BIL"]

start = "2010-01-01"
end   = "2020-01-01"  

data = yf.download(activos, start=start, end=end, auto_adjust=False, progress=False)

# ---------- Adj Close ----------
adj_close = data["Adj Close"]
adj_close.index = pd.to_datetime(adj_close.index).normalize()
adj_close = adj_close.sort_index()
adj_close = adj_close.dropna(how="all")
adj_close = adj_close[~adj_close.index.duplicated(keep="first")]
adj_close = adj_close.ffill().dropna()

adj_close.to_csv(BASE_DIR / "adj_close_2010_2019.csv")

print("CSV guardados en", BASE_DIR)

