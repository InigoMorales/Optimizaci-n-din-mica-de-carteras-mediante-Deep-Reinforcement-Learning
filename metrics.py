# metrics.py
import numpy as np
import pandas as pd

def curve_to_returns(curva: pd.Series) -> pd.Series:
    return curva.pct_change().dropna()

def max_drawdown(curva: pd.Series) -> float:
    peak = curva.cummax()
    dd = (curva / peak) - 1.0
    return float(dd.min())

def resumen_metricas(curva: pd.Series, factor_anualizacion: int = 252) -> dict:
    r = curve_to_returns(curva)
    if len(r) == 0:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}

    # CAGR aproximado usando días
    n_dias = len(curva) - 1
    cagr = float((curva.iloc[-1] / curva.iloc[0]) ** (factor_anualizacion / max(n_dias, 1)) - 1.0)

    vol = float(r.std(ddof=1) * np.sqrt(factor_anualizacion))
    ret_anual = float(r.mean() * factor_anualizacion)
    sharpe = float(ret_anual / vol) if vol > 0 else np.nan
    mdd = max_drawdown(curva)

    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": mdd}

def tabla_metricas(curvas: dict[str, pd.Series], factor_anualizacion: int = 252) -> pd.DataFrame:
    filas = []
    for nombre, curva in curvas.items():
        m = resumen_metricas(curva, factor_anualizacion=factor_anualizacion)
        m["estrategia"] = nombre
        filas.append(m)
    df = pd.DataFrame(filas).set_index("estrategia")
    return df