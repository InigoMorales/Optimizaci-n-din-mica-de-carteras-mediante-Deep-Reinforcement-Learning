import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from entorno_cartera import EntornoCartera
from politicas import (
    pesos_iguales_rebalanceo, pesos_iguales_siempre,
    resolver_minima_varianza_global, resolver_max_retorno, resolver_max_sharpe,
    markowitz_gmv_rolling, markowitz_max_retorno_rolling, markowitz_tangente_rolling, construir_frontera_eficiente
)

# =========================
# Cargar datos
# =========================
retornos = pd.read_csv("returns_train_2010_2015.csv", index_col=0, parse_dates=True).sort_index()
retornos = retornos.dropna()

# =========================
# rf desde IRX (anualizado)
# =========================
if "^IRX" not in retornos.columns:
    raise ValueError("No existe la columna '^IRX' en el CSV. Añádela o cambia el nombre en el código.")

irx = retornos["^IRX"].copy()
# excluir BIL del universo optimizable
retornos_riesgo = retornos.drop(columns=["^IRX"])
rf_dinamico = irx.reindex(retornos_riesgo.index).fillna(0.0)
rf_estatico = float(rf_dinamico.mean() / 100.0)

# Entorno con activos de riesgo (sin BIL)
entorno = EntornoCartera(retornos_riesgo, retornos_riesgo)
construir_frontera_eficiente(retornos_riesgo, rf=rf_estatico)

# =========================
# Baselines
# =========================
curva_rebalance = entorno.ejecutar_backtest(pesos_iguales_rebalanceo(entorno))
curva_hold = entorno.ejecutar_backtest(pesos_iguales_siempre(entorno))

# =========================
# Markowitz ESTÁTICO (3)
# =========================
mu = retornos_riesgo.mean().to_numpy() * 252
Sigma = retornos_riesgo.cov().to_numpy() * 252

res_gmv = resolver_minima_varianza_global(mu, Sigma, solo_largos=True)
w_gmv = res_gmv.x.copy()

res_maxret = resolver_max_retorno(mu, solo_largos=True)
w_maxret = res_maxret.x.copy()

res_tan = resolver_max_sharpe(mu, Sigma, rf=rf_estatico, solo_largos=True)
w_tan = res_tan.x.copy()

def politica_estatica(w):
    w = w.copy()
    def pol(_estado, w=w):
        return w
    return pol

curva_gmv = entorno.ejecutar_backtest(politica_estatica(w_gmv))
curva_maxret = entorno.ejecutar_backtest(politica_estatica(w_maxret))
curva_tan = entorno.ejecutar_backtest(politica_estatica(w_tan))

# =========================
# Markowitz ROLLING (3)
# =========================
ventana = 252
rebalance_cada = 21

curva_gmv_roll = entorno.ejecutar_backtest(
    markowitz_gmv_rolling(entorno, ventana=ventana, rebalance_cada=rebalance_cada, solo_largos=True)
)
curva_maxret_roll = entorno.ejecutar_backtest(
    markowitz_max_retorno_rolling(entorno, ventana=ventana, rebalance_cada=rebalance_cada, solo_largos=True)
)
curva_tan_roll = entorno.ejecutar_backtest(
    markowitz_tangente_rolling(entorno, bil_series=rf_dinamico, ventana=ventana, rebalance_cada=rebalance_cada, solo_largos=True, rf_floor=0.0)
)

# =========================
# Plot final (pocas curvas)
# =========================
plt.figure()
plt.plot(curva_rebalance, label="Iguales rebalanceo")
plt.plot(curva_hold, label="Iguales buy & hold")

plt.plot(curva_gmv, label="Estático GMV (min riesgo)")
plt.plot(curva_maxret, label="Estático Max retorno")
plt.plot(curva_tan, label=f"Estático Tangente (Sharpe, rf={rf_estatico:.2%})")

plt.plot(curva_gmv_roll, "--", label="Rolling GMV")
plt.plot(curva_maxret_roll, "--", label="Rolling Max retorno")
plt.plot(curva_tan_roll, "--", label="Rolling Tangente (Sharpe)")

plt.legend()
plt.grid(True)
plt.show()