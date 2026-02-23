import pandas as pd
from entorno_cartera import EntornoCartera
from politicas import pesos_iguales_rebalanceo, pesos_iguales_siempre, pesos_markowitz


retornos = pd.read_csv("returns_train_2010_2015.csv", index_col=0, parse_dates=True).sort_index()
entorno = EntornoCartera(retornos, retornos)

curva_rebalance = entorno.ejecutar_backtest(pesos_iguales_rebalanceo(entorno))
print("Ajustando para que siempre pesen lo mismo:\n", curva_rebalance.tail())

curva_hold = entorno.ejecutar_backtest(pesos_iguales_siempre(entorno))
print("Inicialmente equiponderado, luego sin ajustes:\n", curva_hold.tail())

curva_markowitz = entorno.ejecutar_backtest(pesos_markowitz(entorno))
print("Markowitz:\n", curva_markowitz.tail())