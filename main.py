import pandas as pd
from entorno_cartera import EntornoCartera
from politicas import pesos_iguales_rebalanceo, pesos_iguales_siempre

retornos = pd.read_csv("returns_train_2010_2015.csv", index_col=0, parse_dates=True).sort_index()
datos_estado = retornos.copy()

entorno = EntornoCartera(datos_estado, retornos)
print("Entorno creado.")

curva_reb = entorno.ejecutar_backtest(pesos_iguales_rebalanceo(entorno))
print("Política de rebalanceo:\n",curva_reb.tail())

curva_hold = entorno.ejecutar_backtest(pesos_iguales_siempre(entorno))
print("Política equal_weights constates:\n",curva_hold.tail())
