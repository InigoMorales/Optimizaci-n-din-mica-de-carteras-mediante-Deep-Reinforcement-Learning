import pandas as pd
import numpy as np
from entorno_cartera import EntornoCartera

# Carga
retornos = pd.read_csv("returns_train_2010_2015.csv", index_col=0, parse_dates=True)
retornos = retornos.sort_index()

datos_estado = retornos.copy()  # estado mínimo para probar

try:
    entorno = EntornoCartera(datos_estado=datos_estado, retornos_diarios=retornos)
    print("Entorno creado.")

    def pesos_iguales(_estado):
        return np.ones(entorno.numero_activos) / entorno.numero_activos

    curva = entorno.ejecutar_backtest(pesos_iguales)
    print(curva.tail())
    print("Rentabilidad = ", (curva.iloc[-1] / curva.iloc[0] - 1) * 100, "%")

except Exception as e:
    print("❌ Error al crear/usar el entorno:", repr(e))
