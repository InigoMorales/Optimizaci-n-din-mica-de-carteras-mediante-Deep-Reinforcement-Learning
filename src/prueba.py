import pandas as pd
import numpy as np

from entorno_cartera import EntornoCartera

# =============================
# 1) Cargar datos
# =============================
features = pd.read_csv(
    "features_train_2010_2015.csv",
    index_col=0,
    parse_dates=True
).sort_index()

retornos = pd.read_csv(
    "retornos_train_2010_2015.csv",
    index_col=0,
    parse_dates=True
).sort_index()

print("Features shape:", features.shape)
print("Retornos shape:", retornos.shape)

rf_anual = (
    pd.read_csv("rf_train_2010_2015.csv", index_col=0, parse_dates=True)
    .squeeze("columns")
    .sort_index()
)

# alinear a fechas de retornos y convertir a diario compuesto
rf_anual = rf_anual.reindex(retornos.index).ffill().fillna(0.0)
rf_diario = (1.0 + rf_anual) ** (1.0 / 252.0) - 1.0

# =============================
# 2) Crear entorno
# =============================
entorno = EntornoCartera(
    datos_estado=features,
    retornos_diarios=retornos,
    coste_transaccion=0.001,
    valor_inicial=1000.0,
    pesos_iniciales="iguales",
    rf_diario=rf_diario
)

# =============================
# 3) Reset y comprobar estado
# =============================
estado = entorno.reset()

print("\nDimensión estado:", estado.shape)
print("Primer estado (primeros 10 valores):")
print(estado[:10])

print("\nNúmero activos:", entorno.numero_activos)

# =============================
# 4) Dar un paso con pesos iguales
# =============================
pesos = np.ones(entorno.numero_activos) / entorno.numero_activos

estado2, reward, done, info = entorno.step(pesos)

print("\nReward primer paso:", reward)
print("Valor cartera tras 1 día:", info["valor_cartera"])
print("Rotación:", info["rotacion"])
print("Peso cash inicio:", info["peso_cash_inicio"])
print("Peso cash fin:", info["peso_cash_fin"])

# =============================
# 5) Ejecutar mini backtest
# =============================
curva = entorno.ejecutar_backtest(
    lambda estado: np.ones(entorno.numero_activos) / entorno.numero_activos
)

print("\nBacktest completado.")
print("Valor final:", curva.iloc[-1])
print("Longitud curva:", len(curva))

# =============================
# 6) Prueba anti-leakage (rápida)
# =============================
# Si tus features usan shift(1), la primera fecha de features debería ser posterior
print("\nPrimera fecha retornos:", retornos.index[0])
print("Primera fecha features:", features.index[0])

# Y además, el estado en t usa features[t], mientras que el retorno aplicado en step(t)
# es retornos[t]. Si no hay leakage, features[t] no debería depender de retornos[t].
# Una prueba práctica: correlación entre una feature de momentum y retorno del mismo día
col = features.columns[0]
aligned = features[[col]].join(retornos.iloc[:, 0].rename("r0"), how="inner")
corr_same_day = aligned[col].corr(aligned["r0"])
print(f"Corr(feature '{col}' vs retorno mismo día del activo0):", corr_same_day)

