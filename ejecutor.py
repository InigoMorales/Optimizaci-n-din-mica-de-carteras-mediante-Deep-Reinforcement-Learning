import pandas as pd
import matplotlib.pyplot as plt
from entorno_cartera import EntornoCartera
from politicas import pesos_iguales_rebalanceo, pesos_iguales_siempre, construir_frontera_eficiente


retornos = pd.read_csv("returns_train_2010_2015.csv", index_col=0, parse_dates=True).sort_index()
retornos = retornos.dropna()
entorno = EntornoCartera(retornos, retornos)

# Ver frontera eficiente
construir_frontera_eficiente(retornos)
frontera = construir_frontera_eficiente(retornos, factor_anualizacion=252,mostrar_grafico=False).sort_values("volatilidad").reset_index(drop=True)

activos = list(retornos.columns)

# 2) Elegir 5 perfiles a lo largo de la frontera
n = len(frontera)
idx = [0, int(0.25*(n-1)), int(0.50*(n-1)), int(0.75*(n-1)), n-1]
perfiles = [frontera.iloc[i] for i in idx]

# 3) Backtests básicos
curva_rebalance = entorno.ejecutar_backtest(pesos_iguales_rebalanceo(entorno))
curva_hold = entorno.ejecutar_backtest(pesos_iguales_siempre(entorno))

# 4) Backtests Markowitz (5 perfiles)
curvas_markowitz = []
labels = [
    "Markowitz P1 (GMV)",
    "Markowitz P2",
    "Markowitz P3",
    "Markowitz P4",
    "Markowitz P5 (máx retorno)"
]
colores_markowitz = [
    "green",     # P1 (GMV)
    "blue",      # P2
    "yellow",    # P3
    "orange",     # P4
    "red"        # P5
]

for perfil in perfiles:
    pesos = perfil[activos].to_numpy()
    pesos = pesos.copy()

    def politica_markowitz(_estado, w = pesos):
        return w
    curvas_markowitz.append(entorno.ejecutar_backtest(politica_markowitz))


# Graficar resultados
plt.figure()
plt.plot(curva_rebalance, label="Iguales rebalanceo")
plt.plot(curva_hold, label="Iguales buy & hold")
for curva, label, color in zip(curvas_markowitz, labels, colores_markowitz):
    plt.plot(curva, label=label, color=color)
plt.legend()
plt.grid(True)
plt.show()

print("Final rebalance:", curva_rebalance.iloc[-1])
print("Final hold:", curva_hold.iloc[-1])
for i, curva in enumerate(curvas_markowitz, 1):
    print(f"Final Markowitz P{i}:", curva.iloc[-1])

