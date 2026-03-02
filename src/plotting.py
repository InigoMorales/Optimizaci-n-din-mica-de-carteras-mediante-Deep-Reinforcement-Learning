# plotting.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_curvas(curvas: dict[str, pd.Series], titulo: str = "Backtest", logy: bool = False):
    """
    curvas: dict {nombre: pd.Series(valor_cartera)}
    """
    plt.figure()
    for nombre, serie in curvas.items():
        plt.plot(serie, label=nombre)

    plt.title(titulo)
    plt.xlabel("Fecha")
    plt.ylabel("Valor cartera")
    plt.grid(True)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.show()