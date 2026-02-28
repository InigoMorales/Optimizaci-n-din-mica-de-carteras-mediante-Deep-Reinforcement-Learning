# ejecutor.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from entorno_cartera import EntornoCartera
from politicas import (
    pesos_iguales_rebalanceo, pesos_iguales_siempre,
    resolver_minima_varianza_global, resolver_max_retorno,
    resolver_max_sharpe, markowitz_gmv_rolling, markowitz_max_retorno_rolling,
    markowitz_tangente_rolling, construir_frontera_eficiente, cartera_optima_en_funcion_riesgo
)

# ============================================================
# CONFIG
# ============================================================
RETURNS_PATH = "retornos_2005_2019.csv"
RF_PATH      = "rf_2005_2019.csv"

VENTANA = 252
REBALANCE_CADA = 21
SOLO_LARGOS = True
ANUALIZAR = 252
ALPHA = 0.7


def backtest_con_pesos(entorno, funcion_pesos):
    """
    Ejecuta un backtest y guarda:
    - pesos_fin (riesgo) de cada día
    - peso_cash_fin
    - valor_cartera
    Devuelve un DataFrame indexado por fecha.
    """
    entorno.reset()

    registros = []
    terminado = False
    estado = entorno._obtener_estado_actual()

    while not terminado:
        pesos = funcion_pesos(estado)
        estado, recompensa, terminado, info = entorno.step(pesos)

        # fecha del día que acabamos de simular
        fecha = entorno.datos_estado.index[entorno.indice_tiempo - 1]

        row = {
            "valor": float(info["valor_cartera"]),
            "cash": float(info.get("peso_cash_fin", 0.0)),
        }

        w_fin = info["pesos_fin"]
        for i, activo in enumerate(entorno.lista_activos):
            row[activo] = float(w_fin[i])

        registros.append((fecha, row))

    df = pd.DataFrame([r for _, r in registros], index=[f for f, _ in registros])
    df.index.name = "fecha"
    return df


def imprimir_pesos_fin_de_ano(df_pesos, titulo, top_n=None):
    """
    Imprime los pesos a fin de año. Si top_n se indica, imprime solo los top_n activos (por peso)
    + cash, para cada año.
    """
    # Nota: si "YE" te diera problemas en tu versión de pandas, cambia a "Y"
    df_y = df_pesos.resample("YE").last()

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

    if top_n is None:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df_y.round(4))
        return

    activos = [c for c in df_y.columns if c not in ("valor", "cash")]
    for fecha, fila in df_y.iterrows():
        pesos_activos = fila[activos].sort_values(ascending=False)
        top = pesos_activos.head(top_n)

        print(f"\n{fecha.date()} | valor={fila['valor']:.2f} | cash={fila['cash']:.4f}")
        for a, w in top.items():
            print(f"  {a:>10s}: {w:.4f}")


def politica_estatica(w: np.ndarray):
    w = np.asarray(w, dtype=float).copy()
    def pol(_estado, w=w):
        return w
    return pol


def main():
    # =========================
    # Cargar datos (universo completo)
    # =========================
    retornos_riesgo = (pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
        .sort_index()
        .dropna(how="all")
    )

    rf_dinamico = (
        pd.read_csv(RF_PATH, index_col=0, parse_dates=True)
        .squeeze("columns")
        .sort_index()
    )

    # Alinear fechas
    fechas = retornos_riesgo.index.intersection(rf_dinamico.index)
    if len(fechas) == 0:
        raise ValueError("No hay fechas comunes entre returns y rf. Revisa los CSV.")

    retornos_riesgo = retornos_riesgo.loc[fechas].dropna()
    rf_dinamico = rf_dinamico.loc[retornos_riesgo.index].ffill().fillna(0.0)

    retornos_riesgo_2005_2009 = retornos_riesgo.loc[
        (retornos_riesgo.index >= "2005-01-01") & (retornos_riesgo.index < "2010-01-01")
    ]

    retornos_riesgo_2010_2019 = retornos_riesgo.loc[
        (retornos_riesgo.index >= "2010-01-01") & (retornos_riesgo.index < "2020-01-01")
    ]

    rf_dinamico_2005_2009 = rf_dinamico.loc[
        (rf_dinamico.index >= "2005-01-01") & (rf_dinamico.index < "2010-01-01")
    ]

    rf_dinamico_2010_2019 = rf_dinamico.loc[
        (rf_dinamico.index >= "2010-01-01") & (rf_dinamico.index < "2020-01-01")
    ]

    rf_estatico = float(rf_dinamico.mean())
    rf_estatico_2005_2009 = float(rf_dinamico_2005_2009.mean())
    rf_estatico_2010_2019 = float(rf_dinamico_2010_2019.mean())

    print(f"N días: {len(retornos_riesgo)} | N activos: {retornos_riesgo.shape[1]}")
    print(f"rf medio anual: {rf_estatico * 100:.3f}%")
    print(f"rf medio anual 2005-2009: {rf_estatico_2005_2009 * 100:.3f}%")
    print(f"rf medio anual 2010-2019: {rf_estatico_2010_2019 * 100:.3f}%")

    # =========================
    # Entorno
    # =========================
    # Mantengo tu creación tal cual (si tu EntornoCartera ahora requiere rf_diario, pásalo aquí)
    entorno = EntornoCartera(retornos_riesgo_2010_2019, retornos_riesgo_2010_2019)

    # =========================
    # Baselines
    # =========================
    curva_rebalance = entorno.ejecutar_backtest(pesos_iguales_rebalanceo(entorno))
    curva_hold = entorno.ejecutar_backtest(pesos_iguales_siempre(entorno))

    # =========================
    # Markowitz ESTÁTICO (3)
    # =========================
    mu = retornos_riesgo_2005_2009.mean().to_numpy() * ANUALIZAR
    Sigma = retornos_riesgo_2005_2009.cov().to_numpy() * ANUALIZAR

    resultado_gmv = resolver_minima_varianza_global(mu, Sigma, solo_largos=SOLO_LARGOS)
    w_gmv = resultado_gmv.x.copy()

    resultado_max_retorno = resolver_max_retorno(mu, solo_largos=SOLO_LARGOS)
    w_max_retorno = resultado_max_retorno.x.copy()

    resultado_sharpe = resolver_max_sharpe(mu, Sigma, rf=rf_estatico_2005_2009, solo_largos=SOLO_LARGOS)
    w_sharpe = resultado_sharpe.x.copy()

    curva_gmv = entorno.ejecutar_backtest(politica_estatica(w_gmv))
    curva_max_retorno = entorno.ejecutar_backtest(politica_estatica(w_max_retorno))
    curva_sharpe = entorno.ejecutar_backtest(politica_estatica(w_sharpe))

    # =========================
    # Markowitz ROLLING (guardar políticas y luego ejecutar)
    # =========================
    pol_gmv_roll = markowitz_gmv_rolling(
        entorno, retornos_full=retornos_riesgo, window_years=5, rebalance_cada=REBALANCE_CADA,
        anualizar=ANUALIZAR, solo_largos=SOLO_LARGOS, min_obs=VENTANA
    )

    pol_max_retorno_roll = markowitz_max_retorno_rolling(
        entorno, retornos_full=retornos_riesgo, window_years=5, rebalance_cada=REBALANCE_CADA,
        anualizar=ANUALIZAR, solo_largos=SOLO_LARGOS, min_obs=VENTANA
    )

    pol_sharpe_roll = markowitz_tangente_rolling(
        entorno, retornos_full=retornos_riesgo, rf_full=rf_dinamico, window_years=5, rebalance_cada=REBALANCE_CADA,
        anualizar=ANUALIZAR, solo_largos=SOLO_LARGOS, rf_floor=0.0, min_obs=VENTANA
    )

    # =========================
    # Cartera óptima en función del riesgo (política) y luego backtest
    # =========================
    pol_optima_con_rf = cartera_optima_en_funcion_riesgo(
        entorno, retornos_full=retornos_riesgo, rf_full=rf_dinamico, rebalance_cada=REBALANCE_CADA,
        anualizar=ANUALIZAR, solo_largos=SOLO_LARGOS, rf_floor=0.0, min_obs=VENTANA, alpha=ALPHA
    )

    # Ahora sí: crear curvas llamando al entorno
    curva_gmv_roll = entorno.ejecutar_backtest(pol_gmv_roll)
    curva_max_retorno_roll = entorno.ejecutar_backtest(pol_max_retorno_roll)
    curva_sharpe_roll = entorno.ejecutar_backtest(pol_sharpe_roll)
    curva_optima_con_rf = entorno.ejecutar_backtest(pol_optima_con_rf)

    # =========================
    # Ejecutar y printear fin de año (PASANDO POLÍTICAS, no curvas)
    # =========================
    df_gmv = backtest_con_pesos(entorno, pol_gmv_roll)
    imprimir_pesos_fin_de_ano(df_gmv, "Rolling GMV - pesos fin de año", top_n=6)

    df_maxret = backtest_con_pesos(entorno, pol_max_retorno_roll)
    imprimir_pesos_fin_de_ano(df_maxret, "Rolling Max Retorno - pesos fin de año", top_n=6)

    df_tan = backtest_con_pesos(entorno, pol_sharpe_roll)
    imprimir_pesos_fin_de_ano(df_tan, "Rolling Tangente (Sharpe) - pesos fin de año", top_n=6)

    df_tan_alpha = backtest_con_pesos(entorno, pol_optima_con_rf)
    imprimir_pesos_fin_de_ano(df_tan_alpha, f"Rolling Tangente + rf (alpha={ALPHA}) - pesos fin de año", top_n=6)

    # =========================
    # Plot final (lo dejas como lo tenías, lo comento tal cual)
    # =========================
    # plt.figure()
    # plt.plot(curva_rebalance, label="Iguales rebalanceo")
    # plt.plot(curva_hold, label="Iguales buy & hold")
    #
    # plt.plot(curva_gmv, label="Estático GMV (min riesgo)")
    # plt.plot(curva_max_retorno, label="Estático Max retorno")
    # plt.plot(curva_sharpe, label=f"Estático Tangente (Sharpe, rf={rf_estatico_2010_2019:.2%})")
    #
    # plt.plot(curva_gmv_roll, "--", label="Rolling GMV")
    # plt.plot(curva_max_retorno_roll, "--", label="Rolling Max retorno")
    # plt.plot(curva_sharpe_roll, "--", label="Rolling Tangente (Sharpe)")
    #
    # plt.plot(curva_optima_con_rf, "--", label=f"Cartera óptima función riesgo (alpha={ALPHA})")
    #
    # plt.title("Backtest (Universo completo)")
    # plt.grid(True)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()