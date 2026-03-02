import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    # RUTAS CSV (Datos/*)
    # =========================
    BASE_DIR = Path(__file__).resolve().parent
    DATOS_DIR = BASE_DIR / "Datos"  # si ejecutor.py está en la raíz
    # Si ejecutor.py está en src/, usa esto:
    # DATOS_DIR = BASE_DIR.parent / "Datos"

    RAW_DIR = DATOS_DIR / "Raw"
    TRAIN_DIR = DATOS_DIR / "Train"
    VAL_DIR = DATOS_DIR / "Validation"
    TEST_DIR = DATOS_DIR / "Test"

    # =========================
    # Cargar datos
    # - Train: para estimar Markowitz estático
    # - Train+Validation: para rolling (histórico disponible antes de test)
    # - Test: para evaluar (entorno principal)
    # =========================
    retornos_train = pd.read_csv(TRAIN_DIR / "retornos_train.csv", index_col=0, parse_dates=True).sort_index()
    rf_anual_train = pd.read_csv(TRAIN_DIR / "rf_anual_train.csv", index_col=0, parse_dates=True).squeeze("columns").sort_index()

    retornos_validation = pd.read_csv(VAL_DIR / "retornos_validation.csv", index_col=0, parse_dates=True).sort_index()
    rf_anual_validation = pd.read_csv(VAL_DIR / "rf_anual_validation.csv", index_col=0, parse_dates=True).squeeze("columns").sort_index()

    datos_estado_test = pd.read_csv(TEST_DIR / "datos_estado_test.csv", index_col=0, parse_dates=True).sort_index()
    retornos_test = pd.read_csv(TEST_DIR / "retornos_test.csv", index_col=0, parse_dates=True).sort_index()
    rf_diario_test = pd.read_csv(TEST_DIR / "rf_diario_test.csv", index_col=0, parse_dates=True).squeeze("columns").sort_index()

    # rf anual para rolling / info (no lo necesita el entorno, pero sí tus políticas rolling de tangencia)
    rf_anual_test = pd.read_csv(TEST_DIR / "rf_anual_test.csv", index_col=0, parse_dates=True).squeeze("columns").sort_index()

    # Full (opcional para logs)
    # datos_estado_full = pd.read_csv(RAW_DIR / "datos_estado_full.csv", index_col=0, parse_dates=True).sort_index()

    # Construir series "full-histórico disponible antes de test" para rolling
    retornos_hist = pd.concat([retornos_train, retornos_validation]).sort_index()
    rf_anual_hist = pd.concat([rf_anual_train, rf_anual_validation]).sort_index()

    print(f"[TEST] N días: {len(retornos_test)} | N activos: {retornos_test.shape[1]}")
    print(f"[TEST] rf medio diario: {rf_diario_test.mean() * 100:.3f}%")
    print(f"[TRAIN] rf medio anual: {rf_anual_train.mean() * 100:.3f}%")

    # =========================
    # Entorno (EVALUACIÓN EN TEST)
    # =========================
    entorno = EntornoCartera(
        datos_estado=datos_estado_test,
        retornos_diarios=retornos_test,
        coste_transaccion=0.001,
        valor_inicial=1000.0,
        pesos_iniciales="iguales",
        rf_diario=rf_diario_test
    )

    # =========================
    # Baselines
    # =========================
    curva_rebalance = entorno.ejecutar_backtest(pesos_iguales_rebalanceo(entorno))
    curva_hold = entorno.ejecutar_backtest(pesos_iguales_siempre(entorno))

    # =========================
    # Markowitz ESTÁTICO (estimado en TRAIN, evaluado en TEST)
    # =========================
    mu = retornos_train.mean().to_numpy() * ANUALIZAR
    Sigma = retornos_train.cov().to_numpy() * ANUALIZAR

    resultado_gmv = resolver_minima_varianza_global(mu, Sigma, solo_largos=SOLO_LARGOS)
    w_gmv = resultado_gmv.x.copy()

    resultado_max_retorno = resolver_max_retorno(mu, solo_largos=SOLO_LARGOS)
    w_max_retorno = resultado_max_retorno.x.copy()

    rf_estatico_train = float(rf_anual_train.mean())  # rf anual medio en entrenamiento
    resultado_sharpe = resolver_max_sharpe(mu, Sigma, rf=rf_estatico_train, solo_largos=SOLO_LARGOS)
    w_sharpe = resultado_sharpe.x.copy()

    curva_gmv = entorno.ejecutar_backtest(politica_estatica(w_gmv))
    curva_max_retorno = entorno.ejecutar_backtest(politica_estatica(w_max_retorno))
    curva_sharpe = entorno.ejecutar_backtest(politica_estatica(w_sharpe))

    # =========================
    # Markowitz ROLLING (usa histórico TRAIN+VAL, evalúa en TEST)
    # =========================
    pol_gmv_roll = markowitz_gmv_rolling(
        entorno,
        retornos_full=retornos_hist,
        window_years=5,
        rebalance_cada=REBALANCE_CADA,
        anualizar=ANUALIZAR,
        solo_largos=SOLO_LARGOS,
        min_obs=VENTANA
    )

    pol_max_retorno_roll = markowitz_max_retorno_rolling(
        entorno,
        retornos_full=retornos_hist,
        window_years=5,
        rebalance_cada=REBALANCE_CADA,
        anualizar=ANUALIZAR,
        solo_largos=SOLO_LARGOS,
        min_obs=VENTANA
    )

    pol_sharpe_roll = markowitz_tangente_rolling(
        entorno,
        retornos_full=retornos_hist,
        rf_full=rf_anual_hist,
        window_years=5,
        rebalance_cada=REBALANCE_CADA,
        anualizar=ANUALIZAR,
        solo_largos=SOLO_LARGOS,
        rf_floor=0.0,
        min_obs=VENTANA
    )

    pol_optima_con_rf = cartera_optima_en_funcion_riesgo(
        entorno,
        retornos_full=retornos_hist,
        rf_full=rf_anual_hist,
        rebalance_cada=REBALANCE_CADA,
        anualizar=ANUALIZAR,
        solo_largos=SOLO_LARGOS,
        rf_floor=0.0,
        min_obs=VENTANA,
        alpha=ALPHA
    )

    curva_gmv_roll = entorno.ejecutar_backtest(pol_gmv_roll)
    curva_max_retorno_roll = entorno.ejecutar_backtest(pol_max_retorno_roll)
    curva_sharpe_roll = entorno.ejecutar_backtest(pol_sharpe_roll)
    curva_optima_con_rf = entorno.ejecutar_backtest(pol_optima_con_rf)

    # =========================
    # Plot final
    # =========================
    plt.figure()
    plt.plot(curva_rebalance, label="Iguales rebalanceo")
    plt.plot(curva_hold, label="Iguales buy & hold")

    plt.plot(curva_gmv, label="Estático GMV (min riesgo)")
    plt.plot(curva_max_retorno, label="Estático Max retorno")
    plt.plot(curva_sharpe, label=f"Estático Tangente (Sharpe, rf={rf_estatico_train:.2%})")

    plt.plot(curva_gmv_roll, "--", label="Rolling GMV")
    plt.plot(curva_max_retorno_roll, "--", label="Rolling Max retorno")
    plt.plot(curva_sharpe_roll, "--", label="Rolling Tangente (Sharpe)")

    plt.plot(curva_optima_con_rf, "--", label=f"Cartera óptima función riesgo (alpha={ALPHA})")

    plt.title("Backtest (TEST)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()