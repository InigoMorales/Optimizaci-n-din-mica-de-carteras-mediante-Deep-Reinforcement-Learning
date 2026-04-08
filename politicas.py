import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ============================================================
# FUNCIONES FINANCIERAS
# ============================================================

def calcular_retorno_cartera(pesos, retornos_esperados):
    retorno = 0.0
    for i in range(len(pesos)):
        retorno += pesos[i] * retornos_esperados[i]
    return float(retorno)

def calcular_varianza_cartera(pesos, matriz_covarianzas):
    n = len(pesos)
    varianza = 0.0
    for i in range(n):
        for j in range(n):
            varianza += pesos[i] * matriz_covarianzas[i, j] * pesos[j]
    return float(varianza)

def calcular_metricas_cartera(pesos, retornos_esperados, matriz_covarianzas):
    varianza = calcular_varianza_cartera(pesos, matriz_covarianzas)
    retorno = calcular_retorno_cartera(pesos, retornos_esperados)
    volatilidad = float(np.sqrt(max(varianza, 0.0)))
    return retorno, volatilidad, varianza

# ============================================================
# POLÍTICAS SENCILLAS
# ============================================================

def pesos_iguales_rebalanceo(entorno):
    def politica(_estado):
        return np.ones(entorno.numero_activos) / entorno.numero_activos
    return politica

def pesos_iguales_siempre(entorno):
    def politica(_estado):
        if entorno.indice_tiempo == 0:
            return np.ones(entorno.numero_activos) / entorno.numero_activos
        return None
    return politica

# ============================================================
# OPTIMIZACIÓN (restricciones / solvers)
# ============================================================

def restriccion_suma_pesos(pesos):
    return float(np.sum(pesos) - 1.0)

def restriccion_retorno_objetivo(pesos, retornos_esperados, retorno_objetivo):
    retorno = calcular_retorno_cartera(pesos, retornos_esperados)
    return float(retorno - retorno_objetivo)

def resolver_minima_varianza_con_retorno_objetivo(
    retornos_esperados,
    matriz_covarianzas,
    retorno_objetivo,
    solo_largos=True
):
    n = len(retornos_esperados)
    x0 = np.ones(n) / n

    def objetivo(w):
        return calcular_varianza_cartera(w, matriz_covarianzas)

    restricciones = [
        {"type": "eq", "fun": restriccion_suma_pesos},
        {"type": "eq", "fun": restriccion_retorno_objetivo, "args": (retornos_esperados, retorno_objetivo)},
    ]

    limites = [(0.0, 1.0)] * n if solo_largos else [(-1.0, 2.0)] * n

    return minimize(objetivo, x0=x0, method="SLSQP", bounds=limites, constraints=restricciones)

def resolver_minima_varianza_global(retornos_esperados, matriz_covarianzas, solo_largos=True):
    n = len(retornos_esperados)
    x0 = np.ones(n) / n

    def objetivo(w):
        return calcular_varianza_cartera(w, matriz_covarianzas)

    restricciones = [{"type": "eq", "fun": restriccion_suma_pesos}]
    limites = [(0.0, 1.0)] * n if solo_largos else [(-1.0, 2.0)] * n

    return minimize(objetivo, x0=x0, method="SLSQP", bounds=limites, constraints=restricciones)

def resolver_max_retorno(mu, solo_largos=True):
    """
    Max retorno esperado: max mu'w s.a sum(w)=1 y bounds.
    """
    n = len(mu)
    x0 = np.ones(n) / n

    def objetivo(w):
        return -float(np.dot(w, mu))  # maximizar retorno -> minimizar negativo

    restricciones = [{"type": "eq", "fun": restriccion_suma_pesos}]
    limites = [(0.0, 1.0)] * n if solo_largos else [(-1.0, 2.0)] * n

    return minimize(objetivo, x0=x0, method="SLSQP", bounds=limites, constraints=restricciones)

def resolver_max_sharpe(mu, Sigma, rf, solo_largos=True):
    """
    Maximiza Sharpe: (mu'w - rf)/sqrt(w'Sigma w) s.a sum(w)=1 y bounds.
    """
    n = len(mu)
    x0 = np.ones(n) / n
    rf = float(rf)

    def objetivo(w):
        retorno = float(np.dot(w, mu))
        varianza = float(np.dot(w, np.dot(Sigma, w)))
        vol = float(np.sqrt(max(varianza, 0.0)))

        if vol <= 0.0:
            return 1e6

        sharpe = (retorno - rf) / vol
        return -sharpe  # minimizar negativo

    restricciones = [{"type": "eq", "fun": restriccion_suma_pesos}]
    limites = [(0.0, 1.0)] * n if solo_largos else [(-1.0, 2.0)] * n

    return minimize(objetivo, x0=x0, method="SLSQP", bounds=limites, constraints=restricciones)

# ============================================================
# FRONTERA EFICIENTE
# ============================================================

def construir_frontera_eficiente(
    retornos_historicos: pd.DataFrame,
    numero_puntos: int = 60,
    rf: float | None = None,
    solo_largos: bool = True,
    factor_anualizacion: int | None = 52,
    mostrar_grafico: bool = True,
):
    retornos_historicos = retornos_historicos.dropna()
    lista_activos = list(retornos_historicos.columns)

    if len(lista_activos) < 2:
        raise ValueError("Se necesitan al menos 2 activos.")

    mu = retornos_historicos.mean().to_numpy()
    matriz_covarianzas = retornos_historicos.cov().to_numpy()

    if factor_anualizacion is not None:
        mu = mu * factor_anualizacion
        matriz_covarianzas = matriz_covarianzas * factor_anualizacion

    # GMV (solo para referencia)
    res_gmv = resolver_minima_varianza_global(mu, matriz_covarianzas, solo_largos=solo_largos)
    w_gmv = res_gmv.x
    ret_gmv, vol_gmv, _ = calcular_metricas_cartera(w_gmv, mu, matriz_covarianzas)

    retornos_obj = np.linspace(float(mu.min()), float(mu.max()), numero_puntos)

    filas = []
    for r_obj in retornos_obj:
        res = resolver_minima_varianza_con_retorno_objetivo(mu, matriz_covarianzas, r_obj, solo_largos=solo_largos)
        if not res.success:
            continue
        w = res.x
        ret, vol, _ = calcular_metricas_cartera(w, mu, matriz_covarianzas)

        fila = {"retorno": ret, "volatilidad": vol}
        for i, a in enumerate(lista_activos):
            fila[a] = w[i]
        filas.append(fila)

    frontera = pd.DataFrame(filas).sort_values("volatilidad")
    frontera = frontera[frontera["retorno"].cummax() == frontera["retorno"]]
    # =========================
    # CARTERA TANGENTE (Sharpe)
    # =========================
    retorno_tan = None
    vol_tan = None
    sharpe_tan = None

    if rf is not None:
        resultado_tan = resolver_max_sharpe(
            mu,
            matriz_covarianzas,
            rf=rf,
            solo_largos=solo_largos
        )

        if resultado_tan.success:
            pesos_tan = resultado_tan.x
            retorno_tan, vol_tan, _ = calcular_metricas_cartera(
                pesos_tan,
                mu,
                matriz_covarianzas
            )

            sharpe_tan = (retorno_tan - rf) / vol_tan

    if mostrar_grafico:
        plt.figure()
        plt.plot(frontera["volatilidad"], frontera["retorno"], linewidth=2, label="Frontera eficiente")
        if rf is not None and sharpe_tan is not None:
            # Línea de mercado de capitales
            vol_range = np.linspace(0, frontera["volatilidad"].max() * 1.1, 100)
            cml = rf + sharpe_tan * vol_range

            plt.plot(vol_range, cml, linestyle="--", label="Capital Market Line")

            # Punto tangente
            plt.scatter(vol_tan, retorno_tan,
                        marker="*", s=180, color="black",
                        label="Cartera Tangente (Sharpe)")
        plt.scatter(vol_gmv, ret_gmv, marker="X", s=90, color="green", label="GMV")
        plt.xlabel("Riesgo (volatilidad)")
        plt.ylabel("Rentabilidad esperada")
        plt.title("Frontera eficiente de Markowitz")
        plt.grid(True)
        plt.legend()
        plt.show()

    return frontera

# ============================================================
# POLÍTICAS ROLLING (3: GMV / MaxRet / Tangente)
# ============================================================

def markowitz_gmv_rolling(
        entorno, retornos_full: pd.DataFrame, window_years: int = 5, rebalance_cada: int = 4, 
        anualizar: int = 52, solo_largos: bool = True, min_obs: int = 52
        ):
    
    retornos_full = retornos_full.sort_index()
    fechas_env = entorno.retornos_diarios.index

    def politica(_estado):
        t = entorno.indice_tiempo
        if (t % rebalance_cada) != 0:
            return None

        fecha_actual = fechas_env[t]
        fecha_inicio = fecha_actual - pd.DateOffset(years=window_years)

        ventana_df = retornos_full.loc[
            (retornos_full.index >= fecha_inicio) & (retornos_full.index < fecha_actual)
        ].dropna()

        if len(ventana_df) < min_obs:
            return None

        mu = ventana_df.mean().to_numpy()
        Sigma = ventana_df.cov().to_numpy()
        if anualizar is not None:
            mu = mu * anualizar
            Sigma = Sigma * anualizar

        res = resolver_minima_varianza_global(mu, Sigma, solo_largos=solo_largos)
        if (not res.success) or (res.x is None):
            return None
        return res.x.copy()

    return politica

def markowitz_max_retorno_rolling(
    entorno, retornos_full: pd.DataFrame, window_years: int = 5, rebalance_cada: int = 4,
    anualizar: int = 52, solo_largos: bool = True, min_obs: int = 52
):
    retornos_full = retornos_full.sort_index()
    fechas_env = entorno.retornos_diarios.index

    def politica(_estado):
        t = entorno.indice_tiempo
        if (t % rebalance_cada) != 0:
            return None

        fecha_actual = fechas_env[t]
        fecha_inicio = fecha_actual - pd.DateOffset(years=window_years)

        ventana_df = retornos_full.loc[
            (retornos_full.index >= fecha_inicio) & (retornos_full.index < fecha_actual)
        ].dropna()

        if len(ventana_df) < min_obs:
            return None

        mu = ventana_df.mean().to_numpy()
        if anualizar is not None:
            mu = mu * anualizar

        res = resolver_max_retorno(mu, solo_largos=solo_largos)
        if (not res.success) or (res.x is None):
            return None
        return res.x.copy()

    return politica

def markowitz_tangente_rolling(
    entorno,
    retornos_full: pd.DataFrame,   # 2005-2020 (retornos diarios)
    rf_full: pd.Series,            # 2005-2020 (rf anual decimal)
    window_years: int = 5, rebalance_cada: int =4, anualizar: int = 52,
    solo_largos: bool = True, rf_floor: float = 0.0, min_obs: int = 52
):
    """
    Entorno empieza en 2010 (capital 1000 en 2010), pero la estimación usa histórico externo:
    en cada rebalanceo usa la ventana [fecha_actual - window_years, fecha_actual).
    """

    # Alinear full a fechas (por seguridad)
    retornos_full = retornos_full.sort_index()
    rf_full = rf_full.sort_index()

    fechas_env = entorno.retornos_diarios.index

    def politica(_estado):
        t = entorno.indice_tiempo

        # rebalanceo periódico
        if (t % rebalance_cada) != 0:
            return None

        fecha_actual = fechas_env[t]
        fecha_inicio = fecha_actual - pd.DateOffset(years=window_years)

        ventana_df = retornos_full.loc[(retornos_full.index >= fecha_inicio) & (retornos_full.index < fecha_actual)].dropna()
        if len(ventana_df) < min_obs:
            return None

        mu = ventana_df.mean().to_numpy()
        Sigma = ventana_df.cov().to_numpy()

        if anualizar is not None:
            mu = mu * anualizar
            Sigma = Sigma * anualizar

        rf_vent = rf_full.loc[(rf_full.index >= fecha_inicio) & (rf_full.index < fecha_actual)]
        rf_t = float(rf_vent.mean()) if len(rf_vent) else 0.0
        if rf_floor is not None:
            rf_t = max(float(rf_floor), rf_t)

        res = resolver_max_sharpe(mu, Sigma, rf=rf_t, solo_largos=solo_largos)
        if (not res.success) or (res.x is None):
            return None

        return res.x.copy()

    return politica

def cartera_optima_en_funcion_riesgo(
        entorno, retornos_full: pd.DataFrame, rf_full: pd.Series, alpha: float,
        window_years: int = 5, rebalance_cada: int = 4, anualizar: int = 52,
    solo_largos: bool = True, rf_floor: float = 0.0, min_obs: int = 52):
    """
    Coge la cartera tangente (max Sharpe) y mezcla con rf 
    según el alpha (aversión al riesgo) de cada ususario"""
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    # Reutilizamos política de markowitz_tangente_rolling para obtener la cartera tangente dinámica
    base_pol = markowitz_tangente_rolling(
        entorno=entorno,
        retornos_full=retornos_full,
        rf_full=rf_full,
        window_years=window_years,
        rebalance_cada=rebalance_cada,
        anualizar=anualizar,
        solo_largos=solo_largos,
        rf_floor=rf_floor,
        min_obs=min_obs,
    )

    def politica(_estado):
        w = base_pol(_estado)
        if w is None:
            return None
        return alpha * np.asarray(w, dtype=float)
    # si alpha = 0.7, entonces se asigna un 70% a la cartera tangente y el 30% restante se deja en cash (rf)
    return politica