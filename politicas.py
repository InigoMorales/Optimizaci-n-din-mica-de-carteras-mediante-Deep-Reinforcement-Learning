import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# FUNCIONES FINANCIERAS
# ============================================================

def calcular_retorno_cartera(pesos, retornos_esperados):
    """
    Retorno esperado = sumatorio (peso_i * mu_i)
    """
    retorno = 0.0
    for i in range(len(pesos)):
        retorno += pesos[i] * retornos_esperados[i]
    return float(retorno)

def calcular_varianza_cartera(pesos, matriz_covarianzas):
    """
    Varianza = sumatorio_i sumatorio_j (peso_i * cov_ij * peso_j)
    """
    numero_activos = len(pesos)
    varianza = 0.0

    for i in range(numero_activos):
        for j in range(numero_activos):
            varianza += (pesos[i] * matriz_covarianzas[i, j] * pesos[j])

    return float(varianza)

def calcular_metricas_cartera(pesos, retornos_esperados, matriz_covarianzas):
    """
    Devuelve:
    - retorno esperado
    - volatilidad
    - varianza
    """
    varianza = calcular_varianza_cartera(pesos, matriz_covarianzas)
    retorno = calcular_retorno_cartera(pesos, retornos_esperados)
    volatilidad = float(np.sqrt(max(varianza, 0.0)))

    return retorno, volatilidad, varianza

# =====================
# POLÍTICAS SENCILLAS
# =====================

def pesos_iguales_rebalanceo(entorno):
    """
    Devuelve una política que equipondera y rebalancea todos los días.
    """
    def politica(_estado):
        return np.ones(entorno.numero_activos) / entorno.numero_activos
    return politica

def pesos_iguales_siempre(entorno):
    """
    Devuelve una política que equipondera solo el primer día y luego NO rebalancea (devuelve None).
    """
    def politica(_estado):
        if entorno.indice_tiempo == 0:
            return np.ones(entorno.numero_activos) / entorno.numero_activos
        return None
    return politica

# ============================================================
# OPTIMIZACIÓN MARKOWITZ
# ============================================================

def restriccion_suma_pesos(pesos):
    """
    Impone que la suma de los pesos sea 1. Devuelve 0 si la suma de pesos es 1
    """
    return float(np.sum(pesos) - 1.0)

def restriccion_retorno_objetivo(pesos, retornos_esperados, retorno_objetivo):
    """
    Impone que el retorno esperado sea igual al objetivo. 
    Devuelve 0 si el retorno esperado es igual al objetivo.
    """
    retorno = calcular_retorno_cartera(pesos, retornos_esperados)
    return float(retorno - retorno_objetivo)

def resolver_minima_varianza_con_retorno_objetivo(
    retornos_esperados,
    matriz_covarianzas,
    retorno_objetivo,
    solo_largos=True
):

    numero_activos = len(retornos_esperados)
    pesos_iniciales = np.ones(numero_activos) / numero_activos

    def funcion_objetivo(pesos):
        return calcular_varianza_cartera(pesos, matriz_covarianzas)

    restricciones = [
        {"type": "eq", "fun": restriccion_suma_pesos},
        {"type": "eq", "fun": restriccion_retorno_objetivo, "args": (retornos_esperados, retorno_objetivo)}
    ]

    if solo_largos:
        limites = [(0.0, 1.0)] * numero_activos
    else:
        limites = [(-1.0, 2.0)] * numero_activos

    resultado = minimize(
        funcion_objetivo,
        x0=pesos_iniciales,
        method="SLSQP",
        bounds=limites,
        constraints=restricciones
    )

    return resultado

def resolver_minima_varianza_global(
    retornos_esperados,
    matriz_covarianzas,
    solo_largos=True
):
    """
    GMV: Minimiza varianza sin fijar retorno.
    """

    numero_activos = len(retornos_esperados)
    pesos_iniciales = np.ones(numero_activos) / numero_activos

    def funcion_objetivo(pesos):
        return calcular_varianza_cartera(pesos, matriz_covarianzas)

    restricciones = [
        {"type": "eq", "fun": restriccion_suma_pesos}
    ]

    if solo_largos:
        limites = [(0.0, 1.0)] * numero_activos
    else:
        limites = [(-1.0, 2.0)] * numero_activos

    resultado = minimize(
        funcion_objetivo,
        x0=pesos_iniciales,
        method="SLSQP",
        bounds=limites,
        constraints=restricciones,
    )

    return resultado

# ============================================================
# CONSTRUCCIÓN DE FRONTERA EFICIENTE
# ============================================================

def construir_frontera_eficiente(
    retornos_historicos: pd.DataFrame,
    numero_puntos: int = 60,
    solo_largos: bool = True,
    factor_anualizacion: int | None = 252,
    mostrar_grafico: bool = True,
):
    """
    Construye la frontera eficiente resolviendo el 
    problema de mínima varianza para distintos retornos objetivo.
    """
    retornos_historicos = retornos_historicos.dropna()
    lista_activos = list(retornos_historicos.columns)
    numero_activos = len(lista_activos)

    if numero_activos < 2:
        raise ValueError("Se necesitan al menos 2 activos.")

    # Estimaciones
    retornos_esperados = retornos_historicos.mean().to_numpy()
    matriz_covarianzas = retornos_historicos.cov().to_numpy()

    if factor_anualizacion is not None:
        retornos_esperados = retornos_esperados * factor_anualizacion
        matriz_covarianzas = matriz_covarianzas * factor_anualizacion

    # GMV
    resultado_gmv = resolver_minima_varianza_global(
        retornos_esperados,
        matriz_covarianzas,
        solo_largos
    )

    pesos_gmv = resultado_gmv.x
    retorno_gmv, volatilidad_gmv, _ = calcular_metricas_cartera(pesos_gmv,retornos_esperados,matriz_covarianzas)

    # Rejilla de retornos objetivo
    retornos_objetivo = np.linspace(
        float(retornos_esperados.min()),
        float(retornos_esperados.max()),
        numero_puntos
    )

    datos_frontera = []

    for retorno_objetivo in retornos_objetivo:
        resultado = resolver_minima_varianza_con_retorno_objetivo(
            retornos_esperados,
            matriz_covarianzas,
            retorno_objetivo,
            solo_largos
        )

        if not resultado.success:
            continue

        pesos = resultado.x
        retorno, volatilidad, _ = calcular_metricas_cartera(pesos,retornos_esperados,matriz_covarianzas) 

        fila = {
            "retorno": retorno,
            "volatilidad": volatilidad
        }

        for i, activo in enumerate(lista_activos):
            fila[activo] = pesos[i]

        datos_frontera.append(fila)

    frontera = pd.DataFrame(datos_frontera).sort_values("volatilidad")
    # quedarse solo con la parte creciente en retorno
    frontera = frontera[frontera["retorno"].cummax() == frontera["retorno"]]

    n = len(frontera)

    perfil_1 = frontera.iloc[0]  # GMV
    perfil_2 = frontera.iloc[int(0.25 * n)]
    perfil_3 = frontera.iloc[int(0.50 * n)]
    perfil_4 = frontera.iloc[int(0.75 * n)]
    perfil_5 = frontera.iloc[-1] 

    # =========================
    # GRÁFICO
    # =========================
    if mostrar_grafico:
        plt.figure()

        plt.plot(frontera["volatilidad"], frontera["retorno"], linewidth=2)

        plt.scatter(perfil_1["volatilidad"], perfil_1["retorno"], marker="X", s=90, color = "green", label="Muy conservador")
        plt.scatter(perfil_2["volatilidad"], perfil_2["retorno"], marker="X", s=90, color = "blue", label="Conservador")
        plt.scatter(perfil_3["volatilidad"], perfil_3["retorno"], marker="X", s=90, color = "yellow", label="Neutro")
        plt.scatter(perfil_4["volatilidad"], perfil_4["retorno"], marker="X", s=90, color = "orange", label="Arriesgado")
        plt.scatter(perfil_5["volatilidad"], perfil_5["retorno"], marker="X", s=90, color = "red", label="Muy Arriesgado")

        plt.xlabel("Riesgo (volatilidad)")
        plt.ylabel("Rentabilidad esperada")
        plt.title("Frontera eficiente de Markowitz")
        plt.grid(True)
        plt.show()

    return frontera