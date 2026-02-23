import numpy as np

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

def pesos_markowitz(
    entorno,
    ventana: int = 60,
    lambda_riesgo: float = 10.0,
):
    """
    Política Markowitz media-varianza (long-only, suma=1).
    - ventana: número de días para estimar media y covarianza
    - lambda_riesgo: aversión al riesgo (mayor => más conservador)
    """

    def politica(_estado):

        t = entorno.indice_tiempo

        # Necesitamos suficientes datos históricos
        if t < ventana:
            # Si no hay suficiente histórico, equiponderamos
            return np.ones(entorno.numero_activos) / entorno.numero_activos

        # Ventana rolling de retornos pasados
        retornos_hist = entorno.retornos_diarios.iloc[t-ventana:t].to_numpy(dtype=np.float64)

        # Estimaciones
        mu = np.mean(retornos_hist, axis=0)          # vector medias
        Sigma = np.cov(retornos_hist, rowvar=False)  # matriz covarianza

        # Regularización numérica para estabilidad
        Sigma += 1e-6 * np.eye(entorno.numero_activos)

        try:
            # Solución analítica sin restricciones:
            # w* ∝ Σ^{-1} μ
            inv_Sigma = np.linalg.inv(Sigma)
            w = inv_Sigma @ mu

        except np.linalg.LinAlgError:
            # Si falla inversión → equiponderar
            return np.ones(entorno.numero_activos) / entorno.numero_activos

        # Forzar long-only
        w = np.clip(w, 0.0, None)

        # Normalizar para que sumen 1
        suma = w.sum()
        if suma <= 0:
            return np.ones(entorno.numero_activos) / entorno.numero_activos

        w = w / suma

        return w

    return politica