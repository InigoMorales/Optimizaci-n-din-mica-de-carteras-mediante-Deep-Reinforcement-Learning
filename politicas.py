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

