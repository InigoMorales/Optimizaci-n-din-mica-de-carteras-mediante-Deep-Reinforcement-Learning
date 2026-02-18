import numpy as np
import pandas as pd


def normalizar_pesos(pesos: np.ndarray) -> np.ndarray:
    """
    Convierte un vector de pesos en un vector válido de cartera: No permite negativos (long-only) y fuerza que sumen 1
    """
    pesos = np.asarray(pesos, dtype=np.float64)

    # Evitar negativos
    pesos = np.clip(pesos, 0.0, None)

    suma = pesos.sum()
    if suma <= 0:
        # Si todo es 0, devolvemos pesos iguales
        return np.ones_like(pesos) / len(pesos)

    return pesos / suma

class EntornoCartera:
    """
    Entorno de inversión (MDP) para asignación dinámica de cartera.
    - Estado: un vector numérico para el día actual
    - Acción: pesos de cartera para los activos
    - Recompensa: retorno de cartera - coste de transacción
    """
    def __init__(
        self,
        datos_estado: pd.DataFrame, #DataFrame index=fechas, columns=features
        retornos_diarios: pd.DataFrame, #DataFrame index=fechas, columns=activos con retornos diarios
        coste_transaccion: float = 0.001,  # 10 bps por rebalanceo
        valor_inicial: float = 1000.0,
        pesos_iniciales: str = "iguales"
    ):
        # Alineamos fechas por seguridad: solo usamos fechas comunes
        fechas_comunes = datos_estado.index.intersection(retornos_diarios.index)
        if len(fechas_comunes) == 0:
            raise ValueError(
                "No hay fechas comunes entre datos_estado y retornos_diarios. "
                "Comprueba que ambos índices son fechas (DatetimeIndex) y coinciden."
            )

        self.datos_estado = datos_estado.loc[fechas_comunes].copy()
        self.retornos_diarios = retornos_diarios.loc[fechas_comunes].copy()

        self.lista_activos = list(self.retornos_diarios.columns)
        self.numero_activos = len(self.lista_activos)

        self.coste_transaccion = float(coste_transaccion)
        self.valor_inicial = float(valor_inicial)
        self.pesos_iniciales = pesos_iniciales

        # Variables internas del episodio
        self.indice_tiempo = 0
        self.valor_cartera = self.valor_inicial
        self.pesos_anteriores = None

        # Arrancamos
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reinicia el entorno al inicio del dataset. Devuelve el estado inicial.
        """
        self.indice_tiempo = 0
        self.valor_cartera = self.valor_inicial

        self.pesos_anteriores = np.ones(self.numero_activos) / self.numero_activos

        return self._obtener_estado_actual()
    
    def _obtener_estado_actual(self) -> np.ndarray:
        """
        Devuelve el estado (features) del día actual como vector numpy.
        """
        fila_estado = self.datos_estado.iloc[self.indice_tiempo]
        return fila_estado.to_numpy(dtype=np.float32)
    
    def step(self, nuevos_pesos) -> tuple[np.ndarray | None, float, bool, dict]:
        """
        Aplica una acción (pesos), calcula reward, avanza el tiempo.
        Incluye DRIFT: aunque no rebalancees, los pesos cambian al final del día por los retornos.
        """
        # 1) Retornos del día actual (los necesitamos tanto para retorno como para drift)
        retornos_hoy = self.retornos_diarios.iloc[self.indice_tiempo].to_numpy(dtype=np.float64)
        retornos_hoy = np.nan_to_num(retornos_hoy, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) Elegir pesos al INICIO del día (después de rebalance si lo hay)
        pesos_previos = self.pesos_anteriores.copy()

        if nuevos_pesos is None:
            # No operamos: mantenemos los pesos con los que llegamos al día
            pesos_actuales = pesos_previos
            rotacion = 0.0
            coste = 0.0
        else:
            # Operamos: rebalanceamos a los pesos objetivo
            pesos_actuales = normalizar_pesos(nuevos_pesos)

            # 3) Coste de transacción (turnover) por cambiar de pesos_previos -> pesos_actuales
            rotacion = float(np.sum(np.abs(pesos_actuales - pesos_previos)))
            coste = self.coste_transaccion * rotacion

        # 4) Retorno de la cartera (bruto, antes de costes)
        retorno_cartera = float(np.dot(pesos_actuales, retornos_hoy))

        # 5) Recompensa (retorno neto de costes)
        recompensa = retorno_cartera - coste

        # 6) Actualizar valor cartera
        self.valor_cartera *= (1.0 + recompensa)

        # 7) DRIFT: actualizar pesos al FINAL del día por los retornos (buy-and-hold real)
        crecimiento = 1.0 + retornos_hoy
        pesos_despues = pesos_actuales * crecimiento  # riqueza relativa por activo
        suma = float(pesos_despues.sum())

        if suma > 0.0:
            pesos_despues = pesos_despues / suma
        else:
            # Caso extremo: si algo va muy mal numéricamente, mantenemos los pesos actuales
            pesos_despues = pesos_actuales.copy()

        # Guardar pesos (ya con drift) para el siguiente paso
        self.pesos_anteriores = pesos_despues

        # 8) Avanzar día
        self.indice_tiempo += 1
        terminado = self.indice_tiempo >= len(self.datos_estado)

        # 9) Preparar outputs
        siguiente_estado = None if terminado else self._obtener_estado_actual()

        info = {
            "valor_cartera": self.valor_cartera,
            "retorno_cartera": retorno_cartera,
            "coste_transaccion": coste,
            "rotacion": rotacion,
            "pesos_inicio": pesos_actuales,   # pesos usados para el retorno del día (post-rebalance)
            "pesos_fin": pesos_despues        # pesos tras drift (para el siguiente día)
        }

        return siguiente_estado, recompensa, terminado, info

    def ejecutar_backtest(self, funcion_pesos) -> pd.Series:
        """
        Ejecuta un backtest llamando a funcion_pesos(estado) -> pesos
        Devuelve una serie con el valor de la cartera en el tiempo.
        """
        self.reset()

        fechas = list(self.datos_estado.index)
        valores = [self.valor_cartera]

        terminado = False
        estado = self._obtener_estado_actual()

        while not terminado:
            pesos = funcion_pesos(estado)
            estado, recompensa, terminado, info = self.step(pesos)
            valores.append(info["valor_cartera"])

        # valores tiene una entrada más (incluye el valor inicial)
        fechas_con_inicial = [fechas[0]] + fechas
        return pd.Series(valores, index=fechas_con_inicial, name="valor_cartera")



