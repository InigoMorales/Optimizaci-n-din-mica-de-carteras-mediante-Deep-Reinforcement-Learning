import numpy as np
import pandas as pd


def normalizar_pesos(pesos: np.ndarray) -> np.ndarray:
    """
    Convierte un vector cualquiera en una cartera válida long-only con suma 1.
    Incluye cash explícito como una dimensión más.
    """
    pesos = np.asarray(pesos, dtype=np.float64)
    pesos = np.nan_to_num(pesos, nan=0.0, posinf=0.0, neginf=0.0)
    pesos = np.clip(pesos, 0.0, None)

    suma = float(pesos.sum())
    if suma <= 0.0:
        return np.ones_like(pesos, dtype=np.float64) / len(pesos)

    return pesos / suma


class EntornoCartera:
    """
    Entorno de inversión (MDP) para asignación dinámica de cartera con CASH EXPLÍCITO.

    Convención de la acción:
    - La acción tiene dimensión N+1:
        [w_1, ..., w_N, w_cash]
    - Los N primeros pesos corresponden a activos riesgosos.
    - El último peso corresponde al activo cash (remunerado a rf_diario).
    - Long-only: todos los pesos >= 0
    - Sin apalancamiento: la suma total se fuerza a 1

    Notas:
    - retornos_diarios contiene SOLO activos riesgosos
    - rf_diario contiene el retorno diario del cash
    - El estado incluye:
        [features_del_mercado, pesos_previos_completos]
    """

    def __init__(
        self,
        datos_estado: pd.DataFrame,          # index=fechas, columns=features
        retornos_diarios: pd.DataFrame,      # index=fechas, columns=activos riesgosos
        coste_transaccion: float = 0.001,    # 10 bps por rebalanceo
        valor_inicial: float = 1000.0,
        pesos_iniciales: str = "iguales",
        rf_diario: pd.Series | None = None,  # index=fechas, retorno diario (decimal)
    ):
        # Alinear fechas comunes
        fechas_comunes = datos_estado.index.intersection(retornos_diarios.index)
        if len(fechas_comunes) == 0:
            raise ValueError(
                "No hay fechas comunes entre datos_estado y retornos_diarios. "
                "Comprueba que ambos índices son fechas (DatetimeIndex) y coinciden."
            )

        self.datos_estado = datos_estado.loc[fechas_comunes].copy()
        self.retornos_diarios = retornos_diarios.loc[fechas_comunes].copy()

        self.lista_activos_riesgo = list(self.retornos_diarios.columns)
        self.numero_activos_riesgo = len(self.lista_activos_riesgo)
        self.numero_activos_totales = self.numero_activos_riesgo + 1  # + cash explícito

        self.coste_transaccion = float(coste_transaccion)
        self.valor_inicial = float(valor_inicial)
        self.pesos_iniciales = pesos_iniciales

        # rf diario
        if rf_diario is not None:
            rf_diario = rf_diario.sort_index()
            rf_diario = rf_diario.reindex(self.retornos_diarios.index).ffill().fillna(0.0)
            self.rf_diario = rf_diario.astype(float)
        else:
            self.rf_diario = pd.Series(0.0, index=self.retornos_diarios.index, dtype=float)

        # Variables internas del episodio
        self.indice_tiempo = 0
        self.valor_cartera = self.valor_inicial

        # Ahora los pesos previos incluyen cash explícito:
        # [w_1, ..., w_N, w_cash]
        self.pesos_anteriores = None

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reinicia el entorno al inicio del dataset. Devuelve el estado inicial.
        """
        self.indice_tiempo = 0
        self.valor_cartera = self.valor_inicial


        if self.pesos_iniciales == "incluyendo_cash":
            # Reparte por igual entre riesgo y cash
            self.pesos_anteriores = (
                np.ones(self.numero_activos_totales, dtype=np.float64) / self.numero_activos_totales
            )      
        else:
            # Por defecto: fully invested en riesgo
            pesos_riesgo = np.ones(self.numero_activos_riesgo, dtype=np.float64) / self.numero_activos_riesgo
            peso_cash = np.array([0.0], dtype=np.float64)
            self.pesos_anteriores = np.concatenate([pesos_riesgo, peso_cash])

        return self._obtener_estado_actual()

    def _obtener_estado_actual(self) -> np.ndarray:
        fila_estado = self.datos_estado.iloc[self.indice_tiempo].to_numpy(dtype=np.float32)
        w_prev = self.pesos_anteriores.astype(np.float32)
        return np.concatenate([fila_estado, w_prev])

    def _rf_hoy(self) -> float:
        return float(self.rf_diario.iloc[self.indice_tiempo])

    def _preparar_pesos_objetivo(self, nuevos_pesos) -> np.ndarray:
        """
        Convierte la acción en un vector válido de pesos completos:
        [w_riesgo_1, ..., w_riesgo_N, w_cash]

        - Long-only
        - Sin apalancamiento
        - Suma total = 1
        """
        w = np.asarray(nuevos_pesos, dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)

        if w.shape[0] != self.numero_activos_totales:
            raise ValueError(
                f"La acción tiene dimensión {w.shape[0]}, pero se esperaban "
                f"{self.numero_activos_totales} pesos "
                f"({self.numero_activos_riesgo} activos riesgosos + 1 cash)."
            )

        suma = float(w.sum())
        if suma <= 0.0:
            w = np.ones(self.numero_activos_totales, dtype=np.float64)
            w = w / w.sum()
        else:
            w = w / suma

        return w

    def step(self, nuevos_pesos) -> tuple[np.ndarray | None, float, bool, dict]:
        """
        Aplica una acción (pesos completos), calcula reward, avanza el tiempo.

        La acción debe tener forma:
            [w_1, ..., w_N, w_cash]
        """
        # 1) Retornos del día actual (solo riesgosos)
        retornos_hoy = self.retornos_diarios.iloc[self.indice_tiempo].to_numpy(dtype=np.float64)
        retornos_hoy = np.nan_to_num(retornos_hoy, nan=0.0, posinf=0.0, neginf=0.0)

        rf_hoy = self._rf_hoy()

        # 2) Pesos al inicio del día
        pesos_previos = self.pesos_anteriores.copy()

        if nuevos_pesos is None:
            pesos_actuales = pesos_previos
            rotacion = 0.0
            coste = 0.0
        else:
            pesos_actuales = self._preparar_pesos_objetivo(nuevos_pesos)
            rotacion = 0.5 * float(np.sum(np.abs(pesos_actuales - pesos_previos)))
            coste = self.coste_transaccion * rotacion

        # Separar riesgo y cash
        pesos_riesgo_inicio = pesos_actuales[:-1]
        peso_cash_inicio = float(pesos_actuales[-1])

        # 3) Retorno cartera (bruto)
        retorno_riesgo = float(np.dot(pesos_riesgo_inicio, retornos_hoy))
        retorno_cash = peso_cash_inicio * rf_hoy
        retorno_cartera = retorno_riesgo + retorno_cash

        # 4) Recompensa neta
        recompensa = retorno_cartera - coste

        # 5) Actualizar valor cartera
        self.valor_cartera *= (1.0 + recompensa)

        # 6) DRIFT al final del día
        riqueza_riesgo_fin = pesos_riesgo_inicio * (1.0 + retornos_hoy)
        riqueza_cash_fin = np.array([peso_cash_inicio * (1.0 + rf_hoy)], dtype=np.float64)

        riqueza_total_fin = np.concatenate([riqueza_riesgo_fin, riqueza_cash_fin])
        total_riqueza_fin = float(riqueza_total_fin.sum())

        if total_riqueza_fin > 0.0:
            pesos_fin = riqueza_total_fin / total_riqueza_fin
        else:
            pesos_fin = pesos_actuales.copy()

        self.pesos_anteriores = pesos_fin

        # 7) Avanzar día
        self.indice_tiempo += 1
        terminado = self.indice_tiempo >= len(self.datos_estado)

        siguiente_estado = None if terminado else self._obtener_estado_actual()

        info = {
            "valor_cartera": self.valor_cartera,
            "retorno_cartera": retorno_cartera,
            "retorno_riesgo": retorno_riesgo,
            "retorno_cash": retorno_cash,
            "rf_hoy": rf_hoy,
            "coste_transaccion": coste,
            "rotacion": rotacion,
            "pesos_inicio": pesos_actuales,              # incluye cash
            "pesos_riesgo_inicio": pesos_riesgo_inicio,
            "peso_cash_inicio": peso_cash_inicio,
            "pesos_fin": pesos_fin,                      # incluye cash
            "pesos_riesgo_fin": pesos_fin[:-1],
            "peso_cash_fin": float(pesos_fin[-1]),
        }

        return siguiente_estado, recompensa, terminado, info

    def ejecutar_backtest(self, funcion_pesos) -> pd.Series:
        """
        Ejecuta un backtest llamando a:
            funcion_pesos(estado) -> pesos completos [w_1, ..., w_N, w_cash]

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

        fechas_con_inicial = [fechas[0]] + fechas
        return pd.Series(valores, index=fechas_con_inicial, name="valor_cartera")