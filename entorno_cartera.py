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
    Supuestos fijados:
    - SIEMPRE se permite cash remunerado a rf diario.
      Si sum(w_riesgo) < 1, el resto (1 - sum) se invierte en rf.
      Esto permite implementar mezclas: alpha * w_tangente + (1-alpha) * rf.
    - NUNCA se permite apalancamiento:
      Si sum(w_riesgo) > 1, se reescala proporcionalmente para que sum=1.

    Notas:
    - La acción/pesos SOLO incluye activos riesgosos (las columnas de retornos_diarios).
    - El cash/rf es implícito.
    """

    def __init__(
        self,
        datos_estado: pd.DataFrame,         # index=fechas, columns=features
        retornos_diarios: pd.DataFrame,     # index=fechas, columns=activos riesgosos (retornos diarios)
        coste_transaccion: float = 0.001,   # 10 bps por rebalanceo
        valor_inicial: float = 1000.0,
        pesos_iniciales: str = "iguales",
        rf_diario: pd.Series | None = None,  # index=fechas, retorno diario (decimal). Si None => rf=0
    ):
        # Alinear fechas por seguridad: solo usamos fechas comunes
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

        # rf diario (si None, se asume rf=0.0)
        if rf_diario is not None:
            rf_diario = rf_diario.sort_index()
            rf_diario = rf_diario.reindex(self.retornos_diarios.index).ffill().fillna(0.0)
            self.rf_diario = rf_diario.astype(float)
        else:
            self.rf_diario = pd.Series(0.0, index=self.retornos_diarios.index, dtype=float)

        # Variables internas del episodio
        self.indice_tiempo = 0
        self.valor_cartera = self.valor_inicial
        # Pesos SOLO de riesgosos; el cash es implícito (1 - sum)
        self.pesos_anteriores = None

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reinicia el entorno al inicio del dataset. Devuelve el estado inicial.
        """
        self.indice_tiempo = 0
        self.valor_cartera = self.valor_inicial

        # Pesos iniciales sobre activos riesgosos
        if self.pesos_iniciales == "iguales":
            # fully invested al inicio en riesgosos (cash=0)
            self.pesos_anteriores = np.ones(self.numero_activos) / self.numero_activos
        else:
            self.pesos_anteriores = np.ones(self.numero_activos) / self.numero_activos

        return self._obtener_estado_actual()

    def _obtener_estado_actual(self) -> np.ndarray:
        fila_estado = self.datos_estado.iloc[self.indice_tiempo].to_numpy(dtype=np.float32)

        # pesos previos (riesgosos)
        w_prev = self.pesos_anteriores.astype(np.float32)

        # cash implícito al inicio del día (antes de aplicar retornos del día)
        cash_prev = np.float32(max(0.0, 1.0 - float(self.pesos_anteriores.sum())))

        return np.concatenate([fila_estado, w_prev, np.array([cash_prev], dtype=np.float32)])

    def _rf_hoy(self) -> float:
        return float(self.rf_diario.iloc[self.indice_tiempo])

    def _preparar_pesos_objetivo(self, nuevos_pesos) -> np.ndarray:
        """
        Convierte la acción en pesos válidos sobre activos riesgosos.
        - Long-only: clip >= 0
        - Sin apalancamiento: fuerza sum(w) <= 1 (si sum>1, reescala a sum=1)
        - Cash residual permitido siempre: si sum(w)<1, el resto es cash.
        """
        w = np.asarray(nuevos_pesos, dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)

        suma = float(w.sum())
        if suma > 1.0:
            w = w / suma  # recorte proporcional (no leverage)

        return w

    def step(self, nuevos_pesos) -> tuple[np.ndarray | None, float, bool, dict]:
        """
        Aplica una acción (pesos riesgosos), calcula reward, avanza el tiempo.
        Incluye DRIFT con cash:
        - La parte en riesgosos deriva con los retornos de mercado.
        - La parte en cash deriva con (1 + rf_diario).
        """
        # 1) Retornos del día actual (riesgosos)
        retornos_hoy = self.retornos_diarios.iloc[self.indice_tiempo].to_numpy(dtype=np.float64)
        retornos_hoy = np.nan_to_num(retornos_hoy, nan=0.0, posinf=0.0, neginf=0.0)

        rf_hoy = self._rf_hoy()

        # 2) Pesos al INICIO del día
        pesos_previos = self.pesos_anteriores.copy()

        if nuevos_pesos is None:
            # No operamos
            pesos_actuales = pesos_previos
            rotacion = 0.0
            coste = 0.0
        else:
            # Rebalanceamos (pero permitimos cash residual)
            pesos_actuales = self._preparar_pesos_objetivo(nuevos_pesos)
            rotacion = float(np.sum(np.abs(pesos_actuales - pesos_previos)))
            coste = self.coste_transaccion * rotacion

        suma_riesgo_inicio = float(pesos_actuales.sum())
        peso_cash_inicio = max(0.0, 1.0 - suma_riesgo_inicio)  # siempre permitido

        # 3) Retorno cartera (bruto), incluyendo cash
        retorno_riesgo = float(np.dot(pesos_actuales, retornos_hoy))
        retorno_cash = peso_cash_inicio * rf_hoy
        retorno_cartera = retorno_riesgo + retorno_cash

        # 4) Recompensa neta de costes
        recompensa = retorno_cartera - coste

        # 5) Actualizar valor cartera
        self.valor_cartera *= (1.0 + recompensa)

        # 6) DRIFT: actualizar pesos al FINAL del día (incluye cash)
        riqueza_riesgo_fin = pesos_actuales * (1.0 + retornos_hoy)
        riqueza_cash_fin = peso_cash_inicio * (1.0 + rf_hoy)

        total_riqueza_fin = float(riqueza_riesgo_fin.sum() + riqueza_cash_fin)

        if total_riqueza_fin > 0.0:
            pesos_riesgo_fin = riqueza_riesgo_fin / total_riqueza_fin
            peso_cash_fin = riqueza_cash_fin / total_riqueza_fin
        else:
            pesos_riesgo_fin = pesos_actuales.copy()
            peso_cash_fin = peso_cash_inicio

        # Guardar pesos (solo riesgosos) para el siguiente paso
        self.pesos_anteriores = pesos_riesgo_fin

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
            "pesos_inicio": pesos_actuales,      # riesgosos (post-rebalance)
            "peso_cash_inicio": peso_cash_inicio,
            "pesos_fin": pesos_riesgo_fin,       # riesgosos (post-drift)
            "peso_cash_fin": peso_cash_fin,
        }

        return siguiente_estado, recompensa, terminado, info

    def ejecutar_backtest(self, funcion_pesos) -> pd.Series:
        """
        Ejecuta un backtest llamando a funcion_pesos(estado) -> pesos (riesgosos).
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