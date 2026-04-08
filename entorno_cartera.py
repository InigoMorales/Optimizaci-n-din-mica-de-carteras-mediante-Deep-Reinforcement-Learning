from __future__ import annotations

import numpy as np
import pandas as pd


def normalizar_pesos(pesos: np.ndarray) -> np.ndarray:
    pesos = np.asarray(pesos, dtype=np.float64)
    pesos = np.nan_to_num(pesos, nan=0.0, posinf=0.0, neginf=0.0)
    pesos = np.clip(pesos, 0.0, None)
    suma = float(pesos.sum())
    if suma <= 0.0:
        return np.ones_like(pesos, dtype=np.float64) / len(pesos)
    return pesos / suma


def interpolar_riesgo(riesgo: float, valor_conservador: float, valor_arriesgado: float) -> float:
    riesgo = float(np.clip(riesgo, 0.0, 1.0))
    return valor_conservador + riesgo * (valor_arriesgado - valor_conservador)


class EntornoCartera:
    """
    Entorno de asignación dinámica de cartera long-only con cash explícito
    y perfil de riesgo exógeno.

    Estado:
        [features_mercado, pesos_previos_completos, riesgo, vol_ema, drawdown_actual]

    Acción:
        [w_1, ..., w_N, w_cash]

    Reward:
        retorno_cartera
        - coste_transaccion
        - pen_dd          (daño real ya ocurrido, retrospectiva, escala con riesgo)
        - pen_varianza    (riesgo absoluto prospectivo w^T*Sigma*w, escala con riesgo)
        - pen_correlacion (diversificación: exceso sobre correlación objetivo por perfil)
        - pen_turnover    (penalización por rotación)

    Concentración:
        Controlada en el espacio de acciones via max_peso_activo en Entrenamiento_SAC.
        El actor no puede producir acciones con más de max_peso_activo en un solo activo.

    Matriz de covarianzas extendida con cash (cov=0 con todo y consigo mismo).
    Correlación extraída de covarianzas: Corr_ij = Sigma_ij / (sigma_i * sigma_j).
    """

    def __init__(
        self,
        datos_estado: pd.DataFrame,
        retornos_semanales: pd.DataFrame,
        coste_transaccion: float = 0.001,
        valor_inicial: float = 100000.0,
        rf_semanal: pd.Series | None = None,
        riesgo: float = 0.5,
        incluir_riesgo_en_estado: bool = True,
        incluir_metricas_riesgo_en_estado: bool = True,
        alpha_vol_ema: float = 0.06,

        # Penalización por drawdown (retrospectiva).
        # lambda_dd define el máximo (perfil conservador, riesgo=0).
        # El mínimo se deriva internamente: lambda_dd * 0.025
        # Ejemplo: lambda_dd=0.20 → max=0.20, min=0.005 (igual que antes)
        lambda_dd: float = 0.40,

        # Penalización por varianza de cartera (riesgo absoluto).
        # El mínimo se deriva internamente: lambda_varianza * 0.01
        # Ejemplo: lambda_varianza=0.10 → max=0.10, min=0.001
        lambda_varianza: float = 0.30,

        # Penalización por correlación (diversificación).
        # El mínimo se deriva internamente: lambda_correlacion * 0.05
        # Ejemplo: lambda_correlacion=0.10 → max=0.10, min=0.005
        lambda_correlacion: float = 0.20,

        # Objetivo de correlación por perfil — fijos, no en HPO
        correlacion_objetivo_min: float = 0.15,  # perfil muy conservador
        correlacion_objetivo_max: float = 0.70,  # perfil muy arriesgado

        # Penalización por turnover — fija, no en HPO
        lambda_turnover_min: float = 0.001,
        lambda_turnover_max: float = 0.005,

        # Penalización por concentración en un único activo de riesgo.
        # Fijo (no interpolado por riesgo): ningún perfil debería ir
        # al 100% en un activo, independientemente de lo arriesgado que sea.
        # Umbral: 0.50. Por encima, penalización cuadrática.
        # lambda=0.01 hace que pasar de 50%→100% en un activo cueste ~0.0025,
        # del orden de un retorno semanal típico.
        lambda_concentracion: float = 0.10,

        # Ventana rolling para covarianzas (en pasos semanales)
        ventana_covarianza: int = 52,

        # Matriz de covarianzas iniciales (precalculada sobre train sin leakage)
        covarianzas_iniciales: pd.DataFrame | np.ndarray | None = None,

        # Ventana de cash dependiente del riesgo
        min_cash: float = 0.20,
        max_cash: float = 0.65,
        cash_floor: float = 0.05,
    ):
        fechas_comunes = datos_estado.index.intersection(retornos_semanales.index)
        if len(fechas_comunes) == 0:
            raise ValueError("No hay fechas comunes entre datos_estado y retornos_semanales.")

        self.datos_estado = datos_estado.loc[fechas_comunes].copy()
        self.retornos_semanales = retornos_semanales.loc[fechas_comunes].copy()

        self.lista_activos_riesgo = list(self.retornos_semanales.columns)
        self.numero_activos_riesgo = len(self.lista_activos_riesgo)
        self.numero_activos_totales = self.numero_activos_riesgo + 1

        self.coste_transaccion = float(coste_transaccion)
        self.valor_inicial = float(valor_inicial)

        if rf_semanal is not None:
            rf_semanal = rf_semanal.sort_index()
            rf_semanal = rf_semanal.reindex(self.retornos_semanales.index).ffill().fillna(0.0)
            self.rf_semanal = rf_semanal.astype(float)
        else:
            self.rf_semanal = pd.Series(0.0, index=self.retornos_semanales.index, dtype=float)

        self.incluir_riesgo_en_estado = bool(incluir_riesgo_en_estado)
        self.incluir_metricas_riesgo_en_estado = bool(incluir_metricas_riesgo_en_estado)
        self.alpha_vol_ema = float(alpha_vol_ema)

        # lambda_dd: max = valor pasado, min = max * 0.025
        # Reproduce: lambda_dd_max=0.20, lambda_dd_min=0.005
        self.lambda_dd_max = float(lambda_dd)
        self.lambda_dd_min = float(lambda_dd) * 0.025

        # lambda_varianza: max = valor pasado, min = max * 0.01
        # Reproduce: lambda_varianza_max=0.10, lambda_varianza_min=0.001
        self.lambda_varianza_max = float(lambda_varianza)
        self.lambda_varianza_min = float(lambda_varianza) * 0.01

        # lambda_correlacion: max = valor pasado, min = max * 0.05
        # Reproduce: lambda_correlacion_max=0.10, lambda_correlacion_min=0.005
        self.lambda_correlacion_max = float(lambda_correlacion)
        self.lambda_correlacion_min = float(lambda_correlacion) * 0.05

        self.correlacion_objetivo_min = float(correlacion_objetivo_min)
        self.correlacion_objetivo_max = float(correlacion_objetivo_max)

        self.lambda_turnover_min = float(lambda_turnover_min)
        self.lambda_turnover_max = float(lambda_turnover_max)

        # lambda_concentracion: interpolado cuadráticamente igual que el resto.
        # riesgo bajo → penaliza mucho la concentración
        # riesgo alto → permite concentrarse (es su estrategia natural)
        # min = max * 0.025 (mismo ratio que lambda_dd)
        self.lambda_concentracion_max = float(lambda_concentracion)
        self.lambda_concentracion_min = float(lambda_concentracion) * 0.025

        self.ventana_covarianza = int(ventana_covarianza)

        self.min_cash = float(min_cash)
        self.max_cash = float(max_cash)
        self.cash_floor = float(cash_floor)

        if not (0.0 <= self.min_cash <= self.max_cash <= 1.0):
            raise ValueError("Se requiere 0 <= min_cash <= max_cash <= 1")
        if not (0.0 <= self.cash_floor <= 1.0):
            raise ValueError("cash_floor debe estar en [0, 1]")

        # Covarianzas iniciales
        if covarianzas_iniciales is not None:
            if isinstance(covarianzas_iniciales, pd.DataFrame):
                self._covarianza_base = covarianzas_iniciales.values.astype(np.float64)
            else:
                self._covarianza_base = np.asarray(covarianzas_iniciales, dtype=np.float64)
        else:
            self._covarianza_base = self.retornos_semanales.cov().values.astype(np.float64)

        # Varianza máxima para normalizar pen_varianza en [0, ~1]
        self._varianza_max = float(np.max(np.diag(self._covarianza_base)))
        if self._varianza_max <= 1e-12:
            self._varianza_max = 1.0

        self.covarianza_actual = self._covarianza_base.copy()
        self.correlacion_actual = self._covarianza_a_correlacion(self._covarianza_base)

        self.indice_tiempo = 0
        self.valor_cartera = self.valor_inicial
        self.valor_maximo = self.valor_inicial
        self.vol_ema = 0.0
        self.drawdown_actual = 0.0
        self.pesos_anteriores = None

        self.establecer_riesgo(riesgo)
        self.reset()

    @staticmethod
    def _covarianza_a_correlacion(cov: np.ndarray) -> np.ndarray:
        """Extrae la matriz de correlaciones a partir de la de covarianzas."""
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        outer = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(outer > 1e-12, cov / outer, 0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def establecer_riesgo(self, riesgo: float) -> None:
        riesgo = float(riesgo)
        if not (0.0 <= riesgo <= 1.0):
            raise ValueError(f"riesgo debe estar en [0,1], recibido {riesgo}")
        self.riesgo = riesgo

    def _interpolar_lambda(self, riesgo: float, lam_min: float, lam_max: float) -> float:
        # Interpolación cuadrática: perfiles conservadores penalizan mucho más,
        # perfiles arriesgados penalizan mucho menos que con interpolación lineal.
        # conservador (0.3): ~0.49x del max | arriesgado (0.7): ~0.09x del max
        return lam_max * (1.0 - riesgo) ** 2 + lam_min * riesgo ** 2

    @property
    def lambda_dd(self) -> float:
        return self._interpolar_lambda(self.riesgo, self.lambda_dd_min, self.lambda_dd_max)

    @property
    def lambda_varianza(self) -> float:
        return self._interpolar_lambda(self.riesgo, self.lambda_varianza_min, self.lambda_varianza_max)

    @property
    def lambda_correlacion(self) -> float:
        return self._interpolar_lambda(self.riesgo, self.lambda_correlacion_min, self.lambda_correlacion_max)

    @property
    def lambda_turnover(self) -> float:
        return self._interpolar_lambda(self.riesgo, self.lambda_turnover_min, self.lambda_turnover_max)

    @property
    def lambda_concentracion(self) -> float:
        return self._interpolar_lambda(self.riesgo, self.lambda_concentracion_min, self.lambda_concentracion_max)

    @property
    def correlacion_objetivo(self) -> float:
        """Correlación media objetivo para este perfil de riesgo."""
        return interpolar_riesgo(
            self.riesgo,
            self.correlacion_objetivo_min,
            self.correlacion_objetivo_max,
        )

    @property
    def cash_min_aceptable(self) -> float:
        return (1.0 - self.riesgo) * self.min_cash

    @property
    def cash_max_aceptable(self) -> float:
        return (1.0 - self.riesgo) * self.max_cash + self.riesgo * self.cash_floor

    def reset(
        self,
        riesgo: float | None = None,
        aleatorio: bool = False,
        pesos_iniciales: np.ndarray | None = None,
    ) -> np.ndarray:
        if riesgo is not None:
            self.establecer_riesgo(riesgo)

        if pesos_iniciales is not None:
            self.pesos_anteriores = np.asarray(pesos_iniciales, dtype=np.float64)
            self.indice_tiempo = 0
        elif aleatorio:
            max_inicio = max(0, len(self.datos_estado) - 52)
            self.indice_tiempo = np.random.randint(0, max_inicio)
            self.pesos_anteriores = np.random.dirichlet(
                np.ones(self.numero_activos_totales)
            ).astype(np.float64)
        else:
            self.indice_tiempo = 0
            pesos_riesgo = np.zeros(self.numero_activos_riesgo, dtype=np.float64)
            peso_cash = np.array([1.0], dtype=np.float64)
            self.pesos_anteriores = np.concatenate([pesos_riesgo, peso_cash])

        self.valor_cartera = self.valor_inicial
        self.valor_maximo = self.valor_inicial
        self.vol_ema = 0.0
        self.drawdown_actual = 0.0
        self.covarianza_actual = self._covarianza_base.copy()
        self.correlacion_actual = self._covarianza_a_correlacion(self._covarianza_base)

        return self._obtener_estado_actual()

    def _obtener_estado_actual(self) -> np.ndarray:
        fila_estado = self.datos_estado.iloc[self.indice_tiempo].to_numpy(dtype=np.float32)
        w_prev = self.pesos_anteriores.astype(np.float32)
        bloques = [fila_estado, w_prev]

        if self.incluir_riesgo_en_estado:
            bloques.append(np.array([self.riesgo], dtype=np.float32))

        if self.incluir_metricas_riesgo_en_estado:
            bloques.append(np.array([self.vol_ema, self.drawdown_actual], dtype=np.float32))

        return np.concatenate(bloques)

    def _rf_hoy(self) -> float:
        return float(self.rf_semanal.iloc[self.indice_tiempo])

    def _preparar_pesos_objetivo(self, nuevos_pesos) -> np.ndarray:
        w = np.asarray(nuevos_pesos, dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)

        if w.shape[0] != self.numero_activos_totales:
            raise ValueError(
                f"La acción tiene dimensión {w.shape[0]}, pero se esperaban "
                f"{self.numero_activos_totales} pesos."
            )

        suma = float(w.sum())
        if suma <= 0.0:
            w = np.ones(self.numero_activos_totales, dtype=np.float64)
            w = w / w.sum()
        else:
            w = w / suma

        return w

    def _actualizar_matrices_rolling(self) -> None:
        """
        Actualiza covarianzas y correlaciones con ventana rolling.
        Si no hay suficientes datos, mantiene las matrices base.
        """
        if self.indice_tiempo >= self.ventana_covarianza:
            inicio = self.indice_tiempo - self.ventana_covarianza
            retornos_ventana = self.retornos_semanales.iloc[inicio:self.indice_tiempo]
            cov = retornos_ventana.cov().values
            cov = np.nan_to_num(cov, nan=0.0)
            np.fill_diagonal(cov, np.maximum(np.diag(cov), 0.0))
            self.covarianza_actual = cov

            var_max = float(np.max(np.diag(cov)))
            if var_max > 1e-12:
                self._varianza_max = var_max

            self.correlacion_actual = self._covarianza_a_correlacion(cov)

    def _penalizacion_varianza(self, pesos_actuales: np.ndarray) -> float:
        """
        Varianza de cartera w^T * Sigma_ext * w normalizada por varianza_max.
        Cash tiene covarianza 0 con todo → actúa como diversificador natural.
        ~100% cash: penalización implícita = riesgo (para diferenciar perfiles).
        """
        pesos_riesgo = pesos_actuales[:-1]
        suma_riesgo = float(pesos_riesgo.sum())

        if suma_riesgo < 0.01:
            return float(self.riesgo)

        n = self.numero_activos_riesgo
        cov_ext = np.zeros((n + 1, n + 1))
        cov_ext[:n, :n] = self.covarianza_actual
        cov_ext[n, n] = 0.0

        varianza_cartera = float(pesos_actuales @ cov_ext @ pesos_actuales)
        return max(0.0, varianza_cartera) / self._varianza_max

    def _penalizacion_correlacion(self, pesos_actuales: np.ndarray) -> float:
        """
        Penaliza el exceso de correlación media de cartera sobre el objetivo del perfil.

        Calcula w_riesgo_norm^T * Corr * w_riesgo_norm sobre la parte invertida.
        Solo penaliza si la correlación supera correlacion_objetivo del perfil:
            exceso = max(0, correlacion_cartera - correlacion_objetivo)

        ~100% cash: correlación = 0 (sin penalización, el cash no está correlacionado).
        Poca inversión: se escala por factor_exposicion para no penalizar carteras
        casi en cash que nominalmente tienen alta correlación entre sus pocos activos.
        """
        pesos_riesgo = pesos_actuales[:-1]
        suma_riesgo = float(pesos_riesgo.sum())

        if suma_riesgo < 0.01:
            return 0.0

        pesos_norm = pesos_riesgo / suma_riesgo
        correlacion_cartera = float(pesos_norm @ self.correlacion_actual @ pesos_norm)
        correlacion_cartera = float(np.clip(correlacion_cartera, 0.0, 1.0))

        exceso = max(0.0, correlacion_cartera - self.correlacion_objetivo)

        # Escalar por exposición: si hay poca inversión, penalizar menos
        factor_exposicion = min(1.0, suma_riesgo / 0.30)
        return exceso * factor_exposicion

    def _penalizacion_concentracion(self, pesos_actuales: np.ndarray) -> float:
        """
        Penaliza la concentración excesiva en un único activo de riesgo.

        Umbral fijo en 0.50: por debajo no hay penalización, por encima
        crece cuadráticamente. Cash excluido: concentrarse en cash ya está
        desincentivado implícitamente por pen_varianza (devuelve self.riesgo
        cuando todo es cash).

        Ejemplos con lambda_concentracion=0.01:
            peso_max=0.50 → pen=0.000  (justo en el límite)
            peso_max=0.70 → pen=0.004  (exceso=0.20, 0.20²*0.01)
            peso_max=1.00 → pen=0.025  (exceso=0.50, 0.50²*0.01)
        """
        pesos_riesgo = pesos_actuales[:-1]
        if len(pesos_riesgo) == 0:
            return 0.5 ** 3
        peso_max = float(np.max(pesos_riesgo))
        exceso = max(0.0, peso_max - 0.50)
        return exceso ** 2

    def step(self, nuevos_pesos) -> tuple[np.ndarray | None, float, bool, dict]:
        retornos_hoy = self.retornos_semanales.iloc[self.indice_tiempo].to_numpy(dtype=np.float64)
        retornos_hoy = np.nan_to_num(retornos_hoy, nan=0.0, posinf=0.0, neginf=0.0)
        rf_hoy = self._rf_hoy()

        pesos_previos = self.pesos_anteriores.copy()

        if nuevos_pesos is None:
            pesos_actuales = pesos_previos
            rotacion = 0.0
            coste = 0.0
        else:
            pesos_actuales = self._preparar_pesos_objetivo(nuevos_pesos)
            rotacion = 0.5 * float(np.sum(np.abs(pesos_actuales - pesos_previos)))
            coste = self.coste_transaccion * rotacion

        pesos_riesgo_inicio = pesos_actuales[:-1]
        peso_cash_inicio = float(pesos_actuales[-1])

        retorno_riesgo = float(np.dot(pesos_riesgo_inicio, retornos_hoy))
        retorno_cash = peso_cash_inicio * rf_hoy
        retorno_cartera = retorno_riesgo + retorno_cash

        retorno_cartera_cuadrado = retorno_cartera ** 2
        self.vol_ema = (
            (1.0 - self.alpha_vol_ema) * self.vol_ema
            + self.alpha_vol_ema * retorno_cartera_cuadrado
        )
        vol_ema_raiz = float(np.sqrt(max(self.vol_ema, 0.0)))

        valor_antes = self.valor_cartera
        valor_despues_bruto = valor_antes * (1.0 + retorno_cartera - coste)
        self.valor_maximo = max(self.valor_maximo, valor_despues_bruto)
        self.drawdown_actual = 1.0 - (valor_despues_bruto / (self.valor_maximo + 1e-12))
        self.drawdown_actual = float(np.clip(self.drawdown_actual, 0.0, 1.0))

        exposicion = 1.0 - peso_cash_inicio

        self._actualizar_matrices_rolling()

        pen_dd = self.lambda_dd * self.drawdown_actual
        pen_varianza = self.lambda_varianza * self._penalizacion_varianza(pesos_actuales)
        pen_correlacion = self.lambda_correlacion * self._penalizacion_correlacion(pesos_actuales)
        pen_turnover = self.lambda_turnover * rotacion
        pen_concentracion = self.lambda_concentracion * self._penalizacion_concentracion(pesos_actuales)

        pen_total = pen_dd + pen_varianza + pen_correlacion + pen_turnover + pen_concentracion

        recompensa = retorno_cartera - coste - pen_total

        self.valor_cartera *= (1.0 + retorno_cartera - coste)

        riqueza_riesgo_fin = pesos_riesgo_inicio * (1.0 + retornos_hoy)
        riqueza_cash_fin = np.array([peso_cash_inicio * (1.0 + rf_hoy)], dtype=np.float64)
        riqueza_total_fin = np.concatenate([riqueza_riesgo_fin, riqueza_cash_fin])
        total_riqueza_fin = float(riqueza_total_fin.sum())

        if total_riqueza_fin > 0.0:
            pesos_fin = riqueza_total_fin / total_riqueza_fin
        else:
            pesos_fin = pesos_actuales.copy()

        self.pesos_anteriores = pesos_fin

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
            "turnover": rotacion,
            "riesgo": self.riesgo,
            "lambda_dd": self.lambda_dd,
            "lambda_varianza": self.lambda_varianza,
            "lambda_correlacion": self.lambda_correlacion,
            "lambda_turnover": self.lambda_turnover,
            "correlacion_objetivo": self.correlacion_objetivo,
            "vol_ema": vol_ema_raiz,
            "drawdown_actual": self.drawdown_actual,
            "cash_min_aceptable": self.cash_min_aceptable,
            "cash_max_aceptable": self.cash_max_aceptable,
            "pen_dd": pen_dd,
            "pen_varianza": pen_varianza,
            "pen_correlacion": pen_correlacion,
            "pen_turnover": pen_turnover,
            "pen_concentracion": pen_concentracion,
            "pen_total": pen_total,
            "pesos_inicio": pesos_actuales,
            "pesos_riesgo_inicio": pesos_riesgo_inicio,
            "peso_cash_inicio": peso_cash_inicio,
            "pesos_fin": pesos_fin,
            "pesos_riesgo_fin": pesos_fin[:-1],
            "peso_cash_fin": float(pesos_fin[-1]),
            "exposicion": exposicion,
            "peso_max_activo": float(np.max(pesos_riesgo_inicio)) if len(pesos_riesgo_inicio) > 0 else 0.0,
        }

        return siguiente_estado, recompensa, terminado, info

    def ejecutar_backtest(
        self,
        funcion_pesos,
        riesgo: float | None = None,
        pesos_iniciales: np.ndarray | None = None,
    ) -> pd.DataFrame:
        self.reset(riesgo=riesgo, pesos_iniciales=pesos_iniciales)

        fechas = list(self.datos_estado.index)
        registros = []

        terminado = False
        estado = self._obtener_estado_actual()

        while not terminado:
            pesos = funcion_pesos(estado)
            estado, recompensa, terminado, info = self.step(pesos)

            registros.append({
                "fecha": fechas[self.indice_tiempo - 1],
                "valor_cartera": info["valor_cartera"],
                "recompensa": recompensa,
                "retorno_cartera": info["retorno_cartera"],
                "riesgo": info["riesgo"],
                "vol_ema": info["vol_ema"],
                "drawdown_actual": info["drawdown_actual"],
                "peso_cash": info["peso_cash_inicio"],
                "exposicion": info["exposicion"],
                "turnover": info["turnover"],
                "pen_dd": info["pen_dd"],
                "pen_varianza": info["pen_varianza"],
                "pen_correlacion": info["pen_correlacion"],
                "pen_turnover": info["pen_turnover"],
                "pen_total": info["pen_total"],
            })

        return pd.DataFrame(registros).set_index("fecha")