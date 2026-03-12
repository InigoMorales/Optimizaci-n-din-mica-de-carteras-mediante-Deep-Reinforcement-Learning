from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import random
import numpy as np
import torch

from entorno_cartera import EntornoCartera
from Replay_buffer import BufferRepeticion
from Agente_SAC import AgenteSAC


# ============================================================
# Utilidades
# ============================================================

def fijar_semillas(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def media_movil(valores: List[float], ventana: int) -> float:
    if len(valores) == 0:
        return 0.0
    if len(valores) < ventana:
        return float(np.mean(valores))
    return float(np.mean(valores[-ventana:]))


def accion_aleatoria_valida(dimension_accion: int) -> np.ndarray:
    """
    Genera una acción válida sobre el simplex:
        [w_1, ..., w_N, w_cash]
    con w_i >= 0 y suma(w) = 1
    """
    w = np.random.dirichlet(alpha=np.ones(dimension_accion)).astype(np.float32)
    return w


# ============================================================
# Configuración de entrenamiento
# ============================================================

@dataclass
class ConfigEntrenamiento:
    semilla: int = 42

    pasos_totales: int = 200_000
    tamano_buffer: int = 200_000
    tamano_batch: int = 256
    pasos_warmup: int = 10_000

    gamma: float = 0.9
    tau: float = 0.0005

    lr_actor: float = 1e-4
    lr_criticos: float = 5e-4
    lr_alpha: float = 1e-3

    target_entropy: Optional[float] = None
    max_concentracion_total_extra: float = 10.0

    frecuencia_actualizacion: int = 1
    actualizaciones_por_step: int = 1

    ventana_log_recompensa: int = 200
    frecuencia_log: int = 1000

    reward_scale: float = 3000.0
    offset_target_entropy: float = -1.0


# ============================================================
#               Script principal de entrenamiento
# ============================================================

def entrenar_sac(entorno: EntornoCartera, config: ConfigEntrenamiento, devolver_agente: bool = False) -> Dict[str, List[float]]:
    
    fijar_semillas(config.semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dimensión del estado: lo que devuelve el entorno
    estado_inicial = entorno.reset()
    dimension_estado = int(estado_inicial.shape[0])

    # Dimensión de acción: activos de riesgo + cash explícito
    dimension_accion = int(entorno.numero_activos_totales)

    # ============================================================
    #           Buffer
    # ============================================================
    buffer = BufferRepeticion(capacidad_maxima=config.tamano_buffer, dimension_estado=dimension_estado, 
                              dimension_accion=dimension_accion, dispositivo=dispositivo, semilla=config.semilla)

    # ============================================================
    #           Agente SAC
    # ============================================================
    agente = AgenteSAC(
        dimension_estado=dimension_estado,
        dimension_accion=dimension_accion,
        dispositivo=dispositivo,
        gamma=config.gamma,
        tau=config.tau,
        tasa_aprendizaje_actor=config.lr_actor,
        tasa_aprendizaje_criticos=config.lr_criticos,
        tasa_aprendizaje_alpha=config.lr_alpha,
        target_entropy=config.target_entropy,
        reward_scale=config.reward_scale,
        offset_target_entropy=config.offset_target_entropy,
        max_concentracion_total_extra=config.max_concentracion_total_extra,
    )

    # ============================================================
    # Históricos para logging
    # ============================================================
    historico_recompensas: List[float] = []
    historico_perdida_critic1: List[float] = []
    historico_perdida_critic2: List[float] = []
    historico_perdida_actor: List[float] = []
    historico_alpha: List[float] = []
    historico_residual_entropia: List[float] = []

    historico_q_min: List[float] = []
    historico_q1: List[float] = []
    historico_q2: List[float] = []
    historico_target_q: List[float] = []
    historico_gap_critics: List[float] = []

    historico_log_prob: List[float] = []
    historico_log_prob_std: List[float] = []

    historico_entropia: List[float] = []

    historico_concentracion_min: List[float] = []
    historico_concentracion_max: List[float] = []
    historico_concentracion_media: List[float] = []
    historico_concentracion_total: List[float] = []
    historico_concentracion_total_std: List[float] = []

    historico_accion_min: List[float] = []
    historico_accion_max: List[float] = []
    historico_peso_cash: List[float] = []

    # ============================================================
    # Loop principal
    # ============================================================
    estado = entorno.reset()
    terminado = False

    for paso_global in range(1, config.pasos_totales + 1):
        # --------------------------------------------------------
        # 1) Elegir acción
        # --------------------------------------------------------
        if paso_global <= config.pasos_warmup:
            accion = accion_aleatoria_valida(dimension_accion)
        else:
            estado_tensor = torch.as_tensor(
                estado,
                dtype=torch.float32,
                device=dispositivo,
            ).unsqueeze(0)

            with torch.no_grad():
                accion_tensor = agente.seleccionar_accion(
                    estado_tensor,
                    determinista=False,
                )

            accion = accion_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        # --------------------------------------------------------
        # 2) Step del entorno
        # --------------------------------------------------------
        siguiente_estado, recompensa, terminado, info = entorno.step(accion)

        if siguiente_estado is None:
            siguiente_estado_guardar = np.zeros_like(estado, dtype=np.float32)
        else:
            siguiente_estado_guardar = np.asarray(siguiente_estado, dtype=np.float32)

        # --------------------------------------------------------
        # 3) Guardar transición
        # --------------------------------------------------------
        buffer.guardar_transicion(
            estado=np.asarray(estado, dtype=np.float32),
            accion=np.asarray(accion, dtype=np.float32),
            recompensa=float(recompensa),
            siguiente_estado=siguiente_estado_guardar,
            terminado=bool(terminado),
        )

        historico_recompensas.append(float(recompensa))

        # --------------------------------------------------------
        # 4) Entrenar
        # --------------------------------------------------------
        if buffer.esta_listo(config.tamano_batch) and (
            paso_global % config.frecuencia_actualizacion == 0
        ):
            for _ in range(config.actualizaciones_por_step):
                lote = buffer.muestrear_lote(config.tamano_batch)
                metricas = agente.paso_entrenamiento(lote)

                historico_perdida_critic1.append(metricas.perdida_critic1)
                historico_perdida_critic2.append(metricas.perdida_critic2)
                historico_perdida_actor.append(metricas.perdida_actor)
                historico_alpha.append(metricas.coeficiente_entropia_alpha)
                historico_residual_entropia.append(metricas.residual_entropia_medio)

                historico_q_min.append(metricas.valor_q_min_medio)
                historico_q1.append(metricas.q1_medio)
                historico_q2.append(metricas.q2_medio)
                historico_target_q.append(metricas.target_q_medio)
                historico_gap_critics.append(metricas.gap_critics_medio)

                historico_log_prob.append(metricas.log_prob_medio)
                historico_log_prob_std.append(metricas.log_prob_std)
                historico_entropia.append(metricas.entropia_media)

                historico_concentracion_min.append(metricas.concentracion_min)
                historico_concentracion_max.append(metricas.concentracion_max)
                historico_concentracion_media.append(metricas.concentracion_media)
                historico_concentracion_total.append(metricas.concentracion_total_media)
                historico_concentracion_total_std.append(metricas.concentracion_total_std)

                historico_accion_min.append(metricas.accion_min)
                historico_accion_max.append(metricas.accion_max)
                historico_peso_cash.append(metricas.peso_cash_medio)

        # --------------------------------------------------------
        # 5) Avanzar / resetear episodio
        # --------------------------------------------------------
        if terminado:
            estado = entorno.reset()
            terminado = False
        else:
            estado = siguiente_estado

        # --------------------------------------------------------
        # 6) Logging
        # --------------------------------------------------------
        if paso_global % config.frecuencia_log == 0:
            recompensa_media = media_movil(
                historico_recompensas,
                config.ventana_log_recompensa,
            )

            ultima_pcritic1 = (
                historico_perdida_critic1[-1]
                if historico_perdida_critic1
                else float("nan")
            )
            ultima_pcritic2 = (
                historico_perdida_critic2[-1]
                if historico_perdida_critic2
                else float("nan")
            )
            ultima_pactor = (
                historico_perdida_actor[-1]
                if historico_perdida_actor
                else float("nan")
            )
            ultimo_alpha = (
                historico_alpha[-1]
                if historico_alpha
                else float(agente.alpha.detach().cpu().item())
            )
            ultimo_residual_entropia = (
                historico_residual_entropia[-1]
                if historico_residual_entropia
                else float("nan")
            )

            ultimo_q_min = (
                historico_q_min[-1]
                if historico_q_min
                else float("nan")
            )
            ultimo_q1 = (
                historico_q1[-1]
                if historico_q1
                else float("nan")
            )
            ultimo_q2 = (
                historico_q2[-1]
                if historico_q2
                else float("nan")
            )
            ultimo_target_q = (
                historico_target_q[-1]
                if historico_target_q
                else float("nan")
            )
            ultimo_gap_critics = (
                historico_gap_critics[-1]
                if historico_gap_critics
                else float("nan")
            )

            ultimo_log_prob = (
                historico_log_prob[-1]
                if historico_log_prob
                else float("nan")
            )
            ultimo_log_prob_std = (
                historico_log_prob_std[-1]
                if historico_log_prob_std
                else float("nan")
            )
            ultima_entropia = (
                historico_entropia[-1]
                if historico_entropia
                else float("nan")
            )

            ultima_concentracion_min = (
                historico_concentracion_min[-1]
                if historico_concentracion_min
                else float("nan")
            )
            ultima_concentracion_max = (
                historico_concentracion_max[-1]
                if historico_concentracion_max
                else float("nan")
            )
            ultima_concentracion_media = (
                historico_concentracion_media[-1]
                if historico_concentracion_media
                else float("nan")
            )
            ultima_concentracion_total = (
                historico_concentracion_total[-1]
                if historico_concentracion_total
                else float("nan")
            )
            ultima_concentracion_total_std = (
                historico_concentracion_total_std[-1]
                if historico_concentracion_total_std
                else float("nan")
            )

            ultima_accion_min = (
                historico_accion_min[-1]
                if historico_accion_min
                else float("nan")
            )
            ultima_accion_max = (
                historico_accion_max[-1]
                if historico_accion_max
                else float("nan")
            )
            ultimo_peso_cash = (
                historico_peso_cash[-1]
                if historico_peso_cash
                else float("nan")
            )

            print(
                f"[Paso {paso_global:>7}] "
                f"recompensa_media({config.ventana_log_recompensa})={recompensa_media:+.6f} | "
                f"perdida_critic1={ultima_pcritic1:.6f} | "
                f"perdida_critic2={ultima_pcritic2:.6f} | "
                f"perdida_actor={ultima_pactor:.6f} | "
                f"alpha={ultimo_alpha:.6f} | "
                f"Q_min={ultimo_q_min:+.6f} | "
                f"Q1={ultimo_q1:+.6f} | "
                f"Q2={ultimo_q2:+.6f} | "
                f"target_Q={ultimo_target_q:+.6f} | "
                f"gap_critics={ultimo_gap_critics:.6f} | "
                f"log_prob={ultimo_log_prob:+.6f} | "
                f"log_prob_std={ultimo_log_prob_std:.6f} | "
                f"entropia={ultima_entropia:+.6f} | "
                f"conc_min={ultima_concentracion_min:.6f} | "
                f"conc_max={ultima_concentracion_max:.6f} | "
                f"conc_media={ultima_concentracion_media:.6f} | "
                f"conc_total={ultima_concentracion_total:.6f} | "
                f"conc_total_std={ultima_concentracion_total_std:.6f} | "
                f"accion_min={ultima_accion_min:.6e} | "
                f"accion_max={ultima_accion_max:.6f} | "
                f"peso_cash={ultimo_peso_cash:.6f} | "
                f"residual_entropia={ultimo_residual_entropia:+.6f}"
            )

    history = {
        "recompensas": historico_recompensas,
        "perdida_critic1": historico_perdida_critic1,
        "perdida_critic2": historico_perdida_critic2,
        "perdida_actor": historico_perdida_actor,
        "alpha": historico_alpha,
        "residual_entropia": historico_residual_entropia,
        "q_min": historico_q_min,
        "q1": historico_q1,
        "q2": historico_q2,
        "target_q": historico_target_q,
        "gap_critics": historico_gap_critics,
        "log_prob": historico_log_prob,
        "log_prob_std": historico_log_prob_std,
        "entropia": historico_entropia,
        "concentracion_min": historico_concentracion_min,
        "concentracion_max": historico_concentracion_max,
        "concentracion_media": historico_concentracion_media,
        "concentracion_total": historico_concentracion_total,
        "concentracion_total_std": historico_concentracion_total_std,
        "accion_min": historico_accion_min,
        "accion_max": historico_accion_max,
        "peso_cash": historico_peso_cash,
    }

    if devolver_agente:
        return history, agente

    return history

# ============================================================
# Ejecución manual
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        "Carga tus datos reales y crea el entorno desde tu notebook o script principal. "
        "Este módulo expone entrenar_sac() y ConfigEntrenamiento."
    )