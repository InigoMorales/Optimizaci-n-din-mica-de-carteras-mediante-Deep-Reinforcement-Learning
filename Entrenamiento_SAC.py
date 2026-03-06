from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import os
import random

from dataclasses import dataclass
from typing import Dict, Optional, List
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

def accion_aleatoria_valida(numero_activos: int) -> np.ndarray:
    """
    Genera una acción long-only con suma <= 1 (cash implícito).
    Para warmup inicial del buffer.
    """
    # Dirichlet te da suma=1; luego multiplicamos por un factor <1 para permitir cash
    w = np.random.dirichlet(alpha=np.ones(numero_activos)).astype(np.float32)
    factor_inversion = np.random.uniform(0.0, 1.0)  # permite desde 0% hasta 100% invertido
    w = w * factor_inversion
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

    gamma: float = 0.99
    tau: float = 0.005

    lr_actor: float = 3e-4
    lr_criticos: float = 3e-4
    lr_alpha: float = 3e-4

    target_entropy: Optional[float] = None  # si None, se usa -numero_activos

    frecuencia_actualizacion: int = 1       # cada cuántos steps entrenamos
    actualizaciones_por_step: int = 1       # cuántos updates por step (1 es lo típico)

    ventana_log_recompensa: int = 200
    frecuencia_log: int = 500

# ============================================================
# Script principal de entrenamiento
# ============================================================

def entrenar_sac(
    entorno: EntornoCartera,
    config: ConfigEntrenamiento,
) -> Dict[str, List[float]]:
    fijar_semillas(config.semilla)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dimension del estado: lo que devuelve entorno.reset()/step()
    estado_inicial = entorno.reset()
    dimension_estado = int(estado_inicial.shape[0])
    numero_activos = int(entorno.numero_activos)

    # Buffer
    buffer = BufferRepeticion(
        capacidad_maxima=config.tamano_buffer,
        dimension_estado=dimension_estado,
        numero_activos=numero_activos,
        dispositivo=dispositivo,
        semilla=config.semilla,
    )

    # Agente SAC
    agente = AgenteSAC(
        dimension_estado=dimension_estado,
        numero_activos=numero_activos,
        dispositivo=dispositivo,
        gamma=config.gamma,
        tau=config.tau,
        tasa_aprendizaje_actor=config.lr_actor,
        tasa_aprendizaje_criticos=config.lr_criticos,
        tasa_aprendizaje_alpha=config.lr_alpha,
        target_entropy=config.target_entropy,
    )

    # Históricos para logging
    historico_recompensas: List[float] = []
    historico_perdida_critic1: List[float] = []
    historico_perdida_critic2: List[float] = []
    historico_perdida_actor: List[float] = []
    historico_alpha: List[float] = []

    # ============================================================
    # Loop
    # ============================================================
    estado = entorno.reset()
    terminado = False

    for paso_global in range(1, config.pasos_totales + 1):
        # --------------------------------------------------------
        # 1) Elegir acción (warmup aleatorio o política actual)
        # --------------------------------------------------------
        if paso_global <= config.pasos_warmup:
            accion = accion_aleatoria_valida(numero_activos)
        else:
            estado_tensor = torch.as_tensor(estado, dtype=torch.float32, device=dispositivo).unsqueeze(0)
            with torch.no_grad():
                accion_tensor = agente.seleccionar_accion(estado_tensor, determinista=False)
            accion = accion_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        # --------------------------------------------------------
        # 2) Step del entorno
        # --------------------------------------------------------
        siguiente_estado, recompensa, terminado, info = entorno.step(accion)

        # Si terminó, para el buffer necesitamos un "siguiente_estado" válido en forma tensor.
        # En SAC se suele guardar el siguiente_estado cualquiera, pero como tu entorno devuelve None,
        # guardamos un vector cero (y usamos 'terminado' para anular el bootstrap con (1-d)).
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
        # 4) Entrenar (si hay suficiente buffer y si toca por frecuencia)
        # --------------------------------------------------------
        if buffer.esta_listo(config.tamano_batch) and (paso_global % config.frecuencia_actualizacion == 0):
            for _ in range(config.actualizaciones_por_step):
                lote = buffer.muestrear_lote(config.tamano_batch)
                metricas = agente.paso_entrenamiento(lote)

                historico_perdida_critic1.append(metricas.perdida_critic1)
                historico_perdida_critic2.append(metricas.perdida_critic2)
                historico_perdida_actor.append(metricas.perdida_actor)
                historico_alpha.append(metricas.coeficiente_entropia_alpha)

        # --------------------------------------------------------
        # 5) Avanzar estado / reset si terminó
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
            recompensa_media = media_movil(historico_recompensas, config.ventana_log_recompensa)

            # En warmup aún no hay pérdidas registradas
            ultima_pcritic1 = historico_perdida_critic1[-1] if historico_perdida_critic1 else float("nan")
            ultima_pcritic2 = historico_perdida_critic2[-1] if historico_perdida_critic2 else float("nan")
            ultima_pactor = historico_perdida_actor[-1] if historico_perdida_actor else float("nan")
            ultimo_alpha = historico_alpha[-1] if historico_alpha else float(agente.alpha.detach().cpu().item())

            print(
                f"[Paso {paso_global:>7}] "
                f"recompensa_media({config.ventana_log_recompensa})={recompensa_media:+.6f} | "
                f"perdida_critic1={ultima_pcritic1:.6f} | "
                f"perdida_critic2={ultima_pcritic2:.6f} | "
                f"perdida_actor={ultima_pactor:.6f} | "
                f"alpha={ultimo_alpha:.6f}"
            )

    return {
        "recompensas": historico_recompensas,
        "perdida_critic1": historico_perdida_critic1,
        "perdida_critic2": historico_perdida_critic2,
        "perdida_actor": historico_perdida_actor,
        "alpha": historico_alpha,
    }


# ============================================================
# Ejecución (ejemplo)
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # AQUÍ debes cargar tus datos:
    #  - datos_estado: DataFrame index=fechas, columns=features
    #  - retornos_diarios: DataFrame index=fechas, columns=activos (retornos diarios)
    #  - rf_diario: Series index=fechas, retorno diario (decimal), o None
    # --------------------------------------------------------

    # EJEMPLO (rellena con tu pipeline real):
    # datos_estado = pd.read_csv("...", index_col=0, parse_dates=True)
    # retornos_diarios = pd.read_csv("...", index_col=0, parse_dates=True)
    # rf_diario = pd.read_csv("...", index_col=0, parse_dates=True).iloc[:, 0]

    raise SystemExit(
        "Rellena la carga de datos (datos_estado, retornos_diarios, rf_diario) "
        "según tu proyecto, y elimina este SystemExit."
    )

    # entorno = EntornoCartera(
    #     datos_estado=datos_estado,
    #     retornos_diarios=retornos_diarios,
    #     coste_transaccion=0.001,
    #     valor_inicial=1000.0,
    #     pesos_iniciales="iguales",
    #     rf_diario=rf_diario,
    # )
    #
    # config = ConfigEntrenamiento(
    #     pasos_totales=200_000,
    #     pasos_warmup=10_000,
    #     tamano_buffer=200_000,
    #     tamano_batch=256,
    # )
    #
    # historicos = entrenar_sac(entorno, config)