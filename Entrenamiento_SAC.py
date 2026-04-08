from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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
    tipo = np.random.randint(0, 3)
    if tipo == 0:
        w = np.zeros(dimension_accion, dtype=np.float32)
        w[np.random.randint(0, dimension_accion)] = 1.0
    elif tipo == 1:
        w = np.zeros(dimension_accion, dtype=np.float32)
        w[-1] = 1.0
    else:
        w = np.random.dirichlet(
            alpha=np.ones(dimension_accion) * 1.0
        ).astype(np.float32)
    return w


def _ultimo_o_nan(historico: List[float]) -> float:
    return historico[-1] if historico else float("nan")


def _obtener_nombres_activos(entorno: EntornoCartera, dimension_accion: int) -> List[str]:
    candidatos = [
        getattr(entorno, "nombres_activos", None),
        getattr(entorno, "columnas_activos", None),
        list(getattr(entorno, "retornos_semanales", []).columns) if hasattr(getattr(entorno, "retornos_semanales", None), "columns") else None,
        list(getattr(entorno, "retornos", []).columns) if hasattr(getattr(entorno, "retornos", None), "columns") else None,
    ]
    for nombres in candidatos:
        if nombres is None:
            continue
        nombres = list(nombres)
        if len(nombres) == dimension_accion - 1:
            return nombres
    return [f"activo_{i}" for i in range(dimension_accion - 1)]


def resumir_cartera(
    accion: np.ndarray,
    nombres_activos: Optional[List[str]] = None,
    top_k: int = 5,
    umbral_activo: float = 0.01,
) -> Tuple[str, Dict[str, Any]]:
    accion = np.asarray(accion, dtype=float).reshape(-1)
    peso_cash = float(accion[-1])
    pesos_riesgo = accion[:-1]

    if nombres_activos is None or len(nombres_activos) != len(pesos_riesgo):
        nombres_activos = [f"activo_{i}" for i in range(len(pesos_riesgo))]

    idx_orden = np.argsort(pesos_riesgo)[::-1]
    top_idx = idx_orden[:top_k]

    top_activos = []
    partes = []
    for i in top_idx:
        peso = float(pesos_riesgo[i])
        if peso <= 1e-12:
            continue
        nombre = str(nombres_activos[i])
        top_activos.append({"activo": nombre, "peso": peso})
        partes.append(f"{nombre}={peso:.3f}")

    n_activos_relevantes = int(np.sum(pesos_riesgo > umbral_activo))
    peso_max = float(np.max(pesos_riesgo)) if len(pesos_riesgo) > 0 else 0.0
    exposicion = float(1.0 - peso_cash)
    resumen_top = " | ".join(partes) if partes else "sin activos"

    resumen = (
        f"cash={peso_cash:.3f} | "
        f"exposicion={exposicion:.3f} | "
        f"peso_max={peso_max:.3f} | "
        f"n_activos>{umbral_activo:.0%}={n_activos_relevantes} | "
        f"top={resumen_top}"
    )

    payload = {
        "peso_cash": peso_cash,
        "exposicion": exposicion,
        "peso_max": peso_max,
        "n_activos_relevantes": n_activos_relevantes,
        "top_activos": top_activos,
    }
    return resumen, payload


# ============================================================
# Configuración de entrenamiento
# ============================================================

@dataclass
class ConfigEntrenamiento:
    semilla: int = 42

    pasos_totales: int = 200_000
    tamano_buffer: int = 200_000
    tamano_batch: int = 256
    pasos_warmup: int = 2_000

    gamma: float = 0.90
    tau: float = 0.005

    lr_actor: float = 3e-4
    lr_criticos: float = 3e-4
    lr_alpha: float = 3e-4

    target_entropy: Optional[float] = None
    max_concentracion_total_extra: float = 7.0

    frecuencia_actualizacion: int = 1
    actualizaciones_por_step: int = 1

    ventana_log_recompensa: int = 200
    frecuencia_log: int = 1000

    reward_scale: float = 20.0
    offset_target_entropy: float = 0.0

    # Snapshots de cartera
    frecuencia_snapshot_cartera: int = 1000
    top_k_cartera: int = 5


# ============================================================
# Script principal de entrenamiento
# ============================================================

def entrenar_sac(
        entorno: EntornoCartera,
        config: ConfigEntrenamiento,
        riesgo: float = 1.0,
        devolver_agente: bool = False,
) -> Union[Dict[str, List[float]], Tuple[Dict[str, List[float]], AgenteSAC]]:

    fijar_semillas(config.semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    estado_inicial = entorno.reset(riesgo=riesgo)
    dimension_estado = int(estado_inicial.shape[0])
    dimension_accion = int(entorno.numero_activos_totales)
    nombres_activos = _obtener_nombres_activos(entorno, dimension_accion)

    buffer = BufferRepeticion(
        capacidad_maxima=config.tamano_buffer,
        dimension_estado=dimension_estado,
        dimension_accion=dimension_accion,
        dispositivo=dispositivo,
        semilla=config.semilla,
    )

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

    with torch.no_grad():
        estado_test = torch.as_tensor(
            estado_inicial, dtype=torch.float32, device=dispositivo,
        ).unsqueeze(0)

        mu_test, log_std_test = agente.actor.forward(estado_test)
        std_test = torch.exp(log_std_test)
        accion_test, log_prob_test, _, _ = agente.actor.sample_action(estado_test)

        residual_ini = log_prob_test.item() - agente.target_entropy

        print("[DIAGNÓSTICO INICIAL]")
        print(
            f"  mu: min={mu_test.min().item():.4f}  "
            f"max={mu_test.max().item():.4f}  "
            f"media={mu_test.mean().item():.4f}"
        )
        print(
            f"  std: min={std_test.min().item():.4f}  "
            f"max={std_test.max().item():.4f}  "
            f"media={std_test.mean().item():.4f}"
        )
        print(
            f"  log_prob: {log_prob_test.item():.4f}  "
            f"(target_entropy={agente.target_entropy:.4f})"
        )
        print(
            f"  residual inicial: {residual_ini:.4f}  "
            f"(>0 → alpha sube, <0 → alpha baja)"
        )
        print(f"  accion suma: {accion_test.sum().item():.6f}")
        print(f"  peso_cash inicial: {accion_test[0, -1].item():.4f}")

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
    historico_snapshots_cartera: List[Dict[str, Any]] = []

    estado = entorno.reset(riesgo=riesgo)
    terminado = False

    for paso_global in range(1, config.pasos_totales + 1):
        if paso_global <= config.pasos_warmup:
            accion = accion_aleatoria_valida(dimension_accion)
        else:
            estado_tensor = torch.as_tensor(
                estado, dtype=torch.float32, device=dispositivo,
            ).unsqueeze(0)
            with torch.no_grad():
                accion_tensor = agente.seleccionar_accion(estado_tensor, determinista=False)
            accion = accion_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        siguiente_estado, recompensa, terminado, info = entorno.step(accion)

        if paso_global % config.frecuencia_snapshot_cartera == 0:
            resumen_cartera, payload_cartera = resumir_cartera(
                accion=accion,
                nombres_activos=nombres_activos,
                top_k=config.top_k_cartera,
            )
            snapshot = {
                "paso": int(paso_global),
                "riesgo": float(riesgo),
                "reward": float(recompensa),
                "retorno": float(info["retorno_cartera"]),
                "drawdown": float(info["drawdown_actual"]),
                "vol": float(info["vol_ema"]),
                "rotacion": float(info["rotacion"]),
                **payload_cartera,
            }
            historico_snapshots_cartera.append(snapshot)

            print("[CARTERA ACTUAL]")
            print(resumen_cartera)

        if paso_global % 1000 == 0:
            print(
                f"retorno={info['retorno_cartera']:+.6f} | "
                f"pen_total={info['pen_total']:+.6f} | "
                f"reward={recompensa:+.6f} | "
                f"dd={info['drawdown_actual']:.4f} | "
                f"vol={info['vol_ema']:.6f} | "
                f"cash={info['peso_cash_inicio']:.3f} | "
                f"riesgo={info['riesgo']:.2f} || "
                f"pen_dd={info['pen_dd']:.6f} | "
                f"pen_varianza={info['pen_varianza']:.6f} | "
                f"pen_correlacion={info['pen_correlacion']:.6f} | "
                f"pen_concentracion={info['pen_concentracion']:.6f} | "
                f"pen_turn={info['pen_turnover']:.6f} | "
            )

        siguiente_estado_guardar = (
            np.zeros_like(estado, dtype=np.float32)
            if siguiente_estado is None
            else np.asarray(siguiente_estado, dtype=np.float32)
        )

        buffer.guardar_transicion(
            estado=np.asarray(estado, dtype=np.float32),
            accion=np.asarray(accion, dtype=np.float32),
            recompensa=float(recompensa),
            siguiente_estado=siguiente_estado_guardar,
            terminado=bool(terminado),
        )
        historico_recompensas.append(float(recompensa))

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

        if terminado:
            estado = entorno.reset(riesgo=riesgo, aleatorio=True)
            terminado = False
        else:
            estado = siguiente_estado

        if paso_global % config.frecuencia_log == 0:
            recompensa_media = media_movil(historico_recompensas, config.ventana_log_recompensa)
            alpha_actual = (
                _ultimo_o_nan(historico_alpha)
                if historico_alpha
                else float(agente.alpha.detach().cpu().item())
            )
            print(
                f"[Paso {paso_global:>7}] "
                f"reward_media({config.ventana_log_recompensa})={recompensa_media:+.6f} | "
                f"critic1={_ultimo_o_nan(historico_perdida_critic1):.6f} | "
                f"critic2={_ultimo_o_nan(historico_perdida_critic2):.6f} | "
                f"actor={_ultimo_o_nan(historico_perdida_actor):.6f} | "
                f"alpha={alpha_actual:.6f} | "
                f"Q_min={_ultimo_o_nan(historico_q_min):+.6f} | "
                f"Q1={_ultimo_o_nan(historico_q1):+.6f} | "
                f"Q2={_ultimo_o_nan(historico_q2):+.6f} | "
                f"target_Q={_ultimo_o_nan(historico_target_q):+.6f} | "
                f"gap_critics={_ultimo_o_nan(historico_gap_critics):.6f} | "
                f"log_prob={_ultimo_o_nan(historico_log_prob):+.6f} | "
                f"log_prob_std={_ultimo_o_nan(historico_log_prob_std):.6f} | "
                f"entropia={_ultimo_o_nan(historico_entropia):+.6f} | "
                f"mu_min={_ultimo_o_nan(historico_concentracion_min):.4f} | "
                f"mu_max={_ultimo_o_nan(historico_concentracion_max):.4f} | "
                f"mu_media={_ultimo_o_nan(historico_concentracion_media):.4f} | "
                f"std_media={_ultimo_o_nan(historico_concentracion_total):.4f} | "
                f"std_std={_ultimo_o_nan(historico_concentracion_total_std):.4f} | "
                f"accion_min={_ultimo_o_nan(historico_accion_min):.6e} | "
                f"accion_max={_ultimo_o_nan(historico_accion_max):.6f} | "
                f"peso_cash={_ultimo_o_nan(historico_peso_cash):.6f} | "
                f"residual={_ultimo_o_nan(historico_residual_entropia):+.6f} | "
                f"riesgo={riesgo:.2f}"
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
        "snapshots_cartera": historico_snapshots_cartera,
    }

    if devolver_agente:
        return history, agente

    return history


if __name__ == "__main__":
    raise SystemExit(
        "Carga tus datos reales y crea el entorno desde tu notebook o script principal. "
        "Este módulo expone entrenar_sac() y ConfigEntrenamiento."
    )