from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from redes_neuronales import Actor, Critic
from Replay_buffer import LoteTransiciones


@dataclass
class MetricasActualizacion:
    perdida_critic1: float
    perdida_critic2: float
    perdida_actor: float
    perdida_alpha: float

    coeficiente_entropia_alpha: float
    entropia_media: float
    residual_entropia_medio: float

    valor_q_min_medio: float
    q1_medio: float
    q2_medio: float
    target_q_medio: float
    gap_critics_medio: float

    log_prob_medio: float
    log_prob_std: float

    concentracion_min: float
    concentracion_max: float
    concentracion_media: float
    concentracion_total_media: float
    concentracion_total_std: float

    accion_min: float
    accion_max: float
    peso_cash_medio: float


class AgenteSAC:
    """
    Implementación de Soft Actor-Critic (SAC) para carteras long-only
    con cash explícito.

    La acción tiene dimensión:
        dimension_accion = numero_activos_riesgo + 1

    y representa:
        [w_1, ..., w_N, w_cash]

    El agente no conoce el entorno.
    Recibe lotes de transiciones desde el replay buffer.
    """

    def __init__(
        self,
        dimension_estado: int,
        dimension_accion: int,
        dispositivo: torch.device,
        gamma: float = 0.99,
        tau: float = 0.005,
        tasa_aprendizaje_actor: float = 3e-4,
        tasa_aprendizaje_criticos: float = 3e-4,
        tasa_aprendizaje_alpha: float = 3e-4,
        target_entropy: Optional[float] = None,
        dimensiones_ocultas: Tuple[int, int] = (256, 256),
        epsilon_numerico: float = 1e-6,
        max_norm_gradiente: float = 1.0,
        reward_scale: float = 1000.0,
        offset_target_entropy: float = -5.0,
        max_concentracion_total_extra: float = 7.0,
    ) -> None:
        self.dimension_estado = int(dimension_estado)
        self.dimension_accion = int(dimension_accion)
        self.dispositivo = dispositivo

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.epsilon_numerico = float(epsilon_numerico)
        self.max_norm_gradiente = float(max_norm_gradiente)
        self.reward_scale = float(reward_scale)
        self.offset_target_entropy = float(offset_target_entropy)
        self.max_concentracion_total_extra = float(max_concentracion_total_extra)

        # ============================================================
        # Redes principales
        # ============================================================
        self.actor = Actor(
            dimension_estado=self.dimension_estado,
            dimension_accion=self.dimension_accion,
            dimensiones_ocultas=dimensiones_ocultas,
            epsilon_numerico=self.epsilon_numerico,
            max_concentracion_total_extra=self.max_concentracion_total_extra,
        ).to(self.dispositivo)

        self.critic1 = Critic(
            dimension_estado=self.dimension_estado,
            dimension_accion=self.dimension_accion,
            dimensiones_ocultas=dimensiones_ocultas,
        ).to(self.dispositivo)

        self.critic2 = Critic(
            dimension_estado=self.dimension_estado,
            dimension_accion=self.dimension_accion,
            dimensiones_ocultas=dimensiones_ocultas,
        ).to(self.dispositivo)

        # ============================================================
        # Redes target (solo critics)
        # ============================================================
        self.critic1_target = Critic(
            dimension_estado=self.dimension_estado,
            dimension_accion=self.dimension_accion,
            dimensiones_ocultas=dimensiones_ocultas,
        ).to(self.dispositivo)

        self.critic2_target = Critic(
            dimension_estado=self.dimension_estado,
            dimension_accion=self.dimension_accion,
            dimensiones_ocultas=dimensiones_ocultas,
        ).to(self.dispositivo)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # ============================================================
        # Optimizadores
        # ============================================================
        self.optimizador_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=tasa_aprendizaje_actor,
        )
        self.optimizador_critic1 = torch.optim.Adam(
            self.critic1.parameters(),
            lr=tasa_aprendizaje_criticos,
        )
        self.optimizador_critic2 = torch.optim.Adam(
            self.critic2.parameters(),
            lr=tasa_aprendizaje_criticos,
        )

        # ============================================================
        # Alpha automático
        # ============================================================
        if target_entropy is None:
            alpha_ref = torch.ones(
                self.dimension_accion,
                dtype=torch.float32,
                device=self.dispositivo,
            )
            dist_ref = torch.distributions.Dirichlet(alpha_ref)
            entropia_uniforme = float(dist_ref.entropy().detach().cpu().item())
            target_entropy = entropia_uniforme + self.offset_target_entropy

        self.target_entropy = float(target_entropy)

        self.log_alpha = torch.tensor(
            [0.0],
            dtype=torch.float32,
            device=self.dispositivo,
            requires_grad=True,
        )
        self.optimizador_alpha = torch.optim.Adam(
            [self.log_alpha],
            lr=tasa_aprendizaje_alpha,
        )

    @property
    def alpha(self) -> torch.Tensor:
        """
        Coeficiente de entropía:
            alpha = exp(log_alpha)
        """
        return torch.exp(self.log_alpha)

    def seleccionar_accion(
        self,
        estado: torch.Tensor,
        determinista: bool = False,
    ) -> torch.Tensor:
        """
        Devuelve solo la acción para interactuar con el entorno.
        """
        salida = self.actor.obtener_accion(estado, determinista=determinista)
        return salida.accion

    # ============================================================
    # Actualización de críticos
    # ============================================================
    def actualizar_criticos(
        self,
        lote: LoteTransiciones,
    ) -> Tuple[
        torch.Tensor,  # perdida_critic1
        torch.Tensor,  # perdida_critic2
        torch.Tensor,  # valor_q_min_medio
        torch.Tensor,  # q1_medio
        torch.Tensor,  # q2_medio
        torch.Tensor,  # target_q_medio
        torch.Tensor,  # gap_critics_medio
    ]:
        """
        Actualiza critic1 y critic2 con MSE frente al target SAC.
        """
        estado = lote.estado
        accion = lote.accion
        recompensa = lote.recompensa
        siguiente_estado = lote.siguiente_estado
        terminado = lote.terminado

        with torch.no_grad():
            salida_actor_siguiente = self.actor.obtener_accion(
                siguiente_estado,
                determinista=False,
            )
            accion_siguiente = salida_actor_siguiente.accion
            log_prob_siguiente = salida_actor_siguiente.log_prob_accion

            valor_q1_target = self.critic1_target(siguiente_estado, accion_siguiente)
            valor_q2_target = self.critic2_target(siguiente_estado, accion_siguiente)
            valor_q_min_target = torch.min(valor_q1_target, valor_q2_target)

            recompensa_escalada = self.reward_scale * recompensa

            target_q = recompensa_escalada + self.gamma * (1.0 - terminado) * (
                valor_q_min_target - self.alpha.detach() * log_prob_siguiente
            )

        valor_q1 = self.critic1(estado, accion)
        valor_q2 = self.critic2(estado, accion)

        perdida_critic1 = F.mse_loss(valor_q1, target_q)
        perdida_critic2 = F.mse_loss(valor_q2, target_q)

        self.optimizador_critic1.zero_grad(set_to_none=True)
        perdida_critic1.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic1.parameters(),
            self.max_norm_gradiente,
        )
        self.optimizador_critic1.step()

        self.optimizador_critic2.zero_grad(set_to_none=True)
        perdida_critic2.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic2.parameters(),
            self.max_norm_gradiente,
        )
        self.optimizador_critic2.step()

        valor_q_min_medio = torch.mean(torch.min(valor_q1, valor_q2)).detach()
        q1_medio = torch.mean(valor_q1).detach()
        q2_medio = torch.mean(valor_q2).detach()
        target_q_medio = torch.mean(target_q).detach()
        gap_critics_medio = torch.mean(torch.abs(valor_q1 - valor_q2)).detach()

        return (
            perdida_critic1,
            perdida_critic2,
            valor_q_min_medio,
            q1_medio,
            q2_medio,
            target_q_medio,
            gap_critics_medio,
        )

    # ============================================================
    # Actualización del actor
    # ============================================================
    def actualizar_actor(
        self,
        lote: LoteTransiciones,
    ) -> Tuple[
        torch.Tensor,  # perdida_actor
        torch.Tensor,  # entropia_media
        torch.Tensor,  # log_prob_medio
        torch.Tensor,  # log_prob_std
        torch.Tensor,  # concentracion_min
        torch.Tensor,  # concentracion_max
        torch.Tensor,  # concentracion_media
        torch.Tensor,  # concentracion_total_media
        torch.Tensor,  # concentracion_total_std
        torch.Tensor,  # accion_min
        torch.Tensor,  # accion_max
        torch.Tensor,  # peso_cash_medio
    ]:
        """
        Actualiza el actor y además calcula métricas para depuración
        de la política Dirichlet.
        """
        estado = lote.estado

        salida_actor = self.actor.obtener_accion(estado, determinista=False)
        accion_nueva = salida_actor.accion
        log_prob_accion = salida_actor.log_prob_accion
        entropia_media = torch.mean(salida_actor.entropia_dirichlet).detach()

        concentraciones = self.actor.forward(estado)
        concentracion_total = torch.sum(concentraciones, dim=1)

        log_prob_medio = torch.mean(log_prob_accion).detach()
        log_prob_std = torch.std(log_prob_accion).detach()

        concentracion_min = torch.min(concentraciones).detach()
        concentracion_max = torch.max(concentraciones).detach()
        concentracion_media = torch.mean(concentraciones).detach()
        concentracion_total_media = torch.mean(concentracion_total).detach()
        concentracion_total_std = torch.std(concentracion_total).detach()

        accion_min = torch.min(accion_nueva).detach()
        accion_max = torch.max(accion_nueva).detach()
        peso_cash_medio = torch.mean(accion_nueva[:, -1]).detach()

        valor_q1 = self.critic1(estado, accion_nueva)
        valor_q2 = self.critic2(estado, accion_nueva)
        valor_q_min = torch.min(valor_q1, valor_q2)

        perdida_actor = torch.mean(
            self.alpha.detach() * log_prob_accion - valor_q_min
        )

        self.optimizador_actor.zero_grad(set_to_none=True)
        perdida_actor.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.max_norm_gradiente,
        )
        self.optimizador_actor.step()

        return (
            perdida_actor,
            entropia_media,
            log_prob_medio,
            log_prob_std,
            concentracion_min,
            concentracion_max,
            concentracion_media,
            concentracion_total_media,
            concentracion_total_std,
            accion_min,
            accion_max,
            peso_cash_medio,
        )

    # ============================================================
    # Actualización automática de alpha
    # ============================================================
    def actualizar_alpha(
        self,
        lote: LoteTransiciones,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ajuste automático de alpha para alcanzar target_entropy.

        Devuelve:
          - perdida_alpha
          - residual_entropia_medio
        """
        estado = lote.estado

        with torch.no_grad():
            salida_actor = self.actor.obtener_accion(estado, determinista=False)
            log_prob_accion = salida_actor.log_prob_accion

        residual_entropia = log_prob_accion + self.target_entropy
        residual_entropia_medio = torch.mean(residual_entropia).detach()

        perdida_alpha = torch.mean(-self.log_alpha * residual_entropia)

        self.optimizador_alpha.zero_grad(set_to_none=True)
        perdida_alpha.backward()
        self.optimizador_alpha.step()

        with torch.no_grad():
            self.log_alpha.clamp_(min=-20.0, max=2.0)

        return perdida_alpha, residual_entropia_medio

    # ============================================================
    # Soft update de targets
    # ============================================================
    def actualizar_targets_suavemente(self) -> None:
        """
        Actualización suave:
            theta_target = tau * theta_online + (1 - tau) * theta_target
        """
        with torch.no_grad():
            for parametro_online, parametro_target in zip(
                self.critic1.parameters(),
                self.critic1_target.parameters(),
            ):
                parametro_target.data.mul_(1.0 - self.tau)
                parametro_target.data.add_(self.tau * parametro_online.data)

            for parametro_online, parametro_target in zip(
                self.critic2.parameters(),
                self.critic2_target.parameters(),
            ):
                parametro_target.data.mul_(1.0 - self.tau)
                parametro_target.data.add_(self.tau * parametro_online.data)

    # ============================================================
    # Paso completo de entrenamiento
    # ============================================================
    def paso_entrenamiento(self, lote: LoteTransiciones) -> MetricasActualizacion:
        """
        Ejecuta un paso completo:
        - actualizar críticos
        - actualizar actor
        - actualizar alpha
        - actualizar targets suavemente
        """
        (
            perdida_critic1,
            perdida_critic2,
            valor_q_min_medio,
            q1_medio,
            q2_medio,
            target_q_medio,
            gap_critics_medio,
        ) = self.actualizar_criticos(lote)

        (
            perdida_actor,
            entropia_media,
            log_prob_medio,
            log_prob_std,
            concentracion_min,
            concentracion_max,
            concentracion_media,
            concentracion_total_media,
            concentracion_total_std,
            accion_min,
            accion_max,
            peso_cash_medio,
        ) = self.actualizar_actor(lote)

        perdida_alpha, residual_entropia_medio = self.actualizar_alpha(lote)
        self.actualizar_targets_suavemente()

        return MetricasActualizacion(
            perdida_critic1=float(perdida_critic1.detach().cpu().item()),
            perdida_critic2=float(perdida_critic2.detach().cpu().item()),
            perdida_actor=float(perdida_actor.detach().cpu().item()),
            perdida_alpha=float(perdida_alpha.detach().cpu().item()),
            coeficiente_entropia_alpha=float(self.alpha.detach().cpu().item()),
            entropia_media=float(entropia_media.cpu().item()),
            residual_entropia_medio=float(residual_entropia_medio.cpu().item()),
            valor_q_min_medio=float(valor_q_min_medio.cpu().item()),
            q1_medio=float(q1_medio.cpu().item()),
            q2_medio=float(q2_medio.cpu().item()),
            target_q_medio=float(target_q_medio.cpu().item()),
            gap_critics_medio=float(gap_critics_medio.cpu().item()),
            log_prob_medio=float(log_prob_medio.cpu().item()),
            log_prob_std=float(log_prob_std.cpu().item()),
            concentracion_min=float(concentracion_min.cpu().item()),
            concentracion_max=float(concentracion_max.cpu().item()),
            concentracion_media=float(concentracion_media.cpu().item()),
            concentracion_total_media=float(concentracion_total_media.cpu().item()),
            concentracion_total_std=float(concentracion_total_std.cpu().item()),
            accion_min=float(accion_min.cpu().item()),
            accion_max=float(accion_max.cpu().item()),
            peso_cash_medio=float(peso_cash_medio.cpu().item()),
        )