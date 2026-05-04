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

    # Reutilizamos estos nombres para no romper el resto del pipeline
    # aunque ahora ya no sean "concentraciones Dirichlet":
    #   concentracion_min         -> mu_min
    #   concentracion_max         -> mu_max
    #   concentracion_media       -> mu_media
    #   concentracion_total_media -> std_media
    #   concentracion_total_std   -> std_std
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
    Soft Actor-Critic para carteras long-only con cash explícito,
    usando política gaussiana sobre logits + softmax.

    Convención:
    - El actor produce logits gaussianos.
    - La acción final son pesos long-only que suman 1 tras softmax.
    - log_prob se calcula sobre los logits gaussianos pre-softmax.
    - Alpha se ajusta automáticamente como en SAC clásico.
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
        reward_scale: float = 5.0,
        offset_target_entropy: float = 0.0,
        max_concentracion_total_extra: float = 7.0,  # compatibilidad, ya no se usa
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

        hidden_dim = int(dimensiones_ocultas[0])

        # ============================================================
        # Redes principales
        # ============================================================
        self.actor = Actor(
            obs_dim=self.dimension_estado,
            act_dim=self.dimension_accion,
            hidden_dim=hidden_dim,
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
        # Targets
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
            # SAC clásico en acción continua:
            # target por defecto ~ -dimensión de acción
            target_entropy = -float(self.dimension_accion) + float(self.offset_target_entropy)

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
        return torch.exp(self.log_alpha)

    # ============================================================
    # Acción
    # ============================================================
    def seleccionar_accion(
        self,
        estado: torch.Tensor,
        determinista: bool = False,
    ) -> torch.Tensor:
        if determinista:
            accion, _, _ = self.actor.deterministic_action(estado)
            return accion

        accion, _, _, _ = self.actor.sample_action(estado)
        return accion

    # ============================================================
    # Críticos
    # ============================================================
    def actualizar_criticos(
        self,
        lote: LoteTransiciones,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        estado = lote.estado
        accion = lote.accion
        recompensa = lote.recompensa
        siguiente_estado = lote.siguiente_estado
        terminado = lote.terminado

        with torch.no_grad():
            accion_siguiente, log_prob_siguiente, _, _ = self.actor.sample_action(
                siguiente_estado
            )

            q1_target = self.critic1_target(siguiente_estado, accion_siguiente)
            q2_target = self.critic2_target(siguiente_estado, accion_siguiente)
            q_min_target = torch.min(q1_target, q2_target)

            recompensa_escalada = self.reward_scale * recompensa

            target_q = recompensa_escalada + self.gamma * (1.0 - terminado) * (
                q_min_target - self.alpha.detach() * log_prob_siguiente
            )

        q1 = self.critic1(estado, accion)
        q2 = self.critic2(estado, accion)

        perdida_critic1 = F.mse_loss(q1, target_q)
        perdida_critic2 = F.mse_loss(q2, target_q)

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

        q_min_medio = torch.mean(torch.min(q1, q2)).detach()
        q1_medio = torch.mean(q1).detach()
        q2_medio = torch.mean(q2).detach()
        target_q_medio = torch.mean(target_q).detach()
        gap_medio = torch.mean(torch.abs(q1 - q2)).detach()

        return (
            perdida_critic1,
            perdida_critic2,
            q_min_medio,
            q1_medio,
            q2_medio,
            target_q_medio,
            gap_medio,
        )

    # ============================================================
    # Actor
    # ============================================================
    def actualizar_actor(
        self,
        lote: LoteTransiciones,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        estado = lote.estado

        accion_nueva, log_prob_accion, mu, std = self.actor.sample_action(estado)

        # Como métrica de "entropía" para logging, usamos -log_prob medio.
        # No es la entropía exacta diferencial de toda la política transformada,
        # pero sí una métrica consistente con el ajuste de alpha en SAC.
        entropia_media = torch.mean(-log_prob_accion).detach()

        log_prob_medio = torch.mean(log_prob_accion).detach()
        log_prob_std = torch.std(log_prob_accion).detach()

        mu_min = torch.min(mu).detach()
        mu_max = torch.max(mu).detach()
        mu_media = torch.mean(mu).detach()
        std_media = torch.mean(std).detach()
        std_std = torch.std(std).detach()

        accion_min = torch.min(accion_nueva).detach()
        accion_max = torch.max(accion_nueva).detach()
        peso_cash_medio = torch.mean(accion_nueva[:, -1]).detach()

        # Stop-gradient sobre log_std: Q no propaga gradientes a log_std
        log_std_actual = torch.clamp(
            self.actor.log_std, self.actor.log_std_min, self.actor.log_std_max
        )
        log_prob_sg = (
            log_prob_accion
            + log_std_actual.sum()
            - log_std_actual.detach().sum()
        )

        q1 = self.critic1(estado, accion_nueva)
        q2 = self.critic2(estado, accion_nueva)
        q_min = torch.min(q1, q2)

        # Regularización L2 sobre mu para evitar explosión
        reg_mu = 1e-3 * (mu ** 2).mean()

        perdida_actor = torch.mean(
            self.alpha.detach() * log_prob_sg - q_min
        ) + reg_mu

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
            mu_min,
            mu_max,
            mu_media,
            std_media,
            std_std,
            accion_min,
            accion_max,
            peso_cash_medio,
            log_prob_accion.detach(),  # para actualizar_alpha — sin warning, tensor ya en device
        )

    # ============================================================
    # Alpha
    # ============================================================
    def actualizar_alpha(
        self,
        lote: LoteTransiciones,
        log_prob_detached: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # log_prob_detached viene de actualizar_actor (ya sin grafo computacional).
        # Lo usamos para alpha — no necesita gradiente hacia el actor.
        residual_entropia = log_prob_detached - self.target_entropy
        residual_entropia_medio = torch.mean(residual_entropia).detach()

        # Actualizar alpha
        perdida_alpha = torch.mean(-self.log_alpha * residual_entropia.detach())
        self.optimizador_alpha.zero_grad(set_to_none=True)
        perdida_alpha.backward()
        self.optimizador_alpha.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=-5.0, max=4.0)

        # perdida_log_std necesita grafo fresco — el de actualizar_actor ya fue liberado.
        # Solo es un forward del actor (sin critics), coste pequeño.
        if isinstance(self.actor.log_std, torch.nn.Parameter):
            _, log_prob_fresco, _, _ = self.actor.sample_action(lote.estado)
            perdida_log_std = torch.mean((log_prob_fresco - self.target_entropy) ** 2)
            self.optimizador_actor.zero_grad(set_to_none=True)
            perdida_log_std.backward()
            self.optimizador_actor.step()

        return perdida_alpha.detach(), residual_entropia_medio

    # ============================================================
    # Targets
    # ============================================================
    def actualizar_targets_suavemente(self) -> None:
        with torch.no_grad():
            for online, target in zip(
                self.critic1.parameters(),
                self.critic1_target.parameters(),
            ):
                target.data.mul_(1.0 - self.tau)
                target.data.add_(self.tau * online.data)

            for online, target in zip(
                self.critic2.parameters(),
                self.critic2_target.parameters(),
            ):
                target.data.mul_(1.0 - self.tau)
                target.data.add_(self.tau * online.data)

    # ============================================================
    # Paso completo
    # ============================================================
    def paso_entrenamiento(
        self,
        lote: LoteTransiciones,
    ) -> MetricasActualizacion:
        (
            perdida_critic1,
            perdida_critic2,
            q_min_medio,
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
            mu_min,
            mu_max,
            mu_media,
            std_media,
            std_std,
            accion_min,
            accion_max,
            peso_cash_medio,
            log_prob_accion_detached,
        ) = self.actualizar_actor(lote)

        # Pasamos log_prob ya detacheado — alpha no necesita grafo del actor
        perdida_alpha, residual_entropia_medio = self.actualizar_alpha(
            lote, log_prob_detached=log_prob_accion_detached
        )
        self.actualizar_targets_suavemente()

        return MetricasActualizacion(
            perdida_critic1=float(perdida_critic1.detach().cpu().item()),
            perdida_critic2=float(perdida_critic2.detach().cpu().item()),
            perdida_actor=float(perdida_actor.detach().cpu().item()),
            perdida_alpha=float(perdida_alpha.detach().cpu().item()),
            coeficiente_entropia_alpha=float(self.alpha.detach().cpu().item()),
            entropia_media=float(entropia_media.cpu().item()),
            residual_entropia_medio=float(residual_entropia_medio.cpu().item()),
            valor_q_min_medio=float(q_min_medio.cpu().item()),
            q1_medio=float(q1_medio.cpu().item()),
            q2_medio=float(q2_medio.cpu().item()),
            target_q_medio=float(target_q_medio.cpu().item()),
            gap_critics_medio=float(gap_critics_medio.cpu().item()),
            log_prob_medio=float(log_prob_medio.cpu().item()),
            log_prob_std=float(log_prob_std.cpu().item()),
            concentracion_min=float(mu_min.cpu().item()),
            concentracion_max=float(mu_max.cpu().item()),
            concentracion_media=float(mu_media.cpu().item()),
            concentracion_total_media=float(std_media.cpu().item()),
            concentracion_total_std=float(std_std.cpu().item()),
            accion_min=float(accion_min.cpu().item()),
            accion_max=float(accion_max.cpu().item()),
            peso_cash_medio=float(peso_cash_medio.cpu().item()),
        )