from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from redes_neuronales import Actor, Critic, SalidaActor
from Replay_buffer import LoteTransiciones

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MetricasActualizacion:
    perdida_critic1: float
    perdida_critic2: float
    perdida_actor: float
    perdida_alpha: float
    coeficiente_entropia_alpha: float
    entropia_media: float
    valor_q_min_medio: float

class AgenteSAC:
    """
    Implementación completa del algoritmo Soft Actor-Critic (SAC)
    adaptada a acciones tipo pesos long-only con cash implícito.

    El agente NO conoce el entorno.
    Recibe lotes de transiciones desde el replay buffer.
    """
    def __init__(
        self,
        dimension_estado: int,
        numero_activos: int,
        dispositivo: torch.device,
        gamma: float = 0.99, # Cuanto más cercano a 1 más valora el retorno futuro
        tau: float = 0.005,
        tasa_aprendizaje_actor: float = 3e-4,
        tasa_aprendizaje_criticos: float = 3e-4,
        tasa_aprendizaje_alpha: float = 3e-4,
        target_entropy: Optional[float] = None,
        dimensiones_ocultas: Tuple[int, int] = (256, 256),
        epsilon_numerico: float = 1e-6,
    ) -> None:
        self.dimension_estado = int(dimension_estado)
        self.numero_activos = int(numero_activos)
        self.dispositivo = dispositivo
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.epsilon_numerico = float(epsilon_numerico)

        # =========================
        # Redes principales
        # =========================
        self.actor = Actor(
            dimension_estado=self.dimension_estado,
            numero_activos=self.numero_activos,
            dimensiones_ocultas=dimensiones_ocultas,
            epsilon_numerico=self.epsilon_numerico,
        ).to(self.dispositivo)

        self.critic1 = Critic(
            dimension_estado=self.dimension_estado,
            numero_activos=self.numero_activos,
            dimensiones_ocultas=dimensiones_ocultas,
        ).to(self.dispositivo)

        self.critic2 = Critic(
            dimension_estado=self.dimension_estado,
            numero_activos=self.numero_activos,
            dimensiones_ocultas=dimensiones_ocultas,
        ).to(self.dispositivo)

        # =========================
        # Redes target (solo critics)
        # =========================
        self.critic1_target = Critic( #copia lenta de critic1
            dimension_estado=self.dimension_estado,
            numero_activos=self.numero_activos,
            dimensiones_ocultas=dimensiones_ocultas,
        ).to(self.dispositivo)

        self.critic2_target = Critic( #copia lenta de critic2
            dimension_estado=self.dimension_estado,
            numero_activos=self.numero_activos,
            dimensiones_ocultas=dimensiones_ocultas,
        ).to(self.dispositivo)

        # Inicialización: target = online
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # =========================
        # Optimizadores
        # =========================
        #Adam es un optimizador del descenso del gradiente
        self.optimizador_actor = torch.optim.Adam(self.actor.parameters(), lr=tasa_aprendizaje_actor)
        self.optimizador_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=tasa_aprendizaje_criticos)
        self.optimizador_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=tasa_aprendizaje_criticos)

        # =========================
        # Alpha automático (log_alpha)
        # =========================
        # target_entropy típico: -dimensión_accion
        if target_entropy is None:
            target_entropy = -float(self.numero_activos)
        self.target_entropy = float(target_entropy)

        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.dispositivo, requires_grad=True)
        self.optimizador_alpha = torch.optim.Adam([self.log_alpha], lr=tasa_aprendizaje_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        """
        coeficiente de entropía alpha = exp(log_alpha)
        """
        return torch.exp(self.log_alpha)

    def seleccionar_accion(
        self, estado: torch.Tensor, determinista: bool = False,) -> torch.Tensor:
        """
        Devuelve solo la acción (pesos) para interactuar con el entorno.
        """
        salida = self.actor.obtener_accion(estado, determinista=determinista)
        return salida.accion

    # ============================================================
    # Actualización de críticos
    # ============================================================
    def actualizar_criticos(self, lote: LoteTransiciones) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actualiza critic1 y critic2 con MSE frente al target SAC.

        Devuelve:
          - perdida_critic1 (tensor escalar)
          - perdida_critic2 (tensor escalar)
          - valor_q_min_medio (tensor escalar) para logging
        """
        estado = lote.estado
        accion = lote.accion
        recompensa = lote.recompensa
        siguiente_estado = lote.siguiente_estado
        terminado = lote.terminado

        with torch.no_grad():
            # Acción siguiente según la política actual
            salida_actor_siguiente = self.actor.obtener_accion(siguiente_estado, determinista=False)
            accion_siguiente = salida_actor_siguiente.accion
            log_prob_siguiente = salida_actor_siguiente.log_prob_accion

            # Q targets
            valor_q1_target = self.critic1_target(siguiente_estado, accion_siguiente)
            valor_q2_target = self.critic2_target(siguiente_estado, accion_siguiente)
            valor_q_min_target = torch.min(valor_q1_target, valor_q2_target)

            # Target SAC completo:
            # y = r + gamma*(1-d)*(min(Q1',Q2') - alpha*log_pi(a'|s'))
            target_q = recompensa + self.gamma * (1.0 - terminado) * (
                valor_q_min_target - self.alpha.detach() * log_prob_siguiente
            )

        # Q online actuales
        valor_q1 = self.critic1(estado, accion)
        valor_q2 = self.critic2(estado, accion)

        # pérdidas (MSE)
        perdida_critic1 = F.mse_loss(valor_q1, target_q)
        perdida_critic2 = F.mse_loss(valor_q2, target_q)

        # Optimizar critic1
        self.optimizador_critic1.zero_grad(set_to_none=True)
        perdida_critic1.backward()
        self.optimizador_critic1.step()

        # Optimizar critic2
        self.optimizador_critic2.zero_grad(set_to_none=True)
        perdida_critic2.backward()
        self.optimizador_critic2.step()

        valor_q_min_medio = torch.mean(torch.min(valor_q1, valor_q2)).detach()

        return perdida_critic1, perdida_critic2, valor_q_min_medio

    # ============================================================
    # Actualización del actor
    # ============================================================
    def actualizar_actor(self, lote: LoteTransiciones) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Actualiza el actor maximizando min(Q1,Q2) y entropía.

        L_actor = E[alpha*log_pi(a|s) - min(Q1,Q2)(s,a)]

        Devuelve:
          - perdida_actor (tensor escalar)
          - entropia_media (tensor escalar)
        """
        estado = lote.estado

        salida_actor = self.actor.obtener_accion(estado, determinista=False)
        accion_nueva = salida_actor.accion
        log_prob_accion = salida_actor.log_prob_accion
        entropia_media = torch.mean(salida_actor.entropia_aproximada).detach()

        valor_q1 = self.critic1(estado, accion_nueva)
        valor_q2 = self.critic2(estado, accion_nueva)
        valor_q_min = torch.min(valor_q1, valor_q2)

        perdida_actor = torch.mean(self.alpha.detach() * log_prob_accion - valor_q_min)

        self.optimizador_actor.zero_grad(set_to_none=True)
        perdida_actor.backward()
        self.optimizador_actor.step()

        return perdida_actor, entropia_media

    # ============================================================
    # Actualización automática de alpha
    # ============================================================
    def actualizar_alpha(self, lote: LoteTransiciones) -> torch.Tensor:
        """
        Ajuste automático de alpha para alcanzar target_entropy.

        L_alpha = E[ -log_alpha * (log_pi(a|s) + target_entropy) ]
        """
        estado = lote.estado

        with torch.no_grad():
            salida_actor = self.actor.obtener_accion(estado, determinista=False)
            log_prob_accion = salida_actor.log_prob_accion

        # La optimización se hace sobre log_alpha
        perdida_alpha = torch.mean(
            -self.log_alpha * (log_prob_accion + self.target_entropy)
        )

        self.optimizador_alpha.zero_grad(set_to_none=True)
        perdida_alpha.backward()
        self.optimizador_alpha.step()

        return perdida_alpha

    # ============================================================
    # Soft update de targets
    # ============================================================
    def actualizar_targets_suavemente(self) -> None:
        """
        Actualización suave:
          theta_target = tau * theta_online + (1-tau) * theta_target
        """
        with torch.no_grad():
            for parametro_online, parametro_target in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                parametro_target.data.mul_(1.0 - self.tau)
                parametro_target.data.add_(self.tau * parametro_online.data)

            for parametro_online, parametro_target in zip(self.critic2.parameters(), self.critic2_target.parameters()):
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
        perdida_critic1, perdida_critic2, valor_q_min_medio = self.actualizar_criticos(lote)
        perdida_actor, entropia_media = self.actualizar_actor(lote)
        perdida_alpha = self.actualizar_alpha(lote)
        self.actualizar_targets_suavemente()

        return MetricasActualizacion(
            perdida_critic1=float(perdida_critic1.detach().cpu().item()),
            perdida_critic2=float(perdida_critic2.detach().cpu().item()),
            perdida_actor=float(perdida_actor.detach().cpu().item()),
            perdida_alpha=float(perdida_alpha.detach().cpu().item()),
            coeficiente_entropia_alpha=float(self.alpha.detach().cpu().item()),
            entropia_media=float(entropia_media.detach().cpu().item()),
            valor_q_min_medio=float(valor_q_min_medio.cpu().item()),
        )