from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from torch.distributions import Normal


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _construir_mlp(
    dimension_entrada: int,
    dimensiones_ocultas: Tuple[int, ...],
    dimension_salida: int,
    activar_salida: bool = False,
) -> nn.Sequential:
    capas = []
    dimension_actual = dimension_entrada

    for dimension_oculta in dimensiones_ocultas:
        capas.append(nn.Linear(dimension_actual, dimension_oculta))
        capas.append(nn.ReLU())
        dimension_actual = dimension_oculta

    capas.append(nn.Linear(dimension_actual, dimension_salida))
    if activar_salida:
        capas.append(nn.ReLU())

    return nn.Sequential(*capas)


@dataclass
class SalidaActor:
    accion: torch.Tensor          # pesos de cartera en el simplex  [B, A]
    log_prob_accion: torch.Tensor  # log_prob de la acción           [B, 1]
    entropia: torch.Tensor         # entropía gaussiana estimada     [B, 1]


# ============================================================
# Nota sobre el cálculo de log_prob
# ============================================================
# La política muestrea x ~ N(mu, std²·I) y luego aplica softmax(x).
# El log_prob exacto sobre el simplex requeriría el jacobiano de softmax,
# que es difícil de calcular de forma estable y compacta.
#
# Usamos la aproximación estándar en SAC continuo:
#   log π(a|s)  ≈  Σ_i log N(x_i | mu_i, std_i)
#
# Es decir, calculamos la log-densidad gaussiana ANTES de la transformación
# softmax. Esto es una aproximación (ignoramos el jacobiano), pero es la
# misma estrategia que se usa en SAC con tanh (donde sí se corrige el
# jacobiano). Aquí, el jacobiano de softmax no tiene una forma diagonal
# y su corrección exacta no mejora el aprendizaje en la práctica.
#
# Lo importante para SAC es que log_prob sea:
#   - diferenciable con respecto a los parámetros del actor
#   - monótono en concentración: carteras más concentradas → log_prob más alto
#   - coherente entre distintos estados del mismo batch
#
# Estas tres propiedades se cumplen con la aproximación gaussiana.
# ============================================================

LOG_2PI = math.log(2.0 * math.pi)

class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
        temperature: float = 5.0,
        eps_mix_uniforme: float = 0.0,
        bias_cash_inicial: float = 0.0,
        peso_max_activo: float = 0.943,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.temperature = temperature
        self.eps_mix_uniforme = eps_mix_uniforme
        self.peso_max_activo = peso_max_activo

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, act_dim)

        # log_std como parámetro global aprendible (no depende del estado)
        # Sustituye log_std_head que siempre saturaba en log_std_max.
        # log_std_init=-1.0 → std inicial ≈ 0.37
        self.log_std = nn.Parameter(
            torch.full((act_dim,), -1.0)
        )

        # Sesgo inicial hacia cash (cash = última dimensión)
        nn.init.constant_(self.mu_head.bias, -2.0)
        with torch.no_grad():
            self.mu_head.bias[-1] = bias_cash_inicial

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def _weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Centramos logits para estabilidad numérica
        logits_centrados = logits - logits.mean(dim=-1, keepdim=True)

        # Softmax con temperatura
        pesos_soft = torch.softmax(logits_centrados / self.temperature, dim=-1)

        # Mezcla con uniforme
        if self.eps_mix_uniforme > 0.0:
            uniforme = torch.full_like(pesos_soft, 1.0 / self.act_dim)
            pesos = (1.0 - self.eps_mix_uniforme) * pesos_soft + self.eps_mix_uniforme * uniforme
        else:
            pesos = pesos_soft

        # Mezcla con uniforme para garantizar masa mínima en todos los activos
        # Esto evita que softmax colapse a peso=1 con mu muy grandes,
        # lo que hace que el clamp posterior no pueda redistribuir
        eps_piso = 1.0 / (self.act_dim * 2)  # piso mínimo ~3% con 17 activos
        uniforme = torch.full_like(pesos, 1.0 / self.act_dim)
        pesos = (1.0 - eps_piso) * pesos + eps_piso * uniforme

        # Clamp iterativo: limita peso máximo por activo
        for _ in range(20):
            if not (pesos > self.peso_max_activo).any():
                break
            pesos = torch.clamp(pesos, max=self.peso_max_activo)
            pesos = pesos / pesos.sum(dim=-1, keepdim=True)
        return pesos

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Devuelve:
        - action_weights: pesos finales en simplex
        - log_prob: log_prob gaussiano sobre logits pre-softmax
        - mu: medias de logits
        - std: desviaciones típicas
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()

        dist = Normal(mu, std)
        eps = torch.randn_like(mu)
        logits = mu + std * eps

        action_weights = self._weights_from_logits(logits)

        # Log-densidad gaussiana analítica directa
        # gradiente d(log_prob)/d(log_std) = -1 por dimensión, siempre activo
        log_prob = (
            -0.5 * (eps ** 2).sum(dim=-1, keepdim=True)
            - log_std.sum(dim=-1, keepdim=True)
            - 0.5 * self.act_dim * LOG_2PI
        )

        return action_weights, log_prob, mu, std

    def deterministic_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        action_weights = self._weights_from_logits(mu)
        return action_weights, mu, std
    

class Critic(nn.Module):
    """
    Critic Q(s,a): aproxima un valor escalar dado estado y acción.
    Sin cambios respecto a la versión original.
    """

    def __init__(
        self,
        dimension_estado: int,
        dimension_accion: int,
        dimensiones_ocultas: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.dimension_estado = int(dimension_estado)
        self.dimension_accion = int(dimension_accion)
        dimension_entrada = self.dimension_estado + self.dimension_accion
        self.red_q = _construir_mlp(
            dimension_entrada=dimension_entrada,
            dimensiones_ocultas=dimensiones_ocultas,
            dimension_salida=1,
            activar_salida=False,
        )

    def forward(self, estado: torch.Tensor, accion: torch.Tensor) -> torch.Tensor:
        entrada = torch.cat([estado, accion], dim=1)
        return self.red_q(entrada)