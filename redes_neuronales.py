from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _construir_mlp(
    dimension_entrada: int,
    dimensiones_ocultas: Tuple[int, ...],
    dimension_salida: int,
    activar_salida: bool = False,
) -> nn.Sequential:
    """
    Construye un MLP (perceptrón multicapa) denso.
    """
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
    accion: torch.Tensor              # pesos completos [riesgo..., cash]
    log_prob_accion: torch.Tensor     # log pi(a|s)
    entropia_dirichlet: torch.Tensor  # entropía exacta de la Dirichlet


class Actor(nn.Module):
    """
    Actor estocástico para SAC sobre el simplex usando distribución Dirichlet.

    La acción tiene dimensión:
        dimension_accion = numero_activos_riesgo + 1 (cash explícito)

    Propiedades:
    - accion_i >= 0
    - suma(accion) = 1

    Esto encaja de forma natural con una cartera long-only con cash explícito.
    """

    def __init__(
        self,
        dimension_estado: int,
        dimension_accion: int,
        dimensiones_ocultas: Tuple[int, ...] = (256, 256),
        epsilon_numerico: float = 1e-6,
        max_concentracion_total_extra: float = 7.0,
    ) -> None:
        super().__init__()
        self.dimension_estado = int(dimension_estado)
        self.dimension_accion = int(dimension_accion)
        self.epsilon_numerico = float(epsilon_numerico)
        self.max_concentracion_total_extra = float(max_concentracion_total_extra)

        self.red_base = _construir_mlp(
            dimension_entrada=self.dimension_estado,
            dimensiones_ocultas=dimensiones_ocultas,
            dimension_salida=dimensiones_ocultas[-1],
            activar_salida=True,
        )

        # Parametrización estable de la Dirichlet:
        # - una media sobre el simplex
        # - una concentración total escalar
        self.capa_logits_media = nn.Linear(dimensiones_ocultas[-1], self.dimension_accion)
        self.capa_concentracion_total = nn.Linear(dimensiones_ocultas[-1], 1)

    def forward(self, estado: torch.Tensor) -> torch.Tensor:
        """
        Devuelve las concentraciones de la Dirichlet con una parametrización
        más estable:

            concentraciones = 1.0 + media * concentracion_total_extra

        donde:
        - media está en el simplex
        - concentracion_total_extra >= 0
        - el +1.0 garantiza que todas las componentes sean > 1
        """
        representacion = self.red_base(estado)

        logits_media = self.capa_logits_media(representacion)
        media = torch.softmax(logits_media, dim=1)

        concentracion_total_extra = F.softplus(
            self.capa_concentracion_total(representacion)
        )

        concentracion_total_extra = torch.clamp(
            concentracion_total_extra,
            min=1e-3,
            max=self.max_concentracion_total_extra,
        )

        concentraciones = 1.0 + media * concentracion_total_extra
        return concentraciones

    def obtener_accion(
        self,
        estado: torch.Tensor,
        determinista: bool = False,
    ) -> SalidaActor:
        """
        Genera una acción y su log_prob asociado.

        determinista=False:
            usa una muestra reparametrizada de la Dirichlet
            -> esto es lo que se debe usar en entrenamiento

        determinista=True:
            usa la media de la Dirichlet
            -> esto es lo que se debe usar en validación/backtest

        Importante:
        NO hacemos clamp ni renormalización después, porque la muestra de
        Dirichlet ya pertenece al simplex y tocarla rompe la coherencia del log_prob.
        """
        concentraciones = self.forward(estado)
        distribucion = torch.distributions.Dirichlet(concentraciones)

        if determinista:
            accion = concentraciones / torch.sum(concentraciones, dim=1, keepdim=True)
        else:
            accion = distribucion.rsample()

        log_prob_accion = distribucion.log_prob(accion).unsqueeze(1)
        entropia_dirichlet = distribucion.entropy().unsqueeze(1)

        return SalidaActor(
            accion=accion,
            log_prob_accion=log_prob_accion,
            entropia_dirichlet=entropia_dirichlet,
        )


class Critic(nn.Module):
    """
    Critic Q(s,a): aproxima un valor escalar dado estado y acción.

    La acción incluye activos de riesgo + cash explícito.
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
        """
        Devuelve valor_q con shape (batch, 1)
        """
        entrada = torch.cat([estado, accion], dim=1)
        valor_q = self.red_q(entrada)
        return valor_q