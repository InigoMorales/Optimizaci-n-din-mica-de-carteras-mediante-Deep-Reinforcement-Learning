from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

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
    accion: torch.Tensor                 # vector de pesos de activos
    log_prob_accion: torch.Tensor        # logaritmo de la probabilidad de que el actor haga esa acción (log pi(a|s))
    entropia_aproximada: torch.Tensor    # entropía aprox = -log_prob

class Actor(nn.Module):
    """
    Actor estocástico para SAC.

    Produce una distribución Gaussiana en espacio latente z, y transforma:
        z -> u = sigmoid(z) en (0,1)
        u -> pesos (stick-breaking) en [0,1] con suma <= 1

    Devuelve:
        - accion: pesos long-only con cash implícito
        - log_prob_accion: log pi(a|s) con correcciones por transformaciones
    """

    def __init__(
        self,
        dimension_estado: int,
        numero_activos: int,
        dimensiones_ocultas: Tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        epsilon_numerico: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dimension_estado = int(dimension_estado)
        self.numero_activos = int(numero_activos)

        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.epsilon_numerico = float(epsilon_numerico)

        # Red común
        self.red_base = _construir_mlp(
            dimension_entrada=self.dimension_estado,
            dimensiones_ocultas=dimensiones_ocultas,
            dimension_salida=dimensiones_ocultas[-1],
            activar_salida=True,
        )

        # Cabezas de media y log_std (por activo)
        self.capa_media = nn.Linear(dimensiones_ocultas[-1], self.numero_activos)
        self.capa_log_std = nn.Linear(dimensiones_ocultas[-1], self.numero_activos)

    def forward(self, estado: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve (media, log_std) del Gaussiano latente.
        """
        representacion = self.red_base(estado)
        media = self.capa_media(representacion)
        log_std = self.capa_log_std(representacion)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return media, log_std

    def obtener_accion(
        self,
        estado: torch.Tensor,
        determinista: bool = False,
    ) -> SalidaActor:
        """
        Genera una acción (pesos) y su log_prob asociado.

        determinista=False: usa muestreo reparametrizado (entrenamiento)
        determinista=True: usa la media (evaluación/backtest determinista)
        """
        media, log_std = self.forward(estado)
        desviacion = torch.exp(log_std)

        if determinista:
            z = media
        else:
            ruido = torch.randn_like(media)
            z = media + desviacion * ruido

        # Transformación a (0,1): u = sigmoid(z)
        u = torch.sigmoid(z)
        u = torch.clamp(u, self.epsilon_numerico, 1.0 - self.epsilon_numerico)

        # Stick-breaking: garantiza pesos en [0,1] y suma <= 1
        accion, log_det_jacobiano_stick = self._stick_breaking(u)

        # log_prob en z (Normal independiente por dimensión)
        # log N(z; media, std)
        log_prob_z = self._log_prob_normal_independiente(z, media, log_std)

        # Corrección Jacobiano sigmoid: u = sigmoid(z)
        # du/dz = u(1-u)  ->  log|dz/du| = -log(u(1-u))  (porque necesitamos p(u))
        # p(u) = p(z) * |dz/du|
        # log p(u) = log p(z) - sum log(u(1-u))
        log_det_jacobiano_sigmoid = -torch.sum(torch.log(u * (1.0 - u)), dim=1, keepdim=True)

        # Transformación u -> accion (stick-breaking):
        # p(accion) = p(u) * |du/daccion|
        # log p(accion) = log p(u) + log|du/daccion|
        # Aquí calculamos log|du/daccion| = -log|daccion/du| = - log_det(daccion/du)
        log_prob_accion = log_prob_z + log_det_jacobiano_sigmoid - log_det_jacobiano_stick

        entropia_aproximada = -log_prob_accion

        return SalidaActor(
            accion=accion,
            log_prob_accion=log_prob_accion,
            entropia_aproximada=entropia_aproximada,
        )

    @staticmethod
    def _log_prob_normal_independiente(
        z: torch.Tensor,
        media: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log-prob de una Normal factorized N(media, std) en cada dimensión.
        Devuelve shape (batch, 1).
        """
        # std = exp(log_std)
        # log N = -0.5 * [ ((z-media)/std)^2 + 2 log_std + log(2pi) ]
        constante = torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))
        varianza_normalizada = ((z - media) ** 2) * torch.exp(-2.0 * log_std)
        log_prob_por_dim = -0.5 * (varianza_normalizada + 2.0 * log_std + constante)
        return torch.sum(log_prob_por_dim, dim=1, keepdim=True)

    def _stick_breaking(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convierte u en pesos accion usando stick-breaking (sin softmax).
        Garantiza sum(pesos) <= 1.

        También devuelve log_det(d_accion/d_u) (Jacobiano).
        Para densidades necesitamos log|d_u/d_accion|, que será -este valor.

        Si definimos:
            restante_0 = 1
            w_i = u_i * restante_{i-1}
            restante_i = restante_{i-1} * (1 - u_i)

        Entonces el Jacobiano d(w)/d(u) es triangular y:
            det = prod_i restante_{i-1}
            log_det = sum_i log(restante_{i-1})
        """
        batch_size = u.shape[0]
        numero_activos = u.shape[1]

        pesos = torch.zeros_like(u)
        restante = torch.ones((batch_size, 1), device=u.device, dtype=u.dtype)

        # log det del jacobiano daccion/du
        log_det = torch.zeros((batch_size, 1), device=u.device, dtype=u.dtype)

        for i in range(numero_activos):
            ui = u[:, i:i+1]  # (batch, 1)

            # w_i = ui * restante
            wi = ui * restante
            pesos[:, i:i+1] = wi

            # contribución al log_det: log(restante_{i-1})
            # (restante es el restante_{i-1} antes de actualizar)
            restante_clamp = torch.clamp(restante, self.epsilon_numerico, 1.0)
            log_det = log_det + torch.log(restante_clamp)

            # actualizar restante: restante *= (1 - ui)
            restante = restante * (1.0 - ui)

        # seguridad numérica final
        pesos = torch.clamp(pesos, 0.0, 1.0)

        # Por construcción suma <= 1, pero por seguridad:
        suma = torch.sum(pesos, dim=1, keepdim=True)
        exceso = torch.clamp(suma - 1.0, min=0.0)
        if torch.any(exceso > 0):
            # si hubiera exceso por precisión numérica, re-normalizamos suavemente
            pesos = pesos / (suma + self.epsilon_numerico)

        return pesos, log_det


class Critic(nn.Module):
    """
    Critic Q(s,a): aproxima un valor escalar dado estado y acción.

    Se instanciará dos veces en el agente (Critic1 y Critic2).
    """

    def __init__(
        self,
        dimension_estado: int,
        numero_activos: int,
        dimensiones_ocultas: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.dimension_estado = int(dimension_estado)
        self.numero_activos = int(numero_activos)

        dimension_entrada = self.dimension_estado + self.numero_activos
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