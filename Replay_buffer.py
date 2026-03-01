from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch

@dataclass
class LoteTransiciones:
    estado: torch.Tensor
    accion: torch.Tensor
    recompensa: torch.Tensor
    siguiente_estado: torch.Tensor
    terminado: torch.Tensor

class BufferRepeticion:
    """
    Buffer circular de transiciones para entrenamiento off-policy.

    Guarda:
      - estado:             (dimension_estado,)
      - accion:             (numero_activos,)
      - recompensa:         (1,)
      - siguiente_estado:   (dimension_estado,)
      - terminado:          (1,)  1.0 si episodio terminado, si no 0.0
    """

    def __init__(
        self,
        capacidad_maxima: int,
        dimension_estado: int,
        numero_activos: int,
        dispositivo: torch.device,
        semilla: Optional[int] = None,
    ) -> None:
        self.capacidad_maxima = int(capacidad_maxima)
        self.dimension_estado = int(dimension_estado)
        self.numero_activos = int(numero_activos)
        self.dispositivo = dispositivo

        if semilla is not None:
            np.random.seed(int(semilla))

        # Memoria pre-asignada (más eficiente que listas)
        self._estados = np.zeros((self.capacidad_maxima, self.dimension_estado), dtype=np.float32)
        self._acciones = np.zeros((self.capacidad_maxima, self.numero_activos), dtype=np.float32)
        self._recompensas = np.zeros((self.capacidad_maxima, 1), dtype=np.float32)
        self._siguientes_estados = np.zeros((self.capacidad_maxima, self.dimension_estado), dtype=np.float32)
        self._terminados = np.zeros((self.capacidad_maxima, 1), dtype=np.float32)

        # Puntero circular y tamaño actual
        self._indice = 0
        self._tamano = 0

    def __len__(self) -> int:
        return self._tamano

    def esta_listo(self, tamano_batch: int) -> bool:
        """
        Indica si hay suficientes transiciones para muestrear un batch.
        """
        return self._tamano >= int(tamano_batch)

    def guardar_transicion(
        self,
        estado: np.ndarray,
        accion: np.ndarray,
        recompensa: float,
        siguiente_estado: np.ndarray,
        terminado: bool,
    ) -> None:
        """
        Inserta una transición en el buffer (sobrescribe en modo circular).
        """
        i = self._indice

        # Asegurar dtype/shape
        self._estados[i] = np.asarray(estado, dtype=np.float32).reshape(-1)
        self._acciones[i] = np.asarray(accion, dtype=np.float32).reshape(-1)
        self._recompensas[i] = np.asarray([recompensa], dtype=np.float32)
        self._siguientes_estados[i] = np.asarray(siguiente_estado, dtype=np.float32).reshape(-1)
        self._terminados[i] = np.asarray([1.0 if terminado else 0.0], dtype=np.float32)

        # actualizar puntero circular
        self._indice = (self._indice + 1) % self.capacidad_maxima
        self._tamano = min(self._tamano + 1, self.capacidad_maxima)

    def muestrear_lote(self, tamano_batch: int) -> LoteTransiciones:
        """
        Muestreo uniforme aleatorio de transiciones.
        Devuelve tensores ya movidos al dispositivo.
        """
        tamano_batch = int(tamano_batch)
        if self._tamano < tamano_batch:
            raise ValueError(
                f"No hay suficientes transiciones para muestrear: "
                f"tamano={self._tamano}, requerido={tamano_batch}"
            )

        indices = np.random.randint(0, self._tamano, size=tamano_batch)

        estado = torch.as_tensor(self._estados[indices], device=self.dispositivo, dtype=torch.float32)
        accion = torch.as_tensor(self._acciones[indices], device=self.dispositivo, dtype=torch.float32)
        recompensa = torch.as_tensor(self._recompensas[indices], device=self.dispositivo, dtype=torch.float32)
        siguiente_estado = torch.as_tensor(self._siguientes_estados[indices], device=self.dispositivo, dtype=torch.float32)
        terminado = torch.as_tensor(self._terminados[indices], device=self.dispositivo, dtype=torch.float32)

        return LoteTransiciones(
            estado=estado,
            accion=accion,
            recompensa=recompensa,
            siguiente_estado=siguiente_estado,
            terminado=terminado,
        )