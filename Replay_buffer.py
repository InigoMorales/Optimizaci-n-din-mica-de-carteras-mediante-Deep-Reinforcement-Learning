from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

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
    Guarda: estado, accion, recompensa, siguiente_estado y terminado(1 si ha terminado)
    dimension_accion = numero_activos_riesgo + 1
    """

    def __init__(
        self,
        capacidad_maxima: int,
        dimension_estado: int,
        dimension_accion: int,
        dispositivo: torch.device,
        semilla: Optional[int] = None,
    ) -> None:
        self.capacidad_maxima = int(capacidad_maxima)
        self.dimension_estado = int(dimension_estado)
        self.dimension_accion = int(dimension_accion)
        self.dispositivo = dispositivo

        if semilla is not None:
            np.random.seed(int(semilla))

        # Memoria preasignada
        self._estados = np.zeros((self.capacidad_maxima, self.dimension_estado), dtype=np.float32)
        self._acciones = np.zeros((self.capacidad_maxima, self.dimension_accion), dtype=np.float32)
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
        terminado: bool
    ) -> None:
        """
        Inserta una transición en el buffer (sobrescribe en modo circular).
        """
        i = self._indice

        self._estados[i] = np.asarray(estado, dtype=np.float32).reshape(-1)
        self._acciones[i] = np.asarray(accion, dtype=np.float32).reshape(-1)
        self._recompensas[i] = np.asarray([recompensa], dtype=np.float32)
        self._siguientes_estados[i] = np.asarray(siguiente_estado, dtype=np.float32).reshape(-1)
        self._terminados[i] = np.asarray([1.0 if terminado else 0.0], dtype=np.float32)

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

        return LoteTransiciones(estado=estado, accion=accion, recompensa=recompensa,
            siguiente_estado=siguiente_estado, terminado=terminado)