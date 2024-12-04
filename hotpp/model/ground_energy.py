from typing import Callable, List, Dict, Optional
import torch
from torch import nn
from .base import AtomicModule
from ..layer.embedding import AtomicOneHot
from ..utils import _scatter_add, EnvPara


class GroundEnergy(AtomicModule):

    def __init__(
        self,
        atomic_number: List[int],
        ground_energy: List[float],
    ) -> None:
        super().__init__()
        self.n_channel = 1
        self.target_way = {"site_energy": 0}
        self.one_hot = AtomicOneHot(atomic_number, trainable=False)
        self.register_buffer(
            "ground_energy", torch.tensor(ground_energy, dtype=EnvPara.FLOAT_PRECISION)
        )

    def calculate(
        self,
        batch_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_data['site_energy'] = self.one_hot(batch_data) @ self.ground_energy
        return batch_data
