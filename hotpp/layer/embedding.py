import torch
from torch import nn
from typing import List, Optional, Dict
from .base import CutoffLayer, EmbeddingLayer
from ..utils import find_distances, _scatter_add, EnvPara

__all__ = [
    "AtomicOneHot",
    "AtomicNumber",
    "AtomicEmbedding",
]


class AtomicOneHot(EmbeddingLayer):
    def __init__(self, elements: List[int], trainable: bool = False) -> None:
        super().__init__(elements)
        max_atomic_number = max(elements)
        n_elements = len(elements)
        weights = torch.zeros(
            max_atomic_number + 1,
            n_elements,
            dtype=EnvPara.FLOAT_PRECISION,
        )
        for idx, z in enumerate(elements):
            weights[z, idx] = 1.0
        self.z_weights = nn.Embedding(max_atomic_number + 1, n_elements)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False
        self.n_channel = n_elements

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])


class AtomicNumber(EmbeddingLayer):
    def __init__(self, elements: List[int], trainable: bool = False) -> None:
        super().__init__(elements)
        max_atomic_number = max(elements)
        weights = torch.arange(max_atomic_number + 1)[:, None].float()
        self.z_weights = nn.Embedding(max_atomic_number + 1, 1)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False
        self.n_channel = 1

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])


class AtomicEmbedding(EmbeddingLayer):
    def __init__(
        self,
        elements: List[int],
        n_channel: int,
    ) -> None:
        super().__init__(elements)
        max_atomic_number = int(max(elements))
        self.z_weights = nn.Embedding(max_atomic_number + 1, n_channel)
        self.n_channel = n_channel

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])
