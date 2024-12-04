import torch
from torch import nn
from typing import List, Dict, Callable, Any, Optional, Union
from .equivalent import TensorLinear, ElementTensorLinear
from .activate import TensorActivateDict
from ..utils import EnvPara


__all__ = ["ReadoutLayer"]


class ReadoutMLP(nn.Module):
    def __init__(
        self,
        n_dim: int,
        way: int,
        activate_fn: str = "silu",
    ) -> None:
        super().__init__()
        self.activate_fn = TensorActivateDict[activate_fn](n_dim)
        self.layer1 = TensorLinear(n_dim, n_dim, bias=(way == 0))
        self.layer2 = TensorLinear(n_dim, 1, bias=(way == 0))

    def forward(
        self,
        input_tensor: torch.Tensor,  # [n_batch, n_channel, n_dim, n_dim, ...]
        batch_data: Dict[str, torch.Tensor],  # [n_batch, n_channel]
    ):
        return self.layer2(self.activate_fn(self.layer1(input_tensor)))


class ElementReadoutMLP(nn.Module):
    def __init__(
        self,
        n_dim: int,
        way: int,
        activate_fn: str = "silu",
    ) -> None:
        super().__init__()
        self.activate_fn = TensorActivateDict[activate_fn](n_dim)
        elements = EnvPara.ELEMENTS
        self.layer1 = ElementTensorLinear(elements, n_dim, n_dim, bias=(way == 0))
        self.layer2 = ElementTensorLinear(elements, n_dim, 1, bias=(way == 0))

    def forward(
        self,
        input_tensor: torch.Tensor,  # [n_batch, n_channel, n_dim, n_dim, ...]
        batch_data: Dict[str, torch.Tensor],  # [n_batch, n_channel]
    ):
        return self.layer2(
            self.activate_fn(self.layer1(input_tensor, batch_data)), batch_data
        )


class ReadoutLayer(nn.Module):

    def __init__(
        self,
        n_dim: int,
        target_way: Dict[str, int] = {"site_energy": 0},
        activate_fn: str = "silu",
    ) -> None:
        super().__init__()
        self.target_way = target_way
        self.layer_dict = nn.ModuleDict()
        for prop, way in target_way.items():
            if EnvPara.ELEMENT_MODE == 'none':
                self.layer_dict[prop] = ReadoutMLP(
                    n_dim=n_dim, way=way, activate_fn=activate_fn
                )
            else:
                self.layer_dict[prop] = ElementReadoutMLP(
                    n_dim=n_dim, way=way, activate_fn=activate_fn
                )

    def forward(
        self,
        input_tensors: Dict[int, torch.Tensor],
        batch_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[str, torch.Tensor], {})
        for prop, readout_layer in self.layer_dict.items():
            way = self.target_way[prop]
            output_tensors[prop] = readout_layer(
                input_tensors[way], batch_data
            ).squeeze(1)
        # delete channel dim
        return output_tensors
