# TODO
# 我看以后atomic number改type得了，呵呵
import torch
from typing import List, Dict
from torch import nn
from ..utils import EnvPara


class ElementIndependentLinear(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(
        self,
        input_features: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.linear(input_features)


class ElementTypeLinear(nn.Module):
    """This is linear layer but with different weight to different element type.
    It can only apply to features with shape (n_atoms, n_dim) or (n_edges, n_dim)"""

    def __init__(
        self, elements: List[int], input_dim: int, output_dim: int, bias: bool = True
    ):
        super().__init__()
        atomic_number_to_type = torch.zeros(max(elements) + 1, dtype=torch.long) - 1
        for i, atomic_number in enumerate(elements):
            atomic_number_to_type[atomic_number] = i
        self.register_buffer("atomic_number_to_type", atomic_number_to_type)

        weights = torch.empty(
            (len(elements), input_dim, output_dim),
            dtype=EnvPara.FLOAT_PRECISION,
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

        self.bias = None
        if bias:
            bias = torch.empty(
                (len(elements), output_dim),
                dtype=EnvPara.FLOAT_PRECISION,
            )
            torch.nn.init.xavier_uniform_(bias)
            self.bias = torch.nn.Parameter(bias)

    def get_indices(
        self,
        input_features: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if input_features.shape[0] == len(batch_data["atomic_number"]):
            type_i = self.atomic_number_to_type[
                batch_data["atomic_number"]
            ]  # [n_atoms] (type of atom i)
        elif input_features.shape[0] == len(batch_data["idx_i"]):
            type_i = self.atomic_number_to_type[
                batch_data["atomic_number"][batch_data["idx_i"]]
            ]  # [n_edge] (type of atom i)
        else:
            raise Exception(
                "Input of 'ElementTypeLinear' must be [n_atoms, ...] or [n_edges, ...],"
                f"but got {input_features.shape}"
            )
        return type_i

    def forward(
        self,
        input_features: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        idx = self.get_indices(input_features, batch_data)
        output_features = torch.einsum('bij,bi->bj', self.weights[idx], input_features)
        if self.bias is not None:
            return output_features + self.bias[idx]
        return output_features


class ElementPairLinear(ElementTypeLinear):
    """This is linear layer but with different weight to different element type pair.
    It can only apply to features with shape (n_edges, n_dim)"""

    def __init__(
        self, elements: List[int], input_dim: int, output_dim: int, bias: bool = True
    ):
        super().__init__()
        atomic_number_to_type = torch.zeros(max(elements) + 1, dtype=torch.long) - 1
        for i, atomic_number in enumerate(elements):
            atomic_number_to_type[atomic_number] = i
        self.register_buffer("atomic_number_to_type", atomic_number_to_type)
        self.register_buffer("n_elements", torch.tensor(len(elements)))

        weights = torch.empty(
            (len(elements) ** 2, input_dim, output_dim),
            dtype=EnvPara.FLOAT_PRECISION,
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

        self.bias = None
        if bias:
            bias = torch.empty(
                (len(elements) ** 2, output_dim),
                dtype=EnvPara.FLOAT_PRECISION,
            )
            torch.nn.init.xavier_uniform_(bias)
            self.bias = torch.nn.Parameter(bias)

    def get_indices(
        self,
        input_features: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        assert input_features.shape[0] == len(batch_data["idx_i"]), print(
            "Input of 'ElementPairLinear' must be [n_edges, ...],"
            f"but got {input_features.shape}"
        )
        type_i = self.atomic_number_to_type[
            batch_data["atomic_number"][batch_data["idx_i"]]
        ]  # [n_edge] (type of atom i)
        type_j = self.atomic_number_to_type[
            batch_data["atomic_number"][batch_data["idx_j"]]
        ]  # [n_edge] (type of atom j)
        return type_i * self.n_elements + type_j


class ElementLinear(nn.Module):

    def __init__(
        self, elements: List[int], input_dim: int, output_dim: int, bias: bool = True
    ):
        super().__init__()
        if EnvPara.ELEMENT_MODE == 'none':
            self.linear = ElementIndependentLinear(input_dim, output_dim, bias=bias)
        elif EnvPara.ELEMENT_MODE == 'node_i':
            self.linear = ElementTypeLinear(elements, input_dim, output_dim, bias=bias)
        elif EnvPara.ELEMENT_MODE == 'node_edge':
            self.linear = ElementTypeLinear(elements, input_dim, output_dim, bias=bias)

    def forward(
        self,
        input_features: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.linear(input_features, batch_data)


class MLP(nn.Module):

    def __init__(
        self,
        n_hidden: List[int],
        activate_fn: nn.Module = nn.SiLU(),
        bias: bool = True,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.activate_fn = activate_fn
        self.mlp = nn.ModuleList(
            [
                nn.Linear(n_in, n_out, bias=bias)
                for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:])
            ]
        )

    def forward(
        self,
        input_features: torch.Tensor,
    ) -> torch.Tensor:
        output_features = input_features
        for layer in self.mlp[:-1]:
            output_features = self.activate_fn(layer(output_features))
        return self.mlp[-1](output_features)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.n_hidden}"


class ElementMLP(nn.Module):

    def __init__(
        self,
        elements: List[int],
        n_hidden: List[int],
        activate_fn: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.activate_fn = activate_fn
        self.mlp = nn.ModuleList(
            [
                ElementLinear(elements, n_in, n_out)
                for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:])
            ]
        )

    def forward(
        self,
        input_features: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        output_features = input_features
        for layer in self.mlp[:-1]:
            output_features = self.activate_fn(layer(output_features, batch_data))
        return self.mlp[-1](output_features, batch_data)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.n_hidden}"
