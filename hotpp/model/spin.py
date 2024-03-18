from typing import Callable, List, Dict, Optional
import torch
from torch import nn
from .base import AtomicModule
from ..layer import TensorLinear, RadialLayer, ReadoutLayer, SelfInteractionLayer
from ..layer.activate import TensorSilu
from ..utils import expand_para, find_distances, find_moment, _aggregate, _scatter_add

# TODO: std and mean should be move to data prepare?


class SpinAggregateLayer(nn.Module):

    def __init__(self,
                 radial_fn      : RadialLayer,
                 n_channel      : int,
                 max_r_way      : int=2,
                 norm_factor    : float=1.,
                 ) -> None:
        super().__init__()
        # get all possible "i, r, o" combinations
        self.rbf_mixing_dict = nn.ModuleDict()
        self.max_r_way = max_r_way
        self.inout_combinations = {r_way: [] for r_way in range(max_r_way + 1)}
        for r_way in range(max_r_way + 1):
            self.rbf_mixing_dict[str(r_way)] = nn.Linear(radial_fn.n_features, n_channel, bias=False)
            for in_way in range(3):
                for z_way in range(min(2, r_way) + 1):
                    out_way = in_way + r_way - 2 * z_way
                    if out_way == 1:
                        self.inout_combinations[r_way].append((in_way, out_way))
        self.radial_fn = radial_fn
        self.register_buffer("norm_factor", torch.tensor(norm_factor).float())

    def forward(self,
                input_tensor  : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        # These 3 rows are required by torch script
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']

        n_atoms = batch_data['atomic_number'].shape[0]
        _, dij, uij = find_distances(batch_data)
        rbf_ij = self.radial_fn(dij)    # [n_edge, n_rbf]

        input_tensors = {
            0: torch.sum(input_tensor[idx_i] * input_tensor[idx_j], dim=-1), # ne, nc
            1: torch.cross(input_tensor[idx_i], input_tensor[idx_j], dim=-1), 
            2: input_tensor[idx_i].unsqueeze(-1) * input_tensor[idx_j].unsqueeze(-2)
        }
        results = torch.zeros_like(input_tensor)
        for r_way, rbf_mixing in self.rbf_mixing_dict.items():
            r_way = int(r_way)
            fn = rbf_mixing(rbf_ij) # [n_edge, n_channel]
            moment_tensor = find_moment(batch_data, r_way)  # [n_edge, n_dim ^ r_way]
            for in_way, out_way in self.inout_combinations[r_way]:
                output_tensor = _aggregate(moment_tensor, fn, input_tensors[in_way], in_way, r_way, out_way) # [n_edge, n_channel, n_dim ^ out_way]
                output_tensor = _scatter_add(output_tensor, idx_i, dim_size=n_atoms) / self.norm_factor
                results += output_tensor
        return results


class SpinEquivalentLayer(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='jilu',
                 mode           : str='normal',
                 ) -> None:
        super().__init__()
        self.tensor_aggregate = SpinAggregateLayer(radial_fn=radial_fn,
                                                            n_channel=input_dim,
                                                            max_r_way=max_r_way,
                                                            norm_factor=norm_factor,)
        # input for SelfInteractionLayer and NonLinearLayer is the output of TensorAggregateLayer
        # so the max_in_way should equal to max_out_way of TensorAggregateLayer
        self.self_interact = TensorLinear(input_dim=input_dim,
                                          output_dim=output_dim, 
                                          bias=False)
        self.non_linear = TensorSilu(output_dim)

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        input_tensors = self.propagate(input_tensors, batch_data)
        return input_tensors

    # TODO: sparse version
    def message_and_aggregate(self,
                              input_tensor : Dict[int, torch.Tensor],
                              batch_data    : Dict[str, torch.Tensor],
                              ) -> Dict[int, torch.Tensor]:
        output_tensor =  self.tensor_aggregate(input_tensor=input_tensor,
                                                batch_data=batch_data) + input_tensor
        return output_tensor

    def update(self,
               input_tensor : Dict[int, torch.Tensor],
               batch_data    : Dict[str, torch.Tensor],
               ) -> Dict[int, torch.Tensor]:
        output_tensor = self.self_interact(input_tensor)
        output_tensor = self.non_linear(output_tensor)
        output_tensor += input_tensor
        return output_tensor

    def propagate(self,
                  input_tensor,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[int, torch.Tensor]:
        output_tensor = self.message_and_aggregate(input_tensor, batch_data)
        output_tensor = self.update(output_tensor, batch_data)
        return output_tensor

    
class SpinMiaoNet(AtomicModule):
    """
    Miao nei ga
    duo xi da miao nei
    """
    def __init__(self,
                 radial_fn       : RadialLayer,
                 n_layers        : int,
                 max_r_way       : int or List,
                 output_dim      : int or List,
                 activate_fn     : str="jilu",
                 target_way      : Dict[str, int]={"site_energy": 0},
                 mean            : float=0.,
                 std             : float=1.,
                 norm_factor     : float=1.,
                 mode            : str='normal',
                 bilinear        : bool=False,
                 ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float())
        self.register_buffer("std", torch.tensor(std).float())
        self.register_buffer("norm_factor", torch.tensor(norm_factor).float())
        self.spin_embedding_layer = TensorLinear(1, output_dim, False)
        max_r_way = expand_para(max_r_way, n_layers)
        hidden_nodes = [output_dim] + expand_para(output_dim, n_layers)
        self.son_equivalent_layers = nn.ModuleList([
            SpinEquivalentLayer(activate_fn=activate_fn,
                               radial_fn=radial_fn.replicate(),
                               # Use factory method, so the radial_fn in each layer are different
                               max_r_way=max_r_way[i],
                               input_dim=hidden_nodes[i],
                               output_dim=hidden_nodes[i + 1],
                               norm_factor=norm_factor,
                               mode=mode) for i in range(n_layers)])
        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                          target_way=target_way,
                                          activate_fn=activate_fn,
                                          bilinear=bilinear,
                                          e_dim=output_dim)

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        find_distances(batch_data)
        output_tensor = self.spin_embedding_layer(batch_data['spin'].unsqueeze(1))
        for son_equivalent_layer in self.son_equivalent_layers:
            output_tensor = son_equivalent_layer(output_tensor, batch_data)
            
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']
        output_tensors = {0: _scatter_add(
            torch.sum(output_tensor[idx_i] * output_tensor[idx_j], dim=-1),
            idx_i, dim_size=batch_data['atomic_number'].shape[0])}
        output_tensors = self.readout_layer(output_tensors, None)
        if 'site_energy' in output_tensors:
            output_tensors['site_energy'] = output_tensors['site_energy'] * self.std + self.mean
        if 'direct_forces' in output_tensors:
            output_tensors['direct_forces'] = output_tensors['direct_forces'] * self.std
        return output_tensors
