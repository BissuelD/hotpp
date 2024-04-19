from typing import Callable, List, Dict, Optional
import torch
from torch import nn
from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import MultiBodyLayer, SimpleTensorAggregateLayer, NonLinearLayer
from ..utils import find_distances


class MiaoMiaoBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_n_body     : int,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 ) -> None:
        super().__init__()
        self.tensor_aggregate = SimpleTensorAggregateLayer(radial_fn=radial_fn,
                                                           n_channel=input_dim,
                                                           max_in_way=max_in_way,
                                                           max_out_way=max_out_way,
                                                           max_r_way=max_r_way,
                                                           norm_factor=norm_factor,)
        # input for SelfInteractionLayer and NonLinearLayer is the output of TensorAggregateLayer
        # so the max_in_way should equal to max_out_way of TensorAggregateLayer
        self.self_interact = MultiBodyLayer(max_n_body=max_n_body,
                                            input_dim=input_dim, 
                                            output_dim=output_dim,
                                            max_way=max_out_way)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        input_tensors = self.propagate(input_tensors, batch_data)
        return input_tensors

    # TODO: sparse version
    def message_and_aggregate(self,
                              input_tensors : Dict[int, torch.Tensor],
                              batch_data    : Dict[str, torch.Tensor],
                              ) -> Dict[int, torch.Tensor]:
        output_tensors =  self.tensor_aggregate(input_tensors=input_tensors,
                                                batch_data=batch_data)
        # resnet
        for r_way in input_tensors.keys():
            output_tensors[r_way] += input_tensors[r_way]
        return output_tensors

    def update(self,
               input_tensors : Dict[int, torch.Tensor],
               ) -> Dict[int, torch.Tensor]:
        output_tensors = self.self_interact(input_tensors)
        output_tensors = self.non_linear(output_tensors)
        # resnet
        for r_way in input_tensors.keys():
            output_tensors[r_way] += input_tensors[r_way]
        return output_tensors

    def propagate(self,
                  input_tensors : Dict[int, torch.Tensor],
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[int, torch.Tensor]:
        output_tensors = self.message_and_aggregate(input_tensors, batch_data)
        output_tensors = self.update(output_tensors)
        return output_tensors


class MiaoMiaoNet(AtomicModule):

    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 n_layers        : int,
                 max_n_body      : List[int],
                 max_r_way       : List[int],
                 max_out_way     : List[int],
                 output_dim      : List[int],
                 activate_fn     : str="silu",
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
        self.embedding_layer = embedding_layer

        max_in_way = [0] + max_out_way[1:]
        hidden_nodes = [embedding_layer.n_channel] + output_dim
        self.son_equivalent_layers = nn.ModuleList([
            MiaoMiaoBlock(activate_fn=activate_fn,
                          radial_fn=radial_fn.replicate(),
                          # Use factory method, so the radial_fn in each layer are different
                          max_n_body=max_n_body[i],
                          max_r_way=max_r_way[i],
                          max_in_way=max_in_way[i],
                          max_out_way=max_out_way[i],
                          input_dim=hidden_nodes[i],
                          output_dim=hidden_nodes[i + 1],
                          norm_factor=norm_factor,
                          ) for i in range(n_layers)])
        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                          target_way=target_way,
                                          activate_fn=activate_fn,
                                          bilinear=bilinear,
                                          e_dim=embedding_layer.n_channel)

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        find_distances(batch_data)
        emb = self.embedding_layer(batch_data=batch_data)
        output_tensors = {0: emb}
        for son_equivalent_layer in self.son_equivalent_layers:
            output_tensors = son_equivalent_layer(output_tensors, batch_data)
        output_tensors = self.readout_layer(output_tensors, emb)
        if 'site_energy' in output_tensors:
            output_tensors['site_energy'] = output_tensors['site_energy'] * self.std + self.mean
        if 'direct_forces' in output_tensors:
            output_tensors['direct_forces'] = output_tensors['direct_forces'] * self.std
        return output_tensors
