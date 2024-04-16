from typing import Callable, List, Dict, Optional
import torch
from torch import nn
from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import TensorAggregateLayer, SimpleTensorAggregateLayer, SelfInteractionLayer, NonLinearLayer, TensorLinear
from ..utils import find_distances, _aggregate_new


# TODO
# Test allowing higher order such as (2, 2) -> 4, (4, 2) -> 2 ?
class MultiBodyLayer(nn.Module):

    def __init__(self,
                 input_dim      : int,
                 output_dim     : int,
                 max_n_body     : int=3,
                 max_in_way     : int=2,
                 ) -> None:
        super().__init__()
        self.max_n_body = max_n_body
        self.max_in_way = max_in_way
        n_body_tensors = [[1] *  (max_in_way + 1)]
        for n in range(max_n_body - 1):
            n_body_tensors.append([0] *  (max_in_way + 1))
            for way1 in range(max_in_way + 1):
                for way2 in range(way1, max_in_way + 1):
                    for way3 in range(abs(way2 - way1), min(max_in_way, way1 + way2) + 1, 2):
                        n_body_tensors[n + 1][way3] += n_body_tensors[n][way1]

        self.linear_list = nn.ModuleList([
            TensorLinear(input_dim * sum([n_body_tensors[n][way] for n in range(max_n_body)]), 
                         output_dim, 
                         bias=(way==0)) 
            for way in range(max_in_way + 1)])

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        n_body_tensors = {0: {way: [input_tensors[way]] for way in input_tensors}}
        for n in range(self.max_n_body):
            n_body_tensors[n + 1] = {way: [] for way in range(self.max_in_way + 1)}
            for way1 in range(self.max_in_way + 1):
                for way2 in range(self.max_in_way + 1):
                    for way3 in range(abs(way2 - way1), min(self.max_in_way, way1 + way2), 2):
                        for tensor in n_body_tensors[n][way1]:
                            n_body_tensors[n + 1][way3].append(_aggregate_new(tensor, input_tensors[way2], way1, way2, way3))
        for way, linear in enumerate(self.linear_list):
            tensor = torch.cat([n_body_tensors[n][way] for n in range(self.max_n_body)], dim=1)  # nb, nc*n, nd, nd, ...
            output_tensors[way] = linear(tensor)
        return output_tensors


class MessagePassingBlock(nn.Module):
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
                                            max_in_way=max_out_way)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_in_way=max_out_way,
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
            MessagePassingBlock(activate_fn=activate_fn,
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
