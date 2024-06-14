import torch
from ..model.base import AtomicModule
from ase.data import chemical_symbols, atomic_numbers
from typing import Dict, List
from ..utils import _scatter_add


class FreezeAtomicModule(AtomicModule):
    def __init__(self, model: AtomicModule):
        super().__init__()
        self.model = model
        self.register_buffer("all_elements", model.all_elements)
        self.register_buffer("cutoff", model.cutoff)
        self.register_buffer("mean", model.mean)
        self.register_buffer("std", model.std)


class EnergyModel(FreezeAtomicModule):

    def forward(self,
                batch_data   : Dict[str, torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        output_tensors = self.model.calculate(batch_data)
        site_energy = output_tensors['site_energy']
        batch_data['energy_p'] = _scatter_add(site_energy, batch_data['batch'])
        return batch_data


class EnergyForcesModel(FreezeAtomicModule):

    def forward(self,
                batch_data   : Dict[str, torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        batch_data['coordinate'].requires_grad_()
        output_tensors = self.model.calculate(batch_data)
        site_energy = output_tensors['site_energy']
        batch_data['energy_p'] = _scatter_add(site_energy, batch_data['batch'])
        grads = torch.autograd.grad([site_energy.sum()],
                                    [batch_data['coordinate']],
                                    )
        batch_data['forces_p'] = -grads[0]
        return batch_data


def main(*args, model="model.pt", device="cpu", output="infer.pt", 
         symbols=None, properties=None, **kwargs):
    model = torch.load(model, map_location=torch.device(device))
    if properties is None:
        properties = ['energy']
    if 'energy' in properties:
        model = EnergyModel(model)
    if 'forces' in properties:
        model = EnergyForcesModel(model)
    model.eval()
    # change embedding layer
    if symbols is not None:
        all_elements = [atomic_numbers[s] for s in symbols]
    else:
        all_elements = model.all_elements.cpu().numpy()
    for params in model.parameters():
        params.requires_grad=False
    ase_infer = torch.jit.script(model)
    ase_infer.save(f'ase-{output}')
    # lammps symbol index is different
    for name, value in model.named_parameters():
        if "embedding_layer" in name:
            new_weight = torch.zeros(len(all_elements), value.shape[1])
            for i, n in enumerate(all_elements):
                new_weight[i] = value[n].data
            value.data = new_weight
    lammps_infer = torch.jit.script(model)
    print(lammps_infer)
    lammps_infer = torch.jit.freeze(lammps_infer)
    lammps_infer.save(f'lammps-{output}')


if __name__ == "__main__":
    main()
