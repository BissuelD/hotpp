import torch
from torch import nn
from typing import List, Dict, Optional
from ..utils import _scatter_add, find_distances, add_scaling


class AtomicModule(nn.Module):

    def forward(self,
                batch_data   : Dict[str, torch.Tensor],
                properties   : Optional[List[str]]=None,
                create_graph : bool=True,
                ) -> Dict[str, torch.Tensor]:
        # Use properties=None instead of properties=['energy'] because
        # Mutable default parameters are not supported because
        # Python binds them to the function and they persist across function calls.
        if properties is None:
            properties = ['energy']
        required_derivatives = torch.jit.annotate(List[str], [])
        if 'forces' in properties:
            required_derivatives.append('coordinate')
            batch_data['coordinate'].requires_grad_()
        if 'virial' in properties or 'stress' in properties:
            required_derivatives.append('scaling')
            batch_data['scaling'].requires_grad_()
            add_scaling(batch_data)
        if 'spin_torques' in properties:
            required_derivatives.append('spin')
            batch_data['spin'].requires_grad_()
        output_tensors = self.calculate(batch_data)
        #######################################
        if 'dipole' in output_tensors:
            batch_data['dipole_p'] = _scatter_add(output_tensors['dipole'], batch_data['batch'])
        if 'polar_diag' in output_tensors:
            polar_diag = _scatter_add(output_tensors['polar_diag'], batch_data['batch'])
            polar_off_diagonal = _scatter_add(output_tensors['polar_off_diagonal'], batch_data['batch'])
            polar = polar_off_diagonal + polar_off_diagonal.transpose(1, 2)
            polar[:, 0, 0] += polar_diag
            polar[:, 1, 1] += polar_diag
            polar[:, 2, 2] += polar_diag
            batch_data['polarizability_p'] = polar 
        if 'site_energy' in output_tensors:
            site_energy = output_tensors['site_energy']
        #######################################
        # for torch.jit.script
        else:
            site_energy = batch_data['n_atoms']
        if 'direct_forces' in output_tensors:
            batch_data['direct_forces_p'] = output_tensors['direct_forces']
        if 'direct_stress' in output_tensors:
            batch_data['direct_stress_p'] = _scatter_add(output_tensors['direct_stress'], batch_data['batch'])
        if ('site_energy' in properties) or ('energies' in properties):
            batch_data['site_energy_p'] = site_energy
        if 'energy' in properties:
            batch_data['energy_p'] = _scatter_add(site_energy, batch_data['batch'])
        if len(required_derivatives) > 0:
            grads = torch.autograd.grad([site_energy.sum()],
                                        [batch_data[prop] for prop in required_derivatives],
                                        create_graph=create_graph)
        #######################################
        # for torch.jit.script 
        else:
            grads = torch.jit.annotate(List[Optional[torch.Tensor]], [])
        #######################################
        if 'forces' in properties:
            #######################################
            # for torch.jit.script 
            dE_dr = grads[required_derivatives.index('coordinate')]
            if dE_dr is not None:
                batch_data['forces_p'] = -dE_dr
            #######################################
        if 'virial' in properties or 'stress' in properties:
            #######################################
            # for torch.jit.script 
            dE_dl = grads[required_derivatives.index('scaling')]
            if dE_dl is not None:
                batch_data['virial_p'] = -dE_dl
            #######################################
        if 'spin_torques' in properties:
            dE_dS = grads[required_derivatives.index('spin')]
            if dE_dS is not None:
                batch_data['spin_torques_p'] = -dE_dS
        return batch_data

    def calculate(self):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'calculate'!")


# Only support energy model now!
class MultiAtomicModule(AtomicModule):
    def __init__(self, models: Dict[str, AtomicModule]) -> None:
        super().__init__()
        self.target_way = None
        for name, model in models.items():
            if self.target_way is None:
                self.target_way = model.target_way
            assert self.target_way == model.target_way, f"{name} target way {model.target_way} is different from {self.target_way}"
        self.models = nn.ModuleDict(models)

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        output_tensors = {}
        n_atoms = len(batch_data['atomic_number'])
        n_dim = batch_data['coordinate'].shape[1]
        device = batch_data['coordinate'].device
        if 'site_energy' in self.target_way:
            output_tensors['site_energy'] = torch.zeros((n_atoms), dtype=batch_data['coordinate'].dtype, device=device)
        if 'direct_forces' in self.target_way:
            output_tensors['direct_forces'] = torch.zeros((n_atoms, n_dim), dtype=batch_data['coordinate'].dtype, device=device)
        if 'direct_stress' in self.target_way:
            output_tensors['direct_stress'] = torch.zeros((n_atoms, n_dim, n_dim), dtype=batch_data['coordinate'].dtype, device=device)
        if 'dipole' in self.target_way:
            output_tensors['dipole'] = torch.zeros((n_atoms, n_dim), dtype=batch_data['coordinate'].dtype, device=device)
        if 'polar_diag' in self.target_way:
            output_tensors['polar_diag'] = torch.zeros((n_atoms), dtype=batch_data['coordinate'].dtype, device=device)
            output_tensors['polar_off_diagonal'] = torch.zeros((n_atoms, n_dim, n_dim), dtype=batch_data['coordinate'].dtype, device=device)

        for name, model in self.models.items():
            for target in self.target_way:
                output_tensors[target] += model.calculate(batch_data)[target]
        return output_tensors
