import numpy as np
from .utils import AtomsDataset, register_dataset
from typing import List, Optional
from ase import Atoms


@register_dataset("rmd17")
class RevisedMD17(AtomsDataset):

    def __init__(
        self,
        datapath: str,
        indices: Optional[List[int]] = None,
        cutoff: float = 4.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(indices=indices, cutoff=cutoff)
        raw_data = np.load(datapath)
        self.symbols = raw_data["nuclear_charges"]
        self.energy = raw_data["energies"]
        self.forces = raw_data["forces"]
        self.coords = raw_data["coords"]
        self.cutoff = cutoff

    def __len__(self):
        if self.indices is None:
            return len(self.energy)
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        atoms = Atoms(
            symbols=self.symbols,
            positions=self.coords[idx],
            info={"energy": self.energy[idx], "forces": self.forces[idx]},
        )
        data = self.atoms_to_data(atoms, cutoff=self.cutoff)
        return data
