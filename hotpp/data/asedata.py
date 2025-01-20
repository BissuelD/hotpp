from git import Union
from .utils import AtomsDataset, register_dataset
from typing import List, Optional
from ase import Atoms
from ase.io import read
from ase.db import connect


@register_dataset("ase")
class ASEData(AtomsDataset):

    def __init__(
        self,
        frames: Union[List[Atoms], str, None] = None,
        indices: Optional[List[int]] = None,
        properties: Optional[List[str]] = ['energy', 'forces'],
        spin: bool = False,
        cutoff: float = 4.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(indices=indices, cutoff=cutoff)
        if frames is None:
            self.frames = []
        elif isinstance(frames, str):
            self.frames = read(frames, index=":")
        else:
            self.frames = frames
        self.properties = properties
        self.spin = spin

    def __len__(self):
        if self.indices is None:
            return len(self.frames)
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        data = self.atoms_to_data(
            self.frames[idx],
            properties=self.properties,
            cutoff=self.cutoff,
            spin=self.spin,
        )
        return data

    def extend(self, frames: Union[List[Atoms], str]):
        if isinstance(frames, str):
            frames = read(frames, index=":")
        self.frames.extend(frames)


@register_dataset("ase-db")
class ASEDBData(AtomsDataset):

    def __init__(
        self,
        datapath: Optional[str] = None,
        indices: Optional[List[int]] = None,
        properties: Optional[List[str]] = ['energy', 'forces'],
        spin: bool = False,
        cutoff: float = 4.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(indices=indices, cutoff=cutoff)
        self.conn = connect(datapath, use_lock_file=False)
        self.properties = properties
        self.spin = spin

    def __len__(self):
        if self.indices is None:
            return self.conn.count()
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        row = self.conn.get(idx + 1)
        atoms = Atoms(
            numbers=row['numbers'],
            cell=row['cell'],
            positions=row['positions'],
            pbc=row['pbc'],
            info=row.data,
        )
        data = self.atoms_to_data(
            atoms, properties=self.properties, cutoff=self.cutoff, spin=self.spin
        )
        return data
