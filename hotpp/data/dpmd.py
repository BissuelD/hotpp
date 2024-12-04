from .base import AtomsDataset
from typing import List, Optional
import glob, os
import numpy as np
from ase import Atoms


def expand_path(path_list: List[str]):
    all_path = []
    for path in path_list:
        for p in glob.glob(path):
            if p not in all_path:
                all_path.append(p)
    return all_path


def read_dpdata(path: str,
                properties: Optional[List[str]]=['energy', 'forces']):
    frames = []
    with open(os.path.join(path, "type_map.raw")) as f:
        type_map_list = f.read().splitlines()
    with open(os.path.join(path, "type.raw")) as f:
        type_list = f.read().splitlines()
    symbols = [type_map_list[int(i)] for i in type_list]
    for dataset in glob.glob(os.path.join(path, "set.*")):
        # 不是，这什么人里面弄个real atom啊我请问了？
        if os.path.exists(os.path.join(dataset, "real_atom_types.npy")):
            real_type_list = np.load(os.path.join(dataset, "real_atom_types.npy"))
        else:
            real_type_list = None
        box = np.load(os.path.join(dataset, "box.npy"))
        coord = np.load(os.path.join(dataset, "coord.npy"))
        atoms_info = {}
        for prop in properties:
            atoms_info[prop] = np.load(os.path.join(dataset, f"{prop}.npy"))

        for i in range(len(box)):
            if real_type_list is not None:
                real_symbols = [type_map_list[t] for t in real_type_list[i]]
            else:
                real_symbols = symbols
            atoms = Atoms(
                pbc=True,
                symbols=real_symbols,
                positions=coord[i].reshape(-1, 3),
                cell=box[i].reshape(3, 3))
            info = {}
            if "energy" in properties:
                info["energy"] = atoms_info["energy"][i]
            if "forces" in properties:
                info["forces"] = atoms_info["forces"][i].reshape(-1, 3)
            if "virial" in properties:
                info["virial"] = atoms_info["virial"][i].reshape(3, 3)
            if "polarizability" in properties:
                info["polarizability"] = atoms_info["polarizability"][i].reshape(3, 3)
            atoms.info = info
            frames.append(atoms)
    return frames


class DeePMDData(AtomsDataset):

    def __init__(self,
                 path_list  : Optional[List[str]]=None,
                 indices    : Optional[List[int]]=None,
                 properties : Optional[List[str]]=['energy', 'forces'],
                 spin       : bool=False,
                 cutoff     : float=4.0,
                 ) -> None:
        super().__init__(indices=indices, cutoff=cutoff)
        frames = []
        for path in expand_path(path_list):
            frames.extend(read_dpdata(path, properties))
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
        data = self.atoms_to_data(self.frames[idx],
                                  properties=self.properties,
                                  cutoff=self.cutoff,
                                  spin=self.spin)
        return data

    def extend(self,
               frames: Optional[List[Atoms]]=None,
               path_list: Optional[List[str]]=None):
        if frames is not None:
            self.frames.extend(frames)
        if path_list is not None:
            frames = []
            for path in expand_path(path_list):
                frames.extend(read_dpdata(path, self.properties))
            self.frames.extend(frames)

