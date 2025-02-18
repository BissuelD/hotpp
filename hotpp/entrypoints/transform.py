import torch
import numpy as np
import tqdm
import copy
from ..data import get_dataset
from torch.utils.data import DataLoader


def _collate_fn(batch):
    coll_batch = {}
    number = {}
    for key in batch[0]:
        coll_batch[key] = torch.cat([d[key] for d in batch], dim=0)
        number[key] = [len(d[key]) for d in batch]
    return coll_batch, number


def main(
    *args,
    cutoff,
    indices=None,
    datapath=None,
    datatype=None,
    properties=["energy", "forces"],
    spin=False,
    batchsize=32,
    num_workers=4,
    **kwargs,
):
    if indices is not None:
        indices = np.loadtxt(indices, dtype=int)

    if isinstance(datapath, list) and len(datapath) == 1:
        datapath = datapath[0]
    dataset = get_dataset(
        cutoff=cutoff,
        datatype=datatype,
        datapath=datapath,
        properties=properties,
        spin=spin,
        indices=indices,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
    data_list = {k: [] for k in dataset[0]}
    number_list = {k: [0] for k in dataset[0]}
    for batch_data, number in tqdm.tqdm(data_loader):
        for key in data_list:
            data_list[key].append(
                copy.deepcopy(batch_data[key])
            )  # 否则不会释放内存，闹麻了
            number_list[key].extend(copy.deepcopy(number[key]))
    collated_data = {key: torch.cat(data_list[key], dim=0) for key in data_list}
    collated_number = {
        key: torch.cumsum(torch.tensor(number_list[key], dtype=torch.long), dim=0)
        for key in number_list
    }
    meta_data = {"cutoff": cutoff, "n_data": len(dataset)}
    torch.save((collated_data, collated_number, meta_data), f"{datapath}-{'-'.join(properties)}-{cutoff}.pt")


if __name__ == "__main__":
    main()
