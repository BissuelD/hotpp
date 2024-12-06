import torch
from ase.data import chemical_symbols, atomic_numbers


def main(*args, model="model.pt", device="cpu", output="infer.pt", symbols=None, double=False, **kwargs):
    model = torch.load(model, map_location=torch.device(device))
    if double:
        model = model.double()
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
    for name, value in model.named_buffers():
        if "atomic_number_to_type" in name:
            value.data = torch.arange(0, len(all_elements), 1, dtype=torch.long)
    lammps_infer = torch.jit.script(model)
    lammps_infer.save(f'lammps-{output}')


if __name__ == "__main__":
    main()
