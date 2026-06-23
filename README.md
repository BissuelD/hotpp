# **HotPP: High order tensor Passing Potential**

This repository is a fork of the [hotPP](https://gitlab.com/bigd4/hotpp) [1] code to allow for various `l3_tensor` symmetries (Kleinmann, intrinsic, none) and aggregation of computed properties for only a subset of atoms in the system.

[![Documentation Status](https://readthedocs.org/projects/hotpp/badge/?version=latest)](https://hotpp.readthedocs.io/en/latest/?badge=latest)

## Introduction

`HotPP` is an open-source package designed for constructing message passing network interatomic potentials. It facilitates the utilization of arbitrary order Cartesian tensors as messages while maintaining equivalence maintenance.

## Current Features

* Building machine learning potentials for molecular and periodic systems;
* Learning dipole moments and polarizability tensors;
* Interface to LAMMPS and ASE;
* Optional first-N atom aggregation for dipole/polarizability (Model.aggregateFirstN)

## Documentation

* An overview of code documentation and tutorials for getting started with `HotPP` can be found [at this link](https://hotpp.readthedocs.io/en/latest/).

## Installation

### Recommended installation for this fork

This fork is under active development, and the latest code may not be stable. We recommend installing as follows to get the latest updates.

It relies on the following libraries:

* ase
* lightning
* pyyaml
* setuptools
* tensorboard
* torch

To automate the installation of these libraries, we rely on `uv`, a package manager for Python. To have more information about `uv`, and its installation, you can visit the [uv documentation](https://uv.readthedocs.io/en/latest/).

#### Clone the repository

First, clone the repository with

```bash
git clone https://github.com/BissuelD/hotpp.git
```

Then, navigate to the project directory with

```bash
cd hotpp
```

#### Create a virtual environment

After installing uv, simply create a virtual environment with

```bash
uv venv
```

Then activate the virtual environment with

```bash
source .venv/bin/activate
```

#### Install the required libraries

You can then simply run

```bash
PYTHONPATH=$(pwd) uv sync
```

to install the required libraries.

#### Day-to-day usage

After installation, you no longer need to use `uv` commands, as the virtual environment is now set up with all the necessary dependencies.

Simply activate the virtual environment when you need it with

```bash
source .venv/bin/activate
```

and deactivate the virtual environment after usage with

```bash
deactivate
```

#### Sporadic updates of the libraries

If the software environment of the project gets updated, you can run

```bash
git pull
```

to get the latest version of the project, including the `uv` related configuration files. Then, just repeat the steps presented earlier:

* Activate the virtual environment

```bash
source .venv/bin/activate
```

* Update the libraries

```bash
PYTHONPATH=$(pwd) uv sync
```

* Deactivate the virtual environment

```bash
deactivate
```

### Other way to install the original version: using pip

You can use https:

```shell
pip install git+https://gitlab.com/bigd4/hotpp.git
```

or use [ssh](https://docs.gitlab.com/ee/user/ssh.html)

```shell
pip install git+ssh://git@gitlab.com/bigd4/hotpp.git
```

Your may need to add `--user` if you do not have the root permission. Or use `--force-reinstall` if you already  have `HotPP` (add `--no-dependencies` if you do not want to reinstall the dependencies).

### Other way to install the original version: from source

Frist, use git clone to get the source code:

```shell
git clone https://gitlab.com/bigd4/hotpp.git
```

Alternatively, you can download the source code from website.

Then, go into the directory and install with pip:

```shell
pip install -e .
```

pip will read **setup.py** in your current directory and install. The `-e` option means python will directly import the module from the current path, but not copy the codes to the default lib path and import the module there, which is convenient for modifying in the future. If you do not have the need, you can remove the option.

### Check

You can use

```shell
hotpp -v
```

to check if you have installed successfully

### Optional First-N Aggregation

You can limit reductions (dipole, polarizability, ...) to the first N atoms in each structure by adding in your input YAML under `Model`:

```yaml
Model:
    aggregateFirstN: 3  # only sum first 3 atomic contributions
```

Set to `null` (or omit) to use all atoms.

### Update

If you installed by pip, use:

```shell
hotpp update

```

If you installed from source, use:

```shell
cd <path-to-magus-package>
git pull origin master
```

## Interface

`HotPP` now support [ASE](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-ase.calculators) and [lammps](https://www.lammps.org/).

### ASE

### LAMMPS

## Contributors

HotPP is developed by Prof. Jian Sun's group at the School of Physics at Nanjing University.

The contributors are:

* Jian Sun
* Junjie Wang
* Yong Wang
* Haoting Zhang
* Ziyang Yang
* Zhixin Liang
* Jiuyang Shi

## Citations

| Reference | cite for what                         |
| --------- | ------------------------------------- |
| [1]       | for any work that used `HotPP`        |

## Reference

[1] [Wang, J. et al. E(n)-Equivariant cartesian tensor message passing interatomic potential. Nat Commun 15, 7607 (2024).](https://doi.org/10.1038/s41467-024-51886-6)
