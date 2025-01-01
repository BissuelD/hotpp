Installation
============

-----
Base
-----

By pip
------

.. code:: shell

   $ pip install git+https://gitlab.com/bigd4/hotpp.git

From Source
-----------

.. code:: shell

   $ git clone https://gitlab.com/bigd4/hotpp.git
   $ cd hotpp
   $ pip install -e .

---
DLC
---

ASE Interface
----------------
Just add `$HOTPP_PATH/interface/ase/hotase.py` to your `$PYTHONPATH`. 
And an example can be found in :doc:`examples/carbon/carbon`.

Lammps Interface
----------------

````````````
Requirements
````````````
.. warning:: 
   Here are the versions I used during compilation.
   Other versions of libtorch_, lammps_, and gcc_ (support c17) should also work, 
   but they haven't been fully tested. If something goes wrong, 
   something may go wrong.

+------------------+-----------------+
| Software/Library | Tested versions |
+==================+=================+
| libtorch_        | 2.2.2+cu12.1    |
+------------------+-----------------+
| lammps_          | 20230802        |
+------------------+-----------------+
| cmake_           | 3.22.1          |
+------------------+-----------------+
| gcc_             | 11.4.0          |
+------------------+-----------------+

````````````
Installation
````````````
- Prepare the required files:

.. code-block:: bash

   cp -r $LIBTORCH_PATH/libtorch $LAMMPS_PATH 
   cp -r $HOTPP_PATH/interface/lammps/src/* $LAMMPS_PATH/src 
   cp -r $HOTPP_PATH/interface/lammps/cmake/* $LAMMPS_PATH/cmake 

And so compared to the original lammps folder, 
the following files has been added:

::

   lammps-2Aug2023/
    ├── libtorch/
    │   ├── bin/
    │   ├── lib/
    │   └── ...
    ├── src/ 
    │   ├── pair_miao.cpp
    │   ├── pair_miao.h
    │   ├── compute_miao_dipole.cpp
    │   ├── compute_miao_dipole.h
    │   ├── compute_miao_polarizability.cpp
    │   ├── compute_miao_polarizability.h
    │   └── ...
    ├── cmake/
    │   ├── CMakeLists.txt
    │   └── ...
    └── ...

- Build the lammps binary:

.. code-block:: bash

   mkdir $LAMMPS_PATH/build
   cd $LAMMPS_PATH/build
   cmake -D BUILD_MPI=ON -D BUILD_OMP=ON -D CAFFE2_USE_CUDNN=1 -D LAMMPS_MACHINE=hotpp -D CMAKE_BUILD_TYPE=RELEASE ../cmake
   make -j4

And the executable file **lmp_hotpp** can be seen in the folder.

````````````
Usage
````````````
An MD simulation example can be found in :doc:`examples/water/water`.


.. _libtorch: https://pytorch.org/
.. _lammps: https://www.lammps.org/
.. _cmake: https://cmake.org/
.. _gcc: https://gcc.gnu.org/


