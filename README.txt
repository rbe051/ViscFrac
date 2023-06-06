This repository contains the toolbox ViscFrac for simulating viscous fingering in fractured porous media.

As an introduction to the ViscFrac toolbox it is recommended to inspect and run the 

  demo.py

script. This provides an interface and demonstration for how to define parameters, set up a problem and run the simulator. Note that the demo uses a coarse mesh and time-stepping to speed up the simulations to demonstrate the software. To obtain reliable results a much finer grid and time-stepping should be used.

The implementation of the simulator is based on the paper:

  Numerical simulations of viscous fingering in fractured porous media
    by Runar L. Berge, Inga Berre, Eirik Keilegavlen, and Jan M. Nordbotten

and the repository also contains the runscripts for reproducing the results of this paper. These runscripts are located as

  run_homogeneous.py
  run_parallel_case.py
  run_brick_case.py
  run_random_case.py

Please refer to the comments in these files for how to use these scripts.

ViscFrac is based on a modified version of the simulation tool PorePy (Copyright 2016-2017: University of Bergen) that is located in the subfolder 

    viscous_porepy

The included version of PorePy is modified to allow for adaptivity of the grid. The modified version of PorePy is included in this repository as viscous_porepy, but can also be found here: https://github.com/rbe051/porepy/tree/leaf_grid

The required packages needed to run the examples can be found in:

  requirements.txt


To speed up the simulation times we recommend that you install pypardiso or umfpack on your system. See:
https://pypi.org/project/pypardiso/
or
https://pypi.org/project/scikit-umfpack/


This file is part of ViscFrac.

ViscFrac is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ViscFrac is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with ViscFrac. If not, see <https://www.gnu.org/licenses/>. 
