This repository contains the toolbox ViscFrac for simulating viscous fingering in fractured porous media.

The implementation is based on the paper:

Numerical simulations of viscous fingering in fractured porous media
by Runar L. Berge, Inga Berre, Eirik Keilegavlen, and Jan M. Nordbotten

and the repository also contains the runscripts for reproducing the results of this paper.

ViscFrac is based on a modified version of the simulation tool PorePy. PorePy is modified to allow for adaptivity of the grid. The modified version of PorePy is included in this repository as viscous_porepy, but can also be found here: https://github.com/rbe051/porepy/tree/leaf_grid

The required packages needed to run the examples can be found in requirements.txt

To speed up the simulation times we recomend that you install pypardiso or umfpack on your system. See:
https://pypi.org/project/pypardiso/
or
https://pypi.org/project/scikit-umfpack/

As an introduction to the ViscFrac toolbox it is recomended to run the demo.py script. This provides an interface for running the simulation cases from the paper. Note that the demo uses a coarse mesh and timestepping to speed up the simulations in order to demonstrate the software. To obtain reliable results a much finer grid and time-stepping should be used.
