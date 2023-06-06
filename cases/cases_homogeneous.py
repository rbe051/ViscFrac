"""
Module for setting up and running the cases without fractures from Paper:
    Berge, R.L,. Berre, I., Keilekavlen, E., & Nordbotten, J.,M.. (2023)
        Numerical simulations of viscous fingering in fractured porous media


Copyright 2023 Runar Lie Berge

This file is part of ViscFrac.

ViscFrac is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

ViscFrac is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
ViscFrac. If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import viscous_porepy as pp
import sys
import os
from multiprocessing import Pool

from cases import base_problem as problems
from simulator import discretizations
from simulator import models
from utils import viz


def run_Pe_R_variation(run_id, local_grid_adaptation=False):
    np.random.seed()
    Pes = [
        1000, 1000, 1000, 1000,
        500, 1000, 2000, 4000, 8000,
    ]
    if local_grid_adaptation:
        mesh_size = 20
        num_grid_levels = 6
    else:
        mesh_size = 512
        num_grid_levels = 1

    Rs = [
        1, 2, 3, 4,
        2, 2, 2, 2, 2
    ]
    physdims = [2, 1]

    R = Rs[run_id]
    Pe = Pes[run_id]

    time_step_param = {
        "initial_dt": 1e-5 * pp.SECOND,
        "end_time": 100 * pp.SECOND,
        "max_dt": 0.01 * pp.SECOND,
        "vtk_folder_name": "res_no_frac/vtk",
        "csv_folder_name": "res_no_frac/csv",
        "adapt_dt_on_grid_coarsening": True,
    }

    param = {
        "km": 1 * pp.METER ** 2,
        "Dm": 1 / Pe * pp.METER ** 2 / pp.SECOND,
        "aperture": 1 * pp.METER,
        "R": R,
        "time_step_param": time_step_param,
    }

    mesh_args = {
        "physdims": physdims,
        "mesh_size": [physdims[0] * mesh_size, physdims[1] * mesh_size],
        "fracture_file_name": "no_fractures.csv",
        "num_grid_levels": num_grid_levels,
        "max_number_of_cells": 600000,
    }

    file_name = "Pe_{}_R_{}_mesh_{}_task_{}".format(Pe, R, mesh_size, run_id)
    file_name = viz.set_unique_file_name(time_step_param["vtk_folder_name"], file_name)
    time_step_param["file_name"] = file_name
    print("Solve with viscosity model with mesh size:{} ".format(mesh_size))

    print("Mesh and assign problem")
    homogeneous_problem = problems.BaseData(mesh_args, param)
    print("Discretize")
    disc = discretizations.ViscousFlow(homogeneous_problem)
    print("Call viscous flow model")
    models.viscous_flow(disc, homogeneous_problem)
    print("Solved with meshsize {}\n".format(mesh_size))
    print("--------------------------------------------------------------\n")


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = "2"
    if len(sys.argv) > 1 and sys.argv[1] != "-i":
        run_id = int(sys.argv[1])
        run_Pe_R_variation(run_id)
    else:
        # Define the dataset
        run_id = np.arange(9)
        # Output the dataset
        print('Dataset: ' + str(run_id))
        agents = run_id.size
        chunksize = 1
        with Pool(processes=agents) as pool:
            result = pool.map(run_Pe_R_variation, run_id, chunksize)
        # Output the result
        print('Result:  ' + str(result))
