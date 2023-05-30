"""
Module for setting up and running the cases with parallel fracture networks
from Paper:
    Berge, R.L,. Berre, I., Keilekavlen, E., & Nordbotten, J.,M.. (2023) 
        Numerical simulations of viscous fingering in fractured porous media
"""
import numpy as np
import viscous_porepy as pp
import os

from cases import parallel_problem
from simulator import discretizations
from simulator import models
from utils import viz


def run_K_A_variation(run_id):
    np.random.seed()
    As = [
        1e-2, 1e-2, 1e-2, 1e-2, 1e-2,
        1e-3, 1e-3, 1e-3, 1e-3, 1e-3,
        1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
    ]
    Ks = [
        1.1, 2, 10, 100, 1000,
        1.1, 2, 10, 100, 1000,
        1.1, 2, 10, 100, 1000,
    ]

    R = 2
    Pe = 1000
    K = Ks[run_id]
    A = As[run_id]
    N = 1
    file_name = "Pe_{}_R_{}_K_{}_A_{}_N_{}_task_{}".format(Pe, R, K, A, N, run_id)
    folder = os.path.join("results", "res_parallel_fracs","KA")
    return run_parallel_frac(Pe, R, K, A, N, file_name, folder)


def run_Pe_R_variation(run_id):
    np.random.seed()
    Rs = [
        1, 2, 3, 4,
        2, 2, 2, 2, 2,
    ]
    Pes = [
        1000, 1000, 1000, 1000,
        100, 500, 1000, 2000, 4000,
    ]

    R = Rs[run_id]
    Pe = Pes[run_id]
    K = 10
    A = 1e-3
    N = 1
    file_name = "Pe_{}_R_{}_K_{}_A_{}_N_{}_task_{}".format(Pe, R, K, A, N, run_id)
    folder = os.path.join("results", "res_parallel_fracs","PeR")
    return run_parallel_frac(Pe, R, K, A, N, file_name, folder)


def run_N_variation(run_id):
    np.random.seed()
    num_fracs = [1, 2, 3, 4, 5]
    mesh_sizes = [512, 512, 384, 512, 320]
    R = 2
    Pe = 1000
    K = 10
    A = 1e-3
    N = num_fracs[run_id]
    file_name = "Pe_{}_R_{}_K_{}_A_{}_N_{}_task_{}".format(Pe, R, K, A, N, run_id)
    folder = os.path.join("results", "res_parallel_fracs","N")
    return run_parallel_frac(Pe, R, K, A, N, file_name, folder, mesh_sizes[run_id], 200)


def run_F_variation(run_id):
    np.random.seed()
    As = [
        1e-2,              # F = 10
        1e-2, 1e-3,        # F = 1
        1e-2, 1e-3, 1e-4,  # F = 0.1
        1e-3, 1e-4, 1e-5,  # F = 0.01
        1e-4, 1e-5, 1e-6,  # F = 0.001
    ]
    Ks = [
        1000,
        100, 1000,
        10, 100, 1000,
        10, 100, 1000,
        10, 100, 1000,
    ]

    R = 2
    Pe = 1000
    K = Ks[run_id]
    A = As[run_id]
    N = 1
    file_name = "Pe_{}_R_{}_K_{}_A_{}_N_{}_task_{}".format(Pe, R, K, A, N, run_id)
    folder = os.path.join("results", "res_parallel_fracs", "F")
    return run_parallel_frac(Pe, R, K, A, N, file_name, folder)


def run_parallel_frac(Pe, R, K, A, N, file_name, folder, mesh_size=None, end_time=100, local_grid_adaptation=False):
    if local_grid_adaptation:
        if mesh_size is None:
            mesh_size = 20
        num_grid_levels = 6
    else:
        if mesh_size is None:
            mesh_size = 512
        num_grid_levels = 1

    physdims = [2, 1]

    time_step_param = {
        "initial_dt": 1e-5 * pp.SECOND,
        "end_time": end_time,
        "max_dt": 0.01 * pp.SECOND,
        "vtk_folder_name": os.path.join(folder, "vtk"),
        "csv_folder_name": os.path.join(folder, "csv"),
        "adapt_dt_on_grid_coarsening": True,
    }

    param = {
        "km": 1 * pp.METER ** 2,
        "kf": K * pp.METER ** 2,
        "kn": K * pp.METER ** 2,
        "Dm": 1 / Pe * pp.METER ** 2 / pp.SECOND,
        "Df": 1 / Pe * pp.METER ** 2 / pp.SECOND,
        "Dn": 1 / Pe * pp.METER ** 2 / pp.SECOND,
        "aperture": A * pp.METER,
        "R": R,
        "N": N,
        "time_step_param": time_step_param,
    }

    mesh_args = {
        "physdims": physdims,
        "mesh_size": [physdims[0] * mesh_size, physdims[1] * mesh_size],
        "num_grid_levels": num_grid_levels,
        "max_number_of_cells": 600000,
    }

    file_name = viz.set_unique_file_name(time_step_param["vtk_folder_name"], file_name)
    time_step_param["file_name"] = file_name
    print("Solve with viscosity model with mesh size:{} ".format(mesh_size))

    print("Mesh and assign problem")
    problem = parallel_problem.ParallelData(mesh_args, param)
    print("Discretize")
    disc = discretizations.ViscousFlow(problem)
    print("Call viscous flow model")
    models.viscous_flow(disc, problem)
    print("Solved with meshsize {}\n".format(mesh_size))
    print("--------------------------------------------------------------\n")
    return 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Invalid argument list. Usage: parallel_fractures.py <case_id> <run_id>")
        raise ValueError()

    # Set the number of threads per process
    os.environ['OMP_NUM_THREADS'] = "2"
    case = int(sys.argv[1])
    run_id = int(sys.argv[2])
    if case == 0:
        result = run_K_A_variation(run_id)
    elif case == 1:
        result = run_Pe_R_variation(run_id)
    elif case == 2:
        result = run_N_variation(run_id)
    elif case == 3:
        result = run_F_variation(run_id)
    else:
        print("Unknow argument '{}', valid arguments are 0, 1, 2".format(case))
        raise ValueError()
    print("result: ", result)
