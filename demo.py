# -*- coding: utf-8 -*-
"""
A script for demonstrating the use of the software. In this demonstration the
cases are run on a coarse mesh, long time steps and a short end time to limit
the computational time.
"""
import os

from cases import base_problem, parallel_problem, brick_problem, random_problem
from simulator import discretizations
from simulator import models

# Possible cases: homogeneous, parallel, brick, random
case = "homogeneous"

# Case parameters that can be changed
Pe = 500
R = 2
K = 5
A = 1e-3
N = 1

# Define mesh parameters
mesh_size = 60
num_grid_levels = 1
physdims = [2, 1]

# File name for output file
file_name = case
folder = os.path.join("results", "demo_" + case)

# Set time step parameters
if case=="homogeneous":
    end_time = 4
else:
    end_time = 1.5
    
time_step_param = {
    "initial_dt": 1e-2, 
    "end_time": end_time,
    "max_dt": 0.2,
    "export_vtk_all_timesteps": True,
    "vtk_folder_name": os.path.join(folder, "vtk"),
    "csv_folder_name": os.path.join(folder, "csv"),
    "adapt_dt_on_grid_coarsening": True,
    "file_name": file_name
}

# Store parameters to dictionaries
param = {
    "km": 1,
    "kf": K ,
    "kn": K,
    "Dm": 1 / Pe,
    "Df": 1 / Pe,
    "Dn": 1 / Pe,
    "aperture": A,
    "R": R,
    "N": N,
    "time_step_param": time_step_param,
}

mesh_args = {
    "physdims": physdims,
    "mesh_size": [physdims[0] * mesh_size, physdims[1] * mesh_size],
    "num_grid_levels": num_grid_levels,
    "max_number_of_cells": 100000,
    "frac_resolution": 0.1
}

print("Solve with viscosity model with mesh size:{} ".format(mesh_size))
print("Mesh and assign problem")
# Define problem
if case=="homogeneous":
    problem = base_problem.BaseData(mesh_args, param)
elif case=="parallel":
    problem = parallel_problem.ParallelData(mesh_args, param)
elif case=="brick":
    problem = brick_problem.BrickData(mesh_args, param)
elif case=="random":
    problem = random_problem.RandomData(mesh_args, param)
else:
    raise ValueError("Unknown case {}".format(case))

# Define discretization
print("Discretize")
# If you have umfpack or pardiso install on your system, changing the linear
# solver from "superlu" to "umfpack" or "pardiso" will significantly speed up
# the simulations for finer mesh resolutions.
disc = discretizations.ViscousFlow(problem, linear_solver="superlu")

# Call simulator
print("Call viscous flow model")
models.viscous_flow(disc, problem)
print("Solved with meshsize {}\n".format(mesh_size))
print("--------------------------------------------------------------\n")
