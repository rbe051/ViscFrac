"""
Simulation script for reproducing the test cases with brick fracture
networks in the paper:
    Berge, R.L,. Berre, I., Keilekavlen, E., & Nordbotten, J.,M.. (2023) 
        Numerical simulations of viscous fingering in fractured porous media

To run a case type:
    python run_brick_case.py <CASE_ID> <NUM_PARALLEL>
    
where CASE_ID is a integer 0-3:
    CASE_ID = 0 Vary the permeability ratio, K, and dimensionless aperture A
    CASE_ID = 1 Vary the Peclet number Pe and log viscosity ratio R
    CASE_ID = 2 Vary the Fracture density N
    CASE_ID = 3 Vary the volumetric flux ratio AK
    CASE_ID = 4 Repeat each simulation of the volumetric flux 5 times

and NUM_PARALLEL is the number of cases run in parallel
"""
from multiprocessing import Pool
import numpy as np
import sys
import os

import cases.cases_random as network_type


if __name__ == "__main__":
    # Select case, 0, 1 or 2.
    #   Case 0: Vary the permeability ratio, K, and dimensionless aperture A
    #   Case 1: Vary the Peclet number Pe and log viscosity ratio R
    #   Case 2: Vary the Fracture density N
    #   Case 3: Vary the volumetric flux ratio AK
    #   Case 4: Repead F variation 5 times
    if len(sys.argv) != 3:
        print("Invalid argument list. Usage: run_random_case.py <case_id> <num_parallel>")
        raise ValueError("Invalid argument list. Usage: run_random_case.py <case_id> <num_parallel>")

    case = int(sys.argv[1])
    # Set the number of threads per process
    os.environ['OMP_NUM_THREADS'] = "2"
    # Set the number of processes. If it is set to None the number of
    # processes will equal the number of parameter evaluations.
    num_agents = int(sys.argv[2])

    if case == 0:
        # Define the dataset
        run_id = np.arange(12)
        # Output the dataset
        print('Dataset: ' + str(run_id))
        if num_agents is None:
            num_agents = run_id.size
        chunksize = 1
        with Pool(processes=num_agents) as pool:
            result = pool.map(network_type.run_K_A_variation, run_id, chunksize)

    elif case == 1:
        # Define the dataset
        run_id = np.arange(9)
        # Output the dataset
        print('Dataset: ' + str(run_id))
        if num_agents is None:
            num_agents = run_id.size
        chunksize = 1
        with Pool(processes=num_agents) as pool:
            result = pool.map(network_type.run_Pe_R_variation, run_id, chunksize)

    elif case == 2:
        # Define the dataset
        run_id = np.arange(5)
        # Output the dataset
        print('Dataset: ' + str(run_id))
        if num_agents is None:
            num_agents = run_id.size
        chunksize = 1
        with Pool(processes=num_agents) as pool:
            result = pool.map(network_type.run_N_variation, run_id, chunksize)

    elif case == 3:
        # Define the dataset
        run_id = np.arange(12)
        # Output the dataset
        print('Dataset: ' + str(run_id))
        if num_agents is None:
            num_agents = run_id.size
        chunksize = 1
        with Pool(processes=num_agents) as pool:
            result = pool.map(network_type.run_F_variation, run_id, chunksize)
    elif case == 4:
        # Define the dataset
        run_id = np.arange(0, 15)
        # Output the dataset
        print('Dataset: ' + str(run_id))
        if num_agents is None:
            num_agents = run_id.size
        chunksize = 1
        with Pool(processes=num_agents) as pool:
            result = pool.map(network_type.run_F_variation_repeated, run_id, chunksize)

    else:
        print("Unknow argument '{}', valid arguments are 0, 1, 2, 3, 4".format(case))
        raise ValueError() 

    # Output the result
    print('Result:  ' + str(result))