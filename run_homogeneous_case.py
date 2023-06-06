"""
Simulation script for reproducing the test cases without fractures in the
paper:
    Berge, R.L,. Berre, I., Keilekavlen, E., & Nordbotten, J.,M.. (2023) 
        Numerical simulations of viscous fingering in fractured porous media

To run a case type:
    python run_homogeneous_case.py <CASE_ID> <NUM_PARALLEL>
    
where CASE_ID is a integer 0 (only one case available for homogeneous domain):
    CASE_ID = 0 Vary the Peclet number Pe and log viscosity ratio R

and NUM_PARALLEL is the number of cases run in parallel


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
from multiprocessing import Pool
import numpy as np
import sys
import os

import cases.cases_homogeneous as network_type


if __name__ == "__main__":
    # Select case, 0 (Only 1 case available)
    #   Case 0: Vary the Peclet number Pe and log viscosity ratio R
    if len(sys.argv) != 3:
        print("Invalid argument list. Usage: run_homogeneous_case.py <case_id> <num_parallel>")
        raise ValueError("Invalid argument list. Usage: run_homogeneous_case.py <case_id> <num_parallel>")

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
            result = pool.map(network_type.run_Pe_R_variation, run_id, chunksize)
    else:
        print("Unknow argument '{}', valid arguments are 0, 1, 2, 3, 4".format(case))
        raise ValueError() 

    # Output the result
    print('Result:  ' + str(result))
