"""
Module for setting up and running the cases with parallel fracture networks
from Paper:
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

from utils import merge_grids
import cases.base_problem as default_problem
from utils import frac_gen


class BrickData(default_problem.BaseData):
    def __init__(self, mesh_args, param):
        self.initial_physdims_ = mesh_args["physdims"].copy()
        super().__init__(mesh_args, param)

    def create_gb(self, mesh_args=None):
        """ Load fractures and create grid bucket
        """
        if mesh_args is not None:
            self.mesh_args = mesh_args

        nx = self.mesh_args["mesh_size"]
        physdims = self.mesh_args["physdims"]

        # Generate fractures
        nfrac = self.param["N"]
        dx = physdims[0] / nx[0]
        dy = physdims[1] / nx[1]

        self.frac_pos_x = 2.0 / nfrac
        self.frac_pos_y = physdims[1] / (4 * nfrac)

        if not np.isclose(self.frac_pos_x / dx, int(self.frac_pos_x / dx)):
            raise ValueError('''
            The mesh must be conforming to the fractures. Number of cells
            in the x-direction must be an integer factor of: {}
            '''.format(physdims[0] / self.frac_pos_x))
        if not np.isclose(self.frac_pos_y / dy, int(self.frac_pos_y / dy)):
            raise ValueError('''
            The mesh must be conforming to the fractures. Number of cells
            in the y-direction must be an integer factor of: {}
            '''.format(physdims[1] / self.frac_pos_y))

        nfracx = int(round(0.5 * physdims[0] * nfrac))

        fracs = frac_gen.brick(physdims[0], physdims[1], nfracx, nfrac, True)
        grid_list = pp.fracs.structured._cart_grid_2d(fracs, nx, physdims, pp.CartLeafGrid)
        self.gb = pp.meshing.grid_list_to_grid_bucket(grid_list)

        # Center grid at x=0
        if not self.domain is None:
            shift = self.domain["xmin"]
        else:
            shift = - self.initial_physdims_[0] / 2
        for g, _ in self.gb:
            g.nodes[0] += shift
            if g.dim==0:
                g.cell_centers[0] += shift
            if hasattr(g, "level_grids"):
                for lg in g.level_grids:
                    lg.nodes[0] += shift
                    lg.compute_geometry()
        self.gb.compute_geometry()

        self.gb = merge_grids.mergeGridsOfEqualDim(self.gb)

        g_max = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        (xmin, ymin, zmin) = np.min(g_max.nodes, axis=1)
        (xmax, ymax, zmax) = np.max(g_max.nodes, axis=1)
        self.domain = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
            "L": xmax - xmin,
        }
        for g, _ in self.gb:
            g.set_periodic_map(np.zeros((2, 0), dtype=int))

        default_problem.assign_periodic_bc(self.gb, "ymin")

    def frame_velocity(self):
        return np.array([0.0, 0.0, 0.0])

    def extend_domain(self, c):
        xmin = np.min(self.gb.cell_centers()[0])
        xmax = np.max(self.gb.cell_centers()[0])
        xvar_max = np.max(self.gb.cell_centers()[0, c > 0.01])

        L = xmax - xmin
        if (xmax - xvar_max) / L < 0.2:
            return True
        else:
            return False

    def shift_domain(self, c):
        xmin = np.min(self.gb.cell_centers()[0])
        xmax = np.max(self.gb.cell_centers()[0])
        xvar_min = np.min(self.gb.cell_centers()[0, c < 0.99])

        L = xmax - xmin
        if (xvar_min - xmin) / L > 0.3:
            return True
        else:
            return False

    def post_adaption(self, c):
        if self.gb.num_cells() < self.mesh_args["max_number_of_cells"]:
            return c

        nx_old = self.mesh_args["mesh_size"].copy()
        nx_min = self.mesh_args["physdims"][0] / self.frac_pos_x
        ny_min = self.mesh_args["physdims"][1] / self.frac_pos_y

        nx_new = [nx_old[0] // 2, nx_old[1] // 2]

        dx = self.mesh_args["physdims"][0] / nx_new[0]
        dy = self.mesh_args["physdims"][1] / nx_new[1]
        if nx_new[0] < nx_min or nx_new[1] < ny_min:
            return c
        if not np.isclose(nx_new[0], nx_old[0] / 2):
            return c
        if not np.isclose(nx_new[1], nx_old[1] / 2):
            return c
        if not np.isclose(self.frac_pos_x / dx, int(self.frac_pos_x / dx)):
            return c
        if not np.isclose(self.frac_pos_y / dy, int(self.frac_pos_y / dy)):
            return c
        return self.coarsen_domain(c)
