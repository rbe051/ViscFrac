# -*- coding: utf-8 -*-
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
import viscous_porepy.fracs.structured

from utils import merge_grids
from cases import brick_problem
from cases import base_problem as default_problem


class RandomData(brick_problem.BrickData):
    def __init__(self, mesh_args, param):
        self.initial_physdims_ = mesh_args["physdims"].copy()
        self.has_generated_fracs = False
        self.frac_pos_x = mesh_args["frac_resolution"]        
        super().__init__(mesh_args, param)

    def generate_fracs(self):
        dx = self.mesh_args["frac_resolution"]
        dy = dx
        xmin = self.domain["xmin"] - 5
        xmax = min(2 * self.param["time_step_param"]["end_time"] * self.param["kf"], 2000)
        ymin = 0
        ymax = self.mesh_args["physdims"][1]
        area = (xmax - xmin)*(ymax - ymin)
        avg_frac_length = 2.5                
        num_x_fracs = int(self.param["N"] * area / avg_frac_length)
        xfracs = []
        for i in range(num_x_fracs):
            xLength = round((dx + np.random.uniform(0, avg_frac_length * 2)) / dx) * dx
            xStart = round(np.random.uniform(xmin, xmax) / dx) * dx
            xEnd = xStart + xLength
            y = round(np.random.uniform(ymin + dy, ymax - dy) / dy ) * dy
            xfracs.append(np.array([[xStart, xEnd], [y, y]]))

        num_y_fracs = int(self.param["N"] * (xmax - xmin))
        yfracs = []
        for i in range(num_y_fracs):
            yLength = np.random.uniform(dy, ymax - dy)
            yStart = round(np.random.uniform(ymin, ymax - dy) / dy) * dy
            yEnd = round((yStart + yLength) / dy) * dy
            x = round(np.random.uniform(0 + dx, xmax) / dx) * dx
            yfracs.append(np.array([[x, x], [yStart, yEnd]]))
            if yEnd >= 1 + dy: # Wrap around periodic bc
                yStartPer = 0
                yEndPer = 0 + yEnd - 1
                yfracs.append(np.array([[x, x], [yStartPer, yEndPer]]))

        # remove overlapping fractures:
        pts = np.hstack(xfracs + yfracs)
        edg = np.vstack([np.arange(0, pts.shape[1], 2),
                          np.arange(1, pts.shape[1], 2)])
        pts, edges = pp.intersections.split_intersecting_segments_2d(
            pts, edg
        )
        self.fracs = []
        for e in range(edges.shape[1]):
            x = np.round(pts[0, edges[:, e]].ravel() / dx) * dx
            y = np.round(pts[1, edges[:, e]].ravel() / dy) * dy
            self.fracs.append(np.vstack([x, y]))

    def get_fracs_in_domain(self):
        fracs = []
        xmin = self.domain["xmin"]
        xmax = xmin + self.mesh_args["physdims"][0]
        ymin = self.domain["ymin"]
        ymax = ymin + self.mesh_args["physdims"][1]
        dx = self.mesh_args["frac_resolution"]
        dy = dx
        tol = 1e-8
        for f in self.fracs:
            if f[0, 0] < xmin + dx-tol and f[0, 1] < xmin + dx-tol: # left of domain
                continue
            elif f[0, 0] > xmax-dx + tol and f[0, 1] > xmax-dx + tol: # right of domain
                continue
            elif f[1, 0] < ymin + dy-tol and f[0, 1] < ymin + dy - tol: # below of domain
                continue
            elif f[1, 0] > ymax-dy+tol and f[1, 1] > ymax-dy + tol: # above of domain
                continue
            
            fcp = f.copy()
            fcp[0] -= xmin
            fracs.append(fcp)
        return fracs
    
    def create_gb(self, mesh_args=None):
        """ Load fractures and create grid bucket
        """
        if mesh_args is not None:
            self.mesh_args = mesh_args

        nx = self.mesh_args["mesh_size"]
        physdims = self.mesh_args["physdims"]
        # Generate fractures
        if not self.has_generated_fracs:
            self.domain = {
                        "xmin": - self.initial_physdims_[0] / 2,
                        "xmax": + self.initial_physdims_[0] / 2,
                        "ymin": 0,
                        "ymax": self.initial_physdims_[1],
                        "zmin": 0,
                        "zmax": 0,
                        "L": self.initial_physdims_[0],
                    }

            self.generate_fracs()
            self.has_generated_fracs = True

        fracs = self.get_fracs_in_domain()
        grid_list = pp.fracs.structured._cart_grid_2d(fracs, nx, physdims, pp.CartLeafGrid)
        # Sort nodes acording to global indices.
        for grid in grid_list[1]:
            IA = np.argsort(grid.global_point_ind)
            grid.nodes = grid.nodes[:, IA]
            grid.global_point_ind = grid.global_point_ind[IA]
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
        if ((xvar_min - xmin) / L > 0.3) and ((xvar_min - xmin)>self.initial_physdims_[0]):
            return True
        else:
            return False
    
    def post_shift(self):
        self.create_gb()

    def post_adaption(self, c):
        if self.gb.num_cells() < self.mesh_args["max_number_of_cells"]:
            return c

        nx_old = self.mesh_args["mesh_size"].copy()

        nx_new = [nx_old[0] // 2, nx_old[1] // 2]

        dx = self.mesh_args["physdims"][0] / nx_new[0]
        dy = self.mesh_args["physdims"][1] / nx_new[1]
        nx_min = self.mesh_args["physdims"][0] / self.mesh_args["frac_resolution"]
        ny_min = self.mesh_args["physdims"][1] / self.mesh_args["frac_resolution"]
        if nx_new[0] < nx_min or nx_new[1] < ny_min:
            return c
        if not np.isclose(nx_new[0], nx_old[0] / 2):
            return c
        if not np.isclose(nx_new[1], nx_old[1] / 2):
            return c
        if not np.isclose(self.mesh_args["frac_resolution"] / dx, int(self.mesh_args["frac_resolution"] / dx)):
            return c
        if not np.isclose(self.mesh_args["frac_resolution"] / dy, int(self.mesh_args["frac_resolution"] / dy)):
            return c
        return self.coarsen_domain(c)
