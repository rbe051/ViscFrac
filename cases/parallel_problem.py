# -*- coding: utf-8 -*-

import numpy as np
import viscous_porepy as pp
import scipy.sparse as sps
import copy

from utils import merge_grids
from cases import base_problem


class ParallelData(base_problem.BaseData):

    def create_gb(self, mesh_args=None):
        """ Load fractures and create grid bucket
        """
        if mesh_args is not None:
            self.mesh_args = mesh_args

        nx = self.mesh_args["mesh_size"]
        physdims = self.mesh_args["physdims"]
        nfrac = self.param["N"]

        # Generate fractures
        dy = physdims[1] / nfrac
        fracs = []
        self.frac_pos_y = physdims[1] / (2 * nfrac)
        for i in np.linspace(dy/2, physdims[1] - dy/2, nfrac):
            x = [0, physdims[0]]
            y = [i] * 2
            fracs.append(np.array([x, y]))

        physdims_split = physdims.copy()

        # Generate fractures
        num_dom_y = nfrac + 1
        if not isinstance(nfrac, int) and not nfrac.is_integer():
            raise ValueError('Numbe of fractures must be integer')
        if not (nx[1] / nfrac / 2).is_integer():
            raise ValueError('Number of cells and number of fractures not compatible')

        nx_split = nx.copy()
        nx_split[1] = int(nx[1] / nfrac / 2)

        dy = physdims[1] / nfrac / 2
        physdims_split[1] = dy
        
        domains2 = np.empty(num_dom_y, dtype=object)
        lg0 = pp.CartLeafGrid(nx_split, physdims_split, self.max_grid_level + 1)
        lgm = pp.CartLeafGrid(
            [nx_split[0], nx_split[1] * 2], [physdims_split[0], dy * 2],
            self.max_grid_level + 1
        )
        for y in range(num_dom_y):
            if y == 0 or (y + 1) == num_dom_y:
                lg = lg0.copy()
            else:
                lg = lgm.copy()
            lg.nodes[1] += max(0, y * 2 * dy - dy)
            for sg in lg.level_grids:
                sg.nodes[1] += max(0, y * 2 * dy - dy)
            domains2[y] = lg

        domains1 = np.empty(nfrac, dtype=object)
        lgf = pp.CartLeafGrid(nx[0], physdims_split[0], self.max_grid_level + 1)
        for y in range(nfrac):
            lg = lgf.copy()
            if y == 0:
                offset = dy
            else:
                offset = (2 * y + 1) * dy
            lg.nodes[1] += offset
            for sg in lgf.level_grids:
                sg.nodes[1] += offset
            domains1[y] = lg

        self.gb = pp.GridBucket()
        self.gb.add_nodes(domains2)
        self.gb.add_nodes(domains1)

        # Center grid at x=0
        for g, _ in self.gb:
            g.nodes[0] -= physdims[0] / 2
            for lg in g.level_grids:
                lg.nodes[0] -= physdims[0] / 2
                lg.compute_geometry()
        self.gb.compute_geometry()

        self.gb.assign_node_ordering()
        self.gb = merge_grids.mergeGridsOfEqualDim(self.gb)

        face_cells = np.empty((nfrac, num_dom_y), dtype=object)
        face_cells_bot = np.empty((nfrac, num_dom_y), dtype=object)
        face_cells_top = np.empty((nfrac, num_dom_y), dtype=object)
        for i in range(nfrac):
            lgf = domains1[i]
            lg_bot = domains2[i]
            lg_top = domains2[i + 1]
            zer = sps.csc_matrix((lgf.num_cells, lg_bot.num_faces - lgf.num_cells))
            one = sps.identity(lgf.num_cells)
            face_cells0 = sps.bmat([[zer, one]],format="csc",)

            if (i + 1) == nfrac:
                nxy = nx_split[1]
            else:
                nxy = 2 * nx_split[1]
            indices = np.arange(
                lg_top.num_faces - nx[0] * (nxy + 1),
                lg_top.num_faces - nx[0] * nxy,
            )
            indptr = np.arange(lgf.num_cells)
            values = np.ones(indices.size, dtype=bool)

            face_cells1 = sps.csr_matrix(
                (values, (indices, indptr)), (lg_top.num_faces, lgf.num_cells)).T

            face_cells_bot[i, i] = face_cells0
            face_cells_bot[i, i + 1] = sps.csc_matrix(face_cells1.shape)
            face_cells_top[i, i + 1] = face_cells1
            face_cells_top[i, i] = sps.csc_matrix(face_cells0.shape)

            face_cells[i, i] = face_cells0
            face_cells[i, i + 1] = face_cells1

        face_cells = sps.bmat(face_cells, format="csc")
        face_cells_bot = sps.bmat(face_cells_bot, format="csc")
        face_cells_top = sps.bmat(face_cells_top, format="csc")

        lgh = self.gb.grids_of_dimension(2)[0]
        lgf = self.gb.grids_of_dimension(1)[0]

        lgh.frac_pairs = [sps.find(face_cells_bot)[1], sps.find(face_cells_top)[1]]

        lgh.level_grids[0].frac_pairs = copy.deepcopy(lgh.frac_pairs)
        for level in range(lgh.num_levels - 1):
            proj_f = lgh.face_proj_level(level, level + 1).T
            is_bot = np.zeros(lgh.level_grids[level].num_faces)
            is_top = np.zeros(lgh.level_grids[level].num_faces)

            is_bot[lgh.level_grids[level].frac_pairs[0]] = True
            is_top[lgh.level_grids[level].frac_pairs[1]] = True
            bot = np.argwhere(proj_f * is_bot)
            top = np.argwhere(proj_f * is_top)
            lgh.level_grids[level + 1].frac_pairs = [bot, top]

        self.gb.add_edge((lgh, lgf), face_cells)

        self.gb.assign_node_ordering()
        pp.meshing.create_mortar_grids(self.gb)

        _, fi,_ = sps.find(face_cells)
        lgh.tags["fracture_faces"][fi] = True

#        self.gb = pp.meshing.cart_grid(fracs, nx, physdims=physdims)


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

        base_problem.assign_periodic_bc(self.gb, "ymin")

    def post_adaption(self, c):
        if self.gb.num_cells() < self.mesh_args["max_number_of_cells"]:
            return c

        nx_old = self.mesh_args["mesh_size"].copy()
        nx_min = 1
        ny_min = self.mesh_args["physdims"][1] / self.frac_pos_y

        nx_new = [nx_old[0] // 2, nx_old[1] // 2]

        dy = self.mesh_args["physdims"][1] / nx_new[1]
        if nx_new[0] < nx_min or nx_new[1] < ny_min:
            return c
        if not np.isclose(nx_new[0], nx_old[0] / 2):
            return c
        if not np.isclose(nx_new[1], nx_old[1] / 2):
            return c
        if not np.isclose(self.frac_pos_y / dy, int(self.frac_pos_y / dy)):
            return c
        return self.coarsen_domain(c)