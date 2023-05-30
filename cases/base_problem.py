import numpy as np
import viscous_porepy as pp
import viscous_porepy.ad
import scipy.spatial.distance as spsd
import scipy.sparse as sps
import scipy as sp


class BaseData(object):
    """ Data class for copuled flow and temperature transport.
    """
    def __init__(self, mesh_args, param):
        """
        Parameters:
        mesh_args(dictionary): Dictionary containing meshing parameters.
        """
        self.gb = None
        self.domain = None
        self.mesh_args = mesh_args
        self.max_grid_level = self.mesh_args["num_grid_levels"] - 1

        self.tol = 1e-8
        self.flow_keyword = "flow"
        self.transport_keyword = "transport"
        self.implicit_fracture_ = True

        self.param = param
        self.time_step_param = param['time_step_param']

        self.max_time_step = self.time_step_param['max_dt']
        T = self.time_step_param["end_time"]
        self.vtk_export_all = self.time_step_param.get("export_vtk_all_timesteps", False)
        t_init = self.time_step_param["initial_dt"] * 10
        self.vtk_times_ = np.logspace(np.log10(t_init), np.log10(T), 100)
        self.vtk_times_ = np.r_[self.vtk_times_, np.inf]
        self.next_vtk_time = -1

        self.prev_vtk_time = -np.infty

        self.create_gb()

        self.add_data()

    # ------------------------------------------------------------------------ #

    def create_gb(self, mesh_args=None):
        """ Load fractures and create grid bucket
        """
        if mesh_args is not None:
            self.mesh_args = mesh_args

        nx = self.mesh_args["mesh_size"]
        physdims = self.mesh_args["physdims"]

        lg = pp.CartLeafGrid(nx, physdims, self.max_grid_level + 1)
        self.gb = pp.GridBucket()
        self.gb.add_nodes(lg)
        self.gb.assign_node_ordering()
        #        self.gb = pp.meshing.cart_grid([], nx, physdims=physdims)
        # Center grid at x=0
        for g, _ in self.gb:
            g.nodes[0] -= physdims[0] / 2
        self.gb.compute_geometry()
        for level_grid in lg.level_grids:
            level_grid.nodes[0] -= physdims[0] / 2
            level_grid.compute_geometry()

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
        assign_periodic_bc(self.gb, "ymin")

    def frame_velocity(self):
        return np.array([1.0, 0.0, 0.0])

    def viscosity(self, c):
        """ Return the viscosity as a function of temperature
        """
        return pp.ad.exp(-self.param["R"] * c)

    def add_data(self):
        """ Add data to the GridBucket
        """
        self.add_flow_data()
        self.add_transport_data()

    def initial_concentration(self):
        t0 = 1e-5
        x = self.gb.cell_centers()[0]
        pert = 1e-4 * np.random.rand(x.size) * np.exp(-(x ** 2) / t0)
        c = 0.5 + 0.5 * sp.special.erf(-x / np.sqrt(t0)) + pert
        return c

    def add_flow_data(self):
        """ Add the flow data to the grid bucket
        """
        keyword = self.flow_keyword
        # Iterate over nodes and assign data
        for g, d in self.gb:
            param = {}
            # Shorthand notation
            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            # Specific volume.
            specific_volume = np.power(
                self.param["aperture"], self.gb.dim_max() - g.dim
            )
            param["specific_volume"] = specific_volume
            # Tangential permeability
            if g.dim == self.gb.dim_max():
                kxx = self.param["km"]
            else:
                kxx = self.param["kf"] * specific_volume

            perm = pp.SecondOrderTensor(kxx * unity)
            param["second_order_tensor"] = perm

            # Source term
            param["source"] = zeros

            # Boundaries
            bound_faces = g.get_boundary_faces()
            bc_val = np.zeros(g.num_faces)

            face_centers = g.face_centers
            # Find faces on right and left side of domain
            east = face_centers[0, :] > self.domain["xmax"] - self.tol
            west = face_centers[0, :] < self.domain["xmin"] + self.tol
            north = face_centers[1, :] > self.domain["ymax"] - self.tol
            south = face_centers[1, :] < self.domain["ymin"] + self.tol
            # Add dirichlet condition to left and right faces
            # Add boundary condition values
            area = g.face_areas[west]
            bc_val[west] = -1.0 * area * kxx

            param["bc"] = pp.BoundaryCondition(g, east, "dir")

            param["bc_values"] = bc_val

            pp.initialize_data(g, d, keyword, param)

        # Loop over edges and set coupling parameters
        for e, d in self.gb.edges():
            # Get higher dimensional grid
            g_h = self.gb.nodes_of_edge(e)[1]
            param_h = self.gb.node_props(g_h, pp.PARAMETERS)
            mg = d["mortar_grid"]
            specific_volume_h = (
                np.ones(mg.num_cells) * param_h[keyword]["specific_volume"]
            )
            kn = self.param["kn"] * specific_volume_h / (self.param["aperture"] / 2)
            param = {"normal_diffusivity": kn}
            pp.initialize_data(e, d, keyword, param)


    def add_transport_data(self):
        """ Add the transport data to the grid bucket
        """
        keyword = self.transport_keyword
        self.gb.add_node_props(["param", "is_tangential"])

        for g, d in self.gb:
            param = {}
            d["is_tangential"] = True

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            # Specific volume.
            specific_volume = np.power(
                self.param["aperture"], self.gb.dim_max() - g.dim
            )
            param["specific_volume"] = specific_volume
            # Tangential diffusivity
            if g.dim == self.gb.dim_max():
                kxx = self.param["Dm"] * unity
            else:
                kxx = self.param["Df"] * specific_volume * unity
            perm = pp.SecondOrderTensor(kxx)
            param["second_order_tensor"] = perm

            # Source term
            param["source"] = zeros

            # Mass weight
            param["mass_weight"] = specific_volume * unity

            # Boundaries
            bound_faces = g.get_boundary_faces()
            bc_val = np.zeros(g.num_faces)
            if bound_faces.size == 0:
                param["bc"] = pp.BoundaryCondition(g, empty, empty)
            else:
                bound_face_centers = g.face_centers[:, bound_faces]

                west = bound_face_centers[0, :] < self.domain["xmin"] + self.tol
                east = bound_face_centers[0, :] > self.domain["xmax"] - self.tol
                north = bound_face_centers[1, :] > self.domain["ymax"] - self.tol
                south = bound_face_centers[1, :] < self.domain["ymin"] + self.tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[west + east] = "dir"

                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[west]] = 1

                param["bc"] = pp.BoundaryCondition(g, bound_faces, labels)

            param["bc_values"] = bc_val

            # Add theta value
            if self.implicit_fracture_ and g.dim < 2:
                theta = 1 * unity
            else:
                theta = 0.5 * unity
            param["theta"] = theta
            pp.initialize_data(g, d, keyword, param)

        # Normal diffusivity
        for e, d in self.gb.edges():
            # Get higher dimensional grid
            g_h = self.gb.nodes_of_edge(e)[1]
            param_h = self.gb.node_props(g_h, pp.PARAMETERS)
            mg = d["mortar_grid"]
            specific_volume_h = (
                np.ones(mg.num_cells) * param_h[keyword]["specific_volume"]
            )
            dn = self.param["Dn"] * specific_volume_h / (self.param["aperture"] / 2)
            param = {"normal_diffusivity": dn}
            pp.initialize_data(e, d, keyword, param)

    def write_vtk_for_time(self, t, k):
        if self.vtk_export_all:
            return True
        elif t - self.next_vtk_time > 0 - self.tol:
            self.next_vtk_time = self.vtk_times_[0]
            self.prev_vtk_time = t
            self.vtk_times_ = np.delete(self.vtk_times_, 0)
            return True
        elif np.abs(t - self.time_step_param['end_time']) < 1e-10:
            self.prev_vtk_time = t
            return True
        else:
            return False

    def adapt_domain(self, t, k):
        return k % 2 == 0

    def extend_domain(self, c):
        xmin = np.min(self.gb.cell_centers()[0])
        xmax = np.max(self.gb.cell_centers()[0])
        xvar_min = np.min(self.gb.cell_centers()[0, c < 0.99])
        xvar_max = np.max(self.gb.cell_centers()[0, c > 0.01])

        L = xmax - xmin
        if (xvar_min - xmin) / L < 0.2 or (xmax - xvar_max) / L < 0.2:
            return True
        else:
            return False

    def shift_domain(self, c):
        return False

    def post_shift(self):
        return False
    
    def sugguest_time_step(self, t, dt):
        dt = min(dt, self.time_step_param['max_dt'])
        T = self.time_step_param['end_time']
        if np.abs(t - T) > dt / 100:
            dt = min(dt, T - t)
        return dt

    def coarsen_domain(self, c):
        if self.gb.num_cells() < self.mesh_args["max_number_of_cells"]:
            return c

        nx_old = self.mesh_args["mesh_size"].copy()
        nx_min = 1
        ny_min = 1
        nx_new = [nx_old[0] // 2, nx_old[1] // 2]

        if nx_new[0] < nx_min or nx_new[1] < ny_min:
            return c
        if not np.isclose(nx_new[0], nx_old[0] / 2):
            return c
        if not np.isclose(nx_new[1], nx_old[1] / 2):
            return c

        cell_volumes_old = []
        for g, _ in self.gb:
            cell_volumes_old.append(g.cell_volumes)

        self.mesh_args["mesh_size"] = nx_new
        self.create_gb()
        i = 0
        old2new_glob = []
        for g, _ in self.gb:
            if g.dim == 2:
                indices = np.arange(cell_volumes_old[i].size)
                indices = indices.reshape(nx_old, order='F')
                indices_first = indices[:, ::2].ravel('F')
                indices_second = indices[:, 1::2].ravel('F')
                indices = np.ravel((indices_first, indices_second),'F')
#                data = cell_volumes_old[i][indices]
                data = np.ones(indices.size, dtype=bool)
                indptr = np.arange(0, nx_old[0] * nx_old[1] + 1, 4)
                old2newMap = sps.csr_matrix((data, indices, indptr))
                weight = (old2newMap.T * (1.0 / g.cell_volumes))
                data_weighted = cell_volumes_old[i][indices] * weight[indices]
                old2new = sps.csr_matrix((data_weighted, indices, indptr))

            if g.dim == 1:
                indices = np.arange(cell_volumes_old[i].size)
                data = cell_volumes_old[i][indices]
                indptr = np.arange(0, cell_volumes_old[i].size + 1, 2)
                old2new = sps.csr_matrix((data, indices, indptr))
                old2new.data = old2new.T * (1.0 / g.cell_volumes)
            if g.dim == 0:
                old2new = sps.identity(g.num_cells)

            old2new_glob.append(old2new)
            i += 1

        adapt_dt = self.time_step_param.get("adapt_dt_on_grid_coarsening", False)
        if adapt_dt:
            self.time_step_param["max_dt"] *= 2

        return sps.block_diag(old2new_glob) * c

    def post_adaption(self, c):
        return self.coarsen_domain(c)




def assign_periodic_bc(gb, side, tol=1e-6):
    xmax = np.max([np.max(g.nodes[0, :]) for g, _ in gb if g.dim > 0])
    ymax = np.max([np.max(g.nodes[1, :]) for g, _ in gb if g.dim > 0])

    for g, d in gb:
        if hasattr(g, "level_grids"):
            for lg in g.level_grids:
                tag_periodic_bc_grid(lg, side, xmax, ymax, tol)
        tag_periodic_bc_grid(g, side, xmax, ymax, tol)

    for dim in [2, 1]:
        grids = gb.grids_of_dimension(dim)
        if len(grids) == 0:
            continue
        assert len(grids) == 1
        g = grids[0]
        if hasattr(g, "level_grids"):
            for lg in g.level_grids:
                match_periodic_bc(lg)
        match_periodic_bc(g)

    for g, _ in gb:
        if hasattr(g, "level_grids"):
            for lg in g.level_grids:
                del lg.face_centers_periodic
        del g.face_centers_periodic


def match_periodic_bc(g):
    if g.left.size < 1:
        g.set_periodic_map(np.zeros((2, 0), dtype=int))
        return
    left, right = match_grids(g, g)
    g.set_periodic_map(np.vstack((left, right)))


def tag_periodic_bc_grid(g, side, xmax, ymax, tol):
    # map right faces to left
    g.face_centers_periodic = g.face_centers.copy()
    if side == "xmin" or side == "west":
        left = np.argwhere(g.face_centers[0] < tol).ravel()
        right = np.argwhere(g.face_centers[0] > xmax - tol).ravel()
        g.face_centers_periodic[0, right] = 0
    elif side == "xmax" or side == "east":
        left = np.argwhere(g.face_centers[0] > xmax - tol).ravel()
        right = np.argwhere(g.face_centers[0] < tol).ravel()
        g.face_centers_periodic[0, right] = xmax
    elif side == "ymin" or side == "south":
        left = np.argwhere(g.face_centers[1] < tol).ravel()
        right = np.argwhere(g.face_centers[1] > ymax - tol).ravel()
        g.face_centers_periodic[1, right] = 0
    elif side == "ymax" or side == "north":
        left = np.argwhere(g.face_centers[1] > ymax - tol).ravel()
        right = np.argwhere(g.face_centers[1] < tol).ravel()
        g.face_centers_periodic[1, right] = ymax
    else:
        raise ValueError("Unknow face side: " + side)
    g.left = left
    g.right = right


def match_grids(gi, gj):
    tol = 1e-6
    fi = gi.face_centers_periodic[:, gi.left]
    fj = gj.face_centers_periodic[:, gj.right]

    left_right_faces = np.argwhere(np.abs(spsd.cdist(fi[:2].T, fj[:2].T) < tol))
    left = gi.left[left_right_faces[:, 0]]
    right = gj.right[left_right_faces[:, 1]]

    assert left.size == np.unique(left).size
    assert right.size == np.unique(right).size

    return left, right
