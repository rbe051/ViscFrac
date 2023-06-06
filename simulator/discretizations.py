"""
The module contains the discretizations needed for the viscous flow
problem. It contains a class ViscousFlow that is the main access point
to the discretizations, and it has functions for discretizing and assembling
the needed discretization matrices. The ViscouFlow class is the one that should
be supplied to the models.viscous_flow function.


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

import viscous_porepy as pp
import numpy as np
import scipy.sparse as sps

import viscous_porepy.ad as ad
from utils import projection
from utils import linear_solvers


class ViscousFlow(object):
    """
    Class for discretizing the operators needed to formulate the viscous flow
    problem.
    """

    def __init__(self, problem, use_mpfa=False, linear_solver="umfpack"):
        """
        Initialize data and discretize.
        """
        self._use_mpfa = use_mpfa
        self.problem = problem
        self.mat = {}
        self.proj = {}
        self.geo = {}
        self.discretize()
        self.calculate_projections()
        self.lin_solv = linear_solver

    def linear_solver(self, A, b):
        """
        Solves a linear system Ax = b and returns the solution x.

        Parameters
        ----------
        A : Matrix
        b : Array

        Raises
        ------
        ValueError
            If spesified linear solver is unknown.

        Returns
        -------
        Array
            Solution to linear system.

        """
        if self.lin_solv=="umfpack":
            return linear_solvers.umfpack(A, b)
        elif self.lin_solv=="pardiso":
            return linear_solvers.pardiso(A, b)
        elif self.lin_solv=="superlu":
            return linear_solvers.superlu(A, b)
        else:
            raise ValueError("Unknown linear solver {}".format(self.linear_solver))
    
    def update_problem(self, problem):
        """
        Update the problem. Should be called after e.g. the domain is adapted

        Parameters
        ----------
        problem : BaseData
            see cases.base_problem.

        Returns
        -------
        None.

        """
        self.problem = problem
        self.mat = {}
        self.proj = {}
        self.geo = {}
        self.discretize()
        self.calculate_projections()

    def discretize(self):
        """
        Discretize all operators
        """
        self.geo["mortar_volumes"] = self.problem.gb.cell_volumes_mortar()
        self.discretize_flow()
        self.discretize_transport()

    def calculate_projections(self):
        """
        Calculate mortar projections between different domains.
        """
        gb = self.problem.gb
        mstr2mrtr, slv2mrtr, mrtr2mstr, mrtr2slv = projection.mixed_dim_projections(gb)
        avg = projection.cells2faces_avg(gb)
        div = projection.faces2cells(gb)

        self.proj["primary2mortar"] = mstr2mrtr
        self.proj["secondary2mortar"] = slv2mrtr
        self.proj["mortar2primary"] = mrtr2mstr
        self.proj["mortar2secondary"] = mrtr2slv
        self.proj["avg"] = avg
        self.proj["div"] = div

    def viscosity(self):
        """
        Returns
        -------
        Function
            Function describing the viscosity of the fluid.

        """
        return self.problem.viscosity

    def discretize_flow(self):
        """
        Discretize the flow operators
        """
        gb = self.problem.gb
        flow_kw = self.problem.flow_keyword
        elliptic_disc(gb, flow_kw, self._use_mpfa)
        # Assemble matrices
        assembler = pp.Assembler(gb)

        # Fluid flow. Darcy + mass conservation
        flux = assembler.assemble_operator(flow_kw, "flux")
        bound_flux = assembler.assemble_operator(flow_kw, "bound_flux")
        trace_cell = assembler.assemble_operator(flow_kw, "bound_pressure_cell")
        trace_face = assembler.assemble_operator(flow_kw, "bound_pressure_face")

        # Assemble discrete parameters and geometric values
        bc_val = assembler.assemble_parameter(flow_kw, "bc_values")
        kn = [d[pp.PARAMETERS][flow_kw]["normal_diffusivity"] for _, d in gb.edges()]
        if len(kn) > 0:
            kn = np.hstack(kn)
        else:
            kn = np.array([], dtype=float)

        # assemble moving reference frame
        i_x = self.problem.frame_velocity()
        for g, d in gb:
            cross_sec_area = d[pp.PARAMETERS][flow_kw]["specific_volume"]
            d[pp.PARAMETERS][flow_kw]["i_x"] = np.dot(i_x, g.face_normals)
            d[pp.PARAMETERS][flow_kw]["i_x"] *= np.max(cross_sec_area)
        i_x = assembler.assemble_parameter(flow_kw, "i_x")

        self.mat[flow_kw] = {
            "flux": flux,
            "bound_flux": bound_flux,
            "trace_cell": trace_cell,
            "trace_face": trace_face,
            "bc_values": bc_val,
            "kn": kn,
            "i_x": i_x,
        }

    def discretize_transport(self):
        """
        Discretize the transport operators
        """
        gb = self.problem.gb
        transport_kw = self.problem.transport_keyword
        elliptic_disc(gb, transport_kw, self._use_mpfa)
        upwind_disc(gb, transport_kw)

        # Assemble global matrices
        assembler = pp.Assembler(gb)

        # Transport. Upwind + mass conservation
        diff = assembler.assemble_operator(transport_kw, "flux")
        bound_diff = assembler.assemble_operator(transport_kw, "bound_flux")
        trace_cell = assembler.assemble_operator(transport_kw, "bound_pressure_cell")
        trace_face = assembler.assemble_operator(transport_kw, "bound_pressure_face")

        pos_cells = assembler.assemble_operator(transport_kw, "pos_cells")
        neg_cells = assembler.assemble_operator(transport_kw, "neg_cells")

        # Assemble discrete parameters and geometric values
        bc_val = assembler.assemble_parameter(transport_kw, "bc_values")
        bc_sgn = assembler.assemble_parameter(transport_kw, "bc_sgn")
        frac_bc = assembler.assemble_parameter(transport_kw, "frac_bc")

        mass_weight = assembler.assemble_parameter(transport_kw, "mass_weight")
        theta = assembler.assemble_parameter(transport_kw, "theta")
        dn = [
            d[pp.PARAMETERS][transport_kw]["normal_diffusivity"] for _, d in gb.edges()
        ]
        if len(dn) > 0:
            dn = np.hstack(dn)
        else:
            dn = np.array([], dtype=float)

        # Store  discretizations
        self.mat[transport_kw] = {
            "flux": diff,
            "bound_flux": bound_diff,
            "trace_cell": trace_cell,
            "trace_face": trace_face,
            "bc_values": bc_val,
            "pos_cells": pos_cells,
            "neg_cells": neg_cells,
            "bc_sgn": bc_sgn,
            "frac_bc": frac_bc,
            "dn": dn,
            "mass_weight": mass_weight * gb.cell_volumes(),
            "theta": theta,
        }

    def upwind(self, c, q):
        """
        Get the upwind weights for a flux field q
        """
        kw = self.problem.transport_keyword
        if isinstance(q, ad.Ad_array):
            q_flux = q.val
        else:
            q_flux = q

        flag = (q_flux > 0).astype(np.int)
        bc_flag = (self.mat[kw]["bc_sgn"] * q_flux < 0).astype(np.int)
        # The coupling flux is added by function mortar_upwind
        not_frac_flag = ~self.mat[kw]["frac_bc"]
        pos_flag = sps.diags(flag * not_frac_flag, dtype=np.int)
        neg_flag = sps.diags((1 - flag) * not_frac_flag, dtype=np.int)

        pos_cells = self.mat[kw]["pos_cells"]
        neg_cells = self.mat[kw]["neg_cells"]
        T_upw = (pos_flag * pos_cells + neg_flag * neg_cells) * c
        return (T_upw + self.mat[kw]["bc_values"] * bc_flag)

    def mortar_upwind(
        self, c, lam, div, avg, primary2mortar, secondary2mortar, mortar2primary, mortar2secondary
    ):
        """
        Get the upwind weights between dimensions
        """
        if isinstance(lam, ad.Ad_array):
            lam_flux = lam.val
        else:
            lam_flux = lam

        # Upwind coupling between dimensions:
        flag_m = (lam_flux > 0).astype(np.int)

        primary_flag = sps.diags(flag_m, dtype=np.int)
        secondary_flag = sps.diags(1 - flag_m, dtype=np.int)

        # Outflow of primary and secondary
        out_primary = (avg * c) * (mortar2primary * primary_flag * lam)
        out_secondary = c * (mortar2secondary * secondary_flag * lam)
        # What flows out of primary/secondary flows into secondary/primary
        inn_secondary = mortar2secondary * (primary2mortar * avg * c * (primary_flag * lam))
        inn_primary = mortar2primary * (secondary2mortar * c * (secondary_flag * lam))

        upwind_primary = np.abs(div) * (out_primary + inn_primary)
        upwind_secondary = out_secondary + inn_secondary

        return upwind_primary - upwind_secondary


def elliptic_disc(gb, keyword, use_mpfa=False):
    """
    Discretize the elliptic operator on each graph node
    """
    for g, d in gb:
        if use_mpfa:
            pp.Mpfa(keyword).discretize(g, d)
        else:
            pp.Tpfa(keyword).discretize(g, d)
        d[pp.DISCRETIZATION_MATRICES][keyword]["div"] = pp.fvutils.scalar_divergence(g)


def upwind_disc(gb, keyword):
    """
    Discretize the upwind operator on each graph node
    """
    for g, d in gb:
        pos_cells = g.cell_faces.copy()
        neg_cells = g.cell_faces.copy()
        # Add periodic bc
        left = g.periodic_face_map[0]
        right = g.periodic_face_map[1]

        left_fi, left_ci, left_sgn = sps.find(g.cell_faces[left])
        right_fi, right_ci, right_sgn = sps.find(g.cell_faces[right])
        shape = [g.num_faces, g.num_cells]
        IA = np.argsort(left_fi)
        left_fi, left_ci, left_sgn = left_fi[IA], left_ci[IA], left_sgn[IA]
        IA = np.argsort(right_fi)
        right_fi, right_ci, right_sgn = right_fi[IA], right_ci[IA], right_sgn[IA]

        # import matplotlib.pyplot as plt
        # for c in range(left_ci.size):
        #     pp.plot_grid(g, if_plot=False,alpha=0)
        #     upstream = left_ci[c]
        #     downstream = right_ci[c]
        #     face_up = left[left_fi[c]]
        #     face_dw = right[right_fi[c]]

        #     plt.plot([g.cell_centers[0, upstream]], [g.cell_centers[1, upstream]], 'r.',markersize=5)
        #     plt.plot([g.cell_centers[0, downstream]], [g.cell_centers[1, downstream]], 'b.',markersize=5)
        #     plt.plot([g.face_centers[0, face_up]], [g.face_centers[1, face_up]], 'rx',markersize=5)
        #     plt.plot([g.face_centers[0, face_dw]], [g.face_centers[1, face_dw]], 'bx',markersize=5)
        #     plt.show()

        left_to_right = sps.coo_matrix(
            (right_sgn, (left[left_fi], right_ci)), shape=shape
        )
        right_to_left = sps.coo_matrix(
            (left_sgn, (right[right_fi], left_ci)), shape=shape
        )

        pos_cells = pos_cells + left_to_right + right_to_left
        neg_cells = neg_cells + left_to_right + right_to_left

        pos_cells.data = pos_cells.data.clip(min=0)
        neg_cells.data = -neg_cells.data.clip(max=0)
        d[pp.DISCRETIZATION_MATRICES][keyword]["pos_cells"] = pos_cells
        d[pp.DISCRETIZATION_MATRICES][keyword]["neg_cells"] = neg_cells

        # Get sign of boundary
        bc_sgn = np.zeros(g.num_faces)
        bc_sgn[g.get_boundary_faces()] = _sign_of_boundary_faces(g)
        d[pp.PARAMETERS][keyword]["bc_sgn"] = bc_sgn
        d[pp.PARAMETERS][keyword]["frac_bc"] = g.tags["fracture_faces"]


def mass_matrix(gb, keyword):
    """
    Discretize the mass matrix on each graph node
    """
    for g, d in gb:
        volumes = d[pp.PARAMETERS][keyword]["mass_weight"] * g.cell_volumes
        d[pp.DISCRETIZATION_MATRICES][keyword]["mass_matrix"] = sps.diags(volumes)


def mortar_weight(gb, keyword):
    """
    Discretize the mortar coupling on each graph edge
    """
    for e, d in gb.edges():
        gs, gm = gb.nodes_of_edge(e)
        if gs == gm:
            W = sps.csc_matrix((d["mortar_grid"].num_cells, d["mortar_grid"].num_cells))
        else:
            Dn = d[pp.PARAMETERS][keyword]["normal_diffusivity"]
            W = sps.eye(d["mortar_grid"].num_cells) / Dn
        d[pp.DISCRETIZATION_MATRICES][keyword]["mortar_weight"] = W


def mortar_projections(gb, keyword):
    """
    Obtain projections between mortar grids, secondary grids and primary grids.
    """
    for e, d in gb.edges():
        gs, gm = gb.nodes_of_edge(e)
        if gs.dim == gm.dim:
            d[pp.DISCRETIZATION_MATRICES][keyword]["mortar2secondary_face"] = d[
                "mortar_grid"
            ].mortar_to_secondary_avg()
            d[pp.DISCRETIZATION_MATRICES][keyword]["mortar2primary_face"] = d[
                "mortar_grid"
            ].mortar_to_primary_avg()
            d[pp.DISCRETIZATION_MATRICES][keyword][
                "mortar2secondary_cell"
            ] = sps.csc_matrix((gs.num_cells, d["mortar_grid"].num_cells))
            d[pp.DISCRETIZATION_MATRICES][keyword][
                "mortar2primary_cell"
            ] = sps.csc_matrix((gm.num_cells, d["mortar_grid"].num_cells))
        elif gs.dim <= gm.dim:
            d[pp.DISCRETIZATION_MATRICES][keyword][
                "mortar2secondary_face"
            ] = sps.csc_matrix((gs.num_faces, d["mortar_grid"].num_cells))
            d[pp.DISCRETIZATION_MATRICES][keyword]["mortar2primary_face"] = d[
                "mortar_grid"
            ].mortar_to_primary_avg()
            d[pp.DISCRETIZATION_MATRICES][keyword]["mortar2secondary_cell"] = d[
                "mortar_grid"
            ].mortar_to_secondary_avg()
            d[pp.DISCRETIZATION_MATRICES][keyword][
                "mortar2primary_cell"
            ] = sps.csc_matrix((gm.num_cells, d["mortar_grid"].num_cells))
        elif gs.dim >= gm.dim:
            d[pp.DISCRETIZATION_MATRICES][keyword]["mortar2secondary_face"] = d[
                "mortar_grid"
            ].mortar_to_secondary_avg()
            d[pp.DISCRETIZATION_MATRICES][keyword][
                "mortar2primary_face"
            ] = sps.csc_matrix((gm.num_faces, d["mortar_grid"].num_cells))
            d[pp.DISCRETIZATION_MATRICES][keyword][
                "mortar2secondary_cell"
            ] = sps.csc_matrix((gs.num_cells, d["mortar_grid"].num_cells))
            d[pp.DISCRETIZATION_MATRICES][keyword]["mortar2primary_cell"] = d[
                "mortar_grid"
            ].mortar_to_primary_avg()


def _sign_of_boundary_faces(g):
    """
    returns the sign of boundary faces as defined by g.cell_faces.
    Parameters:
    g: (Grid Object)
    Returns:
    sgn: (ndarray) the sign of the faces
    """
    faces = g.get_boundary_faces()

    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn = sps.find(g.cell_faces[faces[IA], :])
    assert fi.size == faces.size, "sign of internal faces does not make sense"
    I = np.argsort(fi)
    sgn = sgn[I]
    sgn = sgn[IC]
    return sgn
