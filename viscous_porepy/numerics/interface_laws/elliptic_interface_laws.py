"""
Coupling conditions between subdomains for elliptic equations.

Current content:
    Robin-type couplings, as decsribed by Martin et al 2005.
    Full continuity conditions between subdomains
"""
from typing import Tuple

import numpy as np
import scipy.sparse as sps

import viscous_porepy as pp
import viscous_porepy.numerics.interface_laws.abstract_interface_law


class RobinCoupling(
    pp.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    """A condition with resistance to flow between subdomains. Implementation
    of the model studied (though not originally proposed) by Martin et
    al 2005.

    """

    def __init__(self, keyword, discr_primary, discr_secondary=None):
        super(RobinCoupling, self).__init__(keyword)
        if discr_secondary is None:
            discr_secondary = discr_primary
        self.discr_primary = discr_primary
        self.discr_secondary = discr_secondary

        # This interface law will have direct interface coupling to represent
        # the influence of the flux boundary condition of the secondary
        # interface on the pressure trace on the first interface.
        self.edge_coupling_via_high_dim = True
        # No coupling via lower-dimensional interfaces.
        self.edge_coupling_via_low_dim = False

        # Keys used to identify the discretization matrices of this discretization
        self.mortar_discr_key = "robin_mortar_discr"
        self.mortar_vector_source_key = "robin_vector_source_discr"
        self.mortar_scaling_key = "mortar_scaling"

        # Decide on whether to scale the mortar flux with K^-1 or not.
        # This is the scaling of Darcy's law in mixed methods, and should be used in the
        # interface law if the full system is on mixed form.
        # We decide on this based on whether both neigboring discretizations are mixed
        # or not. This leaves the case when one neighbor is mixed, the other is FV; in
        # this case, we use a K-scaling, but it is not clear what is best.
        if isinstance(
            discr_primary, pp.numerics.vem.dual_elliptic.DualElliptic
        ) and isinstance(discr_secondary, pp.numerics.vem.dual_elliptic.DualElliptic):
            self.kinv_scaling = True
        else:
            # At least one of the neighboring discretizations is FV.
            self.kinv_scaling = False

    def ndof(self, mg):
        return mg.num_cells

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """Discretize the interface law and store the discretization in the
        edge data.

        Parameters:
            g_h: Grid of the primary domanin.
            g_l: Grid of the secondary domain.
            data_h: Data dictionary for the primary domain.
            data_l: Data dictionary for the secondary domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]
        parameter_dictionary_h = data_h[pp.PARAMETERS][self.discr_primary.keyword]
        # Mortar data structure.
        mg = data_edge["mortar_grid"]

        kn = parameter_dictionary_edge["normal_diffusivity"]
        # If normal diffusivity is given as a constant, parse to np.array
        if not isinstance(kn, np.ndarray):
            kn *= np.ones(mg.num_cells)

        inv_M = sps.diags(1.0 / mg.cell_volumes)
        inv_k = 1.0 / kn
        Eta = sps.diags(inv_k)
        matrix_dictionary_edge[self.mortar_discr_key] = -inv_M * Eta

        ## Vector source.
        # This contribution is last term of
        # lambda = -\int{\kappa_n [p_l - p_h +  a/2 g \cdot n]} dV,
        # where n is the outwards normal and the integral is taken over the mortar cell.
        # (Note: This assumes a P0 discretization of mortar fluxes).

        # Ambient dimension of the problem, as specified for the higher-dimensional
        # neighbor.
        # IMPLEMENTATION NOTE: The default value is needed to avoid that
        # ambient_dimension becomes a required parameter. If neither ambient dimension,
        # nor the actual vector_source is specified, there will be no problems (in the
        # assembly, a zero vector soucre of a size that fits with the discretization is
        # created). If a vector_source is specified, but the ambient dimension is not,
        # a dimension mismatch will result unless the ambient dimension implied by
        # the size of the vector source matches g_h.dim. This is okay for domains with
        # no subdomains with co-dimension more than 1, but will fail for fracture
        # intersections. The default value is thus the least bad option in this case.
        vector_source_dim: int = parameter_dictionary_h.get(
            "ambient_dimension", g_h.dim
        )
        # The ambient dimension cannot be less than the dimension of g_h.
        # If this is broken, we risk ending up with zero normal vectors below, so it is
        # better to break this off now
        if vector_source_dim < g_h.dim:
            raise ValueError(
                "Ambient dimension cannot be lower than the grid dimension"
            )

        # Construct the dot product between normals on fracture faces and the identity
        # matrix.

        # Find the mortar normal vectors by projection of the normal vectors in g_h
        normals_h = g_h.face_normals.copy()

        # projection matrix
        proj = mg.primary_to_mortar_avg()

        # Ensure that the normal vectors point out of g_h
        # Indices of faces neighboring this mortar grid
        _, fi_h, _ = sps.find(proj)
        # Switch direction of vectors if relevant
        normals_h[:, fi_h] *= g_h.sign_of_faces(fi_h)

        # Project the normal vectors, we need to do some transposes to get this right
        normals_mortar = (proj * normals_h.T).T
        nrm = np.linalg.norm(normals_mortar, axis=0)
        # Sanity check
        assert np.all(nrm > 1e-10)
        outwards_unit_mortar_normals = normals_mortar / nrm

        # We know that the ambient dimension for the vector source must be at least as
        # high as g_h, thus taking the first vector_source_dim components of the normal
        # vector should be fine.
        vals = outwards_unit_mortar_normals[:vector_source_dim].ravel("f")

        # The values in vals are sorted by the mortar cell index ordering (proj is a
        # csr matrix).
        ci_mortar = np.arange(mg.num_cells, dtype=np.int)

        # The mortar cell indices are expanded to account for the vector source
        # having multiple dimensions
        rows = np.tile(ci_mortar, (vector_source_dim, 1)).ravel("f")
        # Columns must account for the values being vector values.
        cols = pp.fvutils.expand_indices_nd(ci_mortar, vector_source_dim)

        # And we have the normal vectors
        mortar_normals = sps.coo_matrix((vals, (rows, cols))).tocsr()

        # On assembly, the outwards normals on the mortars will be multiplied by the
        # interface vector source.
        matrix_dictionary_edge[self.mortar_vector_source_key] = mortar_normals
        if self.kinv_scaling:
            # Use a discretization fit for mixed methods, with a K^-1 scaling of the
            # mortar flux
            # In this case, the scaling of the pressure blocks on the mortar rows is
            # simple.
            matrix_dictionary_edge[self.mortar_scaling_key] = sps.diags(
                np.ones(mg.num_cells)
            )

        else:
            # Scale the the mortar equations with K, so that the this becomes a
            # Darcy-type equation on standard form.
            matrix_dictionary_edge[self.mortar_scaling_key] = sps.diags(
                mg.cell_volumes * kn
            )

    def assemble_matrix_rhs(
        self, g_primary, g_secondary, data_primary, data_secondary, data_edge, matrix
    ):
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        Parameters:
            g_primary: Grid on one neighboring subdomain.
            g_secondary: Grid on the other neighboring subdomain.
            data_primary: Data dictionary for the primary suddomain
            data_secondary: Data dictionary for the secondary subdomain.
            data_edge: Data dictionary for the edge between the subdomains.
                If gravity is taken into consideration, the parameter sub-
                dictionary should contain something like a/2 * g, where
                g is the ambient_dimension x n_mortar_cells gravity vector
                as used in Starnoni et al 2020, typically with
                    g[ambient_dimension]= -G * rho.
            matrix: original discretization

            The discretization matrices must be included since they will be
            changed by the imposition of Neumann boundary conditions on the
            internal boundary in some numerical methods (Read: VEM, RT0)

        """
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]
        mg = data_edge["mortar_grid"]

        primary_ind = 0
        secondary_ind = 1
        cc, rhs = self._define_local_block_matrix(
            g_primary, g_secondary, self.discr_primary, self.discr_secondary, mg, matrix
        )

        # The convention, for now, is to put the higher dimensional information
        # in the first column and row in matrix, lower-dimensional in the second
        # and mortar variables in the third
        cc[2, 2] = matrix_dictionary_edge[self.mortar_discr_key]

        self.discr_primary.assemble_int_bound_pressure_trace(
            g_primary, data_primary, data_edge, cc, matrix, rhs, primary_ind
        )
        self.discr_primary.assemble_int_bound_flux(
            g_primary, data_primary, data_edge, cc, matrix, rhs, primary_ind
        )

        self.discr_secondary.assemble_int_bound_pressure_cell(
            g_secondary, data_secondary, data_edge, cc, matrix, rhs, secondary_ind
        )
        self.discr_secondary.assemble_int_bound_source(
            g_secondary, data_secondary, data_edge, cc, matrix, rhs, secondary_ind
        )
        # Also assemble vector sources.
        # Discretization of the vector source term
        vector_source_discr: sps.spmatrix = matrix_dictionary_edge[
            self.mortar_vector_source_key
        ]
        # The vector source, defaults to zero if not specified.
        vector_source: np.ndarray = parameter_dictionary_edge.get(
            "vector_source", np.zeros(vector_source_discr.shape[1])
        )
        if vector_source_discr.shape[1] != vector_source.size:
            # If this happens chances are that either the ambient dimension was not set,
            # and thereby its default value was used. Another not unlikely reason is
            # that the ambient dimension is set, but with a value that does not match
            # the specified vector source.
            raise ValueError(
                """Mismatch in vector source dimensions.
                Did you forget to specify the ambient dimension?"""
            )

        rhs[2] = rhs[2] - vector_source_discr * vector_source

        for block in range(cc.shape[1]):
            # Scale the pressure blocks in the mortar problem
            cc[2, block] = (
                matrix_dictionary_edge[self.mortar_scaling_key] * cc[2, block]
            )
        rhs[2] = matrix_dictionary_edge[self.mortar_scaling_key] * rhs[2]

        matrix += cc

        self.discr_primary.enforce_neumann_int_bound(
            g_primary, data_edge, matrix, primary_ind
        )

        return matrix, rhs

    def assemble_edge_coupling_via_high_dim(
        self,
        g,
        data_grid,
        edge_primary,
        data_primary_edge,
        edge_secondary,
        data_secondary_edge,
        matrix,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Represent the impact on a primary interface of the mortar (thus boundary)
        flux on a secondary interface.

        Parameters:
            g (pp.Grid): Grid of the higher dimensional neighbor to the main interface.
            data_grid (dict): Data dictionary of the intermediate grid.
            edge_primary (tuple of grids): The grids of the primary edge
            data_edge_primary (dict): Data dictionary of the primary interface.
            edge_secondary (tuple of grids): The grids of the secondary edge.
            data_edge_secondary (dict): Data dictionary of the secondary interface.
            matrix: original discretization.

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary, secondary and mortar variable, respectively.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the primary,
                secondary and mortar variable, respectively.

        """
        mg_primary = data_primary_edge["mortar_grid"]
        mg_secondary = data_secondary_edge["mortar_grid"]

        # Normally, the projections will be pressure from the primary (high-dim node)
        # to the primary mortar, and flux from secondary mortar to primary
        proj_pressure = mg_primary.primary_to_mortar_avg()
        proj_flux = mg_secondary.mortar_to_primary_int()

        # If the primary and / or secondary mortar is a boundary mortar grid, things
        # become more complex. This probably assumes that a FluxPressureContinuity
        # discretization is applied on the relevant mortar grid.
        if edge_primary[0].dim == edge_primary[1].dim and edge_primary[0] == g:
            proj_pressure = mg_primary.secondary_to_mortar_avg()
        if edge_secondary[0].dim == edge_secondary[1].dim and edge_secondary[0] == g:
            proj_flux = mg_secondary.mortar_to_secondary_int()

        cc, rhs = self._define_local_block_matrix_edge_coupling(
            g, self.discr_primary, mg_primary, mg_secondary, matrix
        )

        # Assemble contribution between higher dimensions.
        self.discr_primary.assemble_int_bound_pressure_trace_between_interfaces(
            g, data_grid, proj_pressure, proj_flux, cc, matrix, rhs
        )
        # Scale the equations (this will modify from K^-1 to K scaling if relevant)
        matrix_dictionary_edge = data_primary_edge[pp.DISCRETIZATION_MATRICES][
            self.keyword
        ]
        for block in range(cc.shape[1]):
            # Scale the pressure blocks in the row of the primary mortar problem.
            # The secondary mortar will be treated somewhere else (handled by the
            # assembler).
            cc[1, block] = (
                matrix_dictionary_edge[self.mortar_scaling_key] * cc[1, block]
            )
        rhs[1] = matrix_dictionary_edge[self.mortar_scaling_key] * rhs[1]

        return cc, rhs


class FluxPressureContinuity(RobinCoupling):
    """A condition for flux and pressure continuity between two domains. A particular
    attention is devoted in the case if these domanins are of equal
    dimension. This can be used to specify full continuity between fractures,
    two domains or a periodic boundary condition for a single domain. The faces
    coupled by flux and pressure condition must be specified by a MortarGrid on
    a graph edge.
    For each face we will impose
    v_m = lambda
    v_s = -lambda
    p_m - p_s = 0
    where subscript m and s is for primary and secondary, v is the flux, p the pressure,
    and lambda the mortar variable.

    """

    def __init__(self, keyword, discr_primary, discr_secondary=None):
        if discr_secondary is None:
            discr_secondary = discr_primary
        self.discr_primary = discr_primary
        self.discr_secondary = discr_secondary

        # This interface law will have direct interface coupling to represent
        # the influence of the flux boundary condition of the secondary
        # interface on the pressure trace on the first interface.
        self.edge_coupling_via_high_dim = False
        # No coupling via lower-dimensional interfaces.
        self.edge_coupling_via_low_dim = False

    def discretize(self, g_h, g_l, data_h, data_l, data_edge):
        """Nothing really to do here

        Parameters:
            g_h: Grid of the primary domanin.
            g_l: Grid of the secondary domain.
            data_h: Data dictionary for the primary domain.
            data_l: Data dictionary for the secondary domain.
            data_edge: Data dictionary for the edge between the domains.

        """
        pass

    def assemble_rhs(
        self, g_primary, g_secondary, data_primary, data_secondary, data_edge, matrix
    ):
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        Parameters:
            g_primary: Grid on one neighboring subdomain.
            g_secondary: Grid on the other neighboring subdomain.
            data_primary: Data dictionary for the primary suddomain
            data_secondary: Data dictionary for the secondary subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain
            matrix_secondary: original discretization for the secondary subdomain

        """
        # IMPLEMENTATION NOTE: This function is aimed at computational savings in a case
        # where the same linear system must be solved with multiple right hand sides.
        # Compared to self.assemble_matrix_rhs(), the method short cuts parts of the
        # assembly that only give contributions to the matrix.

        primary_ind = 0
        secondary_ind = 1

        # Generate matrix for the coupling.
        mg = data_edge["mortar_grid"]
        cc_primary, rhs_primary = self._define_local_block_matrix(
            g_primary, g_secondary, self.discr_primary, self.discr_secondary, mg, matrix
        )
        # I got some problems with pointers when doing rhs_primary = rhs_secondary.copy()
        # so just reconstruct everything.
        rhs_secondary = np.empty(3, dtype=np.object)
        rhs_secondary[primary_ind] = np.zeros_like(rhs_primary[primary_ind])
        rhs_secondary[secondary_ind] = np.zeros_like(rhs_primary[secondary_ind])
        rhs_secondary[2] = np.zeros_like(rhs_primary[2])

        # If primary and secondary is the same grid, they should contribute to the same
        # row and coloumn. When the assembler assigns matrix[idx] it will only add
        # the secondary information because of duplicate indices (primary and secondary
        # is the same). We therefore write the both primary and secondary info to the
        # secondary index.
        if g_primary == g_secondary:
            primary_ind = 1
        else:
            primary_ind = 0

        self.discr_primary.assemble_int_bound_pressure_trace_rhs(
            g_primary, data_primary, data_edge, cc_primary, rhs_primary, primary_ind
        )

        if g_primary.dim == g_secondary.dim:
            rhs_secondary[2] = -rhs_secondary[2]
        rhs = rhs_primary + rhs_secondary

        return rhs

    def assemble_matrix_rhs(
        self, g_primary, g_secondary, data_primary, data_secondary, data_edge, matrix
    ):
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        Parameters:
            g_primary: Grid on one neighboring subdomain.
            g_secondary: Grid on the other neighboring subdomain.
            data_primary: Data dictionary for the primary suddomain
            data_secondary: Data dictionary for the secondary subdomain.
            data_edge: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain
            matrix_secondary: original discretization for the secondary subdomain

        """
        primary_ind = 0
        secondary_ind = 1

        # Generate matrix for the coupling.
        mg = data_edge["mortar_grid"]
        cc_primary, rhs_primary = self._define_local_block_matrix(
            g_primary, g_secondary, self.discr_primary, self.discr_secondary, mg, matrix
        )

        cc_secondary = cc_primary.copy()

        # I got some problems with pointers when doing rhs_primary = rhs_secondary.copy()
        # so just reconstruct everything.
        rhs_secondary = np.empty(3, dtype=np.object)
        rhs_secondary[primary_ind] = np.zeros_like(rhs_primary[primary_ind])
        rhs_secondary[secondary_ind] = np.zeros_like(rhs_primary[secondary_ind])
        rhs_secondary[2] = np.zeros_like(rhs_primary[2])

        # The convention, for now, is to put the primary grid information
        # in the first column and row in matrix, secondary grid in the second
        # and mortar variables in the third
        # If primary and secondary is the same grid, they should contribute to the same
        # row and column. When the assembler assigns matrix[idx] it will only add
        # the secondary information because of duplicate indices (primary and secondary
        # is the same). We therefore write the both primary and secondary info to the
        # secondary index.
        if g_primary == g_secondary:
            primary_ind = 1
        else:
            primary_ind = 0

        self.discr_primary.assemble_int_bound_pressure_trace(
            g_primary,
            data_primary,
            data_edge,
            cc_primary,
            matrix,
            rhs_primary,
            primary_ind,
        )
        self.discr_primary.assemble_int_bound_flux(
            g_primary,
            data_primary,
            data_edge,
            cc_primary,
            matrix,
            rhs_primary,
            primary_ind,
        )

        if g_primary.dim == g_secondary.dim:
            # Consider this terms only if the grids are of the same dimension, by
            # imposing the same condition with a different sign, due to the normal
            self.discr_secondary.assemble_int_bound_pressure_trace(
                g_secondary,
                data_secondary,
                data_edge,
                cc_secondary,
                matrix,
                rhs_secondary,
                secondary_ind,
                use_secondary_proj=True,
            )

            self.discr_secondary.assemble_int_bound_flux(
                g_secondary,
                data_secondary,
                data_edge,
                cc_secondary,
                matrix,
                rhs_secondary,
                secondary_ind,
                use_secondary_proj=True,
            )
            # We now have to flip the sign of some of the matrices
            # First we flip the sign of the secondary flux because the mortar flux points
            # from the primary to the secondary, i.e., flux_s = -mortar_flux
            cc_secondary[secondary_ind, 2] = -cc_secondary[secondary_ind, 2]
            # Then we flip the sign for the pressure continuity since we have
            # We have that p_m - p_s = 0.
            cc_secondary[2, secondary_ind] = -cc_secondary[2, secondary_ind]
            rhs_secondary[2] = -rhs_secondary[2]
            # Note that cc_secondary[2, 2] is fliped twice, first for pressure continuity
        else:
            # Consider this terms only if the grids are of different dimension, by
            # imposing pressure trace continuity and conservation of the normal flux
            # through the lower dimensional object.
            self.discr_secondary.assemble_int_bound_pressure_cell(
                g_secondary,
                data_secondary,
                data_edge,
                cc_secondary,
                matrix,
                rhs_secondary,
                secondary_ind,
            )

            self.discr_secondary.assemble_int_bound_source(
                g_secondary,
                data_secondary,
                data_edge,
                cc_secondary,
                matrix,
                rhs_secondary,
                secondary_ind,
            )

        # Now, the matrix cc = cc_secondary + cc_primary expresses the flux and pressure
        # continuities over the mortars.
        # cc[0] -> flux_m = mortar_flux
        # cc[1] -> flux_s = -mortar_flux
        # cc[2] -> p_m - p_s = 0

        # Computational savings: Only add non-zero components.
        # Exception: The case with equal dimension of the two neighboring grids.
        if g_primary.dim == g_secondary.dim:
            matrix += cc_primary + cc_secondary
        else:
            matrix[0, 2] += cc_primary[0, 2]
            matrix[1, 2] += cc_secondary[1, 2]
            for col in range(3):
                matrix[2, col] += cc_primary[2, col] + cc_secondary[2, col]

        rhs = rhs_primary + rhs_secondary

        self.discr_primary.enforce_neumann_int_bound(
            g_primary, data_edge, matrix, primary_ind
        )

        # Consider this terms only if the grids are of the same dimension
        if g_primary.dim == g_secondary.dim:
            self.discr_secondary.enforce_neumann_int_bound(
                g_secondary, data_edge, matrix, secondary_ind
            )

        return matrix, rhs
