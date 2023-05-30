"""
Module for handling adaptivity of the domain.
"""
import numpy as np
import scipy.sparse as sps
import viscous_porepy as pp

from utils.projection import ProjectionOperator


def adapt_domain(disc, var):
    """
    Wrapper function for handeling the adaptivity.

    Parameters
    ----------
    disc : ViscousFlow
        A discretization class. See simulator.discretization.
    var : Array
        An array of the cell-centred variables.

    Returns
    -------
    array
        The cell-centered variables for the adapted grid.
    bool
        returns True if domain is adapted, False otherwise.
    """   
    adapted_domain = False
    problem = disc.problem

    if problem.shift_domain(var):
        var, shifted = shift_domain(problem, var)
        adapted_domain = adapted_domain or shifted
    if problem.extend_domain(var):
        var = extend_domain(problem, var)
        adapted_domain = True
    
    if adapted_domain:
        var = problem.post_adaption(var)
        problem.add_data()

    return var, adapted_domain


def shift_domain(problem, var):
    """
    Function that shift the domain a quarter of the domain length
    Parameters
    ----------
    problem : BaseData
        Problem class that has inherited from BaseData. 
        See also cases.base_problem.
    var : Array
        An array of the cell-centred variables.

    Returns
    -------
    array
        The cell-centered variables for the adapted grid.
    bool
        returns True if domain is adapted, False otherwise.
    """
    xmin = problem.gb.grids_of_dimension(2)[0].nodes[0].min()
    xmax = problem.gb.grids_of_dimension(2)[0].nodes[0].max()
    L = xmax - xmin
    nx = problem.mesh_args["mesh_size"]
    frac_dx = 2 * problem.frac_pos_x
    desired_offset = L / 4
    offset = round(desired_offset / frac_dx) * frac_dx
    dx = L / nx[0]
    if not np.isclose(offset / dx, round(offset / dx)):
        return var, False
    if offset == 0:
        return var, False

    problem.domain["xmin"] += offset
    problem.domain["xmax"] += offset

    old_grid_var = []
    min_cell_size = 0.5 * dx / 2**problem.max_grid_level
    Ly = problem.domain["ymax"] - problem.domain["ymin"]
    lexsort_shift = lambda x: 0.5 * min_cell_size * x[1] / Ly
    for g, _ in problem.gb:
        projectionOperator = ProjectionOperator(g)
        Pg2l = projectionOperator.global_to_local(problem.gb, 'cell')
        local_var = Pg2l * var
        cc = g.cell_centers.copy()
        cc[0] += lexsort_shift(cc)
        local_sort = np.lexsort(cc[1::-1])
        keep_local = g.cell_centers[0] - xmin > offset + 1e-8
        old_grid_var.append(local_var[local_sort][keep_local[local_sort]])
        shift_grid(g, offset)

    for e, d in problem.gb.edges():
        mg = d['mortar_grid']
        mg.cell_centers[0] += offset
        for sg in mg.side_grids.values():
            shift_grid(sg, offset)

    problem.post_shift()
    
    i = 0
    new_var = []
    for g, _ in problem.gb:
        if g.dim==0 and (len(old_grid_var) < 3):
            new_var.append(np.zeros(g.num_cells))
            continue
        elif g.dim==1 and( len(old_grid_var) < 2):
            new_var.append(np.zeros(g.num_cells))
            continue
        nfrac_max = np.sum(g.cell_centers[0] > xmax - 1e-8)
        right_value = np.zeros(nfrac_max, dtype=float)
        new_grid_var = np.r_[old_grid_var[i], right_value]

        cc = g.cell_centers.copy()
        cc[0] += lexsort_shift(cc)
        IA = np.lexsort(cc[1::-1])
        IC = np.argsort(IA)
        new_grid_var = new_grid_var[IC]
        new_var.append(new_grid_var)
        i += 1
    return np.hstack(new_var), True


def shift_grid(g, offset):
    """
    Shift grid a given offset
    Parameters
    ----------
    g : Grid
        a PorePy grid.
    offset : float
        The distance the grid should be shifted in x-direction.

    Returns
    -------
    None.

    """
    g.nodes[0] += offset
    g.face_centers[0] += offset
    g.cell_centers[0] += offset
    if hasattr(g, "level_grids"):
        for gl in g.level_grids:
            gl.nodes[0] += offset
            gl.face_centers[0] += offset
            gl.cell_centers[0] += offset




def extend_domain(problem, var):
    """
    Function that extend the domain by doubling the size of the domain.

    Parameters
    ----------
    problem : BaseData
        Problem class that has inherited from BaseData. 
        See also cases.base_problem.
    var : Array
        An array of the cell-centred variables.

    Raises
    ------
    RuntimeError
        If there is an issue with the adaption.
    AssertionError
        If assumtions on the cell order fails.

    Returns
    -------
    array
        The cell-centered variables for the adapted grid.

    """
    mesh_args = problem.mesh_args
    old_min = np.min(problem.gb.face_centers()[0])
    old_max = np.max(problem.gb.face_centers()[0])

    L = old_max - old_min
    nx = problem.mesh_args["mesh_size"]
    dx = L / nx[0]

    mesh_args["physdims"][0] *= 2
    mesh_args["mesh_size"][0] *= 2

    min_cell_size = 0.5 * dx / 2**problem.max_grid_level
    Ly = problem.domain["ymax"] - problem.domain["ymin"]
    lexsort_shift = lambda x: 0.5 * min_cell_size * x[1] / Ly

    old_grid_var = []
    for g, _ in problem.gb:
        projectionOperator = ProjectionOperator(g)
        Pg2l = projectionOperator.global_to_local(problem.gb, 'cell')
        local_var = Pg2l * var
        if g.dim==2:
            level_var = []
            old_active_cells = g.active_cells
            for level in range(g.num_levels):
                level_var.append(g.project_level_to_leaf(level, 'cell').T * local_var)
            old_grid_var.append(level_var)
        else:
            cc = g.cell_centers.copy()
            cc[0] += lexsort_shift(cc)
            local_sort = np.lexsort(cc[1::-1])
            old_grid_var.append(local_var[local_sort])    

    problem.create_gb(mesh_args)
    # Refine new cells to match old cells
    nx = mesh_args["mesh_size"][0] // 2
    ny = mesh_args["mesh_size"][1]
    for g, _ in problem.gb:
        if g.dim!=2:
            continue
        g._init_projection()
        if problem.max_grid_level > 0:
            for level in range(g.num_levels):
                active_row = np.reshape(old_active_cells[level], (-1, nx))
                if not (nx / 2).is_integer():
                    raise RuntimeError("Number of cells in x-direction must be even")
                new_cells = np.ones((ny, nx // 2), dtype=bool)
                new_active = (np.c_[new_cells, active_row, new_cells]).ravel('C')
                g.refine_cells(g.project_level_to_leaf(level, 'cell') * ~new_active)
                nx *= 2
                ny *= 2

            for e, de in problem.gb.edges_of_node(g):
                mg = de['mortar_grid']
                lg, face_id, _ = pp.partition.extract_subgrid(g, g.frac_pairs[0], faces=True, is_planar=True)
                if np.any(face_id != g.frac_pairs[0]):
                    raise AssertionError("Assume extract subgrid does not change ordering")

                side_grids = {
                    pp.grids.mortar_grid.LEFT_SIDE: lg.copy(),
                    pp.grids.mortar_grid.RIGHT_SIDE: lg.copy(),
                }

                indices = np.asarray(g.frac_pairs).ravel('F')
                indptr = np.arange(2 * (lg.num_cells + 1), step=2)
                values = np.ones(indices.size, dtype=bool)

                face_cells = sps.csc_matrix((values, indices, indptr), (g.num_faces, lg.num_cells)).T
                face_cells = face_cells.tocsr()
                mg = pp.MortarGrid(lg.dim, side_grids, face_cells, face_duplicate_ind=g.frac_pairs[1])
                de['mortar_grid'] = mg

                gl, _ = problem.gb.nodes_of_edge(e)

                gl.num_nodes = lg.num_nodes
                gl.num_faces = lg.num_faces
                gl.num_cells = lg.num_cells
                gl.nodes = lg.nodes
                gl.face_nodes = lg.face_nodes
                gl.cell_faces = lg.cell_faces
                gl.cell_volumes = lg.cell_volumes
                gl.cell_centers = lg.cell_centers
                gl.face_centers = lg.face_centers
                gl.face_normals = lg.face_normals
                gl.face_areas = lg.face_areas
                gl.tags = lg.tags

                # Sanity check
                eps = (
                    mg.secondary_to_mortar_int() * gl.cell_centers[0]
                    -mg.primary_to_mortar_int() * g.face_centers[0]
                    )
                if np.abs(eps).sum() > 1e-10:
                    raise RuntimeError("Mortar mapping is wrong")

    # Update variable

    new_var = []
    i=-1
    tol = 1e-10
    for g, _ in problem.gb:
        if g.dim==0 and (len(old_grid_var) < 3):
            new_var.append(np.zeros(g.num_cells))
            continue
        elif g.dim==1 and( len(old_grid_var) < 2):
            new_var.append(np.zeros(g.num_cells))
            continue
        i += 1
        nx = mesh_args["mesh_size"][0] // 2
        ny = mesh_args["mesh_size"][1]
        if g.dim==2:
            new_grid_var = np.zeros(g.num_cells)
            for level in range(g.num_levels):
                gl = g.level_grids[level]
                # Find projection from old level grid to new level grid
                cells_left = np.sum(gl.cell_centers[0] < old_min + tol)
                cells_right = np.sum(gl.cell_centers[0] > old_max - tol)

                old_cells = np.ones((ny, nx), dtype=bool)
                new_left = np.zeros(cells_left, dtype=bool).reshape((ny, -1))
                new_right = np.zeros(cells_right, dtype=bool).reshape((ny, -1))
                oldInNew = (np.c_[new_left, old_cells, new_right]).ravel()
                size = oldInNew.sum()
                indices = np.arange(size)
                indptr = np.zeros(oldInNew.size + 1, dtype=int)
                indptr[1:][oldInNew] = 1
                indptr = np.cumsum(indptr)
                data = np.ones(indices.size, dtype=bool)
                old2new = sps.csr_matrix((data, indices, indptr), shape=( oldInNew.size, size))

                # Project old values to new grid.
                right_value = np.zeros(cells_right, dtype=float).reshape((ny, -1))
                left_value = np.ones(cells_left, dtype=float).reshape((ny, -1))
                center_value = np.zeros((ny, nx), dtype=float)
                # initial_var assigns 1 to the new left part of the domain and 0 to the new
                # right part
                initial_var = (np.c_[left_value, center_value, right_value]).ravel()
                add_to_new = initial_var + old2new * old_grid_var[i][level]

                new_grid_var += g.project_level_to_leaf(level, 'cell') * add_to_new
                nx *= 2
                ny *= 2
        else:
            nfrac_min = np.sum(g.cell_centers[0] < old_min + tol)
            nfrac_max = np.sum(g.cell_centers[0] > old_max - tol)
            right_value = np.zeros(nfrac_max, dtype=float)
            left_value = np.ones(nfrac_min, dtype=float)
            new_grid_var = np.r_[left_value, old_grid_var[i], right_value]

            cc = g.cell_centers.copy()
            cc[0] += lexsort_shift(cc)
            IA = np.lexsort(cc[1::-1])
            IC = np.argsort(IA)
            new_grid_var = new_grid_var[IC]

        new_var.append(new_grid_var)
            
    return np.hstack(new_var)