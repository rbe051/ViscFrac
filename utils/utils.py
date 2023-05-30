"""
Module containing different utility functions
"""
import numpy as np
import scipy.spatial.distance as spsd
import os

import viscous_porepy as pp


def initiate_csc_file(out_file_name):
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    out_file = open(out_file_name, "w")
    out_file.write("time, average_c, L05, Lm05, Lf05, L01, Lm01, Lf01, L001, Lm001, Lf001, num_fingers, num_fingers_scaled\n")
    return out_file


def store_time_data(problem, out_file, gb, t, var_name="concentration"):
    c_avg = volume_average_value(gb, var_name)
    L05 = finger_length(gb, var_name, [1, 2], tol=0.05)
    Lm05 = finger_length(gb, var_name, [2], tol=0.05)
    Lf05 = finger_length(gb, var_name, [1], tol=0.05)
    L01 = finger_length(gb, var_name, [1, 2], tol=0.01)
    Lm01 = finger_length(gb, var_name, [2], tol=0.01)
    Lf01 = finger_length(gb, var_name, [1], tol=0.01)
    L001 = finger_length(gb, var_name, [1, 2], tol=0.001)
    Lm001 = finger_length(gb, var_name, [2], tol=0.001)
    Lf001 = finger_length(gb, var_name, [1], tol=0.001)

    A = np.sum(gb.cell_volumes())
    nx = round(np.sqrt(gb.num_cells() / A))
#    num_fingers = number_of_fingers(gb, var_name, nx, 0.01)
#    num_fingers = number_of_fingers_line(gb, var_name, nx, 0.01)
    num_fingers, num_f_scaled = number_of_fingers_fine(problem, gb, var_name, 0.01)

    out_file.write(
        "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
            t, c_avg, L05, Lm05, Lf05, L01, Lm01, Lf01, L001, Lm001, Lf001, num_fingers, num_f_scaled
        )
    )


def volume_average_value(gb, var_name):
    area = 0
    value = 0
    for g, d in gb:
        phi = d[pp.PARAMETERS]["transport"]["mass_weight"]
        value += np.sum(d[pp.STATE][var_name] * phi)
        area += np.sum(phi)

    return value / area


# utility functions
def finger_length(gb, var_name, dims=None, tol=0.1):
    xmin, xmax = finger_region(gb, var_name, dims=dims, tol=tol)
    return xmax - xmin


def finger_region(gb, var_name, dims=None, tol=0.1):
    xmax = -np.inf
    xmin = np.inf
    if dims is None:
        dims = range(gb.dim_max() + 1)

    gbmin = np.min(gb.cell_centers()[0])
    gbmax = np.max(gb.cell_centers()[0])
    for dim in dims:
        for g in gb.grids_of_dimension(dim):
            d = gb.node_props(g)
            c = d[pp.STATE][var_name]
            xmaxd = np.max(
                np.hstack((g.cell_centers[0, c > tol], gbmin))
            )
            xmind = np.min(
                np.hstack((g.cell_centers[0, c < 1 - tol], gbmax))
            )
            xmax = max(xmax, xmaxd)
            xmin = min(xmin, xmind)
    return xmin, xmax


# Number of fingers
def number_of_fingers(gb, var_name, nx, tol):
    var = []
    for g, d in gb:
        var.append(d[pp.STATE][var_name])
    var = np.hstack(var)
    # First filter out finger region:

    finger_cells = np.argwhere(np.bitwise_and(var > tol, var < 1 - tol)).ravel()
    if finger_cells.size == 0:
        return 0
    xmin = np.min(gb.cell_centers()[0, finger_cells]) - 1e-8
    xmax = np.max(gb.cell_centers()[0, finger_cells]) + 1e-8
    var_f = var[finger_cells]
    cc_f = gb.cell_centers()[:, finger_cells]

    # Then we sample the grid in bins of equal size
    bin_edges_x = np.linspace(xmin, xmax, np.round((xmax - xmin) * nx).astype(int) + 1)
    bin_idx_x = np.digitize(cc_f[0], bin_edges_x)
    dx = np.diff(bin_edges_x)

    ymin = np.min(gb.face_centers()[1])
    ymax = np.max(gb.face_centers()[1])
    bin_edges_y = np.linspace(ymin, ymax, np.round((ymax - ymin) * nx).astype(int) + 1)
    # slice per x bin
    num_f = 0
    for line_number in range(len(bin_edges_x) - 1):
        x_idx = bin_idx_x - 1 == line_number
        var_line, _ = np.histogram(
            cc_f[1, x_idx], bin_edges_y, weights=var_f[x_idx]
        )
        counts, _ = np.histogram(
            cc_f[1, x_idx], bin_edges_y
        )
        var_line = var_line / counts
        # pad for periodic bc
        var_line = np.hstack([var_line[-1], var_line, var_line[0]])

        minima = np.bitwise_and(
            var_line[1:-1] < var_line[0:-2], var_line[1:-1] < var_line[2:]
        ).sum()
        maxima = np.bitwise_and(
            var_line[1:-1] > var_line[0:-2], var_line[1:-1] > var_line[2:]
        ).sum()

        if minima != maxima:
            Warning("Different number of minima than maxima. Taking average")

        num_f += (minima + maxima) / 2 * dx[line_number]

    return num_f / (xmax - xmin)


def number_of_fingers_line(gb, var_name, nx, tol):
    var = []
    for g, d in gb:
        var.append(d[pp.STATE][var_name])
    var = np.hstack(var)
    # First filter out finger region:
    finger_cells = np.argwhere(np.bitwise_and(var > tol, var < 1 - tol)).ravel()
    if finger_cells.size == 0:
        return 0
    xmin = np.min(gb.cell_centers()[0, finger_cells]) - 1e-8
    xmax = np.max(gb.cell_centers()[0, finger_cells]) + 1e-8
    ymin= np.min(gb.cell_centers()[1, finger_cells]) - 1e-8
    ymax = np.max(gb.cell_centers()[1, finger_cells]) + 1e-8

    var_f = var[finger_cells]
    cc_f = gb.cell_centers()[:, finger_cells]

    
    # Then we sample the grid in bins of equal size
    lines_x = np.linspace(xmin, xmax, 50)
    lines_y = np.linspace(ymin, ymax, 200)
    dx = lines_x[1] - lines_x[0] 
    dy = lines_y[1] - lines_y[0] 
    num_f = 0
    i = 0
    for x in lines_x:
        i += 1
        line = np.vstack((x * np.ones(lines_y.size), lines_y)).T

        Y = spsd.cdist(line, cc_f[:2].T)
        min_dist = np.argmin(Y, axis=1)
        _, IA = np.unique(min_dist, return_index=True)
        IA = np.sort(IA)
        var_line = var_f[min_dist][IA]

        # pad for periodic bc
        var_line = np.hstack([var_line[-1], var_line, var_line[0]])

        # Smooth data
#        var_line = (var_line[::2] + var_line[1::2]) / 2
        minima = np.bitwise_and(
            var_line[1:-1] < var_line[0:-2], var_line[1:-1] <= var_line[2:]
        ).sum()

        maxima = np.bitwise_and(
            var_line[1:-1] >= var_line[0:-2], var_line[1:-1] > var_line[2:]
        ).sum()

        if minima != maxima:
            Warning("Different number of minima than maxima. Taking average")

        num_f += (minima + maxima) / 2 * dx

        if i == 20:
            import matplotlib.pyplot as plt
            plt.plot(lines_y[IA], var_line[1:-1])
            plt.title(minima)
            plt.show()

            import pdb; pdb.set_trace()

    return num_f / (xmax - xmin)


def number_of_fingers_fine(problem, gb, var_name, tol):
    var = []
    for g in gb.grids_of_dimension(gb.dim_max()):
        d = gb.node_props(g)
        var.append(d[pp.STATE][var_name])
    var = np.hstack(var)
    # First filter out finger region:
    leaf_finger_cells = np.argwhere(np.bitwise_and(var > tol, var < 1 - tol)).ravel()
    if leaf_finger_cells.size == 0:
        return 0, 0

    grid_max = gb.grids_of_dimension(gb.dim_max())
    if len(grid_max)!=1:
        raise ValueError('number_of_fingers_fine(): Only implement for single highest dim grid')
    g = grid_max[0]

    if "CartLeafGrid" in g.name:
        max_level = np.max(g.cell_level)
        g_level = g.level_grids[max_level]
    else:
        max_level = 0, 0
        g_level = g


    # Project solution to finest level
    if max_level > 0:
        var_level = g.project_level_to_leaf(0, 'cell').T * var # leaf to coarsest
    else:
        var_level = var
    for level in range(max_level):
        var_level = (
            g.project_level_to_leaf(level + 1, 'cell').T * var # Leaf to level
            + g.cell_proj_level(level, level + 1).T * var_level # Propogate to coarse to fine
            )

    xmin = np.min(g.cell_centers[0, leaf_finger_cells]) - 1e-8
    xmax = np.max(g.cell_centers[0, leaf_finger_cells]) + 1e-8

    finger_region = np.bitwise_and(g_level.cell_centers[0] > xmin, g_level.cell_centers[0] < xmax)

    nx = problem.mesh_args['mesh_size'][0] * 2**max_level
    ny = problem.mesh_args['mesh_size'][1] * 2**max_level

    var_f = var_level[finger_region]

    var_f_y = np.reshape(var_f, (ny, -1))

    # Merge values with equal value
    i = 0
    tol = 1e-9
    num_f = 0
    num_f_scaled = 0
    dx = 1 / nx
    L = 0
    for x in range(var_f_y.shape[1]):
        i += 1
        var_f_line = var_f_y[:, x]
        _, IA = np.unique(np.floor(var_f_line / tol).astype(int), return_index=True)
        var_f_line = var_f_line[np.sort(IA)]

        # pad for periodic bc:
        var_f_line = np.hstack([var_f_line[-1], var_f_line, var_f_line[0]])


        is_minima = np.bitwise_and(
             var_f_line[1:-1] < var_f_line[0:-2], var_f_line[1:-1] <= var_f_line[2:]
        )
        minima = is_minima.sum(axis=0)
        is_maxima = np.bitwise_and(
             var_f_line[1:-1] >= var_f_line[0:-2], var_f_line[1:-1] > var_f_line[2:]
        )
        maxima = is_maxima.sum(axis=0)
        

        factor = var_f_line[1:-1][is_maxima] - var_f_line[1:-1][is_minima]
        num_f_scaled += factor.sum(axis=0) * dx

        if np.all(np.equal(minima, maxima)):
            Warning("Different number of minima than maxima. Taking average")

        num_f += (minima + maxima) / 2 * dx
        L += dx
        # if i ==var_f_y.shape[1] // 2:
        #     import matplotlib.pyplot as plt
        #     cc = g_level.cell_centers[1, finger_region].reshape((ny, -1))
        #     plt.plot(cc[:, x][np.sort(IA)], var_f_line[1:-1])
        #     plt.plot(cc[:, x], var_f_y[:, x], '--')
        #     plt.title(minima)
        #     plt.xlabel(num_f)
        #     plt.show()
    if L < 1e-10:
        num_f = 0
        num_f_scaled = 0
    else:
        num_f = num_f / L
        num_f_scaled = num_f_scaled / L
    return num_f, num_f_scaled
