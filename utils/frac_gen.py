"""
Module for generating different fracture networks.
"""
import numpy as np


def brick(xmax, ymax, num_x, num_y, swap_xy):
    if swap_xy:
        temp_max = xmax
        xmax = ymax
        ymax = temp_max
        frac_temp = num_y
        num_y = num_x
        num_x = frac_temp

    # Horizontal fractures
    fracs = []
    dy = ymax / num_y
    dx = xmax / num_x
    for i, y in enumerate(np.linspace(dy/2, ymax - dy/2, num_y)):
        x_vec = [0, xmax]
        y_vec = [y] * 2
        fracs.append(np.array([x_vec, y_vec]))
        # Vertical fractures:
        for x in np.linspace(dx/4, xmax - 3*dx/4, num_x):
            y_vec = [y - dy, y]
            x_vec = [x + dx/2 * np.mod(i, 2)] * 2
            fracs.append(np.array([x_vec, y_vec]))
    for x in np.linspace(dx/4, xmax - 3*dx/4, num_x):
        y_vec = [y, y + dy/2]
        x_vec = [x + dx/2 * np.mod(i + 1, 2)] * 2
        fracs.append(np.array([x_vec, y_vec]))
    frac_inv = []
    for frac in fracs:
        frac_inv.append(np.vstack((frac[1], frac[0])))
    if swap_xy:
    #     num_x = num_y
    #     num_y = num_temp
    #     ymax = 
    #     xmax = aspect_ratio * W
        fracs = frac_inv

    return fracs


def hexagon(gb, xmax, ymax, num_x, num_y, swap_xy, stretch=1/np.sqrt(3)):
    gb = gb.copy()
    dy = ymax / num_y
    dx = xmax / num_x

    if swap_xy:
        y_shift = lambda x: stretch * (
            (np.mod(x[1], dy) - dy / 4) * (np.mod(x[1], dy) < dy / 2) -
            (np.mod(x[1], dy) - 3 * dy / 4) * (np.mod(x[1], dy) >= dy / 2)
        )
        x_shift = lambda x:-2/dx*(
            (np.mod(x[0], dx) - dx / 2) * (np.mod(x[0], 2*dx) < dx) -
            (np.mod(x[0], dx) - dx / 2) * (np.mod(x[0], 2*dx) >= dx)
        )
    else:
        x_shift = lambda x: stretch * (
            (np.mod(x[0], dx) - dx / 4) * (np.mod(x[0], dx) < dx / 2) -
            (np.mod(x[0], dx) - 3 * dx / 4) * (np.mod(x[0], dx) >= dx / 2)
        )
        y_shift = lambda x:-2/dy*(
            (np.mod(x[1], dy) - dy / 2) * (np.mod(x[1], 2*dy) < dy) -
            (np.mod(x[1], dy) - dy / 2) * (np.mod(x[1], 2*dy) >= dy)
        )

    def shift_grid(g):
        if swap_xy:
            g.nodes[1] += 3 * dy / 4
            g.nodes[0] += dx / 2
            g.nodes[0] += -x_shift(g.nodes) * y_shift(g.nodes)

            g.nodes[1] -= 3*dy / 4
            g.nodes[0] -= dx / 2
        else:
            g.nodes[0] += 3 * dx / 4
            g.nodes[1] += dy / 2
            g.nodes[1] += -x_shift(g.nodes) * y_shift(g.nodes)

            g.nodes[0] -= 3*dx / 4
            g.nodes[1] -= dy / 2

    for g, _ in gb:
        shift_grid(g)
    for e, d in gb.edges():
        mg = d['mortar_grid']
        for side_g in mg.side_grids.values():
            shift_grid(side_g)
    gb.compute_geometry()
    return gb
