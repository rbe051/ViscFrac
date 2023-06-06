"""
Module for merging all grids in a GridBucket that has equal dimension


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
import scipy.sparse as sps

import viscous_porepy as pp


def merge_grids(grids):
    grid = grids[0].copy()
    if hasattr(grids[0], "cell_facetag"):
        grid.cell_facetag = grids[0].cell_facetag
    if hasattr(grids[0], "frac_pairs"):
        grid.frac_pairs = grids[0].frac_pairs

    for sg in grids[1:]:
        grid.num_cells += sg.num_cells
        grid.num_faces += sg.num_faces
        grid.num_nodes += sg.num_nodes
        grid.nodes = np.hstack((grid.nodes, sg.nodes))
        grid.face_nodes = sps.block_diag(
            (grid.face_nodes, sg.face_nodes), dtype=np.bool, format='csc'
        )
        grid.cell_faces = sps.block_diag(
            (grid.cell_faces, sg.cell_faces), dtype=np.int, format='csc'
        )

        grid.face_areas = np.hstack((grid.face_areas, sg.face_areas))
        grid.face_centers = np.hstack((grid.face_centers, sg.face_centers))
        grid.face_normals = np.hstack((grid.face_normals, sg.face_normals))
        grid.cell_volumes = np.hstack((grid.cell_volumes, sg.cell_volumes))
        grid.cell_centers = np.hstack((grid.cell_centers, sg.cell_centers))

        if hasattr(grid, "cell_facetag"):
            grid.cell_facetag = np.hstack((grid.cell_facetag, sg.cell_facetag))
        if hasattr(grid, "frac_pairs"):
            grid.frac_pairs = np.hstack((grid.frac_pairs, sg.frac_pairs))

        if hasattr(grid, "level_grids"):
            for level in range(grid.num_levels):
                if hasattr(grid, "cell_facetag"):
                    grid.cell_facetag = np.hstack((grid.cell_facetag, sg.cell_facetag))
                if hasattr(grid, "frac_pairs"):
                    grid.frac_pairs = np.hstack((grid.frac_pairs, sg.frac_pairs))

                grid.level_grids[level] = merge_grids(
                    [grid.level_grids[level], sg.level_grids[level]]
                )
                grid.cell_projections[level] = sps.block_diag(
                    [grid.cell_projections[level], sg.cell_projections[level]]
                )
                grid.face_projections[level] = sps.block_diag(
                    [grid.face_projections[level], sg.face_projections[level]]
                )
                grid.node_projections[level] = sps.block_diag(
                    [grid.node_projections[level], sg.node_projections[level]]
                )

                if level < grid.num_levels - 1:
                    grid.cell_projections_level[level] = sps.block_diag(
                        [grid.cell_projections_level[level], sg.cell_projections_level[level]]
                    )
                    grid.face_projections_level[level] = sps.block_diag(
                        [grid.face_projections_level[level], sg.face_projections_level[level]]
                    )
                    grid.node_projections_level[level] = sps.block_diag(
                        [grid.node_projections_level[level], sg.node_projections_level[level]]
                    )
                    
                grid.active_cells[level] = np.hstack(
                    [grid.active_cells[level], sg.active_cells[level]]
                )
                grid.active_faces[level] = np.hstack(
                    [grid.active_faces[level], sg.active_faces[level]]
                )
                grid.active_nodes[level] = np.hstack(
                    [grid.active_nodes[level], sg.active_nodes[level]]
                )

            grid.cell_level = np.hstack([grid.cell_level, sg.cell_level])
            
            for level in range(grid.num_levels):
                grid.update_level_to_leaf(level, "cell")
                grid.update_level_to_leaf(level, "face")
                grid.update_level_to_leaf(level, "node")

        for key in grid.tags.keys():
            grid.tags[key] = np.hstack((grid.tags[key], sg.tags[key]))

    return grid


def mergeGridsOfEqualDim(gb):
    dimMax = gb.dim_max()
    
    mergedGrids = []
    gridsOfDim = np.empty(dimMax + 1, dtype=list)
    gridIdx = np.empty(dimMax + 1, dtype=list)
    for i in range(dimMax + 1):
        gridIdx[i] =[]
        gridsOfDim[i] = gb.grids_of_dimension(i)
        if len(gridsOfDim[i])==0:
            mergedGrids.append([])
            continue

        mergedGrids.append(merge_grids(gridsOfDim[i]))
        for grid in gridsOfDim[i]:
            d = gb.node_props(grid)
            gridIdx[i].append(d['node_number'])
        
    mortarsOfDim = np.empty(dimMax + 1, dtype=list)
    for i in range(len(mortarsOfDim)):
        mortarsOfDim[i] = []
        

    for e, d in gb.edges():
        mortar_grids = []
        mg = d['mortar_grid']
        for sg in mg.side_grids.values():
            mortar_grids.append(sg)
        mortarsOfDim[mg.dim].append(merge_grids(mortar_grids))

    primary2mortar = np.empty(dimMax + 1, dtype=np.ndarray)
    secondary2mortar = np.empty(dimMax + 1, dtype=np.ndarray)

    for i in range(dimMax):
        primary2mortar[i] = np.empty((len(mortarsOfDim[i]), len(gridsOfDim[i+1])),dtype=np.object)
        secondary2mortar[i] = np.empty((len(mortarsOfDim[i]), len(gridsOfDim[i])), dtype=np.object)
        
        # Add an empty grid for mortar row. This is to let the block matrices
        # mergedSecondary2Mortar and mergedPrimary2Mortar know the correct dimension
        # if there is an empty mapping. It should be sufficient to add zeros to
        # one of the mortar grids.
        for j in range(len(gridsOfDim[i+1])):
            if len(mortarsOfDim[i])==0:
                continue
            numMortarCells = mortarsOfDim[i][0].num_cells
            numGridFaces = gridsOfDim[i+1][j].num_faces
            primary2mortar[i][0][j] = sps.csc_matrix((numMortarCells, numGridFaces))

        for j in range(len(gridsOfDim[i])):
            if len(mortarsOfDim[i])==0:
                continue
            numMortarCells = mortarsOfDim[i][0].num_cells
            numGridCells = gridsOfDim[i][j].num_cells
            secondary2mortar[i][0][j] = sps.csc_matrix((numMortarCells, numGridCells))
                       

    mortarPos = np.zeros(dimMax + 1, dtype=np.int)
    for e, d in gb.edges():
        mg = d['mortar_grid']
        gs, gm = gb.nodes_of_edge(e)
        ds = gb.node_props(gs)
        dm = gb.node_props(gm)
        assert gs.dim==mg.dim and gm.dim==mg.dim + 1

        secondaryPos = np.argwhere(np.array(gridIdx[mg.dim]) == ds['node_number']).ravel()
        primaryPos = np.argwhere(np.array(gridIdx[mg.dim + 1]) == dm['node_number']).ravel()

        assert (secondaryPos.size==1 and primaryPos.size==1)

        
        secondary2mortar[mg.dim][mortarPos[mg.dim], secondaryPos] = mg.secondary_to_mortar_int()
        primary2mortar[mg.dim][mortarPos[mg.dim], primaryPos] = mg.primary_to_mortar_int()
        mortarPos[mg.dim] += 1

    mergedMortars = []
    mergedSecondary2Mortar = []
    mergedPrimary2Mortar = []
    for dim in range(dimMax + 1):
        if len(mortarsOfDim[dim])==0:
            mergedMortars.append([])
            mergedSecondary2Mortar.append([])
            mergedPrimary2Mortar.append([])
        else:
            mergedMortars.append(merge_grids(mortarsOfDim[dim]))
            mergedSecondary2Mortar.append(sps.bmat(secondary2mortar[dim], format="csc"))
            mergedPrimary2Mortar.append(sps.bmat(primary2mortar[dim], format="csc"))

    mergedGb = pp.GridBucket()
    mergedGb.add_nodes([g for g in mergedGrids if g != []])

    for dim in range(dimMax):
        mg = mergedMortars[dim]
        if (mg == list([])):
            continue
        gm = mergedGrids[dim + 1]
        gs = mergedGrids[dim]
        mergedGb.add_edge((gm, gs), np.empty(0))
        # Make mortar grid with dummy projection
        mg = pp.MortarGrid(gs.dim, {'0': mg}, sps.identity(mg.num_cells))
        # then, set the correct projection
        mg._primary_to_mortar_int = mergedPrimary2Mortar[dim]
        mg._secondary_to_mortar_int = mergedSecondary2Mortar[dim]

        d = mergedGb.edge_props((gm, gs))
        d['mortar_grid'] = mg

    mergedGb.assign_node_ordering()

    return mergedGb
