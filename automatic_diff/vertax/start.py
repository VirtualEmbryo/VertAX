import os
import numpy as np

import jax.numpy as jnp

from scipy.spatial import Voronoi

from vertax.plot import plot_geograph


def create_geograph(seeds: np.array, show: bool):

    n_cells = len(seeds)
    L_box = np.sqrt(n_cells)

    # PERIODIC VORONOI - VERTICES EDGES FACES

    if n_cells < 20:
        print('\nWarning: [n_cells < 20] initial condition may not work as expected.\n')

    padded_seeds = np.concatenate((seeds,
                                   np.add(seeds, np.full((n_cells, 2), [-L_box, +L_box])),
                                   np.add(seeds, np.full((n_cells, 2), [0, +L_box])),
                                   np.add(seeds, np.full((n_cells, 2), [L_box, +L_box])),
                                   np.add(seeds, np.full((n_cells, 2), [-L_box, 0])),
                                   np.add(seeds, np.full((n_cells, 2), [L_box, 0])),
                                   np.add(seeds, np.full((n_cells, 2), [-L_box, -L_box])),
                                   np.add(seeds, np.full((n_cells, 2), [0, -L_box])),
                                   np.add(seeds, np.full((n_cells, 2), [L_box, -L_box]))), axis=0)

    voronoi = Voronoi(padded_seeds)

    vertices = voronoi.vertices
    edges = voronoi.ridge_vertices
    faces = voronoi.regions

    col0_mask = (vertices[:, 0] >= 0.) & (vertices[:, 0] <= L_box)
    col1_mask = (vertices[:, 1] >= 0.) & (vertices[:, 1] <= L_box)

    periodic_voronoi_vertices_idx = np.arange(len(vertices))[col0_mask & col1_mask]
    periodic_voronoi_vertices_pos = vertices[col0_mask & col1_mask]

    edges_inside = []
    edges_outside = []
    offsets_inside = {}
    offsets_outside = {}
    visited = []

    for e in edges:
        if e[0] in periodic_voronoi_vertices_idx and e[1] in periodic_voronoi_vertices_idx:
            edges_inside.append(tuple(sorted((e[0], e[1]))))
            offsets_inside[(e[0], e[1])] = (0, 0)
            offsets_inside[(e[1], e[0])] = (0, 0)
        if bool(e[0] in periodic_voronoi_vertices_idx) != bool(e[1] in periodic_voronoi_vertices_idx):
            if e[0] in periodic_voronoi_vertices_idx:
                if vertices[e[1]][0] < 0.:
                    x = vertices[e[1]][0] + L_box
                    offset_x1 = -1
                elif vertices[e[1]][0] > L_box:
                    x = vertices[e[1]][0] - L_box
                    offset_x1 = 1
                else:
                    x = vertices[e[1]][0]
                    offset_x1 = 0
                if vertices[e[1]][1] < 0.:
                    y = vertices[e[1]][1] + L_box
                    offset_y1 = -1
                elif vertices[e[1]][1] > L_box:
                    y = vertices[e[1]][1] - L_box
                    offset_y1 = 1
                else:
                    y = vertices[e[1]][1]
                    offset_y1 = 0
                for (idx, pos) in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos):
                    if ((np.abs(pos[0] - x)) < 10 ** -8) and ((np.abs(pos[1] - y)) < 10 ** -8):
                        edges_outside.append(tuple(sorted((e[0], idx))))
                        if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                            offsets_outside[(e[0], idx)] = (offset_x1, offset_y1)
                            offsets_outside[(idx, e[0])] = (-offset_x1, -offset_y1)
                            visited.append((e[0], e[1]))
                            visited.append((e[1], e[0]))
                        break
            else:
                if vertices[e[0]][0] < 0.:
                    x = vertices[e[0]][0] + L_box
                    offset_x0 = -1
                elif vertices[e[0]][0] > L_box:
                    x = vertices[e[0]][0] - L_box
                    offset_x0 = 1
                else:
                    x = vertices[e[0]][0]
                    offset_x0 = 0
                if vertices[e[0]][1] < 0.:
                    y = vertices[e[0]][1] + L_box
                    offset_y0 = -1
                elif vertices[e[0]][1] > L_box:
                    y = vertices[e[0]][1] - L_box
                    offset_y0 = 1
                else:
                    y = vertices[e[0]][1]
                    offset_y0 = 0
                for (idx, pos) in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos):
                    if ((np.abs(pos[0] - x)) < 10 ** -8) and ((np.abs(pos[1] - y)) < 10 ** -8):
                        edges_outside.append(tuple(sorted((idx, e[1]))))
                        if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                            offsets_outside[(idx, e[1])] = (-offset_x0, -offset_y0)
                            offsets_outside[(e[1], idx)] = (offset_x0, offset_y0)
                            visited.append((e[0], e[1]))
                            visited.append((e[1], e[0]))
                        break

    periodic_voronoi_edges = list(set(edges_inside)) + list(set(edges_outside))

    offsets = offsets_inside | offsets_outside

    faces_inside = []
    faces_inside_outside = []
    for face in faces:
        if face:
            if all(item in periodic_voronoi_vertices_idx for item in face):
                faces_inside.append(tuple(sorted(face)))
            if any(item in face for item in periodic_voronoi_vertices_idx):
                face_inside_outside = []
                for f in face:
                    if f in periodic_voronoi_vertices_idx:
                        face_inside_outside.append(f)
                    else:
                        if vertices[f][0] < 0.:
                            x = vertices[f][0] + L_box
                        elif vertices[f][0] > L_box:
                            x = vertices[f][0] - L_box
                        else:
                            x = vertices[f][0]
                        if vertices[f][1] < 0.:
                            y = vertices[f][1] + L_box
                        elif vertices[f][1] > L_box:
                            y = vertices[f][1] - L_box
                        else:
                            y = vertices[f][1]
                        for (idx, pos) in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos):
                            if ((np.abs(pos[0] - x)) < 10 ** -8) and ((np.abs(pos[1] - y)) < 10 ** -8):
                                face_inside_outside.append(idx)
                                break

                faces_inside_outside.append(tuple(sorted(face_inside_outside)))

    periodic_voronoi_faces = list(set(faces_inside_outside))

    # HALF EDGE DATA STRUCTURE

    # reciprocating edges

    periodic_voronoi_half_edges = []
    for e in periodic_voronoi_edges:
        periodic_voronoi_half_edges.append(e)
        periodic_voronoi_half_edges.append((e[1], e[0]))

    # finding clockwise (or counterclockwise) half edge set for each face

    ordered_edges_periodic_voronoi_faces = []
    for face in periodic_voronoi_faces:
        edges_face = []
        for f1 in face:
            for f2 in face:
                if (f1, f2) in periodic_voronoi_edges:
                    edges_face.append((f1, f2))
        i = 0
        start_edge = edges_face[i]
        ordered_face = [start_edge]
        e = start_edge
        visited = [e]
        while sorted(edges_face) != sorted(visited):
            if e[0] == start_edge[1] and e not in visited:
                ordered_face.append(e)
                start_edge = e
                visited.append(e)
            if e[1] == start_edge[1] and e not in visited:
                ordered_face.append((e[1], e[0]))
                start_edge = (e[1], e[0])
                visited.append(e)
            i += 1
            e = edges_face[i % len(face)]

        order = 0
        sum0_offsets = 0
        sum1_offsets = 0
        points = []
        for e in ordered_face:
            idx0 = list(periodic_voronoi_vertices_idx).index(e[0])
            idx1 = list(periodic_voronoi_vertices_idx).index(e[1])
            e_offsets = offsets[e]

            prev_sum0_offsets = sum0_offsets
            prev_sum1_offsets = sum1_offsets
            sum0_offsets += e_offsets[0]
            sum1_offsets += e_offsets[1]

            order += (((periodic_voronoi_vertices_pos[idx1][0] + sum0_offsets * L_box) - (
                        periodic_voronoi_vertices_pos[idx0][0] + prev_sum0_offsets * L_box)) * (
                                  (periodic_voronoi_vertices_pos[idx1][1] + sum1_offsets * L_box) + (
                                      periodic_voronoi_vertices_pos[idx0][1] + prev_sum1_offsets * L_box)))

            points.append((periodic_voronoi_vertices_pos[idx0][0] + prev_sum0_offsets * L_box,
                           periodic_voronoi_vertices_pos[idx0][1] + prev_sum1_offsets * L_box))

            points.append((periodic_voronoi_vertices_pos[idx1][0] + sum0_offsets * L_box,
                           periodic_voronoi_vertices_pos[idx1][1] + sum1_offsets * L_box))

        if order < 0:
            ordered_edges_periodic_voronoi_faces.append(ordered_face)
        if order > 0:
            new_ordered_face = []
            for e in reversed(ordered_face):
                new_ordered_face.append((e[1], e[0]))
            ordered_edges_periodic_voronoi_faces.append(new_ordered_face)
        if order == 0:
            print('\nError: no order detected for face ' + str(face) +'\n')
            exit()

    # VERT FACE HE TABLES

    vertTable = np.zeros((len(periodic_voronoi_vertices_idx), 3))
    for i, (idx, pos) in enumerate(zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos)):
        for j, he in enumerate(periodic_voronoi_half_edges):
            if idx == he[0]:
                idx_selected_he = j
                break
        vertTable[i][0] = pos[0]  # x pos vert
        vertTable[i][1] = pos[1]  # y pos vert
        vertTable[i][2] = idx_selected_he  # he vert source (random among three)

    faceTable = np.zeros((len(periodic_voronoi_faces), 1))
    for i, hedges_face in enumerate(ordered_edges_periodic_voronoi_faces):
        for j, he in enumerate(periodic_voronoi_half_edges):
            if he == hedges_face[0]:
                faceTable[i] = j  # he_inside

    heTable = np.zeros((len(periodic_voronoi_half_edges), 8))
    for i, he in enumerate(periodic_voronoi_half_edges):
        for hedges_face in ordered_edges_periodic_voronoi_faces:
            if he in hedges_face:
                idx = hedges_face.index(he)
                heTable[i][0] = periodic_voronoi_half_edges.index(hedges_face[(idx-1) % len(hedges_face)])  # he_prev
                heTable[i][1] = periodic_voronoi_half_edges.index(hedges_face[(idx+1) % len(hedges_face)])  # he_next
                heTable[i][3] = list(periodic_voronoi_vertices_idx).index(he[0])  # vert source
                heTable[i][4] = list(periodic_voronoi_vertices_idx).index(he[1])  # vert target
                heTable[i][5] = ordered_edges_periodic_voronoi_faces.index(hedges_face)  # face
                break
        heTable[i][2] = periodic_voronoi_half_edges.index((he[1], he[0]))  # he twin
        heTable[i][6] = offsets[he][0]  # he_offset x vert target
        heTable[i][7] = offsets[he][1]  # he_offset y vert target

    os.makedirs('./initial_condition/simulation/', exist_ok=True)

    np.savetxt('./initial_condition/simulation/vertTable.csv', vertTable, delimiter='\t', fmt='%1.18f')
    np.savetxt('./initial_condition/simulation/faceTable.csv', faceTable, delimiter='\t', fmt='%1.0d')
    np.savetxt('./initial_condition/simulation/heTable.csv', heTable, delimiter='\t', fmt='%1.0d')

    vertTable = np.loadtxt('./initial_condition/simulation/vertTable.csv', delimiter='\t', dtype=np.float64)
    faceTable = np.loadtxt('./initial_condition/simulation/faceTable.csv', delimiter='\t', dtype=np.int32)
    heTable = np.loadtxt('./initial_condition/simulation/heTable.csv', delimiter='\t', dtype=np.int32)

    np.save('./initial_condition/simulation/vertTable', vertTable)
    np.save('./initial_condition/simulation/faceTable', faceTable)
    np.save('./initial_condition/simulation/heTable', heTable)

    os.remove('./initial_condition/simulation/vertTable.csv')
    os.remove('./initial_condition/simulation/faceTable.csv')
    os.remove('./initial_condition/simulation/heTable.csv')

    if show:

        plot_geograph(vertTable.astype(float), 
                    faceTable.astype(int), 
                    heTable.astype(int), 
                    L_box=L_box, 
                    multicolor=True, 
                    lines=True, 
                    vertices=False, 
                    path='./initial_condition/simulation/', 
                    name='simulation', 
                    save=True, 
                    show=True)

    return jnp.load('./initial_condition/simulation/vertTable.npy'), jnp.load('./initial_condition/simulation/faceTable.npy'), jnp.load('./initial_condition/simulation/heTable.npy')


def load_geograph(path: str):

    return jnp.load(path + 'vertTable.npy'), jnp.load(path + 'faceTable.npy'), jnp.load(path + 'heTable.npy')


