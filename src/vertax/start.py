"""Module for mesh creation, from given seeds or an image."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.spatial import Voronoi

from vertax.mask_analysis import find_vertices_edges_faces, mask_from_image, pad


def load_mesh(path: str) -> tuple[Array, Array, Array]:
    """Load a mesh from a file.

    Args:
        path (str): Path to the mesh file.

    Returns:
        tuple[Array, Array, Array]: Jax arrays of vertices (2D, float), edges and faces.
    """
    mesh_file = np.load(path)
    return (jnp.array(mesh_file["vertices"]), jnp.array(mesh_file["edges"]), jnp.array(mesh_file["faces"]))
    # return jnp.load(path + "vertTable.npy"), jnp.load(path + "heTable.npy"), jnp.load(path + "faceTable.npy")


def save_mesh(path: str, vertTable: Array, heTable: Array, faceTable: Array) -> None:
    """Save a mesh to a file.

    Args:
        path (str): Path to the saved file.
        vertTable (Array): The vertices of the mesh.
        heTable (Array): The half-edges of the mesh.
        faceTable (Array): The faces of the mesh.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, allow_pickle=False, vertices=vertTable, edges=heTable, faces=faceTable)
    # jnp.save(path + "vertTable", vertTable)
    # jnp.save(path + "heTable", heTable)
    # jnp.save(path + "faceTable", faceTable)


def create_mesh_from_seeds(seeds: Array) -> tuple[Array, Array, Array]:  # noqa: C901
    """Create a Periodic Voronoi Mesh from a list of seeds.

    The result is given in a Half-Edge structure.

    Args:
        seeds (Array): 2D float jax array of seed positions.

    Returns:
        tuple[Array, Array, Array]: The vertices, half-edges and faces table of the mesh.
    """
    n_cells = len(seeds)
    L_box = np.sqrt(n_cells)

    # PERIODIC VORONOI - VERTICES EDGES FACES

    if n_cells < 20:
        print("\nWarning: [n_cells < 20] initial condition may not work as expected.\n")

    padded_seeds = np.concatenate(
        (
            seeds,
            np.add(seeds, np.full((n_cells, 2), [-L_box, +L_box])),
            np.add(seeds, np.full((n_cells, 2), [0, +L_box])),
            np.add(seeds, np.full((n_cells, 2), [L_box, +L_box])),
            np.add(seeds, np.full((n_cells, 2), [-L_box, 0])),
            np.add(seeds, np.full((n_cells, 2), [L_box, 0])),
            np.add(seeds, np.full((n_cells, 2), [-L_box, -L_box])),
            np.add(seeds, np.full((n_cells, 2), [0, -L_box])),
            np.add(seeds, np.full((n_cells, 2), [L_box, -L_box])),
        ),
        axis=0,
    )

    voronoi = Voronoi(padded_seeds)

    vertices = voronoi.vertices
    edges = voronoi.ridge_vertices
    faces = voronoi.regions

    col0_mask = (vertices[:, 0] >= 0.0) & (vertices[:, 0] <= L_box)
    col1_mask = (vertices[:, 1] >= 0.0) & (vertices[:, 1] <= L_box)

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
                if vertices[e[1]][0] < 0.0:
                    x = vertices[e[1]][0] + L_box
                    offset_x1 = -1
                elif vertices[e[1]][0] > L_box:
                    x = vertices[e[1]][0] - L_box
                    offset_x1 = 1
                else:
                    x = vertices[e[1]][0]
                    offset_x1 = 0
                if vertices[e[1]][1] < 0.0:
                    y = vertices[e[1]][1] + L_box
                    offset_y1 = -1
                elif vertices[e[1]][1] > L_box:
                    y = vertices[e[1]][1] - L_box
                    offset_y1 = 1
                else:
                    y = vertices[e[1]][1]
                    offset_y1 = 0
                for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                    if ((np.abs(pos[0] - x)) < 10**-8) and ((np.abs(pos[1] - y)) < 10**-8):
                        edges_outside.append(tuple(sorted((e[0], idx))))
                        if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                            offsets_outside[(e[0], idx)] = (offset_x1, offset_y1)
                            offsets_outside[(idx, e[0])] = (-offset_x1, -offset_y1)
                            visited.append((e[0], e[1]))
                            visited.append((e[1], e[0]))
                        break
            else:
                if vertices[e[0]][0] < 0.0:
                    x = vertices[e[0]][0] + L_box
                    offset_x0 = -1
                elif vertices[e[0]][0] > L_box:
                    x = vertices[e[0]][0] - L_box
                    offset_x0 = 1
                else:
                    x = vertices[e[0]][0]
                    offset_x0 = 0
                if vertices[e[0]][1] < 0.0:
                    y = vertices[e[0]][1] + L_box
                    offset_y0 = -1
                elif vertices[e[0]][1] > L_box:
                    y = vertices[e[0]][1] - L_box
                    offset_y0 = 1
                else:
                    y = vertices[e[0]][1]
                    offset_y0 = 0
                for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                    if ((np.abs(pos[0] - x)) < 10**-8) and ((np.abs(pos[1] - y)) < 10**-8):
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
                        if vertices[f][0] < 0.0:
                            x = vertices[f][0] + L_box
                        elif vertices[f][0] > L_box:
                            x = vertices[f][0] - L_box
                        else:
                            x = vertices[f][0]
                        if vertices[f][1] < 0.0:
                            y = vertices[f][1] + L_box
                        elif vertices[f][1] > L_box:
                            y = vertices[f][1] - L_box
                        else:
                            y = vertices[f][1]
                        for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                            if ((np.abs(pos[0] - x)) < 10**-8) and ((np.abs(pos[1] - y)) < 10**-8):
                                face_inside_outside.append(idx)
                                break

                faces_inside_outside.append(tuple(sorted(face_inside_outside)))

    periodic_voronoi_faces = list(set(faces_inside_outside))

    # HALF EDGE DATA STRUCTURE

    # Reciprocating edges
    periodic_voronoi_half_edges = []
    for e in periodic_voronoi_edges:
        periodic_voronoi_half_edges.append(e)
        periodic_voronoi_half_edges.append((e[1], e[0]))

    # Finding clockwise (or counterclockwise) half edge set for each face
    ordered_edges_periodic_voronoi_faces = []
    for face in periodic_voronoi_faces:
        edges_face = [(f1, f2) for f1 in face for f2 in face if (f1, f2) in periodic_voronoi_edges]
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
            e = edges_face[i % len(edges_face)]

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

            order += (
                (periodic_voronoi_vertices_pos[idx1][0] + sum0_offsets * L_box)
                - (periodic_voronoi_vertices_pos[idx0][0] + prev_sum0_offsets * L_box)
            ) * (
                (periodic_voronoi_vertices_pos[idx1][1] + sum1_offsets * L_box)
                + (periodic_voronoi_vertices_pos[idx0][1] + prev_sum1_offsets * L_box)
            )

            points.append(
                (
                    periodic_voronoi_vertices_pos[idx0][0] + prev_sum0_offsets * L_box,
                    periodic_voronoi_vertices_pos[idx0][1] + prev_sum1_offsets * L_box,
                )
            )

            points.append(
                (
                    periodic_voronoi_vertices_pos[idx1][0] + sum0_offsets * L_box,
                    periodic_voronoi_vertices_pos[idx1][1] + sum1_offsets * L_box,
                )
            )

        if order < 0:
            ordered_edges_periodic_voronoi_faces.append(ordered_face)
        if order > 0:
            new_ordered_face = [(e[1], e[0]) for e in reversed(ordered_face)]
            ordered_edges_periodic_voronoi_faces.append(new_ordered_face)
        if order == 0:
            print("\nError: no order detected for face " + str(face) + "\n")
            exit()

    # VERT FACE HE TABLES

    vertTable = np.zeros((len(periodic_voronoi_vertices_idx), 2))
    for i, (_, pos) in enumerate(zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False)):
        vertTable[i][0] = pos[0]  # x pos vert
        vertTable[i][1] = pos[1]  # y pos vert
        # vertTable[i][2] = idx_selected_he  # he vert source (random among three)

    faceTable = np.zeros(len(periodic_voronoi_faces))
    for i, hedges_face in enumerate(ordered_edges_periodic_voronoi_faces):
        for j, he in enumerate(periodic_voronoi_half_edges):
            if he == hedges_face[0]:
                faceTable[i] = j  # he_inside

    heTable = np.zeros((len(periodic_voronoi_half_edges), 8))
    for i, he in enumerate(periodic_voronoi_half_edges):
        for hedges_face in ordered_edges_periodic_voronoi_faces:
            if he in hedges_face:
                idx = hedges_face.index(he)
                heTable[i][0] = periodic_voronoi_half_edges.index(hedges_face[(idx - 1) % len(hedges_face)])  # he_prev
                heTable[i][1] = periodic_voronoi_half_edges.index(hedges_face[(idx + 1) % len(hedges_face)])  # he_next
                heTable[i][3] = list(periodic_voronoi_vertices_idx).index(he[0])  # vert source
                heTable[i][4] = list(periodic_voronoi_vertices_idx).index(he[1])  # vert target
                heTable[i][5] = ordered_edges_periodic_voronoi_faces.index(hedges_face)  # face
                break
        heTable[i][2] = periodic_voronoi_half_edges.index((he[1], he[0]))  # he twin
        heTable[i][6] = offsets[he][0]  # he_offset x vert target
        heTable[i][7] = offsets[he][1]  # he_offset y vert target

    return (
        jnp.array(vertTable, dtype=np.float32),
        jnp.array(heTable, dtype=np.int32),
        jnp.array(faceTable, dtype=np.int32),
    )


def create_mesh_from_image(image: NDArray) -> tuple[Array, Array, Array]:
    """Create a rudimentary mesh with periodic boundary conditions from an image.

    To do that, we perform a segmentation using Cellpose and we try to fill the holes.
    The result will probably be imperfect and it will always be better if you
    provide directly a mask (with no holes) with the function "create_mesh_from_mask".

    Args:
        image (NDArray): The image which will act as a template for the mesh.

    Returns:
        tuple[Array, Array, Array]:  The vertices, half-edges and faces table of the mesh.
    """
    return create_mesh_from_mask(mask_from_image(image))


def create_mesh_from_mask(mask: NDArray) -> tuple[Array, Array, Array]:  # noqa: C901
    """Create a rudimentary mesh with periodic boundary conditions from a mask with no holes.

    Args:
        mask (NDArray): The mask with no holes which will act as a template for the mesh.

    Returns:
        tuple[Array, Array, Array]:  The vertices, half-edges and faces table of the mesh.
    """
    padded_mask = pad(mask, save=False, output_path="refined_and_padded_image.tiff")
    # Find vertices, edges, faces
    vertices, edges, faces = find_vertices_edges_faces(padded_mask)

    # imread tiff = Y is the first axis, X the second.
    height: int = mask.shape[0]  # original image length. Padded is 3 times bigger.
    y_min = height / 2
    y_max = 2 * height + (height / 2)
    width: int = mask.shape[1]
    x_min = width / 2
    x_max = 2 * width + (width / 2)

    col0_mask = (vertices[:, 0] >= x_min) & (vertices[:, 0] < x_max)
    col1_mask = (vertices[:, 1] >= y_min) & (vertices[:, 1] < y_max)

    periodic_vertices_idx = np.arange(len(vertices))[col0_mask & col1_mask]
    periodic_vertices_pos = vertices[col0_mask & col1_mask]

    # store map between vertex id -> inside vertex id
    inside_vertex: dict[int, int] = {idx: idx for idx in periodic_vertices_idx}
    for i, vertex in enumerate(vertices):
        if i not in periodic_vertices_idx:
            x, y = vertex
            if x < x_min:
                x += 2 * width
            elif x >= x_max:
                x -= 2 * width

            if y < y_min:
                y += 2 * height
            elif y >= y_max:
                y -= 2 * height

            # Find corresponding inside vertex to the outside dest vertex
            for idx, pos in zip(periodic_vertices_idx, periodic_vertices_pos, strict=True):
                if np.max(np.abs(pos - [x, y])) < 1:
                    inside_vertex[i] = idx
                    break

    edges_inside = []
    edges_outside = []
    offsets_inside = {}
    offsets_outside = {}
    visited = []

    for e in edges:
        if e[0] in periodic_vertices_idx and e[1] in periodic_vertices_idx:
            edges_inside.append(tuple(sorted((e[0], e[1]))))
            offsets_inside[(e[0], e[1])] = (0, 0)
            offsets_inside[(e[1], e[0])] = (0, 0)
        elif bool(e[0] in periodic_vertices_idx) != bool(e[1] in periodic_vertices_idx):
            if e[0] in periodic_vertices_idx:
                # origin in, dest out
                # check x coord
                if vertices[e[1]][0] < x_min:
                    offset_x1 = -1
                elif vertices[e[1]][0] >= x_max:
                    offset_x1 = 1
                else:
                    offset_x1 = 0

                # Now check y coord
                if vertices[e[1]][1] < y_min:
                    offset_y1 = -1
                elif vertices[e[1]][1] >= y_max:
                    offset_y1 = 1
                else:
                    offset_y1 = 0

                # Find corresponding inside vertex to the outside dest vertex
                if e[1] not in inside_vertex:
                    print(f"Error, no inside vertex found for vertex {e[1]}.")
                else:
                    idx = inside_vertex[e[1]]
                    edges_outside.append(tuple(sorted((e[0], idx))))
                    if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                        offsets_outside[(e[0], idx)] = (offset_x1, offset_y1)
                        offsets_outside[(idx, e[0])] = (-offset_x1, -offset_y1)
                        visited.append((e[0], e[1]))
                        visited.append((e[1], e[0]))
            else:
                # dest in, origin out
                if vertices[e[0]][0] < x_min:
                    offset_x0 = -1
                elif vertices[e[0]][0] >= x_max:
                    offset_x0 = 1
                else:
                    offset_x0 = 0

                if vertices[e[0]][1] < y_min:
                    offset_y0 = -1
                elif vertices[e[0]][1] >= y_max:
                    offset_y0 = 1
                else:
                    offset_y0 = 0

                # Find corresponding inside vertex to the outside dest vertex
                if e[0] not in inside_vertex:
                    print(f"Error, no inside vertex found for vertex {e[0]}.")
                else:
                    idx = inside_vertex[e[0]]
                    edges_outside.append(tuple(sorted((idx, e[1]))))
                    if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                        offsets_outside[(idx, e[1])] = (-offset_x0, -offset_y0)
                        offsets_outside[(e[1], idx)] = (offset_x0, offset_y0)
                        visited.append((e[0], e[1]))
                        visited.append((e[1], e[0]))

    periodic_edges = list(set(edges_inside)) + list(set(edges_outside))
    offsets = offsets_inside | offsets_outside

    periodic_faces: list[set[int]] = [
        {inside_vertex[i] for i in face} for face in faces if any(v_id in periodic_vertices_idx for v_id in face)
    ]

    # HALF EDGE DATA STRUCTURE

    # Reciprocating edges
    periodic_half_edges = []
    for e in periodic_edges:
        periodic_half_edges.append(e)
        periodic_half_edges.append((e[1], e[0]))

    # Finding clockwise (or counterclockwise) half edge set for each face
    ordered_edges_periodic_faces = []
    for k, face in enumerate(periodic_faces):
        edges_face = [(f1, f2) for f1 in face for f2 in face if (f1, f2) in periodic_edges]
        i = 0
        start_edge = edges_face[0]
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
            e = edges_face[i % len(edges_face)]
        order = 0
        sum0_offsets = 0
        sum1_offsets = 0
        points = []
        for e in ordered_face:
            idx0 = list(periodic_vertices_idx).index(e[0])
            idx1 = list(periodic_vertices_idx).index(e[1])
            e_offsets = offsets[e]
            prev_sum0_offsets = sum0_offsets
            prev_sum1_offsets = sum1_offsets
            sum0_offsets += e_offsets[0]
            sum1_offsets += e_offsets[1]
            order += (
                (periodic_vertices_pos[idx1][0] + sum0_offsets * (2 * width))
                - (periodic_vertices_pos[idx0][0] + prev_sum0_offsets * (2 * width))
            ) * (
                (periodic_vertices_pos[idx1][1] + sum1_offsets * (2 * height))
                + (periodic_vertices_pos[idx0][1] + prev_sum1_offsets * (2 * height))
            )
            points.append(
                (
                    periodic_vertices_pos[idx0][0] + prev_sum0_offsets * (2 * width),
                    periodic_vertices_pos[idx0][1] + prev_sum1_offsets * (2 * height),
                )
            )
            points.append(
                (
                    periodic_vertices_pos[idx1][0] + sum0_offsets * (2 * width),
                    periodic_vertices_pos[idx1][1] + sum1_offsets * (2 * height),
                )
            )
        if order < 0:
            ordered_edges_periodic_faces.append(ordered_face)
        if order > 0:
            new_ordered_face = [(e[1], e[0]) for e in reversed(ordered_face)]
            ordered_edges_periodic_faces.append(new_ordered_face)
        if order == 0:
            print(f"f\n{k} face Error: no order detected for face " + str(face) + "\n")
            print(vertices[np.array(face)])
            exit()

    # VERT FACE HE TABLES
    vertTable = periodic_vertices_pos - [x_min, y_min]

    faceTable = np.zeros(len(periodic_faces))
    for i, hedges_face in enumerate(ordered_edges_periodic_faces):
        for j, he in enumerate(periodic_half_edges):
            if he == hedges_face[0]:
                faceTable[i] = j  # he_inside

    heTable = np.zeros((len(periodic_half_edges), 8))
    for i, he in enumerate(periodic_half_edges):
        for hedges_face in ordered_edges_periodic_faces:
            if he in hedges_face:
                idx = hedges_face.index(he)
                heTable[i][0] = periodic_half_edges.index(hedges_face[(idx - 1) % len(hedges_face)])  # he_prev
                heTable[i][1] = periodic_half_edges.index(hedges_face[(idx + 1) % len(hedges_face)])  # he_next
                heTable[i][3] = list(periodic_vertices_idx).index(he[0])  # vert source
                heTable[i][4] = list(periodic_vertices_idx).index(he[1])  # vert target
                heTable[i][5] = ordered_edges_periodic_faces.index(hedges_face)  # face
                break
        heTable[i][2] = periodic_half_edges.index((he[1], he[0]))  # he twin
        heTable[i][6] = offsets[he][0]  # he_offset x vert target
        heTable[i][7] = offsets[he][1]  # he_offset y vert target

    return (
        jnp.array(vertTable, dtype=np.float32),
        jnp.array(heTable, dtype=np.int32),
        jnp.array(faceTable, dtype=np.int32),
    )
