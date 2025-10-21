"""Module for mesh creation, from given seeds or an image."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import tifffile as tiff
from jax import Array
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import Voronoi


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

    vertTable = np.zeros((len(periodic_voronoi_vertices_idx), 3))
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


def create_mesh_from_image(image: NDArray, path: str = "./") -> tuple[Array, Array, Array]:  # noqa: C901
    def segment(image):
        from cellpose import models

        # Ensure the image is in the correct format
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)  # Convert to (H, W, 1)
        # Blur the image with gaussian filter sigma 10
        blurred_image = gaussian_filter(image, sigma=10)
        # Segment the image using Cellpose's `cyto` model
        model = models.Cellpose(model_type="cyto", gpu=True)
        mask, _, _, _ = model.eval(blurred_image, channels=[0, 0], diameter=None)  # type: ignore
        # Save the resulting image
        output_path = path + "segmented_image.tiff"
        tiff.imwrite(output_path, mask.astype(np.uint16), imagej=True)

        return mask

    def refine_and_pad(mask):
        from skimage.measure import label

        # Relabel the mask
        labeled_mask = label(mask)
        unique_labels, counts = np.unique(labeled_mask, return_counts=True)
        # Create a mask for small labels
        min_size = 9
        small_labels = unique_labels[counts < min_size]
        # Replace small labels with 0
        updated_mask = labeled_mask.copy()
        for label_i in small_labels:
            if label_i != 0:  # Skip the background label
                updated_mask[labeled_mask == label_i] = 0
        # Compute Euclidean Distance Transform (EDT) for background
        distances, indices = distance_transform_edt(updated_mask == 0, return_indices=True)
        # Assign background pixels to the nearest label
        nearest_labels = updated_mask[tuple(indices)]
        expanded_mask = updated_mask.copy()
        expanded_mask[updated_mask == 0] = nearest_labels[updated_mask == 0]
        # Apply reflect padding with the size of the image itself
        height, width = expanded_mask.shape
        padded_image = np.pad(
            expanded_mask,
            ((height, height), (width, width)),  # Reflect padding on top and left only
            mode="reflect",
        )
        # Save the resulting image
        output_path = path + "refined_and_padded_image.tiff"
        tiff.imwrite(output_path, padded_image.astype(np.uint16), imagej=True)

        return label(padded_image)

    def find_vertices_edges_faces(mask):
        # Find unique three-junction points
        three_junctions = []
        unique_label_sets = set()
        height, width = mask.shape
        # Traverse the image
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                # Extract 3x3 neighborhood
                neighborhood = mask[row - 1 : row + 2, col - 1 : col + 2]
                unique_labels = tuple(sorted(np.unique(neighborhood)))
                # Check if there are exactly 3 unique labels and ensure uniqueness of label set
                if len(unique_labels) == 3 and unique_labels not in unique_label_sets:
                    unique_label_sets.add(unique_labels)
                    three_junctions.append([row, col])
        junction_labels = {}
        for idx, (row, col) in enumerate(three_junctions):
            neighborhood = mask[row - 1 : row + 2, col - 1 : col + 2]
            unique_labels = tuple(sorted(np.unique(neighborhood)))
            junction_labels[idx] = unique_labels
        # Find edges between three-junction points
        edges = []
        connections = dict.fromkeys(junction_labels.keys(), 0)  # Track connections for each junction
        for i, (idx1, labels1) in enumerate(junction_labels.items()):
            for idx2, labels2 in list(junction_labels.items())[i + 1 :]:
                # Check if they share exactly two labels
                shared_labels = set(labels1).intersection(set(labels2))
                if len(shared_labels) == 2 and connections[idx1] < 3 and connections[idx2] < 3:
                    # Ensure neither point exceeds 3 connections
                    edges.append([idx1, idx2])
                    connections[idx1] += 1
                    connections[idx2] += 1
        # Find faces as sets of three-junctions sharing a unique label
        faces = []
        for label_i in np.unique(mask):
            label_junctions = [idx for idx, labels in junction_labels.items() if label_i in labels]
            if len(label_junctions) > 2:
                faces.append(label_junctions)

        return three_junctions, edges, faces

    image_shape = image.shape
    L_box = image_shape[0]

    # Segment and refine and pad the image
    input_image = refine_and_pad(segment(image))

    # Find vertices, edges, faces
    vertices, edges, faces = find_vertices_edges_faces(input_image)

    # Saving vertices, edges, faces
    np.save(path + "./vertices.npy", vertices)
    with open(path + "edges.txt", "w") as file:
        for row in edges:
            # Convert each row to a string with tab separation, then write to the file
            file.write("\t".join(map(str, row)) + "\n")
    with open(path + "faces.txt", "w") as file:
        for row in faces:
            # Convert each row to a string with tab separation, then write to the file
            file.write("\t".join(map(str, row)) + "\n")

    L_min = L_box / 2
    L_max = 2 * L_box + (L_box / 2)

    col0_mask = (vertices[:, 0] >= L_min) & (vertices[:, 0] < L_max)
    col1_mask = (vertices[:, 1] >= L_min) & (vertices[:, 1] < L_max)

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
                if vertices[e[1]][0] < L_min:
                    x = vertices[e[1]][0] + (2 * L_box)
                    offset_x1 = -1
                elif vertices[e[1]][0] > L_max:
                    x = vertices[e[1]][0] - (2 * L_box)
                    offset_x1 = 1
                else:
                    x = vertices[e[1]][0]
                    offset_x1 = 0
                if vertices[e[1]][1] < L_min:
                    y = vertices[e[1]][1] + (2 * L_box)
                    offset_y1 = -1
                elif vertices[e[1]][1] > L_max:
                    y = vertices[e[1]][1] - (2 * L_box)
                    offset_y1 = 1
                else:
                    y = vertices[e[1]][1]
                    offset_y1 = 0
                for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                    if ((np.abs(pos[0] - x)) < 3) and ((np.abs(pos[1] - y)) < 3):
                        edges_outside.append(tuple(sorted((e[0], idx))))
                        if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                            offsets_outside[(e[0], idx)] = (offset_x1, offset_y1)
                            offsets_outside[(idx, e[0])] = (-offset_x1, -offset_y1)
                            visited.append((e[0], e[1]))
                            visited.append((e[1], e[0]))
                        break
            else:
                if vertices[e[0]][0] < L_min:
                    x = vertices[e[0]][0] + (2 * L_box)
                    offset_x0 = -1
                elif vertices[e[0]][0] > L_max:
                    x = vertices[e[0]][0] - (2 * L_box)
                    offset_x0 = 1
                else:
                    x = vertices[e[0]][0]
                    offset_x0 = 0
                if vertices[e[0]][1] < L_min:
                    y = vertices[e[0]][1] + (2 * L_box)
                    offset_y0 = -1
                elif vertices[e[0]][1] > L_max:
                    y = vertices[e[0]][1] - (2 * L_box)
                    offset_y0 = 1
                else:
                    y = vertices[e[0]][1]
                    offset_y0 = 0
                for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                    if ((np.abs(pos[0] - x)) < 3) and ((np.abs(pos[1] - y)) < 3):
                        edges_outside.append(tuple(sorted((idx, e[1]))))
                        if (e[0], e[1]) not in visited and (e[1], e[0]) not in visited:
                            offsets_outside[(idx, e[1])] = (-offset_x0, -offset_y0)
                            offsets_outside[(e[1], idx)] = (offset_x0, offset_y0)
                            visited.append((e[0], e[1]))
                            visited.append((e[1], e[0]))
                        break

    periodic_voronoi_edges = list(set(edges_inside)) + list(set(edges_outside))

    offsets = offsets_inside | offsets_outside

    faces_inside_outside = []
    for face in faces:
        if any(item in face for item in periodic_voronoi_vertices_idx):
            face_inside_outside = []
            for f in face:
                if f in periodic_voronoi_vertices_idx:
                    face_inside_outside.append(f)
                else:
                    if vertices[f][0] < L_min:
                        x = vertices[f][0] + (2 * L_box)
                    elif vertices[f][0] >= L_max:
                        x = vertices[f][0] - (2 * L_box)
                    else:
                        x = vertices[f][0]
                    if vertices[f][1] < L_min:
                        y = vertices[f][1] + (2 * L_box)
                    elif vertices[f][1] >= L_max:
                        y = vertices[f][1] - (2 * L_box)
                    else:
                        y = vertices[f][1]
                    for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                        if ((np.abs(pos[0] - x)) < 3) and ((np.abs(pos[1] - y)) < 3):
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
            order += (
                (periodic_voronoi_vertices_pos[idx1][0] + sum0_offsets * (2 * L_box))
                - (periodic_voronoi_vertices_pos[idx0][0] + prev_sum0_offsets * (2 * L_box))
            ) * (
                (periodic_voronoi_vertices_pos[idx1][1] + sum1_offsets * (2 * L_box))
                + (periodic_voronoi_vertices_pos[idx0][1] + prev_sum1_offsets * (2 * L_box))
            )
            points.append(
                (
                    periodic_voronoi_vertices_pos[idx0][0] + prev_sum0_offsets * (2 * L_box),
                    periodic_voronoi_vertices_pos[idx0][1] + prev_sum1_offsets * (2 * L_box),
                )
            )
            points.append(
                (
                    periodic_voronoi_vertices_pos[idx1][0] + sum0_offsets * (2 * L_box),
                    periodic_voronoi_vertices_pos[idx1][1] + sum1_offsets * (2 * L_box),
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

    vertTable = np.zeros((len(periodic_voronoi_vertices_idx), 3))
    for i, (idx, pos) in enumerate(zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False)):
        for j, he in enumerate(periodic_voronoi_half_edges):
            if idx == he[0]:
                break
        vertTable[i][0] = pos[0] - L_min  # y pos vert
        vertTable[i][1] = pos[1] - L_min  # x pos vert
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
