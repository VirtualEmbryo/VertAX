"""Periodic Boundary Condition on a mesh."""

from pathlib import Path
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.spatial import Voronoi

from vertax.geo import get_area, get_length, get_length_with_offset, get_perimeter, update_pbc
from vertax.mask_analysis import find_vertices_edges_faces, mask_from_image, pad
from vertax.meshes.mesh import Mesh

__all__ = ["PbcMesh"]


class PbcMesh(Mesh):
    """Periodic Boundary Condition on a mesh."""

    def __init__(self) -> None:
        """Do not call the constructor."""
        super().__init__()

        self.MAX_EDGES_IN_ANY_FACE: int = 20

    @classmethod
    def copy_mesh(cls, other_mesh: Self) -> Self:
        """Copy all parameters from another mesh in a new mesh."""
        mesh = cls._create()
        mesh.vertices = other_mesh.vertices.copy()
        mesh.edges = other_mesh.edges.copy()
        mesh.faces = other_mesh.faces.copy()
        mesh.width = other_mesh.width
        mesh.height = other_mesh.height
        mesh.vertices_params = other_mesh.vertices_params.copy()
        mesh.edges_params = other_mesh.edges_params.copy()
        mesh.faces_params = other_mesh.faces_params.copy()
        mesh.MAX_EDGES_IN_ANY_FACE = other_mesh.MAX_EDGES_IN_ANY_FACE

        return mesh

    def save_mesh_txt(
        self,
        directory: str,
        vertices_filename: str = "vertTable.txt",
        edges_filename: str = "heTable.txt",
        faces_filename: str = "faceTable.txt",
        vertices_params_filename: str = "vertParamsTable.txt",
        edges_params_filename: str = "heParamsTable.txt",
        faces_params_filename: str = "faceParamsTable.txt",
        constants_filename: str = "constants.txt",
    ) -> None:
        """Save a mesh in separate text files that can be read by numpy.

        Only save the vertices, edges and faces, not other parameters.

        Args:
            directory (str): Path to the directory where to save the files.
            vertices_filename (str, optional): Filename for the vertices table. Defaults to "vertTable.txt".
            edges_filename (str, optional): Filename for the half-edges table. Defaults to "heTable.txt".
            faces_filename (str, optional): Filename for the faces table. Defaults to "faceTable.txt".
            vertices_params_filename (str, optional): Filename for the vertices parameters table.
                    Defaults to "vertParamsTable.txt".
            edges_params_filename (str, optional): Filename for the half-edges parameters table.
                    Defaults to "heParamsTable.txt".
            faces_params_filename (str, optional): Filename for the faces parameters table.
                    Defaults to "faceParamsTable.txt".
            constants_filename (str, optional): Filename for width/height/MAX_EDGES_IN_ANY_FACE.
                    Defaults to "constants.txt".
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)
        np.savetxt(dirpath / vertices_filename, self.vertices)
        np.savetxt(dirpath / edges_filename, self.edges)
        np.savetxt(dirpath / faces_filename, self.faces)
        np.savetxt(dirpath / vertices_params_filename, self.vertices_params)
        np.savetxt(dirpath / edges_params_filename, self.edges_params)
        np.savetxt(dirpath / faces_params_filename, self.faces_params)
        with (dirpath / constants_filename).open("w") as f:
            f.write(f"{self.width} {self.height} {self.MAX_EDGES_IN_ANY_FACE}")

    def save_mesh(self, path: str) -> None:
        """Save mesh to a file.

        All PBCMesh data is saved.

        Args:
            path (str): Path to the saved file. The extension is .npz, a numpy format.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            allow_pickle=False,
            vertices=self.vertices,
            edges=self.edges,
            faces=self.faces,
            width=self.width,
            height=self.height,
            vertices_params=self.vertices_params,
            edges_params=self.edges_params,
            faces_params=self.faces_params,
            MAX_EDGES_IN_ANY_FACE=self.MAX_EDGES_IN_ANY_FACE,
        )

    @classmethod
    def load_mesh(cls, path: str) -> Self:
        """Load a mesh from a file.

        All PBCMesh data is reloaded.

        Args:
            path (str): Path to the mesh file (.npz), numpy format.

        Returns:
            Mesh: the mesh loaded from the numpy .npz file.
        """
        mesh_file = np.load(path)
        mesh = cls._create()
        mesh.vertices, mesh.edges, mesh.faces = mesh_file["vertices"], mesh_file["edges"], mesh_file["faces"]
        mesh.width, mesh.height = mesh_file["width"], mesh_file["height"]
        mesh.vertices_params = mesh_file["vertices_params"]
        mesh.edges_params = mesh_file["edges_params"]
        mesh.faces_params = mesh_file["faces_params"]
        mesh.MAX_EDGES_IN_ANY_FACE = mesh_file["MAX_EDGES_IN_ANY_FACE"]
        return mesh

    @classmethod
    def load_mesh_txt(
        cls,
        directory: str,
        vertices_filename: str = "vertTable.txt",
        edges_filename: str = "heTable.txt",
        faces_filename: str = "faceTable.txt",
        vertices_params_filename: str = "vertParamsTable.txt",
        edges_params_filename: str = "heParamsTable.txt",
        faces_params_filename: str = "faceParamsTable.txt",
        constants_filename: str = "constants.txt",
    ) -> Self:
        """Load a mesh from text files.

        Only load the vertices, edges and faces, not other parameters.

        Args:
            directory (str): Directory where the text files are stored.
            vertices_filename (str, optional): Filename for the vertices table. Defaults to "vertTable.txt".
            edges_filename (str, optional): Filename for the half-edges table. Defaults to "heTable.txt".
            faces_filename (str, optional): Filename for the faces table. Defaults to "faceTable.txt".
            vertices_params_filename (str, optional): Filename for the vertices parameters table.
                    Defaults to "vertParamsTable.txt".
            edges_params_filename (str, optional): Filename for the half-edges parameters table.
                    Defaults to "heParamsTable.txt".
            faces_params_filename (str, optional): Filename for the faces parameters table.
                    Defaults to "faceParamsTable.txt".
            constants_filename (str, optional): Filename for width/height/MAX_EDGES_IN_ANY_FACE.
                    Defaults to "constants.txt".

        Returns:
            Self: The loaded mesh.
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        mesh = cls._create()
        mesh.vertices = jnp.array(np.loadtxt(dirpath / vertices_filename, dtype=np.float64))
        mesh.edges = jnp.array(np.loadtxt(dirpath / edges_filename, dtype=np.int64))
        mesh.faces = jnp.array(np.loadtxt(dirpath / faces_filename, dtype=np.int64))
        mesh.vertices_params = jnp.array(np.loadtxt(dirpath / vertices_params_filename, dtype=np.float64))
        mesh.edges_params = jnp.array(np.loadtxt(dirpath / edges_params_filename, dtype=np.int64))
        mesh.faces_params = jnp.array(np.loadtxt(dirpath / faces_params_filename, dtype=np.int64))
        with (dirpath / constants_filename).open("r") as f:
            numbers = f.readline().split()
            mesh.width = float(numbers[0])
            mesh.height = float(numbers[1])
            mesh.MAX_EDGES_IN_ANY_FACE = int(numbers[2])
        return mesh

    def get_length(self, half_edge_id: Array) -> Array:
        """Get the length of an edge."""

        def _get_length(half_edge_id: Array) -> Array:
            return get_length(half_edge_id, self.vertices, self.edges, self.faces, self.width, self.height)

        return jax.vmap(_get_length)(half_edge_id)

    def get_length_with_offset(self, half_edge_id: Array) -> Array:
        """Get the length of an edge along with its offsets in an array (length, offset x, offset y)."""

        def _get_length_with_offset(half_edge_id: Array) -> Array:
            return get_length_with_offset(half_edge_id, self.vertices, self.edges, self.faces, self.width, self.height)

        return jax.vmap(_get_length_with_offset)(half_edge_id)

    def get_perimeter(self, face_id: Array) -> Array:
        """Get the perimeter of a face."""

        def _get_perimeter(face_id: Array) -> Array:
            return get_perimeter(
                face_id, self.vertices, self.edges, self.faces, self.width, self.height, self.MAX_EDGES_IN_ANY_FACE
            )

        return jax.vmap(_get_perimeter)(face_id)

    def get_area(self, face_id: Array) -> Array:
        """Get the area of a face."""

        def _get_area(face_id: Array) -> Array:
            return get_area(
                face_id, self.vertices, self.edges, self.faces, self.width, self.height, self.MAX_EDGES_IN_ANY_FACE
            )

        return jax.vmap(_get_area)(face_id)

    def update_boundary_conditions(self) -> None:
        """Force periodic boundary conditions again after an update."""
        self.vertices, self.edges, self.faces = update_pbc(
            self.vertices, self.edges, self.faces, self.width, self.height
        )

    @classmethod
    def periodic_voronoi_from_random_seeds(cls, nb_seeds: int, width: float, height: float, random_key: int) -> Self:
        """Create a Periodic Voronoi Mesh from random seeds.

        Args:
            nb_seeds (int): Number of random seeds to use.
            width (float): Width of the rectangular domains the seeds will be in.
            height (float): Height of the rectangular domains the seeds will be in.
            random_key (int): Set the random key for reproducibility.

        Returns:
            Self: The corresponding mesh.
        """
        key = jax.random.PRNGKey(random_key)
        seeds = jnp.array((width, height)) * jax.random.uniform(key, (nb_seeds, 2))
        return cls.periodic_voronoi_from_seeds(seeds, width, height)

    @classmethod
    def periodic_voronoi_from_seeds(cls, seeds: Array, width: float, height: float) -> Self:
        """Create a Periodic Voronoi Mesh from a list of seeds.

        The seeds are assumed to have positive x and y positions.

        Args:
            seeds (Array[float32]): jax float array of seed positions of shape (nbSeeds, 2).
            width (float): width of the box containing the seeds.
            height (float): height of the box containing the seeds.
        """
        (
            periodic_voronoi_vertices_idx,
            periodic_voronoi_vertices_pos,
            periodic_voronoi_edges,
            offsets,
            periodic_voronoi_faces,
        ) = _make_periodic(seeds, width, height)

        vertices, edges, faces = _make_he_structure(
            width,
            height,
            periodic_voronoi_vertices_idx,
            periodic_voronoi_vertices_pos,
            periodic_voronoi_edges,
            offsets,
            periodic_voronoi_faces,
        )

        pbc_mesh = cls._create()
        pbc_mesh.vertices = jnp.array(vertices, dtype=np.float32)
        pbc_mesh.edges = jnp.array(edges, dtype=np.int32)
        pbc_mesh.faces = jnp.array(faces, dtype=np.int32)
        pbc_mesh.width = width
        pbc_mesh.height = height

        return pbc_mesh

    @classmethod
    def periodic_from_image(
        cls,
        image: NDArray,
    ) -> Self:
        """Create a rudimentary mesh with periodic boundary conditions from an image.

        To do that, we perform a segmentation using Cellpose and we try to fill the holes.
        The result will probably be imperfect and it will always be better if you
        provide directly a mask (with no holes) with the function "periodic_from_mask".

        Args:
            image (NDArray): The image which will act as a template for the mesh.

        Returns:
            tuple[Array, Array, Array]:  The vertices, half-edges and faces table of the mesh.
        """
        return cls.periodic_from_mask(mask_from_image(image))

    @classmethod
    def periodic_from_mask(  # noqa: C901
        cls,
        mask: NDArray,
    ) -> Self:
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

        vertices, edges, faces = _make_he_structure(
            2 * width,
            2 * height,
            periodic_vertices_idx,  # ty:ignore[invalid-argument-type]
            periodic_vertices_pos,
            periodic_edges,
            offsets,
            periodic_faces,
            vertices_offset=(x_min, y_min),
        )

        pbc_mesh = cls._create()
        pbc_mesh.vertices = jnp.array(vertices, dtype=np.float32)
        pbc_mesh.edges = jnp.array(edges, dtype=np.int32)
        pbc_mesh.faces = jnp.array(faces, dtype=np.int32)
        pbc_mesh.width = 2 * width
        pbc_mesh.height = 2 * height

        return pbc_mesh


def _make_periodic(  # noqa: C901
    seeds: Array,
    width: float,
    height: float,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    list[tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    list[set[int]],
]:
    n_cells = len(seeds)

    # PERIODIC VORONOI - VERTICES EDGES FACES

    if n_cells < 20:
        print("\nWarning: [n_cells < 20] initial condition may not work as expected.\n")

    # add eight neighbor copies of the seeds
    padded_seeds = np.concatenate(
        (
            seeds,
            np.add(seeds, np.full((n_cells, 2), [-width, +height])),
            np.add(seeds, np.full((n_cells, 2), [0, +height])),
            np.add(seeds, np.full((n_cells, 2), [width, +height])),
            np.add(seeds, np.full((n_cells, 2), [-width, 0])),
            np.add(seeds, np.full((n_cells, 2), [width, 0])),
            np.add(seeds, np.full((n_cells, 2), [-width, -height])),
            np.add(seeds, np.full((n_cells, 2), [0, -height])),
            np.add(seeds, np.full((n_cells, 2), [width, -height])),
        ),
        axis=0,
    )

    voronoi = Voronoi(padded_seeds)

    vertices = voronoi.vertices
    edges = voronoi.ridge_vertices
    faces = voronoi.regions

    # original vertices and not from neighbor copies
    col0_mask = (vertices[:, 0] >= 0.0) & (vertices[:, 0] <= width)
    col1_mask = (vertices[:, 1] >= 0.0) & (vertices[:, 1] <= height)

    periodic_voronoi_vertices_idx: NDArray[np.int32] = np.arange(len(vertices))[col0_mask & col1_mask]  # ty:ignore[invalid-assignment]
    periodic_voronoi_vertices_pos: NDArray[np.float64] = vertices[col0_mask & col1_mask]

    edges_inside = []
    edges_outside = []
    offsets_inside = {}
    offsets_outside = {}
    visited = []

    for e in edges:
        source_in = e[0] in periodic_voronoi_vertices_idx
        target_in = e[1] in periodic_voronoi_vertices_idx
        if source_in and target_in:
            edges_inside.append(tuple(sorted((e[0], e[1]))))
            offsets_inside[(e[0], e[1])] = (0, 0)
            offsets_inside[(e[1], e[0])] = (0, 0)
        elif source_in:  # and not target_in
            if vertices[e[1]][0] < 0.0:
                x = vertices[e[1]][0] + width
                offset_x1 = -1
            elif vertices[e[1]][0] > width:
                x = vertices[e[1]][0] - width
                offset_x1 = 1
            else:
                x = vertices[e[1]][0]
                offset_x1 = 0
            if vertices[e[1]][1] < 0.0:
                y = vertices[e[1]][1] + height
                offset_y1 = -1
            elif vertices[e[1]][1] > height:
                y = vertices[e[1]][1] - height
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
        elif target_in:  # and not source_in
            if vertices[e[0]][0] < 0.0:
                x = vertices[e[0]][0] + width
                offset_x0 = -1
            elif vertices[e[0]][0] > width:
                x = vertices[e[0]][0] - width
                offset_x0 = 1
            else:
                x = vertices[e[0]][0]
                offset_x0 = 0
            if vertices[e[0]][1] < 0.0:
                y = vertices[e[0]][1] + height
                offset_y0 = -1
            elif vertices[e[0]][1] > height:
                y = vertices[e[0]][1] - height
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

    periodic_voronoi_edges: list[tuple[int, int]] = list(set(edges_inside)) + list(set(edges_outside))

    offsets: dict[tuple[int, int], tuple[int, int]] = offsets_inside | offsets_outside

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
                            x = vertices[f][0] + width
                        elif vertices[f][0] > width:
                            x = vertices[f][0] - width
                        else:
                            x = vertices[f][0]
                        if vertices[f][1] < 0.0:
                            y = vertices[f][1] + height
                        elif vertices[f][1] > height:
                            y = vertices[f][1] - height
                        else:
                            y = vertices[f][1]
                        for idx, pos in zip(periodic_voronoi_vertices_idx, periodic_voronoi_vertices_pos, strict=False):
                            if ((np.abs(pos[0] - x)) < 10**-8) and ((np.abs(pos[1] - y)) < 10**-8):
                                face_inside_outside.append(idx)
                                break

                faces_inside_outside.append(tuple(sorted(face_inside_outside)))

    periodic_voronoi_faces: list[set[int]] = list(set(faces_inside_outside))
    return (
        periodic_voronoi_vertices_idx,
        periodic_voronoi_vertices_pos,
        periodic_voronoi_edges,
        offsets,
        periodic_voronoi_faces,
    )


def _make_he_structure(  # noqa: C901
    width: float,
    height: float,
    periodic_vertices_idx: NDArray[np.int32],
    periodic_vertices_positions: NDArray[np.float64],
    periodic_edges: list[tuple[int, int]],
    offsets: dict[tuple[int, int], tuple[int, int]],
    periodic_faces: list[set[int]],
    vertices_offset: tuple[float, float] = (0, 0),
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Return half-edge structure with vertices, edges, faces.

    vertices is the position of vertices.
    edges is a table of int with:
        previous half-edge id in face,
        next half-edge id in face,
        twin half-edge id,
        source vertex id,
        target vertex id,
        face it belongs id,
        offset x,
        offset y.
    faces records just for each face the id of one of its edges.
    """
    # HALF EDGE DATA STRUCTURE

    # Reciprocating edges
    periodic_half_edges = []
    for e in periodic_edges:
        periodic_half_edges.append(e)
        periodic_half_edges.append((e[1], e[0]))

    # Finding clockwise (or counterclockwise) half edge set for each face
    ordered_edges_periodic_faces = []
    for face in periodic_faces:
        edges_face = [(f1, f2) for f1 in face for f2 in face if (f1, f2) in periodic_edges]
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
            idx0 = list(periodic_vertices_idx).index(e[0])
            idx1 = list(periodic_vertices_idx).index(e[1])
            e_offsets = offsets[e]

            prev_sum0_offsets = sum0_offsets
            prev_sum1_offsets = sum1_offsets
            sum0_offsets += e_offsets[0]
            sum1_offsets += e_offsets[1]

            order += (
                (periodic_vertices_positions[idx1][0] + sum0_offsets * width)
                - (periodic_vertices_positions[idx0][0] + prev_sum0_offsets * width)
            ) * (
                (periodic_vertices_positions[idx1][1] + sum1_offsets * height)
                + (periodic_vertices_positions[idx0][1] + prev_sum1_offsets * height)
            )

            points.append(
                (
                    periodic_vertices_positions[idx0][0] + prev_sum0_offsets * width,
                    periodic_vertices_positions[idx0][1] + prev_sum1_offsets * height,
                )
            )

            points.append(
                (
                    periodic_vertices_positions[idx1][0] + sum0_offsets * width,
                    periodic_vertices_positions[idx1][1] + sum1_offsets * height,
                )
            )

        if order < 0:
            ordered_edges_periodic_faces.append(ordered_face)
        if order > 0:
            new_ordered_face = [(e[1], e[0]) for e in reversed(ordered_face)]
            ordered_edges_periodic_faces.append(new_ordered_face)
        if order == 0:
            print("\nError: no order detected for face " + str(face) + "\n")
            exit()

    # VERT FACE HE TABLES

    vertTable = periodic_vertices_positions - vertices_offset

    faceTable = np.zeros(len(periodic_faces), dtype=np.int32)
    for i, hedges_face in enumerate(ordered_edges_periodic_faces):
        for j, he in enumerate(periodic_half_edges):
            if he == hedges_face[0]:
                faceTable[i] = j  # he_inside

    heTable = np.zeros((len(periodic_half_edges), 8), dtype=np.int32)
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
    return (vertTable, heTable, faceTable)
