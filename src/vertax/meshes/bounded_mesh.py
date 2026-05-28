"""Bounded mesh are useful to represent finite tissue clusters with curved interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Voronoi

from vertax.geo import get_any_length, get_area_bounded, get_perimeter_bounded
from vertax.meshes.mesh import Mesh

if TYPE_CHECKING:
    from jax import Array
    from numpy.random import Generator
    from numpy.typing import NDArray

__all__ = ["BoundedMesh"]


class BoundedMesh(Mesh):
    """Bounded mesh with arc circles for boundary cells.

    For a BoundedMesh, `vertices` is a 2D array of floats of size (nb_vertices, 2) ;
    with the coordinates of the vertices (in ]0, width[ x ]0, height[ ).

    `edges` is a  2D array of integers of size (nb_half_edges, 8), with:

    - id of previous half-edge,
    - id of next half-edge,
    - id of twin half-edge,
    - id of source vertex + 2 if current half-edge is an inside edge, else 0,
    - id of target vertex + 2 if current half-edge is an inside edge, else 1,
    - id of source vertex + 2 if current half-edge is an outside edge, else 0,
    - id of target vertex + 2 if current half-edge is an outside edge, else 1,
    - id of the face containing the half-edge.

    `faces` is a 1D array of integers of size (nb_faces) containing the id of a half-edge belonging to this face.

    `angles` is a 1D array of floats of size (nb_angles), with the angles sustaining the arcs of the free interfaces ;
    between 0 and PI / 2.
    """

    def __init__(self) -> None:
        """Do not call the constructor."""
        super().__init__()
        self.angles: Array = jnp.array([])
        """Angle sustaining the arced free interfaces. Between 0 and PI / 2."""

    @property
    def nb_angles(self) -> int:
        """Get the number of angles (free interfaces) of the mesh."""
        return len(self.angles)

    @classmethod
    def from_random_seeds(cls, nb_seeds: int, width: float, height: float, random_key: int, nb_fates: int = 2) -> Self:
        """Create a bounded Mesh from random seeds, based on a Voronoi diagram with arced free interfaces.

        Args:
            nb_seeds (int): Number of random seeds to use.
            width (float): width of the box containing the seeds.
            height (float): height of the box containing the seeds.
            random_key (int): seed for a random number generator to add new seeds if needed, decide on cell fates...
            nb_fates (int, default=2): number of possible different fate marker for a cell.

        Returns:
            Self: The corresponding mesh.
        """
        rng = np.random.default_rng(seed=random_key)
        seeds = rng.random((nb_seeds, 2)) * (width, height)
        return cls.from_seeds(seeds, width, height, random_key, nb_fates)

    @classmethod
    def from_seeds(cls, seeds: NDArray, width: float, height: float, random_key: int, nb_fates: int = 2) -> Self:  # noqa: C901
        """Create a bounded Mesh from a list of given seeds.

        The seeds are assumed to have x-coordinate in ]0, width[ and y-coordinate in ]0, height[.
        Note that the final mesh might not use your seeds if they don't work to create a correct
        bounded mesh via our method.

        Args:
            seeds (Array[float32]): jax float array of seed positions of shape (nbSeeds, 2).
            width (float): width of the box containing the seeds.
            height (float): height of the box containing the seeds.
            random_key (int): seed for a random number generator to add new seeds if needed, decide on cell fates...
            nb_fates (int, default=2): number of possible different fate marker for a cell.
        """
        rng = np.random.default_rng(seed=random_key)
        n_cells = len(seeds)  # starting number of seeds must be equal to the desired number of cells (faces)

        # We'll try to construct a Voronoi diagram with n_cells closed (bounded) cells.
        # This will not work with exactly n_cells seeds because there will be unbounded cells.
        # If the number of bounded cells is insufficient, we add a new seed to the list.
        # If there is too many bounded cells, we retry with entirely new seeds.
        # We also check that the bounded cells are connected.
        while True:
            success = 0  # if 1 : not enough bounded cells. If 2 : too many bounded cells.

            # Create the Voronoi diagrams from current seed list.
            voronoi = Voronoi(seeds)
            vertices = voronoi.vertices
            edges = voronoi.ridge_vertices
            faces = voronoi.regions  # regions = faces = cells

            # We count the number of bounded cells and the connectivity of vertices.
            inbound_faces = []
            inbound_vertices = np.zeros(vertices.shape[0], dtype=np.int32)
            for face in faces:
                if face and all(item > -1 for item in face):  # the face must not be an empty list
                    face_vertices_positions = vertices[face]
                    # We check that all of the face's vertices are in a box [0, width]x[0, height]
                    if (
                        np.all(face_vertices_positions[:, 0] < width)
                        and np.all(face_vertices_positions[:, 1] < height)
                        and np.all(face_vertices_positions > 0)
                    ):
                        inbound_faces.append(face)  # the face is bounded
                        inbound_vertices[face] += 1  # +1 to the connectivity of the face's vertices.

            # getting rid of faces connected to a single other inbound face
            # (these can be problematic and lead to many special cases later on)
            while True:
                num_infaces = len(inbound_faces)
                del_count = 0
                for i, face in enumerate(reversed(inbound_faces)):
                    # only 2 (or less (not sure it's possible)) vertices of the face are shared with other faces :
                    # that means it is connected to one other bounded face only -> We remove it.
                    if np.sum(inbound_vertices[face] > 1) <= 2:
                        inbound_vertices[face] -= 1
                        del inbound_faces[num_infaces - i - 1]
                        del_count += 1
                # Removing one face can possibly alter other faces so we might do another loop.
                # We stop when there is no more face to remove.
                if del_count == 0:
                    break

            # Check that we have the correct number of cells.
            if num_infaces < n_cells:
                success = 1
            elif num_infaces > n_cells:
                success = 2
            else:
                # There is exactly n_cells connected bounded faces.
                # Now, it is possible that a bounded face has vertices or edges
                # that are not shared with other faces. We get rid of those,
                # in order to have only one exterior edge that will be an arc circle.
                for i, face in enumerate(inbound_faces):
                    useful_vertices = []  # List of the face vertices that are shared with other.
                    # (We'll keep them and call them "useful").
                    extra_edges = []  # List of edges to replace what we have removed
                    last_useful = -1  # ID of the last "useful" vertex. -1 at the beginning (we'll treat that case)
                    new_edge = []  # New edge that will replace current vertices we're trying to remove
                    incomplete_new_edge = False  # State boolean : are we replacing vertices right now ?
                    for vertex in face:
                        if inbound_vertices[vertex] == 1:  # We found a vertex that is not shared with other faces.
                            # We plan to remove it by not adding it to the useful vertices list,
                            # and by creating a new edge from last useful vertex to the next one.
                            if not incomplete_new_edge:  # Detect if we're not already in the incomplete edge state
                                new_edge = []  # re-init
                                new_edge.append(last_useful)
                                incomplete_new_edge = True  # Move to incomplete edge state.
                        else:  # We found a useful vertex
                            useful_vertices.append(vertex)
                            last_useful = vertex
                            if incomplete_new_edge:  # If in incomplete edge state we can finally close the new edge.
                                new_edge.append(vertex)
                                extra_edges.append(new_edge)
                                incomplete_new_edge = False
                    # After looping through the vertices of the face, we need to take care of
                    # two special cases : the first or the last vertex is not shared.
                    if extra_edges and extra_edges[0][0] == -1:
                        extra_edges[0][0] = useful_vertices[-1]
                    elif incomplete_new_edge:
                        new_edge.append(useful_vertices[0])
                        extra_edges.append(new_edge)
                    # The extra edges are added to the list of all edges
                    edges.extend(extra_edges)
                    inbound_faces[i] = tuple(
                        sorted(useful_vertices)
                    )  # And the face itself is replaced by only the useful vertices.
                    # Note that the vertices here are not ordered in clockwise or counterclockwise order anymore.
                useful_vertices_set = set(np.where(inbound_vertices > 1)[0])  # We filter the useful vertices.

                # HALF EDGE DATA STRUCTURE
                # Filter edges with useful vertices only.
                useful_edges = [tuple(sorted(e)) for e in edges if set(e).issubset(useful_vertices_set)]

                # failing to abide by the following relation results in disconnected topologies
                if len(useful_edges) != (n_cells - 1) * 3:
                    success = 2  # Case : we want to restart with new seeds because current solution is not OK.
                else:
                    # We construct the half-edges.
                    half_edges = []
                    for e in useful_edges:
                        half_edges.append(e)
                        # reciprocating edges
                        half_edges.append((e[1], e[0]))

                    # finding clockwise (or counterclockwise) half edge set for each face,
                    # as we broke it earlier.
                    ordered_edges_inbound_faces = []
                    for face in inbound_faces:
                        # Find all edges fon this face and we'll loow through them to order them.
                        edges_face = [(f1, f2) for f1 in face for f2 in face if (f1, f2) in useful_edges]

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
                            # We must be careful because some edges might be in the wrong order.
                            if e[1] == start_edge[1] and e not in visited:
                                ordered_face.append((e[1], e[0]))
                                start_edge = (e[1], e[0])
                                visited.append(e)
                            i += 1
                            e = edges_face[i % len(edges_face)]

                        # Sanity check : do we have a correct ordering ?
                        order = 0
                        for e in ordered_face:
                            idx0 = e[0]
                            idx1 = e[1]

                            order += (vertices[idx1][0] - vertices[idx0][0]) * (vertices[idx1][1] + vertices[idx0][1])

                        if order < 0:
                            ordered_edges_inbound_faces.append(ordered_face)
                        if order > 0:
                            ordered_edges_inbound_faces.append([(e[1], e[0]) for e in reversed(ordered_face)])
                        if order == 0:
                            print("\nError: no order detected for face " + str(face) + "\n")
                            exit()

                    # Now we fill the tables with the info we have.
                    useful_vertices_list = list(useful_vertices_set)
                    vertTable = np.zeros((len(useful_vertices_list), 2))
                    for i, idx in enumerate(useful_vertices_list):
                        pos = vertices[idx]
                        vertTable[i][0] = pos[0]  # x pos vert
                        vertTable[i][1] = pos[1]  # y pos vert

                    faceTable = np.zeros((len(inbound_faces), 1), dtype=np.int32)
                    for i, hedges_face in enumerate(ordered_edges_inbound_faces):
                        for j, he in enumerate(half_edges):
                            if he == hedges_face[0]:
                                faceTable[i] = j  # he_inside
                    faceTable = _fate_selection(faceTable, nb_fates, rng)

                    nb_half_edges = len(half_edges)
                    heTable = np.zeros((nb_half_edges, 8), dtype=np.int32)
                    heTable[:, 4] = 1
                    heTable[:, 6] = 1
                    relevant_twins = []
                    # HE TABLE :
                    # 0 : previous half-edge.
                    # 1 : next half-edge.
                    # 2 : twin half-edge.
                    # 3 : source vertex id + 2 if edge is inside, 0 if the edge is outside
                    # 4 : target vertex id + 2 if edge is inside, 0 if the edge is outside
                    # 5 : source vertex id + 2 if edge is outside, 0 if the edge is inside
                    # 6 : target vertex id + 2 if edge is outside, 0 if the edge is inside
                    # 7 : id of the face containing this half-edge.
                    for i, he in enumerate(half_edges):
                        belongs_to_any_face = False
                        for hedges_face in ordered_edges_inbound_faces:
                            if he in hedges_face:
                                idx = hedges_face.index(he)
                                heTable[i][0] = half_edges.index(hedges_face[(idx - 1) % len(hedges_face)])  # he_prev
                                heTable[i][1] = half_edges.index(hedges_face[(idx + 1) % len(hedges_face)])  # he_next
                                # indices 0 and 1 are reserved for source or target vertices of "outside" edges.
                                # So we have to add +2 to other indices.
                                heTable[i][3] = useful_vertices_list.index(he[0]) + 2  # vert source inner edges
                                heTable[i][4] = useful_vertices_list.index(he[1]) + 2  # vert target inner edges
                                heTable[i][7] = ordered_edges_inbound_faces.index(hedges_face)  # face
                                belongs_to_any_face = True
                                break
                        twin_idx = half_edges.index((he[1], he[0]))
                        heTable[i][2] = twin_idx  # he twin
                        if not belongs_to_any_face:
                            relevant_twins.append(twin_idx)

                    # Angles are randomly chosen between 0 and pi/2 (with some margin to avoid extreme cases).
                    angTable = np.ones(nb_half_edges // 2)
                    for tidx in relevant_twins:
                        angTable[tidx // 2] = rng.random() * (np.pi / 2 - 0.018) + 0.017
                        heTable[tidx][5] = heTable[tidx][3]  # vert source surface edges
                        heTable[tidx][6] = heTable[tidx][4]  # vert target surface edges
                        heTable[tidx][3] = 0
                        heTable[tidx][4] = 1

                    bounded_mesh = cls._create()
                    bounded_mesh.vertices = jnp.array(vertTable, dtype=np.float32)
                    bounded_mesh.angles = jnp.array(angTable, dtype=np.float32)
                    bounded_mesh.faces = jnp.array(faceTable, dtype=np.int32)
                    bounded_mesh.edges = jnp.array(heTable, dtype=np.int32)
                    bounded_mesh.width = width
                    bounded_mesh.height = height

                    return bounded_mesh

            # If success was 1, we had not enough bounded faces, we add a new seed to see if it helps.
            # Otherwise we retry with new seeds entirely.
            seeds = (
                np.vstack([seeds, (width, height) * rng.random((1, 2))])
                if success == 1
                else (width, height) * rng.random((n_cells, 2))
            )  # type: ignore

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty mesh. Use if you know what you're doing !"""
        return cls._create()

    @classmethod
    def copy_mesh(cls, other_mesh: Self) -> Self:
        """Copy all parameters from another mesh in a new mesh."""
        mesh = cls._create()
        mesh.vertices = other_mesh.vertices.copy()
        mesh.edges = other_mesh.edges.copy()
        mesh.faces = other_mesh.faces.copy()
        mesh.angles = other_mesh.angles.copy()
        mesh.width = other_mesh.width
        mesh.height = other_mesh.height
        mesh.vertices_params = other_mesh.vertices_params.copy()
        mesh.edges_params = other_mesh.edges_params.copy()
        mesh.faces_params = other_mesh.faces_params.copy()

        return mesh

    def save_mesh(self, path: str) -> None:
        """Save mesh to a file.

        All BoundedMesh data is saved.

        Args:
            path (str): Path to the saved file. The extension is .npz.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            allow_pickle=False,
            vertices=self.vertices,
            edges=self.edges,
            faces=self.faces,
            angles=self.angles,
            width=self.width,
            height=self.height,
            vertices_params=self.vertices_params,
            edges_params=self.edges_params,
            faces_params=self.faces_params,
        )

    @classmethod
    def load_mesh(cls, path: str) -> Self:
        """Load a mesh from a file.

        All BoundedMesh data is reloaded.

        Args:
            path (str): Path to the mesh file (.npz).

        Returns:
            Mesh: the mesh loaded from the .npz file.
        """
        mesh_file = np.load(path)
        mesh = cls._create()
        mesh.vertices, mesh.edges, mesh.faces, mesh.angles = (
            mesh_file["vertices"],
            mesh_file["edges"],
            mesh_file["faces"],
            mesh_file["angles"],
        )
        mesh.width = mesh_file["width"]
        mesh.height = mesh_file["height"]
        mesh.vertices_params = mesh_file["vertices_params"]
        mesh.edges_params = mesh_file["edges_params"]
        mesh.faces_params = mesh_file["faces_params"]
        return mesh

    def save_mesh_txt(
        self,
        directory: str,
        vertices_filename: str = "vertTable.txt",
        angles_filename: str = "angTable.txt",
        edges_filename: str = "heTable.txt",
        faces_filename: str = "faceTable.txt",
        vertices_params_filename: str = "vertParamsTable.txt",
        edges_params_filename: str = "heParamsTable.txt",
        faces_params_filename: str = "faceParamsTable.txt",
        constants_filename: str = "constants.txt",
    ) -> None:
        """Save a mesh in separate text files that can be read by numpy.

        Only save the vertices, angles, edges and faces, not other parameters.

        Args:
            directory (str): Path to the directory where to save the files.
            vertices_filename (str, optional): Filename for the vertices table. Defaults to "vertTable.txt".
            angles_filename (str, optional): Filename for the angles table. Defaults to "angTable.txt".
            edges_filename (str, optional): Filename for the half-edges table. Defaults to "heTable.txt".
            faces_filename (str, optional): Filename for the faces table. Defaults to "faceTable.txt".
            vertices_params_filename (str, optional): Filename for the vertices parameters table.
                    Defaults to "vertParamsTable.txt".
            edges_params_filename (str, optional): Filename for the half-edges parameters table.
                    Defaults to "heParamsTable.txt".
            faces_params_filename (str, optional): Filename for the faces parameters table.
                    Defaults to "faceParamsTable.txt".
            constants_filename (str, optional): Filename for width/height.
                    Defaults to "constants.txt".
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)
        np.savetxt(dirpath / vertices_filename, self.vertices)
        np.savetxt(dirpath / angles_filename, self.angles)
        np.savetxt(dirpath / edges_filename, self.edges)
        np.savetxt(dirpath / faces_filename, self.faces)
        np.savetxt(dirpath / vertices_params_filename, self.vertices_params)
        np.savetxt(dirpath / edges_params_filename, self.edges_params)
        np.savetxt(dirpath / faces_params_filename, self.faces_params)
        with (dirpath / constants_filename).open("w") as f:
            f.write(f"{self.width} {self.height}")

    @classmethod
    def load_mesh_txt(
        cls,
        directory: str,
        vertices_filename: str = "vertTable.txt",
        angles_filename: str = "angTable.txt",
        edges_filename: str = "heTable.txt",
        faces_filename: str = "faceTable.txt",
        vertices_params_filename: str = "vertParamsTable.txt",
        edges_params_filename: str = "heParamsTable.txt",
        faces_params_filename: str = "faceParamsTable.txt",
        constants_filename: str = "constants.txt",
    ) -> Self:
        """Load a mesh from text files.

        Only load the vertices, angles, edges and faces, not other parameters.

        Args:
            directory (str): Directory where the text files are stored.
            vertices_filename (str, optional): Filename for the vertices table. Defaults to "vertTable.txt".
            angles_filename (str, optional): Filename for the angles table. Defaults to "angTable.txt".
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
        mesh.angles = jnp.array(np.loadtxt(dirpath / angles_filename, dtype=np.float64))
        mesh.edges = jnp.array(np.loadtxt(dirpath / edges_filename, dtype=np.int64))
        mesh.faces = jnp.array(np.loadtxt(dirpath / faces_filename, dtype=np.int64))
        mesh.vertices_params = jnp.array(np.loadtxt(dirpath / vertices_params_filename, dtype=np.float64))
        mesh.edges_params = jnp.array(np.loadtxt(dirpath / edges_params_filename, dtype=np.int64))
        mesh.faces_params = jnp.array(np.loadtxt(dirpath / faces_params_filename, dtype=np.int64))
        with (dirpath / constants_filename).open("r") as f:
            numbers = f.readline().split()
            mesh.width = float(numbers[0])
            mesh.height = float(numbers[1])
        return mesh

    def get_length(self, half_edge_id: Array) -> Array:
        """Get the length of an edge."""
        vertTable = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), self.vertices])
        angTable = jnp.repeat(self.angles, 2)

        def _get_length(half_edge_id: Array) -> Array:
            return get_any_length(half_edge_id, vertTable, angTable, self.edges)

        return jax.vmap(_get_length)(half_edge_id)

    def get_perimeter(self, face_id: Array) -> Array:
        """Get the area of a face."""
        vertTable = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), self.vertices])
        angTable = jnp.repeat(self.angles, 2)

        def _get_perimeter(face_id: Array) -> Array:
            return get_perimeter_bounded(face_id, vertTable, angTable, self.edges, self.faces)

        return jax.vmap(_get_perimeter)(face_id)

    def get_area(self, face_id: Array) -> Array:
        """Get the area of a face."""
        vertTable = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), self.vertices])
        angTable = jnp.repeat(self.angles, 2)

        def _get_area(face_id: Array) -> Array:
            return get_area_bounded(face_id, vertTable, angTable, self.edges, self.faces)

        return jax.vmap(_get_area)(face_id)


def _fate_selection(faceTable: NDArray, n_fates: int, rng: Generator) -> NDArray:
    n_cells = faceTable.size
    n_cells_per_fate = n_cells // n_fates
    n_cells_left = n_cells % n_fates
    cell_fates = np.repeat(np.arange(n_fates), n_cells_per_fate)
    cell_fates = np.concatenate([cell_fates, np.arange(n_cells_left)])
    rng.shuffle(cell_fates)
    return np.hstack([faceTable, cell_fates[:, None]])
