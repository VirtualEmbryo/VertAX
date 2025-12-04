"""Bounded mesh with arc circles for boundary cells."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import jax.numpy as jnp
import numpy as np
from scipy.spatial import Voronoi

from vertax.geo import get_area_bounded, get_edge_length, get_surface_length
from vertax.mesh import Mesh
from vertax.opt_bounded import InnerLossFunction, OuterLossFunction, bilevel_opt_bounded, inner_opt_bounded

if TYPE_CHECKING:
    from matplotlib.pylab import Generator
    from numpy.typing import NDArray


class BoundedMesh(Mesh):
    """Bounded mesh with arc circles for boundary cells."""

    def __init__(self) -> None:
        """Do not call the constructor."""
        super().__init__()
        self.angles = jnp.array([])
        self.angles_target = jnp.array([])

    def save_mesh(self, path: str) -> None:
        """Save mesh to a file.

        Args:
            path (str): Path to the saved file. The extension is .npz.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, allow_pickle=False, vertices=self.vertices, edges=self.edges, faces=self.faces, angles=self.angles
        )

    @classmethod
    def load_mesh(cls, path: str) -> Self:
        """Load a mesh from a file.

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
        return mesh

    @classmethod
    def copy_mesh(cls, other_mesh: Self) -> Self:
        """Copy all parameters from another mesh in a new mesh."""
        mesh = cls._create()
        mesh.vertices = other_mesh.vertices.copy()
        mesh.edges = other_mesh.edges.copy()
        mesh.faces = other_mesh.faces.copy()
        mesh.angles = other_mesh.angles.copy()
        mesh.angles_target = other_mesh.angles_target.copy()
        mesh.width = other_mesh.width
        mesh.height = other_mesh.height
        mesh.vertices_params = other_mesh.vertices_params.copy()
        mesh.edges_params = other_mesh.edges_params.copy()
        mesh.faces_params = other_mesh.faces_params.copy()
        mesh.vertices_target = other_mesh.vertices_target.copy()
        mesh.edges_target = other_mesh.edges_target.copy()
        mesh.faces_target = other_mesh.faces_target.copy()
        mesh.image_target = other_mesh.image_target.copy()
        mesh.bilevel_optimization_method = other_mesh.bilevel_optimization_method
        mesh.beta = other_mesh.beta
        mesh.min_dist_T1 = other_mesh.min_dist_T1
        mesh.max_nb_iterations = other_mesh.max_nb_iterations
        mesh.tolerance = other_mesh.tolerance
        mesh.patience = other_mesh.patience
        mesh.inner_solver = other_mesh.inner_solver
        mesh.outer_solver = other_mesh.outer_solver

        return mesh

    @property
    def nb_angles(self) -> int:
        """Get the number of angles of the mesh."""
        return len(self.angles)

    def get_length(self, half_edge_id: int) -> float:
        """Get the length of an edge."""
        return get_edge_length(half_edge_id, self.vertices, self.edges) + get_surface_length(
            half_edge_id, self.vertices, self.angles, self.edges
        )

    def get_area(self, face_id: int) -> float:
        """Get the area of a face."""
        return float(
            get_area_bounded(face_id, self.vertices, self.edges, self.faces, self.width, self.height)
        )  # if bug maybe remove the conversion to a float.

    @classmethod
    def from_random_seeds(cls, nb_seeds: int, width: float, height: float, random_key: int) -> Self:
        """Create a bounded Mesh from random seeds.

        Args:
            nb_seeds (int): Number of random seeds to use.
            width (float): width of the box containing the seeds.
            height (float): height of the box containing the seeds.
            random_key (int): seed for a random number generator to add new seeds if needed, decide on cell fates...

        Returns:
            Self: The corresponding mesh.
        """
        rng = np.random.default_rng(seed=random_key)
        seeds = (width, height) * rng.random((nb_seeds, 2))
        return cls.from_seeds(seeds, width, height, random_key)

    @classmethod
    def from_seeds(cls, seeds: NDArray, width: float, height: float, random_key: int) -> Self:  # noqa: C901
        """Create a bounded Mesh from a list of seeds.

        The seeds are assumed to have x-coordinate in ]0, width[ and y-coordinate in ]0, height[.

        Args:
            seeds (Array[float32]): jax float array of seed positions of shape (nbSeeds, 2).
            width (float): width of the box containing the seeds.
            height (float): height of the box containing the seeds.
            random_key (int): seed for a random number generator to add new seeds if needed, decide on cell fates...
        """
        rng = np.random.default_rng(seed=random_key)
        n_cells = len(seeds)  # starting number of seeds must be equal to the desired number of cells (faces)

        while True:
            success = 0
            voronoi = Voronoi(seeds)

            vertices = voronoi.vertices
            edges = voronoi.ridge_vertices
            faces = voronoi.regions

            inbound_faces = []
            inbound_vertices = np.zeros(vertices.shape[0], dtype=np.int32)
            for face in faces:
                if face and all(item > -1 for item in face):  # the face must not be an empty list
                    face_vertices_positions = vertices[face]
                    if (
                        np.all(face_vertices_positions[:, 0] < width)
                        and np.all(face_vertices_positions[:, 1] < height)
                        and np.all(face_vertices_positions > 0)
                    ):
                        inbound_faces.append(face)
                        inbound_vertices[face] += 1

            # getting rid of faces connected to a single other inbound face
            # (these can be problematic and lead to many special cases later on)
            while True:
                num_infaces = len(inbound_faces)
                del_count = 0
                for i, face in enumerate(reversed(inbound_faces)):
                    if np.sum(inbound_vertices[face] > 1) <= 2:
                        inbound_vertices[face] -= 1
                        del inbound_faces[num_infaces - i - 1]
                        del_count += 1
                if del_count == 0:
                    break

            if num_infaces < n_cells:
                success = 1
            elif num_infaces > n_cells:
                success = 2
            else:
                for i, face in enumerate(inbound_faces):
                    useful_vertices = []
                    extra_edges = []
                    last_useful = -1
                    incomplete_new_edge = False
                    new_edge = []  # now there
                    for vertex in face:
                        if inbound_vertices[vertex] == 1:
                            if not incomplete_new_edge:
                                # new_edge = []  # previously here
                                new_edge.append(last_useful)
                                incomplete_new_edge = True
                        else:
                            useful_vertices.append(vertex)
                            last_useful = vertex
                            if incomplete_new_edge:
                                new_edge.append(vertex)
                                extra_edges.append(new_edge)
                                incomplete_new_edge = False
                    if extra_edges and extra_edges[0][0] == -1:
                        extra_edges[0][0] = useful_vertices[-1]
                    elif incomplete_new_edge:
                        new_edge.append(useful_vertices[0])
                        extra_edges.append(new_edge)
                    edges.extend(extra_edges)
                    inbound_faces[i] = tuple(sorted(useful_vertices))
                useful_vertices_set = set(np.where(inbound_vertices > 1)[0])

                # HALF EDGE DATA STRUCTURE

                # reciprocating edges

                useful_edges = [tuple(sorted(e)) for e in edges if set(e).issubset(useful_vertices_set)]

                # failing to abide by the following relation results in disconnected topologies
                if len(useful_edges) != (n_cells - 1) * 3:
                    success = 2
                else:
                    half_edges = []
                    for e in useful_edges:
                        half_edges.append(e)
                        half_edges.append((e[1], e[0]))

                    # finding clockwise (or counterclockwise) half edge set for each face

                    ordered_edges_inbound_faces = []
                    for face in inbound_faces:
                        ### I think scipy give you everything already ordered for finite faces
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
                            if e[1] == start_edge[1] and e not in visited:
                                ordered_face.append((e[1], e[0]))
                                start_edge = (e[1], e[0])
                                visited.append(e)
                            i += 1
                            e = edges_face[i % len(edges_face)]

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
                    faceTable = _fate_selection(faceTable, 2, rng)

                    L_he = len(half_edges)
                    heTable = np.zeros((L_he, 8), dtype=np.int32)
                    heTable[:, 4] = 1
                    heTable[:, 6] = 1
                    relevant_twins = []
                    for i, he in enumerate(half_edges):
                        belongs_to_any_face = False
                        for hedges_face in ordered_edges_inbound_faces:
                            if he in hedges_face:
                                idx = hedges_face.index(he)
                                heTable[i][0] = half_edges.index(hedges_face[(idx - 1) % len(hedges_face)])  # he_prev
                                heTable[i][1] = half_edges.index(hedges_face[(idx + 1) % len(hedges_face)])  # he_next
                                heTable[i][3] = useful_vertices_list.index(he[0]) + 2  # vert source inner edges
                                heTable[i][4] = useful_vertices_list.index(he[1]) + 2  # vert target inner edges
                                heTable[i][7] = ordered_edges_inbound_faces.index(hedges_face)  # face
                                belongs_to_any_face = True
                                break
                        twin_idx = half_edges.index((he[1], he[0]))
                        heTable[i][2] = twin_idx  # he twin
                        if not belongs_to_any_face:
                            relevant_twins.append(twin_idx)

                    angTable = np.ones(L_he // 2)
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

            seeds = (
                np.vstack([seeds, (width, height) * rng.random((1, 2))])
                if success == 1
                else (width, height) * rng.random((n_cells, 2))
            )  # type: ignore

    def inner_opt(
        self,
        loss_function_inner: InnerLossFunction,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> list[float]:
        """Optimize the mesh for the loss function given.

        Args:
            loss_function_inner (InnerLossFunction): Loss function to optimize.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.

        Returns:
            list[float]: History of loss values during optimization.
        """
        selected_vertices, selected_edges, selected_faces = None, None, None
        if only_on_vertices is not None:
            selected_vertices = jnp.array(only_on_vertices)
        if only_on_edges is not None:
            selected_edges = jnp.array(only_on_edges)
        if only_on_faces is not None:
            selected_faces = jnp.array(only_on_faces)

        (self.vertices, self.angles, self.edges, self.faces), energies = inner_opt_bounded(
            self.vertices,
            self.angles,
            self.edges,
            self.faces,
            self.vertices_params,
            self.edges_params,
            self.faces_params,
            loss_function_inner,
            self.inner_solver,
            self.min_dist_T1,
            self.max_nb_iterations,
            self.tolerance,
            self.patience,
            selected_verts=selected_vertices,
            selected_hes=selected_edges,
            selected_faces=selected_faces,
        )
        return list(energies)

    def bilevel_opt(
        self,
        loss_function_inner: InnerLossFunction,
        loss_function_outer: OuterLossFunction,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> list[float]:
        """Optimize the mesh for the loss function given.

        Args:
            loss_function_inner (InnerLossFunction): Loss function to optimize.
            loss_function_outer (OuterLossFunction): Loss function to optimize.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.

        Returns:
            list[float]: History of loss values during optimization.
        """
        selected_vertices, selected_edges, selected_faces = None, None, None
        if only_on_vertices is not None:
            selected_vertices = jnp.array(only_on_vertices)
        if only_on_edges is not None:
            selected_edges = jnp.array(only_on_edges)
        if only_on_faces is not None:
            selected_faces = jnp.array(only_on_faces)
        (
            (
                self.vertices,
                self.angles,
                self.edges,
                self.faces,
                self.vertices_params,
                self.edges_params,
                self.faces_params,
            ),
            loss_history,
        ) = bilevel_opt_bounded(
            vertTable=self.vertices,
            angTable=self.angles,
            heTable=self.edges,
            faceTable=self.faces,
            vert_params=self.vertices_params,
            he_params=self.edges_params,
            face_params=self.faces_params,
            vertTable_target=self.vertices_target,
            angTable_target=self.angles_target,
            heTable_target=self.edges_target,
            faceTable_target=self.faces_target,
            L_in=loss_function_inner,
            L_out=loss_function_outer,
            solver_inner=self.inner_solver,
            solver_outer=self.outer_solver,
            min_dist_T1=self.min_dist_T1,
            iterations_max=self.max_nb_iterations,
            tolerance=self.tolerance,
            patience=self.patience,
            selected_verts=selected_vertices,
            selected_hes=selected_edges,
            selected_faces=selected_faces,
            image_target=self.image_target,
            beta=self.beta,
            optimization_method=self.bilevel_optimization_method,
        )

        return list(loss_history)


def _fate_selection(faceTable: NDArray, n_fates: int, rng: Generator) -> NDArray:
    n_cells = faceTable.size
    n_cells_per_fate = n_cells // n_fates
    n_cells_left = n_cells % n_fates
    cell_fates = np.repeat(np.arange(n_fates), n_cells_per_fate)
    cell_fates = np.concatenate([cell_fates, np.arange(n_cells_left)])
    rng.shuffle(cell_fates)
    return np.hstack([faceTable, cell_fates[:, None]])
