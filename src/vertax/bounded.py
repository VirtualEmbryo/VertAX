"""Bounded mesh with arc circles for boundary cells."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import Voronoi

from vertax.geo import get_any_length, get_area_bounded, get_perimeter_bounded
from vertax.mesh import Mesh
from vertax.opt import BilevelOptimizationMethod
from vertax.opt_bounded import InnerLossFunction, OuterLossFunction, bilevel_opt_bounded, inner_opt_bounded
from vertax.plot import EdgePlot, FacePlot, VertexPlot, add_colorbar, adjust_lightness, get_cmap
from vertax.topo import do_not_update_T1, update_T1_bounded

if TYPE_CHECKING:
    from jax import Array
    from matplotlib.pylab import Generator
    from numpy.typing import NDArray


class BoundedMesh(Mesh):
    """Bounded mesh with arc circles for boundary cells."""

    def __init__(self) -> None:
        """Do not call the constructor."""
        super().__init__()
        self.angles = jnp.array([])
        self.angles_target = jnp.array([])
        self._update_T1_func = update_T1_bounded

    def save_mesh_txt(
        self,
        directory: str,
        vertices_filename: str = "vertTable.txt",
        angles_filename: str = "angTable.txt",
        edges_filename: str = "heTable.txt",
        faces_filename: str = "faceTable.txt",
    ) -> None:
        """Save a mesh in separate text files that can be read by numpy.

        Only save the vertices, angles, edges and faces, not other parameters.

        Args:
            directory (str): Path to the directory where to save the files.
            vertices_filename (str, optional): Filename for the vertices table. Defaults to "vertTable.txt".
            angles_filename (str, optional): Filename for the angles table. Defaults to "angTable.txt".
            edges_filename (str, optional): Filename for the half-edges table. Defaults to "heTable.txt".
            faces_filename (str, optional): Filename for the faces table. Defaults to "faceTable.txt".
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)
        vertpath = dirpath / vertices_filename
        angpath = dirpath / angles_filename
        hepath = dirpath / edges_filename
        facepath = dirpath / faces_filename
        np.savetxt(vertpath, self.vertices)
        np.savetxt(angpath, self.angles)
        np.savetxt(hepath, self.edges)
        np.savetxt(facepath, self.faces)

    def save_mesh(self, path: str) -> None:
        """Save mesh to a file.

        All BoundedMesh data is saved, except for the solvers that are not saved.

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
            vertices_target=self.vertices_target,
            edges_target=self.edges_target,
            faces_target=self.faces_target,
            angles_target=self.angles_target,
            image_target=self.image_target,
            bilevel_optimization_method=self.bilevel_optimization_method.value,
            beta=self.beta,
            min_dist_T1=self.min_dist_T1,
            max_nb_iterations=self.max_nb_iterations,
            tolerance=self.tolerance,
            patience=self.patience,
            update_t1=self.update_t1,
        )

    @classmethod
    def load_mesh(cls, path: str) -> Self:
        """Load a mesh from a file.

        All BoundedMesh data is reloaded, except for the solvers that are not saved.

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
        mesh.vertices_target = mesh_file["vertices_target"]
        mesh.edges_target = mesh_file["edges_target"]
        mesh.faces_target = mesh_file["faces_target"]
        mesh.angles_target = mesh_file["angles_target"]
        mesh.image_target = mesh_file["image_target"]
        mesh.bilevel_optimization_method = BilevelOptimizationMethod(mesh_file["bilevel_optimization_method"])
        mesh.beta = mesh_file["beta"]
        mesh.min_dist_T1 = mesh_file["min_dist_T1"]
        mesh.max_nb_iterations = mesh_file["max_nb_iterations"]
        mesh.tolerance = mesh_file["tolerance"]
        mesh.patience = mesh_file["patience"]
        mesh.update_t1 = mesh_file["update_t1"]
        return mesh

    @classmethod
    def load_mesh_txt(
        cls,
        directory: str,
        vertices_filename: str = "vertTable.txt",
        angles_filename: str = "angTable.txt",
        edges_filename: str = "heTable.txt",
        faces_filename: str = "faceTable.txt",
    ) -> Self:
        """Load a mesh from text files.

        Only load the vertices, angles, edges and faces, not other parameters.

        Args:
            directory (str): Directory where the text files are stored.
            vertices_filename (str, optional): Filename for the vertices table. Defaults to "vertTable.txt".
            angles_filename (str, optional): Filename for the angles table. Defaults to "angTable.txt".
            edges_filename (str, optional): Filename for the half-edges table. Defaults to "heTable.txt".
            faces_filename (str, optional): Filename for the faces table. Defaults to "faceTable.txt".

        Returns:
            Self: The loaded mesh.
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)
        vertpath = dirpath / vertices_filename
        angpath = dirpath / angles_filename
        hepath = dirpath / edges_filename
        facepath = dirpath / faces_filename

        mesh = cls._create()
        mesh.vertices = jnp.array(np.loadtxt(vertpath, dtype=np.float64))
        mesh.angles = jnp.array(np.loadtxt(angpath, dtype=np.float64))
        mesh.edges = jnp.array(np.loadtxt(hepath, dtype=np.int64))
        mesh.faces = jnp.array(np.loadtxt(facepath, dtype=np.int64))
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
        mesh._update_T1_func = other_mesh._update_T1_func

        return mesh

    @property
    def udpate_t1(self) -> bool:
        """Whether or not update the mesh by applying T1 transitions."""
        return self._update_T1_func != do_not_update_T1

    @udpate_t1.setter
    def update_t1(self, b: bool) -> None:
        """Whether or not update the mesh by applying T1 transitions."""
        if b:
            self._update_T1_func = update_T1_bounded
        else:
            self._update_T1_func = do_not_update_T1

    @property
    def nb_angles(self) -> int:
        """Get the number of angles of the mesh."""
        return len(self.angles)

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
            update_T1_func=self._update_T1_func,
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
            update_T1_func=self._update_T1_func,
        )

        return list(loss_history)

    def plot(
        self,
        vertex_plot: VertexPlot = VertexPlot.BLACK,
        edge_plot: EdgePlot = EdgePlot.BLACK,
        face_plot: FacePlot = FacePlot.MULTICOLOR,
        vertex_parameters_name: str = "",
        edge_parameters_name: str = "",
        face_parameters_name: str = "",
        show: bool = True,
        save: bool = False,
        save_path: str = "bounded_mesh.png",
        faces_cmap_name: str = "cividis",
        edges_cmap_name: str = "coolwarm",
        edges_width: float = 2,
        vertices_cmap_name: str = "spring",
        vertices_size: float = 20,
        title: str = "",
    ) -> None:
        """Plot the mesh and decide to save and/or show the mesh or not."""
        fig, _ax = self.get_plot(
            vertex_plot,
            edge_plot,
            face_plot,
            vertex_parameters_name,
            edge_parameters_name,
            face_parameters_name,
            faces_cmap_name,
            edges_cmap_name,
            edges_width,
            vertices_cmap_name,
            vertices_size,
            title,
        )

        if save:
            plt.savefig(save_path)  # format maybe should be left as a choice

        if show:
            plt.show()

        plt.close(fig)

    def get_plot(
        self,
        vertex_plot: VertexPlot = VertexPlot.BLACK,
        edge_plot: EdgePlot = EdgePlot.BLACK,
        face_plot: FacePlot = FacePlot.MULTICOLOR,
        vertex_parameters_name: str = "",
        edge_parameters_name: str = "",
        face_parameters_name: str = "",
        faces_cmap_name: str = "cividis",
        edges_cmap_name: str = "coolwarm",
        edges_width: float = 2,
        vertices_cmap_name: str = "spring",
        vertices_size: float = 20,
        title: str = "",
    ) -> tuple[Figure, Axes]:
        """Get the matplotlib figure and and ax for one plot."""
        fig, _ = plt.subplots(layout="constrained")
        ax = plt.gca()
        self._plot_faces(fig, ax, face_plot, faces_cmap_name, face_parameters_name)
        self._plot_edges(fig, ax, edge_plot, edges_cmap_name, edges_width, edge_parameters_name)
        self._plot_vertices(fig, ax, vertex_plot, vertices_cmap_name, vertices_size, vertex_parameters_name)

        ax.set_title(title)
        # unlike the pbc case, here is not easy to know a priori
        # what the limits of the progressively optimized cell cluster will be (across a stack of images)
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height + 0.5)
        ax.set_aspect((self.height + 1) / (self.width + 1))

        return fig, ax

    def _plot_faces(
        self, fig: Figure, ax: Axes, face_plot: FacePlot, faces_cmap_name: str, face_parameters_name: str
    ) -> None:
        multicolor_cmap = self._get_multicolor_face_cmap()
        faces_cmap = matplotlib.colormaps.get_cmap(faces_cmap_name)

        v_max = 1
        v_min = 0
        values = jnp.array([1])
        # set the correct colorbar if needed
        # Get values, min and max
        cbar_label = "Face parameter" if face_parameters_name == "" else face_parameters_name
        match face_plot:
            case FacePlot.FACE_PAREMETER:
                values = self.faces_params
            case FacePlot.AREA:
                values = self.get_area(jnp.arange(self.nb_faces))
                cbar_label = "Area of cell"
            case FacePlot.PERIMETER:
                values = self.get_perimeter(jnp.arange(self.nb_faces))
                cbar_label = "Perimeter of cell"
            case FacePlot.FATES:
                values = self.faces[:, 1]
        v_max = float(values.max())
        v_min = float(values.min())
        match face_plot:
            case FacePlot.MULTICOLOR | FacePlot.WHITE | FacePlot.FATES:
                pass
            case _:
                cbar = add_colorbar(fig, ax, v_min, v_max, faces_cmap)
                cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
                cbar.ax.yaxis.set_ticks_position("left")

        # Draw each face
        for face in range(len(self.faces)):
            match face_plot:
                # Find correct color depending on chosen colormap
                case FacePlot.MULTICOLOR:
                    color = multicolor_cmap(face)
                case FacePlot.WHITE:
                    color = (1, 1, 1, 1)
                case _:
                    norm_val = 1 if v_max == v_min else (values[face] - v_min) / (v_max - v_min)
                    color = faces_cmap(norm_val)
            # Find face's vertices and draw the corresponding polygon.
            face_vertices = self._get_face_vertices(face)
            ax.fill(face_vertices[:, 0], face_vertices[:, 1], color=color)

    def _get_face_vertices(self, face_id: int) -> NDArray:
        """Get all vertices positions corresponding to a face."""
        face_vertices = []

        draw_curve_threshold = 0.01  # radians. Must be above 0 to avoid overcomplicating a simple plot
        # one will be 0, other real id offseted by 2 as the first two values 0 and 1 are reserved
        # initialize the loop on every face's edges.
        start_he = int(self.faces[face_id][0])
        he = start_he

        do_while = False
        while not do_while or (do_while and he != start_he):
            do_while = True  # Just for the first time
            # Always add origin point
            v_source_id = int(self.edges[he][3] + self.edges[he][5] - 2)
            pos_source = self.vertices[v_source_id]

            face_vertices.append(pos_source)
            ang = float(self.angles[he // 2])
            if self.edges[he][3] != 0 or ang <= draw_curve_threshold:
                # means it's an inside edge, we do not need to do anything,
                # as the target will be the source of next half-edge. Or the angle is too flat.
                pass
            else:  # means it's an outside edge so it might be drawn as an arc
                pass
                # one will be 1, other real id offseted by 2 as the first two values 0 and 1 are reserved
                # so real_id + 2 - 1
                v_target_id = int(self.edges[he][4] + self.edges[he][6] - 3)
                pos_target = self.vertices[v_target_id]
                arc_points = _get_arc_with_n_points(ang, pos_source, pos_target)
                face_vertices.extend(arc_points)

            # Next edge for next loop
            he = int(self.edges[he][1])

        face_vertices.append(face_vertices[0])
        return np.array(face_vertices)

    def _get_multicolor_face_cmap(self) -> Callable[[int], tuple[float, float, float, float]]:
        def cmap_light_hsv(n: int) -> Callable[[int], tuple[float, float, float, Literal[1]]]:
            def light_hsv(i: int) -> tuple[float, float, float, Literal[1]]:
                fun: Callable[[int], tuple[float, float, float, float]] = get_cmap(n, name="hsv")
                return (*adjust_lightness(fun(i)[:3], 1.4), 1)

            return light_hsv

        return cmap_light_hsv(len(self.faces))

    def _plot_edges(
        self,
        fig: Figure,
        ax: Axes,
        edge_plot: EdgePlot,
        edges_cmap_name: str,
        edges_width: float,
        edge_parameters_name: str,
    ) -> None:
        if edge_plot != EdgePlot.INVISIBLE:
            edge_params_cmap = matplotlib.colormaps.get_cmap(edges_cmap_name)
            # set the correct colorbar

            v_max = 1
            v_min = 0
            values = jnp.array([1])
            # Get values, min and max
            cbar_label = "Edge parameter" if edge_parameters_name == "" else edge_parameters_name
            match edge_plot:
                case EdgePlot.EDGE_PAREMETER:
                    values = self.edges_params
                case EdgePlot.LENGTH:
                    values = self.get_length(jnp.arange(2 * self.nb_edges))
                    cbar_label = "Length of edge"
            v_max = float(values.max())
            v_min = float(values.min())

            # set the correct colorbar if needed
            match edge_plot:
                case EdgePlot.BLACK:
                    pass
                case _:
                    cbar = add_colorbar(fig, ax, v_min, v_max, edge_params_cmap)
                    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
                    cbar.ax.yaxis.set_ticks_position("left")

            # Draw each edge
            for i in range(len(self.edges)):
                he = i
                # Find correct color depending on chosen colormap
                match edge_plot:
                    case EdgePlot.BLACK:
                        color = (0, 0, 0, 1)
                    case _:
                        norm_val = 1 if v_max == v_min else (values[he] - v_min) / (v_max - v_min)
                        color = edge_params_cmap(norm_val)
                # Draw the edge with color
                self._draw_edge(ax, he, color, edges_width)

    def _draw_edge(
        self,
        ax: Axes,
        he: int,
        color: tuple[float, float, float, float] | NDArray,
        edges_width: float,
    ) -> None:
        draw_curve_threshold = 0.01  # radians. Must be above 0 to avoid overcomplicating a simple plot

        v_source_id = int(self.edges[he][3] + self.edges[he][5] - 2)
        v_target_id = int(self.edges[he][4] + self.edges[he][6] - 3)
        if v_source_id >= 0 and v_target_id >= 0:  # else = twin of surface edges
            pos_source = self.vertices[v_source_id]
            pos_target = self.vertices[v_target_id]
            ang = float(self.angles[he // 2])

            if self.edges[he][3] != 0 or ang <= draw_curve_threshold:
                points = np.array([pos_source, pos_target])
            else:
                points = np.vstack(
                    (pos_source, np.array(_get_arc_with_n_points(ang, pos_source, pos_target)), pos_target)
                )
            x = points[:, 0]
            y = points[:, 1]
            ax.plot(x, y, color=color, linewidth=edges_width)

    def _plot_vertices(
        self,
        fig: Figure,
        ax: Axes,
        vertex_plot: VertexPlot,
        vertices_cmap_name: str,
        vertices_size: float,
        vertex_parameters_name: str,
    ) -> None:
        if vertex_plot != VertexPlot.INVISIBLE:
            # set the correct colorbar
            vertices_params_cmap = matplotlib.colormaps.get_cmap(vertices_cmap_name)
            v_max = 1
            v_min = 0
            if vertex_plot == VertexPlot.VERTEX_PAREMETER:
                v_max = float(self.vertices_params.max())
                v_min = float(self.vertices_params.min())
                cbar = add_colorbar(fig, ax, v_min, v_max, vertices_params_cmap)
                cbar_label = "Vertex parameter" if vertex_parameters_name == "" else vertex_parameters_name
                cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
                cbar.ax.yaxis.set_ticks_position("left")
            # Draw each vertex
            for i, vertex in enumerate(self.vertices):
                # Find correct color depending on chosen colormap
                match vertex_plot:
                    case VertexPlot.VERTEX_PAREMETER:
                        norm_val = 1 if v_max == v_min else (self.vertices_params[i] - v_min) / (v_max - v_min)
                        color = vertices_params_cmap(norm_val)
                    case VertexPlot.BLACK:
                        color = (0, 0, 0, 1)
                    case _:
                        color = (0, 0, 0, 0)
                # Draw the vertex with color
                ax.scatter(vertex[0], vertex[1], color=color, s=vertices_size)


def _fate_selection(faceTable: NDArray, n_fates: int, rng: Generator) -> NDArray:
    n_cells = faceTable.size
    n_cells_per_fate = n_cells // n_fates
    n_cells_left = n_cells % n_fates
    cell_fates = np.repeat(np.arange(n_fates), n_cells_per_fate)
    cell_fates = np.concatenate([cell_fates, np.arange(n_cells_left)])
    rng.shuffle(cell_fates)
    return np.hstack([faceTable, cell_fates[:, None]])


def _get_arc_with_n_points(ang: float, pos_source: Array, pos_target: Array, nb_draw_points: int = 40) -> list[NDArray]:
    edge_vector = pos_target - pos_source
    edge_half_length = np.linalg.norm(edge_vector) / 2
    # shouldn't it be ang/2 ? No, ang is between 0 and pi/2, it's already divided by 2 (if you ask me)
    radius = edge_half_length / np.sin(ang)
    # diameter = 2 * radius
    d_midpoint_to_center = np.sqrt(radius**2 - edge_half_length**2)
    midpoint = (pos_source + pos_target) / 2
    unit_vector = edge_vector / np.linalg.norm(edge_vector)
    unit_vector_midpoint_to_center = np.array([-unit_vector[1], unit_vector[0]])
    center = midpoint + unit_vector_midpoint_to_center * d_midpoint_to_center
    # angles in radians between -pi and pi
    ang_source = np.angle(np.dot((pos_source - center), np.array([1, 1j])))  # angle of a + bi
    ang_target = np.angle(np.dot((pos_target - center), np.array([1, 1j])))
    # rad2deg = 180 / np.pi
    # # angle in degrees between -180 and 180
    # normalized_ang_source_degrees = ang_source * rad2deg
    # normalized_ang_target_degrees = ang_target * rad2deg
    # # angle in degrees between 0 and 360
    # if normalized_ang_source_degrees < 0.0:
    #     normalized_ang_source_degrees += 360
    # if normalized_ang_target_degrees < 0.0:
    #     normalized_ang_target_degrees += 360

    # # draw the curved edge
    # if lines:
    #     surface_arc = Arc(
    #         center,
    #         diameter,
    #         diameter,
    #         theta1=normalized_ang_source_degrees,
    #         theta2=normalized_ang_target_degrees,
    #         color="black",
    #         linewidth=2.0,
    #     )
    #     plt.gca().add_patch(surface_arc)

    # Now we'll take nb_draw_points the points along the curved edge and close the figure.
    tau = 2 * np.pi
    if abs(ang_target - ang_source) > np.pi:  # can it happen ?
        if ang_source < ang_target:
            ang_source += tau
        else:
            ang_target += tau
    intermediate_angles = np.linspace(ang_source, ang_target, nb_draw_points, False)[1:]  # without both endpoints
    # points = [pos_source]
    # points.extend([radius * np.array([np.cos(a), np.sin(a)]) + center for a in intermediate_angles])
    # points.append(pos_target)
    # points.append(pos_source)  # make it a closed form
    # points = np.array(points)
    # x, y = points[:, 0], points[:, 1]
    return [np.array(radius * np.array([np.cos(a), np.sin(a)]) + center) for a in intermediate_angles]
