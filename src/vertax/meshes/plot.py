"""Plotting module for meshes."""

import colorsys
from collections.abc import Callable
from enum import Enum
from typing import Literal

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, to_rgb
from matplotlib.figure import Figure
from numpy.typing import NDArray

from vertax.meshes.bounded_mesh import BoundedMesh
from vertax.meshes.mesh import Mesh
from vertax.meshes.pbc_mesh import PbcMesh


class FacePlot(Enum):
    """What it is possible to show on a face."""

    MULTICOLOR = 1
    """Each face get a random color."""
    FACE_PARAMETER = 2
    """The face color depends on its parameter."""
    AREA = 3
    """The face color depends on its area."""
    PERIMETER = 4
    """The face color depends on its perimeter."""
    WHITE = 5
    """All faces are just white."""
    FATES = 6
    """Faces are colored depending on their fate marker (if any)."""


class EdgePlot(Enum):
    """What it is possible to show on an edge."""

    BLACK = 1
    """All edges are black."""
    EDGE_PARAMETER = 2
    """The edge color depends on its parameter."""
    LENGTH = 3
    """The edge color depends on its length."""
    INVISIBLE = 4
    """Do not show edges."""


class VertexPlot(Enum):
    """What it is possible to show on a vertex."""

    BLACK = 1
    """Vertices are black."""
    VERTEX_PARAMETER = 2
    """The vertex color depends on its parameter."""
    INVISIBLE = 3
    """Do not show vertices."""


def add_colorbar(fig: Figure, ax: Axes, v_min: float, v_max: float, cmap: Colormap) -> Colorbar:
    """Add a colorbar to the figure, with given min and max values and colormap."""
    fake_im = ax.imshow([[1]], vmin=v_min, vmax=v_max, cmap=cmap)
    colorbar = fig.colorbar(fake_im, ax=ax, shrink=0.7)
    fake_im.set_alpha(0)
    return colorbar


def adjust_lightness(color: tuple[float, float, float], amount: float = 0.5) -> tuple[float, float, float]:
    """Adjust lightness of a color."""
    color = colorsys.rgb_to_hls(*to_rgb(color))
    return colorsys.hls_to_rgb(color[0], max(0, min(1, amount * color[1])), color[2])


def get_cmap(n: int, name: str = "hsv") -> Callable[[int], tuple[float, float, float, float]]:
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color.

    The keyword argument name must be a standard mpl colormap name.
    """

    def cmap_with_n_colors(i: int) -> tuple[float, float, float, float]:
        cmap = matplotlib.colormaps.get_cmap(name)
        return cmap(i / (n - 1))

    return cmap_with_n_colors


def save_simple_xy_graph(
    filename: str, x_data: list[float], y_data: list[float], title: str = "", x_label: str = "", y_label: str = ""
) -> None:
    """Save a simple line plot."""
    fig, _ = plt.subplots(layout="constrained")
    ax = plt.gca()

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(x_data, y_data)
    plt.savefig(str(filename))
    plt.close(fig)


def plot_mesh(
    mesh: Mesh,
    vertex_plot: VertexPlot = VertexPlot.INVISIBLE,
    edge_plot: EdgePlot = EdgePlot.BLACK,
    face_plot: FacePlot = FacePlot.MULTICOLOR,
    vertex_parameters_name: str = "",
    edge_parameters_name: str = "",
    face_parameters_name: str = "",
    show: bool = True,
    save: bool = False,
    save_path: str = "pbc_mesh.png",
    faces_cmap_name: str = "cividis",
    edges_cmap_name: str = "coolwarm",
    edges_width: float = 2,
    vertices_cmap_name: str = "spring",
    vertices_size: float = 20,
    title: str = "",
    forced_vertex_scale: tuple[float, float] | None = None,
    forced_edge_scale: tuple[float, float] | None = None,
    forced_face_scale: tuple[float, float] | None = None,
) -> None:
    """Plot the mesh and decide to save and/or show the mesh or not."""
    if isinstance(mesh, PbcMesh):
        _plot_pbc_mesh(
            mesh,
            vertex_plot,
            edge_plot,
            face_plot,
            vertex_parameters_name,
            edge_parameters_name,
            face_parameters_name,
            show,
            save,
            save_path,
            faces_cmap_name,
            edges_cmap_name,
            edges_width,
            vertices_cmap_name,
            vertices_size,
            title,
            forced_vertex_scale,
            forced_edge_scale,
            forced_face_scale,
        )
    elif isinstance(mesh, BoundedMesh):
        _plot_bounded_mesh(
            mesh,
            vertex_plot,
            edge_plot,
            face_plot,
            vertex_parameters_name,
            edge_parameters_name,
            face_parameters_name,
            show,
            save,
            save_path,
            faces_cmap_name,
            edges_cmap_name,
            edges_width,
            vertices_cmap_name,
            vertices_size,
            title,
            forced_vertex_scale,
            forced_edge_scale,
            forced_face_scale,
        )


def get_plot_mesh(
    mesh: Mesh,
    vertex_plot: VertexPlot = VertexPlot.INVISIBLE,
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
    forced_vertex_scale: tuple[float, float] | None = None,
    forced_edge_scale: tuple[float, float] | None = None,
    forced_face_scale: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Get the matplotlib figure and and ax for one plot."""
    if isinstance(mesh, PbcMesh):
        return _get_plot_pbc_mesh(
            mesh,
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
            forced_vertex_scale,
            forced_edge_scale,
            forced_face_scale,
        )
    elif isinstance(mesh, BoundedMesh):
        return _get_plot_bounded_mesh(
            mesh,
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
            forced_vertex_scale,
            forced_edge_scale,
            forced_face_scale,
        )
    else:
        msg = f"Expected either a PbcMesh or a BoundedMesh. Got {mesh} instead."
        raise ValueError(msg)


def _plot_pbc_mesh(
    mesh: PbcMesh,
    vertex_plot: VertexPlot = VertexPlot.INVISIBLE,
    edge_plot: EdgePlot = EdgePlot.BLACK,
    face_plot: FacePlot = FacePlot.MULTICOLOR,
    vertex_parameters_name: str = "",
    edge_parameters_name: str = "",
    face_parameters_name: str = "",
    show: bool = True,
    save: bool = False,
    save_path: str = "pbc_mesh.png",
    faces_cmap_name: str = "cividis",
    edges_cmap_name: str = "coolwarm",
    edges_width: float = 2,
    vertices_cmap_name: str = "spring",
    vertices_size: float = 20,
    title: str = "",
    forced_vertex_scale: tuple[float, float] | None = None,
    forced_edge_scale: tuple[float, float] | None = None,
    forced_face_scale: tuple[float, float] | None = None,
) -> None:
    """Plot the mesh and decide to save and/or show the mesh or not."""
    fig, _ax = _get_plot_pbc_mesh(
        mesh,
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
        forced_vertex_scale,
        forced_edge_scale,
        forced_face_scale,
    )

    if save:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close(fig)


def _get_plot_pbc_mesh(
    mesh: PbcMesh,
    vertex_plot: VertexPlot = VertexPlot.INVISIBLE,
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
    forced_vertex_scale: tuple[float, float] | None = None,
    forced_edge_scale: tuple[float, float] | None = None,
    forced_face_scale: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Get the matplotlib figure and and ax for one plot."""
    # Fates not used for pbc.
    if face_plot == FacePlot.FATES:
        face_plot = FacePlot.WHITE

    fig, _ = plt.subplots(layout="constrained")
    ax = plt.gca()
    _plot_faces_pbc(mesh, fig, ax, face_plot, faces_cmap_name, face_parameters_name, forced_face_scale)
    _plot_edges_pbc(mesh, fig, ax, edge_plot, edges_cmap_name, edges_width, edge_parameters_name, forced_edge_scale)
    _plot_vertices_pbc(
        mesh, fig, ax, vertex_plot, vertices_cmap_name, vertices_size, vertex_parameters_name, forced_vertex_scale
    )

    ax.set_title(title)
    ax.set_aspect(mesh.height / mesh.width)
    ax.set_xlim(0, mesh.width)
    ax.set_ylim(0, mesh.height)

    return fig, ax


def _plot_faces_pbc(
    mesh: PbcMesh,
    fig: Figure,
    ax: Axes,
    face_plot: FacePlot,
    faces_cmap_name: str,
    face_parameters_name: str,
    forced_scale: tuple[float, float] | None = None,
) -> None:
    multicolor_cmap = _get_multicolor_face_cmap(mesh)
    faces_cmap = matplotlib.colormaps.get_cmap(faces_cmap_name)

    v_max = 1
    v_min = 0
    values = jnp.array([1])
    # set the correct colorbar if needed
    # Get values, min and max
    cbar_label = "Face parameter" if face_parameters_name == "" else face_parameters_name
    match face_plot:
        case FacePlot.FACE_PARAMETER:
            values = mesh.faces_params
        case FacePlot.AREA:
            values = mesh.get_area(jnp.arange(mesh.nb_faces))
            cbar_label = "Area of cell"
        case FacePlot.PERIMETER:
            values = mesh.get_perimeter(jnp.arange(mesh.nb_faces))
            cbar_label = "Perimeter of cell"
    if forced_scale is None:
        v_max = float(values.max())
        v_min = float(values.min())
    else:
        v_min, v_max = forced_scale
    match face_plot:
        case FacePlot.MULTICOLOR | FacePlot.WHITE:
            pass
        case _:
            cbar = add_colorbar(fig, ax, v_min, v_max, faces_cmap)
            cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
            cbar.ax.yaxis.set_ticks_position("left")

    # Draw each face
    for face in range(len(mesh.faces)):
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
        face_vertices = _get_face_vertices_vertices(mesh, face)
        _draw_face_pbc(mesh, ax, face_vertices, color)


def _get_face_vertices_vertices(mesh: PbcMesh, face: int) -> NDArray:
    """Get all vertices positions corresponding to a face."""
    start_he = mesh.faces[face]
    he = start_he

    source_vertex_id = mesh.edges[he][3]
    verts_sources = np.array([mesh.vertices[source_vertex_id]])
    # all_verts.append(mesh.vertices[v_source])

    he_offset_x = mesh.edges[he][6]
    he_offset_y = mesh.edges[he][7]
    sum0_offsets = he_offset_x
    sum1_offsets = he_offset_y

    he = mesh.edges[he][1]

    while he != start_he:
        source_vertex_id = mesh.edges[he][3]

        # all_verts.append(mesh.vertices[v_source])

        verts_sources = np.concatenate(
            (
                verts_sources,
                (
                    np.array([mesh.vertices[source_vertex_id]])
                    + np.array([sum0_offsets * mesh.width, sum1_offsets * mesh.height])
                ),
            ),
            axis=0,
        )

        he_offset_x = mesh.edges[he][6]
        he_offset_y = mesh.edges[he][7]
        sum0_offsets += he_offset_x
        sum1_offsets += he_offset_y

        he = mesh.edges[he][1]

    source_vertex_id = mesh.edges[he][3]
    verts_sources = np.concatenate((verts_sources, (np.array([mesh.vertices[source_vertex_id]]))), axis=0)
    return verts_sources


def _draw_face_pbc(
    mesh: PbcMesh, ax: Axes, face_vertices: NDArray, color: tuple[float, float, float, float] | NDArray
) -> None:
    x = face_vertices[:, 0]
    y = mesh.height - face_vertices[:, 1]

    ax.fill(x, y, color=color)
    ax.fill(np.add(x, mesh.width), np.add(y, mesh.height), color=color)
    ax.fill(np.add(x, -mesh.width), np.add(y, -mesh.height), color=color)
    ax.fill(np.add(x, -mesh.width), np.add(y, mesh.height), color=color)
    ax.fill(np.add(x, mesh.width), np.add(y, -mesh.height), color=color)
    ax.fill(x, np.add(y, mesh.height), color=color)
    ax.fill(x, np.add(y, -mesh.height), color=color)
    ax.fill(np.add(x, mesh.width), y, color=color)
    ax.fill(np.add(x, -mesh.width), y, color=color)


def _get_multicolor_face_cmap(mesh: Mesh) -> Callable[[int], tuple[float, float, float, float]]:
    def cmap_light_hsv(n: int) -> Callable[[int], tuple[float, float, float, Literal[1]]]:
        def light_hsv(i: int) -> tuple[float, float, float, Literal[1]]:
            fun: Callable[[int], tuple[float, float, float, float]] = get_cmap(n, name="hsv")
            return (*adjust_lightness(fun(i)[:3], 1.4), 1)

        return light_hsv

    return cmap_light_hsv(len(mesh.faces))


def _plot_edges_pbc(
    mesh: PbcMesh,
    fig: Figure,
    ax: Axes,
    edge_plot: EdgePlot,
    edges_cmap_name: str,
    edges_width: float,
    edge_parameters_name: str,
    forced_scale: tuple[float, float] | None = None,
) -> None:
    if edge_plot != EdgePlot.INVISIBLE:
        edge_params_cmap = matplotlib.colormaps.get_cmap(edges_cmap_name)
        # set the correct colorbar

        v_max = 1
        v_min = 0
        values = jnp.array([1])
        # set the correct colorbar if needed
        # Get values, min and max
        cbar_label = "Edge parameter" if edge_parameters_name == "" else edge_parameters_name
        match edge_plot:
            case EdgePlot.EDGE_PARAMETER:
                values = mesh.edges_params
            case EdgePlot.LENGTH:
                values = mesh.get_length(jnp.arange(2 * mesh.nb_edges))
                cbar_label = "Length of edge"
        if forced_scale is None:
            v_max = float(values.max())
            v_min = float(values.min())
        else:
            v_min, v_max = forced_scale

        match edge_plot:
            case EdgePlot.BLACK:
                pass
            case _:
                cbar = add_colorbar(fig, ax, v_min, v_max, edge_params_cmap)
                cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
                cbar.ax.yaxis.set_ticks_position("left")

        # Draw each edge
        for i, edge_entry in enumerate(mesh.edges):
            # Find correct color depending on chosen colormap
            match edge_plot:
                case EdgePlot.BLACK:
                    color = (0, 0, 0, 1)
                case _:
                    norm_val = 1 if v_max == v_min else (values[i] - v_min) / (v_max - v_min)
                    color = edge_params_cmap(norm_val)
            # Draw the edge with color
            _draw_edge_pbc(mesh, ax, edge_entry, color, edges_width)


def _draw_edge_pbc(
    mesh: PbcMesh,
    ax: Axes,
    edge_entry: tuple[int, int, int, int, int, int, int, int],
    color: tuple[float, float, float, float] | NDArray,
    edges_width: float,
) -> None:
    origin_id = edge_entry[3]
    target_id = edge_entry[4]
    he_offset_x = edge_entry[6]
    he_offset_y = edge_entry[7]

    origin = mesh.vertices[origin_id]
    target = mesh.vertices[target_id] + np.array([he_offset_x * mesh.width, he_offset_y * mesh.height])

    points = np.array([origin, target])
    x = points[:, 0]
    y = mesh.height - points[:, 1]
    ax.plot(x, y, color=color, linewidth=edges_width)


def _plot_vertices_pbc(
    mesh: PbcMesh,
    fig: Figure,
    ax: Axes,
    vertex_plot: VertexPlot,
    vertices_cmap_name: str,
    vertices_size: float,
    vertex_parameters_name: str,
    forced_scale: tuple[float, float] | None = None,
) -> None:
    if vertex_plot != VertexPlot.INVISIBLE:
        # set the correct colorbar
        vertices_params_cmap = matplotlib.colormaps.get_cmap(vertices_cmap_name)
        v_max = 1
        v_min = 0
        if vertex_plot == VertexPlot.VERTEX_PARAMETER:
            if forced_scale is None:
                v_max = float(mesh.vertices_params.max())
                v_min = float(mesh.vertices_params.min())
            else:
                v_min, v_max = forced_scale
            cbar = add_colorbar(fig, ax, v_min, v_max, vertices_params_cmap)
            cbar_label = "Vertex parameter" if vertex_parameters_name == "" else vertex_parameters_name
            cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
            cbar.ax.yaxis.set_ticks_position("left")
        # Draw each vertex
        for i, vertex in enumerate(mesh.vertices):
            # Find correct color depending on chosen colormap
            match vertex_plot:
                case VertexPlot.VERTEX_PARAMETER:
                    norm_val = 1 if v_max == v_min else (mesh.vertices_params[i] - v_min) / (v_max - v_min)
                    color = vertices_params_cmap(norm_val)
                case VertexPlot.BLACK:
                    color = (0, 0, 0, 1)
                case _:
                    color = (0, 0, 0, 0)
            # Draw the vertex with color
            ax.scatter(vertex[0], mesh.height - vertex[1], color=color, s=vertices_size)


def _plot_bounded_mesh(
    mesh: BoundedMesh,
    vertex_plot: VertexPlot = VertexPlot.INVISIBLE,
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
    forced_vertex_scale: tuple[float, float] | None = None,
    forced_edge_scale: tuple[float, float] | None = None,
    forced_face_scale: tuple[float, float] | None = None,
) -> None:
    """Plot the mesh and decide to save and/or show the mesh or not."""
    fig, _ax = _get_plot_bounded_mesh(
        mesh,
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
        forced_vertex_scale,
        forced_edge_scale,
        forced_face_scale,
    )

    if save:
        plt.savefig(save_path)  # format maybe should be left as a choice

    if show:
        plt.show()

    plt.close(fig)


def _get_plot_bounded_mesh(
    mesh: BoundedMesh,
    vertex_plot: VertexPlot = VertexPlot.INVISIBLE,
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
    forced_vertex_scale: tuple[float, float] | None = None,
    forced_edge_scale: tuple[float, float] | None = None,
    forced_face_scale: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Get the matplotlib figure and and ax for one plot."""
    fig, _ = plt.subplots(layout="constrained")
    ax = plt.gca()
    _plot_faces_bounded(mesh, fig, ax, face_plot, faces_cmap_name, face_parameters_name, forced_face_scale)
    _plot_edges_bounded(mesh, fig, ax, edge_plot, edges_cmap_name, edges_width, edge_parameters_name, forced_edge_scale)
    _plot_vertices_bounded(
        mesh, fig, ax, vertex_plot, vertices_cmap_name, vertices_size, vertex_parameters_name, forced_vertex_scale
    )

    ax.set_title(title)
    # unlike the pbc case, here is not easy to know a priori
    # what the limits of the progressively optimized cell cluster will be (across a stack of images)
    ax.set_xlim(-0.5, mesh.width + 0.5)
    ax.set_ylim(-0.5, mesh.height + 0.5)
    ax.set_aspect((mesh.height + 1) / (mesh.width + 1))

    return fig, ax


def _plot_faces_bounded(
    mesh: BoundedMesh,
    fig: Figure,
    ax: Axes,
    face_plot: FacePlot,
    faces_cmap_name: str,
    face_parameters_name: str,
    forced_scale: tuple[float, float] | None = None,
) -> None:
    multicolor_cmap = _get_multicolor_face_cmap(mesh)
    faces_cmap = matplotlib.colormaps.get_cmap(faces_cmap_name)

    v_max = 1
    v_min = 0
    values = jnp.array([1])
    # set the correct colorbar if needed
    # Get values, min and max
    cbar_label = "Face parameter" if face_parameters_name == "" else face_parameters_name
    match face_plot:
        case FacePlot.FACE_PARAMETER:
            values = mesh.faces_params
        case FacePlot.AREA:
            values = mesh.get_area(jnp.arange(mesh.nb_faces))
            cbar_label = "Area of cell"
        case FacePlot.PERIMETER:
            values = mesh.get_perimeter(jnp.arange(mesh.nb_faces))
            cbar_label = "Perimeter of cell"
        case FacePlot.FATES:
            values = mesh.faces[:, 1]
    if forced_scale is None:
        v_max = float(values.max())
        v_min = float(values.min())
    else:
        v_min, v_max = forced_scale
    match face_plot:
        case FacePlot.MULTICOLOR | FacePlot.WHITE | FacePlot.FATES:
            pass
        case _:
            cbar = add_colorbar(fig, ax, v_min, v_max, faces_cmap)
            cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
            cbar.ax.yaxis.set_ticks_position("left")

    # Draw each face
    for face in range(len(mesh.faces)):
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
        face_vertices = _get_face_vertices_bounded(mesh, face)
        ax.fill(face_vertices[:, 0], face_vertices[:, 1], color=color)


def _get_face_vertices_bounded(mesh: BoundedMesh, face_id: int) -> NDArray:
    """Get all vertices positions corresponding to a face."""
    face_vertices = []

    draw_curve_threshold = 0.01  # radians. Must be above 0 to avoid overcomplicating a simple plot
    # one will be 0, other real id offsetted by 2 as the first two values 0 and 1 are reserved
    # initialize the loop on every face's edges.
    start_he = int(mesh.faces[face_id][0])
    he = start_he

    # Do the while loop at least one time and add a max iter to "catch" bugs of infinite loop (should not happen !)
    do_while = False
    max_iter = 1000
    nb_iter = 0
    while not do_while or ((do_while and he != start_he) and nb_iter < max_iter):
        do_while = True  # Just for the first time
        nb_iter += 1
        # Always add origin point
        v_source_id = int(mesh.edges[he][3] + mesh.edges[he][5] - 2)
        pos_source = mesh.vertices[v_source_id]

        face_vertices.append(pos_source)
        ang = float(mesh.angles[he // 2])
        if mesh.edges[he][3] != 0 or ang <= draw_curve_threshold:
            # means it's an inside edge, we do not need to do anything,
            # as the target will be the source of next half-edge. Or the angle is too flat.
            pass
        else:  # means it's an outside edge so it might be drawn as an arc
            # one will be 1, other real id offsetted by 2 as the first two values 0 and 1 are reserved
            # so real_id + 2 - 1
            v_target_id = int(mesh.edges[he][4] + mesh.edges[he][6] - 3)
            pos_target = mesh.vertices[v_target_id]
            arc_points = _get_arc_with_n_points(ang, pos_source, pos_target)
            face_vertices.extend(arc_points)

        # Next edge for next loop
        he = int(mesh.edges[he][1])

    face_vertices.append(face_vertices[0])
    return np.array(face_vertices)


def _plot_edges_bounded(
    mesh: BoundedMesh,
    fig: Figure,
    ax: Axes,
    edge_plot: EdgePlot,
    edges_cmap_name: str,
    edges_width: float,
    edge_parameters_name: str,
    forced_scale: tuple[float, float] | None = None,
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
            case EdgePlot.EDGE_PARAMETER:
                values = mesh.edges_params
            case EdgePlot.LENGTH:
                values = mesh.get_length(jnp.arange(2 * mesh.nb_edges))
                cbar_label = "Length of edge"
        if forced_scale is None:
            v_max = float(values.max())
            v_min = float(values.min())
        else:
            v_min, v_max = forced_scale

        # set the correct colorbar if needed
        match edge_plot:
            case EdgePlot.BLACK:
                pass
            case _:
                cbar = add_colorbar(fig, ax, v_min, v_max, edge_params_cmap)
                cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
                cbar.ax.yaxis.set_ticks_position("left")

        # Draw each edge
        for i in range(len(mesh.edges)):
            he = i
            # Find correct color depending on chosen colormap
            match edge_plot:
                case EdgePlot.BLACK:
                    color = (0, 0, 0, 1)
                case _:
                    norm_val = 1 if v_max == v_min else (values[he] - v_min) / (v_max - v_min)
                    color = edge_params_cmap(norm_val)
            # Draw the edge with color
            _draw_edge_bounded(mesh, ax, he, color, edges_width)


def _draw_edge_bounded(
    mesh: BoundedMesh,
    ax: Axes,
    he: int,
    color: tuple[float, float, float, float] | NDArray,
    edges_width: float,
) -> None:
    draw_curve_threshold = 0.01  # radians. Must be above 0 to avoid overcomplicating a simple plot

    v_source_id = int(mesh.edges[he][3] + mesh.edges[he][5] - 2)
    v_target_id = int(mesh.edges[he][4] + mesh.edges[he][6] - 3)
    if v_source_id >= 0 and v_target_id >= 0:  # else = twin of surface edges
        pos_source = mesh.vertices[v_source_id]
        pos_target = mesh.vertices[v_target_id]
        ang = float(mesh.angles[he // 2])

        if mesh.edges[he][3] != 0 or ang <= draw_curve_threshold:
            points = np.array([pos_source, pos_target])
        else:
            points = np.vstack((pos_source, np.array(_get_arc_with_n_points(ang, pos_source, pos_target)), pos_target))
        x = points[:, 0]
        y = points[:, 1]
        ax.plot(x, y, color=color, linewidth=edges_width)


def _plot_vertices_bounded(
    mesh: BoundedMesh,
    fig: Figure,
    ax: Axes,
    vertex_plot: VertexPlot,
    vertices_cmap_name: str,
    vertices_size: float,
    vertex_parameters_name: str,
    forced_scale: tuple[float, float] | None = None,
) -> None:
    if vertex_plot != VertexPlot.INVISIBLE:
        # set the correct colorbar
        vertices_params_cmap = matplotlib.colormaps.get_cmap(vertices_cmap_name)
        v_max = 1
        v_min = 0
        if vertex_plot == VertexPlot.VERTEX_PARAMETER:
            if forced_scale is None:
                v_max = float(mesh.vertices_params.max())
                v_min = float(mesh.vertices_params.min())
            else:
                v_min, v_max = forced_scale
            cbar = add_colorbar(fig, ax, v_min, v_max, vertices_params_cmap)
            cbar_label = "Vertex parameter" if vertex_parameters_name == "" else vertex_parameters_name
            cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=13)
            cbar.ax.yaxis.set_ticks_position("left")
        # Draw each vertex
        for i, vertex in enumerate(mesh.vertices):
            # Find correct color depending on chosen colormap
            match vertex_plot:
                case VertexPlot.VERTEX_PARAMETER:
                    norm_val = 1 if v_max == v_min else (mesh.vertices_params[i] - v_min) / (v_max - v_min)
                    color = vertices_params_cmap(norm_val)
                case VertexPlot.BLACK:
                    color = (0, 0, 0, 1)
                case _:
                    color = (0, 0, 0, 0)
            # Draw the vertex with color
            ax.scatter(vertex[0], vertex[1], color=color, s=vertices_size)


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

    # Now we'll take nb_draw_points the points along the curved edge and close the figure.
    tau = 2 * np.pi
    if abs(ang_target - ang_source) > np.pi:  # can it happen ?
        if ang_source < ang_target:
            ang_source += tau
        else:
            ang_target += tau
    intermediate_angles = np.linspace(ang_source, ang_target, nb_draw_points, False)[1:]  # without both endpoints
    return [np.array(radius * np.array([np.cos(a), np.sin(a)]) + center) for a in intermediate_angles]
