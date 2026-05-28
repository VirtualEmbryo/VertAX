"""Simple forward test for the periodic case."""

# Package imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import jax.numpy as jnp
import numpy as np

from vertax import BoundedMesh, EdgePlot, FacePlot, PbcMesh, VertexPlot, plot_mesh


def test_plot() -> None:
    """Check the plot functions (manual)."""
    # Settings
    n_cells = 10
    # Initial condition
    L_box = jnp.sqrt(n_cells)
    width = float(L_box)
    height = float(L_box)
    bounded_mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)
    rng = np.random.default_rng(1337)
    bounded_mesh.vertices_params = jnp.array(rng.random(bounded_mesh.nb_vertices) * 82 + 14)
    bounded_mesh.edges_params = jnp.array(rng.random(bounded_mesh.nb_edges) * 3 + 1)
    bounded_mesh.faces_params = jnp.array(rng.random(bounded_mesh.nb_faces) * 7 + 3)
    plot_mesh(
        mesh=bounded_mesh,
        vertex_plot=VertexPlot.INVISIBLE,
        edge_plot=EdgePlot.LENGTH,
        face_plot=FacePlot.PERIMETER,
        vertex_parameters_name="Random vertex parameter",
        title="Too much colorbar is possible !",
    )

    pbc_mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)
    rng = np.random.default_rng(1337)
    pbc_mesh.vertices_params = jnp.array(rng.random(pbc_mesh.nb_vertices) * 82 + 14)
    pbc_mesh.edges_params = jnp.array(rng.random(pbc_mesh.nb_edges) * 3 + 1)
    pbc_mesh.faces_params = jnp.array(rng.random(pbc_mesh.nb_faces) * 7 + 3)
    plot_mesh(
        mesh=pbc_mesh,
        vertex_plot=VertexPlot.VERTEX_PARAMETER,
        edge_plot=EdgePlot.LENGTH,
        face_plot=FacePlot.AREA,
        vertex_parameters_name="Random vertex parameter",
        title="Too much colorbar is possible !",
    )


if __name__ == "__main__":
    test_plot()
