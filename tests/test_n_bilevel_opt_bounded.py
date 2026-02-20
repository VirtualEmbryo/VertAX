"""Test the whole pipeline of n bilevel optimization with the new API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import optax

from vertax import BoundedBilevelOptimizer, BoundedMesh
from vertax.cost import cost_ratio
from vertax.energy import energy_bounded
from vertax.method_enum import BilevelOptimizationMethod

if TYPE_CHECKING:
    pass


def test_n_bilevel_opt() -> None:
    """Check the whole function of n bilevel opt which makes a lot of things."""
    # Settings
    n_cells = 20
    n_edges = (n_cells - 1) * 3
    epochs = 10
    L_box = jnp.sqrt(n_cells)
    width = float(L_box)
    height = float(L_box)

    # Set periodic boundary mesh and some of its properties
    rng_seed = 2
    bounded_mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=rng_seed)
    # Note: those are base values so the following can be omitted
    bilevel_optimizer = BoundedBilevelOptimizer()
    bilevel_optimizer.min_dist_T1 = 0.005
    bilevel_optimizer.max_nb_iterations = 1000
    bilevel_optimizer.tolerance = 1e-4
    bilevel_optimizer.patience = 5
    bilevel_optimizer.inner_solver = optax.sgd(learning_rate=0.01)  # inner solver
    bilevel_optimizer.outer_solver = optax.adam(learning_rate=0.0001, nesterov=True)  # outer solver
    bilevel_optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
    bilevel_optimizer.beta = 0.01

    # Initial condition (parameters)
    rng = np.random.default_rng(seed=rng_seed)
    bounded_mesh.vertices_params = jnp.asarray([0.0 for _ in range(bounded_mesh.nb_vertices)])
    bounded_mesh.edges_params = jnp.asarray(rng.random(n_edges) * 20 - 10)
    bounded_mesh.faces_params = jnp.asarray([0.0 for _ in range(bounded_mesh.nb_faces)])

    # Energy minimization (init cond equilibrium)
    bilevel_optimizer.loss_function_inner = energy_bounded
    bilevel_optimizer.inner_optimization(mesh=bounded_mesh)
    # If you want to select only a subset of vertices, edges, and faces, it's possible:
    # pbc_mesh.inner_opt(
    #     loss_function_inner=energy_bounded,
    #     only_on_vertices=[list_vertex_ids],
    #     only_on_edges=[list_edges_id],
    #     only_on_faces=[list_faces_id],
    # )

    def energy_metric(mesh: BoundedMesh, _bilevel_opt: BoundedBilevelOptimizer) -> float:
        return float(
            energy_bounded(
                mesh.vertices,
                mesh.angles,
                mesh.edges,
                mesh.faces,
                None,
                None,
                None,
                mesh.vertices_params,
                mesh.edges_params,
                mesh.faces_params,
            )
        )

    bilevel_optimizer.add_custom_metric("Inner Energy", energy_metric)

    bilevel_optimizer.loss_function_outer = cost_ratio

    print(bilevel_optimizer.self_summary())
    bilevel_optimizer.do_n_bilevel_optimization(
        nb_epochs=epochs,
        mesh=bounded_mesh,
        report_every=1,
        also_report_to_stdout=True,
        save_plotmesh_every=10,
        save_mesh_data_every=10,
        save_folder="test_n_bilevel_bounded",
    )


if __name__ == "__main__":
    test_n_bilevel_opt()
