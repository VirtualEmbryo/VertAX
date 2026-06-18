"""Test the whole pipeline of bilevel optimization with conjugate gradient."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from vertax import (
    BoundedBilevelOptimizer,
    BoundedMesh,
    cost_ratio,
    energy_line_tensions_bounded,
    plot_mesh,
)
from vertax.method_enum import BilevelOptimizationMethod

if TYPE_CHECKING:
    pass


def create_optimizer() -> BoundedBilevelOptimizer:
    """Get the optimizer for the experiments."""
    bop = BoundedBilevelOptimizer()
    bop.min_dist_T1 = 0.05
    bop.max_nb_iterations = 1000
    bop.tolerance = 0.00001
    bop.patience = 5
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    # bop.inner_solver = nonlinear_cg(restart_every=10)
    bop.outer_solver = optax.adam(learning_rate=0.0001, nesterov=True)
    bop.bilevel_optimization_method = BilevelOptimizationMethod.ADJOINT_STATE
    bop.loss_function_outer = cost_ratio
    # bop.loss_function_outer = cost_v2v_ias
    return bop


def test_cg() -> None:
    """Check identical result of a standard test with previous results (november 2025)."""
    t_start = perf_counter()
    Path("tests/cg/results").mkdir(exist_ok=True)
    nb_epochs = 10000
    n_cells = 20
    width = math.sqrt(n_cells)
    height = width

    bop = create_optimizer()

    mesh = BoundedMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1290)

    # Initial condition (parameters)
    mesh.vertices_params = jnp.asarray([0.0 for _ in range(mesh.nb_vertices)])

    mu_tensions = 1.0
    std_tensions = 0.2
    key = jax.random.PRNGKey(643517)  # change the seed for different results
    mesh.edges_params = mu_tensions + std_tensions * jax.random.normal(key, shape=(mesh.nb_edges,))

    mesh.faces_params = jnp.asarray([0.5 for _ in range(mesh.nb_faces)])

    # Energy minimization (init cond equilibrium)
    bop.loss_function_inner = energy_line_tensions_bounded
    bop.inner_optimization(mesh)
    plot_mesh(mesh, show=False, save=True, save_path="tests/cg/results/base_mesh.png", title="Base mesh")
    mesh.save_mesh("tests/cg/results/target_mesh.npz")

    # bop.vertices_target = mesh_target.vertices.copy()
    # bop.edges_target = mesh_target.edges.copy()
    # bop.faces_target = mesh_target.faces.copy()

    bop.do_n_bilevel_optimization(
        nb_epochs,
        mesh,
        report_every=10,
        save_plotmesh_every=10,
        save_mesh_data_every=100,
        also_report_to_stdout=True,
        save_folder="tests/cg/results",
    )

    t_end = perf_counter()
    elapsed_times = t_end - t_start
    print(f"Test cg took {elapsed_times:.2f} s.")


if __name__ == "__main__":
    test_cg()
