"""Create figure of cpu time vs gpu time consumption in function of the number of cells.

You'll have to run this script three times :
    - First set plot to False, and use_gpu to False.
    - Then set use_gpu to True.
    - Then set plot to True.
"""

from __future__ import annotations

import os

plot = False
use_gpu = False
if use_gpu:
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
else:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

from time import perf_counter
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import vmap

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.geo import get_area, get_length
from vertax.method_enum import BilevelOptimizationMethod

if TYPE_CHECKING:
    from jax import Array
    from numpy.typing import NDArray


def perform_bilevel_opt(n_cells: int, n_epochs: int) -> float:  # noqa: C901
    """Perform n_epochs of bilevel optimization on a mesh with n_cells and return the time it took."""
    print(f"Perform BO with {n_cells} cells.")
    t_start = perf_counter()

    # Settings
    L_box = jnp.sqrt(n_cells)
    width = float(L_box)
    height = float(L_box)
    MAX_EDGES_IN_ANY_FACE = 20

    # Set periodic boundary mesh and some of its properties
    pbc_mesh = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)
    # Note: those are base values so the following can be omitted
    bilevel_optimizer = PbcBilevelOptimizer()
    bilevel_optimizer.min_dist_T1 = 0.005
    bilevel_optimizer.max_nb_iterations = 1000
    bilevel_optimizer.tolerance = 1e-4
    bilevel_optimizer.patience = 5
    bilevel_optimizer.inner_solver = optax.sgd(learning_rate=0.01)  # inner solver
    bilevel_optimizer.outer_solver = optax.adam(learning_rate=0.0001, nesterov=True)  # outer solver
    bilevel_optimizer.bilevel_optimization_method = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
    # Other parameters are image_target (for cost_mesh2image), beta (for EP).

    # Initial condition (parameters)
    mu_tensions = 1.2
    std_tensions = 0.1
    mu_areas = 1.0
    std_areas = 0.0
    key = jax.random.PRNGKey(1)  # change the seed for different results
    he_params = mu_tensions + std_tensions * jax.random.normal(key, shape=(pbc_mesh.nb_edges,))
    he_params_reference = he_params[0]
    # Set mesh parameters
    pbc_mesh.vertices_params = jnp.asarray([0.0])
    pbc_mesh.edges_params = jnp.repeat(he_params, 2)
    pbc_mesh.faces_params = jnp.asarray(mu_areas + std_areas * jax.random.normal(key, shape=(n_cells,)))

    # Energy functions : Note that they use the width and height parameters now, defined earlier
    def area_part(face: Array, face_param: Array, vertTable: Array, heTable: Array, faceTable: Array) -> Array:
        a = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
        return (a - face_param) ** 2

    def hedge_part(he: Array, he_param: Array, vertTable: Array, heTable: Array, faceTable: Array) -> Array:
        edge_lengths = get_length(he, vertTable, heTable, faceTable, width, height)
        return he_param * edge_lengths

    # It is important to define the energy function with this exact signature,
    # even though the "_vert_params" is unused, we still keep it.
    def energy(
        vertTable: Array, heTable: Array, faceTable: Array, _vert_params: Array, he_params: Array, face_params: Array
    ) -> Array:
        K_areas = 20

        def mapped_areas_part(face: Array, face_param: Array) -> Array:
            return area_part(face, face_param, vertTable, heTable, faceTable)

        def mapped_hedges_part(he: Array, he_param: Array) -> Array:
            return hedge_part(he, he_param, vertTable, heTable, faceTable)

        areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), face_params)
        hedges_part = vmap(mapped_hedges_part)(jnp.arange(2, len(heTable)), he_params[2:])
        return (
            (2 * he_params_reference * get_length(0, vertTable, heTable, faceTable, width, height))
            + jnp.sum(hedges_part)
            + (0.5 * K_areas) * jnp.sum(areas_part)
        )

    # Energy minimization (init cond equilibrium)
    bilevel_optimizer.loss_function_inner = energy
    bilevel_optimizer.inner_optimization(mesh=pbc_mesh)
    # If you want to select only a subset of vertices, edges, and faces, it's possible:
    # pbc_mesh.inner_opt(
    #     loss_function_inner=energy,
    #     only_on_vertices=[list_vertex_ids],
    #     only_on_edges=[list_edges_id],
    #     only_on_faces=[list_faces_id],
    # )

    # Target (vertices)
    pbc_mesh_target = PbcMesh.copy_mesh(pbc_mesh)

    # Target (parameters)
    key = jax.random.PRNGKey(2)  # change the seed for different results
    he_params_target = mu_tensions + std_tensions * jax.random.normal(key, shape=(pbc_mesh_target.nb_edges,))
    he_params_reference_target = he_params_target[0]
    # Set target parameters
    pbc_mesh_target.vertices_params = jnp.asarray([0.0])  # Same as the first mesh we copied so we could have omit that.
    pbc_mesh_target.edges_params = jnp.repeat(he_params_target, 2)
    pbc_mesh_target.faces_params = jnp.asarray(mu_areas + std_areas * jax.random.normal(key, shape=(n_cells,)))

    # Energy minimization (target equilibrium)
    bilevel_optimizer.inner_optimization(mesh=pbc_mesh_target)

    # OK now we obtained our target vertices, edges and faces, let's store them in pbc_mesh too,
    # so they become the mesh target.
    bilevel_optimizer.vertices_target = pbc_mesh_target.vertices.copy()
    bilevel_optimizer.edges_target = pbc_mesh_target.edges.copy()
    bilevel_optimizer.faces_target = pbc_mesh_target.faces.copy()
    # We can discard pbc_mesh_target, which is of no use now
    del pbc_mesh_target

    # Areas target are the actual target ones at equilibrium (and remain fixed during BO)
    def mapped_fixed_areas_target(face: Array) -> Array:
        return get_area(
            face,
            bilevel_optimizer.vertices_target,
            bilevel_optimizer.edges_target,
            bilevel_optimizer.faces_target,
            width,
            height,
            MAX_EDGES_IN_ANY_FACE,
        )

    fixed_areas_target = vmap(mapped_fixed_areas_target)(jnp.arange(pbc_mesh.nb_faces))

    # Redefined energy function (with fixed areas and fixed first tension equal to the target ones)
    def area_part_v2(face: Array, vertTable: Array, heTable: Array, faceTable: Array) -> Array:
        a = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
        return (a - fixed_areas_target[face]) ** 2

    def hedge_part_v2(he: Array, he_param: Array, vertTable: Array, heTable: Array, faceTable: Array) -> Array:
        edge_lengths = get_length(he, vertTable, heTable, faceTable, width, height)
        return he_param * edge_lengths

    # It is important to define the energy function with this exact signature,
    # even though the "_vert_params" and "_face_params" are unused, we still keep them.
    def energy_v2(
        vertTable: Array, heTable: Array, faceTable: Array, _vert_params: Array, he_params: Array, _face_params: Array
    ) -> Array:
        K_areas = 20

        def mapped_areas_part(face: Array) -> Array:
            return area_part_v2(face, vertTable, heTable, faceTable)

        def mapped_hedges_part(he: Array, he_param: Array) -> Array:
            return hedge_part_v2(he, he_param, vertTable, heTable, faceTable)

        areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)))
        hedges_part = vmap(mapped_hedges_part)(jnp.arange(2, len(heTable)), he_params[2:])
        return (
            (2 * he_params_reference_target * get_length(0, vertTable, heTable, faceTable, width, height))
            + jnp.sum(hedges_part)
            + (0.5 * K_areas) * jnp.sum(areas_part)
        )

    bilevel_optimizer.loss_function_outer = cost_v2v
    for j in range(n_epochs + 1):
        t1 = perf_counter()
        print(
            "epoch: "
            + str(j)
            + "/"
            + str(n_epochs)
            + "\t cost: "
            + str(
                cost_v2v(
                    pbc_mesh.vertices,
                    pbc_mesh.edges,
                    pbc_mesh.faces,
                    width,
                    height,
                    bilevel_optimizer.vertices_target,
                    bilevel_optimizer.edges_target,
                    bilevel_optimizer.faces_target,
                )
            )
        )

        bilevel_optimizer.bilevel_optimization(mesh=pbc_mesh)
        print(f"epoch took {perf_counter() - t1} seconds")

    t_end = perf_counter()
    elapsed_times = t_end - t_start
    print(f"Test inverse modelling took {elapsed_times:.2f} s.")
    return elapsed_times


def collect_times(n_cells_list: list[int], n_epochs: int) -> list[float]:
    """Perform several bilevel optimizations with different number of cells."""
    print(f"Will perform BO for {n_cells_list} cells and {n_epochs} epochs.")
    return [perform_bilevel_opt(n_cells, n_epochs) for n_cells in n_cells_list]


def get_cpu_times(n_cells_list: list[int], n_epochs: int) -> list[float]:
    """Perform several bilevel optimizations with different number of cells on the CPU."""
    print(f"JAX default device set to {jax.numpy.ones(3).device}.")
    return collect_times(n_cells_list, n_epochs)


def get_gpu_times(n_cells_list: list[int], n_epochs: int) -> list[float]:
    """Perform several bilevel optimizations with different number of cells on the GPU."""
    print(f"JAX default device set to {jax.numpy.ones(3).device}.")
    return collect_times(n_cells_list, n_epochs)


def perform_and_save(n_cells_list: list[int], n_epochs: int, save_filename: str = "results.npz") -> None:
    """Perform all computations and save the results as numpy arrays in numpy file."""
    if use_gpu:
        save_filename = "gpu_" + save_filename
        gpu_times = np.array(get_gpu_times(n_cells_list, n_epochs))
        np.savez(save_filename, times=gpu_times, n_cells_list=n_cells_list, n_epochs=n_epochs)
    else:
        save_filename = "cpu_" + save_filename
        cpu_times = np.array(get_cpu_times(n_cells_list, n_epochs))
        np.savez(save_filename, times=cpu_times, n_cells_list=n_cells_list, n_epochs=n_epochs)


def load_previous_computations(
    save_filename: str = "cpu_results.npz",
) -> tuple[list[int], int, NDArray]:
    """Load cpu and gpu times saved previously in a numpy file."""
    res = np.load(save_filename)
    return [res["n_cells_list"], res["n_epochs"], res["times"]]  # type: ignore


def plot_cpu_vs_gpu_times(
    save_filename: str = "results.npz", save_plot_filename: str = "cpu_vs_gpu_results.png"
) -> None:
    """Plot a previous result, cpu vs gpu."""
    n_cells_list, n_epochs, cpu_times = load_previous_computations("cpu_" + save_filename)
    n_cells_list2, n_epochs2, gpu_times = load_previous_computations("gpu_" + save_filename)
    assert len(n_cells_list2) == len(n_cells_list)
    assert n_epochs == n_epochs2
    assert len(cpu_times) == len(gpu_times)
    plt.plot(n_cells_list, cpu_times, "blue", label="CPU")
    plt.plot(n_cells_list, gpu_times, "red", label="GPU")
    plt.legend()
    plt.xlabel("Numbel of cells")
    plt.ylabel(f"Time to perform {n_epochs} epochs (s)")
    plt.title("Time taken on CPU vs GPU in function of the number of cells")
    plt.savefig(save_plot_filename)
    plt.show()


if __name__ == "__main__":
    n_cells_list = [20, 40, 60, 80, 100]
    n_epochs = 10
    filename = "results.npz"
    plot_filename = "cpu_vs_gpu_results.png"
    if plot:
        plot_cpu_vs_gpu_times(filename, plot_filename)
    else:
        perform_and_save(n_cells_list=n_cells_list, n_epochs=n_epochs)
