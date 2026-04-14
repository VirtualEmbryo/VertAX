"""Test the whole pipeline of bilevel optimization with the new API."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import vmap
from numpy.typing import NDArray

from vertax import PbcBilevelOptimizer, PbcMesh, plot_mesh
from vertax.cost import cost_v2v_ias
from vertax.geo import get_area, get_length
from vertax.method_enum import BilevelOptimizationMethod

if TYPE_CHECKING:
    from jax import Array


def load_geograph(path: str) -> tuple[Array, Array, Array]:
    """Load a mesh the old way."""
    return jnp.load(path + "vertTable.npy"), jnp.load(path + "heTable.npy"), jnp.load(path + "faceTable.npy")


def translate_base_mesh() -> None:
    """Translate old mesh data to new version."""
    vertices, edges, faces = load_geograph("tests/correlation/input/")
    vertices = vertices[:, :2]
    mesh = PbcMesh.create_empty()
    mesh.vertices = vertices
    mesh.edges = edges.reshape(-1, 8)
    mesh.faces = faces
    mesh.width = math.sqrt(20)
    mesh.height = math.sqrt(20)
    mesh.vertices_params = jnp.asarray([0.0])
    init_path = "tests/correlation/input/line_tensions_init.txt"
    init_data = np.loadtxt(init_path)
    init_values = init_data[:, 1]

    he_params = jnp.asarray(init_values[::2])
    mesh.edges_params = he_params
    mesh.faces_params = jnp.asarray([0.0 for i in range(20)])
    mesh.save_mesh("tests/correlation/base_mesh.npz")


def translate_target_mesh() -> None:
    """Translate old mesh data to new version (target mesh)."""
    vertices, edges, faces = load_geograph("tests/correlation/target/")
    vertices = vertices[:, :2]
    mesh = PbcMesh.create_empty()
    mesh.vertices = vertices
    mesh.edges = edges.reshape(-1, 8)
    mesh.faces = faces
    mesh.width = math.sqrt(20)
    mesh.height = math.sqrt(20)
    mesh.save_mesh("tests/correlation/target_mesh.npz")


def load_target_mesh() -> PbcMesh:
    """Load target mesh."""
    return PbcMesh.load_mesh("tests/correlation/target_mesh.npz")


def load_base_mesh() -> PbcMesh:
    """Load the base PBC mesh for correlation experiments."""
    return PbcMesh.load_mesh("tests/correlation/base_mesh.npz")


def create_optimizer() -> PbcBilevelOptimizer:
    """Get the optimizer for the experiments."""
    bop = PbcBilevelOptimizer()
    bop.min_dist_T1 = 0.05
    bop.max_nb_iterations = 1000
    bop.tolerance = 0.00001
    bop.patience = 5
    bop.inner_solver = optax.sgd(learning_rate=0.01)
    bop.outer_solver = optax.adam(learning_rate=0.0001, nesterov=True)
    bop.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
    bop.loss_function_outer = cost_v2v_ias
    return bop


def load_areas_target() -> Array:
    """Load target area for energy."""
    init_path_target = "tests/correlation/target/areas_target.txt"
    init_data_target = np.loadtxt(init_path_target)
    init_values_target = init_data_target[:, 1]
    return jnp.asarray(init_values_target)


def load_tensions_target() -> Array:
    """Load line tensions target for energy."""
    init_path_target = "tests/correlation/target/line_tensions_target.txt"
    init_data_target = np.loadtxt(init_path_target)
    init_values_target = init_data_target[:, 1]
    return jnp.asarray(init_values_target)


def _load_and_sort(target_path: str, init_path: str) -> tuple[NDArray, NDArray]:
    target_data = np.loadtxt(target_path)
    init_data = np.loadtxt(init_path)

    target_values = target_data[:, 1]
    init_values = init_data[:, 1]

    sorted_indices = np.argsort(target_values)
    sorted_target_values = target_values[sorted_indices]
    corresponding_init_values = init_values[sorted_indices]

    return sorted_target_values, corresponding_init_values


def _expected_result() -> None:
    nb_epochs = 2_400
    target1, init1 = _load_and_sort(
        "OLD_E2_working/OLD_E2_working/2_inference_multiple/energy_line_tensions_COUPLED/target/line_tensions_target.txt",
        "OLD_E2_working/OLD_E2_working/2_inference_multiple/energy_line_tensions_COUPLED/starting_0.1/line_tensions_init.txt",
    )
    _, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot for Initial Condition
    axes[0].scatter(target1, init1, color="blue", alpha=0.7)
    axes[0].plot([0.8, 1.7], [0.8, 1.7], color="red", linestyle="--", linewidth=1)  # Bisecting line
    axes[0].set_title("Initial Condition")
    axes[0].set_xlabel("Sorted Target Values")
    axes[0].set_ylabel("Corresponding Simulation Values")
    axes[0].set_xlim(0.65, 1.85)
    axes[0].set_ylim(0.65, 1.85)
    axes[1].set_xlim(0.65, 1.85)
    axes[1].set_ylim(0.65, 1.85)

    # Calculate and display correlation coefficient for Initial Condition
    corr1 = np.corrcoef(target1, init1)[0, 1]
    axes[0].text(
        0.05, 0.95, f"Corr. coeff. = {corr1:.2f}", transform=axes[0].transAxes, fontsize=12, verticalalignment="top"
    )

    target2, init2 = _load_and_sort(
        "OLD_E2_working/OLD_E2_working/2_inference_multiple/energy_line_tensions_COUPLED/target/line_tensions_target.txt",
        f"OLD_E2_working/OLD_E2_working/2_inference_multiple/bilevel_opt_lines_LESS_COUPLED_0.1/configurations/{nb_epochs}/line_tensions_final.txt",
    )

    # Plot for Final Condition
    axes[1].scatter(target2, init2, color="green", alpha=0.7)
    axes[1].plot([0.8, 1.7], [0.8, 1.7], color="red", linestyle="--", linewidth=1)  # Bisecting line
    axes[1].set_title("Final Condition")
    axes[1].set_xlabel("Sorted Target Values")
    axes[1].set_ylabel("Corresponding Simulation Values")

    # # Ensure both subplots share the same x and y limits
    # x_min = min(min(target1)-0.05, min(init1)-0.05, min(target2)-0.05, min(init2)-0.05)
    # x_max = max(max(target1)+0.05, max(init1)+0.05, max(target2)+0.05, max(init2)+0.05)
    # y_min = x_min
    # y_max = x_max

    # Calculate and display correlation coefficient for Final Condition
    corr2 = np.corrcoef(target2, init2)[0, 1]
    axes[1].text(
        0.05, 0.95, f"Corr. coeff. = {corr2:.2f}", transform=axes[1].transAxes, fontsize=12, verticalalignment="top"
    )

    plt.show()


def test_pearson_e2_t1() -> None:
    """Check identical result of a standard test with previous results (november 2025)."""
    t_start = perf_counter()
    nb_epochs = 10000
    MAX_EDGES_IN_ANY_FACE = 20
    areas_target = load_areas_target()

    n_cells = 20
    width = math.sqrt(n_cells)
    height = width

    # target_mesh = load_target_mesh()

    bop = create_optimizer()

    mesh_target = PbcMesh.from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1290)

    # Initial condition (parameters)
    mesh_target.vertices_params = jnp.asarray([0.0 for _ in range(mesh_target.nb_vertices)])

    mu_tensions = 1.0
    std_tensions = 0.3
    key = jax.random.PRNGKey(613714)  # change the seed for different results
    target_he_params = mu_tensions + std_tensions * jax.random.normal(key, shape=(mesh_target.nb_edges,))
    # Set mesh parameters
    mesh_target.edges_params = jnp.repeat(target_he_params, 2)

    mesh_target.faces_params = jnp.asarray([1.0 for _ in range(mesh_target.nb_faces)])

    he_params_reference = target_he_params[0]

    mesh = PbcMesh.copy_mesh(mesh_target)
    key = jax.random.PRNGKey(238721)  # change the seed for different results
    # base_he_params = target_he_params + std_tensions * jax.random.normal(key, shape=(mesh.nb_edges,))
    base_he_params = mu_tensions + std_tensions * jax.random.normal(key, shape=(mesh.nb_edges,))
    # Set mesh parameters
    mesh.edges_params = jnp.repeat(base_he_params, 2)

    # Energy functions : Note that they use the width and height parameters now, defined earlier
    def area_part(face: Array, _face_param: Array, vertTable: Array, heTable: Array, faceTable: Array) -> Array:
        a = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
        # return (a - face_param) ** 2
        return (a - areas_target[face]) ** 2

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
    bop.loss_function_inner = energy
    bop.inner_optimization(mesh_target)
    bop.inner_optimization(mesh)
    plot_mesh(mesh, show=False, save=True, save_path="tests/correlation/results/base_mesh.png", title="Base mesh")
    plot_mesh(
        mesh_target, show=False, save=True, save_path="tests/correlation/results/target_mesh.png", title="Target mesh"
    )
    mesh_target.save_mesh("tests/correlation/results/target_mesh.npz")

    bop.vertices_target = mesh_target.vertices.copy()
    bop.edges_target = mesh_target.edges.copy()
    bop.faces_target = mesh_target.faces.copy()

    def pearson_correlation(mesh: PbcMesh, _bop: PbcBilevelOptimizer) -> float:
        return float(jnp.corrcoef(mesh.edges_params, mesh_target.edges_params)[0, 1])

    bop.add_custom_metric("Pearson correlation", pearson_correlation)
    bop.do_n_bilevel_optimization(
        nb_epochs,
        mesh,
        report_every=10,
        save_plotmesh_every=100,
        save_mesh_data_every=100,
        also_report_to_stdout=True,
        save_folder="tests/correlation/results",
    )
    # for j in range(epochs + 1):
    #     t1 = perf_counter()
    #     print(
    #         "epoch: "
    #         + str(j)
    #         + "/"
    #         + str(epochs)
    #         + "\t cost: "
    #         + str(
    #             cost_v2v(
    #                 pbc_mesh.vertices,
    #                 pbc_mesh.edges,
    #                 pbc_mesh.faces,
    #                 pbc_mesh.width,
    #                 pbc_mesh.height,
    #                 bilevel_optimizer.vertices_target,
    #                 bilevel_optimizer.edges_target,
    #                 bilevel_optimizer.faces_target,
    #             )
    #         )
    #     )
    #
    #     bilevel_optimizer.bilevel_optimization(pbc_mesh)
    #     print(perf_counter() - t1)
    #     pearson_corr = float(jnp.corrcoef(pbc_mesh.edges_params, pbc_mesh_target.edges_params)[0, 1])
    #     print("Pearson", pearson_corr)
    #     np_corr = np.corrcoef(pbc_mesh.edges_params, pbc_mesh_target.edges_params)[0, 1]
    #     print("Pearson np", np_corr)

    t_end = perf_counter()
    elapsed_times = t_end - t_start
    print(f"Test correlation took {elapsed_times:.2f} s.")


def read_result() -> None:
    """Demonstrates how to extract tension data from saved meshes."""
    # First get all mesh filenames.
    mesh_filenames = Path("tests/correlation/results/meshes_data/").glob("mesh*.npz")
    # Load mesh file and extract in particular the tensions (edges params)
    tensions = [PbcMesh.load_mesh(str(filename)).edges_params for filename in mesh_filenames]
    print(tensions[0])
    print(tensions[-1])
    print(f"{len(tensions)} tension arrays acquired.")


if __name__ == "__main__":
    translate_base_mesh()
    translate_target_mesh()
    test_pearson_e2_t1()
    # read_result()
    # _expected_result()
    # print(load_base_mesh().edges_params)
