# Package imports
from time import perf_counter

import jax.numpy as jnp
import jax.random
import optax
from jax import jit, vmap
from numpy.testing import assert_allclose
from functools import partial

from vertax.geo import get_area, get_perimeter
from vertax.opt import OptimizationTarget, minimize
from vertax.start import create_mesh_from_seeds, load_mesh


def test_minimization_for_regressions() -> None:
    """Check identical result of a standard test with previous results (october 2025)."""
    t_start = perf_counter()

    # Settings
    n_cells = 100
    min_dist_T1 = 0.2
    vert_params = jnp.asarray([0.0])
    he_params = jnp.asarray([0.0])
    face_params = jnp.asarray([3.7])

    # Solver
    sgd = optax.sgd(learning_rate=0.01)
    iterations_max = 1000
    tolerance = 1e-4
    patience = 5

    MAX_EDGES_IN_ANY_FACE = 20

    # Energy function
    @partial(jit, static_argnums=(5, 6))
    def cell_energy(face, face_param, vertTable, heTable, faceTable, width, height):
        area = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
        perimeter = get_perimeter(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
        return ((area - 1) ** 2) + ((perimeter - face_param) ** 2)

    @partial(jit, static_argnums=(3, 4))
    def energy(vertTable, heTable, faceTable, width, height, vert_params, he_params, face_params):
        mapped_fn = lambda face, param: cell_energy(face, param, vertTable, heTable, faceTable, width, height)
        face_params_broadcasted = jnp.broadcast_to(face_params, (len(faceTable),))
        cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params_broadcasted)
        return jnp.sum(cell_energies)

    # Initial condition
    key = jax.random.PRNGKey(1)
    L_box = jnp.sqrt(n_cells)
    width = float(L_box)
    height = float(L_box)
    seeds = L_box * jax.random.uniform(key, (n_cells, 2))
    vertTable, heTable, faceTable = create_mesh_from_seeds(seeds)

    # Energy minimization
    (vertTable_eq, heTable_eq, faceTable_eq, vert_params_eq, he_params_eq, face_params_eq), energies = minimize(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        energy,
        sgd,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts=None,
        selected_hes=None,
        selected_faces=None,
        optimization_target=OptimizationTarget.VERTICES,
    )

    t_end = perf_counter()
    elapsed_times = t_end - t_start
    print(f"Test forward modelling took {elapsed_times:.2f} s.")

    # To create a new reference for the regression test only
    # from vertax.start import save_mesh
    # save_mesh("tests/reference_result_test_minimization.npz", vertTable_eq, heTable_eq, faceTable_eq)

    ref_vertices, ref_edges, ref_faces = load_mesh("tests/reference_result_test_minimization.npz")

    assert_allclose(vertTable_eq, ref_vertices, rtol=0.001)
    assert_allclose(heTable_eq, ref_edges, rtol=0.001)
    assert_allclose(faceTable_eq, ref_faces, rtol=0.001)

    # # Plotting/saving
    # plot_mesh(
    #     vertTable_eq, heTable_eq, faceTable_eq,
    #     L_box, multicolor=True, lines=True, vertices=False,
    #     path='./', name='energy_minimization', show=True, save=True)


if __name__ == "__main__":
    test_minimization_for_regressions()
