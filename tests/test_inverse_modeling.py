"""Test the whole pipeline of bilevel optimization."""

from time import perf_counter

import jax.numpy as jnp
import jax.random
import optax
from jax import jit, vmap
from numpy.testing import assert_allclose

from vertax.cost import cost_v2v
from vertax.geo import get_area, get_length
from vertax.opt import BilevelOptimizationMethod, bilevel_opt, inner_opt
from vertax.start import create_mesh_from_seeds, load_mesh


def test_inverse_modeling_for_regressions() -> None:
    """Check identical result of a standard test with previous results (october 2025)."""
    t_start = perf_counter()

    # Settings
    n_cells = 20
    min_dist_T1 = 0.005
    mu_tensions = 1.2
    std_tensions = 0.1
    mu_areas = 1.0
    std_areas = 0.0

    # Solvers
    sgd = optax.sgd(learning_rate=0.01)  # inner solver
    adam = optax.adam(learning_rate=0.0001, nesterov=True)  # outer solver
    iterations_max = 1000
    tolerance = 1e-4
    patience = 5
    epochs = 2

    # Energy function
    @jit
    def area_part(face, face_param, vertTable, heTable, faceTable):
        a = get_area(face, vertTable, heTable, faceTable)
        return (a - face_param) ** 2

    @jit
    def hedge_part(he, he_param, vertTable, heTable, faceTable):
        edge_lengths = get_length(he, vertTable, heTable, faceTable)
        return he_param * edge_lengths

    @jit
    def energy(vertTable, heTable, faceTable, vert_params, he_params, face_params):
        K_areas = 20

        def mapped_areas_part(face, face_param):
            return area_part(face, face_param, vertTable, heTable, faceTable)

        def mapped_hedges_part(he, he_param):
            return hedge_part(he, he_param, vertTable, heTable, faceTable)

        areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), face_params)
        hedges_part = vmap(mapped_hedges_part)(jnp.arange(2, len(heTable)), he_params[2:])
        return (
            (2 * he_params_reference * get_length(0, vertTable, heTable, faceTable))
            + jnp.sum(hedges_part)
            + (0.5 * K_areas) * jnp.sum(areas_part)
        )

    # Initial condition (vertices)
    key = jax.random.PRNGKey(0)  # change the seed for different results
    L_box = jnp.sqrt(n_cells)
    seeds = L_box * jax.random.uniform(key, shape=(n_cells, 2))
    vertTable, heTable, faceTable = create_mesh_from_seeds(seeds)

    # Initial condition (parameters)
    key = jax.random.PRNGKey(1)  # change the seed for different results
    vert_params = jnp.asarray([0.0])
    he_params = mu_tensions + std_tensions * jax.random.normal(key, shape=(len(heTable) // 2,))
    he_params = jnp.repeat(he_params, 2)
    he_params_reference = he_params[0]
    face_params = jnp.asarray(mu_areas + std_areas * jax.random.normal(key, shape=(n_cells,)))

    # Energy minimization (init cond equilibrium)
    (vertTable, heTable, faceTable), _ = inner_opt(
        vertTable,
        heTable,
        faceTable,
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
    )

    # Target (vertices)
    vertTable_target, heTable_target, faceTable_target = vertTable.copy(), heTable.copy(), faceTable.copy()

    # Target (parameters)
    key = jax.random.PRNGKey(2)  # change the seed for different results
    vert_params_target = jnp.asarray([0.0])
    he_params_target = mu_tensions + std_tensions * jax.random.normal(key, shape=(len(heTable_target) // 2,))
    he_params_target = jnp.repeat(he_params_target, 2)
    he_params_reference_target = he_params_target[0]
    face_params_target = jnp.asarray(mu_areas + std_areas * jax.random.normal(key, shape=(n_cells,)))

    # Energy minimization (target equilibrium)
    (vertTable_target, heTable_target, faceTable_target), _ = inner_opt(
        vertTable_target,
        heTable_target,
        faceTable_target,
        vert_params_target,
        he_params_target,
        face_params_target,
        energy,
        sgd,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts=None,
        selected_hes=None,
        selected_faces=None,
    )

    # Areas target are the actual target ones at equilibrium (and remain fixed during BO)
    def mapped_fixed_areas_target(face):
        return get_area(face, vertTable_target, heTable_target, faceTable)

    fixed_areas_target = vmap(mapped_fixed_areas_target)(jnp.arange(len(faceTable)))

    # Redefined energy function (with fixed areas and fixed first tension equal to the target ones)
    @jit
    def area_part_v2(face, vertTable, heTable, faceTable):
        a = get_area(face, vertTable, heTable, faceTable)
        return (a - fixed_areas_target[face]) ** 2

    @jit
    def hedge_part_v2(he, he_param, vertTable, heTable, faceTable):
        edge_lengths = get_length(he, vertTable, heTable, faceTable)
        return he_param * edge_lengths

    @jit
    def energy_v2(vertTable, heTable, faceTable, vert_params, he_params, face_params):
        K_areas = 20

        def mapped_areas_part(face):
            return area_part_v2(face, vertTable, heTable, faceTable)

        def mapped_hedges_part(he, he_param):
            return hedge_part_v2(he, he_param, vertTable, heTable, faceTable)

        areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)))
        hedges_part = vmap(mapped_hedges_part)(jnp.arange(2, len(heTable)), he_params[2:])
        return (
            (2 * he_params_reference_target * get_length(0, vertTable, heTable, faceTable))
            + jnp.sum(hedges_part)
            + (0.5 * K_areas) * jnp.sum(areas_part)
        )

    for j in range(epochs + 1):
        print(
            "epoch: "
            + str(j)
            + "/"
            + str(epochs)
            + "\t cost: "
            + str(
                cost_v2v(
                    vertTable,
                    heTable,
                    faceTable,
                    vertTable_target,
                    heTable_target,
                    faceTable_target,
                    selected_verts=None,
                    selected_hes=None,
                    selected_faces=None,
                    image_target=None,
                )
            )
        )

        (vertTable, heTable, faceTable, vert_params, he_params, face_params), _ = bilevel_opt(
            vertTable,
            heTable,
            faceTable,
            vert_params,
            he_params,
            face_params,
            vertTable_target,
            heTable_target,
            faceTable_target,
            L_in=energy_v2,
            L_out=cost_v2v,
            solver_inner=sgd,
            solver_outer=adam,
            min_dist_T1=min_dist_T1,
            iterations_max=iterations_max,
            tolerance=tolerance,
            patience=patience,
            selected_verts=None,
            selected_hes=None,
            selected_faces=None,
            image_target=None,
            beta=None,
            method=BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION,  # change to EP, ID, AS
        )

    t_end = perf_counter()
    elapsed_times = t_end - t_start
    print(f"Test forward modelling took {elapsed_times:.2f} s.")

    ref_vertices, ref_edges, ref_faces = load_mesh("tests/reference_result_test_inverse_modeling.npz")

    assert_allclose(vertTable, ref_vertices, rtol=0.001)
    assert_allclose(heTable, ref_edges, rtol=0.001)
    assert_allclose(faceTable, ref_faces, rtol=0.001)


if __name__ == "__main__":
    test_inverse_modeling_for_regressions()
