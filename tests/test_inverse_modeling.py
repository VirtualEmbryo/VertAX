"""Test the whole pipeline of bilevel optimization with the new API."""

from __future__ import annotations

import math
from time import perf_counter
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random
import optax
from jax import vmap
from numpy.testing import assert_allclose

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.geo import get_area, get_length
from vertax.method_enum import BilevelOptimizationMethod

if TYPE_CHECKING:
    from jax import Array


def test_inverse_modeling_for_regressions() -> None:  # noqa: C901
    """Check identical result of a standard test with previous results (november 2025)."""
    t_start = perf_counter()

    # Settings
    n_cells = 20
    width = math.sqrt(n_cells)
    height = width
    epochs = 2
    MAX_EDGES_IN_ANY_FACE = 20

    # Set periodic boundary mesh and some of its properties
    pbc_mesh = PbcMesh.periodic_voronoi_from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=0)

    # Initial condition (parameters)
    pbc_mesh.vertices_params = jnp.asarray([0.0])

    mu_tensions = 1.2
    std_tensions = 0.1
    key = jax.random.PRNGKey(1)  # change the seed for different results
    he_params = mu_tensions + std_tensions * jax.random.normal(key, shape=(pbc_mesh.nb_edges,))
    # Set mesh parameters
    pbc_mesh.edges_params = jnp.repeat(he_params, 2)

    pbc_mesh.faces_params = jnp.asarray([1.0 for _ in range(pbc_mesh.nb_faces)])

    he_params_reference = he_params[0]

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

    # Note: those are base values so the following can be omitted
    bilevel_optimizer = PbcBilevelOptimizer()
    bilevel_optimizer.update_T1 = True
    bilevel_optimizer.min_dist_T1 = 0.005
    bilevel_optimizer.max_nb_iterations = 1000
    bilevel_optimizer.tolerance = 1e-4
    bilevel_optimizer.patience = 5
    bilevel_optimizer.inner_solver = optax.sgd(learning_rate=0.01)  # inner solver
    bilevel_optimizer.outer_solver = optax.adam(learning_rate=0.0001, nesterov=True)  # outer solver
    bilevel_optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
    # Other parameters are image_target (for cost_mesh2image), beta (for EP).

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
    pbc_mesh_target.faces_params = jnp.asarray([1.0 for _ in range(pbc_mesh_target.nb_faces)])

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

    bilevel_optimizer.loss_function_inner = energy_v2
    bilevel_optimizer.loss_function_outer = cost_v2v
    for j in range(epochs + 1):
        t1 = perf_counter()
        print(
            "epoch: "
            + str(j)
            + "/"
            + str(epochs)
            + "\t cost: "
            + str(
                cost_v2v(
                    pbc_mesh.vertices,
                    pbc_mesh.edges,
                    pbc_mesh.faces,
                    pbc_mesh.width,
                    pbc_mesh.height,
                    bilevel_optimizer.vertices_target,
                    bilevel_optimizer.edges_target,
                    bilevel_optimizer.faces_target,
                )
            )
        )

        bilevel_optimizer.bilevel_optimization(pbc_mesh)
        print(perf_counter() - t1)

    t_end = perf_counter()
    elapsed_times = t_end - t_start
    print(f"Test inverse modelling took {elapsed_times:.2f} s.")

    # To create a new reference for the regression test only
    # pbc_mesh.save_mesh("tests/reference_result_test_inverse_modeling.npz")

    ref_mesh = PbcMesh.load_mesh("tests/reference_result_test_inverse_modeling.npz")

    assert_allclose(pbc_mesh.vertices, ref_mesh.vertices, rtol=0.001)
    assert_allclose(pbc_mesh.edges, ref_mesh.edges, rtol=0.001)
    assert_allclose(pbc_mesh.faces, ref_mesh.faces, rtol=0.001)


if __name__ == "__main__":
    test_inverse_modeling_for_regressions()
