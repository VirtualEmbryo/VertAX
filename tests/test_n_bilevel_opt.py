"""Test the whole pipeline of n bilevel optimization with the new API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random
import optax
from jax import vmap

from vertax import PbcBilevelOptimizer, PbcMesh
from vertax.cost import cost_v2v
from vertax.geo import get_area, get_length
from vertax.method_enum import BilevelOptimizationMethod

if TYPE_CHECKING:
    from jax import Array


def test_n_bilevel_opt() -> None:  # noqa: C901
    """Check the whole function of n bilevel opt which makes a lot of things."""
    # Settings
    n_cells = 20
    epochs = 10
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
    bilevel_optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
    bilevel_optimizer.beta = 0.01

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

    def energy_metric(mesh: PbcMesh, _bilevel_opt: PbcBilevelOptimizer) -> float:
        return float(
            energy_v2(mesh.vertices, mesh.edges, mesh.faces, mesh.vertices_params, mesh.edges_params, mesh.faces_params)
        )

    bilevel_optimizer.add_custom_metric("Inner Energy", energy_metric)

    bilevel_optimizer.loss_function_inner = energy_v2
    bilevel_optimizer.loss_function_outer = cost_v2v

    print(bilevel_optimizer.self_summary())
    bilevel_optimizer.do_n_bilevel_optimization(
        nb_epochs=epochs,
        mesh=pbc_mesh,
        report_every=1,
        also_report_to_stdout=True,
        save_plotmesh_every=10,
        save_mesh_data_every=10,
        save_folder="test_n_bilevel",
    )


if __name__ == "__main__":
    test_n_bilevel_opt()
