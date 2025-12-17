"""Energy related functions."""

import jax
import jax.numpy as jnp
from jax import Array, jit, vmap

from vertax.geo import get_area, get_area_bounded, get_edge_length, get_length, get_perimeter, get_surface_length

TARGET_AREA = 0.6

MAX_EDGES_IN_ANY_FACE = 20


def cell_energy(
    face: Array, face_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
) -> Array:
    """Example of a cell energy function."""
    area = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
    return ((area - 1) ** 2) + ((perimeter - face_param) ** 2)


def energy_shape_factor_homo(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    face_params: Array,
) -> Array:
    """Example of an energy function."""

    def mapped_fn(face: Array, param: Array) -> Array:
        return cell_energy(face, param, vertTable, heTable, faceTable, width, height)

    face_params_broadcasted = jnp.broadcast_to(face_params, (len(faceTable), *face_params.shape[1:]))
    cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params_broadcasted)
    return jnp.sum(cell_energies)


def energy_shape_factor_hetero(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    selected_faces: Array,
    face_params: Array,
) -> Array:
    """Example of an energy function."""

    def mapped_fn(face: Array, param: Array) -> Array:
        return cell_energy(face, param, vertTable, heTable, faceTable, width, height)

    cell_energies = vmap(mapped_fn)(selected_faces, face_params[selected_faces])
    # cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params)
    return jnp.sum(cell_energies)


def area_part(
    face: Array, face_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
) -> Array:
    """Part of an energy function."""
    a = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
    return (a - face_param) ** 2


def hedge_part(
    he: Array, he_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
) -> Array:
    """Part of an energy function."""
    length = get_length(he, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_ANY_FACE)
    return he_param * length


def energy_line_tensions(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    he_params: Array,
    face_params: Array,
) -> Array:
    """Example of an energy function."""
    K_areas = 20

    def mapped_areas_part(face: Array, face_param: Array) -> Array:
        return area_part(face, face_param, vertTable, heTable, faceTable, width, height)

    def mapped_hedges_part(he: Array, he_param: Array) -> Array:
        return hedge_part(he, he_param, vertTable, heTable, faceTable, width, height)

    areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), face_params)
    hedges_part = vmap(mapped_hedges_part)(jnp.arange(len(heTable)), he_params)
    # areas_part = vmap(mapped_areas_part)(selected_faces, face_params[selected_faces])
    # hedges_part = vmap(mapped_hedges_part)(selected_hes, he_params[selected_hes])
    return jnp.sum(hedges_part) + (0.5 * K_areas) * jnp.sum(areas_part)


# ==========
# Bounded
# ==========
@jit
def cell_area_energy(face: Array, vertTable: Array, angTable: Array, heTable: Array, faceTable: Array) -> Array:
    """Part of an energy function."""
    area = get_area_bounded(face, vertTable, angTable, heTable, faceTable)
    return (area - TARGET_AREA) ** 2


@jit
def surface_edge_energy(edge: Array, tension: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    """Part of an energy function."""
    length = get_surface_length(edge, vertTable, angTable, heTable)
    return length * tension


@jit
def inner_edge_energy(edge: Array, tension: Array, vertTable: Array, heTable: Array) -> Array:
    """Part of an energy function."""
    length = get_edge_length(edge, vertTable, heTable)
    return length * tension


@jit
def energy_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    _selected_verts: Array | None,
    _selected_hes: Array | None,
    _selected_faces: Array | None,
    _vert_params: Array,
    he_params: Array,
    _face_params: Array,
) -> Array:
    """Base energy function for bounded meshes."""
    num_faces = faceTable.shape[0]
    faces = jnp.arange(num_faces)
    num_edges = angTable.size
    num_half_edges = num_edges * 2
    unique_edges = jnp.arange(num_edges) * 2
    edges = jnp.arange(num_half_edges)
    vertTable = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), vertTable])
    angTable = jnp.repeat(angTable, 2)
    he_params = jax.nn.sigmoid(he_params) + 1

    def mapped_fn_area(face: Array) -> Array:
        return cell_area_energy(face, vertTable, angTable, heTable, faceTable)

    cell_area_energies = jnp.sum(vmap(mapped_fn_area)(faces))

    def mapped_fn_inner(edge: Array, tension: Array) -> Array:
        return inner_edge_energy(edge, tension, vertTable, heTable)

    inner_edge_energies = jnp.sum(vmap(mapped_fn_inner)(unique_edges, he_params))

    def mapped_fn_surface(edge: Array, tension: Array) -> Array:
        return surface_edge_energy(edge, tension, vertTable, angTable, heTable)

    surface_edge_energies = jnp.sum(vmap(mapped_fn_surface)(edges, jnp.repeat(he_params, 2)))
    return 20 * cell_area_energies + inner_edge_energies + surface_edge_energies
