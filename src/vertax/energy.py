import jax.numpy as jnp
from jax import Array, jit, vmap
import jax

from vertax.geo import get_area, get_length, get_perimeter, get_area_bounded, get_surface_length, get_edge_length

TARGET_AREA = 0.6


@jit
def cell_energy(face, face_param, vertTable, heTable, faceTable, width: float, height: float):
    area = get_area(face, vertTable, heTable, faceTable, width, height)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable, width, height)
    return ((area - 1) ** 2) + ((perimeter - face_param) ** 2)


@jit
def energy_shape_factor_homo(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    selected_verts,
    selected_hes,
    selected_faces,
    vert_params,
    he_params,
    face_params,
):
    def mapped_fn(face, param):
        return cell_energy(face, param, vertTable, heTable, faceTable, width, height)

    face_params_broadcasted = jnp.broadcast_to(face_params, (len(faceTable),) + face_params.shape[1:])
    cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params_broadcasted)
    return jnp.sum(cell_energies)


@jit
def energy_shape_factor_hetero(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    selected_verts,
    selected_hes,
    selected_faces,
    vert_params,
    he_params,
    face_params,
):
    def mapped_fn(face, param):
        return cell_energy(face, param, vertTable, heTable, faceTable, width, height)

    cell_energies = vmap(mapped_fn)(selected_faces, face_params[selected_faces])
    # cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params)
    return jnp.sum(cell_energies)


@jit
def area_part(
    face: float, face_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
):
    a = get_area(face, vertTable, heTable, faceTable, width, height)
    return (a - face_param) ** 2


@jit
def hedge_part(
    he: float, he_param: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
):
    length = get_length(he, vertTable, heTable, faceTable, width, height)
    return he_param * length


@jit
def energy_line_tensions(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    selected_verts,
    selected_hes,
    selected_faces,
    vert_params,
    he_params,
    face_params,
):
    K_areas = 20

    def mapped_areas_part(face, face_param):
        return area_part(face, face_param, vertTable, heTable, faceTable, width, height)

    def mapped_hedges_part(he, he_param):
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
def cell_area_energy(face: float, vertTable: Array, angTable: Array, heTable: Array, faceTable: Array):
    area = get_area_bounded(face, vertTable, angTable, heTable, faceTable)
    return (area - TARGET_AREA) ** 2


@jit
def surface_edge_energy(edge: float, tension: Array, vertTable: Array, angTable: Array, heTable: Array):
    length = get_surface_length(edge, vertTable, angTable, heTable)
    return length * tension


@jit
def inner_edge_energy(edge: float, tension: Array, vertTable: Array, heTable: Array):
    length = get_edge_length(edge, vertTable, heTable)
    return length * tension


@jit
def energy_bounded(
    vertTable,
    angTable,
    heTable,
    faceTable,
    selected_verts,
    selected_hes,
    selected_faces,
    vert_params,
    he_params,
    face_params,
):
    num_faces = faceTable.shape[0]
    faces = jnp.arange(num_faces)
    num_edges = angTable.size
    num_half_edges = num_edges * 2
    unique_edges = jnp.arange(num_edges) * 2
    edges = jnp.arange(num_half_edges)
    vertTable = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), vertTable])
    angTable = jnp.repeat(angTable, 2)
    he_params = jax.nn.sigmoid(he_params) + 1
    mapped_fn_area = lambda face, area: cell_area_energy(face, vertTable, angTable, heTable, faceTable)
    face_params_broadcasted = jnp.broadcast_to(face_params, (len(faces),))
    cell_area_energies = jnp.sum(vmap(mapped_fn_area)(faces, face_params_broadcasted))
    mapped_fn_inner = lambda edge, tension: inner_edge_energy(edge, tension, vertTable, heTable)
    inner_edge_energies = jnp.sum(vmap(mapped_fn_inner)(unique_edges, he_params))
    mapped_fn_surface = lambda edge, tension: surface_edge_energy(edge, tension, vertTable, angTable, heTable)
    surface_edge_energies = jnp.sum(vmap(mapped_fn_surface)(edges, jnp.repeat(he_params, 2)))
    return 20 * cell_area_energies + inner_edge_energies + surface_edge_energies
