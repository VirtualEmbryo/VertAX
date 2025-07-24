import jax.numpy as jnp 
from jax import jit, vmap

from vertax.geo import get_area, get_perimeter, get_length


@jit
def cell_energy(face, face_param, vertTable, heTable, faceTable):
    area = get_area(face, vertTable, heTable, faceTable)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable)
    return ((area - 1) ** 2) + ((perimeter - face_param) ** 2)

@jit
def energy_shape_factor_homo(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params):
    mapped_fn = lambda face, param: cell_energy(face, param, vertTable, heTable, faceTable)
    face_params_broadcasted = jnp.broadcast_to(face_params, (len(faceTable),) + face_params.shape[1:])
    cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params_broadcasted)
    return jnp.sum(cell_energies)

@jit
def energy_shape_factor_hetero(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params):
    mapped_fn = lambda face, param: cell_energy(face, param, vertTable, heTable, faceTable)
    cell_energies = vmap(mapped_fn)(selected_faces, face_params[selected_faces])
    # cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params)
    return jnp.sum(cell_energies)


@jit
def area_part(face: float, face_param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    a = get_area(face, vertTable, heTable, faceTable)
    return (a - face_param) ** 2

@jit
def hedge_part(he: float, he_param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    l = get_length(he, vertTable, heTable, faceTable)
    return he_param * l

@jit
def energy_line_tensions(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params):
    K_areas = 20
    mapped_areas_part = lambda face, face_param: area_part(face, face_param, vertTable, heTable, faceTable)
    mapped_hedges_part = lambda he, he_param: hedge_part(he, he_param, vertTable, heTable, faceTable)
    areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), face_params)
    hedges_part = vmap(mapped_hedges_part)(jnp.arange(len(heTable)), he_params)
    # areas_part = vmap(mapped_areas_part)(selected_faces, face_params[selected_faces])
    # hedges_part = vmap(mapped_hedges_part)(selected_hes, he_params[selected_hes])
    return  jnp.sum(hedges_part) + (0.5 * K_areas) * jnp.sum(areas_part)
