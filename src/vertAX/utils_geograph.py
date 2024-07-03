from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import while_loop


@partial(jit, static_argnums=(3,))
def sum_edges(face: int, t_heTable: jnp.array, t_faceTable: jnp.array, fun):

    start_he = t_faceTable.at[face].get()
    he = start_he
    res = fun(he, jnp.array([0., 0., 0.]))
    he = t_heTable.at[he, 1].get()

    # stacked_data = (current_he, current_res)
    # res is the sum of contributions before current_he
    def cond_fun(stacked_data):
        he, _ = stacked_data
        return he != start_he

    def body_fun(stacked_data):
        he, res = stacked_data
        next_he = t_heTable.at[he, 1].get()
        res += fun(he, res)
        return next_he, res

    _, res = while_loop(cond_fun, body_fun, (he, res))
    return res

@jit
def length(he, vertTable: jnp.array, t_heTable: jnp.array, L_box: float):

    v_source = t_heTable.at[he, 3].get()
    v_target = t_heTable.at[he, 4].get()
    x0 = vertTable.at[v_source, 0].get()  # source
    y0 = vertTable.at[v_source, 1].get()
    he_offset_x1 = t_heTable.at[he, 6].get() * L_box  # offset target
    he_offset_y1 = t_heTable.at[he, 7].get() * L_box
    x1 = vertTable.at[v_target, 0].get() + he_offset_x1  # target
    y1 = vertTable.at[v_target, 1].get() + he_offset_y1

    return jnp.array([jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), he_offset_x1, he_offset_y1])

@jit
def get_perimeter(face: int, vertTable: jnp.array, t_heTable: jnp.array, t_faceTable: jnp.array):

    def fun(he, res):
        return length(he, vertTable, t_heTable, jnp.sqrt(len(t_faceTable)))

    return sum_edges(face, t_heTable, t_faceTable, fun)[0]

@jit
def compute_numerator(he, res, vertTable: jnp.array, t_heTable: jnp.array, L_box: float):

    x_offset, y_offset = res.at[1].get(), res.at[2].get()
    v_source = t_heTable.at[he, 3].get()
    v_target = t_heTable.at[he, 4].get()
    x0 = vertTable.at[v_source, 0].get() + x_offset  # source
    y0 = vertTable.at[v_source, 1].get() + y_offset
    he_offset_x1 = t_heTable.at[he, 6].get() * L_box  # offset target
    he_offset_y1 = t_heTable.at[he, 7].get() * L_box
    x1 = vertTable.at[v_target, 0].get() + x_offset + he_offset_x1  # target
    y1 = vertTable.at[v_target, 1].get() + y_offset + he_offset_y1

    return jnp.array([(x0 * y1) - (x1 * y0), he_offset_x1, he_offset_y1])

@jit
def get_area(face: int, vertTable: jnp.array, t_heTable: jnp.array, t_faceTable: jnp.array):

    def fun(he, res):
        return compute_numerator(he, res, vertTable, t_heTable, jnp.sqrt(len(t_faceTable)))

    return 0.5 * jnp.abs(sum_edges(face, t_heTable, t_faceTable, fun)[0])

@jit
def compute_single_face(face, vertTable, t_heTable, t_faceTable):

    perimeter = get_perimeter(face, vertTable, t_heTable, t_faceTable)
    area = get_area(face, vertTable, t_heTable, t_faceTable)

    return perimeter, area

# computing center given two vertices with their offsets
# v1 = (v1_x, v1_y, v1_offset_x * L_box, v1_offset_y * L_box)
# v2 = (v2_x, v2_y, v2_offset_x * L_box, v2_offset_y * L_box)
@jit
def get_edge_center(v1: jnp.array, v2: jnp.array):

    cx = ((v1.at[0].get() + v1.at[2].get()) + (v2.at[0].get() + v2.at[2].get())) / 2
    cy = ((v1.at[1].get() + v1.at[3].get()) + (v2.at[1].get() + v2.at[3].get())) / 2

    return cx, cy

@jit
def update_he(he, t_heTable, vertTable, L_box: float):

    v_idx_target = t_heTable.at[he, 4].get()
    v_x = vertTable.at[v_idx_target, 0].get()
    v_y = vertTable.at[v_idx_target, 1].get()
    offset_x_target = jnp.where(v_x < 0., -1, jnp.where(v_x > L_box, +1, 0))
    offset_y_target = jnp.where(v_y < 0., -1, jnp.where(v_y > L_box, +1, 0))

    v_idx_source = t_heTable.at[he, 3].get()
    v_x = vertTable.at[v_idx_source, 0].get()
    v_y = vertTable.at[v_idx_source, 1].get()
    offset_x_source = jnp.where(v_x < 0., -1, jnp.where(v_x > L_box, +1, 0))
    offset_y_source = jnp.where(v_y < 0., -1, jnp.where(v_y > L_box, +1, 0))


    return offset_x_target, offset_y_target, offset_x_source, offset_y_source

@jit
def move_vertex(v, vertTable, L_box: float):

    v_x = vertTable.at[v, 0].get()
    v_y = vertTable.at[v, 1].get()
    #jax.debug.print("old: v_x v_y {bar1} {bar2}", bar1=v_x, bar2=v_y)
    v_x = jnp.where(v_x < 0., v_x + L_box, jnp.where(v_x > L_box, v_x - L_box, v_x))
    v_y = jnp.where(v_y < 0., v_y + L_box, jnp.where(v_y > L_box, v_y - L_box, v_y))
    #jax.debug.print("new: v_x v_y {bar1} {bar2}", bar1=v_x, bar2=v_y)

    return v_x, v_y

@jit
def get_shape_factor(vertTable, t_heTable, t_faceTable):

    num_faces = len(t_faceTable)
    faces = jnp.arange(num_faces)
    mapped_fn = lambda face: compute_single_face(face, vertTable, t_heTable, t_faceTable)
    perimeters, areas = vmap(mapped_fn)(faces)

    return jnp.sum(perimeters) / jnp.sqrt(num_faces * jnp.sum(areas))
