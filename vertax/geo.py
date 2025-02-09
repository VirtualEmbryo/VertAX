from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import while_loop


@partial(jit, static_argnums=(3,))
def sum_edges(face, 
              heTable: jnp.array, 
              faceTable: jnp.array, 
              fun
              ):

    start_he = faceTable.at[face].get()
    he = start_he
    res = fun(he, jnp.array([0., 0., 0.]))
    he = heTable.at[he, 1].get()

    # stacked_data = (current_he, current_res)
    # res is the sum of contributions before current_he
    def cond_fun(stacked_data):
        he, _ = stacked_data
        return he != start_he

    def body_fun(stacked_data):
        he, res = stacked_data
        next_he = heTable.at[he, 1].get()
        res += fun(he, res)
        return next_he, res

    _, res = while_loop(cond_fun, body_fun, (he, res))
    return res

@jit
def get_length(he, 
               vertTable: jnp.array, 
               heTable: jnp.array, 
               L_box: float
               ):

    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    x0 = vertTable.at[v_source, 0].get()  # source
    y0 = vertTable.at[v_source, 1].get()
    he_offset_x1 = heTable.at[he, 6].get() * L_box  # offset target
    he_offset_y1 = heTable.at[he, 7].get() * L_box
    x1 = vertTable.at[v_target, 0].get() + he_offset_x1  # target
    y1 = vertTable.at[v_target, 1].get() + he_offset_y1

    return jnp.array([jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), he_offset_x1, he_offset_y1])

@jit
def get_perimeter(face, 
                  vertTable: jnp.array, 
                  heTable: jnp.array, 
                  faceTable: jnp.array
                  ):

    def fun(he, res):
        return get_length(he, vertTable, heTable, jnp.sqrt(len(faceTable)))

    return sum_edges(face, heTable, faceTable, fun)[0]

@jit
def compute_numerator(he, 
                      res, 
                      vertTable: jnp.array, 
                      heTable: jnp.array, 
                      L_box: float
                      ):

    x_offset, y_offset = res.at[1].get(), res.at[2].get()
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    x0 = vertTable.at[v_source, 0].get() + x_offset  # source
    y0 = vertTable.at[v_source, 1].get() + y_offset
    he_offset_x1 = heTable.at[he, 6].get() * L_box  # offset target
    he_offset_y1 = heTable.at[he, 7].get() * L_box
    x1 = vertTable.at[v_target, 0].get() + x_offset + he_offset_x1  # target
    y1 = vertTable.at[v_target, 1].get() + y_offset + he_offset_y1

    return jnp.array([(x0 * y1) - (x1 * y0), he_offset_x1, he_offset_y1])

# computing area for a face using  ## shoelace formula ##
@jit
def get_area(face, 
             vertTable: jnp.array, 
             heTable: jnp.array, 
             faceTable: jnp.array
             ):

    def fun(he, res):
        return compute_numerator(he, res, vertTable, heTable, jnp.sqrt(len(faceTable)))

    return 0.5 * jnp.abs(sum_edges(face, heTable, faceTable, fun)[0])

# (only for id implementation)
# listing vertices of a face 
@jit
def get_vertices_id(face, 
                    vertTable: jnp.array, 
                    heTable: jnp.array, 
                    faceTable: jnp.array
                    ):

    n_cells = len(faceTable)
    L_box = jnp.sqrt(n_cells)

    start_he = faceTable.at[face].get()

    he = start_he
    v_source = heTable.at[he, 3].get()

    verts_sources = jnp.array([vertTable.at[v_source].get()])

    verts_offsets = jnp.array([jnp.array([0, 0])])
    he_offset_x = heTable.at[he, 6].get()
    he_offset_y = heTable.at[he, 7].get()
    sum0_offsets = he_offset_x
    sum1_offsets = he_offset_y

    he = heTable.at[he, 1].get()

    for _ in range(20 - 1):
        v_source = heTable.at[he, 3].get()
        verts_sources = jnp.concatenate((verts_sources, jnp.array([vertTable.at[v_source].get()])), axis=0)

        verts_offsets = jnp.where(he != start_he, jnp.concatenate(
            (verts_offsets, jnp.array([jnp.array([sum0_offsets * L_box, sum1_offsets * L_box])])),
            axis=0), jnp.concatenate((verts_offsets, jnp.array([verts_offsets.at[0].get()]))))

        he_offset_x = heTable.at[he, 6].get()
        he_offset_y = heTable.at[he, 7].get()
        sum0_offsets += he_offset_x
        sum1_offsets += he_offset_y

        he = jnp.where(he != start_he, heTable.at[he, 1].get(), he)

    return jnp.hstack((verts_sources.at[:, :-1].get(), verts_offsets))

# (only for id implementation)
# computing area for a face using  ## shoelace formula ##  
@jit
def get_area_id(face, 
                vertTable: jnp.array, 
                heTable: jnp.array, 
                faceTable: jnp.array
                ):

    vertices = get_vertices_id(face, vertTable, heTable, faceTable)

    numerator = 0.

    for i in range(len(vertices) - 1):
        numerator += ((vertices.at[i, 0].get() + vertices.at[i, 2].get()) * 
                      (vertices.at[i + 1, 1].get() + vertices.at[i + 1, 3].get()) - 
                      (vertices.at[i, 1].get() + vertices.at[i, 3].get()) * 
                      (vertices.at[i + 1, 0].get() + vertices.at[i + 1, 2].get()))

    return jnp.abs(numerator / 2.)

@jit
def get_perimeter_area(face, 
                       vertTable: jnp.array, 
                       heTable: jnp.array, 
                       faceTable: jnp.array
                       ):

    perimeter = get_perimeter(face, vertTable, heTable, faceTable)
    area = get_area(face, vertTable, heTable, faceTable)

    return perimeter, area

@jit
def get_shape_factor(vertTable: jnp.array, 
                     heTable: jnp.array, 
                     faceTable: jnp.array
                     ):

    num_faces = len(faceTable)
    faces = jnp.arange(num_faces)
    mapped_fn = lambda face: get_perimeter_area(face, vertTable, heTable, faceTable)
    perimeters, areas = vmap(mapped_fn)(faces)

    return jnp.sum(perimeters) / jnp.sqrt(num_faces * jnp.sum(areas))

@jit
def update_he(he, 
              vertTable: jnp.array, 
              heTable: jnp.array, 
              L_box: float):

    v_idx_target = heTable.at[he, 4].get()
    v_x = vertTable.at[v_idx_target, 0].get()
    v_y = vertTable.at[v_idx_target, 1].get()
    offset_x_target = jnp.where(v_x < 0., -1, jnp.where(v_x > L_box, +1, 0))
    offset_y_target = jnp.where(v_y < 0., -1, jnp.where(v_y > L_box, +1, 0))

    v_idx_source = heTable.at[he, 3].get()
    v_x = vertTable.at[v_idx_source, 0].get()
    v_y = vertTable.at[v_idx_source, 1].get()
    offset_x_source = jnp.where(v_x < 0., -1, jnp.where(v_x > L_box, +1, 0))
    offset_y_source = jnp.where(v_y < 0., -1, jnp.where(v_y > L_box, +1, 0))

    return offset_x_target, offset_y_target, offset_x_source, offset_y_source

@jit
def move_vertex_inside(v: jnp.array, 
                       vertTable: jnp.array, 
                       L_box: float):

    v_x = vertTable.at[v, 0].get()
    v_y = vertTable.at[v, 1].get()
    v_x = jnp.where(v_x < 0., v_x + L_box, jnp.where(v_x > L_box, v_x - L_box, v_x))
    v_y = jnp.where(v_y < 0., v_y + L_box, jnp.where(v_y > L_box, v_y - L_box, v_y))

    return v_x, v_y

# updating vertices positions and offsets for periodic boundary conditions
@jit
def update_pbc(vertTable: jnp.array, 
               heTable: jnp.array, 
               faceTable: jnp.array
               ):

    n_cells = len(faceTable)
    L_box = jnp.sqrt(n_cells)

    mapped_offsets = lambda he: update_he(he, vertTable, heTable, L_box)
    offset_x_target, offset_y_target, offset_x_source, offset_y_source = vmap(mapped_offsets)(jnp.arange(len(heTable)))
    heTable = heTable.at[:, 6].add(+offset_x_target-offset_x_source)
    heTable = heTable.at[:, 7].add(+offset_y_target-offset_y_source)

    mapped_vertices = lambda v: move_vertex_inside(v, vertTable, L_box)
    v_x, v_y = vmap(mapped_vertices)(jnp.arange(len(vertTable)))
    vertTable = vertTable.at[:, 0].set(v_x)
    vertTable = vertTable.at[:, 1].set(v_y)

    return vertTable, heTable, faceTable

# checking T1 transitions and (potentially) updating vertices positions and offsets for periodic boundary conditions
# @jit
def update_T1(vertTable: jnp.array, 
              heTable: jnp.array, 
              faceTable: jnp.array, 
              MIN_DISTANCE: float
              ):

    ### C'È UN PROBLEMA QUANDO 2 LINK CONSECUTIVI DI UNA STESSA CELLULA VOGLIONO TRANSIRE T1 ###
    ### FORSE FARE VARIABILI DI APPOGGIO E AGGIORNARE A OGNI ITERATA?                        ###

    n_cells = len(faceTable)
    L_box = jnp.sqrt(n_cells)
    
    heTable_new = heTable.copy()
    faceTable_new = faceTable.copy()
    vertTable_new = vertTable.copy()

    for idx in range(len(heTable)):

        he = heTable.at[idx].get()

        v_idx_source = he[3]
        v_idx_target = he[4]

        v_pos_source = vertTable.at[v_idx_source].get()
        v_pos_target = vertTable.at[v_idx_target].get()

        v_offset_x_target = he[6] * L_box
        v_offset_y_target = he[7] * L_box

        v1 = jnp.hstack((v_pos_source[:-1], jnp.array([0, 0])))
        v2 = jnp.hstack((v_pos_target[:-1], jnp.array([v_offset_x_target, v_offset_y_target])))

        distance = get_length(idx, vertTable, heTable, L_box)[0]

        if distance < MIN_DISTANCE:

            x1 = v_pos_source.at[0].get()
            y1 = v_pos_source.at[1].get()

            # find the he's center
            cx = ((v1.at[0].get() + v1.at[2].get()) + (v2.at[0].get() + v2.at[2].get())) / 2
            cy = ((v1.at[1].get() + v1.at[3].get()) + (v2.at[1].get() + v2.at[3].get())) / 2

            # rotate of 90 degrees counterclockwise
            angle = -jnp.pi / 2.
            x1_new = cx + jnp.cos(angle) * (x1 - cx) - jnp.sin(angle) * (y1 - cy)
            y1_new = cy + jnp.sin(angle) * (x1 - cx) + jnp.cos(angle) * (y1 - cy)

            # scale at larger size than minimal distance
            x1 = (x1_new - cx) * (((MIN_DISTANCE + 10 ** (-3)) / 2.) / jnp.sqrt(((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cx
            y1 = (y1_new - cy) * (((MIN_DISTANCE + 10 ** (-3)) / 2.) / jnp.sqrt(((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cy

            he_prev = he[0]
            heTable_new = heTable_new.at[he_prev, 1].set(he[1])  # change prev_he's next_he with he's next_he
            heTable_new = heTable_new.at[he_prev, 4].set(v_idx_source)

            he_prev_twin = heTable.at[he_prev, 2].get()
            heTable_new = heTable_new.at[he_prev_twin, 0].set(he[2])  # idx  # change prev_twin_he's prev_he with he
            heTable_new = heTable_new.at[he_prev_twin, 3].set(v_idx_source)

            he_next = he[1]
            heTable_new = heTable_new.at[he_next, 0].set(he[0])  # change next_he's prev_he with he's prev_he
            heTable_new = heTable_new.at[he_next, 3].set(v_idx_source)  # change next_he's source_vertex with he's source_vertex

            he_next_twin = heTable.at[he_next, 2].get()
            heTable_new = heTable_new.at[he_next_twin, 1].set(idx)  # change next_twin_he's next_he with he's twin
            heTable_new = heTable_new.at[he_next_twin, 4].set(v_idx_source)  # change next twin he's target vertex with he's source vertex

            heTable_new = heTable_new.at[idx, 0].set(he_next_twin)
            heTable_new = heTable_new.at[idx, 1].set(heTable.at[he_next_twin, 1].get())
            heTable_new = heTable_new.at[idx, 5].set(heTable.at[he_next_twin, 5].get())

            faceTable_new = faceTable_new.at[he[5]].set(he_next)
            faceTable_new = faceTable_new.at[heTable_new[idx][5]].set(idx)

            vertTable_new = vertTable_new.at[he[3], 0].set(x1)
            vertTable_new = vertTable_new.at[he[3], 1].set(y1)
            vertTable_new = vertTable_new.at[he[3], 2].set(he[1])

        else:
            pass

    return vertTable_new, heTable_new, faceTable_new
