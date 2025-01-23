import jax.numpy as jnp
import jax
import jax.lax
from jax import jit

from vertax.geo import get_length, update_pbc


@jit
def update_T1(vertTable: jnp.array, 
              heTable: jnp.array, 
              faceTable: jnp.array, 
              MIN_DISTANCE: float):

    def body_fun(idx, state):
        vertTable_new, heTable_new, faceTable_new = state

        he = heTable[idx]

        v_idx_source = he[3]
        v_idx_target = he[4]

        v_pos_source = vertTable[v_idx_source]
        v_pos_target = vertTable[v_idx_target]

        v_offset_x_target = he[6] * L_box
        v_offset_y_target = he[7] * L_box

        v1 = jnp.hstack((v_pos_source[:-1], jnp.array([0, 0])))
        v2 = jnp.hstack((v_pos_target[:-1], jnp.array([v_offset_x_target, v_offset_y_target])))

        distance = get_length(idx, vertTable, heTable, L_box)[0]

        def update_state(_state):
            vertTable_new, heTable_new, faceTable_new = _state

            x1 = v_pos_source[0]
            y1 = v_pos_source[1]

            cx = ((v1[0] + v1[2]) + (v2[0] + v2[2])) / 2
            cy = ((v1[1] + v1[3]) + (v2[1] + v2[3])) / 2

            angle = -jnp.pi / 2.0
            x1_new = cx + jnp.cos(angle) * (x1 - cx) - jnp.sin(angle) * (y1 - cy)
            y1_new = cy + jnp.sin(angle) * (x1 - cx) + jnp.cos(angle) * (y1 - cy)

            scale_factor = ((MIN_DISTANCE + 1e-3) / 2.0) / jnp.sqrt((x1_new - cx) ** 2 + (y1_new - cy) ** 2)
            x1 = (x1_new - cx) * scale_factor + cx
            y1 = (y1_new - cy) * scale_factor + cy

            he_prev = he[0]
            he_prev_twin = heTable[he_prev, 2]
            he_next = he[1]
            he_next_twin = heTable[he_next, 2]

            heTable_new = heTable_new.at[he_prev, 1].set(he[1])
            heTable_new = heTable_new.at[he_prev, 4].set(v_idx_source)
            heTable_new = heTable_new.at[he_prev_twin, 0].set(he[2])
            heTable_new = heTable_new.at[he_prev_twin, 3].set(v_idx_source)
            heTable_new = heTable_new.at[he_next, 0].set(he[0])
            heTable_new = heTable_new.at[he_next, 3].set(v_idx_source)
            heTable_new = heTable_new.at[he_next_twin, 1].set(idx)
            heTable_new = heTable_new.at[he_next_twin, 4].set(v_idx_source)
            heTable_new = heTable_new.at[idx, 0].set(he_next_twin)
            heTable_new = heTable_new.at[idx, 1].set(heTable.at[he_next_twin, 1].get())
            heTable_new = heTable_new.at[idx, 5].set(heTable.at[he_next_twin, 5].get())

            faceTable_new = faceTable_new.at[he[5]].set(he_next)
            faceTable_new = faceTable_new.at[heTable_new[idx][5]].set(idx)

            vertTable_new = vertTable_new.at[he[3], 0].set(x1)
            vertTable_new = vertTable_new.at[he[3], 1].set(y1)
            vertTable_new = vertTable_new.at[he[3], 2].set(he[1])

            return vertTable_new, heTable_new, faceTable_new

        vertTable_new, heTable_new, faceTable_new = jax.lax.cond(
            distance < MIN_DISTANCE,
            update_state,
            lambda _state: _state,
            (vertTable_new, heTable_new, faceTable_new)
        )

        return vertTable_new, heTable_new, faceTable_new

    n_cells = len(faceTable)
    L_box = jnp.sqrt(n_cells)

    state = (vertTable.copy(), heTable.copy(), faceTable.copy())
    vertTable_new, heTable_new, faceTable_new = jax.lax.fori_loop(
        0, len(heTable), body_fun, state
    )

    return update_pbc(vertTable_new, heTable_new, faceTable_new)












































































































# # checking T1 transitions and potentially updating vertices positions and offsets for periodic boundary conditions
# # @jit
# def update_T1(vertTable: jnp.array, 
#               heTable: jnp.array, 
#               faceTable: jnp.array, 
#               MIN_DISTANCE: float
#               ):

#     ### C'È UN PROBLEMA QUANDO 2 LINK CONSECUTIVI DI UNA STESSA CELLULA VOGLIONO TRANSIRE T1 ###
#     ### FORSE FARE VARIABILI DI APPOGGIO E AGGIORNARE A OGNI ITERATA?                        ###

#     n_cells = len(faceTable)
#     L_box = jnp.sqrt(n_cells)
    
#     heTable_new = heTable.copy()
#     faceTable_new = faceTable.copy()
#     vertTable_new = vertTable.copy()

#     for idx in range(len(heTable)):

#         he = heTable.at[idx].get()

#         v_idx_source = he[3]
#         v_idx_target = he[4]

#         v_pos_source = vertTable.at[v_idx_source].get()
#         v_pos_target = vertTable.at[v_idx_target].get()

#         v_offset_x_target = he[6] * L_box
#         v_offset_y_target = he[7] * L_box

#         v1 = jnp.hstack((v_pos_source[:-1], jnp.array([0, 0])))
#         v2 = jnp.hstack((v_pos_target[:-1], jnp.array([v_offset_x_target, v_offset_y_target])))

#         distance = get_length(idx, vertTable, heTable, L_box)[0]

#         if distance < MIN_DISTANCE:

#             x1 = v_pos_source.at[0].get()
#             y1 = v_pos_source.at[1].get()

#             # find the he's center
#             cx = ((v1.at[0].get() + v1.at[2].get()) + (v2.at[0].get() + v2.at[2].get())) / 2
#             cy = ((v1.at[1].get() + v1.at[3].get()) + (v2.at[1].get() + v2.at[3].get())) / 2

#             # rotate of 90 degrees counterclockwise
#             angle = -jnp.pi / 2.
#             x1_new = cx + jnp.cos(angle) * (x1 - cx) - jnp.sin(angle) * (y1 - cy)
#             y1_new = cy + jnp.sin(angle) * (x1 - cx) + jnp.cos(angle) * (y1 - cy)

#             # scale at larger size than minimal distance (adding 10**-3)
#             x1 = (x1_new - cx) * (((MIN_DISTANCE + 10 ** (-3)) / 2.) / jnp.sqrt(((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cx
#             y1 = (y1_new - cy) * (((MIN_DISTANCE + 10 ** (-3)) / 2.) / jnp.sqrt(((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cy

#             he_prev = he[0]
#             heTable_new = heTable_new.at[he_prev, 1].set(he[1])  # change prev_he's next_he with he's next_he
#             heTable_new = heTable_new.at[he_prev, 4].set(v_idx_source)

#             he_prev_twin = heTable.at[he_prev, 2].get()
#             heTable_new = heTable_new.at[he_prev_twin, 0].set(he[2])  # idx  # change prev_twin_he's prev_he with he
#             heTable_new = heTable_new.at[he_prev_twin, 3].set(v_idx_source)

#             he_next = he[1]
#             heTable_new = heTable_new.at[he_next, 0].set(he[0])  # change next_he's prev_he with he's prev_he
#             heTable_new = heTable_new.at[he_next, 3].set(v_idx_source)  # change next_he's source_vertex with he's source_vertex

#             he_next_twin = heTable.at[he_next, 2].get()
#             heTable_new = heTable_new.at[he_next_twin, 1].set(idx)  # change next_twin_he's next_he with he's twin
#             heTable_new = heTable_new.at[he_next_twin, 4].set(v_idx_source)  # change next twin he's target vertex with he's source vertex

#             heTable_new = heTable_new.at[idx, 0].set(he_next_twin)
#             heTable_new = heTable_new.at[idx, 1].set(heTable.at[he_next_twin, 1].get())
#             heTable_new = heTable_new.at[idx, 5].set(heTable.at[he_next_twin, 5].get())

#             faceTable_new = faceTable_new.at[he[5]].set(he_next)
#             faceTable_new = faceTable_new.at[heTable_new[idx][5]].set(idx)

#             vertTable_new = vertTable_new.at[he[3], 0].set(x1)
#             vertTable_new = vertTable_new.at[he[3], 1].set(y1)
#             vertTable_new = vertTable_new.at[he[3], 2].set(he[1])

#         else:
#             pass

#     return update_pbc(vertTable_new, heTable_new, faceTable_new)














































































































































# @jit
# def update_T1(vertTable: jnp.array, 
#               heTable: jnp.array, 
#               faceTable: jnp.array, 
#               MIN_DISTANCE: float):

#     def body_fun(idx, state):
#         vertTable_new, heTable_new, faceTable_new = state

#         he = heTable[idx]

#         v_idx_source = he[3]
#         v_idx_target = he[4]

#         v_pos_source = vertTable[v_idx_source]
#         v_pos_target = vertTable[v_idx_target]

#         v_offset_x_target = he[6] * L_box
#         v_offset_y_target = he[7] * L_box

#         v1 = jnp.hstack((v_pos_source[:-1], jnp.array([0, 0])))
#         v2 = jnp.hstack((v_pos_target[:-1], jnp.array([v_offset_x_target, v_offset_y_target])))

#         distance = get_length(idx, vertTable, heTable, L_box)[0]

#         def update_state(_):
#             # nonlocal vertTable_new, heTable_new, faceTable_new

#             x1 = v_pos_source[0]
#             y1 = v_pos_source[1]

#             cx = ((v1[0] + v1[2]) + (v2[0] + v2[2])) / 2
#             cy = ((v1[1] + v1[3]) + (v2[1] + v2[3])) / 2

#             angle = -jnp.pi / 2.0
#             x1_new = cx + jnp.cos(angle) * (x1 - cx) - jnp.sin(angle) * (y1 - cy)
#             y1_new = cy + jnp.sin(angle) * (x1 - cx) + jnp.cos(angle) * (y1 - cy)

#             scale_factor = ((MIN_DISTANCE + 1e-3) / 2.0) / jnp.sqrt((x1_new - cx) ** 2 + (y1_new - cy) ** 2)
#             x1 = (x1_new - cx) * scale_factor + cx
#             y1 = (y1_new - cy) * scale_factor + cy

#             he_prev = he[0]
#             he_prev_twin = heTable[he_prev, 2]
#             he_next = he[1]
#             he_next_twin = heTable[he_next, 2]

#             heTable_new = heTable_new.at[he_prev, 1].set(he[1])
#             # heTable_new = heTable_new.at[he_prev, 4].set(v_idx_source)

#             # heTable_new = heTable_new.at[he_prev_twin, 0].set(he[2])
#             # heTable_new = heTable_new.at[he_prev_twin, 3].set(v_idx_source)

#             # heTable_new = heTable_new.at[he_next, 0].set(he[0])
#             # heTable_new = heTable_new.at[he_next, 3].set(v_idx_source)

#             # heTable_new = heTable_new.at[he_next_twin, 1].set(idx)
#             # heTable_new = heTable_new.at[he_next_twin, 4].set(v_idx_source)

#             # heTable_new = heTable_new.at[idx, 0].set(he_next_twin)
#             # heTable_new = heTable_new.at[idx, 1].set(heTable[he_next_twin, 1])
#             # heTable_new = heTable_new.at[idx, 5].set(heTable[he_next_twin, 5])

#             # faceTable_new = faceTable_new.at[he[5]].set(he_next)
#             # faceTable_new = faceTable_new.at[heTable_new[idx][5]].set(idx)

#             # vertTable_new = vertTable_new.at[he[3], 0].set(x1)
#             # vertTable_new = vertTable_new.at[he[3], 1].set(y1)
#             # vertTable_new = vertTable_new.at[he[3], 2].set(he[1])

#             return vertTable_new, heTable_new, faceTable_new

#         vertTable_new, heTable_new, faceTable_new = jax.lax.cond(distance < MIN_DISTANCE, 
#                                                                  update_state, 
#                                                                  lambda _: (vertTable_new, heTable_new, faceTable_new), 
#                                                                  operand=None
#                                                                  )

#         return vertTable_new, heTable_new, faceTable_new

#     n_cells = len(faceTable)
#     L_box = jnp.sqrt(n_cells)

#     state = (vertTable.copy(), heTable.copy(), faceTable.copy())
#     vertTable_new, heTable_new, faceTable_new = jax.lax.fori_loop(0, 
#                                                                   len(heTable), 
#                                                                   body_fun, 
#                                                                   state
#                                                                   )

#     # return update_pbc(vertTable_new, heTable_new, faceTable_new)
#     return vertTable_new, heTable_new, faceTable_new















# checking T1 transitions and potentially updating vertices positions and offsets for periodic boundary conditions
# # @jit
# def update_T1(vertTable: jnp.array, 
#               heTable: jnp.array, 
#               faceTable: jnp.array, 
#               MIN_DISTANCE: float
#               ):

#     ### C'È UN PROBLEMA QUANDO 2 LINK CONSECUTIVI DI UNA STESSA CELLULA VOGLIONO TRANSIRE T1 ###
#     ### FORSE FARE VARIABILI DI APPOGGIO E AGGIORNARE A OGNI ITERATA?                        ###

#     n_cells = len(faceTable)
#     L_box = jnp.sqrt(n_cells)
    
#     heTable_new = heTable.copy()
#     faceTable_new = faceTable.copy()
#     vertTable_new = vertTable.copy()

#     for idx in range(len(heTable)):

#         he = heTable.at[idx].get()

#         v_idx_source = he[3]
#         v_idx_target = he[4]

#         v_pos_source = vertTable.at[v_idx_source].get()
#         v_pos_target = vertTable.at[v_idx_target].get()

#         v_offset_x_target = he[6] * L_box
#         v_offset_y_target = he[7] * L_box

#         v1 = jnp.hstack((v_pos_source[:-1], jnp.array([0, 0])))
#         v2 = jnp.hstack((v_pos_target[:-1], jnp.array([v_offset_x_target, v_offset_y_target])))

#         distance = get_length(idx, vertTable, heTable, L_box)[0]

#         if distance < MIN_DISTANCE:

#             x1 = v_pos_source.at[0].get()
#             y1 = v_pos_source.at[1].get()

#             # find the he's center
#             cx = ((v1.at[0].get() + v1.at[2].get()) + (v2.at[0].get() + v2.at[2].get())) / 2
#             cy = ((v1.at[1].get() + v1.at[3].get()) + (v2.at[1].get() + v2.at[3].get())) / 2

#             # rotate of 90 degrees counterclockwise
#             angle = -jnp.pi / 2.
#             x1_new = cx + jnp.cos(angle) * (x1 - cx) - jnp.sin(angle) * (y1 - cy)
#             y1_new = cy + jnp.sin(angle) * (x1 - cx) + jnp.cos(angle) * (y1 - cy)

#             # scale at larger size than minimal distance (adding 10**-3)
#             x1 = (x1_new - cx) * (((MIN_DISTANCE + 10 ** (-3)) / 2.) / jnp.sqrt(((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cx
#             y1 = (y1_new - cy) * (((MIN_DISTANCE + 10 ** (-3)) / 2.) / jnp.sqrt(((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cy

#             he_prev = he[0]
#             heTable_new = heTable_new.at[he_prev, 1].set(he[1])  # change prev_he's next_he with he's next_he
#             heTable_new = heTable_new.at[he_prev, 4].set(v_idx_source)

#             he_prev_twin = heTable.at[he_prev, 2].get()
#             heTable_new = heTable_new.at[he_prev_twin, 0].set(he[2])  # idx  # change prev_twin_he's prev_he with he
#             heTable_new = heTable_new.at[he_prev_twin, 3].set(v_idx_source)

#             he_next = he[1]
#             heTable_new = heTable_new.at[he_next, 0].set(he[0])  # change next_he's prev_he with he's prev_he
#             heTable_new = heTable_new.at[he_next, 3].set(v_idx_source)  # change next_he's source_vertex with he's source_vertex

#             he_next_twin = heTable.at[he_next, 2].get()
#             heTable_new = heTable_new.at[he_next_twin, 1].set(idx)  # change next_twin_he's next_he with he's twin
#             heTable_new = heTable_new.at[he_next_twin, 4].set(v_idx_source)  # change next twin he's target vertex with he's source vertex

#             heTable_new = heTable_new.at[idx, 0].set(he_next_twin)
#             heTable_new = heTable_new.at[idx, 1].set(heTable.at[he_next_twin, 1].get())
#             heTable_new = heTable_new.at[idx, 5].set(heTable.at[he_next_twin, 5].get())

#             faceTable_new = faceTable_new.at[he[5]].set(he_next)
#             faceTable_new = faceTable_new.at[heTable_new[idx][5]].set(idx)

#             vertTable_new = vertTable_new.at[he[3], 0].set(x1)
#             vertTable_new = vertTable_new.at[he[3], 1].set(y1)
#             vertTable_new = vertTable_new.at[he[3], 2].set(he[1])

#         else:
#             pass

#     return update_pbc(vertTable_new, heTable_new, faceTable_new)



# import jax
# import jax.numpy as jnp





# @jit
# def update_T1_beta(MIN_DISTANCE: float):
#     num_edges = t_heTable.shape[0]
#     edges = jnp.arange(num_edges // 2) * 2
#     split_idx = 2 * vlinks2he.size
#     scale = MIN_DISTANCE + 10 ** (-3) / 2
#     rot_angle = -jnp.pi / 2
#     num_angs_out = verts_angs.size - split_idx
#     fill_angs = num_edges - num_angs_out
#     verts_angs_arange = jnp.arange(split_idx + num_edges)
#     alinks2he_arange = jnp.arange(num_edges)

#     heTable_out = t_heTable[:]
#     faceTable_out = t_faceTable[:]
#     verts_angs_out = jnp.hstack([verts_angs, jnp.zeros(fill_angs)])
#     vlinks2he_out = vlinks2he[:]
#     alinks2he_out = jnp.hstack([alinks2he, jnp.zeros(fill_angs, dtype='int32')])

#     def angles_renumbering_fun(i, stacked_data):
#         heTable, alinks2he = stacked_data
#         surface_he_idx = alinks2he.at[i].get()
#         ang_idx = heTable.at[surface_he_idx, 8].get()
#         heTable = heTable.at[surface_he_idx, 8].set(ang_idx - 1)
#         return heTable, alinks2he

#     def true_fun1(stacked_data):
#         idx, idx_twin, v_idx_source, v_idx_target, x1, y1, x2, y2, he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face, he_angle, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info = stacked_data

#         he_prev_twin = heTable.at[he_prev, 2].get()
#         he_next_twin = heTable.at[he_next, 2].get()
#         he_next_twin_next = heTable.at[he_next_twin, 1].get()
#         he_prev_twin_prev = heTable.at[he_prev_twin, 0].get()
#         he_next_twin_face = heTable.at[he_next_twin, 7].get()
#         he_prev_twin_face = heTable.at[he_prev_twin, 7].get()

#         heTable = heTable.at[he_prev, 1].set(he_next)
#         heTable = heTable.at[he_prev_twin, 0].set(idx_twin)
#         heTable = heTable.at[he_next, 0].set(he_prev)
#         heTable = heTable.at[he_next, 3].set(v_idx_source)
#         heTable = heTable.at[he_next_twin_next, 0].set(idx)
#         heTable = heTable.at[he_next_twin, 1].set(idx)
#         heTable = heTable.at[he_next_twin, 4].set(v_idx_source)
#         heTable = heTable.at[he_prev_twin_prev, 1].set(idx_twin)
#         heTable = heTable.at[he_prev_twin_prev, 6].set(v_idx_target)

#         heTable = heTable.at[idx, 0].set(he_next_twin)
#         heTable = heTable.at[idx, 1].set(he_next_twin_next)
#         heTable = heTable.at[idx, 3].set(v_idx_source)
#         heTable = heTable.at[idx, 4].set(v_idx_target)
#         heTable = heTable.at[idx, 5].set(0)
#         heTable = heTable.at[idx, 6].set(1)
#         heTable = heTable.at[idx, 7].set(he_next_twin_face)
#         heTable = heTable.at[idx, 8].set(0)
#         heTable = heTable.at[idx_twin, 0].set(he_prev_twin_prev)
#         heTable = heTable.at[idx_twin, 1].set(he_prev_twin)
#         heTable = heTable.at[idx_twin, 3].set(v_idx_target)
#         heTable = heTable.at[idx_twin, 4].set(v_idx_source)
#         heTable = heTable.at[idx_twin, 7].set(he_prev_twin_face)

#         faceTable = faceTable.at[he_face].set(he_next)

#         return heTable, faceTable, info

#     def false_fun1(stacked_data):
#         _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, heTable, faceTable, info = stacked_data
#         return heTable, faceTable, info

#     def true_fun0(stacked_data):
#         x1, y1, x2, y2, idx, v_source, v_target, heTable, faceTable, info = stacked_data

#         cx = (x1 + x2) / 2
#         cy = (y1 + y2) / 2

#         # scale at larger size than minimal distance
#         x1_norm = jnp.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
#         x2_norm = jnp.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)

#         # rotate of 90 degrees counterclockwise
#         sina = jnp.sin(rot_angle)
#         cosa = jnp.cos(rot_angle)
#         rx1 = cx + cosa * (x1 - cx) - sina * (y1 - cy)
#         ry1 = cy + sina * (x1 - cx) + cosa * (y1 - cy)
#         rx2 = cx + cosa * (x2 - cx) - sina * (y2 - cy)
#         ry2 = cy + sina * (x2 - cx) + cosa * (y2 - cy)

#         # scale at larger size than minimal distance
#         nx1 = (rx1 - cx) * scale / x1_norm + cx
#         ny1 = (ry1 - cy) * scale / x1_norm + cy
#         nx2 = (rx2 - cx) * scale / x2_norm + cx
#         ny2 = (ry2 - cy) * scale / x2_norm + cy

#         he = heTable[idx]
#         idx_twin = he[2]
#         he_twin = heTable[idx_twin]

#         he_prev = he[0]
#         he_next = he[1]
#         he_twin_prev = he_twin[0]
#         he_twin_next = he_twin[1]
#         he_face = he[7]
#         he_twin_face = he_twin[7]
#         he_angle = he[8]

#         internalize = he_angle > 0
#         return cond(internalize, true_fun1, false_fun1, (idx, idx_twin, v_source, v_target, nx1, ny1, nx2, ny2, he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face, he_angle, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info))

#     def false_fun0(stacked_data):
#         _, _, _, _, _, _, _, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info = stacked_data
#         return heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info

#     def body_fun(i, stacked_data):
#         heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info = stacked_data
#         vertTable = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), jnp.reshape(verts_angs[:split_idx], [-1, 2])])
#         selector = (heTable.at[2 * i, 3].get() + heTable.at[2 * i, 5].get()) == 0
#         idx = 2 * i + selector.astype('int32')
#         v_source = heTable.at[idx, 3].get() + heTable.at[idx, 5].get()
#         v_target = heTable.at[idx, 4].get() + heTable.at[idx, 6].get() - 1
#         x0 = vertTable.at[v_source, 0].get()
#         y0 = vertTable.at[v_source, 1].get()
#         x1 = vertTable.at[v_target, 0].get()
#         y1 = vertTable.at[v_target, 1].get()
#         dist = jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
#         T1_trigger = dist < MIN_DISTANCE
#         return cond(T1_trigger, true_fun0, false_fun0, (x0, y0, x1, y1, idx, v_source, v_target, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info))

#     heTable_out, faceTable_out, verts_angs_out, vlinks2he_out, alinks2he_out, num_angs_out, info = fori_loop(0, t_heTable.shape[0] // 2, body_fun, (heTable_out, faceTable_out, verts_angs_out, vlinks2he_out, alinks2he_out, num_angs_out, 0))
#     t_heTable = heTable[:]
#     t_faceTable = faceTable[:]
#     verts_angs = verts_angs[:]
#     vlinks2he = vlinks2he[:]
#     alinks2he = alinks2he[:]
#     return info








# # @jit
# # def update_T1_beta(MIN_DISTANCE: float):
# #     num_edges = t_heTable.shape[0]
# #     edges = jnp.arange(num_edges // 2) * 2
# #     split_idx = 2 * vlinks2he.size
# #     scale = MIN_DISTANCE + 10 ** (-3) / 2
# #     rot_angle = -jnp.pi / 2
# #     num_angs_out = verts_angs.size - split_idx
# #     fill_angs = num_edges - num_angs_out
# #     verts_angs_arange = jnp.arange(split_idx + num_edges)
# #     alinks2he_arange = jnp.arange(num_edges)

# #     heTable_out = t_heTable[:]
# #     faceTable_out = t_faceTable[:]
# #     verts_angs_out = jnp.hstack([verts_angs, jnp.zeros(fill_angs)])
# #     vlinks2he_out = vlinks2he[:]
# #     alinks2he_out = jnp.hstack([alinks2he, jnp.zeros(fill_angs, dtype='int32')])

# #     def angles_renumbering_fun(i, stacked_data):
# #         heTable, alinks2he = stacked_data
# #         surface_he_idx = alinks2he.at[i].get()
# #         ang_idx = heTable.at[surface_he_idx, 8].get()
# #         heTable = heTable.at[surface_he_idx, 8].set(ang_idx - 1)
# #         return heTable, alinks2he

# #     def true_fun1(stacked_data):
# #         idx, idx_twin, v_idx_source, v_idx_target, x1, y1, x2, y2, he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face, he_angle, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info = stacked_data

# #         he_prev_twin = heTable.at[he_prev, 2].get()
# #         he_next_twin = heTable.at[he_next, 2].get()
# #         he_next_twin_next = heTable.at[he_next_twin, 1].get()
# #         he_prev_twin_prev = heTable.at[he_prev_twin, 0].get()
# #         he_next_twin_face = heTable.at[he_next_twin, 7].get()
# #         he_prev_twin_face = heTable.at[he_prev_twin, 7].get()

# #         heTable = heTable.at[he_prev, 1].set(he_next)
# #         heTable = heTable.at[he_prev_twin, 0].set(idx_twin)
# #         heTable = heTable.at[he_next, 0].set(he_prev)
# #         heTable = heTable.at[he_next, 3].set(v_idx_source)
# #         heTable = heTable.at[he_next_twin_next, 0].set(idx)
# #         heTable = heTable.at[he_next_twin, 1].set(idx)
# #         heTable = heTable.at[he_next_twin, 4].set(v_idx_source)
# #         heTable = heTable.at[he_prev_twin_prev, 1].set(idx_twin)
# #         heTable = heTable.at[he_prev_twin_prev, 6].set(v_idx_target)

# #         heTable = heTable.at[idx, 0].set(he_next_twin)
# #         heTable = heTable.at[idx, 1].set(he_next_twin_next)
# #         heTable = heTable.at[idx, 3].set(v_idx_source)
# #         heTable = heTable.at[idx, 4].set(v_idx_target)
# #         heTable = heTable.at[idx, 5].set(0)
# #         heTable = heTable.at[idx, 6].set(1)
# #         heTable = heTable.at[idx, 7].set(he_next_twin_face)
# #         heTable = heTable.at[idx, 8].set(0)
# #         heTable = heTable.at[idx_twin, 0].set(he_prev_twin_prev)
# #         heTable = heTable.at[idx_twin, 1].set(he_prev_twin)
# #         heTable = heTable.at[idx_twin, 3].set(v_idx_target)
# #         heTable = heTable.at[idx_twin, 4].set(v_idx_source)
# #         heTable = heTable.at[idx_twin, 7].set(he_prev_twin_face)

# #         faceTable = faceTable.at[he_face].set(he_next)

# #         verts_angs = verts_angs.at[v_idx_source * 2 - 4].set(x1)
# #         verts_angs = verts_angs.at[v_idx_source * 2 - 3].set(y1)
# #         vlinks2he = vlinks2he.at[v_idx_source - 2].set(he_next)
# #         verts_angs = verts_angs.at[v_idx_target * 2 - 4].set(x2)
# #         verts_angs = verts_angs.at[v_idx_target * 2 - 3].set(y2)
# #         vlinks2he = vlinks2he.at[v_idx_target - 2].set(he_next_twin_next)

# #         del_ang = he_angle - 1
# #         verts_angs = select(verts_angs_arange < del_ang + split_idx, verts_angs, jnp.roll(verts_angs, -1))
# #         alinks2he = select(alinks2he_arange < del_ang, alinks2he, jnp.roll(alinks2he, -1))
# #         num_angs = num_angs - 1
# #         heTable, alinks2he = fori_loop(del_ang, num_angs, angles_renumbering_fun, (heTable, alinks2he))
# #         return heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info

# #     def false_fun1(stacked_data):
# #         _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info = stacked_data
# #         return heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info

# #     def true_fun0(stacked_data):
# #         x1, y1, x2, y2, idx, v_source, v_target, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info = stacked_data

# #         cx = (x1 + x2) / 2
# #         cy = (y1 + y2) / 2

# #         # scale at larger size than minimal distance
# #         x1_norm = jnp.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
# #         x2_norm = jnp.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)

# #         # rotate of 90 degrees counterclockwise
# #         sina = np.sin(rot_angle)
# #         cosa = np.cos(rot_angle)
# #         rx1 = cx + cosa * (x1 - cx) - sina * (y1 - cy)
# #         ry1 = cy + sina * (x1 - cx) + cosa * (y1 - cy)
# #         rx2 = cx + cosa * (x2 - cx) - sina * (y2 - cy)
# #         ry2 = cy + sina * (x2 - cx) + cosa * (y2 - cy)

# #         # scale at larger size than minimal distance
# #         nx1 = (rx1 - cx) * scale / x1_norm + cx
# #         ny1 = (ry1 - cy) * scale / x1_norm + cy
# #         nx2 = (rx2 - cx) * scale / x2_norm + cx
# #         ny2 = (ry2 - cy) * scale / x2_norm + cy

# #         he = heTable[idx]
# #         idx_twin = he[2]
# #         he_twin = heTable[idx_twin]

# #         he_prev = he[0]
# #         he_next = he[1]
# #         he_twin_prev = he_twin[0]
# #         he_twin_next = he_twin[1]
# #         he_face = he[7]
# #         he_twin_face = he_twin[7]
# #         he_angle = he[8]

# #         internalize = he_angle > 0
# #         return cond(internalize, true_fun1, false_fun1, (idx, idx_twin, v_source, v_target, nx1, ny1, nx2, ny2, he_prev, he_next, he_twin_prev, he_twin_next, he_face, he_twin_face, he_angle, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info))

# #     def false_fun0(stacked_data):
# #         _, _, _, _, _, _, _, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info = stacked_data
# #         return heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info

# #     def body_fun(i, stacked_data):
# #         heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info = stacked_data
# #         vertTable = jnp.vstack([jnp.array([[0.0, 0.0], [1.0, 1.0]]), jnp.reshape(verts_angs[:split_idx], [-1, 2])])
# #         selector = (heTable.at[2 * i, 3].get() + heTable.at[2 * i, 5].get()) == 0
# #         idx = 2 * i + selector.astype('int32')
# #         v_source = heTable.at[idx, 3].get() + heTable.at[idx, 5].get()
# #         v_target = heTable.at[idx, 4].get() + heTable.at[idx, 6].get() - 1
# #         x0 = vertTable.at[v_source, 0].get()
# #         y0 = vertTable.at[v_source, 1].get()
# #         x1 = vertTable.at[v_target, 0].get()
# #         y1 = vertTable.at[v_target, 1].get()
# #         dist = jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
# #         T1_trigger = dist < MIN_DISTANCE
# #         return cond(T1_trigger, true_fun0, false_fun0, (x0, y0, x1, y1, idx, v_source, v_target, heTable, faceTable, verts_angs, vlinks2he, alinks2he, num_angs, info))

# #     heTable_out, faceTable_out, verts_angs_out, vlinks2he_out, alinks2he_out, num_angs_out, info = fori_loop(0, t_heTable.shape[0] // 2, body_fun, (heTable_out, faceTable_out, verts_angs_out, vlinks2he_out, alinks2he_out, num_angs_out, 0))
# #     t_heTable = heTable[:]
# #     t_faceTable = faceTable[:]
# #     verts_angs = verts_angs[:]
# #     vlinks2he = vlinks2he[:]
# #     alinks2he = alinks2he[:]
# #     return info