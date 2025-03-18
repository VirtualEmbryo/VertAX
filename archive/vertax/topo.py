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

    state = (vertTable, heTable, faceTable)
    vertTable_last, heTable_last, faceTable_last = jax.lax.fori_loop(
        0, len(heTable), body_fun, state
    )

    return update_pbc(vertTable_last, heTable_last, faceTable_last)



# # checking T1 transitions and potentially updating vertices positions and offsets for periodic boundary conditions
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
