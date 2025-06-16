from functools import partial

import jax.numpy as jnp
import jax
import jax.lax
from jax import jit

from vertax.geo import get_length, update_pbc
from vertax.plot import plot_geograph


@partial(jit, static_argnums=(9,))
def update_T1(vertTable, 
              heTable, 
              faceTable, 
              selected_verts, 
              selected_hes, 
              selected_faces, 
              vert_params, 
              he_params, 
              face_params, 
              L_in, 
              min_distance):

    def body_fun(idx, state):

        vertTable_new, heTable_new, faceTable_new = state

        he = heTable_new[idx]
        prev_he = heTable_new[he[0]]
        next_he = heTable_new[he[1]]
        twin_he = heTable_new[he[2]]
        twin_prev_he = heTable_new[prev_he[2]]
        next_twin_he = heTable_new[twin_he[1]]
        twin_next_he = heTable_new[next_he[2]]

        he_idx = idx
        prev_he_idx = heTable_new[he_idx, 0]
        next_he_idx = heTable_new[he_idx, 1]
        twin_he_idx = heTable_new[he_idx, 2]
        twin_prev_he_idx = prev_he[2]
        prev_twin_prev_he_idx = twin_prev_he[0]
        prev_twin_he_idx = twin_he[0]
        next_twin_he_idx = twin_he[1]
        twin_next_he_idx = next_he[2]
        next_twin_next_he_idx = twin_next_he[1]

        # check distance
        distance = get_length(he_idx, vertTable_new, heTable_new, L_box)[0]
        
        # check if the two faces that share the hes are triangles
        he_prev = he[0]
        twin_he_prev = twin_he[0]
        should_update = heTable_new[he_prev, 0] != he[1]
        twin_should_update = heTable_new[twin_he_prev, 0] != twin_he[1]

        def update_state(_state):

            vertTable_new, heTable_new, faceTable_new = _state

            ## heTable
            # he
            heTable_new = heTable_new.at[he_idx, 0].set(prev_twin_prev_he_idx)
            heTable_new = heTable_new.at[he_idx, 1].set(twin_prev_he_idx)
            heTable_new = heTable_new.at[he_idx, 5].set(twin_prev_he[5])

            # twin he
            heTable_new = heTable_new.at[twin_he_idx, 0].set(twin_next_he_idx)
            heTable_new = heTable_new.at[twin_he_idx, 1].set(next_twin_next_he_idx)
            heTable_new = heTable_new.at[twin_he_idx, 5].set(twin_next_he[5])

            # prev he
            heTable_new = heTable_new.at[prev_he_idx, 1].set(next_he_idx)
            heTable_new = heTable_new.at[prev_he_idx, 4].set(he[4])
            heTable_new = heTable_new.at[prev_he_idx, 6].add(he[6])
            heTable_new = heTable_new.at[prev_he_idx, 7].add(he[7])
            
            # next he
            heTable_new = heTable_new.at[next_he_idx, 0].set(prev_he_idx)

            # prev twin he
            heTable_new = heTable_new.at[prev_twin_he_idx, 1].set(next_twin_he_idx)
            heTable_new = heTable_new.at[prev_twin_he_idx, 4].set(twin_he[4])
            heTable_new = heTable_new.at[prev_twin_he_idx, 6].add(twin_he[6])
            heTable_new = heTable_new.at[prev_twin_he_idx, 7].add(twin_he[7])

            # next twin he
            heTable_new = heTable_new.at[next_twin_he_idx, 0].set(prev_twin_he_idx)

            # prev twin prev he
            heTable_new = heTable_new.at[prev_twin_prev_he_idx, 1].set(he_idx)

            # twin prev he
            heTable_new = heTable_new.at[twin_prev_he_idx, 0].set(he_idx)
            heTable_new = heTable_new.at[twin_prev_he_idx, 3].set(he[4])
            heTable_new = heTable_new.at[twin_prev_he_idx, 6].add(-he[6])
            heTable_new = heTable_new.at[twin_prev_he_idx, 7].add(-he[7])

            # twin next he
            heTable_new = heTable_new.at[twin_next_he_idx, 1].set(twin_he_idx)

            # next twin next he
            heTable_new = heTable_new.at[next_twin_next_he_idx, 0].set(twin_he_idx)
            heTable_new = heTable_new.at[next_twin_next_he_idx, 3].set(twin_he[4])
            heTable_new = heTable_new.at[next_twin_next_he_idx, 6].add(-twin_he[6])
            heTable_new = heTable_new.at[next_twin_next_he_idx, 7].add(-twin_he[7])

            ## vertTable
            vertTable_new = vertTable_new.at[he[3], 2].set(he_idx)
            vertTable_new = vertTable_new.at[twin_he[3], 2].set(twin_he_idx)

            # he
            x_source_he = vertTable_new[he[3], 0]
            y_source_he = vertTable_new[he[3], 1]
            x_target_he = vertTable_new[he[4], 0]
            y_target_he = vertTable_new[he[4], 1]
            offset_x_target_he = heTable_new[he_idx, 6] * L_box
            offset_y_target_he = heTable_new[he_idx, 7] * L_box

            center_x_he = ((x_source_he) + (x_target_he + offset_x_target_he)) / 2
            center_y_he = ((y_source_he) + (y_target_he + offset_y_target_he)) / 2
            angle = jnp.pi / 2.
            turn_x_source_he = center_x_he + jnp.cos(angle) * (x_source_he - center_x_he) - jnp.sin(angle) * (y_source_he - center_y_he)
            turn_y_source_he = center_y_he + jnp.sin(angle) * (x_source_he - center_x_he) + jnp.cos(angle) * (y_source_he - center_y_he)
            scale_factor = ((min_distance + min_distance*0.1) / 2.0) / jnp.sqrt((turn_x_source_he - center_x_he) ** 2 + (turn_y_source_he - center_y_he) ** 2)
            last_x_source_he = (turn_x_source_he - center_x_he) * scale_factor + center_x_he
            last_y_source_he = (turn_y_source_he - center_y_he) * scale_factor + center_y_he

            # twin he
            x_source_twin_he = vertTable_new[twin_he[3], 0]
            y_source_twin_he = vertTable_new[twin_he[3], 1]
            x_target_twin_he = vertTable_new[twin_he[4], 0]
            y_target_twin_he = vertTable_new[twin_he[4], 1]
            offset_x_target_twin_he = heTable_new[twin_he_idx, 6] * L_box
            offset_y_target_twin_he = heTable_new[twin_he_idx, 7] * L_box

            center_x_twin_he = ((x_source_twin_he) + (x_target_twin_he + offset_x_target_twin_he)) / 2
            center_y_twin_he = ((y_source_twin_he) + (y_target_twin_he + offset_y_target_twin_he)) / 2
            angle = jnp.pi / 2.
            turn_x_source_twin_he = center_x_twin_he + jnp.cos(angle) * (x_source_twin_he - center_x_twin_he) - jnp.sin(angle) * (y_source_twin_he - center_y_twin_he)
            turn_y_source_twin_he = center_y_twin_he + jnp.sin(angle) * (x_source_twin_he - center_x_twin_he) + jnp.cos(angle) * (y_source_twin_he - center_y_twin_he)
            scale_factor = ((min_distance + min_distance*0.1) / 2.0) / jnp.sqrt((turn_x_source_twin_he - center_x_twin_he) ** 2 + (turn_y_source_twin_he - center_y_twin_he) ** 2)
            last_x_source_twin_he = (turn_x_source_twin_he - center_x_twin_he) * scale_factor + center_x_twin_he
            last_y_source_twin_he = (turn_y_source_twin_he - center_y_twin_he) * scale_factor + center_y_twin_he

            vertTable_new = vertTable_new.at[he[3], 0].set(last_x_source_he)
            vertTable_new = vertTable_new.at[he[3], 1].set(last_y_source_he)
            vertTable_new = vertTable_new.at[he[4], 0].set(last_x_source_twin_he)
            vertTable_new = vertTable_new.at[he[4], 1].set(last_y_source_twin_he)

            vertTable_new = vertTable_new.at[twin_he[3], 0].set(last_x_source_twin_he)
            vertTable_new = vertTable_new.at[twin_he[3], 1].set(last_y_source_twin_he)
            vertTable_new = vertTable_new.at[twin_he[4], 0].set(last_x_source_he)
            vertTable_new = vertTable_new.at[twin_he[4], 1].set(last_y_source_he)

            ## faceTable
            faceTable_new = faceTable_new.at[prev_he[5]].set(prev_he_idx)
            faceTable_new = faceTable_new.at[twin_prev_he[5]].set(twin_prev_he_idx)
            faceTable_new = faceTable_new.at[next_twin_he[5]].set(next_twin_he_idx)
            faceTable_new = faceTable_new.at[twin_next_he[5]].set(twin_next_he_idx)

            vertTable_new_T1, heTable_new_T1, faceTable_new_T1 = update_pbc(vertTable_new, heTable_new, faceTable_new)

            # Compute final L_in after T1
            L_in_T1 = L_in(vertTable_new_T1, heTable_new_T1, faceTable_new_T1, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)

            vertTable_new, heTable_new, faceTable_new = _state

            # he
            x_source_he = vertTable_new[he[3], 0]
            y_source_he = vertTable_new[he[3], 1]
            x_target_he = vertTable_new[he[4], 0]
            y_target_he = vertTable_new[he[4], 1]
            offset_x_target_he = heTable_new[he_idx, 6] * L_box
            offset_y_target_he = heTable_new[he_idx, 7] * L_box

            center_x_he = ((x_source_he) + (x_target_he + offset_x_target_he)) / 2
            center_y_he = ((y_source_he) + (y_target_he + offset_y_target_he)) / 2
            angle = 0.
            turn_x_source_he = center_x_he + jnp.cos(angle) * (x_source_he - center_x_he) - jnp.sin(angle) * (y_source_he - center_y_he)
            turn_y_source_he = center_y_he + jnp.sin(angle) * (x_source_he - center_x_he) + jnp.cos(angle) * (y_source_he - center_y_he)
            scale_factor = ((min_distance + min_distance*0.1) / 2.0) / jnp.sqrt((turn_x_source_he - center_x_he) ** 2 + (turn_y_source_he - center_y_he) ** 2)
            last_x_source_he = (turn_x_source_he - center_x_he) * scale_factor + center_x_he
            last_y_source_he = (turn_y_source_he - center_y_he) * scale_factor + center_y_he

            # twin he
            x_source_twin_he = vertTable_new[twin_he[3], 0]
            y_source_twin_he = vertTable_new[twin_he[3], 1]
            x_target_twin_he = vertTable_new[twin_he[4], 0]
            y_target_twin_he = vertTable_new[twin_he[4], 1]
            offset_x_target_twin_he = heTable_new[twin_he_idx, 6] * L_box
            offset_y_target_twin_he = heTable_new[twin_he_idx, 7] * L_box

            center_x_twin_he = ((x_source_twin_he) + (x_target_twin_he + offset_x_target_twin_he)) / 2
            center_y_twin_he = ((y_source_twin_he) + (y_target_twin_he + offset_y_target_twin_he)) / 2
            angle = 0.
            turn_x_source_twin_he = center_x_twin_he + jnp.cos(angle) * (x_source_twin_he - center_x_twin_he) - jnp.sin(angle) * (y_source_twin_he - center_y_twin_he)
            turn_y_source_twin_he = center_y_twin_he + jnp.sin(angle) * (x_source_twin_he - center_x_twin_he) + jnp.cos(angle) * (y_source_twin_he - center_y_twin_he)
            scale_factor = ((min_distance + min_distance*0.1) / 2.0) / jnp.sqrt((turn_x_source_twin_he - center_x_twin_he) ** 2 + (turn_y_source_twin_he - center_y_twin_he) ** 2)
            last_x_source_twin_he = (turn_x_source_twin_he - center_x_twin_he) * scale_factor + center_x_twin_he
            last_y_source_twin_he = (turn_y_source_twin_he - center_y_twin_he) * scale_factor + center_y_twin_he

            vertTable_new = vertTable_new.at[he[3], 0].set(last_x_source_he)
            vertTable_new = vertTable_new.at[he[3], 1].set(last_y_source_he)
            vertTable_new = vertTable_new.at[he[4], 0].set(last_x_source_twin_he)
            vertTable_new = vertTable_new.at[he[4], 1].set(last_y_source_twin_he)

            vertTable_new = vertTable_new.at[twin_he[3], 0].set(last_x_source_twin_he)
            vertTable_new = vertTable_new.at[twin_he[3], 1].set(last_y_source_twin_he)
            vertTable_new = vertTable_new.at[twin_he[4], 0].set(last_x_source_he)
            vertTable_new = vertTable_new.at[twin_he[4], 1].set(last_y_source_he)

            vertTable_new_no_T1, heTable_new_no_T1, faceTable_new_no_T1 = update_pbc(vertTable_new, heTable_new, faceTable_new)

            # Compute initial L_in
            L_in_no_T1 = L_in(vertTable_new_no_T1, heTable_new_no_T1, faceTable_new_no_T1, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)

            # Accept the update only if L_in_after < L_in_before
            return jax.lax.cond(
                L_in_T1 < L_in_no_T1,
                lambda _: (vertTable_new_T1, heTable_new_T1, faceTable_new_T1),
                lambda _: (vertTable_new_no_T1, heTable_new_no_T1, faceTable_new_no_T1),
                None
            )

        vertTable_new, heTable_new, faceTable_new = jax.lax.cond(
            (distance < min_distance) & should_update & twin_should_update,
            update_state,
            lambda _state: _state,
            (vertTable_new, heTable_new, faceTable_new)
        )

        return vertTable_new, heTable_new, faceTable_new

    n_cells = len(faceTable)
    L_box = jnp.sqrt(n_cells)

    state = (vertTable, heTable, faceTable)

    vertTable_last, heTable_last, faceTable_last = jax.lax.fori_loop(0, len(heTable) // 2, lambda i, s: body_fun(2 * i, s), state)

    return vertTable_last, heTable_last, faceTable_last  


### WORKING BUT WITHOUT CHECKING THE ENERGY DECREASE

# @jit
# def update_T1(vertTable: jnp.array, 
#               heTable: jnp.array, 
#               faceTable: jnp.array, 
#               min_distance: float):

#     def body_fun(idx, state):

#         vertTable_new, heTable_new, faceTable_new = state

#         he = heTable_new[idx]
#         prev_he = heTable_new[he[0]]
#         next_he = heTable_new[he[1]]
#         twin_he = heTable_new[he[2]]
#         twin_prev_he = heTable_new[prev_he[2]]
#         next_twin_he = heTable_new[twin_he[1]]
#         twin_next_he = heTable_new[next_he[2]]

#         he_idx = idx
#         prev_he_idx = heTable_new[he_idx, 0]
#         next_he_idx = heTable_new[he_idx, 1]
#         twin_he_idx = heTable_new[he_idx, 2]
#         twin_prev_he_idx = prev_he[2]
#         prev_twin_prev_he_idx = twin_prev_he[0]
#         prev_twin_he_idx = twin_he[0]
#         next_twin_he_idx = twin_he[1]
#         twin_next_he_idx = next_he[2]
#         next_twin_next_he_idx = twin_next_he[1]

#         # check distance
#         distance = get_length(he_idx, vertTable_new, heTable_new, L_box)[0]
        
#         # check if the two faces that share the hes are triangles
#         he_prev = he[0]
#         twin_he_prev = twin_he[0]
#         should_update = heTable_new[he_prev, 0] != he[1]
#         twin_should_update = heTable_new[twin_he_prev, 0] != twin_he[1]

#         def update_state(_state):

#             vertTable_new, heTable_new, faceTable_new = _state
            
#             ## heTable
#             # he
#             heTable_new = heTable_new.at[he_idx, 0].set(prev_twin_prev_he_idx)
#             heTable_new = heTable_new.at[he_idx, 1].set(twin_prev_he_idx)
#             heTable_new = heTable_new.at[he_idx, 5].set(twin_prev_he[5])

#             # twin he
#             heTable_new = heTable_new.at[twin_he_idx, 0].set(twin_next_he_idx)
#             heTable_new = heTable_new.at[twin_he_idx, 1].set(next_twin_next_he_idx)
#             heTable_new = heTable_new.at[twin_he_idx, 5].set(twin_next_he[5])

#             # prev he
#             heTable_new = heTable_new.at[prev_he_idx, 1].set(next_he_idx)
#             heTable_new = heTable_new.at[prev_he_idx, 4].set(he[4])

#             # next he
#             heTable_new = heTable_new.at[next_he_idx, 0].set(prev_he_idx)

#             # prev twin he
#             heTable_new = heTable_new.at[prev_twin_he_idx, 1].set(next_twin_he_idx)
#             heTable_new = heTable_new.at[prev_twin_he_idx, 4].set(twin_he[4])

#             # next twin he
#             heTable_new = heTable_new.at[next_twin_he_idx, 0].set(prev_twin_he_idx)

#             # prev twin prev he
#             heTable_new = heTable_new.at[prev_twin_prev_he_idx, 1].set(he_idx)

#             # twin prev he
#             heTable_new = heTable_new.at[twin_prev_he_idx, 0].set(he_idx)
#             heTable_new = heTable_new.at[twin_prev_he_idx, 3].set(he[4])

#             # twin next he
#             heTable_new = heTable_new.at[twin_next_he_idx, 1].set(twin_he_idx)

#             # next twin next he
#             heTable_new = heTable_new.at[next_twin_next_he_idx, 0].set(twin_he_idx)
#             heTable_new = heTable_new.at[next_twin_next_he_idx, 3].set(he[3])

#             ## vertTable
#             vertTable_new = vertTable_new.at[he[3], 2].set(he_idx)
#             vertTable_new = vertTable_new.at[twin_he[3], 2].set(twin_he_idx)

#             # he
#             x_source_he = vertTable_new[he[3], 0]
#             y_source_he = vertTable_new[he[3], 1]
#             x_target_he = vertTable_new[he[4], 0]
#             y_target_he = vertTable_new[he[4], 1]
#             offset_x_target_he = heTable_new[he_idx, 6] * L_box
#             offset_y_target_he = heTable_new[he_idx, 7] * L_box

#             center_x_he = ((x_source_he) + (x_target_he + offset_x_target_he)) / 2
#             center_y_he = ((y_source_he) + (y_target_he + offset_y_target_he)) / 2
#             angle = jnp.pi / 2.
#             turn_x_source_he = center_x_he + jnp.cos(angle) * (x_source_he - center_x_he) - jnp.sin(angle) * (y_source_he - center_y_he)
#             turn_y_source_he = center_y_he + jnp.sin(angle) * (x_source_he - center_x_he) + jnp.cos(angle) * (y_source_he - center_y_he)
#             scale_factor = ((min_distance + min_distance*0.1) / 2.0) / jnp.sqrt((turn_x_source_he - center_x_he) ** 2 + (turn_y_source_he - center_y_he) ** 2)
#             last_x_source_he = (turn_x_source_he - center_x_he) * scale_factor + center_x_he
#             last_y_source_he = (turn_y_source_he - center_y_he) * scale_factor + center_y_he

#             center_x_he = ((x_source_he) + (x_target_he + offset_x_target_he)) / 2
#             center_y_he = ((y_source_he) + (y_target_he + offset_y_target_he)) / 2
#             angle = jnp.pi / 2.
#             turn_x_target_he = center_x_he + jnp.cos(angle) * (x_target_he - center_x_he) - jnp.sin(angle) * (y_target_he - center_y_he)
#             turn_y_target_he = center_y_he + jnp.sin(angle) * (x_target_he - center_x_he) + jnp.cos(angle) * (y_target_he - center_y_he)
#             scale_factor = ((min_distance + min_distance*0.1) / 2.0) / jnp.sqrt((turn_x_target_he - center_x_he) ** 2 + (turn_y_target_he - center_y_he) ** 2)
#             last_x_target_he = (turn_x_target_he - center_x_he) * scale_factor + center_x_he
#             last_y_target_he = (turn_y_target_he - center_y_he) * scale_factor + center_y_he

#             # twin he
#             x_source_twin_he = vertTable_new[twin_he[3], 0]
#             y_source_twin_he = vertTable_new[twin_he[3], 1]
#             x_target_twin_he = vertTable_new[twin_he[4], 0]
#             y_target_twin_he = vertTable_new[twin_he[4], 1]
#             offset_x_target_twin_he = heTable_new[twin_he_idx, 6] * L_box
#             offset_y_target_twin_he = heTable_new[twin_he_idx, 7] * L_box

#             center_x_twin_he = ((x_source_twin_he) + (x_target_twin_he + offset_x_target_twin_he)) / 2
#             center_y_twin_he = ((y_source_twin_he) + (y_target_twin_he + offset_y_target_twin_he)) / 2
#             angle = jnp.pi / 2.
#             turn_x_source_twin_he = center_x_twin_he + jnp.cos(angle) * (x_source_twin_he - center_x_twin_he) - jnp.sin(angle) * (y_source_twin_he - center_y_twin_he)
#             turn_y_source_twin_he = center_y_twin_he + jnp.sin(angle) * (x_source_twin_he - center_x_twin_he) + jnp.cos(angle) * (y_source_twin_he - center_y_twin_he)
#             scale_factor = ((min_distance + min_distance*0.1) / 2.0) / jnp.sqrt((turn_x_source_twin_he - center_x_twin_he) ** 2 + (turn_y_source_twin_he - center_y_twin_he) ** 2)
#             last_x_source_twin_he = (turn_x_source_twin_he - center_x_twin_he) * scale_factor + center_x_twin_he
#             last_y_source_twin_he = (turn_y_source_twin_he - center_y_twin_he) * scale_factor + center_y_twin_he

#             center_x_twin_he = ((x_source_twin_he) + (x_target_twin_he + offset_x_target_twin_he)) / 2
#             center_y_twin_he = ((y_source_twin_he) + (y_target_twin_he + offset_y_target_twin_he)) / 2
#             angle = jnp.pi / 2.
#             turn_x_target_twin_he = center_x_twin_he + jnp.cos(angle) * (x_target_twin_he - center_x_twin_he) - jnp.sin(angle) * (y_target_twin_he - center_y_twin_he)
#             turn_y_target_twin_he = center_y_twin_he + jnp.sin(angle) * (x_target_twin_he - center_x_twin_he) + jnp.cos(angle) * (y_target_twin_he - center_y_twin_he)
#             scale_factor = ((min_distance + min_distance*0.1) / 2.0) / jnp.sqrt((turn_x_target_twin_he - center_x_twin_he) ** 2 + (turn_y_target_twin_he - center_y_twin_he) ** 2)
#             last_x_target_twin_he = (turn_x_target_twin_he - center_x_twin_he) * scale_factor + center_x_twin_he
#             last_y_target_twin_he = (turn_y_target_twin_he - center_y_twin_he) * scale_factor + center_y_twin_he

#             vertTable_new = vertTable_new.at[he[3], 0].set(last_x_source_he)
#             vertTable_new = vertTable_new.at[he[3], 1].set(last_y_source_he)
#             vertTable_new = vertTable_new.at[he[4], 0].set(last_x_target_he)
#             vertTable_new = vertTable_new.at[he[4], 1].set(last_y_target_he)

#             vertTable_new = vertTable_new.at[twin_he[3], 0].set(last_x_source_twin_he)
#             vertTable_new = vertTable_new.at[twin_he[3], 1].set(last_y_source_twin_he)
#             vertTable_new = vertTable_new.at[twin_he[4], 0].set(last_x_target_twin_he)
#             vertTable_new = vertTable_new.at[twin_he[4], 1].set(last_y_target_twin_he)

#             ## faceTable
#             faceTable_new = faceTable_new.at[prev_he[5]].set(prev_he_idx)
#             faceTable_new = faceTable_new.at[twin_prev_he[5]].set(twin_prev_he_idx)
#             faceTable_new = faceTable_new.at[next_twin_he[5]].set(next_twin_he_idx)
#             faceTable_new = faceTable_new.at[twin_next_he[5]].set(twin_next_he_idx)

#             return update_pbc(vertTable_new, heTable_new, faceTable_new)

#         vertTable_new, heTable_new, faceTable_new = jax.lax.cond(
#             (distance < min_distance) & should_update & twin_should_update,
#             update_state,
#             lambda _state: _state,
#             (vertTable_new, heTable_new, faceTable_new)
#         )

#         return vertTable_new, heTable_new, faceTable_new

#     n_cells = len(faceTable)
#     L_box = jnp.sqrt(n_cells)

#     state = (vertTable, heTable, faceTable)

#     vertTable_last, heTable_last, faceTable_last = jax.lax.fori_loop(0, len(heTable) // 2, lambda i, s: body_fun(2 * i, s), state)

#     return vertTable_last, heTable_last, faceTable_last  
