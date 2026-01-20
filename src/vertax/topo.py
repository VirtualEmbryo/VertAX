"""Topology module to handle T1 transitions in meshes."""

from functools import partial

import jax
import jax.lax
import jax.numpy as jnp
from jax import Array, jit

from vertax import opt, opt_bounded
from vertax.geo import get_length, update_pbc

ROT_ANGLE = -jnp.pi / 2
SINA = jnp.sin(ROT_ANGLE)
COSA = jnp.cos(ROT_ANGLE)


@partial(jit, static_argnums=(3, 4, 8))
def update_T1(  # noqa: N802
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: opt.InnerLossFunction,
    min_dist_T1: float,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
) -> tuple[Array, Array, Array]:
    """Detect and apply T1 topological changes."""
    # if selected_verts is None:
    #     selected_verts = jnp.arange(vertTable.shape[0])
    # if selected_hes is None:
    #     selected_hes = jnp.arange(heTable.shape[0])
    # if selected_faces is None:
    #     selected_faces = jnp.arange(faceTable.shape[0])

    def body_fun(idx: Array, state: tuple[Array, Array, Array]) -> Array:
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
        distance = get_length(he_idx, vertTable_new, heTable_new, faceTable_new, width, height)

        # check if the two faces that share the hes are triangles
        he_prev = he[0]
        twin_he_prev = twin_he[0]
        should_update = heTable_new[he_prev, 0] != he[1]
        twin_should_update = heTable_new[twin_he_prev, 0] != twin_he[1]

        def update_state(_state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
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
            offset_x_target_he = heTable_new[he_idx, 6] * width
            offset_y_target_he = heTable_new[he_idx, 7] * height

            center_x_he = ((x_source_he) + (x_target_he + offset_x_target_he)) / 2
            center_y_he = ((y_source_he) + (y_target_he + offset_y_target_he)) / 2
            angle = jnp.pi / 2.0
            turn_x_source_he = (
                center_x_he
                + jnp.cos(angle) * (x_source_he - center_x_he)
                - jnp.sin(angle) * (y_source_he - center_y_he)
            )
            turn_y_source_he = (
                center_y_he
                + jnp.sin(angle) * (x_source_he - center_x_he)
                + jnp.cos(angle) * (y_source_he - center_y_he)
            )
            scale_factor = ((min_dist_T1 + min_dist_T1 * 0.1) / 2.0) / jnp.sqrt(
                (turn_x_source_he - center_x_he) ** 2 + (turn_y_source_he - center_y_he) ** 2
            )
            last_x_source_he = (turn_x_source_he - center_x_he) * scale_factor + center_x_he
            last_y_source_he = (turn_y_source_he - center_y_he) * scale_factor + center_y_he

            # twin he
            x_source_twin_he = vertTable_new[twin_he[3], 0]
            y_source_twin_he = vertTable_new[twin_he[3], 1]
            x_target_twin_he = vertTable_new[twin_he[4], 0]
            y_target_twin_he = vertTable_new[twin_he[4], 1]
            offset_x_target_twin_he = heTable_new[twin_he_idx, 6] * width
            offset_y_target_twin_he = heTable_new[twin_he_idx, 7] * height

            center_x_twin_he = ((x_source_twin_he) + (x_target_twin_he + offset_x_target_twin_he)) / 2
            center_y_twin_he = ((y_source_twin_he) + (y_target_twin_he + offset_y_target_twin_he)) / 2
            angle = jnp.pi / 2.0
            turn_x_source_twin_he = (
                center_x_twin_he
                + jnp.cos(angle) * (x_source_twin_he - center_x_twin_he)
                - jnp.sin(angle) * (y_source_twin_he - center_y_twin_he)
            )
            turn_y_source_twin_he = (
                center_y_twin_he
                + jnp.sin(angle) * (x_source_twin_he - center_x_twin_he)
                + jnp.cos(angle) * (y_source_twin_he - center_y_twin_he)
            )
            scale_factor = ((min_dist_T1 + min_dist_T1 * 0.1) / 2.0) / jnp.sqrt(
                (turn_x_source_twin_he - center_x_twin_he) ** 2 + (turn_y_source_twin_he - center_y_twin_he) ** 2
            )
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

            vertTable_new_T1, heTable_new_T1, faceTable_new_T1 = update_pbc(
                vertTable_new, heTable_new, faceTable_new, width, height
            )

            # Compute final L_in after T1
            L_in_T1 = L_in(
                vertTable_new_T1[selected_verts],
                heTable_new_T1[selected_hes],
                faceTable_new_T1[selected_faces],
                vert_params,
                he_params,
                face_params,
            )

            vertTable_new, heTable_new, faceTable_new = _state

            # he
            x_source_he = vertTable_new[he[3], 0]
            y_source_he = vertTable_new[he[3], 1]
            x_target_he = vertTable_new[he[4], 0]
            y_target_he = vertTable_new[he[4], 1]
            offset_x_target_he = heTable_new[he_idx, 6] * width
            offset_y_target_he = heTable_new[he_idx, 7] * height

            center_x_he = ((x_source_he) + (x_target_he + offset_x_target_he)) / 2
            center_y_he = ((y_source_he) + (y_target_he + offset_y_target_he)) / 2
            angle = 0.0
            turn_x_source_he = (
                center_x_he
                + jnp.cos(angle) * (x_source_he - center_x_he)
                - jnp.sin(angle) * (y_source_he - center_y_he)
            )
            turn_y_source_he = (
                center_y_he
                + jnp.sin(angle) * (x_source_he - center_x_he)
                + jnp.cos(angle) * (y_source_he - center_y_he)
            )
            scale_factor = ((min_dist_T1 + min_dist_T1 * 0.1) / 2.0) / jnp.sqrt(
                (turn_x_source_he - center_x_he) ** 2 + (turn_y_source_he - center_y_he) ** 2
            )
            last_x_source_he = (turn_x_source_he - center_x_he) * scale_factor + center_x_he
            last_y_source_he = (turn_y_source_he - center_y_he) * scale_factor + center_y_he

            # twin he
            x_source_twin_he = vertTable_new[twin_he[3], 0]
            y_source_twin_he = vertTable_new[twin_he[3], 1]
            x_target_twin_he = vertTable_new[twin_he[4], 0]
            y_target_twin_he = vertTable_new[twin_he[4], 1]
            offset_x_target_twin_he = heTable_new[twin_he_idx, 6] * width
            offset_y_target_twin_he = heTable_new[twin_he_idx, 7] * height

            center_x_twin_he = ((x_source_twin_he) + (x_target_twin_he + offset_x_target_twin_he)) / 2
            center_y_twin_he = ((y_source_twin_he) + (y_target_twin_he + offset_y_target_twin_he)) / 2
            angle = 0.0
            turn_x_source_twin_he = (
                center_x_twin_he
                + jnp.cos(angle) * (x_source_twin_he - center_x_twin_he)
                - jnp.sin(angle) * (y_source_twin_he - center_y_twin_he)
            )
            turn_y_source_twin_he = (
                center_y_twin_he
                + jnp.sin(angle) * (x_source_twin_he - center_x_twin_he)
                + jnp.cos(angle) * (y_source_twin_he - center_y_twin_he)
            )
            scale_factor = ((min_dist_T1 + min_dist_T1 * 0.1) / 2.0) / jnp.sqrt(
                (turn_x_source_twin_he - center_x_twin_he) ** 2 + (turn_y_source_twin_he - center_y_twin_he) ** 2
            )
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

            vertTable_new_no_T1, heTable_new_no_T1, faceTable_new_no_T1 = update_pbc(
                vertTable_new, heTable_new, faceTable_new, width, height
            )

            # Compute initial L_in
            L_in_no_T1 = L_in(
                vertTable_new_no_T1[selected_verts],
                heTable_new_no_T1[selected_hes],
                faceTable_new_no_T1[selected_faces],
                vert_params,
                he_params,
                face_params,
            )

            # Accept the update only if L_in_after < L_in_before
            return jax.lax.cond(
                L_in_T1 <= L_in_no_T1,
                lambda _: (vertTable_new_T1, heTable_new_T1, faceTable_new_T1),
                lambda _: (vertTable_new_no_T1, heTable_new_no_T1, faceTable_new_no_T1),
                None,
            )

        vertTable_new, heTable_new, faceTable_new = jax.lax.cond(
            (distance <= min_dist_T1) & should_update & twin_should_update,
            update_state,
            lambda _state: _state,
            (vertTable_new, heTable_new, faceTable_new),
        )

        return vertTable_new, heTable_new, faceTable_new  # ty:ignore[invalid-return-type]

    state = (vertTable, heTable, faceTable)

    vertTable_last, heTable_last, faceTable_last = jax.lax.fori_loop(
        0, len(heTable) // 2, lambda i, s: body_fun(2 * i, s), state
    )

    return vertTable_last, heTable_last, faceTable_last


@partial(jit, static_argnums=(3, 4, 8))
def do_not_update_T1(  # noqa: N802
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,  # noqa: ARG001
    height: float,  # noqa: ARG001
    vert_params: Array,  # noqa: ARG001
    he_params: Array,  # noqa: ARG001
    face_params: Array,  # noqa: ARG001
    L_in: opt.InnerLossFunction,  # noqa: ARG001
    min_dist_T1: float,  # noqa: ARG001
    selected_verts: Array | None,  # noqa: ARG001
    selected_hes: Array | None,  # noqa: ARG001
    selected_faces: Array | None,  # noqa: ARG001
) -> tuple[Array, Array, Array]:
    """Do nothing function."""
    return vertTable, heTable, faceTable


# ==========
# Bounded
# ==========
@partial(jit, static_argnums=(7, 8))
def update_T1_bounded(  # noqa: C901, N802
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,  # noqa: ARG001
    he_params: Array,  # noqa: ARG001
    face_params: Array,  # noqa: ARG001
    L_in: opt_bounded.InnerLossFunction,  # noqa: ARG001
    min_dist_T1: float,
    selected_verts: Array | None = None,  # noqa: ARG001
    selected_hes: Array | None = None,  # noqa: ARG001
    selected_faces: Array | None = None,  # noqa: ARG001
) -> tuple[Array, Array, Array, Array]:
    """Find and apply T1 topological transitions for bounded meshes."""
    num_edges = heTable.shape[0] // 2
    angTable = jnp.clip(angTable, 0.017, jnp.pi / 2 - 0.001)
    scale = (min_dist_T1 + 10 ** (-3)) / 2

    def true_fun4(
        stacked_data: tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[Array, Array, Array, Array]:
        (
            idx,
            idx_twin,
            v_idx_source,
            v_idx_target,
            x1,
            y1,
            x2,
            y2,
            he_prev,
            he_next,
            he_twin_prev,
            he_twin_next,
            he_face,
            he_twin_face,
            vertTable_,
            angTable_,
            heTable_,
            faceTable_,
        ) = stacked_data

        he_prev_twin = heTable_.at[he_prev, 2].get()
        he_twin_next_twin = heTable_.at[he_twin_next, 2].get()
        he_twin_next_twin_next = heTable_.at[he_twin_next_twin, 1].get()
        he_twin_next_twin_face = heTable_.at[he_twin_next_twin, 7].get()

        heTable_ = heTable_.at[he_prev, 1].set(he_next)
        heTable_ = heTable_.at[he_twin_prev, 1].set(he_twin_next)
        heTable_ = heTable_.at[he_prev_twin, 0].set(idx_twin)
        heTable_ = heTable_.at[he_next, 0].set(he_prev)
        heTable_ = heTable_.at[he_next, 5].set(v_idx_source)
        heTable_ = heTable_.at[he_twin_next, 0].set(he_twin_prev)
        heTable_ = heTable_.at[he_twin_next, 3].set(v_idx_target)
        heTable_ = heTable_.at[he_twin_next_twin, 1].set(idx_twin)
        heTable_ = heTable_.at[he_twin_next_twin, 4].set(v_idx_target)

        heTable_ = heTable_.at[idx, 0].set(0)
        heTable_ = heTable_.at[idx, 1].set(0)
        heTable_ = heTable_.at[idx, 3].set(0)
        heTable_ = heTable_.at[idx, 4].set(1)
        heTable_ = heTable_.at[idx, 7].set(0)
        heTable_ = heTable_.at[idx_twin, 0].set(he_twin_next_twin)
        heTable_ = heTable_.at[idx_twin, 1].set(he_twin_next_twin_next)
        heTable_ = heTable_.at[idx_twin, 3].set(0)
        heTable_ = heTable_.at[idx_twin, 4].set(1)
        heTable_ = heTable_.at[idx_twin, 5].set(v_idx_target)
        heTable_ = heTable_.at[idx_twin, 6].set(v_idx_source)
        heTable_ = heTable_.at[idx_twin, 7].set(he_twin_next_twin_face)

        faceTable_ = faceTable_.at[he_face, 0].set(he_next)
        faceTable_ = faceTable_.at[he_twin_face, 0].set(he_twin_next)

        vertTable_ = vertTable_.at[v_idx_source - 2, 0].set(x1)
        vertTable_ = vertTable_.at[v_idx_source - 2, 1].set(y1)
        vertTable_ = vertTable_.at[v_idx_target - 2, 0].set(x2)
        vertTable_ = vertTable_.at[v_idx_target - 2, 1].set(y2)

        angTable_ = angTable_.at[idx_twin // 2].set(0.017)
        return vertTable_, angTable_, heTable_, faceTable_

    def false_fun4(
        stacked_data: tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[Array, Array, Array, Array]:
        (
            idx,
            idx_twin,
            v_idx_source,
            v_idx_target,
            x1,
            y1,
            x2,
            y2,
            he_prev,
            he_next,
            he_twin_prev,
            he_twin_next,
            he_face,
            he_twin_face,
            vertTable_,
            angTable_,
            heTable_,
            faceTable_,
        ) = stacked_data

        he_prev_twin = heTable_.at[he_prev, 2].get()
        he_twin_prev_twin = heTable_.at[he_twin_prev, 2].get()
        he_next_twin = heTable_.at[he_next, 2].get()
        he_twin_next_twin = heTable_.at[he_twin_next, 2].get()
        he_next_twin_next = heTable_.at[he_next_twin, 1].get()
        he_next_twin_face = heTable_.at[he_next_twin, 7].get()
        he_twin_next_twin_next = heTable_.at[he_twin_next_twin, 1].get()
        he_twin_next_twin_face = heTable_.at[he_twin_next_twin, 7].get()

        heTable_ = heTable_.at[he_prev, 1].set(he_next)  # change prev_he's next_he with he's next_he
        heTable_ = heTable_.at[he_twin_prev, 1].set(he_twin_next)
        heTable_ = heTable_.at[he_prev_twin, 0].set(idx_twin)  # idx  # change prev_twin_he's prev_he with he
        heTable_ = heTable_.at[he_twin_prev_twin, 0].set(idx)
        heTable_ = heTable_.at[he_next, 0].set(he_prev)  # change next_he's prev_he with he's prev_he
        heTable_ = heTable_.at[he_next, 3].set(v_idx_source)  # change next_he's source_vertex with he's source_vertex
        heTable_ = heTable_.at[he_twin_next, 0].set(he_twin_prev)
        heTable_ = heTable_.at[he_twin_next, 3].set(v_idx_target)
        heTable_ = heTable_.at[he_next_twin, 1].set(idx)  # change next_twin_he's next_he with he's twin
        heTable_ = heTable_.at[he_next_twin, 4].set(
            v_idx_source
        )  # change next twin he's target vertex with he's source vertex
        heTable_ = heTable_.at[he_twin_next_twin, 1].set(idx_twin)
        heTable_ = heTable_.at[he_twin_next_twin, 4].set(v_idx_target)

        heTable_ = heTable_.at[idx, 0].set(he_next_twin)
        heTable_ = heTable_.at[idx, 1].set(he_next_twin_next)
        heTable_ = heTable_.at[idx, 7].set(he_next_twin_face)
        heTable_ = heTable_.at[idx_twin, 0].set(he_twin_next_twin)
        heTable_ = heTable_.at[idx_twin, 1].set(he_twin_next_twin_next)
        heTable_ = heTable_.at[idx_twin, 7].set(he_twin_next_twin_face)

        faceTable_ = faceTable_.at[he_face, 0].set(he_next)
        faceTable_ = faceTable_.at[he_twin_face, 0].set(he_twin_next)

        vertTable_ = vertTable_.at[v_idx_source - 2, 0].set(x1)
        vertTable_ = vertTable_.at[v_idx_source - 2, 1].set(y1)
        vertTable_ = vertTable_.at[v_idx_target - 2, 0].set(x2)
        vertTable_ = vertTable_.at[v_idx_target - 2, 1].set(y2)
        return vertTable_, angTable_, heTable_, faceTable_

    def true_fun3(
        stacked_data: tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[Array, Array, Array, Array]:
        (
            idx,
            idx_twin,
            v_idx_source,
            v_idx_target,
            x1,
            y1,
            x2,
            y2,
            he_prev,
            he_next,
            he_twin_prev,
            he_twin_next,
            he_face,
            he_twin_face,
            vertTable_,
            angTable_,
            heTable_,
            faceTable_,
        ) = stacked_data

        he_twin_prev_twin = heTable_.at[he_twin_prev, 2].get()
        he_next_twin = heTable_.at[he_next, 2].get()
        he_next_twin_next = heTable_.at[he_next_twin, 1].get()
        he_next_twin_face = heTable_.at[he_next_twin, 7].get()

        heTable_ = heTable_.at[he_prev, 1].set(he_next)
        heTable_ = heTable_.at[he_twin_prev, 1].set(he_twin_next)
        heTable_ = heTable_.at[he_twin_prev_twin, 0].set(idx)
        heTable_ = heTable_.at[he_next, 0].set(he_prev)
        heTable_ = heTable_.at[he_next, 3].set(v_idx_source)
        heTable_ = heTable_.at[he_twin_next, 0].set(he_twin_prev)
        heTable_ = heTable_.at[he_twin_next, 5].set(v_idx_target)
        heTable_ = heTable_.at[he_next_twin, 1].set(idx)
        heTable_ = heTable_.at[he_next_twin, 4].set(v_idx_source)

        heTable_ = heTable_.at[idx, 0].set(he_next_twin)
        heTable_ = heTable_.at[idx, 1].set(he_next_twin_next)
        heTable_ = heTable_.at[idx, 3].set(0)
        heTable_ = heTable_.at[idx, 4].set(1)
        heTable_ = heTable_.at[idx, 5].set(v_idx_source)
        heTable_ = heTable_.at[idx, 6].set(v_idx_target)
        heTable_ = heTable_.at[idx, 7].set(he_next_twin_face)
        heTable_ = heTable_.at[idx_twin, 0].set(0)
        heTable_ = heTable_.at[idx_twin, 1].set(0)
        heTable_ = heTable_.at[idx_twin, 3].set(0)
        heTable_ = heTable_.at[idx_twin, 4].set(1)
        heTable_ = heTable_.at[idx_twin, 7].set(0)

        faceTable_ = faceTable_.at[he_face, 0].set(he_next)
        faceTable_ = faceTable_.at[he_twin_face, 0].set(he_twin_next)

        vertTable_ = vertTable_.at[v_idx_source - 2, 0].set(x1)
        vertTable_ = vertTable_.at[v_idx_source - 2, 1].set(y1)
        vertTable_ = vertTable_.at[v_idx_target - 2, 0].set(x2)
        vertTable_ = vertTable_.at[v_idx_target - 2, 1].set(y2)

        angTable_ = angTable_.at[idx // 2].set(0.017)
        return vertTable_, angTable_, heTable_, faceTable_

    def false_fun3(
        stacked_data: tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[Array, Array, Array, Array]:
        _, _, _, _, _, _, _, _, _, he_next, _, _, _, _, _, _, heTable_, _ = stacked_data
        externalize2 = heTable_.at[he_next, 3].get() == 0
        return jax.lax.cond(externalize2, true_fun4, false_fun4, stacked_data)

    def true_fun2(
        stacked_data: tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[Array, Array, Array, Array]:
        (
            idx,
            idx_twin,
            v_idx_source,
            v_idx_target,
            x1,
            y1,
            x2,
            y2,
            he_prev,
            he_next,
            _,
            _,
            he_face,
            _,
            vertTable_,
            angTable_,
            heTable_,
            faceTable_,
        ) = stacked_data

        he_prev_twin = heTable_.at[he_prev, 2].get()
        he_next_twin = heTable_.at[he_next, 2].get()
        he_next_twin_next = heTable_.at[he_next_twin, 1].get()
        he_prev_twin_prev = heTable_.at[he_prev_twin, 0].get()
        he_next_twin_face = heTable_.at[he_next_twin, 7].get()
        he_prev_twin_face = heTable_.at[he_prev_twin, 7].get()

        heTable_ = heTable_.at[he_prev, 1].set(he_next)
        heTable_ = heTable_.at[he_prev_twin, 0].set(idx_twin)
        heTable_ = heTable_.at[he_next, 0].set(he_prev)
        heTable_ = heTable_.at[he_next, 3].set(v_idx_source)
        heTable_ = heTable_.at[he_next_twin_next, 0].set(idx)
        heTable_ = heTable_.at[he_next_twin, 1].set(idx)
        heTable_ = heTable_.at[he_next_twin, 4].set(v_idx_source)
        heTable_ = heTable_.at[he_prev_twin_prev, 1].set(idx_twin)
        heTable_ = heTable_.at[he_prev_twin_prev, 6].set(v_idx_target)

        heTable_ = heTable_.at[idx, 0].set(he_next_twin)
        heTable_ = heTable_.at[idx, 1].set(he_next_twin_next)
        heTable_ = heTable_.at[idx, 3].set(v_idx_source)
        heTable_ = heTable_.at[idx, 4].set(v_idx_target)
        heTable_ = heTable_.at[idx, 5].set(0)
        heTable_ = heTable_.at[idx, 6].set(1)
        heTable_ = heTable_.at[idx, 7].set(he_next_twin_face)
        heTable_ = heTable_.at[idx_twin, 0].set(he_prev_twin_prev)
        heTable_ = heTable_.at[idx_twin, 1].set(he_prev_twin)
        heTable_ = heTable_.at[idx_twin, 3].set(v_idx_target)
        heTable_ = heTable_.at[idx_twin, 4].set(v_idx_source)
        heTable_ = heTable_.at[idx_twin, 7].set(he_prev_twin_face)

        faceTable_ = faceTable_.at[he_face, 0].set(he_next)

        vertTable_ = vertTable_.at[v_idx_source - 2, 0].set(x1)
        vertTable_ = vertTable_.at[v_idx_source - 2, 1].set(y1)
        vertTable_ = vertTable_.at[v_idx_target - 2, 0].set(x2)
        vertTable_ = vertTable_.at[v_idx_target - 2, 1].set(y2)

        angTable_ = angTable_.at[idx // 2].set(1.0)
        return vertTable_, angTable_, heTable_, faceTable_

    def false_fun2(
        stacked_data: tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[Array, Array, Array, Array]:
        _, _, _, _, _, _, _, _, he_prev, _, _, _, _, _, _, _, heTable_, _ = stacked_data
        externalize1 = heTable_.at[he_prev, 3].get() == 0
        return jax.lax.cond(externalize1, true_fun3, false_fun3, stacked_data)

    def true_fun1(
        stacked_data: tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[Array, Array, Array, Array]:
        (
            x1,
            y1,
            x2,
            y2,
            cx,
            cy,
            x1_norm,
            x2_norm,
            _,
            _,
            v_source,
            v_target,
            _,
            _,
            _,
            _,
            _,
            _,
            vertTable_,
            angTable_,
            heTable_,
            faceTable_,
        ) = stacked_data

        # scale at larger size than minimal distance
        sx1 = (x1 - cx) * scale / x1_norm + cx
        sy1 = (y1 - cy) * scale / x1_norm + cy
        sx2 = (x2 - cx) * scale / x2_norm + cx
        sy2 = (y2 - cy) * scale / x2_norm + cy

        vertTable_ = vertTable_.at[v_source - 2, 0].set(sx1)
        vertTable_ = vertTable_.at[v_source - 2, 1].set(sy1)
        vertTable_ = vertTable_.at[v_target - 2, 0].set(sx2)
        vertTable_ = vertTable_.at[v_target - 2, 1].set(sy2)

        return vertTable_, angTable_, heTable_, faceTable_

    def false_fun1(
        stacked_data: tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
    ) -> tuple[Array, Array, Array, Array]:
        (
            x1,
            y1,
            x2,
            y2,
            cx,
            cy,
            x1_norm,
            x2_norm,
            idx,
            idx_twin,
            v_source,
            v_target,
            he_prev,
            he_next,
            he_twin_prev,
            he_twin_next,
            he_face,
            he_twin_face,
            vertTable_,
            angTable_,
            heTable_,
            faceTable_,
        ) = stacked_data

        # rotate of 90 degrees counterclockwise
        rx1 = cx + COSA * (x1 - cx) - SINA * (y1 - cy)
        ry1 = cy + SINA * (x1 - cx) + COSA * (y1 - cy)
        rx2 = cx + COSA * (x2 - cx) - SINA * (y2 - cy)
        ry2 = cy + SINA * (x2 - cx) + COSA * (y2 - cy)

        # scale at larger size than minimal distance
        nx1 = (rx1 - cx) * scale / x1_norm + cx
        ny1 = (ry1 - cy) * scale / x1_norm + cy
        nx2 = (rx2 - cx) * scale / x2_norm + cx
        ny2 = (ry2 - cy) * scale / x2_norm + cy

        internalize = heTable_.at[idx, 3].get() == 0
        return jax.lax.cond(
            internalize,
            true_fun2,
            false_fun2,
            (
                idx,
                idx_twin,
                v_source,
                v_target,
                nx1,
                ny1,
                nx2,
                ny2,
                he_prev,
                he_next,
                he_twin_prev,
                he_twin_next,
                he_face,
                he_twin_face,
                vertTable_,
                angTable_,
                heTable_,
                faceTable_,
            ),
        )

    def true_fun0(
        stacked_data: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        x1, y1, x2, y2, idx, v_source, v_target, vertTable_, angTable_, heTable_, faceTable_ = stacked_data

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        x1_norm = jnp.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        x2_norm = jnp.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)

        he = heTable_[idx]
        idx_twin = he[2]
        he_twin = heTable_[idx_twin]

        he_prev = he[0]
        he_next = he[1]
        he_twin_prev = he_twin[0]
        he_twin_next = he_twin[1]
        he_face = he[7]
        he_twin_face = he_twin[7]

        emergency = jnp.logical_and(heTable_.at[he_prev, 3].get() == 0, heTable_.at[he_next, 3].get() == 0)
        emergency = jnp.logical_or(emergency, heTable_.at[he_next, 1].get() == he_prev)
        emergency = jnp.logical_or(emergency, heTable_.at[he_twin_next, 1].get() == he_twin_prev)
        emergency = jnp.logical_or(emergency, jnp.sum(heTable_[:, 5] != 0) == 2)
        return jax.lax.cond(
            emergency,
            true_fun1,
            false_fun1,
            (
                x1,
                y1,
                x2,
                y2,
                cx,
                cy,
                x1_norm,
                x2_norm,
                idx,
                idx_twin,
                v_source,
                v_target,
                he_prev,
                he_next,
                he_twin_prev,
                he_twin_next,
                he_face,
                he_twin_face,
                vertTable_,
                angTable_,
                heTable_,
                faceTable_,
            ),
        )

    def false_fun0(
        stacked_data: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        _, _, _, _, _, _, _, vertTable_, angTable_, heTable_, faceTable_ = stacked_data
        return vertTable_, angTable_, heTable_, faceTable_

    def body_fun(i: int, stacked_data: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array, Array]:
        vertTable_, angTable_, heTable_, faceTable_ = stacked_data
        selector = (heTable_.at[2 * i, 3].get() + heTable_.at[2 * i, 5].get()) == 0
        idx = 2 * i + selector.astype("int32")
        v_source = heTable_.at[idx, 3].get() + heTable_.at[idx, 5].get()
        v_target = heTable_.at[idx, 4].get() + heTable_.at[idx, 6].get() - 1
        x0 = vertTable_.at[v_source - 2, 0].get()
        y0 = vertTable_.at[v_source - 2, 1].get()
        x1 = vertTable_.at[v_target - 2, 0].get()
        y1 = vertTable_.at[v_target - 2, 1].get()
        dist = jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        T1_trigger = dist < min_dist_T1
        return jax.lax.cond(
            T1_trigger,
            true_fun0,
            false_fun0,
            (x0, y0, x1, y1, idx, v_source, v_target, vertTable_, angTable_, heTable_, faceTable_),
        )

    return jax.lax.fori_loop(0, num_edges, body_fun, (vertTable, angTable, heTable, faceTable))


@partial(jit, static_argnums=(7, 8))
def do_not_update_T1_bounded(  # noqa: N802
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,  # noqa: ARG001
    he_params: Array,  # noqa: ARG001
    face_params: Array,  # noqa: ARG001
    L_in: opt_bounded.InnerLossFunction,  # noqa: ARG001
    min_dist_T1: float,  # noqa: ARG001
    selected_verts: Array | None = None,  # noqa: ARG001
    selected_hes: Array | None = None,  # noqa: ARG001
    selected_faces: Array | None = None,  # noqa: ARG001
) -> tuple[Array, Array, Array, Array]:
    """Do nothing function for bounded meshes."""
    return vertTable, angTable, heTable, faceTable
