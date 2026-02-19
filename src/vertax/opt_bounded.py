"""Optimization methods for VertAX (bounded version)."""

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
from jax import Array, grad, jacfwd, jit, jvp, lax
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator, minres

from vertax.method_enum import BilevelOptimizationMethod
from vertax.topo import update_T1_bounded

InnerLossFunctionBounded = Callable[
    [Array, Array, Array, Array, Array | None, Array | None, Array | None, Array, Array, Array], Array
]
OuterLossFunction = Callable[
    [
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
    ],
    Array,
]

UpdateT1FuncBounded = Callable[
    [Array, Array, Array, Array, Array, Array, Array, InnerLossFunctionBounded, float, Array, Array, Array],
    tuple[Array, Array, Array, Array],
]

LossEPFunction = Callable[
    [
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        InnerLossFunctionBounded,
        OuterLossFunction,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        float,
    ],
    Array,
]
###############################
## AUTOMATIC DIFFERENTIATION ##
###############################


@partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
def _minimize_bounded(  # noqa: C901
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: InnerLossFunctionBounded,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    argnums: int = 0,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], tuple[Array, Array]]:
    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])
    if selected_hes is None:
        selected_hes = jnp.arange(heTable.shape[0])
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    iterations_max = int(iterations_max)
    patience = int(patience)

    # --- select the target array and indices for the optimizer ---
    if argnums == 0:
        x0, sel = vertTable, selected_verts
    elif argnums == 2:
        x0, sel = heTable, selected_hes
    elif argnums == 3:
        x0, sel = faceTable, selected_faces
    elif argnums == 4:
        x0, sel = vert_params, selected_verts
    elif argnums == 5:
        x0, sel = he_params, selected_hes
    elif argnums == 6:
        x0, sel = face_params, selected_faces
    else:
        msg = "argnums must be in {0,2,3,4,5,6}"
        raise ValueError(msg)

    # --- check if the target is float-type before differentiating ---
    if not jnp.issubdtype(x0.dtype, jnp.floating):
        msg = (
            f"Cannot differentiate argnums={argnums}: "
            f"target array has dtype {x0.dtype}. "
            f"Use a float array (vertTable, vert_params, he_params, face_params) only."
        )
        raise TypeError(msg)

    jacnums = argnums
    if argnums == 0:
        opt_state = solver.init((vertTable, angTable))
        jacnums = [0, 1]
    else:
        opt_state = solver.init(x0)

    # Initial loss and bookkeeping
    initial_L = L_in(
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
    )
    L_in_list = jnp.zeros((iterations_max,)).at[0].set(initial_L)
    prev_L_values = jnp.full((patience,), initial_L)
    stagnation_count = jnp.array(0)
    step_count = jnp.array(0)
    should_stop = jnp.array(False)

    @jit
    def scan_step(
        carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], i: int
    ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], None]:
        vt, at, ht, ft, vp, hp, fp, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_list = carry

        is_running = jnp.logical_not(should_stop)

        # Compute loss
        L_current = L_in(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, vp, hp, fp)

        # Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)
        new_stagnation_count = jnp.where(rel_var < tolerance, stagnation_count + 1, 0)
        new_should_stop = (new_stagnation_count >= patience) | (i >= iterations_max - 1)

        # Gradient wrt chosen argnums
        grads = grad(L_in, argnums=jacnums)(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, vp, hp, fp)

        # Optimizer update
        updates: Array
        new_opt_state: Array
        updates, new_opt_state = solver.update(grads, opt_state)  # ty:ignore[invalid-assignment]

        # Apply updates to the chosen array on selected indices
        if argnums == 0:
            new_vt = vt.at[sel].set(vt[sel] + updates[0].at[sel].get())
            new_at = at + updates[1]
            vt = lax.cond(is_running, lambda: new_vt, lambda: vt)
            at = lax.cond(is_running, lambda: new_at, lambda: at)
        elif argnums == 1:
            new_ht = ht.at[sel].set(ht[sel] + updates.at[sel].get())
            ht = lax.cond(is_running, lambda: new_ht, lambda: ht)
        elif argnums == 2:
            new_ft = ft.at[sel].set(ft[sel] + updates.at[sel].get())
            ft = lax.cond(is_running, lambda: new_ft, lambda: ft)
        elif argnums == 3:
            new_vp = vp.at[sel].set(vp[sel] + updates.at[sel].get())
            vp = lax.cond(is_running, lambda: new_vp, lambda: vp)
        elif argnums == 4:
            new_hp = hp.at[sel].set(hp[sel] + updates.at[sel].get())
            hp = lax.cond(is_running, lambda: new_hp, lambda: hp)
        elif argnums == 5:
            new_fp = fp.at[sel].set(fp[sel] + updates.at[sel].get())
            fp = lax.cond(is_running, lambda: new_fp, lambda: fp)

        opt_state = lax.cond(is_running, lambda: new_opt_state, lambda: opt_state)
        stagnation_count = lax.cond(is_running, lambda: new_stagnation_count, lambda: stagnation_count)
        should_stop = new_should_stop

        # Geometry updates
        new_vt_T1, new_at_T1, new_ht_T1, new_ft_T1 = update_T1_func(
            vt, at, ht, ft, vp, hp, fp, L_in, min_dist_T1, selected_verts, selected_hes, selected_faces
        )
        vt = lax.cond(is_running, lambda: new_vt_T1, lambda: vt)
        at = lax.cond(is_running, lambda: new_at_T1, lambda: at)
        ht = lax.cond(is_running, lambda: new_ht_T1, lambda: ht)
        ft = lax.cond(is_running, lambda: new_ft_T1, lambda: ft)

        # Shift prev_L_values
        new_prev_L_values = prev_L_values.at[1:].set(prev_L_values[:-1])
        new_prev_L_values = new_prev_L_values.at[0].set(L_current)
        prev_L_values = lax.cond(is_running, lambda: new_prev_L_values, lambda: prev_L_values)

        L_list = L_list.at[i].set(L_current)
        step_count = i + 1

        return (
            vt,
            at,
            ht,
            ft,
            vp,
            hp,
            fp,
            opt_state,
            prev_L_values,
            stagnation_count,
            step_count,
            should_stop,
            L_list,
        ), None

    init_carry = (
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        opt_state,
        prev_L_values,
        stagnation_count,
        step_count,
        should_stop,
        L_in_list,
    )

    final_state, _ = lax.scan(scan_step, init_carry, xs=jnp.arange(iterations_max))

    # Unpack and return results (for slicing outside JIT)
    vt_f, at_f, ht_f, ft_f, vp_f, hp_f, fp_f, _, _, _, step_f, _, L_hist = final_state

    return (vt_f, at_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist, step_f)


def inner_opt_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: InnerLossFunctionBounded,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[tuple[Array, Array, Array, Array], Array]:
    """Inner optimization for a bounded mesh."""
    # Use the general minimize function with argnums=0 (optimize vertTable)
    (vt_f, at_f, ht_f, ft_f, _vp_f, _hp_f, _fp_f), (L_hist_full, step_f_array) = _minimize_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver,
        min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        selected_verts=selected_verts,
        selected_hes=selected_hes,
        selected_faces=selected_faces,
        argnums=0,
        update_T1_func=update_T1_func,
    )

    # Convert the JAX array step_f to a Python integer
    step_f = step_f_array.item()
    # Now slice using standard Python/NumPy slicing
    final_L_list = L_hist_full[:step_f]
    # Return updated arrays and loss history
    return (vt_f, at_f, ht_f, ft_f), final_L_list


def cost_ad_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    image_target: Array | None = None,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> Array:
    """Automatic differentiation cost function."""
    (vertTable, angTable, heTable, faceTable), _ = inner_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        update_T1_func,
    )

    loss_out_value = L_out(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    return loss_out_value


def outer_opt_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    image_target: Array | None = None,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[Array, Array, Array]:
    """Outer optimization for a bounded mesh."""
    grad_verts = grad(cost_ad_bounded, argnums=4)(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        update_T1_func,
    )

    grad_hes = grad(cost_ad_bounded, argnums=5)(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        update_T1_func,
    )

    grad_faces = grad(cost_ad_bounded, argnums=6)(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        update_T1_func,
    )

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vert_params, he_params, face_params


#############################
## EQUILIBTIUM PROPAGATION ##
#############################


def _loss_ep_static_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
) -> Array:
    loss_inner_value = L_in(
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
    )

    loss_outer_value = L_out(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    return loss_inner_value + (beta * loss_outer_value)


@partial(jit, static_argnums=(7, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25))
def _minimize_ep_bounded(  # noqa: C901
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    loss_fn: LossEPFunction,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    image_target: Array | None = None,
    beta: float = 0.001,
    argnums: int = 0,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], tuple[Array, Array]]:
    def loss_evaluated(vt: Array, at: Array, ht: Array, ft: Array, vp: Array, hp: Array, fp: Array) -> Array:
        return loss_fn(
            vt,
            at,
            ht,
            ft,
            vp,
            hp,
            fp,
            vertTable_target,
            angTable_target,
            heTable_target,
            faceTable_target,
            L_in,
            L_out,
            selected_verts,
            selected_hes,
            selected_faces,
            image_target,
            beta,
        )

    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])
    if selected_hes is None:
        selected_hes = jnp.arange(heTable.shape[0])
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    iterations_max = int(iterations_max)
    patience = int(patience)

    # --- select the target array and indices for the optimizer ---
    if argnums == 0:
        x0, sel = vertTable, selected_verts
    elif argnums == 2:
        x0, sel = heTable, selected_hes
    elif argnums == 3:
        x0, sel = faceTable, selected_faces
    elif argnums == 4:
        x0, sel = vert_params, selected_verts
    elif argnums == 5:
        x0, sel = he_params, selected_hes
    elif argnums == 6:
        x0, sel = face_params, selected_faces
    else:
        msg = "argnums must be in {0,2,3,4,5,6}"
        raise ValueError(msg)

    # --- check if the target is float-type before differentiating ---
    if not jnp.issubdtype(x0.dtype, jnp.floating):
        msg = (
            f"Cannot differentiate argnums={argnums}: "
            f"target array has dtype {x0.dtype}. "
            f"Use a float array (vertTable, vert_params, he_params, face_params) only."
        )
        raise TypeError(msg)

    jacnums = argnums
    if argnums == 0:
        opt_state = solver.init((vertTable, angTable))
        jacnums = [0, 1]
    else:
        opt_state = solver.init(x0)

    # Initial loss and bookkeeping
    initial_L = loss_evaluated(vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params)
    L_in_list = jnp.zeros((iterations_max,)).at[0].set(initial_L)
    prev_L_values = jnp.full((patience,), initial_L)
    stagnation_count = jnp.array(0)
    step_count = jnp.array(0)
    should_stop = jnp.array(False)

    @jit
    def scan_step(
        carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], i: int
    ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], None]:
        vt, at, ht, ft, vp, hp, fp, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_list = carry

        is_running = jnp.logical_not(should_stop)

        # Compute loss
        L_current = loss_evaluated(vt, at, ht, ft, vp, hp, fp)

        # Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)
        new_stagnation_count = jnp.where(rel_var < tolerance, stagnation_count + 1, 0)
        new_should_stop = (new_stagnation_count >= patience) | (i >= iterations_max - 1)

        # Gradient wrt chosen argnums
        grads = grad(loss_evaluated, argnums=jacnums)(vt, at, ht, ft, vp, hp, fp)

        # Optimizer update
        updates: Array
        new_opt_state: Array
        updates, new_opt_state = solver.update(grads, opt_state)  # ty:ignore[invalid-assignment]

        # Apply updates to the chosen array on selected indices
        if argnums == 0:
            new_vt = vt.at[sel].set(vt[sel] + updates[0].at[sel].get())
            new_at = at + updates[1]
            vt = lax.cond(is_running, lambda: new_vt, lambda: vt)
            at = lax.cond(is_running, lambda: new_at, lambda: at)
        elif argnums == 1:
            new_ht = ht.at[sel].set(ht[sel] + updates.at[sel].get())
            ht = lax.cond(is_running, lambda: new_ht, lambda: ht)
        elif argnums == 2:
            new_ft = ft.at[sel].set(ft[sel] + updates.at[sel].get())
            ft = lax.cond(is_running, lambda: new_ft, lambda: ft)
        elif argnums == 3:
            new_vp = vp.at[sel].set(vp[sel] + updates.at[sel].get())
            vp = lax.cond(is_running, lambda: new_vp, lambda: vp)
        elif argnums == 4:
            new_hp = hp.at[sel].set(hp[sel] + updates.at[sel].get())
            hp = lax.cond(is_running, lambda: new_hp, lambda: hp)
        elif argnums == 5:
            new_fp = fp.at[sel].set(fp[sel] + updates.at[sel].get())
            fp = lax.cond(is_running, lambda: new_fp, lambda: fp)

        opt_state = lax.cond(is_running, lambda: new_opt_state, lambda: opt_state)
        stagnation_count = lax.cond(is_running, lambda: new_stagnation_count, lambda: stagnation_count)
        should_stop = new_should_stop

        # Geometry updates
        new_vt_T1, new_at_T1, new_ht_T1, new_ft_T1 = update_T1_func(
            vt, at, ht, ft, vp, hp, fp, L_in, min_dist_T1, selected_verts, selected_hes, selected_faces
        )
        vt = lax.cond(is_running, lambda: new_vt_T1, lambda: vt)
        at = lax.cond(is_running, lambda: new_at_T1, lambda: at)
        ht = lax.cond(is_running, lambda: new_ht_T1, lambda: ht)
        ft = lax.cond(is_running, lambda: new_ft_T1, lambda: ft)

        # Shift prev_L_values
        new_prev_L_values = prev_L_values.at[1:].set(prev_L_values[:-1])
        new_prev_L_values = new_prev_L_values.at[0].set(L_current)
        prev_L_values = lax.cond(is_running, lambda: new_prev_L_values, lambda: prev_L_values)

        L_list = L_list.at[i].set(L_current)
        step_count = i + 1

        return (
            vt,
            at,
            ht,
            ft,
            vp,
            hp,
            fp,
            opt_state,
            prev_L_values,
            stagnation_count,
            step_count,
            should_stop,
            L_list,
        ), None

    init_carry = (
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        opt_state,
        prev_L_values,
        stagnation_count,
        step_count,
        should_stop,
        L_in_list,
    )

    final_state, _ = lax.scan(scan_step, init_carry, xs=jnp.arange(iterations_max))

    # Unpack and return results (for slicing outside JIT)
    vt_f, at_f, ht_f, ft_f, vp_f, hp_f, fp_f, _, _, _, step_f, _, L_hist = final_state

    return (vt_f, at_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist, step_f)


def inner_eq_prop_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    loss_ep_static_bounded: LossEPFunction,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[tuple[Array, Array, Array, Array], Array]:
    """Inner optimization function for equilibrium propagation only and bounded meshes."""
    (vt_f, at_f, ht_f, ft_f, _vp_f, _hp_f, _fp_f), (L_hist_full, step_f_array) = _minimize_ep_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        loss_ep_static_bounded,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
        argnums=0,
        update_T1_func=update_T1_func,
    )

    step_f = step_f_array.item()
    final_L_list = L_hist_full[:step_f]
    return (vt_f, at_f, ht_f, ft_f), final_L_list


def outer_eq_prop_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[Array, Array, Array]:
    """Outer optimization for equilibrium propagation."""
    (vertTable_free, angTable_free, heTable_free, faceTable_free), _loss_free = inner_eq_prop_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        _loss_ep_static_bounded,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=-beta,
        update_T1_func=update_T1_func,
    )

    (vertTable_nudged, angTable_nudged, heTable_nudged, faceTable_nudged), _loss_nudged = inner_eq_prop_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        _loss_ep_static_bounded,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
        update_T1_func,
    )

    grad_loss_ep_free_verts = grad(_loss_ep_static_bounded, argnums=4)(
        vertTable_free,
        angTable_free,
        heTable_free,
        faceTable_free,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=-beta,
    )

    grad_loss_ep_nudged_verts = grad(_loss_ep_static_bounded, argnums=4)(
        vertTable_nudged,
        angTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
    )

    grad_loss_ep_free_hes = grad(_loss_ep_static_bounded, argnums=5)(
        vertTable_free,
        angTable_free,
        heTable_free,
        faceTable_free,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=-beta,
    )

    grad_loss_ep_nudged_hes = grad(_loss_ep_static_bounded, argnums=5)(
        vertTable_nudged,
        angTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
    )

    grad_loss_ep_free_faces = grad(_loss_ep_static_bounded, argnums=6)(
        vertTable_free,
        angTable_free,
        heTable_free,
        faceTable_free,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta=-beta,
    )

    grad_loss_ep_nudged_faces = grad(_loss_ep_static_bounded, argnums=6)(
        vertTable_nudged,
        angTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
        angTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        beta,
    )

    grad_verts = (1.0 / (2 * beta)) * ((grad_loss_ep_nudged_verts) - (grad_loss_ep_free_verts))
    grad_hes = (1.0 / (2 * beta)) * ((grad_loss_ep_nudged_hes) - (grad_loss_ep_free_hes))
    grad_faces = (1.0 / (2 * beta)) * ((grad_loss_ep_nudged_faces) - (grad_loss_ep_free_faces))

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vert_params, he_params, face_params


###########################
## IMPLICIT DIFFERENTION ##
###########################


def outer_implicit_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[Array, Array, Array]:
    """Outer optimization for implicit differentiation method."""

    def L_in_flatten(  # noqa: N802
        vertangTable_flatten: Array,
        heTable: Array,
        faceTable: Array,
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        vertTable_tmp = vertangTable_flatten[: -heTable.shape[0] // 2].reshape(
            (len(vertangTable_flatten) - heTable.shape[0] // 2) // 2, 2
        )
        angTable_tmp = vertangTable_flatten[-heTable.shape[0] // 2 :]
        return L_in(
            vertTable_tmp,
            angTable_tmp,
            heTable,
            faceTable,
            selected_verts,
            selected_hes,
            selected_faces,
            vert_params,
            he_params,
            face_params,
        )

    (vertTable_eq, angTable_eq, heTable_eq, faceTable_eq), _L_in_value = inner_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        update_T1_func,
    )

    ################################################################
    ################            Start mod           ################
    ################################################################

    vertang_flat_eq = jnp.concatenate((vertTable_eq.flatten(), angTable_eq))

    # H = d^2 L_in / d vertangTable^2
    G_op = grad(L_in_flatten, argnums=0)

    dtype = vertang_flat_eq.dtype

    def matvec(v: ArrayLike) -> Array:
        v_casted = jnp.asarray(v, dtype=dtype)
        _, Hv = jvp(
            lambda vfe: G_op(vfe, heTable_eq, faceTable_eq, vert_params, he_params, face_params),
            (vertang_flat_eq,),
            (v_casted,),
        )
        return Hv

    nv = len(vertang_flat_eq)
    H_np = LinearOperator((nv, nv), matvec=matvec)  # type: ignore

    # Compute cross-derivatives (dL_in / d vertangTable d param)
    crossderivative_verts_op = jacfwd(G_op, argnums=3)
    crossderivative_hes_op = jacfwd(G_op, argnums=4)
    crossderivative_faces_op = jacfwd(G_op, argnums=5)

    crossderivative_verts = crossderivative_verts_op(
        vertang_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_hes = crossderivative_hes_op(
        vertang_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_faces = crossderivative_faces_op(
        vertang_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    # Convert JAX arrays to NumPy arrays for use with scipy.sparse.linalg.minres
    # H_np = np.asarray(H) # Replaced by linear operator
    cd_verts_np = np.asarray(crossderivative_verts)
    cd_hes_np = np.asarray(crossderivative_hes)
    cd_faces_np = np.asarray(crossderivative_faces)

    # MINRES solves H * X = B. We want X = -H^{-1} * B, so we solve H * X = -B

    b_verts = -cd_verts_np
    b_verts_np = np.asarray(b_verts)
    N_rhs = b_verts.shape[1]

    Lambda_solutions = []
    # Loop over the columns (right-hand sides)
    for i in range(N_rhs):
        b_vector = b_verts_np[:, i]
        # Solve H_np * Lambda_i = b_vector
        Lambda_i_np, _ = minres(H_np, b_vector)
        Lambda_solutions.append(Lambda_i_np)
    L_in_derivative_verts_np = jnp.stack(Lambda_solutions, axis=1)

    b_hes = -cd_hes_np
    b_hes_np = np.asarray(b_hes)
    N_rhs = b_hes.shape[1]

    Lambda_solutions = []
    # Loop over the columns (right-hand sides)
    for i in range(N_rhs):
        b_vector = b_hes_np[:, i]
        # Solve H_np * Lambda_i = b_vector
        Lambda_i_np, _ = minres(H_np, b_vector)
        Lambda_solutions.append(Lambda_i_np)
    L_in_derivative_hes_np = jnp.stack(Lambda_solutions, axis=1)

    b_faces = -cd_faces_np
    b_faces_np = np.asarray(b_faces)
    N_rhs = b_faces_np.shape[1]

    Lambda_solutions = []
    # Loop over the columns (right-hand sides)
    for i in range(N_rhs):
        b_vector = b_faces_np[:, i]
        # Solve H_np * Lambda_i = b_vector
        Lambda_i_np, _ = minres(H_np, b_vector)
        Lambda_solutions.append(Lambda_i_np)
    L_in_derivative_faces_np = jnp.stack(Lambda_solutions, axis=1)

    # Convert solutions back to JAX arrays for the final gradient calculation
    L_in_derivative_verts = jnp.asarray(L_in_derivative_verts_np)
    L_in_derivative_hes = jnp.asarray(L_in_derivative_hes_np)
    L_in_derivative_faces = jnp.asarray(L_in_derivative_faces_np)

    # The term (dL_out / d vertangTable) evaluated at the equilibrium
    dL_out_dvertang = jnp.concatenate(
        [
            grad(L_out, argnums=0)(
                vertTable_eq,
                angTable_eq,
                heTable_eq,
                faceTable_eq,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            )[:, :2].flatten(),
            grad(L_out, argnums=1)(
                vertTable_eq,
                angTable_eq,
                heTable_eq,
                faceTable_eq,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            ),
        ]
    )

    # dL_out / d param = (dL_in_d param)^T @ (dL_out / d vertangTable)
    grad_verts = L_in_derivative_verts.T @ dL_out_dvertang
    grad_hes = L_in_derivative_hes.T @ dL_out_dvertang
    grad_faces = L_in_derivative_faces.T @ dL_out_dvertang

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vert_params, he_params, face_params


##########################
## ADJOINT STATE METHOD ##
##########################


def outer_adjoint_state_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    angTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[Array, Array, Array]:
    """Outer optimization for adjoint state method."""

    def L_in_flatten(  # noqa: N802
        vertangTable_flatten: Array,
        heTable: Array,
        faceTable: Array,
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        vertTable_tmp = vertangTable_flatten[: -heTable.shape[0] // 2].reshape(
            (len(vertangTable_flatten) - heTable.shape[0] // 2) // 2, 2
        )
        angTable_tmp = vertangTable_flatten[-heTable.shape[0] // 2 :]
        return L_in(
            vertTable_tmp,
            angTable_tmp,
            heTable,
            faceTable,
            selected_verts,
            selected_hes,
            selected_faces,
            vert_params,
            he_params,
            face_params,
        )

    (vertTable_eq, angTable_eq, heTable_eq, faceTable_eq), _L_in_value = inner_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        update_T1_func,
    )

    vertTable_eq_flat = jnp.concatenate((vertTable_eq.flatten(), angTable_eq))

    G_op = grad(L_in_flatten, argnums=0)

    dtype = vertTable_eq_flat.dtype

    def matvec(v: ArrayLike) -> Array:
        v_casted = jnp.asarray(v, dtype=dtype)
        _, Hv = jvp(
            lambda vfe: G_op(vfe, heTable_eq, faceTable_eq, vert_params, he_params, face_params),
            (vertTable_eq_flat,),
            (v_casted,),
        )
        return Hv

    nv = len(vertTable_eq_flat)
    H_np = LinearOperator((nv, nv), matvec=matvec)  # type: ignore

    gradout = jnp.concatenate(
        [
            grad(L_out, argnums=0)(
                vertTable_eq,
                angTable_eq,
                heTable_eq,
                faceTable_eq,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            )[:, :2].flatten(),
            grad(L_out, argnums=1)(
                vertTable_eq,
                angTable_eq,
                heTable_eq,
                faceTable_eq,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            ),
        ]
    )

    # Convert JAX arrays to NumPy arrays for SciPy function call
    gradout_np = np.asarray(gradout)

    # Call SciPy MINRES. H is symmetric and possibly indefinite, so MINRES is suitable.
    Lambda_np, _ = minres(H_np, gradout_np)

    # Convert the result back to a JAX array
    Lambda = jnp.asarray(Lambda_np)

    crossderivative_verts = jacfwd(G_op, argnums=3)(
        vertTable_eq_flat, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_hes = jacfwd(G_op, argnums=4)(
        vertTable_eq_flat, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_faces = jacfwd(G_op, argnums=5)(
        vertTable_eq_flat, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    grad_verts = -Lambda @ crossderivative_verts
    grad_hes = -Lambda @ crossderivative_hes
    grad_faces = -Lambda @ crossderivative_faces

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}

    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    vert_params = updated_params["vert_params"]  # type: ignore
    he_params = updated_params["he_params"]  # type: ignore
    face_params = updated_params["face_params"]  # type: ignore

    return vert_params, he_params, face_params


#############
## WRAPPER ##
#############


def bilevel_opt_bounded(
    vertTable: Array,
    angTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array | None,
    angTable_target: Array | None,
    heTable_target: Array | None,
    faceTable_target: Array | None,
    L_in: InnerLossFunctionBounded,
    L_out: OuterLossFunction,
    solver_inner: optax.GradientTransformation,
    solver_outer: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int,
    tolerance: float,
    patience: int,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float = 0.001,
    optimization_method: BilevelOptimizationMethod = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION,
    update_T1_func: UpdateT1FuncBounded = update_T1_bounded,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array], Array]:
    """Bilevel optimization for bounded meshes."""
    match optimization_method:
        case BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION:
            vert_params, he_params, face_params = outer_opt_bounded(
                vertTable,
                angTable,
                heTable,
                faceTable,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                L_in,
                L_out,
                solver_inner,
                solver_outer,
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
            )
        case BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION:
            vert_params, he_params, face_params = outer_eq_prop_bounded(
                vertTable,
                angTable,
                heTable,
                faceTable,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                L_in,
                L_out,
                solver_inner,
                solver_outer,
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
                beta,
                update_T1_func,
            )
        case BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION:
            vert_params, he_params, face_params = outer_implicit_bounded(
                vertTable,
                angTable,
                heTable,
                faceTable,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
                angTable_target,
                heTable_target,
                faceTable_target,
                L_in,
                L_out,
                solver_inner,
                solver_outer,
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                selected_verts,
                selected_hes,
                selected_faces,
                image_target,
                update_T1_func,
            )
        case _:
            msg = f"{optimization_method} is not implemented for bounded meshes."
            raise ValueError(msg)
    (vertTable, angTable, heTable, faceTable), cost = inner_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        L_in,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        selected_verts,
        selected_hes,
        selected_faces,
        update_T1_func,
    )

    return (vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params), cost
