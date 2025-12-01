"""Optimization methods for VertAX (bounded version)."""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import grad, jacfwd, jit, lax

from vertax.topo import update_T1_bounded

# ==========
# Bounded
# ==========
###############################
## AUTOMATIC DIFFERENTIATION ##
###############################


@partial(jit, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
def minimize_bounded(
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
    iterations_max=1e3,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    argnums=0,
):
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
        raise ValueError("argnums must be in {0,2,3,4,5,6}")

    # --- check if the target is float-type before differentiating ---
    if not jnp.issubdtype(x0.dtype, jnp.floating):
        raise TypeError(
            f"Cannot differentiate argnums={argnums}: "
            f"target array has dtype {x0.dtype}. "
            f"Use a float array (vertTable, vert_params, he_params, face_params) only."
        )

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
    def scan_step(carry, i):
        vt, at, ht, ft, vp, hp, fp, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_list = carry

        is_running = jnp.logical_not(should_stop)

        # Compute loss
        L_current = L_in(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, vp, hp, fp)

        # Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)
        new_stagnation_count = jnp.where(rel_var < tolerance, stagnation_count + 1, 0)
        new_should_stop = (new_stagnation_count >= patience) | (i >= iterations_max - 1)  # type: ignore

        # Gradient wrt chosen argnums
        grads = grad(L_in, argnums=jacnums)(vt, at, ht, ft, selected_verts, selected_hes, selected_faces, vp, hp, fp)

        # Optimizer update
        updates, new_opt_state = solver.update(grads, opt_state)

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
        new_vt_T1, new_at_T1, new_ht_T1, new_ft_T1 = update_T1_bounded(
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
    iterations_max=1e3,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
):
    # Use the general minimize function with argnums=0 (optimize vertTable)
    (vt_f, at_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist_full, step_f_array) = minimize_bounded(
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
    )

    # Convert the JAX array step_f to a Python integer
    step_f = step_f_array.item()
    # Now slice using standard Python/NumPy slicing
    final_L_list = L_hist_full[:step_f]
    # Return updated arrays and loss history
    return (vt_f, at_f, ht_f, ft_f), final_L_list


def cost_ad_bounded(
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
    iterations_max=1e3,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    image_target=None,
):
    (vertTable, angTable, heTable, faceTable), L_in = inner_opt_bounded(
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
):
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


def loss_ep_static_bounded(
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
    selected_verts,
    selected_hes,
    selected_faces,
    image_target,
    beta,
):
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


@partial(jit, static_argnums=(7, 12, 13, 14, 16, 17, 18, 24))
def minimize_ep_bounded(
    vertTable,
    angTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    loss_fn,
    vertTable_target,
    angTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver,
    min_dist_T1,
    iterations_max=1e3,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    image_target=None,
    beta=0.001,
    argnums=0,
):
    def loss_evaluated(vt, at, ht, ft, vp, hp, fp):
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
        raise ValueError("argnums must be in {0,2,3,4,5,6}")

    # --- check if the target is float-type before differentiating ---
    if not jnp.issubdtype(x0.dtype, jnp.floating):
        raise TypeError(
            f"Cannot differentiate argnums={argnums}: "
            f"target array has dtype {x0.dtype}. "
            f"Use a float array (vertTable, vert_params, he_params, face_params) only."
        )

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
    def scan_step(carry, i):
        vt, at, ht, ft, vp, hp, fp, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_list = carry

        is_running = jnp.logical_not(should_stop)

        # Compute loss
        L_current = loss_evaluated(vt, at, ht, ft, vp, hp, fp)

        # Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)
        new_stagnation_count = jnp.where(rel_var < tolerance, stagnation_count + 1, 0)
        new_should_stop = (new_stagnation_count >= patience) | (i >= iterations_max - 1)  # type: ignore

        # Gradient wrt chosen argnums
        grads = grad(loss_evaluated, argnums=jacnums)(vt, at, ht, ft, vp, hp, fp)

        # Optimizer update
        updates, new_opt_state = solver.update(grads, opt_state)

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
        new_vt_T1, new_at_T1, new_ht_T1, new_ft_T1 = update_T1_bounded(
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
):
    (vt_f, at_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist_full, step_f_array) = minimize_ep_bounded(
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
    )

    step_f = step_f_array.item()
    final_L_list = L_hist_full[:step_f]
    return (vt_f, at_f, ht_f, ft_f), final_L_list


def outer_eq_prop_bounded(
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
):
    (vertTable_free, angTable_free, heTable_free, faceTable_free), loss_free = inner_eq_prop_bounded(
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
        beta=0.0,
    )

    (vertTable_nudged, angTable_nudged, heTable_nudged, faceTable_nudged), loss_nudged = inner_eq_prop_bounded(
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
    )

    grad_loss_ep_free_verts = grad(loss_ep_static_bounded, argnums=4)(
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
        beta=0.0,
    )

    grad_loss_ep_nudged_verts = grad(loss_ep_static_bounded, argnums=4)(
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

    grad_loss_ep_free_hes = grad(loss_ep_static_bounded, argnums=5)(
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
        beta=0.0,
    )

    grad_loss_ep_nudged_hes = grad(loss_ep_static_bounded, argnums=5)(
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

    grad_loss_ep_free_faces = grad(loss_ep_static_bounded, argnums=6)(
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
        beta=0.0,
    )

    grad_loss_ep_nudged_faces = grad(loss_ep_static_bounded, argnums=6)(
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
):
    def L_in_flatten(vertangTable_flatten, heTable, faceTable, vert_params, he_params, face_params):
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

    (vertTable_eq, angTable_eq, heTable_eq, faceTable_eq), L_in_value = inner_opt_bounded(
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
    )

    H = jacfwd(grad(L_in_flatten, argnums=0), argnums=0)(
        jnp.concatenate((vertTable_eq.flatten(), angTable_eq)),
        heTable_eq,
        faceTable_eq,
        vert_params,
        he_params,
        face_params,
    )

    crossderivative_verts = jacfwd(jacfwd(L_in_flatten, argnums=0), argnums=3)(
        jnp.concatenate((vertTable_eq.flatten(), angTable_eq)),
        heTable_eq,
        faceTable_eq,
        vert_params,
        he_params,
        face_params,
    )
    crossderivative_hes = jacfwd(jacfwd(L_in_flatten, argnums=0), argnums=4)(
        jnp.concatenate((vertTable_eq.flatten(), angTable_eq)),
        heTable_eq,
        faceTable_eq,
        vert_params,
        he_params,
        face_params,
    )
    crossderivative_faces = jacfwd(jacfwd(L_in_flatten, argnums=0), argnums=5)(
        jnp.concatenate((vertTable_eq.flatten(), angTable_eq)),
        heTable_eq,
        faceTable_eq,
        vert_params,
        he_params,
        face_params,
    )

    L_in_derivative_verts = -jax.numpy.linalg.solve(H, crossderivative_verts)
    L_in_derivative_hes = -jax.numpy.linalg.solve(H, crossderivative_hes)
    L_in_derivative_faces = -jax.numpy.linalg.solve(H, crossderivative_faces)

    grad_verts = L_in_derivative_verts.T @ jnp.concatenate(
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
                image_target,
            ),
        ]
    )
    grad_hes = L_in_derivative_hes.T @ jnp.concatenate(
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
                image_target,
            ),
        ]
    )
    grad_faces = L_in_derivative_faces.T @ jnp.concatenate(
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
                image_target,
            ),
        ]
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


#############
## WRAPPER ##
#############


def bilevel_opt_bounded(
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
    image_target=None,
    beta=None,
    method="ad",
):
    if method == "ad":
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
    elif method == "ep":
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
        )
    elif method == "id":
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
        )
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
    )

    return (vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params), cost
