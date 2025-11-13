"""Optimization methods for VertAX."""

from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import Array, grad, jacfwd, jit, lax

from vertax.geo import update_pbc
from vertax.topo import update_T1


class OptimizationTarget(Enum):
    """What the minimize function will try to optimize."""

    VERTICES = 0
    EDGES = 1
    FACES = 2
    # width = 3
    # height = 4
    VERTEX_PARAMETERS = 5
    EDGE_PARAMETERS = 6
    FACE_PARAMETERS = 7


###############################
## AUTOMATIC DIFFERENTIATION ##
###############################


@partial(jit, static_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17))
def _jit_minimize(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    selected_verts: Array,
    selected_hes: Array,
    selected_faces: Array,
    width: float,
    height: float,
    L_in,
    solver,
    min_dist_T1,
    iterations_max: int = 1000,
    tolerance=1e-4,
    patience=5,
    optimization_target: OptimizationTarget = OptimizationTarget.VERTICES,
):
    x0: Array = [vertTable, heTable, faceTable, None, None, vert_params, he_params, face_params][
        optimization_target.value
    ]  # type: ignore
    sel: Array = [
        selected_verts,
        selected_hes,
        selected_faces,
        None,
        None,
        selected_verts,
        selected_hes,
        selected_faces,
    ][optimization_target.value]  # type: ignore
    opt_state = solver.init(x0)

    # Initial loss and bookkeeping
    initial_L = L_in(
        vertTable[selected_verts],
        heTable[selected_hes],
        faceTable[selected_faces],
        width,
        height,
        vert_params[selected_verts],
        he_params[selected_hes],
        face_params[selected_faces],
    )
    L_in_list = jnp.zeros((iterations_max,)).at[0].set(initial_L)
    prev_L_values = jnp.full((patience,), initial_L)
    stagnation_count = jnp.array(0)
    step_count = jnp.array(0)
    should_stop = jnp.array(False)

    # 2. Define the inner step function for lax.scan
    # The inner function is defined *without* @jit because it's compiled by lax.scan/jit on minimize
    def scan_step(carry, i):
        # Unpack the carry structure:
        (
            vt,
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
        ) = carry

        # Determine if we are still running (i.e., not stopped)
        is_running = jnp.logical_not(should_stop)
        # 1) Compute loss
        L_current = L_in(
            vt[selected_verts],
            ht[selected_hes],
            ft[selected_faces],
            width,
            height,
            vp[selected_verts],
            hp[selected_hes],
            fp[selected_faces],
        )

        # 2) Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)

        new_stagnation_count = jnp.where(rel_var < tolerance, stagnation_count + 1, 0)
        new_should_stop = (new_stagnation_count >= patience) | (i >= iterations_max - 1)  # type: ignore

        # 3) Gradient wrt chosen argnums
        grads = grad(L_in, argnums=optimization_target.value)(vt, ht, ft, width, height, vp, hp, fp)

        # 4) Optimizer update
        updates, new_opt_state = solver.update(grads, opt_state)

        # 5) Apply updates to the chosen array on selected indices
        updates_sel = updates.at[sel].get()
        arrays = [vt, ht, ft, None, None, vp, hp, fp]
        k = optimization_target.value
        # --- Conditional Application of Optimization Update and State ---

        new_optim = arrays[k].at[sel].set(arrays[k][sel] + updates_sel)  # type: ignore
        arrays[k] = lax.cond(is_running, lambda: new_optim, lambda: arrays[k])  # type: ignore

        vt, ht, ft, _, _, vp, hp, fp = arrays
        # new_vt_optim = vt.at[sel].set(vt[sel] + updates_sel)
        # vt = lax.cond(is_running and should_update_vt, lambda: new_vt_optim, lambda: vt)

        opt_state = lax.cond(is_running, lambda: new_opt_state, lambda: opt_state)
        stagnation_count = lax.cond(is_running, lambda: new_stagnation_count, lambda: stagnation_count)
        should_stop = new_should_stop  # should_stop is a flag for the *next* iteration

        # 6) Geometry updates (Must also be conditional/masked)
        new_vt_pbc, new_ht_pbc, new_ft_pbc = update_pbc(vt, ht, ft, width, height)
        vt = lax.cond(is_running, lambda: new_vt_pbc, lambda: vt)
        ht = lax.cond(is_running, lambda: new_ht_pbc, lambda: ht)
        ft = lax.cond(is_running, lambda: new_ft_pbc, lambda: ft)

        new_vt_T1, new_ht_T1, new_ft_T1 = update_T1(vt, ht, ft, width, height, vp, hp, fp, L_in, min_dist_T1)
        vt = lax.cond(is_running, lambda: new_vt_T1, lambda: vt)
        ht = lax.cond(is_running, lambda: new_ht_T1, lambda: ht)
        ft = lax.cond(is_running, lambda: new_ft_T1, lambda: ft)

        # 7) Shift prev_L_values
        new_prev_L_values = prev_L_values.at[1:].set(prev_L_values[:-1])
        new_prev_L_values = new_prev_L_values.at[0].set(L_current)
        prev_L_values = lax.cond(is_running, lambda: new_prev_L_values, lambda: prev_L_values)

        # 8) Update History and Step Count
        L_list = L_list.at[i].set(L_current)
        step_count = i + 1

        new_carry = (
            vt,
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
        )
        return new_carry, None

    # 3. Call lax.scan

    init_carry = (
        vertTable,
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

    # 4. Unpack and return results (for slicing outside JIT)
    vt_f, ht_f, ft_f, vp_f, hp_f, fp_f, _, _, _, step_f, _, L_hist = final_state
    return (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist, step_f)


def minimize(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    L_in,
    solver,
    min_dist_T1,
    iterations_max=1000,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    optimization_target: OptimizationTarget = OptimizationTarget.VERTICES,
):
    # ensure width and height are hashable...
    width = float(width)
    height = float(height)
    iterations_max = int(iterations_max)
    patience = int(patience)
    min_dist_T1 = float(min_dist_T1)
    tolerance = float(tolerance)

    selected_verts = jnp.arange(vertTable.shape[0]) if selected_verts is None else jnp.array(selected_verts)
    selected_hes = jnp.arange(heTable.shape[0]) if selected_hes is None else jnp.array(selected_hes)
    selected_faces = jnp.arange(faceTable.shape[0]) if selected_faces is None else jnp.array(selected_faces)

    iterations_max = int(iterations_max)
    patience = int(patience)
    return _jit_minimize(
        vertTable=vertTable,
        heTable=heTable,
        faceTable=faceTable,
        vert_params=vert_params,
        he_params=he_params,
        face_params=face_params,
        selected_verts=selected_verts,
        selected_hes=selected_hes,
        selected_faces=selected_faces,
        width=width,
        height=height,
        L_in=L_in,
        solver=solver,
        min_dist_T1=min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        optimization_target=optimization_target,
    )


def inner_opt(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    L_in,
    solver,
    min_dist_T1,
    iterations_max=1000,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
):
    # Use the general minimize function with VERTICES (optimize vertTable)
    (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist_full, step_f_array) = minimize(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
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
        optimization_target=OptimizationTarget.VERTICES,
    )

    # Convert the JAX array step_f to a Python integer
    step_f = step_f_array.item()

    # Now slice using standard Python/NumPy slicing
    final_L_list = L_hist_full[:step_f]

    # Return updated arrays and loss history
    return (vt_f, ht_f, ft_f), final_L_list


def cost_ad(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
    heTable_target,
    faceTable_target,
    L_in,
    L_out,
    solver_inner,
    min_dist_T1,
    iterations_max=1000,
    tolerance=1e-4,
    patience=5,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    image_target=None,
):
    (vertTable, heTable, faceTable), L_in = inner_opt(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
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
        heTable,
        faceTable,
        width,
        height,
        vertTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    return loss_out_value


def outer_opt(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
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
    grad_verts = grad(cost_ad, argnums=5)(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_hes = grad(cost_ad, argnums=6)(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_faces = grad(cost_ad, argnums=7)(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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
    new_vert_params: Array = updated_params["vert_params"]
    new_he_params: Array = updated_params["he_params"]
    new_face_params: Array = updated_params["face_params"]

    return new_vert_params, new_he_params, new_face_params


#############################
## EQUILIBTIUM PROPAGATION ##
#############################


def loss_ep(
    vertTable,
    heTable,
    faceTable,
    width,
    height,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
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
    loss_inner_value = L_in(vertTable, heTable, faceTable, width, height, vert_params, he_params, face_params)

    loss_outer_value = L_out(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vertTable_target,
        heTable_target,
        faceTable_target,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
    )

    loss_ep_value = loss_inner_value + (beta * loss_outer_value)

    return loss_ep_value


def forward(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
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
    def loss_ep_forward(vt, ht, ft, width, height, vp, hp, fp):
        return loss_ep(
            vt,
            ht,
            ft,
            width,
            height,
            vp,
            hp,
            fp,
            vertTable_target,
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

    (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist_full, step_f_array) = minimize(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        L_in=loss_ep_forward,
        solver=solver_inner,
        min_dist_T1=min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        selected_verts=selected_verts,
        selected_hes=selected_hes,
        selected_faces=selected_faces,
        optimization_target=OptimizationTarget.VERTICES,
    )

    # Convert the JAX array step_f to a Python integer
    step_f = step_f_array.item()

    # Now slice using standard Python/NumPy slicing
    final_L_list = L_hist_full[:step_f]

    # Return updated arrays and loss history
    return (vt_f, ht_f, ft_f), final_L_list


def outer_eq_prop(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
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
    (vertTable_free, heTable_free, faceTable_free), _ = forward(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    (vertTable_nudged, heTable_nudged, faceTable_nudged), _ = forward(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_loss_ep_free_verts = grad(loss_ep, argnums=5)(
        vertTable_free,
        heTable_free,
        faceTable_free,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_loss_ep_nudged_verts = grad(loss_ep, argnums=5)(
        vertTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_loss_ep_free_hes = grad(loss_ep, argnums=6)(
        vertTable_free,
        heTable_free,
        faceTable_free,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_loss_ep_nudged_hes = grad(loss_ep, argnums=6)(
        vertTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_loss_ep_free_faces = grad(loss_ep, argnums=7)(
        vertTable_free,
        heTable_free,
        faceTable_free,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_loss_ep_nudged_faces = grad(loss_ep, argnums=7)(
        vertTable_nudged,
        heTable_nudged,
        faceTable_nudged,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        vertTable_target,
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

    grad_verts = (1.0 / beta) * ((grad_loss_ep_nudged_verts) - (grad_loss_ep_free_verts))
    grad_hes = (1.0 / beta) * ((grad_loss_ep_nudged_hes) - (grad_loss_ep_free_hes))
    grad_faces = (1.0 / beta) * ((grad_loss_ep_nudged_faces) - (grad_loss_ep_free_faces))

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    new_vert_params: Array = updated_params["vert_params"]
    new_he_params: Array = updated_params["he_params"]
    new_face_params: Array = updated_params["face_params"]

    return new_vert_params, new_he_params, new_face_params


###########################
## IMPLICIT DIFFERENTION ##
###########################


def outer_implicit(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
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
    def L_in_flatten(
        vertTable_flatten, heTable, faceTable, width: float, height: float, vert_params, he_params, face_params
    ):
        vertTable_tmp = jnp.hstack(
            (vertTable_flatten.reshape(len(vertTable_flatten) // 2, 2), jnp.zeros((len(vertTable_flatten) // 2, 1)))
        )
        return L_in(vertTable_tmp, heTable, faceTable, width, height, vert_params, he_params, face_params)

    (vertTable_eq, heTable_eq, faceTable_eq), _ = inner_opt(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
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
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, width, height, vert_params, he_params, face_params
    )

    crossderivative_verts = jacfwd(grad(L_in_flatten, argnums=0), argnums=5)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, width, height, vert_params, he_params, face_params
    )
    crossderivative_hes = jacfwd(grad(L_in_flatten, argnums=0), argnums=6)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, width, height, vert_params, he_params, face_params
    )
    crossderivative_faces = jacfwd(grad(L_in_flatten, argnums=0), argnums=7)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, width, height, vert_params, he_params, face_params
    )

    L_in_derivative_verts = -jax.numpy.linalg.solve(H, crossderivative_verts)
    L_in_derivative_hes = -jax.numpy.linalg.solve(H, crossderivative_hes)
    L_in_derivative_faces = -jax.numpy.linalg.solve(H, crossderivative_faces)

    grad_verts = (
        L_in_derivative_verts.T
        @ grad(L_out, argnums=0)(
            vertTable_eq,
            heTable_eq,
            faceTable_eq,
            width,
            height,
            vertTable_target,
            heTable_target,
            faceTable_target,
            image_target,
        )[:, :2].flatten()
    )
    grad_hes = (
        L_in_derivative_hes.T
        @ grad(L_out, argnums=0)(
            vertTable_eq,
            heTable_eq,
            faceTable_eq,
            width,
            height,
            vertTable_target,
            heTable_target,
            faceTable_target,
            image_target,
        )[:, :2].flatten()
    )
    grad_faces = (
        L_in_derivative_faces.T
        @ grad(L_out, argnums=0)(
            vertTable_eq,
            heTable_eq,
            faceTable_eq,
            width,
            height,
            vertTable_target,
            heTable_target,
            faceTable_target,
            image_target,
        )[:, :2].flatten()
    )

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    new_vert_params: Array = updated_params["vert_params"]
    new_he_params: Array = updated_params["he_params"]
    new_face_params: Array = updated_params["face_params"]

    return new_vert_params, new_he_params, new_face_params


##########################
## ADJOINT STATE METHOD ##
##########################


def outer_adjoint_state(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
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
    def L_in_flatten(
        vertTable_flatten, heTable, faceTable, width: float, height: float, vert_params, he_params, face_params
    ):
        vertTable_tmp = jnp.hstack(
            (vertTable_flatten.reshape(len(vertTable_flatten) // 2, 2), jnp.zeros((len(vertTable_flatten) // 2, 1)))
        )
        return L_in(vertTable_tmp, heTable, faceTable, width, height, vert_params, he_params, face_params)

    (vertTable_eq, heTable_eq, faceTable_eq), _ = inner_opt(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
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
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, width, height, vert_params, he_params, face_params
    )

    # print(jax.numpy.linalg.eig(H)[0].max())
    # print(jax.numpy.linalg.eig(H)[0].min())
    # print('\n')

    crossderivative_verts = jacfwd(grad(L_in_flatten, argnums=5), argnums=0)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, width, height, vert_params, he_params, face_params
    )
    crossderivative_hes = jacfwd(grad(L_in_flatten, argnums=6), argnums=0)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, width, height, vert_params, he_params, face_params
    )
    crossderivative_faces = jacfwd(grad(L_in_flatten, argnums=7), argnums=0)(
        vertTable_eq[:, :2].flatten(), heTable_eq, faceTable_eq, width, height, vert_params, he_params, face_params
    )

    gradout = grad(L_out, argnums=0)(
        vertTable_eq,
        heTable_eq,
        faceTable_eq,
        width,
        height,
        vertTable_target,
        heTable_target,
        faceTable_target,
        image_target,
    )[:, :2].flatten()

    Lambda = -jax.numpy.linalg.solve(H, gradout)

    grad_verts = crossderivative_verts @ Lambda
    grad_hes = crossderivative_hes @ Lambda
    grad_faces = crossderivative_faces @ Lambda

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    new_vert_params: Array = updated_params["vert_params"]
    new_he_params: Array = updated_params["he_params"]
    new_face_params: Array = updated_params["face_params"]

    return new_vert_params, new_he_params, new_face_params


#############
## WRAPPER ##
#############


class BilevelOptimizationMethod(Enum):
    """Which optimization method to use in the bi-level optimization."""

    AUTOMATIC_DIFFERENTIATION = "ad"
    EQUILIBRIUM_PROPAGATION = "ep"
    IMPLICIT_DIFFERENTIATION = "id"
    ADJOINT_STATE = "as"


def bilevel_opt(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vert_params,
    he_params,
    face_params,
    vertTable_target,
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
    beta=0.01,
    method: BilevelOptimizationMethod = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION,
):
    match method:
        case BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION:
            vert_params, he_params, face_params = outer_opt(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
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
            vert_params, he_params, face_params = outer_eq_prop(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
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

        case BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION:
            vert_params, he_params, face_params = outer_implicit(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
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

        case BilevelOptimizationMethod.ADJOINT_STATE:
            vert_params, he_params, face_params = outer_adjoint_state(
                vertTable,
                heTable,
                faceTable,
                width,
                height,
                vert_params,
                he_params,
                face_params,
                vertTable_target,
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

        case _:
            msg = f"Method not recognized. Must be a BilevelOptimizationMethod. Got {method}."
            raise ValueError(msg)

    (vertTable, heTable, faceTable), cost = inner_opt(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
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

    return (vertTable, heTable, faceTable, vert_params, he_params, face_params), cost
