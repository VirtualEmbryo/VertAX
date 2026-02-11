"""Optimization methods for VertAX."""

from collections.abc import Callable
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array, grad, jacfwd, jit, jvp, lax
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator, minres

from vertax.geo import update_pbc
from vertax.method_enum import BilevelOptimizationMethod
from vertax.topo import update_T1


class OptimizationTarget(Enum):
    """What the minimize function will try to optimize."""

    VERTICES = 0
    EDGES = 1
    FACES = 2
    VERTEX_PARAMETERS = 3
    EDGE_PARAMETERS = 4
    FACE_PARAMETERS = 5


InnerLossFunction = Callable[[Array, Array, Array, Array, Array, Array], Array]
OuterLossFunction = Callable[
    [
        Array,
        Array,
        Array,
        float,
        float,
        Array,
        Array,
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
    ],
    Array,
]
LossEPFunction = Callable[
    [
        Array,
        Array,
        Array,
        float,
        float,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        InnerLossFunction,
        OuterLossFunction,
        Array | None,
        Array | None,
        Array | None,
        Array | None,
        float,
    ],
    Array,
]

UpdateT1Func = Callable[
    [
        Array,
        Array,
        Array,
        float,
        float,
        Array,
        Array,
        Array,
        InnerLossFunction,
        float,
        Array,
        Array,
        Array,
    ],
    tuple[Array, Array, Array],
]


###############################
## AUTOMATIC DIFFERENTIATION ##
###############################


@partial(jit, static_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
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
    L_in: InnerLossFunction,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    optimization_target: OptimizationTarget = OptimizationTarget.VERTICES,
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array], tuple[Array, Array]]:
    x0: Array = [vertTable, heTable, faceTable, vert_params, he_params, face_params][optimization_target.value]
    sel: Array = [
        selected_verts,
        selected_hes,
        selected_faces,
        selected_verts,
        selected_hes,
        selected_faces,
    ][optimization_target.value]
    opt_state = solver.init(x0)

    # Initial loss and bookkeeping
    initial_L = L_in(
        vertTable[selected_verts],
        heTable[selected_hes],
        faceTable[selected_faces],
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
    def scan_step(
        carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], i: int
    ) -> tuple[tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, int, Array, Array], None]:
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
            vp[selected_verts],
            hp[selected_hes],
            fp[selected_faces],
        )

        # 2) Early stopping bookkeeping
        denom = jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0)
        rel_var = jnp.abs((L_current - prev_L_values[-1]) / denom)

        new_stagnation_count = jnp.where(rel_var < tolerance, stagnation_count + 1, 0)
        new_should_stop = (new_stagnation_count >= patience) | (i >= iterations_max - 1)

        # 3) Gradient wrt chosen argnums
        grads = grad(L_in, argnums=optimization_target.value)(vt, ht, ft, vp, hp, fp)

        # 4) Optimizer update
        updates, new_opt_state = solver.update(grads, opt_state)

        # 5) Apply updates to the chosen array on selected indices
        updates_sel = updates.at[sel].get()  # type: ignore

        # --- Conditional Application of Optimization Update and State ---
        arrays = [vt, ht, ft, vp, hp, fp]
        k = optimization_target.value
        new_optim = arrays[k].at[sel].set(arrays[k][sel] + updates_sel)
        arrays[k] = lax.cond(is_running, lambda: new_optim, lambda: arrays[k])
        vt, ht, ft, vp, hp, fp = arrays

        # new_vt_optim = vt.at[sel].set(vt[sel] + updates_sel)
        # vt = lax.cond(is_running, lambda: new_vt_optim, lambda: vt)

        opt_state = lax.cond(is_running, lambda: new_opt_state, lambda: opt_state)
        stagnation_count = lax.cond(is_running, lambda: new_stagnation_count, lambda: stagnation_count)
        should_stop = new_should_stop  # should_stop is a flag for the *next* iteration

        # 6) Geometry updates (Must also be conditional/masked)
        new_vt_pbc, new_ht_pbc, new_ft_pbc = update_pbc(vt, ht, ft, width, height)
        vt = lax.cond(is_running, lambda: new_vt_pbc, lambda: vt)
        ht = lax.cond(is_running, lambda: new_ht_pbc, lambda: ht)
        ft = lax.cond(is_running, lambda: new_ft_pbc, lambda: ft)

        new_vt_T1, new_ht_T1, new_ft_T1 = update_t1_func(
            vt, ht, ft, width, height, vp, hp, fp, L_in, min_dist_T1, selected_verts, selected_hes, selected_faces
        )
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

    final_state, _ = lax.scan(scan_step, init_carry, xs=jnp.arange(iterations_max))  # ty:ignore[invalid-argument-type]

    # 4. Unpack and return results (for slicing outside JIT)
    vt_f, ht_f, ft_f, vp_f, hp_f, fp_f, _, _, _, step_f, _, L_hist = final_state
    return (vt_f, ht_f, ft_f, vp_f, hp_f, fp_f), (L_hist, step_f)  # ty:ignore[invalid-return-type]


def minimize(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: InnerLossFunction,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    optimization_target: OptimizationTarget = OptimizationTarget.VERTICES,
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array], tuple[Array, Array]]:
    """General minimization function."""
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

    return jax.block_until_ready(
        _jit_minimize(
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
            update_t1_func=update_t1_func,
        )
    )


def inner_opt(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    L_in: InnerLossFunction,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-4,
    patience: int = 5,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[tuple[Array, Array, Array], Array]:
    """Inner optimization function."""
    # Use the general minimize function with VERTICES (optimize vertTable)
    (vt_f, ht_f, ft_f, _vp_f, _hp_f, _fp_f), (L_hist_full, step_f_array) = minimize(
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
        update_t1_func=update_t1_func,
    )

    # Convert the JAX array step_f to a Python integer
    step_f = step_f_array.item()

    # Now slice using standard Python/NumPy slicing
    final_L_list = L_hist_full[:step_f]

    # Return updated arrays and loss history
    return (vt_f, ht_f, ft_f), final_L_list


def cost_ad(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
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
    update_t1_func: UpdateT1Func = update_T1,
) -> Array:
    """Automatic differentiation cost function."""
    (vertTable, heTable, faceTable), _loss = inner_opt(
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
        update_t1_func,
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
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
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
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[Array, Array, Array]:
    """Outer optimization for Automatic differentiation method."""
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
        update_t1_func,
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
        update_t1_func,
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
        update_t1_func,
    )

    params = {"vert_params": vert_params, "he_params": he_params, "face_params": face_params}
    grads = {"vert_params": grad_verts, "he_params": grad_hes, "face_params": grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    new_vert_params: Array = updated_params["vert_params"]  # type: ignore
    new_he_params: Array = updated_params["he_params"]  # type: ignore
    new_face_params: Array = updated_params["face_params"]  # type: ignore

    return new_vert_params, new_he_params, new_face_params


#############################
## EQUILIBTIUM PROPAGATION ##
#############################


def _loss_ep_static(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
) -> Array:
    loss_inner = L_in(vertTable, heTable, faceTable, vert_params, he_params, face_params)
    loss_outer = L_out(
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
    return loss_inner + (beta * loss_outer)


@partial(jit, static_argnums=(6, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24))
def _minimize_ep(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,  # 0, 1, 2
    vert_params: Array,
    he_params: Array,
    face_params: Array,  # 3, 4, 5
    loss_fn: LossEPFunction,  # 6 [STATIC]
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,  # 7, 8, 9
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,  # 10, 11 [STATIC]
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,  # 12, 13, 14
    image_target: Array | None,
    beta: float,  # 15, 16
    solver: optax.GradientTransformation,  # 17 [STATIC]
    min_dist_T1: float,  # 18
    iterations_max: int = 1000,
    tolerance: float = 1e-3,
    patience: int = 5,  # 19, 20, 21 [STATIC]
    width: float = 1,
    height: float = 1,
    update_t1_func: UpdateT1Func = update_T1,  # 22, 23, 24 : [STATIC]
) -> tuple[tuple[Array, Array, Array, Array, Array, Array], tuple[Array, Array]]:
    # 1. Helper: Bind the fixed arguments to the loss function.
    def loss_evaluated(vt: Array, ht: Array, ft: Array, vp: Array, hp: Array, fp: Array) -> Array:
        return loss_fn(
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

    # 2. Ensure static parameters are concrete integers
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
        L_in=loss_evaluated,
        solver=solver,
        min_dist_T1=min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        optimization_target=OptimizationTarget.VERTICES,
        update_t1_func=update_t1_func,
    )


def inner_eq_prop(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    loss_ep_static: LossEPFunction,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
    L_out: OuterLossFunction,
    selected_verts: Array | None,
    selected_hes: Array | None,
    selected_faces: Array | None,
    image_target: Array | None,
    beta: float,
    solver: optax.GradientTransformation,
    min_dist_T1: float,
    iterations_max: int = 1000,
    tolerance: float = 1e-3,
    patience: int = 5,
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[tuple[Array, Array, Array], Array]:
    """Wrapper to call the JIT-compiled minimize function and handle post-processing (slicing history)."""
    # Call the JIT-compiled minimize function (defined in previous step)
    # We pass all targets and loss components so they can be bound
    # inside the static wrapper.
    (vt_f, ht_f, ft_f, _vp_f, _hp_f, _fp_f), (L_hist_full, step_f_array) = _minimize_ep(
        vertTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        loss_ep_static,
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
        solver,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        width,
        height,
        update_t1_func,
    )

    # Convert the JAX array step_f to a Python integer for slicing
    step_f = step_f_array.item()

    # Slice the loss history to remove padding zeros
    final_L_list = L_hist_full[:step_f]

    # Return updated geometry and the valid loss history
    return (vt_f, ht_f, ft_f), final_L_list


def outer_eq_prop(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
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
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[Array, Array, Array]:
    """Outer optimization for equilibrium propagation method."""
    (vertTable_free, heTable_free, faceTable_free), _ = inner_eq_prop(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        _loss_ep_static,
        vertTable_target,
        heTable_target,
        faceTable_target,
        L_in,
        L_out,
        selected_verts,
        selected_hes,
        selected_faces,
        image_target,
        -beta,
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        update_t1_func,
    )

    (vertTable_nudged, heTable_nudged, faceTable_nudged), _ = inner_eq_prop(
        vertTable,
        heTable,
        faceTable,
        width,
        height,
        vert_params,
        he_params,
        face_params,
        _loss_ep_static,
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
        solver_inner,
        min_dist_T1,
        iterations_max,
        tolerance,
        patience,
        update_t1_func,
    )

    grad_loss_ep_free_verts = grad(_loss_ep_static, argnums=5)(
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
        beta=-beta,
    )

    grad_loss_ep_nudged_verts = grad(_loss_ep_static, argnums=5)(
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

    grad_loss_ep_free_hes = grad(_loss_ep_static, argnums=6)(
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
        beta=-beta,
    )

    grad_loss_ep_nudged_hes = grad(_loss_ep_static, argnums=6)(
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

    grad_loss_ep_free_faces = grad(_loss_ep_static, argnums=7)(
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
        beta=-beta,
    )

    grad_loss_ep_nudged_faces = grad(_loss_ep_static, argnums=7)(
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


def outer_implicit(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
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
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[Array, Array, Array]:
    """Outer optimization for implicit differentiation method."""

    def L_in_flatten(  # noqa: N802
        vertTable_flatten: Array,
        heTable: Array,
        faceTable: Array,
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        vertTable_tmp = vertTable_flatten.reshape(len(vertTable_flatten) // 2, 2)
        return L_in(vertTable_tmp, heTable, faceTable, vert_params, he_params, face_params)

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
        update_t1_func,
    )

    vert_flat_eq = vertTable_eq.flatten()

    # H = d^2 L_in / d vertTable^2
    G_op = grad(L_in_flatten, argnums=0)

    dtype = vert_flat_eq.dtype

    def matvec(v: ArrayLike) -> Array:
        v_casted = jnp.asarray(v, dtype=dtype)
        _, Hv = jvp(
            lambda vfe: G_op(vfe, heTable_eq, faceTable_eq, vert_params, he_params, face_params),
            (vert_flat_eq,),
            (v_casted,),
        )
        return Hv

    nv = len(vert_flat_eq)
    H_np = LinearOperator((nv, nv), matvec=matvec)  # type: ignore

    # Compute cross-derivatives (dL_in / d vertTable d param)
    crossderivative_verts_op = jacfwd(G_op, argnums=3)
    crossderivative_hes_op = jacfwd(G_op, argnums=4)
    crossderivative_faces_op = jacfwd(G_op, argnums=5)

    crossderivative_verts = crossderivative_verts_op(
        vert_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_hes = crossderivative_hes_op(
        vert_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )
    crossderivative_faces = crossderivative_faces_op(
        vert_flat_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params
    )

    # Convert JAX arrays to NumPy arrays for use with scipy.sparse.linalg.minres
    # H_np = np.asarray(H) # Replaced by linear operator
    cd_verts_np = np.asarray(crossderivative_verts)
    cd_hes_np = np.asarray(crossderivative_hes)
    cd_faces_np = np.asarray(crossderivative_faces)

    # MINRES solves H * X = B. We want X = -H^{-1} * B, so we solve H * X = -B

    b_verts = -cd_verts_np
    L_in_derivative_verts_np, _info_v = minres(H_np, b_verts)

    b_hes = -cd_hes_np
    L_in_derivative_hes_np, _info_h = minres(H_np, b_hes)

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

    # The term (dL_out / d vertTable) evaluated at the equilibrium
    dL_out_dvert = grad(L_out, argnums=0)(
        vertTable_eq,
        heTable_eq,
        faceTable_eq,
        width,
        height,
        vertTable_target,
        heTable_target,
        faceTable_target,
        image_target,
    ).flatten()

    # dL_out / d param = (dL_in_d param)^T @ (dL_out / d vertTable)
    grad_verts = L_in_derivative_verts.T @ dL_out_dvert
    grad_hes = L_in_derivative_hes.T @ dL_out_dvert
    grad_faces = L_in_derivative_faces.T @ dL_out_dvert

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


def outer_adjoint_state(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
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
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[Array, Array, Array]:
    """Outer optimization for adjoint state method."""

    def L_in_flatten(  # noqa: N802
        vertTable_flatten: Array,
        heTable: Array,
        faceTable: Array,
        vert_params: Array,
        he_params: Array,
        face_params: Array,
    ) -> Array:
        vertTable_tmp = vertTable_flatten.reshape(len(vertTable_flatten) // 2, 2)
        return L_in(vertTable_tmp, heTable, faceTable, vert_params, he_params, face_params)

    (vertTable_eq, heTable_eq, faceTable_eq), _L_in_value = inner_opt(
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
        update_t1_func,
    )

    vertTable_eq_flat = vertTable_eq.flatten()

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
    ).flatten()

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


def bilevel_opt(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vert_params: Array,
    he_params: Array,
    face_params: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    L_in: InnerLossFunction,
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
    image_target: Array | None = None,
    beta: float = 0.01,
    method: BilevelOptimizationMethod = BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION,
    update_t1_func: UpdateT1Func = update_T1,
) -> tuple[tuple[Array, Array, Array, Array, Array, Array], Array]:
    """Bilevel optimization for PBC meshes."""
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
                update_t1_func,
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
                update_t1_func,
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
                update_t1_func,
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
                update_t1_func,
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
        update_t1_func,
    )

    return (vertTable, heTable, faceTable, vert_params, he_params, face_params), cost
