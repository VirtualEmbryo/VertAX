"""Nonlinear conjugate gradient solver (optax ``GradientTransformation``).

A drop-in optax solver implementing the **Polak-Ribière** nonlinear conjugate
gradient (CG) method with automatic restarts and a zoom line search. It plugs
into the same optimisation loops as ``optax.adam`` / ``optax.sgd`` (e.g.
``vertax.opt_bounded.inner_opt_bounded``), provided that loop forwards the line
search arguments ``value``, ``grad`` and ``value_fn`` to ``solver.update`` — see
the note on the line search below.

Why conjugate gradient
----------------------
Steepest descent (``d = -∇f``) zig-zags in elongated valleys: each step partly
undoes the previous one. CG removes this by making successive search directions
*conjugate* (non-interfering with respect to the local curvature), so progress
along a valley is not destroyed at the next step. On a strictly quadratic
objective in ``n`` variables it converges in at most ``n`` steps; a general
energy is not quadratic, hence *nonlinear* CG with periodic restarts.

The algorithm (one ``update`` call)
-----------------------------------
Given the gradient ``g_k = ∇f(x_k)``, the previous gradient ``g_{k-1}`` and the
previous direction ``d_{k-1}``:

1. **Polak-Ribière coefficient**
   ``β = ⟨g_k, g_k - g_{k-1}⟩ / ‖g_{k-1}‖²``.
   The ``g_k - g_{k-1}`` term injects curvature information without ever forming
   a Hessian.
2. **Restart** (set ``β = 0`` → fall back to ``-g``) when either
   - it has been ``restart_every`` steps since the last restart (conjugacy
     degrades on a non-quadratic objective), or
   - ``β ≤ 0`` — the **PR+** safeguard ``β⁺ = max(0, β)``: a negative ``β`` means
     the old direction is no longer useful, so we drop it.
3. **Direction** ``d_k = -g_k + β · d_{k-1}``
   (``β = 0`` ⇒ steepest descent; ``β > 0`` ⇒ curve the search along the valley).
4. **Descent guard**: a line search is only valid if ``d`` goes downhill, i.e.
   ``⟨d, g⟩ < 0``. Rounding / non-quadratic effects can break this; if so we fall
   back to the always-valid steepest-descent direction ``-g``.
5. **Zoom line search** along ``d``: chooses the step size satisfying the (strong)
   Wolfe conditions by re-evaluating ``f`` along the line. ``update`` returns
   ``step · d`` as the optax updates.

Line search requirement
------------------------
Like ``optax.lbfgs``, this solver wraps ``optax.scale_by_zoom_linesearch`` and
therefore needs the loss value and a callable to re-evaluate it. The enclosing
loop must call::

    updates, state = solver.update(grad, state, params,
                                   value=current_value, grad=grad,
                                   value_fn=loss_fn)

where ``loss_fn(params) -> scalar``. ``optax.adam`` / ``optax.sgd`` ignore these
extra arguments, so a loop that always forwards them stays compatible with both.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu


class NonlinearCGState(NamedTuple):
    """State carried by the nonlinear CG solver between ``update`` calls."""

    count: jax.Array  # iteration counter (since init)
    grad_prev: optax.Updates  # gradient at the previous point  (g_{k-1})
    dir_prev: optax.Updates  # previous conjugate direction     (d_{k-1})
    ls_state: Any  # zoom line-search inner state


def nonlinear_cg(
    restart_every: int = 10,
    max_linesearch_steps: int = 20,
) -> optax.GradientTransformationExtraArgs:
    """Polak-Ribière nonlinear conjugate gradient with restarts + zoom line search.

    Args:
        restart_every: force a steepest-descent restart every this many steps.
            Smaller = more robust on strongly non-quadratic objectives, larger =
            longer conjugate runs.
        max_linesearch_steps: maximum zoom line-search iterations per step.

    Returns:
        An ``optax`` solver. Its ``update`` requires ``value``, ``grad`` and
        ``value_fn`` keyword arguments (see module docstring).
    """
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps)

    def init_fn(params: optax.Params) -> NonlinearCGState:
        return NonlinearCGState(
            count=jnp.zeros([], jnp.int32),
            grad_prev=otu.tree_zeros_like(params),
            dir_prev=otu.tree_zeros_like(params),
            ls_state=linesearch.init(params),
        )

    def update_fn(
        updates: optax.Updates,  # = grad, by optax convention
        state: NonlinearCGState,
        params: optax.Params | None = None,
        *,
        value: jax.Array | None = None,
        grad: optax.Updates | None = None,
        value_fn=None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, NonlinearCGState]:
        if params is None or value is None or grad is None or value_fn is None:
            msg = (
                "nonlinear_cg.update requires `params`, `value`, `grad` and "
                "`value_fn` (it performs a line search). Pass them from the loop."
            )
            raise ValueError(msg)

        g = grad
        g_prev = state.grad_prev
        d_prev = state.dir_prev

        # On the very first step grad_prev is zero → force a steepest-descent start.
        is_first = state.count == 0

        # (1) Polak-Ribière coefficient.
        gg_prev = otu.tree_vdot(g_prev, g_prev) + 1e-12  # ‖g_{k-1}‖²
        y = otu.tree_sub(g, g_prev)  # g_k - g_{k-1}
        beta_pr = otu.tree_vdot(g, y) / gg_prev

        # (2) Restart: periodic, or PR+ (β ≤ 0), or first step.
        do_restart = is_first | (state.count % restart_every == 0) | (beta_pr <= 0.0)
        beta = jnp.where(do_restart, 0.0, beta_pr)

        # (3) Conjugate direction d = -g + β · d_prev.
        d = otu.tree_add_scalar_mul(otu.tree_scalar_mul(-1.0, g), beta, d_prev)

        # (4) Descent-direction guard: ensure ⟨d, g⟩ < 0, else use -g.
        is_descent = otu.tree_vdot(d, g) < 0.0
        d = jax.lax.cond(is_descent, lambda: d, lambda: otu.tree_scalar_mul(-1.0, g))

        # (5) Zoom line search along d → updates = step · d.
        step_updates, ls_state = linesearch.update(
            updates=d,
            state=state.ls_state,
            params=params,
            value=value,
            grad=g,
            value_fn=value_fn,
        )

        new_state = NonlinearCGState(
            count=state.count + 1,
            grad_prev=g,
            dir_prev=d,
            ls_state=ls_state,
        )
        return step_updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)  # ty:ignore[invalid-argument-type]
