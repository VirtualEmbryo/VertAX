"""Enumerations declaration, separated here to avoid dependency cycles."""

from enum import Enum


class BilevelOptimizationMethod(Enum):
    """Which optimization method to use in the bi-level optimization."""

    AUTOMATIC_DIFFERENTIATION = "ad"
    """Unrolls the inner optimization steps ; forward-mode JVP via `jax.jacfwd`,
    cost scales with the number of parameters and iterations."""
    EQUILIBRIUM_PROPAGATION = "ep"
    """Estimates the gradient from perturbed free vs nudged equilibria ; no backdrop required.
    Most efficient but depends on the perturbation size β."""
    IMPLICIT_DIFFERENTIATION = "id"
    """Differentiates the optimality condition ∇ₓE=0 via Implicit Function Theorem ; JVP variant ;
    requires Hessian solve and sensitive to ill-conditioning."""
    ADJOINT_STATE = "as"
    """Differentiates the optimality condition ∇ₓE=0 via Implicit Function Theorem ; VJP variant ;
    requires Hessian solve or sensitive to ill-conditioning."""
