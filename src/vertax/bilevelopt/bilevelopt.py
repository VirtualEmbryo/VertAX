"""Bi-level optimizers abstract class."""

from collections.abc import Callable

import jax.numpy as jnp
import optax
from jax import Array

from vertax.meshes.mesh import Mesh
from vertax.method_enum import BilevelOptimizationMethod

__all__ = ["_BilevelOptimizer"]


class _BilevelOptimizer:
    """Abstract class for Bi-level optimizers."""

    def __init__(self) -> None:
        """Initialize shared parameters and hyper-parameters between Bi-level optimizers."""
        self.mesh: Mesh | None = None
        self.bilevel_optimization_method: BilevelOptimizationMethod = (
            BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
        )
        self.inner_solver: optax.GradientTransformation = optax.sgd(learning_rate=0.01)
        self.outer_solver: optax.GradientTransformation = optax.adam(learning_rate=0.0001, nesterov=True)
        self.loss_function_inner: Callable | None = None
        self.loss_function_outer: Callable | None = None

        self.max_nb_iterations: int = 1000
        self.tolerance: float = 1e-4
        self.patience: int = 5

        self.min_dist_T1: float = 0.005
        self._update_T1: bool = False
        self._update_T1_func: Callable | None = None  # value set by _set_update_T1_func

        # These values will be set in the init function of child classes
        self._inner_opt_func: Callable[[Mesh, Array | None, Array | None, Array | None], list[float]] | None = None
        self._outer_opt_func: Callable[[Mesh, Array | None, Array | None, Array | None], None] | None = None

        self.update_T1 = True  # Force the setting of update T1 func

        # Targets
        self.vertices_target = jnp.array([])
        self.edges_target = jnp.array([])
        self.faces_target = jnp.array([])
        # Those attributes are not always used (depends on the bilevel_optimization_method)
        self.image_target: Array = jnp.array([])
        self.beta = 0.01

    def _set_update_T1_func(self, b: bool) -> None:  # noqa: N802
        """Set the _update_T1_func callable with respect to whether it is needed or not.

        Must be implemented by child classes.
        """
        raise NotImplementedError

    @property
    def update_T1(self) -> bool:  # noqa: N802
        """Whether to process T1 topological operations or not."""
        return self._update_T1

    @update_T1.setter
    def update_T1(self, value: bool) -> None:  # noqa: N802
        self._update_T1 = value
        self._set_update_T1_func(value)

    def inner_optimization(
        self,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> list[float]:
        """Optimize the mesh for the inner loss function.

        Args:
            mesh (Mesh): The mesh to act on.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.

        Returns:
            list[float]: Histor of loss values during optimization.
        """
        # To be defined by child classes
        if self._inner_opt_func is None:
            msg = "The inner function was not initialized."
            raise AttributeError(msg)
        else:
            return self._inner_opt_func(mesh, *_selection_to_jax_arrays(only_on_vertices, only_on_edges, only_on_faces))

    def outer_optimization(
        self,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> None:
        """Optimize the mesh for the outer loss function.

        Args:
            mesh (Mesh): The mesh to act on.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.
        """
        # To be defined by child classes
        if self._outer_opt_func is None:
            msg = "The outer function was not initialized."
            raise AttributeError(msg)
        else:
            self._outer_opt_func(mesh, *_selection_to_jax_arrays(only_on_vertices, only_on_edges, only_on_faces))

    def bilevel_optimization(
        self,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> list[float]:
        """Optimize the mesh for the loss function given.

        Args:
            mesh (Mesh): The mesh to act on.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.

        Returns:
            list[float]: History of loss values during optimization.
        """
        self.outer_optimization(mesh, only_on_vertices, only_on_edges, only_on_faces)
        return self.inner_optimization(mesh, only_on_vertices, only_on_edges, only_on_faces)


def _selection_to_jax_arrays(
    only_on_vertices: None | list[int] = None,
    only_on_edges: None | list[int] = None,
    only_on_faces: None | list[int] = None,
) -> tuple[Array | None, Array | None, Array | None]:
    selected_vertices, selected_edges, selected_faces = None, None, None
    if only_on_vertices is not None:
        selected_vertices = jnp.array(only_on_vertices)
    if only_on_edges is not None:
        selected_edges = jnp.array(only_on_edges)
    if only_on_faces is not None:
        selected_faces = jnp.array(only_on_faces)
    return selected_vertices, selected_edges, selected_faces
