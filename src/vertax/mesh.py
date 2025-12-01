"""Abstract mesh module."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, Self, TypeVar

import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from vertax.opt import BilevelOptimizationMethod, bilevel_opt, inner_opt

T = TypeVar("T")
type InnerLossFunction = Callable[[Array, Array, Array, float, float, Array, Array, Array], float]
type OuterLossFunction = Callable[
    [
        Array,
        Array,
        Array,
        float,
        float,
        Array,
        Array,
        Array,
        None | list[float],
        None | list[float],
        None | list[float],
        Array,
    ],
    float,
]


class NoPublicConstructor(type):
    """Metaclass that ensures a private constructor.

    If a class uses this metaclass like this:

        class SomeClass(metaclass=NoPublicConstructor):
            pass

    If you try to instantiate your class (`SomeClass()`),
    a `TypeError` will be thrown.
    """

    def __call__(cls, *args, **kwargs) -> NoReturn:  # noqa
        """Make it impossible to call with ClassName()."""
        msg = f"{cls.__module__}.{cls.__qualname__} has no public constructor"
        raise TypeError(msg)

    def _create(cls: type[T], *args: Any, **kwargs: Any) -> T:  # noqa: ANN401
        return super().__call__(*args, **kwargs)  # type: ignore


class Mesh(metaclass=NoPublicConstructor):
    """Generic mesh structure."""

    def __init__(self) -> None:
        """Do nothing but create attributes. Do not call this."""
        self.vertices: Array = jnp.array([])
        self.edges: Array = jnp.array([])
        self.faces: Array = jnp.array([])
        self.width: float = 0
        self.height: float = 0

        self.vertices_params: Array = jnp.array([])
        self.edges_params: Array = jnp.array([])
        self.faces_params: Array = jnp.array([])

        self.vertices_target: Array = jnp.array([])
        self.edges_target: Array = jnp.array([])
        self.faces_target: Array = jnp.array([])

        self.image_target: Array = jnp.array([])

        self.bilevel_optimization_method: BilevelOptimizationMethod = (
            BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION
        )
        self.beta: float = 0.01

        self.min_dist_T1: float = 0.005
        self.max_nb_iterations: int = 1000
        self.tolerance: float = 1e-4
        self.patience: int = 5

        self.inner_solver: optax.GradientTransformation = optax.sgd(learning_rate=0.01)
        self.outer_solver: optax.GradientTransformation = optax.adam(learning_rate=0.0001, nesterov=True)

    def save_mesh(self, path: str) -> None:
        """Save mesh to a file.

        Args:
            path (str): Path to the saved file. The extension is .npz.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, allow_pickle=False, vertices=self.vertices, edges=self.edges, faces=self.faces)

    @classmethod
    def load_mesh(cls, path: str) -> Self:
        """Load a mesh from a file.

        Args:
            path (str): Path to the mesh file (.npz).

        Returns:
            Mesh: the mesh loaded from the .npz file.
        """
        mesh_file = np.load(path)
        return cls(mesh_file["vertices"], mesh_file["edges"], mesh_file["faces"])

    def inner_opt(
        self,
        loss_function_inner: InnerLossFunction,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> list[float]:
        """Optimize the mesh for the loss function given.

        Args:
            loss_function_inner (InnerLossFunction): Loss function to optimize.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.

        Returns:
            list[float]: History of loss values during optimization.
        """
        selected_vertices, selected_edges, selected_faces = None, None, None
        if only_on_vertices is not None:
            selected_vertices = jnp.array(only_on_vertices)
        if only_on_edges is not None:
            selected_edges = jnp.array(only_on_edges)
        if only_on_faces is not None:
            selected_faces = jnp.array(only_on_faces)

        (self.vertices, self.edges, self.faces), loss_history = inner_opt(
            vertTable=self.vertices,
            heTable=self.edges,
            faceTable=self.faces,
            width=self.width,
            height=self.height,
            vert_params=self.vertices_params,
            he_params=self.edges_params,
            face_params=self.faces_params,
            L_in=loss_function_inner,
            solver=self.inner_solver,
            min_dist_T1=self.min_dist_T1,
            iterations_max=self.max_nb_iterations,
            tolerance=self.tolerance,
            patience=self.patience,
            selected_verts=selected_vertices,
            selected_hes=selected_edges,
            selected_faces=selected_faces,
        )
        return list(loss_history)

    def bilevel_opt(
        self,
        loss_function_inner: InnerLossFunction,
        loss_function_outer: OuterLossFunction,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> list[float]:
        """Optimize the mesh for the loss function given.

        Args:
            loss_function_inner (InnerLossFunction): Loss function to optimize.
            loss_function_outer (OuterLossFunction): Loss function to optimize.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.

        Returns:
            list[float]: History of loss values during optimization.
        """
        selected_vertices, selected_edges, selected_faces = None, None, None
        if only_on_vertices is not None:
            selected_vertices = jnp.array(only_on_vertices)
        if only_on_edges is not None:
            selected_edges = jnp.array(only_on_edges)
        if only_on_faces is not None:
            selected_faces = jnp.array(only_on_faces)
        (
            (self.vertices, self.edges, self.faces, self.vertices_params, self.edges_params, self.faces_params),
            loss_history,
        ) = bilevel_opt(
            vertTable=self.vertices,
            heTable=self.edges,
            faceTable=self.faces,
            width=self.width,
            height=self.height,
            vert_params=self.vertices_params,
            he_params=self.edges_params,
            face_params=self.faces_params,
            vertTable_target=self.vertices_target,
            heTable_target=self.edges_target,
            faceTable_target=self.faces_target,
            L_in=loss_function_inner,
            L_out=loss_function_outer,
            solver_inner=self.inner_solver,
            solver_outer=self.outer_solver,
            min_dist_T1=self.min_dist_T1,
            iterations_max=self.max_nb_iterations,
            tolerance=self.tolerance,
            patience=self.patience,
            selected_verts=selected_vertices,
            selected_hes=selected_edges,
            selected_faces=selected_faces,
            image_target=self.image_target,
            beta=self.beta,
            method=self.bilevel_optimization_method,
        )

        return list(loss_history)
