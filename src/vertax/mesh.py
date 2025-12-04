"""Abstract mesh module."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, NoReturn, Self, TypeVar

import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

T = TypeVar("T")


class BilevelOptimizationMethod(Enum):
    """Which optimization method to use in the bi-level optimization."""

    AUTOMATIC_DIFFERENTIATION = "ad"
    EQUILIBRIUM_PROPAGATION = "ep"
    IMPLICIT_DIFFERENTIATION = "id"
    ADJOINT_STATE = "as"


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
        mesh = cls._create()
        mesh.vertices, mesh.edges, mesh.faces = mesh_file["vertices"], mesh_file["edges"], mesh_file["faces"]
        return mesh

    @property
    def nb_vertices(self) -> int:
        """Get the number of vertices of the mesh."""
        return len(self.vertices)

    @property
    def nb_edges(self) -> int:
        """Get the number of edges of the mesh."""
        return len(self.edges) // 2

    @property
    def nb_faces(self) -> int:
        """Get the number of faces of the mesh."""
        return len(self.faces)
