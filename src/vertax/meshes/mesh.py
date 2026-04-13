"""Abstract mesh module."""

from __future__ import annotations

from typing import Any, NoReturn, Self, TypeVar

import jax.numpy as jnp
from jax import Array

T = TypeVar("T")

__all__ = ["Mesh"]


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
    """Generic mesh structure. It is an abstract base class, not to be used directly.

    It defines common attributes and functions between `PbcMesh` and `BoundedMesh`.
    """

    def __init__(self) -> None:
        """Do nothing but create attributes. Do not call this, but call specialized class methods to create meshes.

        See `PbcMesh` and `BoundedMesh`.

        Technically, it uses a DCEL structure.
        """
        self.faces: Array = jnp.array([])
        """The cells of the tissue."""
        self.edges: Array = jnp.array([])
        """The interface between cells. Technically half-edges."""
        self.vertices: Array = jnp.array([])
        """The mesh vertices, where cells meet."""
        self.width: float = 0
        """The mesh live in a rectangle of size [0, width] in the X direction."""
        self.height: float = 0
        """The mesh live in a rectangle of size [0, height] in the Y direction."""

        self.faces_params: Array = jnp.array([])
        """Parameters attached to faces. Can be optimized."""
        self.edges_params: Array = jnp.array([])
        """Parameters attached to edges. Can be optimized."""
        self.vertices_params: Array = jnp.array([])
        """Parameters attached to vertices. Can be optimized."""

    @property
    def nb_faces(self) -> int:
        """Get the number of faces of the mesh."""
        return len(self.faces)

    @property
    def nb_edges(self) -> int:
        """Get the number of edges of the mesh."""
        return self.nb_half_edges // 2

    @property
    def nb_half_edges(self) -> int:
        """Get the number of half-edges of the mesh, ie. twice the number of edges."""
        return len(self.edges)

    @property
    def nb_vertices(self) -> int:
        """Get the number of vertices of the mesh."""
        return len(self.vertices)

    def save_mesh(self, path: str) -> None:
        """Save mesh to a file.

        All mesh data is saved.

        Args:
            path (str): Path to the saved file. The extension is .npz.
        """
        raise NotImplementedError

    @classmethod
    def load_mesh(cls, path: str) -> Self:
        """Load a mesh from a file.

        Args:
            path (str): Path to the mesh file (.npz).

        Returns:
            The mesh loaded from the .npz file.
        """
        raise NotImplementedError
