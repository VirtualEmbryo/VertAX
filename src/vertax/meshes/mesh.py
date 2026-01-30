"""Abstract mesh module."""

from __future__ import annotations

from typing import Any, NoReturn, TypeVar

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

    @property
    def nb_vertices(self) -> int:
        """Get the number of vertices of the mesh."""
        return len(self.vertices)

    @property
    def nb_edges(self) -> int:
        """Get the number of edges of the mesh."""
        return self.nb_half_edges // 2

    @property
    def nb_half_edges(self) -> int:
        """Get the number of half-edges of the mesh."""
        return len(self.edges)

    @property
    def nb_faces(self) -> int:
        """Get the number of faces of the mesh."""
        return len(self.faces)
