"""Bilevel optimizer for bounded meshes."""

import jax.numpy as jnp
from jax import Array

from vertax.bilevelopt.bilevelopt import _BilevelOptimizer
from vertax.meshes.bounded_mesh import BoundedMesh
from vertax.meshes.mesh import Mesh
from vertax.opt_bounded import (
    BilevelOptimizationMethod,
    inner_opt_bounded,
    outer_adjoint_state_bounded,
    outer_eq_prop_bounded,
    outer_implicit_bounded,
    outer_opt_bounded,
)
from vertax.topo import do_not_update_T1_bounded, update_T1_bounded

__all__ = ["BoundedBilevelOptimizer"]


class BoundedBilevelOptimizer(_BilevelOptimizer):
    """Bi-level optimizer for periodic boundary condition meshes."""

    def __init__(self) -> None:
        """Create a Bi-level optimizer for periodic boundary condition meshes with default parameters."""
        super().__init__()
        self._inner_opt_func = self._inner_opt
        self._outer_opt_func = self._outer_opt
        self.angles_target = jnp.array([])

    def _set_update_T1_func(self, b: bool) -> None:  # noqa: N802
        """Set the _update_T1_func callable with respect to whether it is needed or not.

        Must be implemented by child classes.
        """
        if b:
            self._update_T1_func = update_T1_bounded
        else:
            self._update_T1_func = do_not_update_T1_bounded

    def compute_outer_loss(
        self,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> float:
        """Get the result of self.loss_function_outer called with the correct arguments.

        Must be implemented by child classes.
        """
        selected_vertices, selected_edges, selected_faces = self._selection_to_jax_arrays(
            only_on_vertices, only_on_edges, only_on_faces
        )
        if not isinstance(mesh, BoundedMesh):
            msg = "The mesh given to a BoundedBilevelOptimizer must be a BoundedMesh."
            raise ValueError(msg)
        elif self.loss_function_outer is None:
            msg = "The outer loss function was not defined."
            raise AttributeError(msg)
        return float(
            self.loss_function_outer(
                mesh.vertices,
                mesh.angles,
                mesh.edges,
                mesh.faces,
                self.vertices_target,
                self.angles_target,
                self.edges_target,
                self.faces_target,
                selected_vertices,
                selected_edges,
                selected_faces,
                self.image_target,
            )
        )

    def compute_inner_loss(
        self,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
    ) -> float:
        """Get the result of self.loss_function_inner called with the correct arguments.

        Must be implemented by child classes.
        """
        selected_vertices, selected_edges, selected_faces = self._selection_to_jax_arrays(
            only_on_vertices, only_on_edges, only_on_faces
        )
        if not isinstance(mesh, BoundedMesh):
            msg = "The mesh given to a BoundedBilevelOptimizer must be a BoundedMesh."
            raise ValueError(msg)
        elif self.loss_function_inner is None:
            msg = "The inner loss function was not defined."
            raise AttributeError(msg)
        return float(
            self.loss_function_inner(
                mesh.vertices,
                mesh.angles,
                mesh.edges,
                mesh.faces,
                selected_vertices,
                selected_edges,
                selected_faces,
                mesh.vertices_params,
                mesh.edges_params,
                mesh.faces_params,
            )
        )

    def _inner_opt(
        self,
        mesh: Mesh,
        only_on_vertices: None | Array = None,
        only_on_edges: None | Array = None,
        only_on_faces: None | Array = None,
    ) -> list:
        """Call the correct inner optimization function for a BoundedMesh."""
        if not isinstance(mesh, BoundedMesh):
            msg = "The mesh given to a BoundedBilevelOptimizer must be a BoundedMesh."
            raise ValueError(msg)
        elif self.loss_function_inner is None:
            msg = "The inner loss function was not defined."
            raise AttributeError(msg)
        elif self._update_T1_func is None:
            msg = "The update T1 method was not set by a boolean."
            raise AttributeError(msg)
        else:
            (mesh.vertices, mesh.angles, mesh.edges, mesh.faces), loss_history = inner_opt_bounded(
                vertTable=mesh.vertices,
                angTable=mesh.angles,
                heTable=mesh.edges,
                faceTable=mesh.faces,
                vert_params=mesh.vertices_params,
                he_params=mesh.edges_params,
                face_params=mesh.faces_params,
                L_in=self.loss_function_inner,
                solver=self.inner_solver,
                min_dist_T1=self.min_dist_T1,
                iterations_max=self.max_nb_iterations,
                tolerance=self.tolerance,
                patience=self.patience,
                selected_verts=only_on_vertices,
                selected_hes=only_on_edges,
                selected_faces=only_on_faces,
                update_T1_func=self._update_T1_func,
            )
        return list(loss_history)

    def _outer_opt(
        self,
        mesh: Mesh,
        selected_vertices: None | Array = None,
        selected_edges: None | Array = None,
        selected_faces: None | Array = None,
    ) -> None:
        """Call the correct outer optimization function for a BoundedMesh."""
        if not isinstance(mesh, BoundedMesh):
            msg = "The mesh given to a BoundedBilevelOptimizer must be a BoundedMesh."
            raise ValueError(msg)
        elif self.loss_function_inner is None:
            msg = "The inner loss function was not defined."
            raise AttributeError(msg)
        elif self.loss_function_outer is None:
            msg = "The outer loss function was not defined."
            raise AttributeError(msg)
        elif self._update_T1_func is None:
            msg = "The update T1 method was not set by a boolean."
            raise AttributeError(msg)
        else:
            match self.bilevel_optimization_method:
                case BilevelOptimizationMethod.AUTOMATIC_DIFFERENTIATION:
                    mesh.vertices_params, mesh.edges_params, mesh.faces_params = outer_opt_bounded(
                        mesh.vertices,
                        mesh.angles,
                        mesh.edges,
                        mesh.faces,
                        mesh.vertices_params,
                        mesh.edges_params,
                        mesh.faces_params,
                        self.vertices_target,
                        self.angles_target,
                        self.edges_target,
                        self.faces_target,
                        self.loss_function_inner,
                        self.loss_function_outer,
                        self.inner_solver,
                        self.outer_solver,
                        self.min_dist_T1,
                        self.max_nb_iterations,
                        self.tolerance,
                        self.patience,
                        selected_vertices,
                        selected_edges,
                        selected_faces,
                        self.image_target,
                        self._update_T1_func,
                    )

                case BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION:
                    mesh.vertices_params, mesh.edges_params, mesh.faces_params = outer_eq_prop_bounded(
                        mesh.vertices,
                        mesh.angles,
                        mesh.edges,
                        mesh.faces,
                        mesh.vertices_params,
                        mesh.edges_params,
                        mesh.faces_params,
                        self.vertices_target,
                        self.angles_target,
                        self.edges_target,
                        self.faces_target,
                        self.loss_function_inner,
                        self.loss_function_outer,
                        self.inner_solver,
                        self.outer_solver,
                        self.min_dist_T1,
                        self.max_nb_iterations,
                        self.tolerance,
                        self.patience,
                        selected_vertices,
                        selected_edges,
                        selected_faces,
                        self.image_target,
                        self.beta,
                        self._update_T1_func,
                    )

                case BilevelOptimizationMethod.IMPLICIT_DIFFERENTIATION:
                    mesh.vertices_params, mesh.edges_params, mesh.faces_params = outer_implicit_bounded(
                        mesh.vertices,
                        mesh.angles,
                        mesh.edges,
                        mesh.faces,
                        mesh.vertices_params,
                        mesh.edges_params,
                        mesh.faces_params,
                        self.vertices_target,
                        self.angles_target,
                        self.edges_target,
                        self.faces_target,
                        self.loss_function_inner,
                        self.loss_function_outer,
                        self.inner_solver,
                        self.outer_solver,
                        self.min_dist_T1,
                        self.max_nb_iterations,
                        self.tolerance,
                        self.patience,
                        selected_vertices,
                        selected_edges,
                        selected_faces,
                        self.image_target,
                        self._update_T1_func,
                    )
                case BilevelOptimizationMethod.ADJOINT_STATE:
                    mesh.vertices_params, mesh.edges_params, mesh.faces_params = outer_adjoint_state_bounded(
                        mesh.vertices,
                        mesh.angles,
                        mesh.edges,
                        mesh.faces,
                        mesh.vertices_params,
                        mesh.edges_params,
                        mesh.faces_params,
                        self.vertices_target,
                        self.angles_target,
                        self.edges_target,
                        self.faces_target,
                        self.loss_function_inner,
                        self.loss_function_outer,
                        self.inner_solver,
                        self.outer_solver,
                        self.min_dist_T1,
                        self.max_nb_iterations,
                        self.tolerance,
                        self.patience,
                        selected_vertices,
                        selected_edges,
                        selected_faces,
                        self.image_target,
                        self._update_T1_func,
                    )
                case _:
                    msg = f"Method not recognized. Must be a BilevelOptimizationMethod. \
                        Got {self.bilevel_optimization_method}."
                    raise AttributeError(msg)
