"""Bi-level optimizers abstract class."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import jax.numpy as jnp
import optax
import pandas
from jax import Array

from vertax.meshes.mesh import Mesh
from vertax.meshes.plot import plot_mesh, save_simple_xy_graph
from vertax.method_enum import BilevelOptimizationMethod

__all__ = ["_BilevelOptimizer"]


class _BilevelOptimizer:
    """Abstract class for Bi-level optimizers."""

    def __init__(self) -> None:
        """Initialize shared parameters and hyper-parameters between Bi-level optimizers."""
        self.custom_metrics: dict[str, tuple[Callable[[Any, Any], float], list[float], list[float]]] = {}
        self.bilevel_optimization_method: BilevelOptimizationMethod = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
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
        raise NotImplementedError

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
        raise NotImplementedError

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
            return self._inner_opt_func(
                mesh, *self._selection_to_jax_arrays(only_on_vertices, only_on_edges, only_on_faces)
            )

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
            self._outer_opt_func(mesh, *self._selection_to_jax_arrays(only_on_vertices, only_on_edges, only_on_faces))

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

    def do_n_bilevel_optimization(  # noqa: C901
        self,
        nb_epochs: int,
        mesh: Mesh,
        only_on_vertices: None | list[int] = None,
        only_on_edges: None | list[int] = None,
        only_on_faces: None | list[int] = None,
        pre_inner_optimization: bool = False,
        report_every: int = 0,
        also_report_to_stdout: bool = False,
        save_mesh_every: int = 0,
        save_folder: str = ".",
    ) -> None:
        """Optimize the mesh for the loss function given.

        Args:
            nb_epochs (int): The number of bilevel optimization steps to perform.
            mesh (Mesh): The mesh to act on.
            only_on_vertices (None | list[int] = None): Consider only a subset of vertices for the loss function.
                                                        All vertices if None.
            only_on_edges (None | list[int] = None): Consider only a subset of edges for the loss function.
                                                        All edges if None.
            only_on_faces (None | list[int] = None): Consider only a subset of faces for the loss function.
                                                        All faces if None.
            pre_inner_optimization (bool = False): Whether to perform an optional initial inner optimization first.
            report_every (int = 0): If strictly positive, report current state (costs, metrics, time) at this frequency.
            also_report_to_stdout (bool = False): Whether to also report to stdout (print).
            save_mesh_every (int = 0): If positive, save the mesh data and a plot at this frequency.
            save_folder: (str = "."): Base folder where to save data if needed.
        """
        save_folder_path = Path(save_folder)
        save_folder_path.mkdir(parents=True, exist_ok=True)
        mesh_save_folder = save_folder_path / "meshes"
        outer_cost_graph_filename = save_folder_path / "outer_cost_over_time.png"
        inner_cost_graph_filename = save_folder_path / "inner_cost_over_time.png"
        summary_filename = save_folder_path / "summary.csv"
        parameters_filename = save_folder_path / "hyper-parameters.txt"

        with parameters_filename.open("w") as f:
            f.write(self.self_summary())

        all_epoch_data = []
        epoch_data = []
        if save_mesh_every > 0:
            mesh_save_folder.mkdir(parents=True, exist_ok=True)
            plot_mesh(mesh, show=False, save=True, save_path=str(mesh_save_folder / "mesh_epoch_0.png"))

        x_epochs: list[float] = []
        y_outer_costs: list[float] = []
        y_inner_costs: list[float] = []
        if report_every > 0:
            epoch_data = [0]
            outer_cost = self.compute_outer_loss(mesh, only_on_vertices, only_on_edges, only_on_faces)
            inner_cost = self.compute_inner_loss(mesh)
            x_epochs.append(0)
            y_outer_costs.append(outer_cost)
            y_inner_costs.append(inner_cost)
            epoch_data.extend([outer_cost, inner_cost, 0, 0])  # the zeros are time and delta time
            # Reset and put initial metrics
            for metric_name in self.custom_metrics:
                self.custom_metrics[metric_name][1].clear()
                self.custom_metrics[metric_name][2].clear()
                self.custom_metrics[metric_name][1].append(0.0)
                metric = self.custom_metrics[metric_name][0](mesh, self)
                self.custom_metrics[metric_name][2].append(metric)
                epoch_data.append(metric)
            all_epoch_data.append(epoch_data)
            pandas.DataFrame(
                all_epoch_data,
                columns=[
                    "Epoch",
                    "Outer cost",
                    "Inner cost",
                    "Time (s)",
                    "Delta Time (s)",
                    *(metric_name for metric_name in self.custom_metrics),
                ],  # ty:ignore[invalid-argument-type]
            ).to_csv(summary_filename)

            if also_report_to_stdout:
                msg = "First epoch may be a bit long."
                msg += f" Initial state: Outer cost = {outer_cost}, Inner cost = {inner_cost}"
                for metric_name in self.custom_metrics:
                    msg += f", {metric_name} = {self.custom_metrics[metric_name][2][-1]}"
                print(msg)

        if pre_inner_optimization:
            self.inner_optimization(mesh, only_on_vertices, only_on_edges, only_on_faces)

        t_initial = perf_counter()
        for epoch in range(1, nb_epochs + 1):
            epoch_data = [epoch]
            t_begin_of_epoch = perf_counter()

            self.bilevel_optimization(mesh, only_on_vertices, only_on_edges, only_on_faces)
            t_end_of_epoch = perf_counter()
            total_time = t_end_of_epoch - t_initial
            delta_time = t_end_of_epoch - t_begin_of_epoch

            if save_mesh_every > 0 and epoch % save_mesh_every == 0:
                plot_mesh(mesh, show=False, save=True, save_path=str(mesh_save_folder / f"mesh_epoch_{epoch}.png"))

            if report_every > 0 and epoch % report_every == 0:
                # Compute results data for this step
                outer_cost = self.compute_outer_loss(mesh, only_on_vertices, only_on_edges, only_on_faces)
                inner_cost = self.compute_inner_loss(mesh)
                x_epochs.append(epoch)
                y_outer_costs.append(outer_cost)
                y_inner_costs.append(inner_cost)
                epoch_data.extend([outer_cost, inner_cost, total_time, delta_time])  # the zeros are time and delta time
                for metric_name in self.custom_metrics:
                    self.custom_metrics[metric_name][1].append(epoch)
                    metric = self.custom_metrics[metric_name][0](mesh, self)
                    self.custom_metrics[metric_name][2].append(metric)
                    epoch_data.append(metric)

                all_epoch_data.append(epoch_data)
                pandas.DataFrame(
                    all_epoch_data,
                    columns=[
                        "Epoch",
                        "Outer cost",
                        "Inner cost",
                        "Time (s)",
                        "Delta Time (s)",
                        *(metric_name for metric_name in self.custom_metrics),
                    ],  # ty:ignore[invalid-argument-type]
                ).to_csv(summary_filename)

                # Report in log.
                if also_report_to_stdout:
                    msg = f"\nEpoch {epoch}/{nb_epochs} : Outer cost = {outer_cost}, Inner cost = {inner_cost}"
                    for metric_name in self.custom_metrics:
                        msg += f", {metric_name} = {self.custom_metrics[metric_name][2][-1]}"
                    msg += f", Time = {total_time}s (+{delta_time}s)"
                    print(msg)

                # Update graphs
                save_simple_xy_graph(
                    str(outer_cost_graph_filename),
                    x_epochs,
                    y_outer_costs,
                    "Outer cost over time",
                    "Epoch",
                    "Outer cost",
                )
                save_simple_xy_graph(
                    str(inner_cost_graph_filename),
                    x_epochs,
                    y_inner_costs,
                    "Inner cost over time",
                    "Epoch",
                    "Inner cost",
                )
                for metric_name in self.custom_metrics:
                    graph_name = save_folder_path / (metric_name + ".png")
                    save_simple_xy_graph(
                        str(graph_name),
                        self.custom_metrics[metric_name][1],
                        self.custom_metrics[metric_name][2],
                        f"{metric_name} over time",
                        "Epoch",
                        metric_name,
                    )

    def self_summary(self) -> str:
        """Give a summary of hyper-parameters of the optimizer."""
        summary = f"Bi-level optimization method: {self.bilevel_optimization_method.value}\n"
        if self.loss_function_inner is not None:
            summary += f"Inner loss function: {self.loss_function_inner.__name__}\n"  # ty:ignore[unresolved-attribute]
        else:
            summary += "Inner loss function: None\n"
        if self.loss_function_outer is not None:
            summary += f"Outer loss function: {self.loss_function_outer.__name__}\n"  # ty:ignore[unresolved-attribute]
        else:
            summary += "Outer loss function: None\n"
        summary += f"Max number of iterations: {self.max_nb_iterations}\n"
        summary += f"Tolerance: {self.tolerance}\n"
        summary += f"Patience: {self.patience}\n"
        summary += f"Minimum distance for T1: {self.min_dist_T1}\n"
        summary += f"Update T1: {self.update_T1}\n"
        if self._update_T1_func is not None:
            summary += f"Update T1 function: {self._update_T1_func.__name__}\n"  # ty:ignore[unresolved-attribute]
        else:
            summary += "Update T1 function: None\n"
        summary += f"Beta: {self.beta}"
        return summary

    def add_custom_metric(self, name: str, function: Callable[[Any, Any], float]) -> None:
        """Add a custom metric to the metrics to save when performing n bilevel optimizations.

        The function must take two arguments : a mesh and a bilevel optimizer.
        """
        self.custom_metrics[name] = (function, [], [])

    def remove_custom_metric(self, name: str) -> None:
        """Remove a custom metric from the metrics to save when performing n bilevel optimizations."""
        if name in self.custom_metrics:
            del self.custom_metrics[name]

    def _selection_to_jax_arrays(
        self,
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
