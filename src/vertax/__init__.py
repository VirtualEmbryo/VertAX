"""A differentiable JAX-based framework for vertex modeling and inverse design of epithelial tissues.

Epithelial tissues dynamically reshape through local mechanical interactions among cells.
Understanding, inferring, and designing these mechanics is a central challenge in developmental biology and biophysics.
**VertAX** is a computational framework built to address this challenge.

**VertAX** is a **framework for vertex-based modeling**:
 it represents epithelial tissues as two-dimensional polygonal meshes in which cells are faces, junctions are edges,
  tricellular contacts are vertices, and mechanical equilibrium is defined by the minimum of a user-specified energy.

  Built on **JAX**, VertAX is designed not only for forward simulation,
  but also for inverse problems such as parameter inference and tissue design.

In **VertAX**, a `Mesh` represents a vertex model of an epithelial tissue. It is made of faces (cells), edges (interface between cells) and vertices (where 3 cells or more meet).

More specifically, two types of meshes are currently supported:
- `PbcMesh` have Periodic Boundary Conditions, and are used for bulk tissue dynamics without explicit external boundaries.
- `BoundedMesh` are designed for finite tissue clusters, with curved free interfaces.

The other first class objects of **VertAX** are called Bilevel Optimizers. They allow to formulate inverse problems as nested optimizations:

$$
\\begin{aligned}
\\textbf{Outer problem (learning):} \\quad
\\theta^{\\ast} &= \\arg\\min_{\\theta} \\mathcal{C}\\left(X^{\\ast}_{\\theta},\\theta\\right)
&& \\leftarrow \\text{fit data or reach a target} \\\\
\\end{aligned}
$$
$$
\\begin{aligned}
\\textbf{Inner problem (physics):} \\quad \\text{s.t.}&
X^{\\ast}_{\\theta} \\in \\arg\\min_{X} \\mathcal{E}(X,\\theta)
&& \\leftarrow \\text{compute mechanical equilibrium}
\\end{aligned}
$$


Here, $X$ denotes the tissue configuration, i.e. the vertex positions of the mesh, and $\\theta$ denotes the model parameters, such as line tensions, target areas, or shape factors.

In other words, VertAX repeatedly solves a mechanical equilibrium problem for a given parameter set $\\theta$, then updates those parameters to better match data or a design objective.

In symmetry with meshes, a base abstract class `_BilevelOptimizer` defines common hyper-parameters and methods for the bilevel optimization, but you need to use the specialized classes:
- `PbcBilevelOptimizer` for `PbcMesh`,
- `BoundedBilevelOptimizer` for `BoundedMesh`.

**VertAX** comes with pre-defined energy and cost functions but you can easily define your own functions. See the `examples` folder in the repository to have a typical example on how to use **VertAX**.

Finally, there are plot functions to easily see the results of your experiments. See `plot_mesh` for example.

Users can define their own energy functions for the inner optimization and cost function for the outer optimization, however we provide some basic ones.
If you use your own make sure to use the exact same signature as we do for these functions, otherwise it won't work. See the cost and energy functions we provide in this documentation.
"""  # noqa: D301, E501

from vertax.bilevelopt.bilevelopt import _BilevelOptimizer
from vertax.bilevelopt.boundedbop import BoundedBilevelOptimizer
from vertax.bilevelopt.pbcbop import PbcBilevelOptimizer
from vertax.cost import (
    cost_areas,
    cost_checkerboard,
    cost_d_IAS,
    cost_IAS,
    cost_mesh2image,
    cost_ratio,
    cost_tem_halfedge,
    cost_v2v,
    cost_v2v_ias,
    cost_v2v_tem,
)
from vertax.energy import (
    energy_line_tensions,
    energy_line_tensions_bounded,
    energy_shape_factor_hetero,
    energy_shape_factor_homo,
)
from vertax.meshes.bounded_mesh import BoundedMesh
from vertax.meshes.mesh import Mesh
from vertax.meshes.pbc_mesh import PbcMesh
from vertax.meshes.plot import EdgePlot, FacePlot, VertexPlot, get_plot_mesh, plot_mesh
from vertax.method_enum import BilevelOptimizationMethod

__all__ = [  # noqa: RUF022
    "Mesh",
    "PbcMesh",
    "BoundedMesh",
    "BilevelOptimizationMethod",
    "_BilevelOptimizer",
    "PbcBilevelOptimizer",
    "BoundedBilevelOptimizer",
    "plot_mesh",
    "get_plot_mesh",
    "FacePlot",
    "EdgePlot",
    "VertexPlot",
    "cost_v2v",
    "cost_mesh2image",
    "cost_areas",
    "cost_IAS",
    "cost_d_IAS",
    "cost_tem_halfedge",
    "cost_v2v_ias",
    "cost_v2v_tem",
    "cost_ratio",
    "cost_checkerboard",
    "energy_shape_factor_hetero",
    "energy_shape_factor_homo",
    "energy_line_tensions",
    "energy_line_tensions_bounded",
]
