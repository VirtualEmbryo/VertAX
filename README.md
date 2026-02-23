<p align="center">
  <img src="./figures/vertax_text.png" alt="VertAX" width="300">
</p>

<div align="center">

<!-- Badges -->
<!-- [![License](https://img.shields.io/pypi/l/vertAX.svg)](https://github.com/vertAX/vertAX/raw/main/LICENSE)
[![Python package index](https://img.shields.io/pypi/v/vertAX.svg)](https://pypi.org/project/vertAX)
[![DOI](https://zenodo.org/badge/144513571.svg)](https://zenodo.org/badge/latestdoi/144513571)
[![Development Status](https://img.shields.io/pypi/status/vertAX.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Beta) -->

**A differentiable, JAX-powered vertex model framework for learning epithelial tissue mechanics.**

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=1200&color=38C2FF&center=true&width=500&height=30&lines=Forward+simulation;+Parameter+inference;Inverse+mechanical+design)](https://gitlab.college-de-france.fr/virtualembryo/vertax)

*— all in one unified Python package.*

</div>

---

## What is VertAX?

Epithelial tissues dynamically reshape through local mechanical interactions among cells. Understanding and *engineering* these forces is a central challenge in developmental biology and biophysics. **VertAX** provides the computational infrastructure to tackle this challenge.

VertAX represents tissues as two-dimensional **vertex models** — polygonal meshes where cells are faces, junctions are edges, and tricellular contacts are vertices. From this geometrically grounded representation, VertAX offers:

- ⚡ **GPU-accelerated** simulation via JAX and JIT compilation
- 🔁 **End-to-end differentiable** forward mechanics and inverse modeling
- 🧩 **Bilevel optimization** framework for parameter inference and inverse design
- 🔬 **Three gradient strategies**: Automatic Differentiation (AD), Implicit Differentiation (ID), and Equilibrium Propagation (EP)
- 🎨 **Fully customizable** energy and cost functions as plain Python
- 🔗 **Seamless ML integration** with JAX/Optax ecosystems

---

## Conceptual Overview

VertAX frames inverse problems within a unified **bilevel optimization** paradigm:

$$
\begin{aligned}
\textbf{Inner problem (physics):} \quad
& X^{\ast}_{\theta} \in \arg\min_{X} E(X,\theta)
&& \leftarrow \text{mechanical equilibrium}\\
\textbf{Outer problem (learning):} \quad
& \theta^{\ast} \in \arg\min_{\theta} C\left(X^{\ast}_{\theta}\right)
&& \leftarrow \text{match data or design target}
\end{aligned}
$$

The inner problem finds the equilibrium configuration $X^*$ of a tissue for a fixed parameter set $\theta$ (tensions, shape factors, ...). The outer problem then adjusts $\theta$ to minimize a user-defined cost $C$ — such as mismatch to microscopy images or deviation from a desired morphology.

<p align="center">
  <img src="./figures/concept.png" alt="VertAX Concept" width="500">
</p>

*Figure: VertAX bilevel optimization loop. Inverse modeling tasks: force inference, mechanical design, patterning.*

---

## Key Features

### 🏗️ Two simulation modes

| Mode | Use case | Initialization |
|---|---|---|
| **Periodic** | Bulk tissue dynamics, no explicit boundaries | Random Voronoi seeds or segmented images (Cellpose) |
| **Bounded** | Finite tissue clusters with curved interfaces | Random Voronoi seeds; boundary arcs as additional DOF |

### 🔀 T1 topological transitions

VertAX handles **T1 neighbor exchanges** automatically: when an edge shortens below a threshold, two candidate configurations are evaluated (flip vs. stretch) and the one with lower energy is accepted.

### 📐 Half-edge data structure

The mesh topology is encoded in three tables (`vertTable`, `heTable`, `faceTable`), separating geometry from connectivity for fast, JIT-friendly access. In bounded mode an additional `angTable` stores boundary arc angles.

---

## Installation

We recommend installing into a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install "vertax"
```

**Dependencies**: JAX, Optax, SciPy (for Voronoi initialization), Matplotlib (for plotting).

For GPU support, install JAX with CUDA as described in the [JAX docs](https://github.com/google/jax#installation) before installing VertAX.

---

## Quick Start — Forward Modeling

The simplest usage: create a periodic tissue mesh, define an energy, minimize.

```python
import jax.numpy as jnp
from jax import Array

from vertax import PbcBilevelOptimizer, PbcMesh, plot_mesh
from vertax.energy import energy_shape_factor_homo

# --- Mesh setup ---
n_cells = 100
L_box = jnp.sqrt(n_cells)
mesh = PbcMesh.periodic_voronoi_from_random_seeds(
    nb_seeds=n_cells, width=float(L_box), height=float(L_box), random_key=1
)

# --- Attach parameters (tensions, target areas, shape factors) ---
mesh.vertices_params = jnp.asarray([0.0])
mesh.edges_params    = jnp.asarray([0.0])
mesh.faces_params    = jnp.asarray([3.7])   # uniform target shape factor

# --- Define energy (wrapping the built-in shape-factor energy) ---
def energy(vertTable, heTable, faceTable, _vert_params, _he_params, face_params):
    return energy_shape_factor_homo(
        vertTable, heTable, faceTable, float(L_box), float(L_box), face_params
    )

# --- Minimize and visualize ---
optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner = energy
optimizer.inner_optimization(mesh)

mesh.save_mesh("equilibrium.npz")
plot_mesh(mesh)
```

---

## Tutorial 1 — Inverse Modeling with Periodic Boundaries

This tutorial recovers edge tensions from a target equilibrium geometry.

### Step 1 — Create and parametrize the mesh

```python
import math
import jax
import jax.numpy as jnp
from vertax import PbcMesh, plot_mesh

n_cells = 20
width = height = math.sqrt(n_cells)

mesh = PbcMesh.periodic_voronoi_from_random_seeds(
    nb_seeds=n_cells, width=width, height=height, random_key=0
)

# Random edge tensions drawn from a Gaussian
key = jax.random.PRNGKey(1)
he_params = 1.2 + 0.1 * jax.random.normal(key, shape=(mesh.nb_edges,))
mesh.edges_params    = jnp.repeat(he_params, 2)   # twin half-edges share tension
mesh.faces_params    = jnp.ones(mesh.nb_faces)     # target area = 1
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)

plot_mesh(mesh, title="Initial mesh")
```

### Step 2 — Define a custom energy function

```python
from jax import Array, vmap
from vertax.geo import get_area, get_length

MAX_EDGES_IN_FACE = 20

def get_energy_function(reference_tension: float):

    def area_part(face, face_param, vertTable, heTable, faceTable):
        a = get_area(face, vertTable, heTable, faceTable, width, height, MAX_EDGES_IN_FACE)
        return (a - face_param) ** 2

    def hedge_part(he, he_param, vertTable, heTable, faceTable):
        return he_param * get_length(he, vertTable, heTable, faceTable, width, height)

    def energy_fct(vertTable, heTable, faceTable, _vert_params, he_params, face_params):
        K_areas = 20
        areas  = vmap(lambda f, fp: area_part(f, fp, vertTable, heTable, faceTable))(
                     jnp.arange(len(faceTable)), face_params)
        hedges = vmap(lambda h, hp: hedge_part(h, hp, vertTable, heTable, faceTable))(
                     jnp.arange(2, len(heTable)), he_params[2:])
        return (2 * reference_tension * get_length(0, vertTable, heTable, faceTable, width, height)
                + jnp.sum(hedges) + 0.5 * K_areas * jnp.sum(areas))

    return energy_fct

energy = get_energy_function(float(he_params[0]))
```

### Step 3 — Inner optimization (find equilibrium)

```python
import optax
from vertax import PbcBilevelOptimizer

optimizer = PbcBilevelOptimizer()
optimizer.loss_function_inner   = energy
optimizer.inner_solver          = optax.sgd(learning_rate=0.01)
optimizer.update_T1             = True
optimizer.min_dist_T1           = 0.005
optimizer.max_nb_iterations     = 1000
optimizer.tolerance             = 1e-4
optimizer.patience              = 5

optimizer.inner_optimization(mesh)
plot_mesh(mesh, title="Mesh after inner optimization")
```

### Step 4 — Build the target and run bilevel optimization

```python
from vertax.cost import cost_v2v
from vertax import BilevelOptimizationMethod

# Create a target mesh with different tensions
target = PbcMesh.copy_mesh(mesh)
key2   = jax.random.PRNGKey(2)
he_params_target   = 1.2 + 0.1 * jax.random.normal(key2, shape=(target.nb_edges,))
target.edges_params = jnp.repeat(he_params_target, 2)

energy_target = get_energy_function(float(he_params_target[0]))
optimizer.inner_optimization(target)

# Register the target
optimizer.vertices_target       = target.vertices.copy()
optimizer.edges_target          = target.edges.copy()
optimizer.faces_target          = target.faces.copy()
optimizer.loss_function_inner   = energy_target
optimizer.loss_function_outer   = cost_v2v
optimizer.outer_solver          = optax.adam(learning_rate=1e-4, nesterov=True)
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION

# Run bilevel optimization
for epoch in range(100):
    optimizer.bilevel_optimization(mesh)
```

### Step 5 — Visualize results

```python
from vertax import EdgePlot, FacePlot, VertexPlot

plot_mesh(
    mesh,
    edge_plot=EdgePlot.EDGE_PARAMETER,   edge_parameters_name="tension",
    face_plot=FacePlot.FACE_PARAMETER,   face_parameters_name="area",
    vertex_plot=VertexPlot.INVISIBLE,
    title="Inferred tensions and areas"
)
```

---

## Tutorial 2 — Inverse Design with Bounded Boundaries

This tutorial uses a **bounded** tissue cluster and optimizes line tensions to achieve a prescribed elongation ratio (convergent extension).

### Setup

```python
from vertax import BoundedBilevelOptimizer, BoundedMesh, plot_mesh
import jax, jax.numpy as jnp, math, optax

n_cells = 20
width = height = math.sqrt(n_cells)

mesh = BoundedMesh.from_random_seeds(
    nb_seeds=n_cells, width=width, height=height, random_key=2
)

key = jax.random.PRNGKey(3)
mesh.edges_params    = 1 + jax.nn.sigmoid(jax.random.uniform(key, (mesh.nb_edges,)) * 20 - 10)
mesh.faces_params    = jnp.full(mesh.nb_faces, 0.6)   # target area = 0.6
mesh.vertices_params = jnp.zeros(mesh.nb_vertices)
```

### Custom energy for bounded meshes

```python
from jax import Array, vmap
from vertax.geo import get_area_bounded, get_edge_length, get_surface_length

def get_energy_function():

    def energy_fct(vertTable, angTable, heTable, faceTable,
                   _sel_v, _sel_he, _sel_f, _vert_params, he_params, face_params):
        vertTable = jnp.vstack([jnp.array([[0.,0.],[1.,1.]]), vertTable])
        num_edges = angTable.size
        angTable  = jnp.repeat(angTable, 2)
        he_params = jax.nn.sigmoid(he_params) + 1     # keep tensions positive

        K_areas = 20
        areas = jnp.sum(vmap(
            lambda f, fp: (get_area_bounded(f, vertTable, angTable, heTable, faceTable) - fp) ** 2
        )(jnp.arange(len(faceTable)), face_params))

        edges = jnp.arange(num_edges * 2)
        surf  = jnp.sum(vmap(
            lambda e, t: get_surface_length(e, vertTable, angTable, heTable) * t
        )(edges, jnp.repeat(he_params, 2)))

        unique = jnp.arange(num_edges) * 2
        inner = jnp.sum(vmap(
            lambda e, t: get_edge_length(e, vertTable, heTable) * t
        )(unique, he_params))

        return K_areas * areas + inner + surf

    return energy_fct
```

### Bilevel optimization toward convergent extension

```python
from vertax.cost import cost_ratio   # minimizes (a₁/a₂ − 2)²
from vertax import BilevelOptimizationMethod

optimizer = BoundedBilevelOptimizer()
optimizer.loss_function_inner         = get_energy_function()
optimizer.loss_function_outer         = cost_ratio   # drive the cluster to 2:1 aspect ratio
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
optimizer.outer_solver                = optax.adam(learning_rate=1e-4, nesterov=True)

# Reach initial equilibrium
optimizer.inner_optimization(mesh)

# Optimize tensions to achieve elongation
for epoch in range(500):
    cost = cost_ratio(mesh.vertices)
    optimizer.bilevel_optimization(mesh)
    if epoch % 50 == 0:
        print(f"epoch {epoch}: cost = {cost:.4f}")

plot_mesh(mesh, title="Convergent extension result")
```

---

## Gradient Strategies

VertAX implements and benchmarks three complementary methods for computing outer gradients through the implicit inner problem:

| Method | How it works | Pros | Cons |
|---|---|---|---|
| **AD** (Automatic Diff.) | Unrolls the inner optimization steps; forward-mode JVP via `jax.jacfwd` | Exact for differentiable pipelines; easy in JAX | Cost scales with # iterations × # parameters |
| **ID** (Implicit Diff.) | Differentiates the optimality condition ∇ₓE=0 via Implicit Function Theorem; JVP or adjoint (VJP) variant | No unrolling; constant memory; exact near equilibrium | Requires Hessian solve; sensitive to ill-conditioning |
| **EP** (Equilibrium Prop.) | Estimates gradient from perturbed free and nudged equilibria; no backprop required | Memory-efficient; works with non-differentiable/incomplete solvers | Approximate; depends on perturbation size β |

**In practice**: AD and EP converge similarly (Pearson ρ > 0.9 after ~200 epochs on synthetic benchmarks), while EP is especially valuable for non-differentiable simulators — including C++ or discrete-update solvers — since it only requires two forward passes.

Selecting the method:

```python
from vertax import BilevelOptimizationMethod

# Automatic differentiation (default)
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.AD

# Implicit differentiation — adjoint-state (VJP) mode
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.ADJOINT_STATE

# Implicit differentiation — sensitivity (JVP) mode
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.SENSITIVITY

# Equilibrium propagation
optimizer.bilevel_optimization_method = BilevelOptimizationMethod.EQUILIBRIUM_PROPAGATION
```

---

## API Reference

### Mesh classes

| Class | Description |
|---|---|
| `PbcMesh` | Periodic boundary conditions mesh |
| `BoundedMesh` | Finite cluster with curved boundary arcs |
| `PbcMesh.periodic_voronoi_from_random_seeds(nb_seeds, width, height, random_key)` | Create periodic mesh from random Voronoi seeds |
| `BoundedMesh.from_random_seeds(nb_seeds, width, height, random_key)` | Create bounded mesh from random seeds |
| `PbcMesh.copy_mesh(mesh)` | Deep copy of a mesh |
| `mesh.save_mesh(path)` | Save mesh to `.npz` |
| `PbcMesh.load_mesh(path)` | Load mesh from `.npz` |

### Optimizer classes

| Class | Description |
|---|---|
| `PbcBilevelOptimizer` | Optimizer for periodic meshes |
| `BoundedBilevelOptimizer` | Optimizer for bounded meshes |
| `.inner_optimization(mesh)` | Run energy minimization (inner problem) |
| `.bilevel_optimization(mesh)` | Run one epoch of bilevel optimization (outer problem) |

### Key optimizer attributes

| Attribute | Default | Description |
|---|---|---|
| `loss_function_inner` | — | Energy function `E(vertTable, heTable, faceTable, vert_params, he_params, face_params)` |
| `loss_function_outer` | — | Cost function `C(vertices, edges, faces, ...)` |
| `inner_solver` | `optax.sgd(lr=0.01)` | Optax optimizer for inner problem |
| `outer_solver` | `optax.adam(lr=1e-4, nesterov=True)` | Optax optimizer for outer problem |
| `bilevel_optimization_method` | `EQUILIBRIUM_PROPAGATION` | Gradient strategy |
| `update_T1` | `True` | Enable T1 topological transitions |
| `min_dist_T1` | `0.005` | Edge length threshold for T1 |
| `max_nb_iterations` | `1000` | Max inner optimization steps |
| `tolerance` | `1e-4` | Loss stagnation threshold |
| `patience` | `5` | Steps before early stopping |

### Geometry utilities (`vertax.geo`)

| Function | Description |
|---|---|
| `get_area(face, vertTable, heTable, faceTable, width, height, max_iter)` | Cell area (periodic) |
| `get_length(he, vertTable, heTable, faceTable, width, height)` | Half-edge length (periodic) |
| `get_area_bounded(face, vertTable, angTable, heTable, faceTable)` | Cell area (bounded) |
| `get_edge_length(edge, vertTable, heTable)` | Inner edge length (bounded) |
| `get_surface_length(edge, vertTable, angTable, heTable)` | Boundary arc length (bounded) |

### Built-in energy functions (`vertax.energy`)

| Energy | Formula | Parameters | Use case |
|---|---|---|---|
| **E₁** — Shape factor | `Σ_α (a_α − 1)² + Σ_α (p_α − p⁰_α)²` | Per-cell target shape factor `p⁰_α`; `a_α = A_α/A₀`, `p_α = P_α/√A₀` | Cell-scale heterogeneity, rigidity transitions |
| **E₂** — Line tension | `½K Σ_α (A_α − A⁰_α)² + Σ_{ij} γ_{ij} ℓ_{ij}` | Elastic modulus `K`, per-cell target area `A⁰_α`, per-edge tension `γ_{ij}` | Force inference, convergent extension |

Both energies can be mixed, extended, or replaced entirely with your own Python function.

### Built-in cost functions (`vertax.cost`)

| Function | Description |
|---|---|
| `cost_v2v` | Vertex-to-vertex MSE |
| `cost_mesh2image` | Mesh-to-image MSE
| `cost_ratio` | Aspect ratio cost `(a₁/a₂ − 2)²` |

Cost functions can be mixed, extended, or replaced entirely with your own Python function.

### Plotting (`vertax`)

```python
from vertax import plot_mesh, EdgePlot, FacePlot, VertexPlot

plot_mesh(
    mesh,
    edge_plot=EdgePlot.EDGE_PARAMETER,    # or INVISIBLE, DEFAULT
    face_plot=FacePlot.FACE_PARAMETER,    # or INVISIBLE, DEFAULT
    vertex_plot=VertexPlot.INVISIBLE,     # or DEFAULT
    edge_parameters_name="tension",
    face_parameters_name="area",
    title="My mesh"
)
```

---

## Data Structure

VertAX uses a **half-edge** data structure separating geometry from topology:

```
vertTable  [n_v  × 2]:   (x, y) vertex coordinates

heTable    [n_he × 8]:   prev | next | twin | source | target | face | offset_x | offset_y

faceTable  [n_f  × 1]:   index of one half-edge belonging to each face

angTable   [n_be × 1]:   arc angle per boundary edge  (bounded mode only)
```

Each edge is split into two **oppositely oriented** half-edges, enabling independent parametrization of the two cell membranes in contact. In periodic mode, half-edges crossing the box boundary carry offset flags `(-1, 0, +1)` to maintain topological continuity.

---

## Benchmarks

On synthetic inverse problems (E₁, 20–60 cells), all three gradient strategies recover ground-truth shape factors with Pearson ρ > 0.98 after 350 epochs. Runtime and memory scaling:

- **AD**: runtime scales linearly with # parameters; memory scales with inner loop depth.
- **ID**: constant memory; but runtime can be dominated by Hessian linear solve.
- **EP**: lowest memory footprint; runtime comparable to 2–3 inner passes; accuracy depends on perturbation size β.

---

## Features Summary

- ✅ Forward vertex modeling (periodic and bounded)

- ✅ Inverse vertex modeling via :
    - ⚙️ Automatic differentiation (AD, forward-mode `jacfwd`)
    - ⚙️ Implicit differentiation (adjoint-state VJP and sensitivity JVP)
    - ⚙️ Equilibrium propagation (first-order and second-order centered estimators)

- ✅ Custom energy and cost functions in plain Python

- ✅ Periodic boundary conditions from random seeds or real images

- ✅ Bounded boundary conditions from random seeds

- ✅ T1 topological transitions

- ✅ Save / load meshes (`.npz`)

- ✅ Visualization with parameter coloring

---

## Tutorials

See the [`docs/`](docs) folder for in-depth examples:

| Notebook | Description |
|---|---|
| `inverse_modelling_example.ipynb` | Inverse modeling with periodic boundary conditions |
| `inverse_modelling_example_bounded.ipynb` | Inverse design with bounded cluster (convergent extension) |

---

## Citing VertAX

If you use VertAX in your research, please cite:

> Pasqui A., Catacora Ocana J.M., Sinha A., Delbary F., Perez M., Gosti G., Miotto M., Caudo D., Ruocco G., Ernoult M.\*, Turlier H.\* (2025). *VertAX: A Differentiable Vertex Model for Learning Epithelial Tissue Mechanics.*

<!-- Repository DOI:
> vertAX contributors (2019). vertAX: a differentiable vertex model framework. [doi:10.5281/zenodo.3555620](https://zenodo.org/record/3555620) -->

---

## Funding

This project received funding from the European Union's Horizon 2020 research and innovation programme under the **European Research Council** grant agreement no. 949267, and under the **Marie Skłodowska-Curie** grant agreement No. 945304 – Cofund AI4theSciences, hosted by PSL University.

---

## License

VertAX is released under the license specified in [`LICENSE`](LICENSE). It is free and open-source software.
