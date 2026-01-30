# VertAX

<!-- Project info -->

[![License](https://img.shields.io/pypi/l/vertAX.svg)](https://github.com/vertAX/vertAX/raw/main/LICENSE)
[![Python package index](https://img.shields.io/pypi/v/vertAX.svg)](https://pypi.org/project/vertAX)
[![DOI](https://zenodo.org/badge/144513571.svg)](https://zenodo.org/badge/latestdoi/144513571)

<!-- Project standards and quality  -->

[![Development Status](https://img.shields.io/pypi/status/vertAX.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Beta)

JAX-based differentiable vertex model suitable for solving inverse problems for
confluent tissues through bilevel optimization.

TODO @Alessandro : more explanations ? Images ?

## installation

For a full installation, we recommend installing vertAX into a virtual environment, like this:

```sh
python -m venv .venv
source .venv/bin/activate
pip install "vertax"
```

## simple example

TODO @Alessandro : explain what it does

```python
import jax.numpy as jnp
from jax import Array

from vertax import PbcBilevelOptimizer, PbcMesh, plot_mesh
from vertax.energy import energy_shape_factor_homo

# Settings
n_cells = 100
# Initial condition
L_box = jnp.sqrt(n_cells)
width = float(L_box)
height = float(L_box)
# Create a mesh with periodic boundary conditions
mesh = PbcMesh.periodic_voronoi_from_random_seeds(nb_seeds=n_cells, width=width, height=height, random_key=1)
# Parameters such as tensions, target areas, ... can be attached to vertices, edges, faces.
mesh.vertices_params = jnp.asarray([0.0])
mesh.edges_params = jnp.asarray([0.0])
mesh.faces_params = jnp.asarray([3.7])

def energy(
    vertTable: Array, heTable: Array, faceTable: Array, _vert_params: Array, _he_params: Array, face_params: Array
) -> Array:
    """We use an energy given in vertAX for this example.

    But only indirectly as the loss function for an inner optimization needs a specific function signature.
    """
    return energy_shape_factor_homo(vertTable, heTable, faceTable, width, height, face_params)

# Energy minimization
bilevel_optimizer = PbcBilevelOptimizer()
bilevel_optimizer.loss_function_inner = energy
bilevel_optimizer.inner_optimization(mesh)

mesh.save_mesh("mesh.npz")
plot_mesh(mesh)
```

## features

- Forward ?
- Inverse ?
- AD, ID, AS, EP...
- periodic boundary conditions from random Voronoi cells
- periodic boundary conditions from an image
- bounded boundaries from random seeds
- custom energy and cost
- plot function
- save / load meshes

## tutorials

See the [docs](docs) folder for more in-depht examples.

## citing VertAX

If you find `vertAX` useful please cite [this repository](https://github.com/vertAX/vertAX) using its DOI as follows:

> vertAX contributors (2019). vertAX: a multi-dimensional image viewer for python. [doi:10.5281/zenodo.3555620](https://zenodo.org/record/3555620)

Note this DOI will resolve to all versions of vertAX. To cite a specific version please find the
DOI of that version on our [zenodo page](https://zenodo.org/record/3555620). The DOI of the latest version is in the badge at the top of this page.

## institutional and funding partners

<a href="https://chanzuckerberg.com/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://chanzuckerberg.com/wp-content/themes/czi/img/logo-white.svg">
    <img alt="CZI logo" src="https://chanzuckerberg.com/wp-content/themes/czi/img/logo.svg">
  </picture>
</a>
