"""Simple forward test for the bounded case."""

import jax.numpy as jnp
import numpy as np
import optax

from vertax.energy import energy_bounded
from vertax.opt_bounded import inner_opt_bounded
from vertax.plot import plot_bounded_mesh
from vertax.start import create_bounded_mesh_from_seeds

# Settings
n_cells = 20
n_edges = (n_cells - 1) * 3
min_dist_T1 = 0.025
vert_params = jnp.asarray([0.0])
face_params = jnp.asarray([0.0])

# Solver
sgd = optax.sgd(learning_rate=0.01)
iterations_max = 1000
tolerance = 1e-6
patience = 5

# Initial condition
L_box = jnp.sqrt(n_cells)
rng_seed = 1  # np.random.randint(0, 2**32 - 1)
rng = np.random.default_rng(seed=rng_seed)
seeds = L_box * rng.random((n_cells, 2))
vertTable, angTable, heTable, faceTable = create_bounded_mesh_from_seeds(seeds, show=True, rng=rng)
he_params = jnp.asarray(rng.random(n_edges) * 20 - 10)

# Energy minimization
(vertTable_eq, angTable_eq, heTable_eq, faceTable_eq), energies = inner_opt_bounded(
    vertTable,
    angTable,
    heTable,
    faceTable,
    vert_params,
    he_params,
    face_params,
    energy_bounded,
    sgd,
    min_dist_T1,
    iterations_max,
    tolerance,
    patience,
)

# Plotting/saving
plot_bounded_mesh(
    vertTable_eq,
    angTable_eq,
    heTable_eq,
    faceTable_eq,
    L_box,
    multicolor=True,
    lines=True,
    vertices=False,
    path="./",
    name="forward_modeling_bounded",
    save=True,
    show=True,
)
