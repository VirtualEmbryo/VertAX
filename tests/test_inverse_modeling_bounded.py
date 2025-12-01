"""Inverse modelling test for the bounded case."""

# Package imports
import jax.numpy as jnp
import numpy as np
import optax

from vertax.cost import cost_ratio
from vertax.energy import energy_bounded
from vertax.opt_bounded import bilevel_opt_bounded, inner_opt_bounded
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
adam = optax.adam(learning_rate=0.01, nesterov=True)
iterations_max = 1000
tolerance = 1e-6
patience = 5
epochs = 5

# Initial condition
L_box = jnp.sqrt(n_cells)
rng_seed = 1  # np.random.randint(0, 2**32 - 1)
rng = np.random.default_rng(seed=rng_seed)
seeds = L_box * rng.random((n_cells, 2))
vertTable, angTable, heTable, faceTable = create_bounded_mesh_from_seeds(seeds, show=True, rng=rng)
he_params = jnp.asarray(rng.random(n_edges) * 20 - 10)

# Energy minimization
(vertTable, angTable, heTable, faceTable), energies = inner_opt_bounded(
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
    vertTable,
    angTable,
    heTable,
    faceTable,
    L_box,
    multicolor=True,
    lines=True,
    vertices=False,
    path="./",
    name="forward_modeling_bounded",
    save=True,
    show=True,
)

for j in range(epochs + 1):
    print("epoch: " + str(j) + "/" + str(epochs) + "\t cost: " + str(cost_ratio(vertTable)))

    (vertTable, angTable, heTable, faceTable, vert_params, he_params, face_params), cost = bilevel_opt_bounded(
        vertTable,
        angTable,
        heTable,
        faceTable,
        vert_params,
        he_params,
        face_params,
        None,
        None,
        None,
        None,
        L_in=energy_bounded,
        L_out=cost_ratio,
        solver_inner=sgd,
        solver_outer=adam,
        min_dist_T1=min_dist_T1,
        iterations_max=iterations_max,
        tolerance=tolerance,
        patience=patience,
        selected_verts=None,
        selected_hes=None,
        selected_faces=None,
        image_target=None,
        beta=None,
        method="ad",
    )

    if j % 100 == 0:
        # Plotting/saving
        plot_bounded_mesh(
            vertTable,
            angTable,
            heTable,
            faceTable,
            L_box,
            multicolor=True,
            lines=True,
            vertices=False,
            path="./",
            name="bilevel_result_" + str(j),
            save=True,
            show=True,
        )
