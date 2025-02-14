import numpy as np 

import jax
import jax.numpy as jnp 
from jax import jit, lax, jacfwd, grad

import matplotlib.pyplot as plt

from vertax.start import create_geograph_from_seeds, load_geograph
from vertax.geo import list_verts_hes_faces
from vertax.cost import gaussian_blur_line_segments, cost_mesh2image
from vertax.plot import plot_geograph

from tqdm import tqdm

import optax 


# Create target configuration
n_cells = 30
L_box = jnp.sqrt(n_cells)
seeds = L_box * np.random.random_sample((n_cells, 2))
vertTable_target, heTable_target, faceTable_target = create_geograph_from_seeds(seeds, show=True)
verts_selected, hes_selected, faces_selected = list_verts_hes_faces(vertTable_target, heTable_target, faceTable_target)  # jnp.arange(len(vertTable_target)), jnp.arange(len(heTable_target)), jnp.arange(len(faceTable_target))  # list_verts_hes_faces(vertTable_target, heTable_target, faceTable_target)

# Identify only cells inside the box
mask = hes_selected # (N,)

# Select only valid edges
starting = (vertTable_target[heTable_target[mask, 3], :2]) * 2 / L_box  # (M, 2)
ending = (vertTable_target[heTable_target[mask, 4], :2]) * 2 / L_box  # (M, 2)
he_edges = jnp.stack((starting, ending), axis=1)  # (N, 2, 2)
x = he_edges.transpose(1, 2, 0) - 1  # (2, 2, N)

# Blur target configuration
image_target=gaussian_blur_line_segments(x).real

plt.imshow(image_target[int(256/6):int(((256*2/6)*2)+(256/6)),int(256/6):int(((256*2/6)*2)+(256/6))])
plt.show()

key = jax.random.PRNGKey(0)  # random seed
noise_strength = .1 
noise = noise_strength * jax.random.normal(key, shape=(vertTable_target.shape[0], 2))

# Add noise to the target
vertTable = vertTable_target.at[:, :2].add(noise)
heTable = heTable_target.copy()
faceTable = faceTable_target.copy()

plot_geograph(vertTable, 
              heTable, 
              faceTable, 
              L_box=L_box, 
              multicolor=True, 
              lines=True, 
              vertices=False, 
              path='./', 
              name='-', 
              save=False, 
              show=True)

# Recover the target configuration

adam = optax.adam(learning_rate=0.001)
@jit
def update_step(carry, _):
    vertTable, heTable, faceTable, opt_state = carry
    jax.debug.print("Loss out: {x}", x=cost_mesh2image(vertTable, heTable, faceTable, verts_selected, hes_selected, faces_selected, image_target))
    jacforward = grad(cost_mesh2image, argnums=0)(vertTable, heTable, faceTable, verts_selected, hes_selected, faces_selected, image_target)
    updates, opt_state = adam.update(jacforward, opt_state)
    vertTable = optax.apply_updates(vertTable, updates)
    return (vertTable, heTable, faceTable, opt_state), None
opt_state = adam.init(vertTable)
(vertTable, heTable, faceTable, _), _ = lax.scan(update_step, (vertTable, heTable, faceTable, opt_state), None, length=1000)

plot_geograph(vertTable.astype(float), 
              heTable.astype(int), 
              faceTable.astype(int), 
              L_box=L_box, 
              multicolor=True, 
              lines=True, 
              vertices=False, 
              path='./', 
              name='-', 
              save=False, 
              show=True)
