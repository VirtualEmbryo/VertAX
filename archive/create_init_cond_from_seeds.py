import numpy as np 

import jax.numpy as jnp 
from jax import jit, vmap

import optax

from vertax.start import create_geograph_from_seeds, save_geograph
from vertax.geo import get_area, get_length
from vertax.opt import fire, inner_optax
from vertax.plot import plot_geograph


################
### SETTINGS ###
################

K_areas = 20
area_param = 0.6  # init cond area params 
edge_param = 0.7  # init cond edge params

min_dist_T1 = 0.05
iterations = 100


############## 
### ENERGY ### 
############## 

@jit
def area_part(face: float, area_param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    a = get_area(face, vertTable, heTable, faceTable)
    return (a - area_param) ** 2

@jit
def edge_part(edge: float, edge_param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    l = get_length(edge, vertTable, heTable, L_box)
    return edge_param * l

@jit
def energy(vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, *params: tuple):
    mapped_areas_part = lambda face, a_param: area_part(face, a_param, vertTable, heTable, faceTable)
    mapped_edges_part = lambda edge, e_param: edge_part(edge, e_param, vertTable, heTable, faceTable)
    areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)), params[0])
    edges_part = vmap(mapped_edges_part)(jnp.arange(len(heTable)), params[1])
    return  jnp.sum(edges_part) + (0.5 * K_areas) * jnp.sum(areas_part)


##################
### SIMULATION ###
##################

n_cells = 20

L_box = jnp.sqrt(n_cells)

seeds = L_box * np.random.random_sample((n_cells, 2))
vertTable_init, heTable_init, faceTable_init = create_geograph_from_seeds(seeds, show=True)

areas_params = []
for face in range(len(faceTable_init)):
    areas_params.append(area_param)
areas_params = jnp.asarray(areas_params)

edges_params = []
for edge in range(len(heTable_init)):
    edges_params.append(edge_param)
edges_params = jnp.asarray(edges_params)

# vertTable_eq, heTable_eq, faceTable_eq = fire(energy, vertTable_init, heTable_init, faceTable_init, areas_params, edges_params, iterations=iterations, min_dist_T1=min_dist_T1)
sgd = optax.sgd(learning_rate=0.01)
vertTable_eq, heTable_eq, faceTable_eq = inner_optax(vertTable_init, heTable_init, faceTable_init, areas_params, edges_params, L_in=energy, solver=sgd, iterations=iterations, min_dist_T1=min_dist_T1)

save_geograph('./initial_condition/equilibrium/', vertTable_eq, heTable_eq, faceTable_eq)

plot_geograph(vertTable_eq.astype(float), 
              heTable_eq.astype(int), 
              faceTable_eq.astype(int), 
              L_box=L_box, 
              multicolor=True, 
              lines=True, 
              vertices=False, 
              path='./initial_condition/equilibrium/', 
              name='equilibrium', 
              save=True, 
              show=True)
