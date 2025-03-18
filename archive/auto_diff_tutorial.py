import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np 

import jax.numpy as jnp 
from jax import jit, vmap

import optax 

from vertax.start import load_geograph
from vertax.geo import get_area, get_length
from vertax.opt import fire, inner_optax, outer_optax
from vertax.cost import cost_v2v
from vertax.plot import plot_geograph


## SETTINGS

K_areas = 20
area_param_init = 0.6  # 0.6  # init cond area params 
edge_param_init = 1.5  # 0.7  # init cond edge params

min_dist_T1 = 0.05
iterations = 100
epochs = 50


## ENERGY 

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


## SIMULATION 

vertTable_init, heTable_init, faceTable_init = load_geograph('./good_initial_condition0/simulation/')
vertTable_target, heTable_target, faceTable_target = load_geograph('./good_initial_condition0/equilibrium/')

L_box = jnp.sqrt(len(faceTable_init))

areas_params = []
for face in range(len(faceTable_init)):
    areas_params.append(area_param_init)
areas_params = jnp.asarray(areas_params)

edges_params = []
for edge in range(len(heTable_init)):
    edges_params.append(edge_param_init)
edges_params = jnp.asarray(edges_params)


## INNER PROCESS

# vertTable_eq, heTable_eq, faceTable_eq = fire(energy, vertTable_init, heTable_init, faceTable_init, areas_params, edges_params, iterations=iterations, min_dist_T1=min_dist_T1)

sgd = optax.sgd(learning_rate=0.01)
sign_sgd = optax.sign_sgd(learning_rate=0.01)
adam = optax.adam(learning_rate=0.0001)

vertTable_eq, heTable_eq, faceTable_eq = inner_optax(vertTable_init, 
                                                     heTable_init, 
                                                     faceTable_init, 
                                                     areas_params, 
                                                     edges_params, 
                                                     L_in=energy, 
                                                     solver=sgd, 
                                                     iterations=iterations, 
                                                     min_dist_T1=min_dist_T1)

# plot initial equilibrium cofiguration 
plot_geograph(vertTable_eq.astype(float), 
              heTable_eq.astype(int), 
              faceTable_eq.astype(int), 
              L_box=L_box, 
              multicolor=True, 
              lines=True, 
              vertices=False, 
              path='./', 
              name='-', 
              save=False, 
              show=True)

# plot target configuration
plot_geograph(vertTable_target.astype(float), 
              heTable_target.astype(int), 
              faceTable_target.astype(int), 
              L_box=L_box, 
              multicolor=True, 
              lines=True, 
              vertices=False, 
              path='./', 
              name='-', 
              save=False, 
              show=True)


## OUTER PROCESS

loss_inner, loss_outer = [], []

for _ in tqdm(range(epochs)):

    # print(cost_v2v(vertTable_eq, heTable_eq, faceTable_eq, vertTable_target))

    loss_inner.append(energy(vertTable_eq, heTable_eq, faceTable_eq, areas_params, edges_params))
    loss_outer.append(cost_v2v(vertTable_eq, heTable_eq, faceTable_eq, vertTable_target))

    areas_params, edges_params = outer_optax(areas_params, 
                                             edges_params, 
                                             vertTable_eq, 
                                             heTable_eq, 
                                             faceTable_eq, 
                                             vertTable_target, 
                                             energy, cost_v2v, 
                                             sgd, adam, 
                                             min_dist_T1=min_dist_T1, 
                                             iterations=iterations)

    vertTable_eq, heTable_eq, faceTable_eq = inner_optax(vertTable_init, 
                                                         heTable_init, 
                                                         faceTable_init, 
                                                         areas_params, 
                                                         edges_params, 
                                                         L_in=energy, 
                                                         solver=sgd, 
                                                         iterations=iterations, 
                                                         min_dist_T1=min_dist_T1)

# plot final equilibrium cofiguration 
plot_geograph(vertTable_eq, 
              heTable_eq, 
              faceTable_eq, 
              L_box=L_box, 
              multicolor=True, 
              lines=True, 
              vertices=False, 
              path='./', 
              name='-', 
              save=False, 
              show=True)

# plot loss_inner and loss_outer
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(loss_inner, color='blue')
ax[0].set_ylabel('Energy')
ax[0].set_title('AUTO DIFF')
ax[1].plot(loss_outer, color='red')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Cost')
plt.show()
