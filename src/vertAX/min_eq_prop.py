###########
## TO DO ##
###########

# 1- implement a warning when it detect a crossing (check sum of the areas, it has to be fixed to L^2).
# 2- implement T1 only when total energy decreases, otherwise reject the T1.
# 3- try jit-ing the functions fro periodic bc and T1 transitions


import os
from functools import partial
import time
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, jacfwd, vmap
from jax.lax import while_loop
from topo_geo_graph import DVM_topology, DVM_geometry
from tqdm import trange, tqdm


class model:

    def __init__(self, geo_graph, energy, cost, params):

        self.geo_graph = geo_graph
        self.energy = energy
        self.cost = cost
        self.params = params
        self.L_box = jnp.sqrt(len(geo_graph.t_faceTable))

    @partial(jit, static_argnums=(0,))
    def update_energy(self, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, step: float):
        return vertTable - step * jacfwd(self.energy, argnums=1)(geo_graph, vertTable, heTable, faceTable, self.params)

    @partial(jit, static_argnums=(0,))
    def lagrangian(self, vertTable: jnp.array, vertTable_target: jnp.array, params: jnp.array, beta: float):
        return energy(geo_graph, vertTable, heTable, faceTable, params) + beta * cost(vertTable, vertTable_target, self.L_box)

    @partial(jit, static_argnums=(0,))
    def update_lagrangian(self, vertTable: jnp.array, vertTable_target: jnp.array, params: jnp.array, beta: float, step: float):
        return vertTable - step * jacfwd(self.lagrangian, argnums=0)(vertTable, vertTable_target, params, beta)

    @partial(jit, static_argnums=(0,))
    def update_params(self, vertTable_zero: jnp.array, vertTable_beta: jnp.array, vertTable_target: jnp.array, params: jnp.array, beta: float, step: float):
        return params - step * ((1/beta) * (jacfwd(self.lagrangian, argnums=2)(vertTable_beta, vertTable_target, params, beta) - jacfwd(self.lagrangian, argnums=2)(vertTable_zero, vertTable_target, params, beta=0.)))


##################
### SIMULATION ###
##################

path = '../../dev/scripts/'
vertTable = jnp.load(path + 'vertTable.npy')  # np.loadtxt(path + 'vertTable.csv', delimiter='\t', dtype=np.float64)
faceTable = jnp.load(path + 'faceTable.npy')  # np.loadtxt(path + 'faceTable.csv', delimiter='\t', dtype=np.int32)
heTable = jnp.load(path + 'heTable.npy')  # np.loadtxt(path + 'heTable.csv', delimiter='\t', dtype=np.int32)

################
### SEETINGS ###
################

n_cells = len(faceTable)
L_box = jnp.sqrt(n_cells)
MIN_DISTANCE = 0.02
tot_time = 20
lagrangian_time = 200
params_step = 0.05
lagrangian_step = 0.02
beta = 0.02

with open('./settings.txt', 'w') as file:
    file.write('n_cells = ' + str(n_cells) + '\n')
    file.write('L_box = ' + str(L_box) + '\n')
    file.write('MIN_DISTANCE = ' + str(MIN_DISTANCE) + '\n')
    file.write('tot_time = ' + str(tot_time) + '\n')
    file.write('lagrangian_time = ' + str(lagrangian_time) + '\n')
    file.write('params_step = ' + str(params_step) + '\n')
    file.write('lagrangian_step = ' + str(lagrangian_step) + '\n')
    file.write('beta = ' + str(beta) + '\n')

##################
### PARAMETERS ###
##################

param_hexagonal_cell = jnp.array([2 * (2 ** 0.5) * (3 ** 0.25)])
# param_pentagonal_cell = 2 * (5 ** 0.5) * ((5 - (2 * (5 ** 0.5))) ** 0.25)
param_liquid_cell = jnp.array([4.1])  # 4.8 was too high

params = []
for face in range(len(faceTable)):
    if face == 2:
        params.append(param_liquid_cell)  # param_liquid_cell
    else:
        params.append(param_hexagonal_cell)
params = jnp.array(params)

##############
### ENERGY ###
##############

@partial(jit, static_argnums=(0,))
def cell_energy(geo_graph, face: float, param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    area = geo_graph.get_area(face, vertTable, heTable, faceTable)
    perimeter = geo_graph.get_perimeter(face, vertTable, heTable, faceTable)
    return ((area - 1) ** 2) + ((perimeter - param[0]) ** 2)

@partial(jit, static_argnums=(0,))
def energy(geo_graph, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, params: jnp.array):
    faces = jnp.arange(len(faceTable))
    mapped_fn = lambda face, param: cell_energy(geo_graph, face, param, vertTable, heTable, faceTable)
    cell_energies = vmap(mapped_fn)(faces, params)
    return jnp.sum(cell_energies)

############
### COST ###
############

# 1. COST FUNCTION AS THE SUM OF THE DISTANCE NORMS OF CORRESPONDING VERTICES

@jit
def min_distance(vertTable: jnp.array, vertTable_target: jnp.array, v: int, L_box: jnp.array):
    return jnp.min(jnp.array([jnp.sqrt((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0] + L_box) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0] - L_box) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - vertTable_target[v][1] + L_box) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - vertTable_target[v][1] - L_box) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0] + L_box) ** 2 + (vertTable[v][1] - vertTable_target[v][1] + L_box) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0] + L_box) ** 2 + (vertTable[v][1] - vertTable_target[v][1] - L_box) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0] - L_box) ** 2 + (vertTable[v][1] - vertTable_target[v][1] + L_box) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0] - L_box) ** 2 + (vertTable[v][1] - vertTable_target[v][1] - L_box) ** 2)]))

@jit
def cost(vertTable: jnp.array, vertTable_target: jnp.array, L_box: jnp.array):
    v = jnp.arange(len(vertTable))
    mapped_fn = lambda vec: min_distance(vertTable, vertTable_target, vec, L_box)
    distances = vmap(mapped_fn)(v)
    return jnp.linalg.norm(distances)

##############
### TARGET ###
##############

path = '../../dev/scripts/'
vertTable_target = jnp.load(path + 'target_vertTable.npy')
faceTable_target = jnp.load(path + 'target_faceTable.npy')
heTable_target = jnp.load(path + 'target_heTable.npy')

graph_target = DVM_topology(heTable_target, faceTable_target)
geo_graph_target = DVM_geometry(graph_target, vertTable_target, L_box=L_box)

########################
### START SIMULATION ###
########################

print('Simulation:  # cells ' + str(n_cells) + ' --> box size ' + str(round(L_box, 3)))

parameter = []

for dt in trange(tot_time, desc='total'):

    # tqdm.write('dt = ' + str(dt) + ' /' + str(tot_time))
    # tqdm.write(str(params))

    parameter.append(params[2][0])

    graph = DVM_topology(heTable, faceTable)
    geo_graph = DVM_geometry(graph, vertTable, L_box=L_box)
    simulation_zero = model(geo_graph, energy, cost, params)

    os.makedirs('./binaries/binaries_zero_' + str(dt) + '/', exist_ok=True)

    energy_zero = []
    cost_zero = []
    lagrangian_zero = []
    shape_factor_zero = []

    energy_zero.append(float(energy(simulation_zero.geo_graph, simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable, params=params)))
    cost_zero.append(float(cost(simulation_zero.geo_graph.vertTable, vertTable_target, L_box)))
    lagrangian_zero.append(float(simulation_zero.lagrangian(simulation_zero.geo_graph.vertTable, vertTable_target, params, beta=0.)))
    shape_factor_zero.append(float(simulation_zero.geo_graph.get_shape_factor(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable)))

    for di in trange(lagrangian_time, desc='zero', leave=False):

        # tqdm.write('di_zero = ' + str(di) + ' /' + str(lagrangian_time))

        simulation_zero.geo_graph.vertTable = simulation_zero.update_lagrangian(simulation_zero.geo_graph.vertTable, vertTable_target, params, beta=0., step=lagrangian_step)
        simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable = simulation_zero.geo_graph.update_vertices_positions_and_offsets(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable)
        simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable = simulation_zero.geo_graph.update_T1(MIN_DISTANCE=MIN_DISTANCE)
        simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable = simulation_zero.geo_graph.update_vertices_positions_and_offsets(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable)
        tqdm.write(str(round(simulation_zero.geo_graph.check_crossings(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable), 1)))

        np.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_vertTable', simulation_zero.geo_graph.vertTable)
        np.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_faceTable', simulation_zero.geo_graph.t_faceTable)
        np.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_heTable', simulation_zero.geo_graph.t_heTable)

        energy_zero.append(float(energy(simulation_zero.geo_graph, simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable, params=params)))
        cost_zero.append(float(cost(simulation_zero.geo_graph.vertTable, vertTable_target, L_box)))
        lagrangian_zero.append(float(simulation_zero.lagrangian(simulation_zero.geo_graph.vertTable, vertTable_target, params, beta=0.)))
        shape_factor_zero.append(float(simulation_zero.geo_graph.get_shape_factor(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable)))

    open('./binaries/binaries_zero_'+str(dt)+'/' + '_energy_zero.txt', "w").write('\n'.join(str(e) for e in energy_zero))
    open('./binaries/binaries_zero_'+str(dt)+'/' + '_cost_zero.txt', "w").write('\n'.join(str(e) for e in cost_zero))
    open('./binaries/binaries_zero_'+str(dt)+'/' + '_lagrangian_zero.txt', "w").write('\n'.join(str(e) for e in lagrangian_zero))
    open('./binaries/binaries_zero_'+str(dt)+'/' + '_shape_factor_zero.txt', "w").write('\n'.join(str(e) for e in shape_factor_zero))

    graph = DVM_topology(heTable, faceTable)
    geo_graph = DVM_geometry(graph, vertTable, L_box=L_box)
    simulation_beta = model(geo_graph, energy, cost, params)

    os.makedirs('./binaries/binaries_beta_' + str(dt) + '/', exist_ok=True)

    energy_beta = []
    cost_beta = []
    lagrangian_beta = []
    shape_factor_beta = []

    energy_beta.append(float(energy(simulation_beta.geo_graph, simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable, simulation_beta.geo_graph.t_faceTable, params=params)))
    cost_beta.append(float(cost(simulation_beta.geo_graph.vertTable, vertTable_target, L_box)))
    lagrangian_beta.append(float(simulation_beta.lagrangian(simulation_beta.geo_graph.vertTable, vertTable_target, params, beta=beta)))
    shape_factor_beta.append(float(simulation_beta.geo_graph.get_shape_factor(simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable, simulation_beta.geo_graph.t_faceTable)))

    for dj in trange(lagrangian_time, desc='beta', leave=False):

        # tqdm.write('dj_beta = ' + str(dj) + ' /' + str(lagrangian_time))

        simulation_beta.geo_graph.vertTable = simulation_beta.update_lagrangian(simulation_beta.geo_graph.vertTable, vertTable_target, params, beta=beta, step=lagrangian_step)
        simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable = simulation_beta.geo_graph.update_vertices_positions_and_offsets(simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable)
        simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable, simulation_beta.geo_graph.t_faceTable = simulation_beta.geo_graph.update_T1(MIN_DISTANCE=MIN_DISTANCE)
        simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable = simulation_beta.geo_graph.update_vertices_positions_and_offsets(simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable)

        np.save('./binaries/binaries_beta_'+str(dt)+'/' + str(dj) + '_vertTable', simulation_beta.geo_graph.vertTable)
        np.save('./binaries/binaries_beta_'+str(dt)+'/' + str(dj) + '_faceTable', simulation_beta.geo_graph.t_faceTable)
        np.save('./binaries/binaries_beta_'+str(dt)+'/' + str(dj) + '_heTable', simulation_beta.geo_graph.t_heTable)

        energy_beta.append(float(energy(simulation_beta.geo_graph, simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable, simulation_beta.geo_graph.t_faceTable, params=params)))
        cost_beta.append(float(cost(simulation_beta.geo_graph.vertTable, vertTable_target, L_box)))
        lagrangian_beta.append(float(simulation_beta.lagrangian(simulation_beta.geo_graph.vertTable, vertTable_target, params, beta=beta)))
        shape_factor_beta.append(float(simulation_beta.geo_graph.get_shape_factor(simulation_beta.geo_graph.vertTable, simulation_beta.geo_graph.t_heTable, simulation_beta.geo_graph.t_faceTable)))

    open('./binaries/binaries_beta_' + str(dt) + '/' + '_energy_beta.txt', "w").write('\n'.join(str(e) for e in energy_beta))
    open('./binaries/binaries_beta_' + str(dt) + '/' + '_cost_beta.txt', "w").write('\n'.join(str(e) for e in cost_beta))
    open('./binaries/binaries_beta_' + str(dt) + '/' + '_lagrangian_beta.txt', "w").write('\n'.join(str(e) for e in lagrangian_beta))
    open('./binaries/binaries_beta_' + str(dt) + '/' + '_shape_factor_beta.txt', "w").write('\n'.join(str(e) for e in shape_factor_beta))

    params = simulation_zero.update_params(simulation_zero.geo_graph.vertTable, simulation_beta.geo_graph.vertTable, vertTable_target, params, beta=beta, step=params_step)

open('./parameter.txt', "w").write('\n'.join(str(e) for e in parameter))
