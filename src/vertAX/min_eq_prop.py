##########
## TODO ##
##########

# 1- implement a warning when it detect a crossing (check sum of the areas, it has to be fixed to L^2).
# 2- implement T1 only when total energy decreases, otherwise reject the T1.
# 3- try jit-ing the functions fro periodic bc and T1 transitions


import os
import time

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap

from tqdm import trange, tqdm

from geograph import topology, geometry
from model import model_eq_prop

from utils_geograph import get_area, get_perimeter, get_shape_factor


##################
### SIMULATION ###
##################

path = './initial_conditions_voronoi/'
vertTable = jnp.load(path + 'simulation_vertTable.npy')
faceTable = jnp.load(path + 'simulation_faceTable.npy')
heTable = jnp.load(path + 'simulation_heTable.npy')

################
### SEETINGS ###
################

n_cells = len(faceTable)
L_box = jnp.sqrt(n_cells)
MIN_DISTANCE = 0.025
tot_time = 1000
lagrangian_time = 150
params_step = 0.1
lagrangian_step = 0.02
#beta = 0.02
beta = 0.2

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

@jit
def cell_energy(face: float, param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    area = get_area(face, vertTable, heTable, faceTable)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable)
    #jax.debug.print("area perimeter {d} {d2}", d=area, d2=perimeter)
    return ((area - 1) ** 2) + ((perimeter - param[0]) ** 2)

@jit
def energy(vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, params: jnp.array):
    faces = jnp.arange(len(faceTable))
    mapped_fn = lambda face, param: cell_energy(face, param, vertTable, heTable, faceTable)
    cell_energies = vmap(mapped_fn)(faces, params)
    return jnp.sum(cell_energies)

############
### COST ###
############

# 1. COST FUNCTION AS THE SUM OF THE DISTANCE NORMS OF CORRESPONDING VERTICES

@jit
def min_distance(vertTable: jnp.array, vertTable_target: jnp.array, v: int, L_box: jnp.array):
    return jnp.min(jnp.array([jnp.sqrt((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2),
                              jnp.sqrt((vertTable[v][0] - (vertTable_target[v][0] + L_box)) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2),
                              jnp.sqrt((vertTable[v][0] - (vertTable_target[v][0] - L_box)) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] + L_box)) ** 2),
                              jnp.sqrt((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] - L_box)) ** 2),
                              jnp.sqrt((vertTable[v][0] - (vertTable_target[v][0] + L_box)) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] + L_box)) ** 2),
                              jnp.sqrt((vertTable[v][0] - (vertTable_target[v][0] + L_box)) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] - L_box)) ** 2),
                              jnp.sqrt((vertTable[v][0] - (vertTable_target[v][0] - L_box)) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] + L_box)) ** 2),
                              jnp.sqrt((vertTable[v][0] - (vertTable_target[v][0] - L_box)) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] - L_box)) ** 2)]))

@jit
def cost(vertTable: jnp.array, vertTable_target: jnp.array, L_box: jnp.array):
    v = jnp.arange(len(vertTable))
    mapped_fn = lambda vec: min_distance(vertTable, vertTable_target, vec, L_box)
    distances = vmap(mapped_fn)(v)
    #jax.debug.print("distances {d}", d=distances)
    return jnp.linalg.norm(distances)

##############
### TARGET ###
##############

path = './initial_conditions_voronoi/'
vertTable_target = jnp.load(path + 'target_vertTable.npy')
faceTable_target = jnp.load(path + 'target_faceTable.npy')
heTable_target = jnp.load(path + 'target_heTable.npy')

graph_target = topology(heTable_target, faceTable_target)
geo_graph_target = geometry(graph_target, vertTable_target, L_box=L_box)

########################
### START SIMULATION ###
########################

print('Simulation:  # cells ' + str(n_cells) + ' --> box size ' + str(round(L_box, 3)))

parameters = []

for dt in trange(tot_time, desc='total'):

    # tqdm.write('dt = ' + str(dt) + ' /' + str(tot_time))
    # trange(tot_time, desc='total').set_description(str(params[2][0]))

    tqdm.write(str(params))

    parameters.append('\t'.join(map(str, params.flatten())))

    graph_zero = topology(heTable, faceTable)
    geo_graph_zero = geometry(graph_zero, vertTable, L_box=L_box)

    simulation_zero = model_eq_prop(energy, cost, L_box, params)

    os.makedirs('./binaries/binaries_zero_' + str(dt) + '/', exist_ok=True)

    energy_zero = []
    cost_zero = []
    lagrangian_zero = []
    shape_factor_zero = []
    lagrangian_step_zero = lagrangian_step

    energy_zero.append(float(energy(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable, params=params)))
    cost_zero.append(float(cost(geo_graph_zero.vertTable, vertTable_target, L_box)))
    lagrangian_zero.append(float(simulation_zero.lagrangian(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable, vertTable_target, params, beta=0.)))
    shape_factor_zero.append(float(get_shape_factor(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable)))

    for di in trange(lagrangian_time, desc='zero', leave=False):

        # tqdm.write('di_zero = ' + str(di) + ' /' + str(lagrangian_time))

        geo_graph_zero.vertTable = simulation_zero.update_lagrangian(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable, vertTable_target, params, beta=0., step=lagrangian_step_zero)

        geo_graph_zero.vertTable, geo_graph_zero.t_heTable = geo_graph_zero.update_vertices_positions_and_offsets(geo_graph_zero.vertTable, geo_graph_zero.t_heTable)
        geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable = geo_graph_zero.update_T1(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable, MIN_DISTANCE=MIN_DISTANCE)
        geo_graph_zero.vertTable, geo_graph_zero.t_heTable = geo_graph_zero.update_vertices_positions_and_offsets(geo_graph_zero.vertTable, geo_graph_zero.t_heTable)

        jnp.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_vertTable', geo_graph_zero.vertTable)
        jnp.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_faceTable', geo_graph_zero.t_faceTable)
        jnp.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_heTable', geo_graph_zero.t_heTable)

        energy_zero.append(float(energy(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable, params=params)))
        cost_zero.append(float(cost(geo_graph_zero.vertTable, vertTable_target, L_box)))
        lagrangian_zero.append(float(simulation_zero.lagrangian(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable, vertTable_target, params, beta=0.)))
        shape_factor_zero.append(float(get_shape_factor(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable)))

    open('./binaries/binaries_zero_'+str(dt)+'/' + '_energy_zero.txt', "w").write('\n'.join(str(e) for e in energy_zero))
    open('./binaries/binaries_zero_'+str(dt)+'/' + '_cost_zero.txt', "w").write('\n'.join(str(e) for e in cost_zero))
    open('./binaries/binaries_zero_'+str(dt)+'/' + '_lagrangian_zero.txt', "w").write('\n'.join(str(e) for e in lagrangian_zero))
    open('./binaries/binaries_zero_'+str(dt)+'/' + '_shape_factor_zero.txt', "w").write('\n'.join(str(e) for e in shape_factor_zero))

    graph_beta = topology(heTable, faceTable)
    geo_graph_beta = geometry(graph_beta, vertTable, L_box=L_box)

    simulation_beta = model_eq_prop(energy, cost, L_box, params)

    os.makedirs('./binaries/binaries_beta_' + str(dt) + '/', exist_ok=True)

    energy_beta = []
    cost_beta = []
    lagrangian_beta = []
    shape_factor_beta = []
    lagrangian_step_beta = lagrangian_step

    energy_beta.append(float(energy(geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable, params=params)))
    cost_beta.append(float(cost(geo_graph_beta.vertTable, vertTable_target, L_box)))
    lagrangian_beta.append(float(simulation_beta.lagrangian(geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable, vertTable_target, params, beta=beta)))
    shape_factor_beta.append(float(get_shape_factor(geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable)))

    for dj in trange(lagrangian_time, desc='beta', leave=False):

        # tqdm.write('dj_beta = ' + str(dj) + ' /' + str(lagrangian_time))

        geo_graph_beta.vertTable = simulation_beta.update_lagrangian(geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable, vertTable_target, params, beta=beta, step=lagrangian_step_zero)

        geo_graph_beta.vertTable, geo_graph_beta.t_heTable = geo_graph_beta.update_vertices_positions_and_offsets(geo_graph_beta.vertTable, geo_graph_beta.t_heTable)
        geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable = geo_graph_beta.update_T1(geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable, MIN_DISTANCE=MIN_DISTANCE)
        geo_graph_beta.vertTable, geo_graph_beta.t_heTable = geo_graph_beta.update_vertices_positions_and_offsets(geo_graph_beta.vertTable, geo_graph_beta.t_heTable)

        jnp.save('./binaries/binaries_beta_'+str(dt)+'/' + str(dj) + '_vertTable', geo_graph_beta.vertTable)
        jnp.save('./binaries/binaries_beta_'+str(dt)+'/' + str(dj) + '_faceTable', geo_graph_beta.t_faceTable)
        jnp.save('./binaries/binaries_beta_'+str(dt)+'/' + str(dj) + '_heTable', geo_graph_beta.t_heTable)

        energy_beta.append(float(energy(geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable, params=params)))
        cost_beta.append(float(cost(geo_graph_beta.vertTable, vertTable_target, L_box)))
        lagrangian_beta.append(float(simulation_beta.lagrangian(geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable, vertTable_target, params, beta=beta)))
        shape_factor_beta.append(float(get_shape_factor(geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable)))

    open('./binaries/binaries_beta_' + str(dt) + '/' + '_energy_beta.txt', "w").write('\n'.join(str(e) for e in energy_beta))
    open('./binaries/binaries_beta_' + str(dt) + '/' + '_cost_beta.txt', "w").write('\n'.join(str(e) for e in cost_beta))
    open('./binaries/binaries_beta_' + str(dt) + '/' + '_lagrangian_beta.txt', "w").write('\n'.join(str(e) for e in lagrangian_beta))
    open('./binaries/binaries_beta_' + str(dt) + '/' + '_shape_factor_beta.txt', "w").write('\n'.join(str(e) for e in shape_factor_beta))

    params = simulation_zero.update_params(geo_graph_zero.vertTable, geo_graph_zero.t_heTable, geo_graph_zero.t_faceTable, geo_graph_beta.vertTable, geo_graph_beta.t_heTable, geo_graph_beta.t_faceTable, vertTable_target, params, beta=beta, step=params_step)

    if dt % 10 == 0:
        with open('parameters.txt', 'a') as file:
            for line in parameters:
                file.write(line + '\n')
        parameters = []

