import os
import time

from functools import partial

import jax.numpy as jnp
from jax import jit, jacfwd, vmap

from tqdm import trange, tqdm

from geograph import topology, geometry
from model import model_energy

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
energy_time = 500
energy_step = 0.02

with open('./settings.txt', 'w') as file:
    file.write('n_cells = ' + str(n_cells) + '\n')
    file.write('L_box = ' + str(L_box) + '\n')
    file.write('MIN_DISTANCE = ' + str(MIN_DISTANCE) + '\n')
    file.write('energy_time = ' + str(energy_time) + '\n')
    file.write('energy_step = ' + str(energy_step) + '\n')

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

#################
### GEO_GRAPH ###
#################

graph = topology(heTable, faceTable)
geo_graph = geometry(graph, vertTable, L_box=L_box)

##############
### ENERGY ###
##############

@jit
def cell_energy(face: float, param: jnp.array, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    area = get_area(face, vertTable, heTable, faceTable)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable)
    return ((area - 1) ** 2) + ((perimeter - param[0]) ** 2)

@jit
def energy(vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, params: jnp.array):
    faces = jnp.arange(len(faceTable))
    mapped_fn = lambda face, param: cell_energy(face, param, vertTable, heTable, faceTable)
    cell_energies = vmap(mapped_fn)(faces, params)
    return jnp.sum(cell_energies)

########################
### START SIMULATION ###
########################

simulation_zero = model_energy(energy, params)

print('Simulation:  # cells ' + str(n_cells) + ' --> box size ' + str(round(L_box, 3)))

energy_zero = []
shape_factor_zero = []

energy_zero.append(float(energy(geo_graph.vertTable, geo_graph.t_heTable, geo_graph.t_faceTable, params=params)))
shape_factor_zero.append(float(get_shape_factor(geo_graph.vertTable, geo_graph.t_heTable, geo_graph.t_faceTable)))

os.makedirs('./binaries/', exist_ok=True)

jnp.save('./binaries/' + str(-1) + '_simulation_vertTable', geo_graph.vertTable)
jnp.save('./binaries/' + str(-1) + '_simulation_faceTable', geo_graph.t_faceTable)
jnp.save('./binaries/' + str(-1) + '_simulation_heTable', geo_graph.t_heTable)

for dt in trange(energy_time, desc='total'):

    start = time.time()
    geo_graph.vertTable = simulation_zero.update(geo_graph.vertTable, geo_graph.t_heTable, geo_graph.t_faceTable, step=energy_step)
    print('0 update step exec time: ' + str(time.time() - start))

    start = time.time()
    geo_graph.vertTable, geo_graph.t_heTable = geo_graph.update_vertices_positions_and_offsets(geo_graph.vertTable, geo_graph.t_heTable)
    print('1 update vertices and offsets exec time: ' + str(time.time() - start))

    start = time.time()
    geo_graph.vertTable, geo_graph.t_heTable, geo_graph.t_faceTable = geo_graph.update_T1(geo_graph.vertTable, geo_graph.t_heTable, geo_graph.t_faceTable, MIN_DISTANCE=MIN_DISTANCE)
    #print(geo_graph.vertTable.shape)
    print('2 update T1 exec time: ' + str(time.time() - start))

    start = time.time()
    geo_graph.vertTable, geo_graph.t_heTable = geo_graph.update_vertices_positions_and_offsets(geo_graph.vertTable, geo_graph.t_heTable)
    print('3 update vertices and offsets exec time: ' + str(time.time() - start))

    jnp.save('./binaries/' + str(dt) + '_simulation_vertTable', geo_graph.vertTable)
    jnp.save('./binaries/' + str(dt) + '_simulation_faceTable', geo_graph.t_faceTable)
    jnp.save('./binaries/' + str(dt) + '_simulation_heTable', geo_graph.t_heTable)

    energy_zero.append(float(energy(geo_graph.vertTable, geo_graph.t_heTable, geo_graph.t_faceTable, params=params)))
    shape_factor_zero.append(float(get_shape_factor(geo_graph.vertTable, geo_graph.t_heTable, geo_graph.t_faceTable)))

open('./binaries/' + '_energy_zero.txt', "w").write('\n'.join(str(e) for e in energy_zero))
open('./binaries/' + '_shape_factor_zero.txt', "w").write('\n'.join(str(sf) for sf in shape_factor_zero))

