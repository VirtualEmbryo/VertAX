###########
## TO DO ##
###########

# 1- implement a warning when it detect a crossing (check sum of the areas, it has to be fixed to L^2).
# 2- implement T1 only when total energy decreases, otherwise reject the T1.
# 3- try jit-ing all the functions

import os
from functools import partial
import time
import jax.numpy as jnp
import jax
from jax import jit, jacfwd, vmap
from jax.lax import while_loop
from topo_geo_graph import DVM_topology, DVM_geometry


class model:

    def __init__(self, geo_graph, energy, params):

        self.geo_graph = geo_graph
        self.energy = energy
        self.params = params

        print('\n')
        print('Simulation: # cells ' + str(len(self.geo_graph.t_faceTable)) + ' --> box size ' + str(round(jnp.sqrt(len(self.geo_graph.t_faceTable)), 3)))

    @partial(jit, static_argnums=(0,))
    def update(self, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, step: float):
        return vertTable - step * jacfwd(self.energy, argnums=1)(geo_graph, vertTable, heTable, faceTable, self.params)


##################
### SIMULATION ###
##################
path = '../../dev/scripts/'
vertTable = jnp.load(path + '/' + 'vertTable.npy')  # jnp.loadtxt(path + 'vertTable.csv', delimiter='\t', dtype=np.float64)
faceTable = jnp.load(path + '/' + 'faceTable.npy')  # jnp.loadtxt(path + 'faceTable.csv', delimiter='\t', dtype=np.int32)
heTable = jnp.load(path + '/' + 'heTable.npy')  # jnp.loadtxt(path + 'heTable.csv', delimiter='\t', dtype=np.int32)

################
### SEETINGS ###
################
n_cells = len(faceTable)
L_box = jnp.sqrt(n_cells)
MIN_DISTANCE = 0.025
cost_time = 1
energy_time = 500
cost_step = 0.5
energy_step = 0.02
beta = 0.05

with open('./settings.txt', 'w') as file:
    file.write('n_cells = ' + str(n_cells) + '\n')
    file.write('L_box = ' + str(L_box) + '\n')
    file.write('MIN_DISTANCE = ' + str(MIN_DISTANCE) + '\n')
    file.write('cost_time = ' + str(cost_time) + '\n')
    file.write('energy_time = ' + str(energy_time) + '\n')
    file.write('cost_step = ' + str(cost_step) + '\n')
    file.write('energy_step = ' + str(energy_step) + '\n')
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

########################
### START SIMULATION ###
########################
for dt in range(cost_time):

    print('dt = ' + str(dt) + ' /' + str(cost_time))

    graph = DVM_topology(heTable, faceTable)
    geo_graph = DVM_geometry(graph, vertTable, L_box=L_box)
    simulation_zero = model(geo_graph, energy, params)

    os.makedirs('./binaries/binaries_zero_' + str(dt) + '/', exist_ok=True)

    energy_zero = []
    shape_factor_zero = []

    energy_zero.append(float(energy(simulation_zero.geo_graph, simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable, params=params)))

    for di in range(energy_time):

        print('di_zero = ' + str(di) + ' /' + str(energy_time))

        # start = time.time()
        simulation_zero.geo_graph.vertTable = simulation_zero.update(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable, step=energy_step)
        # print('0 update exec time : ' + str(time.time() - start))

        # start = time.time()
        simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable = simulation_zero.geo_graph.update_vertices_positions_and_offsets(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable)
        # print('1 update vertices and offsets exec time: ' + str(time.time() - start))

        # start = time.time()
        simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable = simulation_zero.geo_graph.update_T1(MIN_DISTANCE=MIN_DISTANCE)
        # print('2 update T1 exec time: ' + str(time.time() - start))

        # start = time.time()
        simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable = simulation_zero.geo_graph.update_vertices_positions_and_offsets(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable)
        # print('3 update vertices and offsets exec time: ' + str(time.time() - start))

        jnp.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_vertTable', simulation_zero.geo_graph.vertTable)
        jnp.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_faceTable', simulation_zero.geo_graph.t_faceTable)
        jnp.save('./binaries/binaries_zero_'+str(dt)+'/' + str(di) + '_heTable', simulation_zero.geo_graph.t_heTable)

        # print(float(energy(simulation_zero.geo_graph, simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable, params=params)))

        energy_zero.append(float(energy(simulation_zero.geo_graph, simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable, params=params)))
        shape_factor_zero.append(float(simulation_zero.geo_graph.get_shape_factor(simulation_zero.geo_graph.vertTable, simulation_zero.geo_graph.t_heTable, simulation_zero.geo_graph.t_faceTable)))

        # print('\n')

    open('./binaries/binaries_zero_'+str(dt)+'/' + '_energy_zero.txt', "w").write('\n'.join(str(e) for e in energy_zero))
    open('./binaries/binaries_zero_'+str(dt)+'/' + '_shape_factor_zero.txt', "w").write('\n'.join(str(sf) for sf in shape_factor_zero))

