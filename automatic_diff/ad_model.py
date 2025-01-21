import os 

from functools import partial 

import numpy as np 

import jax.numpy as jnp 
from jax import jit, jacfwd, vmap, lax 

import optax 

from vertax.start import create_geograph
from vertax.geo import get_area, get_length
from vertax.plot import plot_geograph

NEW_GRAPH = True

##################
### SIMULATION ###
##################

if NEW_GRAPH == True:

    n_cells = 20
    L_box = jnp.sqrt(n_cells)

    seeds = L_box * np.random.random_sample((n_cells, 2))

    vertTable, faceTable, heTable = create_geograph(seeds, show=True)

else:

    vertTable, faceTable, heTable = load_geograph(path = '../initial_condition/eq_simulation/')

# path = './initial_conditions/'

# vertTable = jnp.load(path + 'eq_simulation_vertTable.npy')
# faceTable = jnp.load(path + 'eq_simulation_faceTable.npy')
# heTable = jnp.load(path + 'eq_simulation_heTable.npy')

# selected_vertices = jnp.array(
#     list(
#         {16, 24, 31, 32, 33, 17, 35, 34, 44, 40, 37, 36, 33, 32, 58, 45, 44, 34, 44, 45, 46, 47, 41, 40, 32, 31, 30, 59, 60, 58, 58, 60, 
#         67, 68, 54, 46, 45, 47, 46, 54, 55, 56, 57, 48, 59, 65, 66, 67, 60, 67, 66, 85, 81, 82, 69, 68, 54, 68, 69, 70, 55, 55, 70, 71, 
#         72, 119, 56, 57, 56, 119, 73}))

# selected_faces = jnp.array([7, 12, 13, 14, 18, 19, 20, 24, 27, 33, 37, 38])

# selected_hes = jnp.array(
#     [42, 43, 44, 45, 46, 47, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 108, 109, 110, 111,112, 113, 114, 
#     115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 146, 147, 148, 149, 150, 162, 163, 164, 165, 166, 167, 168, 198, 
#     199, 200, 201, 202, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230])

# vertTable_selected = vertTable[selected_vertices]
# faceTable_selected = faceTable[selected_faces]
# heTable_selected = heTable[selected_hes]

################
### SEETINGS ###
################

K_areas = 20
outer_epochs = 200
inner_epochs = 100
inner_lr = 0.01
outer_lr = 0.01

with open('./ad_model_settings.txt', 'w') as file:
    file.write('n_cells = ' + str(n_cells) + '\n')
    file.write('L_box = ' + str(L_box) + '\n')
    file.write('K_areas = ' + str(K_areas) + '\n')
    file.write('outer_epochs = ' + str(outer_epochs) + '\n')
    file.write('inner_epochs = ' + str(inner_epochs) + '\n')
    file.write('inner_lr = ' + str(inner_lr) + '\n')
    file.write('outer_lr = ' + str(outer_lr) + '\n')

##################
### PARAMETERS ###
##################

area_param = 0.6  # initial condition areas parameters 
edge_param = 0.7  # initial condition edges parameters 

# areas_params = []
# for face in range(len(faceTable_selected)):
#     areas_params.append(area_param)
# areas_params = jnp.asarray(areas_params)

# edges_params = []
# for edge in range(len(heTable_selected)):
#     edges_params.append(edge_param)
# edges_params = jnp.asarray(edges_params)

areas_params = []
for face in range(len(faceTable)):
    areas_params.append(area_param)
areas_params = jnp.asarray(areas_params)

edges_params = []
for edge in range(len(heTable)):
    edges_params.append(edge_param)
edges_params = jnp.asarray(edges_params)

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
def energy(vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, areas_params: jnp.array, edges_params: jnp.array, selected_faces, selected_hes):
    
    mapped_areas_part = lambda face, a_param: area_part(face, a_param, vertTable, heTable, faceTable)
    mapped_edges_part = lambda edge, e_param: edge_part(edge, e_param, vertTable, heTable, faceTable)
    areas_part = vmap(mapped_areas_part)(selected_faces, areas_params)
    edges_part = vmap(mapped_edges_part)(selected_hes, edges_params)
    
    return  jnp.sum(edges_part) + (0.5 * K_areas) * jnp.sum(areas_part)

##############
### TARGET ###
##############

path = './initial_conditions/'

vertTable_target, heTable_target, faceTable_target = load_geograph(path)
# vertTable_target = np.load(path + 'target_vertTable.npy')
# faceTable_target = jnp.load(path + 'target_faceTable.npy')
# heTable_target = jnp.load(path + 'target_heTable.npy')

vertTable_target[:, :2] = (vertTable_target[:, :2] / 1024.) * L_box
vertTable_target = jnp.array(vertTable_target)

############
### COST ###
############

@jit
def squared_distance(vertTable: jnp.array, vertTable_target: jnp.array, v: int):
    
    return (vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2

@jit
def cost(vertTable: jnp.array,
         vertTable_target: jnp.array,
         selected_vertices):
    
    mapped_fn = lambda vec: (squared_distance(vertTable, vertTable_target, vec))
    distances = vmap(mapped_fn)(selected_vertices)
    return (1./(2*len(distances))) * jnp.sum(distances)

#####################
### INNER PROCESS ###
#####################

def inner(vertTable: jnp.array, 
          heTable: jnp.array, 
          faceTable: jnp.array, 
          areas_params: jnp.array, 
          edges_params: jnp.array, 
          selected_vertices: jnp.array, 
          selected_faces: jnp.array, 
          selected_hes: jnp.array,
          solver_inner, 
          inner_epochs: int
          ):

    @jit
    def update_step(carry, _):
        vertTable, opt_state = carry
        jacforward = jacfwd(energy, argnums=0)(vertTable, heTable, faceTable, areas_params, edges_params, selected_faces, selected_hes)
        updates, opt_state = solver_inner.update(jacforward, opt_state)
        vertTable = optax.apply_updates(vertTable, updates)
        return (vertTable, opt_state), None

    # initialize the optimizer state
    opt_state = solver_inner.init(vertTable)

    # lax.scan to apply the update step inner_time times
    (vertTable, _), _ = lax.scan(update_step, (vertTable, opt_state), None, length=inner_epochs)

    return vertTable

#####################
### OUTER PROCESS ###
#####################

def outer(areas_params: jnp.array, 
          edges_params: jnp.array,
          vertTable: jnp.array, 
          heTable: jnp.array, 
          faceTable: jnp.array, 
          vertTable_target: jnp.array,
          selected_vertices: jnp.array, 
          selected_faces: jnp.array, 
          selected_hes: jnp.array,
          solver_inner,
          inner_epochs: int
          ):

    vertTable = inner(vertTable, heTable, faceTable, areas_params, edges_params, selected_vertices, selected_faces, selected_hes, solver_inner, inner_epochs)

    cost_value = cost(vertTable, vertTable_target, selected_vertices)

    return cost_value

def update_params(areas_params: jnp.array, 
                  edges_params: jnp.array, 
                  vertTable: jnp.array, 
                  heTable: jnp.array, 
                  faceTable: jnp.array, 
                  vertTable_target: jnp.array,
                  selected_vertices: jnp.array, 
                  selected_faces: jnp.array, 
                  selected_hes: jnp.array, 
                  solver_inner,
                  solver_outer, 
                  inner_epochs: int
                  ):

    grad_areas = jnp.sign(jacfwd(outer, argnums=0)(areas_params, 
                                          edges_params, 
                                          vertTable, 
                                          heTable, 
                                          faceTable, 
                                          vertTable_target, 
                                          selected_vertices, 
                                          selected_faces, 
                                          selected_hes, 
                                          solver_inner, 
                                          inner_epochs))
    
    grad_edges = jnp.sign(jacfwd(outer, argnums=1)(areas_params, 
                                          edges_params, 
                                          vertTable, 
                                          heTable, 
                                          faceTable, 
                                          vertTable_target, 
                                          selected_vertices, 
                                          selected_faces, 
                                          selected_hes, 
                                          solver_inner, 
                                          inner_epochs))

    params = {'areas_params': areas_params, 'edges_params': edges_params}
    grads = {'areas_params': grad_areas, 'edges_params': grad_edges}

    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    areas_params = updated_params['areas_params']
    edges_params = updated_params['edges_params']

    return areas_params, edges_params

###############
### SOLVERS ###
###############

sgd_inner = optax.sgd(learning_rate=inner_lr)
sgd_outer = optax.sgd(learning_rate=outer_lr)

####################
### MINIMISATION ###
####################

last_saved = '-'
inner_loss_values, outer_loss_values = [], []

for t in range(outer_epochs):

    if t % 5 ==0:
        
        plot_geograph(vertTable, 
                      faceTable, 
                      heTable, 
                      L_box, 
                      multicolor=True, 
                      lines=True, 
                      vertices=False, 
                      path='./ad_model_sign_sgd_outer', 
                      name=t, 
                      save=True, 
                      show=False)
        
        last_saved = t

    areas_params, edges_params = update_params(areas_params, 
                                               edges_params, 
                                               vertTable, 
                                               heTable, 
                                               faceTable, 
                                               vertTable_target, 
                                               selected_vertices, 
                                               selected_faces, 
                                               selected_hes, 
                                               sgd_inner,
                                               sgd_outer, 
                                               inner_epochs)

    vertTable = inner(vertTable, 
                      heTable, 
                      faceTable, 
                      areas_params, 
                      edges_params, 
                      selected_vertices, 
                      selected_faces, 
                      selected_hes, 
                      sgd_inner, 
                      inner_epochs)

    inner_loss = energy(vertTable, heTable, faceTable, areas_params, edges_params, selected_faces, selected_hes)
    outer_loss = cost(vertTable, vertTable_target, selected_vertices)

    print(f"Epoch: {t}/{outer_epochs-1}, inner loss: {inner_loss:.5f}, outer loss: {outer_loss:.5f}, lr: {outer_lr:.5f}, last saved: {last_saved}/{outer_epochs-1}", end='\r', flush=True)

    inner_loss_values.append(inner_loss)
    outer_loss_values.append(outer_loss) 

with open('./ad_model_inner_loss.txt', 'w') as file:
    for loss in inner_loss_values:
        file.write(f"{loss}\n")  

with open('./ad_model_outer_loss.txt', 'w') as file:
    for loss in outer_loss_values:
        file.write(f"{loss}\n")  

with open('./ad_model_areas_params.txt', 'w') as file:
    for p in areas_params.tolist():
        file.write(f"{p}\n")  

with open('./ad_model_edges_params.txt', 'w') as file:
    for p in edges_params.tolist():
        file.write(f"{p}\n")  

