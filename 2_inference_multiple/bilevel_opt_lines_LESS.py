import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import jit, vmap
import jax.numpy as jnp 
import optax 

from vertax.start import save_geograph, load_geograph
from vertax.opt import bilevel_opt
from vertax.geo import get_area, get_length
from vertax.plot import plot_geograph
from vertax.cost import gaussian_blur_line_segments, cost_mesh2image, cost_v2v

## SETTINGS

path = './2_inference_multiple/bilevel_opt_lines_LESS_COUPLED_0.2/'

init_path_target = "./2_inference_multiple/energy_line_tensions_COUPLED/target/areas_target.txt"
init_data_target = np.loadtxt(init_path_target)
init_values_target = init_data_target[:, 1]
areas_target = jnp.asarray(init_values_target)

init_path_target = "./2_inference_multiple/energy_line_tensions_COUPLED/target/line_tensions_target.txt"
init_data_target = np.loadtxt(init_path_target)
init_values_target = init_data_target[:, 1]
line_tensions_target = jnp.asarray(init_values_target)

vert_params = jnp.asarray([0.])

init_path = "./2_inference_multiple/energy_line_tensions_COUPLED/starting_0.2/line_tensions_init.txt"
init_data = np.loadtxt(init_path)
init_values = init_data[:, 1]

he_params = jnp.asarray(init_values[2::2])

face_params = jnp.asarray([0.])

min_dist_T1 = 0.005

iterations_max = 1000
tolerance = 0.00001
patience=5

epochs = 100

## ENERGY LINE TENSIONS

@jit
def area_part(face: int, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    a = get_area(face, vertTable, heTable, faceTable)
    return (a - areas_target[face]) ** 2

@jit
def hedge_part(he: int, he_param: float, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    l = get_length(he, vertTable, heTable, faceTable)[0]
    return he_param * l

@jit
def energy_line_tensions(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params):
    K_areas = 20
    mapped_areas_part = lambda face: area_part(face, vertTable, heTable, faceTable)
    mapped_hedges_part = lambda he, he_param: hedge_part(he, he_param, vertTable, heTable, faceTable)
    areas_part = vmap(mapped_areas_part)(jnp.arange(len(faceTable)))
    he_params = jnp.repeat(he_params, 2)
    hedges_part = vmap(mapped_hedges_part)(jnp.arange(2, len(heTable)), he_params)
    return (2 * line_tensions_target[0] * get_length(0, vertTable, heTable, faceTable)[0]) + jnp.sum(hedges_part) + (0.5 * K_areas) * jnp.sum(areas_part)

## SOLVER

sgd = optax.sgd(learning_rate=0.01)
adam = optax.adam(learning_rate=0.0001, nesterov=True)

## BILEVEL OPTIMIZATION

vertTable, heTable, faceTable = load_geograph(path='./2_inference_multiple/energy_line_tensions_COUPLED/starting_0.2/')
selected_verts, selected_hes, selected_faces = jnp.arange(len(vertTable)), jnp.arange(len(heTable)), jnp.arange(len(faceTable))

n_cells = len(faceTable)
L_box = jnp.sqrt(n_cells)

vertTable_target, heTable_target, faceTable_target = load_geograph(path='./2_inference_multiple/energy_line_tensions_COUPLED/target/')

energy_list, cost_list, min_dist_T1_list = [], [], []

for j in range(epochs+1):

    print('epoch: '+str(j)+'/'+str(epochs)+'\t min_dist_T1: '+str(min_dist_T1)+'\t cost: '+str(cost_v2v(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vertTable_target, heTable_target, faceTable_target, image_target=None)))
    
    energy_list.append(energy_line_tensions(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params))
    cost_list.append(cost_v2v(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vertTable_target, heTable_target, faceTable_target, image_target=None))
    min_dist_T1_list.append(min_dist_T1)

    vertTable, heTable, faceTable, vert_params, he_params, face_params = bilevel_opt(vertTable,
                                                                                        heTable,
                                                                                        faceTable,
                                                                                        selected_verts,
                                                                                        selected_hes,
                                                                                        selected_faces,
                                                                                        vert_params,
                                                                                        he_params,
                                                                                        face_params,
                                                                                        vertTable_target,
                                                                                        heTable_target,
                                                                                        faceTable_target, 
                                                                                        energy_line_tensions, 
                                                                                        cost_v2v,
                                                                                        sgd,
                                                                                        adam, 
                                                                                        min_dist_T1,
                                                                                        iterations_max,
                                                                                        tolerance,
                                                                                        patience,
                                                                                        image_target=None,
                                                                                        beta=None,
                                                                                        method='ad')
    if j % 100 == 0:

        plot_geograph(vertTable, 
                    heTable, 
                    faceTable, 
                    L_box=L_box, 
                    multicolor=True, 
                    lines=True, 
                    vertices=False, 
                    path=path+'images/', 
                    name=str(j), 
                    save=True, 
                    show=False)

        save_geograph(path+'./configurations/'+str(j)+'/', vertTable, heTable, faceTable)

        with open(path+'./configurations/'+str(j)+'/'+'line_tensions_final.txt', 'w') as f:
            f.write(str(0) + '\t' + str(line_tensions_target[0]) + '\n')
            f.write(str(0) + '\t' + str(line_tensions_target[0]) + '\n')
            for he in range((len(heTable)-2)//2):
                f.write(str(he+1) + '\t' + str(he_params[he]) + '\n')
                f.write(str(he+1) + '\t' + str(he_params[he]) + '\n')

        with open(path+'./configurations/'+str(j)+'/'+'areas_final.txt', 'w') as f:
            for face in range(len(faceTable)):
                f.write(str(face) + '\t' + str(face_params[face]) + '\n')

        # Save energy_list
        with open(path+'./configurations/'+str(j)+'/'+'energy_list.txt', "w") as f:
            for energy in energy_list:
                f.write(f"{energy}\n")

        # Save cost_list
        with open(path+'./configurations/'+str(j)+'/'+'cost_list.txt', "w") as f:
            for cost in cost_list:
                f.write(f"{cost}\n")

        # Save min_dist_T1_list
        with open(path+'./configurations/'+str(j)+'/'+'min_dist_T1_list.txt', "w") as f:
            for dist in min_dist_T1_list:
                f.write(f"{dist}\n")
