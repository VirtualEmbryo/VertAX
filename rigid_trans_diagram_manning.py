import numpy as np 

import jax.numpy as jnp 
from jax import jit, vmap
import optax 

from vertax.start import create_geograph_from_seeds, save_geograph
from vertax.geo import get_area, get_perimeter
from vertax.opt import inner_optax

## SETTINGS

path = './_section1/results_rigid_trans_diagram_100cells_100trials_0.2min_dist_T1/'

vert_param_init = 0.   # init cond vert params 
he_param_init = 0.    # init cond edge params

min_dist_T1 = 0.2

iterations_max = 50000
tolerance = 0.00000001
patience=5

## SOLVER

sgd = optax.sgd(learning_rate=0.01)

## ENERGY

@jit
def cell_energy(face, face_param, vertTable, heTable, faceTable):
    area = get_area(face, vertTable, heTable, faceTable)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable)
    return ((area - 1) ** 2) + ((perimeter - face_param) ** 2)

@jit
def energy(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params):
    mapped_fn = lambda face, param: cell_energy(face, param, vertTable, heTable, faceTable)
    face_params_broadcasted = jnp.broadcast_to(face_params, (len(faceTable),) + face_params.shape[1:])
    cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params_broadcasted)
    return jnp.sum(cell_energies)

## ENERGY MINIMIZATION 

n_cells = 100
L_box = jnp.sqrt(n_cells)

vert_params = [vert_param_init]
vert_params = jnp.asarray(vert_params)

he_params = [he_param_init]
he_params = jnp.asarray(he_params)

p_0 = []
p_0_std = []
for i in range(5):

    face_param_init = 3.8 + i*0.05
    face_params = [face_param_init] 
    face_params = jnp.asarray(face_params)
    print('\nshape factor: ' + str(face_param_init))

    L_in_list = []
    p_0_tmp = []
    for j in range(100):
        print('run: ' + str(j))

        seeds = L_box * np.random.random_sample((n_cells, 2))
        vertTable_init, heTable_init, faceTable_init = create_geograph_from_seeds(seeds, path, show=False)
        selected_verts, selected_hes, selected_faces = jnp.arange(len(vertTable_init)), jnp.arange(len(heTable_init)), jnp.arange(len(faceTable_init))

        (vertTable_eq, heTable_eq, faceTable_eq), L_in = inner_optax(vertTable_init,
                                                                    heTable_init, 
                                                                    faceTable_init,
                                                                    selected_verts,
                                                                    selected_hes,
                                                                    selected_faces,  
                                                                    vert_params,
                                                                    he_params,
                                                                    face_params,
                                                                    energy, 
                                                                    sgd,  
                                                                    min_dist_T1,
                                                                    iterations_max,
                                                                    tolerance,
                                                                    patience)
        
        L_in_list.append(L_in[-1])
        save_geograph(path + 'simulation/' + 'p0_' + str(face_param_init) +'_run_' + str(j) + '/', vertTable_eq, heTable_eq, faceTable_eq)
        
        # p_0_tmp.append(get_mean_shape_factor(vertTable_eq, heTable_eq, faceTable_eq))
    
    with open(path + 'simulation/L_in_values.txt', 'a') as file:
        for value in L_in_list:
            file.write(str(face_param_init) + '\t' + str(value) + "\n")

    # print(p_0_tmp)
    # p_0.append(jnp.mean(jnp.array(p_0_tmp)))
    # p_0_std.append(jnp.std(jnp.array(p_0_tmp)))


import matplotlib.pyplot as plt

# # X values for the plot
# x_values = [3.75, 3.8, 3.85, 3.9, 3.95, 4.0]
# # Y values == # of L_in values that are less than 0.1
# y_values = [0, 0, 1, 6, 10, 9]
# # Create the plot with error bars (standard deviations)
# #plt.errorbar(x_values, p_0, yerr=p_0_std, fmt='o-', color='black', ecolor='black', capsize=3)
# plt.errorbar(x_values, y_values, fmt='o-', color='black')
# # Save the plot
# plt.savefig(path + 'rigid_trans_sigmoid.svg', format='svg')
# # Show the plot
# plt.show()

data = []
with open(path+"simulation/L_in_values.txt", "r") as file:  # Replace "data.txt" with the actual filename
    for line in file:
        values = line.strip().split()
        data.append((float(values[0]), float(values[1])))

# Process data
unique_values = sorted(set(x[0] for x in data))  # Unique sorted values from first column
count_below_0_1 = [
    sum(1 for x in data if x[0] == val and x[1] < 0.1) for val in unique_values
]

print(unique_values)
print(count_below_0_1)

plt.errorbar(unique_values, count_below_0_1, fmt='o-', color='black')
# Save the plot
plt.savefig(path + 'rigid_trans_sigmoid.svg', format='svg')
# Show the plot
plt.show()
