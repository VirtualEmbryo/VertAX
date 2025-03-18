from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np 

import jax.numpy as jnp 
from jax import jit, vmap
import optax 

from vertax.start import load_geograph
from vertax.geo import get_area, get_perimeter, select_verts_hes_faces
from vertax.opt import bilevel_opt
from vertax.cost import cost_v2v, cost_mesh2image, gaussian_blur_line_segments
from vertax.plot import plot_geograph

## SETTINGS

# parameters initial condition
# NB. at the moment, it is necessary to define some parameters for the vertices, for the hes, and for the faces. 
# if your energy function doesn't depend on one (or more) of these parameters, they can conventionally be put to 0. 
vert_param_init = 0.   # init cond vert params 
he_param_init = 0.    # init cond edge params
face_param_init = [3.22+0.01*i for i in range(100)]  # init cond area params --> values distributed around the target value of 3.72

# setting for T1 transitions
min_dist_T1 = 0.2

# settings for energy minimization
iterations_max = 10000
tolerance = 0.000001
patience=5

# setting for cost minimization
epochs = 100

## SOLVERS

# optax solvers for energy and cost minimization
sgd = optax.sgd(learning_rate=0.01)
adam = optax.adam(learning_rate=0.01, nesterov=True)

## ENERGY

# energy function definition 
# NB. at the moment, every new energy function definition has to depend explicitely on these variables, and on these variables only (even if not all of them are eventually used in computing the energy function itself): 
# (vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)
@jit
def cell_energy(face, face_param, vertTable, heTable, faceTable):
    area = get_area(face, vertTable, heTable, faceTable)
    perimeter = get_perimeter(face, vertTable, heTable, faceTable)
    return ((area - 1) ** 2) + ((perimeter - face_param) ** 2)

@jit
def energy(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params):
    mapped_fn = lambda face, param: cell_energy(face, param, vertTable, heTable, faceTable)
    cell_energies = vmap(mapped_fn)(jnp.arange(len(faceTable)), face_params)
    return jnp.sum(cell_energies)

## COST

# weights for cost function
w_cost_v2v = 100.
w_cost_mesh2image = 1.

# cost function definition (weighted sum of cost_v2v and cost_mesh2image)
# NB. at the moment, every new cost function definition has to depend explicitely on these variables, and on these variables only (even if not all of them are eventually used in computing the cost function itself): 
# (vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vertTable_target, heTable_target, faceTable_target, image_target)
def cost(vertTable_eq, heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vertTable_target, heTable_target, faceTable_target, image_target): 
    return (
        w_cost_v2v
        * cost_v2v(
            vertTable_eq,
            heTable_eq,
            faceTable_eq,
            selected_verts,
            selected_hes,
            selected_faces,
            vertTable_target,
            heTable_target,
            faceTable_target,
            image_target,
        )
        + w_cost_mesh2image
        * cost_mesh2image(
            vertTable_eq,
            heTable_eq,
            faceTable_eq,
            selected_verts,
            selected_hes,
            selected_faces,
            vertTable_target,
            heTable_target,
            faceTable_target,
            image_target,
        )
    )

## INITIAL CONFIGURATION AND TARGET CONFIGURATION

# load a target configuration and 
vertTable_target, heTable_target, faceTable_target = load_geograph('/workspace/vertax/_section2/init_cond_100cells_0.2min_dist_T1_3.72p0_homo/')
vertTable_eq, heTable_eq, faceTable_eq = vertTable_target.copy(), heTable_target.copy(), faceTable_target.copy()

# defining the L_box as the sqrt(#cells)
L_box = jnp.sqrt(len(faceTable_eq))

# selecting all the vertices and all the faces of the target, but only the hes which are inside of the smaller box of size (L_box-L_box/10) in the target:
# NB. this is a specific choice for this case:
# here I wanted to use the cost_v2v on all the vertices, and the cost_mesh2image on a subset of hes which are inside of the box (not cut from the walls of the L_box)
selected_verts_tmp, selected_hes_tmp, selected_faces_tmp = select_verts_hes_faces(vertTable_target, heTable_target, faceTable_target, (L_box-L_box/10.))
selected_verts, selected_hes, selected_faces = jnp.arange(len(vertTable_target)), selected_hes_tmp, jnp.arange(len(faceTable_target))
# in principle, to use all the verts, hes and faces, you can set: 
# selected_verts, selected_hes, selected_faces = jnp.arange(len(vertTable_target)), jnp.arange(len(heTable_target)), jnp.arange(len(faceTable_target))

# storing the initial parameters as jnp arrays 
vert_params = [vert_param_init]
vert_params = jnp.asarray(vert_params)
he_params = [he_param_init]
he_params = jnp.asarray(he_params)
face_params = jnp.asarray(face_param_init)

# generating an image out of the target vertax configuration (this will be the cost_mesh2image function's parameter called 'image_target')
# Blur target configuration
mask = selected_hes # (N,)
starting = (vertTable_target[heTable_target[mask, 3], :2]) * 2 / L_box  # (M, 2)
ending = (vertTable_target[heTable_target[mask, 4], :2]) * 2 / L_box  # (M, 2)
he_edges = jnp.stack((starting, ending), axis=1)  # (N, 2, 2)
x = he_edges.transpose(1, 2, 0) - 1  # (2, 2, N)
image_target=gaussian_blur_line_segments(x).real
# plt.imshow(image_target)
# plt.show()  # plot to check that every is fine with the 'image_target'

## BILEVEL OPTIMIZATION

# lists to store energy and cost values after each bilevel optimization step
loss_inner, loss_outer = [], []

# list to store the parameters values after each bilevel optimization step
face_params_list = [face_params]

for _ in tqdm(range(epochs)):
    
    print('\n')
    print(cost(vertTable_eq, heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vertTable_target, heTable_target, faceTable_target, image_target))
    print(face_params)

    loss_inner.append(energy(vertTable_eq, heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params))
    loss_outer.append(cost(vertTable_eq, heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vertTable_target, heTable_target, faceTable_target, image_target))
    
    # bilevel optimization step: it updates the new configuration (vertTable, heTable, faceTable) and the new parameters (vert_params, he_params, face_params)
    vertTable_eq, heTable_eq, faceTable_eq, vert_params, he_params, face_params = bilevel_opt(vertTable_eq,
                                                                                              heTable_eq,
                                                                                              faceTable_eq,
                                                                                              selected_verts,
                                                                                              selected_hes,
                                                                                              selected_faces,
                                                                                              vert_params,
                                                                                              he_params,
                                                                                              face_params,
                                                                                              vertTable_target,
                                                                                              heTable_target,
                                                                                              faceTable_target, 
                                                                                              energy, 
                                                                                              cost,
                                                                                              sgd,
                                                                                              adam, 
                                                                                              min_dist_T1,
                                                                                              iterations_max,
                                                                                              tolerance,
                                                                                              patience,
                                                                                              image_target,
                                                                                              beta=None,
                                                                                              method='ad')

    face_params_list.append(face_params)

# plot final cofiguration 
plot_geograph(vertTable_eq, 
              heTable_eq, 
              faceTable_eq, 
              L_box=L_box, 
              multicolor=True, 
              lines=True, 
              vertices=False, 
              path='-', 
              name='-', 
              save=False, 
              show=True)


