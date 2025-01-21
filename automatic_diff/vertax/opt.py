import jax.numpy as jnp 
from jax import jit, jacfwd, lax 

import optax 

from vertax.topo import update_T1


####################
### MINIMIZATION ###
####################

def min_fun(vertTable: jnp.array, 
            heTable: jnp.array, 
            faceTable: jnp.array, 
            areas_params: jnp.array, 
            edges_params: jnp.array, 
            fun, 
            solver, 
            epochs: int
            ):

    #@jit
    def update_step(carry, _):
        vertTable, heTable, faceTable, opt_state = carry
        jacforward = jacfwd(fun, argnums=0)(vertTable, heTable, faceTable, areas_params, edges_params)
        updates, opt_state = solver.update(jacforward, opt_state)
        vertTable = optax.apply_updates(vertTable, updates)
        # vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, MIN_DISTANCE = 0.0001)
        return (vertTable, heTable, faceTable, opt_state), None

    # initialize the optimizer state
    opt_state = solver.init(vertTable)

    # lax.scan to apply the update step inner_time times
    (vertTable, heTable, faceTable, _), _ = lax.scan(update_step, (vertTable, heTable, faceTable, opt_state), None, length=epochs)

    return vertTable


#####################
### INNER PROCESS ###
#####################

def inner(vertTable: jnp.array, 
          heTable: jnp.array, 
          faceTable: jnp.array, 
          areas_params: jnp.array, 
          edges_params: jnp.array, 
          L_in, 
          solver_inner, 
          inner_epochs: int
          ):

    @jit
    def update_step(carry, _):
        vertTable, opt_state = carry
        jacforward = jacfwd(L_in, argnums=0)(vertTable, heTable, faceTable, areas_params, edges_params)
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
          L_in,
          L_out,
          solver_inner,
          inner_epochs: int
          ):

    vertTable = inner(vertTable, heTable, faceTable, L_in, areas_params, edges_params, solver_inner, inner_epochs)

    L_out_value = L_out(vertTable, vertTable_target)

    return L_out_value


#########################
### PARAMETERS UPDATE ###
#########################

def update_params(areas_params: jnp.array, 
                  edges_params: jnp.array, 
                  vertTable: jnp.array, 
                  heTable: jnp.array, 
                  faceTable: jnp.array, 
                  vertTable_target: jnp.array,
                  L_in,
                  L_out,
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
                                                   L_in, 
                                                   L_out, 
                                                   solver_inner, 
                                                   inner_epochs))
    
    grad_edges = jnp.sign(jacfwd(outer, argnums=1)(areas_params, 
                                          edges_params, 
                                          vertTable, 
                                          heTable, 
                                          faceTable, 
                                          vertTable_target, 
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
