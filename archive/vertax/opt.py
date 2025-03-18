## NOTES:
# 
# ADD one function with options 'ad, ep, id'
# 
# FOR INNER PROCESS: 
# epochs == iterations
# maximum iterations with stopping conditions (certain number of steps energy does not vary relatively == tolerance)
# tolerance = for t times we have to have:  DE/E < 10**-8 (-6)(-5)
# 
# FOR OUTER PROCESS:
# add parameters associated to verteces 


import jax.numpy as jnp 
from jax import jit, jacfwd, lax, grad
import jax

import optax 

from vertax.topo import update_T1
from vertax.geo import update_pbc


######################
### FIRE OPTIMIZER ###
######################

# # Example to run FIRE optimizer
# vertTable_eq, heTable_eq, faceTable_eq = fire(energy, vertTable_init, heTable_init, faceTable_init, areas_params, edges_params, min_dist_T1=min_dist_T1, iterations=iterations)

# @jit
def fire(func, 
         vertTable, 
         heTable, 
         faceTable,
         *params,
         min_dist_T1=0.05,
         iterations=100,
         dt=0.001, 
         alpha_start=0.001, 
         alpha_decay=0.99, 
         max_dt=1.0):

    # @jit
    def grad_func(func, vertTable, heTable, faceTable, *params):
        return jacfwd(func, argnums=0)(vertTable, heTable, faceTable, *params)

    @jit
    def step(state):
        vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count = state

        # jax.debug.print("{x}\n", x=func(vertTable, heTable, faceTable, *params))    

        # Update velocities using forces
        velocities = velocities + dt * forces
        
        # Calculate the power P = sum(v · f) for all vertices
        power = jnp.sum(velocities * forces)

        # Update positions
        vertTable = vertTable + dt * velocities
        vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)
        vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, MIN_DISTANCE=min_dist_T1)

        # Recompute forces
        forces = -grad_func(func, vertTable, heTable, faceTable, *params)

        # Compute norms for velocities and forces
        velocities_norm = jnp.linalg.norm(velocities)
        forces_norm = jnp.linalg.norm(forces) 
        
        # If power is positive, increase dt and reduce alpha
        def positive_power():
            new_dt = jnp.minimum(dt * 1.1, max_dt)
            new_alpha = alpha * alpha_decay
            velocities_adjusted = (1 - new_alpha) * velocities + new_alpha * forces * velocities_norm / (forces_norm + 1e-12)
            return new_dt, new_alpha, velocities_adjusted

        # If power is negative, reset velocities and reduce dt
        def negative_power():
            new_dt = dt * 0.5
            new_alpha = alpha_start
            return new_dt, new_alpha, jnp.zeros_like(velocities)

        dt, alpha, velocities = jax.lax.cond(
            power > 0,
            positive_power,
            negative_power,
        )

        # Convergence check
        converged = jnp.where(step_count == iterations, True, False)

        return (vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count + 1), converged

    # Initialize state
    forces = -grad_func(func, vertTable, heTable, faceTable, *params)
    state = (
        vertTable,
        heTable,
        faceTable,
        jnp.zeros_like(vertTable),  # Initial velocities (N, 2)
        forces,
        alpha_start,
        dt,
        0,  # Step count
    )

    @jit
    def cond_fn(state_converged):
        _, converged = state_converged
        return ~converged

    @jit
    def body_fn(state_converged):
        state, _ = state_converged
        state, converged = step(state)
        return state, converged

    # Iterate the FIRE algorithm
    final_state, converged = jax.lax.while_loop(
        cond_fn, body_fn, (state, False)
    )

    vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count = final_state
    
    return vertTable, heTable, faceTable


#####################
### INNER PROCESS ###
#####################

def inner_optax(vertTable: jnp.array, 
                heTable: jnp.array, 
                faceTable: jnp.array, 
                *params,
                L_in, 
                solver, 
                min_dist_T1,
                iterations):

    @jit
    def update_step(carry, _):
        vertTable, heTable, faceTable, opt_state = carry

        # jax.debug.print("{x}\n", x=L_in(vertTable, heTable, faceTable, *params))    

        jacforward = jacfwd(L_in, argnums=0)(vertTable, heTable, faceTable, *params)
        updates, opt_state = solver.update(jacforward, opt_state)
        vertTable = optax.apply_updates(vertTable, updates)
        vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)
        vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, MIN_DISTANCE=min_dist_T1)
        return (vertTable, heTable, faceTable, opt_state), None

    # initialize the optimizer state
    opt_state = solver.init(vertTable)

    # lax.scan to apply the update step iterations times
    (vertTable, heTable, faceTable, _), _ = lax.scan(update_step, (vertTable, heTable, faceTable, opt_state), None, length=iterations)

    return vertTable, heTable, faceTable


#####################
### OUTER PROCESS ###
#####################

def loss_out_optax(areas_params: jnp.array, 
                   edges_params: jnp.array,
                   vertTable: jnp.array, 
                   heTable: jnp.array, 
                   faceTable: jnp.array, 
                   vertTable_target: jnp.array,
                   L_in,
                   L_out,
                   solver_inner,
                   min_dist_T1,
                   iterations: int):

    vertTable, heTable, faceTable = inner_optax(vertTable, heTable, faceTable, areas_params, edges_params, L_in=L_in, solver=solver_inner, iterations=iterations, min_dist_T1=min_dist_T1)
    
    return L_out(vertTable, heTable, faceTable, vertTable_target)

def outer_optax(areas_params: jnp.array, 
                edges_params: jnp.array, 
                vertTable: jnp.array, 
                heTable: jnp.array, 
                faceTable: jnp.array, 
                vertTable_target: jnp.array,
                L_in,
                L_out,
                solver_inner,
                solver_outer, 
                min_dist_T1,
                iterations):
    
    grad_areas = jacfwd(loss_out_optax, argnums=0)(areas_params, 
                                                   edges_params, 
                                                   vertTable, 
                                                   heTable, 
                                                   faceTable, 
                                                   vertTable_target, 
                                                   L_in, 
                                                   L_out, 
                                                   solver_inner, 
                                                   min_dist_T1,
                                                   iterations)
                
    grad_edges = jacfwd(loss_out_optax, argnums=1)(areas_params, 
                                                   edges_params, 
                                                   vertTable, 
                                                   heTable, 
                                                   faceTable, 
                                                   vertTable_target, 
                                                   L_in, 
                                                   L_out, 
                                                   solver_inner, 
                                                   min_dist_T1,
                                                   iterations)

    params = {'areas_params': areas_params, 'edges_params': edges_params}
    grads = {'areas_params': grad_areas, 'edges_params': grad_edges}

    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    areas_params = updated_params['areas_params']
    edges_params = updated_params['edges_params']

    return areas_params, edges_params


#############################
## EQUILIBRIUM PROPAGATION ##
#############################

## LOSS 

def loss_ep(vertTable: jnp.array, 
         heTable: jnp.array, 
         faceTable: jnp.array, 
         vertTable_target: jnp.array,
         areas_params: jnp.array, 
         edges_params: jnp.array,
         L_in,
         L_out,
         beta: float):
    
    loss_inner_value = L_in(vertTable, heTable, faceTable, areas_params, edges_params)
    loss_outer_value = L_out(vertTable, heTable, faceTable, vertTable_target)
    
    lagrangian_value = loss_inner_value + beta * loss_outer_value

    return lagrangian_value


## FORWARD PROCESS 

def forward(vertTable: jnp.array, 
            heTable: jnp.array, 
            faceTable: jnp.array, 
            vertTable_target: jnp.array,
            areas_params: jnp.array, 
            edges_params: jnp.array, 
            L_in,
            L_out,
            solver, 
            min_dist_T1,
            iterations: int,
            beta: float):

    @jit
    def update_step(carry, _):
        vertTable, heTable, faceTable, opt_state = carry
        jacforward = jacfwd(loss_ep, argnums=0)(vertTable, 
                                                heTable, 
                                                faceTable,
                                                vertTable_target,
                                                areas_params,
                                                edges_params,
                                                L_in,
                                                L_out,
                                                beta)
        
        updates, opt_state = solver.update(jacforward, opt_state)
        vertTable = optax.apply_updates(vertTable, updates)
        vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)
        vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, MIN_DISTANCE=min_dist_T1)
        return (vertTable, heTable, faceTable, opt_state), None

    # initialize the optimizer state
    opt_state = solver.init(vertTable)

    # lax.scan to apply the update step iterations times
    (vertTable, heTable, faceTable, _), _ = lax.scan(update_step, (vertTable, heTable, faceTable, opt_state), None, length=iterations)

    return vertTable, heTable, faceTable


## OUTER EQ PROP

def outer_eq_prop(vertTable_free, 
                  heTable_free, 
                  faceTable_free,
                  vertTable_nudged, 
                  heTable_nudged, 
                  faceTable_nudged,
                  vertTable_target,
                  areas_params, 
                  edges_params, 
                  L_in,
                  L_out,
                  solver, 
                  min_dist_T1,
                  iterations,
                  beta):

    vertTable_free, heTable_free, faceTable_free = forward(vertTable_free, 
                                                           heTable_free, 
                                                           faceTable_free, 
                                                           vertTable_target,
                                                           areas_params, 
                                                           edges_params, 
                                                           L_in,
                                                           L_out,
                                                           solver, 
                                                           min_dist_T1=min_dist_T1,
                                                           iterations=iterations,
                                                           beta=0.)
    
    vertTable_nudged, heTable_nudged, faceTable_nudged = forward(vertTable_nudged, 
                                                                 heTable_nudged, 
                                                                 faceTable_nudged, 
                                                                 vertTable_target,
                                                                 areas_params, 
                                                                 edges_params, 
                                                                 L_in,
                                                                 L_out,
                                                                 solver, 
                                                                 min_dist_T1=min_dist_T1,
                                                                 iterations=iterations,
                                                                 beta=beta)
    
    grad_loss_ep_free_areas = jacfwd(loss_ep, argnums=4)(vertTable_free, 
                                                         heTable_free, 
                                                         faceTable_free, 
                                                         vertTable_target,
                                                         areas_params, 
                                                         edges_params,
                                                         L_in,
                                                         L_out,
                                                         beta=0.)
    
    grad_loss_ep_nudged_areas = jacfwd(loss_ep, argnums=4)(vertTable_nudged, 
                                                           heTable_nudged, 
                                                           faceTable_nudged, 
                                                           vertTable_target,
                                                           areas_params, 
                                                           edges_params,
                                                           L_in,
                                                           L_out,
                                                           beta=beta)
    
    grad_loss_ep_free_edges = jacfwd(loss_ep, argnums=5)(vertTable_free, 
                                                         heTable_free, 
                                                         faceTable_free, 
                                                         vertTable_target,
                                                         areas_params, 
                                                         edges_params,
                                                         L_in,
                                                         L_out,
                                                         beta=0.)
    
    grad_loss_ep_nudged_edges = jacfwd(loss_ep, argnums=5)(vertTable_nudged, 
                                                           heTable_nudged, 
                                                           faceTable_nudged, 
                                                           vertTable_target,
                                                           areas_params, 
                                                           edges_params,
                                                           L_in,
                                                           L_out,
                                                           beta=beta)

    grad_areas = (1./beta) * ((grad_loss_ep_nudged_areas) - (grad_loss_ep_free_areas))
    grad_edges = (1./beta) * ((grad_loss_ep_nudged_edges) - (grad_loss_ep_free_edges))

    params = {'areas_params': areas_params, 'edges_params': edges_params}
    grads = {'areas_params': grad_areas, 'edges_params': grad_edges}

    opt_state = solver.init(params)
    updates, opt_state = solver.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)

    areas_params = updated_params['areas_params']
    edges_params = updated_params['edges_params']

    return areas_params, edges_params 
