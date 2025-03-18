import jax.numpy as jnp 
from jax import jit, jacfwd, lax, grad
import jax

import optax 

from vertax.topo import update_T1
from vertax.geo import update_pbc


############################
### BILEVEL OPTIMIZATION ###
############################

###############################
## AUTOMATIC DIFFERENTIATION ##
###############################

def inner_optax(vertTable, 
                heTable, 
                faceTable, 
                selected_verts,
                selected_hes,
                selected_faces,
                vert_params,
                he_params,
                face_params,
                L_in, 
                solver, 
                min_dist_T1,
                iterations_max=1e3,
                tolerance=1e-8,
                patience=5):
    
    @jit
    def update_step(carry):
        vertTable, heTable, faceTable, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_in_list = carry
        # Compute loss
        L_current = L_in(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)
        # Store loss in preallocated array
        L_in_list = L_in_list.at[step_count].set(L_current)        
        # Compute relative variation using jnp.where to avoid if-conditions
        rel_variation = jnp.abs((L_current - prev_L_values[-1]) / jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0))
        # Update stagnation count using jnp.where
        stagnation_count = stagnation_count + jnp.where(rel_variation < tolerance, 1, -stagnation_count)
        # Determine if we should stop
        should_stop = (stagnation_count >= patience) | (step_count >= iterations_max)
        # Compute gradient and update state
        jacforward = jacfwd(L_in, argnums=0)(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)
        updates, opt_state = solver.update(jacforward, opt_state)
        # Apply updates
        vertTable = optax.apply_updates(vertTable, updates)
        # updates_selected = updates.at[selected_verts].get()
        # vertTable = vertTable.at[selected_verts].set(vertTable[selected_verts] + updates_selected)
        # Apply additional updates
        vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)
        vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params, L_in, min_dist_T1)
        # Update previous loss values (shift array)
        prev_L_values = prev_L_values.at[1:].set(prev_L_values[:-1])
        prev_L_values = prev_L_values.at[0].set(L_current)
        # Increment step count
        step_count += 1
        # Return the updated carry with the same structure
        return (vertTable, heTable, faceTable, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_in_list)

    # Initialize optimizer state
    opt_state = solver.init(vertTable)
    # Initialize tracking variables
    initial_L = L_in(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)
    L_in_list = jnp.zeros((iterations_max,))  # Preallocate with max iterations
    L_in_list = L_in_list.at[0].set(initial_L)  
    prev_L_values = jnp.full((patience,), initial_L)
    stagnation_count = jnp.array(0)
    step_count = jnp.array(0)
    # Use lax.while_loop for early stopping
    def cond_fn(state):
        # Extract the `should_stop` scalar from the state
        _, _, _, _, _, _, _, should_stop, _ = state
        return jnp.logical_not(should_stop)  # Return a boolean scalar for continuation
    
    final_state = lax.while_loop(cond_fn, update_step, 
                                    (vertTable, heTable, faceTable, opt_state, prev_L_values, stagnation_count, step_count, jnp.array(False), L_in_list))

    # Extract the loss values (only valid iterations)
    final_L_list = final_state[-1][:final_state[6]]  # Trim unused part of L_in_list

    return final_state[:3], final_L_list  # Return updated (vertTable, heTable, faceTable)

## OUTER PROCESS 

def loss_out_optax(vertTable, 
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
                   L_in,
                   L_out,
                   solver_inner,
                   min_dist_T1,
                   iterations_max,
                   tolerance,
                   patience,
                   image_target=None):
    
    (vertTable_eq, heTable_eq, faceTable_eq), L_in = inner_optax(vertTable,
                                                        heTable, 
                                                        faceTable,
                                                        selected_verts,
                                                        selected_hes,
                                                        selected_faces,  
                                                        vert_params,
                                                        he_params,
                                                        face_params,
                                                        L_in, 
                                                        solver_inner,  
                                                        min_dist_T1,
                                                        iterations_max,
                                                        tolerance,
                                                        patience)
    
    loss_out_value = L_out(vertTable_eq, 
                           heTable_eq, 
                           faceTable_eq, 
                           selected_verts, 
                           selected_hes, 
                           selected_faces, 
                           vertTable_target, 
                           heTable_target, 
                           faceTable_target, 
                           image_target)
    
    return loss_out_value

def outer_optax(vertTable, 
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
                L_in,
                L_out,
                solver_inner,
                solver_outer, 
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                image_target=None):

    grad_verts = jacfwd(loss_out_optax, argnums=6)(vertTable, 
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
                                                   L_in, 
                                                   L_out, 
                                                   solver_inner, 
                                                   min_dist_T1,
                                                   iterations_max,
                                                   tolerance,
                                                   patience,
                                                   image_target) 

    grad_hes = jacfwd(loss_out_optax, argnums=7)(vertTable, 
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
                                                 L_in, 
                                                 L_out, 
                                                 solver_inner, 
                                                 min_dist_T1,
                                                 iterations_max,
                                                 tolerance,
                                                 patience,
                                                 image_target) 
    
    grad_faces = jacfwd(loss_out_optax, argnums=8)(vertTable, 
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
                                                   L_in, 
                                                   L_out, 
                                                   solver_inner, 
                                                   min_dist_T1,
                                                   iterations_max,
                                                   tolerance,
                                                   patience,
                                                   image_target) 
    
    params = {'vert_params': vert_params, 'he_params': he_params, 'face_params': face_params}
    grads = {'vert_params': grad_verts, 'he_params': grad_hes, 'face_params': grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params['vert_params']
    he_params = updated_params['he_params']
    face_params = updated_params['face_params']
    
    return vert_params, he_params, face_params

# ####################
# ## EQ PROPAGATION ##
# ####################

# def loss_ep(vertTable: jnp.array, 
#             heTable: jnp.array, 
#             faceTable: jnp.array, 
#             areas_params: jnp.array, 
#             edges_params: jnp.array,
#             verts_params: jnp.array,
#             vertTable_target: jnp.array,
#             heTable_target: jnp.array,
#             faceTable_target: jnp.array,
#             image_target: jnp.array,
#             L_in,
#             L_out,
#             beta):
#     loss_inner_value = L_in(vertTable, heTable, faceTable, areas_params, edges_params, verts_params)
#     loss_outer_value = L_out(vertTable, heTable, faceTable, vertTable_target, heTable_target, faceTable_target, image_target)
#     loss_value = loss_inner_value + (beta * loss_outer_value)
#     return loss_value

# def forward(vertTable: jnp.array, 
#             heTable: jnp.array, 
#             faceTable: jnp.array, 
#             areas_params: jnp.array, 
#             edges_params: jnp.array, 
#             verts_params: jnp.array,
#             vertTable_target: jnp.array,
#             heTable_target: jnp.array,
#             faceTable_target: jnp.array,
#             image_target: jnp.array,
#             L_in,
#             L_out,
#             solver, 
#             min_dist_T1,
#             iterations,
#             beta):
#     @jit
#     def update_step(carry, _):
#         vertTable, heTable, faceTable, opt_state = carry
#         jacforward = jacfwd(loss_ep, argnums=0)(vertTable, 
#                                                 heTable, 
#                                                 faceTable, 
#                                                 areas_params, 
#                                                 edges_params,
#                                                 verts_params,
#                                                 vertTable_target,
#                                                 heTable_target,
#                                                 faceTable_target,
#                                                 image_target,
#                                                 L_in,
#                                                 L_out,
#                                                 beta)
#         updates, opt_state = solver.update(jacforward, opt_state)
#         vertTable = optax.apply_updates(vertTable, updates)
#         vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)
#         vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params, L_in, min_dist_T1)
#         return (vertTable, heTable, faceTable, opt_state), None
#     # initialize the optimizer state
#     opt_state = solver.init(vertTable)
#     # lax.scan to apply the update step iterations times
#     (vertTable, heTable, faceTable, _), _ = lax.scan(update_step, (vertTable, heTable, faceTable, opt_state), None, length=iterations)
#     return vertTable, heTable, faceTable

# def outer_eq_prop(vertTable, 
#                     heTable, 
#                     faceTable,
#                     areas_params, 
#                     edges_params, 
#                     verts_params,
#                     vertTable_target,
#                     heTable_target,
#                     faceTable_target,
#                     image_target,
#                     L_in,
#                     L_out,
#                     solver_inner,
#                     solver_outer, 
#                     min_dist_T1,
#                     iterations,
#                     beta):
#     vertTable_free, heTable_free, faceTable_free = forward(vertTable_free, 
#                                                             heTable_free, 
#                                                             faceTable_free, 
#                                                             areas_params, 
#                                                             edges_params, 
#                                                             verts_params,
#                                                             vertTable_target,
#                                                             heTable_target,
#                                                             faceTable_target,
#                                                             L_in,
#                                                             L_out,
#                                                             solver_inner, 
#                                                             min_dist_T1,
#                                                             iterations,
#                                                             beta=0.)
#     vertTable_nudged, heTable_nudged, faceTable_nudged = forward(vertTable, 
#                                                                     heTable, 
#                                                                     faceTable, 
#                                                                     vertTable_target,
#                                                                     areas_params, 
#                                                                     edges_params, 
#                                                                     L_in,
#                                                                     L_out,
#                                                                     solver_inner, 
#                                                                     min_dist_T1,
#                                                                     iterations,
#                                                                     beta)
#     grad_loss_ep_free_areas = jacfwd(loss_ep, argnums=4)(vertTable_free, 
#                                                             heTable_free, 
#                                                             faceTable_free, 
#                                                             areas_params, 
#                                                             edges_params,
#                                                             verts_params,
#                                                             vertTable_target,
#                                                             heTable_target,
#                                                             faceTable_target,
#                                                             L_in,
#                                                             L_out,
#                                                             beta=0.)
#     grad_loss_ep_nudged_areas = jacfwd(loss_ep, argnums=4)(vertTable_nudged, 
#                                                             heTable_nudged, 
#                                                             faceTable_nudged, 
#                                                             areas_params, 
#                                                             edges_params,
#                                                             verts_params,
#                                                             vertTable_target,
#                                                             heTable_target,
#                                                             faceTable_target,
#                                                             L_in,
#                                                             L_out,
#                                                             beta)
#     grad_loss_ep_free_edges = jacfwd(loss_ep, argnums=5)(vertTable_free, 
#                                                             heTable_free, 
#                                                             faceTable_free, 
#                                                             areas_params, 
#                                                             edges_params,
#                                                             verts_params,
#                                                             vertTable_target,
#                                                             heTable_target,
#                                                             faceTable_target,
#                                                             L_in,
#                                                             L_out,
#                                                             beta=0.)
#     grad_loss_ep_nudged_edges = jacfwd(loss_ep, argnums=5)(vertTable_nudged, 
#                                                             heTable_nudged, 
#                                                             faceTable_nudged, 
#                                                             vertTable_target,
#                                                             areas_params, 
#                                                             edges_params,
#                                                             verts_params,
#                                                             vertTable_target,
#                                                             heTable_target,
#                                                             faceTable_target,
#                                                             L_in,
#                                                             L_out,
#                                                             beta)
#     grad_loss_ep_free_verts = jacfwd(loss_ep, argnums=6)(vertTable_free, 
#                                                             heTable_free, 
#                                                             faceTable_free, 
#                                                             areas_params, 
#                                                             edges_params,
#                                                             verts_params,
#                                                             vertTable_target,
#                                                             heTable_target,
#                                                             faceTable_target,
#                                                             L_in,
#                                                             L_out,
#                                                             beta=0.)
#     grad_loss_ep_nudged_verts = jacfwd(loss_ep, argnums=6)(vertTable_nudged, 
#                                                             heTable_nudged, 
#                                                             faceTable_nudged, 
#                                                             areas_params, 
#                                                             edges_params,
#                                                             verts_params,
#                                                             vertTable_target,
#                                                             heTable_target,
#                                                             faceTable_target,
#                                                             L_in,
#                                                             L_out,
#                                                             beta)
#     grad_areas = (1./beta) * ((grad_loss_ep_nudged_areas) - (grad_loss_ep_free_areas))
#     grad_edges = (1./beta) * ((grad_loss_ep_nudged_edges) - (grad_loss_ep_free_edges))
#     grad_verts = (1./beta) * ((grad_loss_ep_nudged_verts) - (grad_loss_ep_free_verts))
#     params = {'areas_params': areas_params, 'edges_params': edges_params, 'verts_params': verts_params}
#     grads = {'areas_params': grad_areas, 'edges_params': grad_edges, 'verts_params': grad_verts}
#     opt_state = solver_outer.init(params)
#     updates, opt_state = solver_outer.update(grads, opt_state, params)
#     updated_params = optax.apply_updates(params, updates)
#     areas_params = updated_params['areas_params']
#     edges_params = updated_params['edges_params']
#     verts_params = updated_params['verts_params']
#     return areas_params, edges_params, verts_params



#############
## WRAPPER ##
#############

def bilevel_opt(vertTable, 
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
                L_in, 
                L_out,
                solver_inner,
                solver_outer, 
                min_dist_T1,
                iterations_max,
                tolerance,
                patience,
                image_target=None,
                beta=None,
                method='ad'):

    if method == 'ad':

        vert_params, he_params, face_params = outer_optax(vertTable, 
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
                                                          L_in,
                                                          L_out,
                                                          solver_inner,
                                                          solver_outer, 
                                                          min_dist_T1,
                                                          iterations_max,
                                                          tolerance,
                                                          patience,
                                                          image_target)
    
        (vertTable, heTable, faceTable), L_in = inner_optax(vertTable,
                                                    heTable, 
                                                    faceTable,
                                                    selected_verts,
                                                    selected_hes,
                                                    selected_faces,  
                                                    vert_params,
                                                    he_params,
                                                    face_params,
                                                    L_in, 
                                                    solver_inner,  
                                                    min_dist_T1,
                                                    iterations_max,
                                                    tolerance,
                                                    patience)

    # elif method == 'ep': 

    #     vertTable_free, heTable_free, faceTable_free = forward(vertTable, 
    #                                                            heTable, 
    #                                                            faceTable,
    #                                                            areas_params, 
    #                                                            edges_params,
    #                                                            verts_params,
    #                                                            vertTable_target,
    #                                                            heTable_target,
    #                                                            faceTable_target,
    #                                                            image_target,
    #                                                            L_in,
    #                                                            L_out,
    #                                                            solver_inner, 
    #                                                            min_dist_T1,
    #                                                            iterations,
    #                                                            beta=0.)
    # 
    #     areas_params_new, edges_params_new, verts_params_new = outer_eq_prop(vertTable, 
    #                                                                          heTable, 
    #                                                                          faceTable,
    #                                                                          areas_params, 
    #                                                                          edges_params, 
    #                                                                          verts_params,
    #                                                                          vertTable_target,
    #                                                                          heTable_target,
    #                                                                          faceTable_target,
    #                                                                          image_target,
    #                                                                          L_in,
    #                                                                          L_out,
    #                                                                          solver_inner,
    #                                                                          solver_outer, 
    #                                                                          min_dist_T1,
    #                                                                          iterations,
    #                                                                          beta)
    
    # elif method=='id':
    #   
    #     ## TODO
    # 
    #     exit()


    return vertTable, heTable, faceTable, vert_params, he_params, face_params




# ######################
# ### FIRE OPTIMIZER ###
# ######################

# # # Example to run FIRE optimizer
# # vertTable_eq, heTable_eq, faceTable_eq = fire(energy, vertTable_init, heTable_init, faceTable_init, areas_params, edges_params, min_dist_T1=min_dist_T1, iterations=iterations)

# @jit
# def fire(func, 
#          vertTable, 
#          heTable, 
#          faceTable,
#          *params,
#          min_dist_T1=0.05,
#          iterations=100,
#          dt=0.001, 
#          alpha_start=0.001, 
#          alpha_decay=0.99, 
#          max_dt=1.0):

#     @jit
#     def grad_func(func, vertTable, heTable, faceTable, *params):
#         return jacfwd(func, argnums=0)(vertTable, heTable, faceTable, *params)

#     @jit
#     def step(state):
#         vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count = state
#         # jax.debug.print("{x}\n", x=func(vertTable, heTable, faceTable, *params))    
#         # Update velocities using forces
#         velocities = velocities + dt * forces
#         # Calculate the power P = sum(v · f) for all vertices
#         power = jnp.sum(velocities * forces)
#         # Update positions
#         vertTable = vertTable + dt * velocities
#         vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)
#         vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params, L_in, min_dist_T1)
#         # Recompute forces
#         forces = -grad_func(func, vertTable, heTable, faceTable, *params)
#         # Compute norms for velocities and forces
#         velocities_norm = jnp.linalg.norm(velocities)
#         forces_norm = jnp.linalg.norm(forces) 
#         # If power is positive, increase dt and reduce alpha
#         def positive_power():
#             new_dt = jnp.minimum(dt * 1.1, max_dt)
#             new_alpha = alpha * alpha_decay
#             velocities_adjusted = (1 - new_alpha) * velocities + new_alpha * forces * velocities_norm / (forces_norm + 1e-12)
#             return new_dt, new_alpha, velocities_adjusted
#         # If power is negative, reset velocities and reduce dt
#         def negative_power():
#             new_dt = dt * 0.5
#             new_alpha = alpha_start
#             return new_dt, new_alpha, jnp.zeros_like(velocities)
#         dt, alpha, velocities = jax.lax.cond(
#             power > 0,
#             positive_power,
#             negative_power,
#         )
#         # Convergence check
#         converged = jnp.where(step_count == iterations, True, False)
#         return (vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count + 1), converged
#     # Initialize state
#     forces = -grad_func(func, vertTable, heTable, faceTable, *params)
#     state = (
#         vertTable,
#         heTable,
#         faceTable,
#         jnp.zeros_like(vertTable),  # Initial velocities (N, 2)
#         forces,
#         alpha_start,
#         dt,
#         0,  # Step count
#     )
#     @jit
#     def cond_fn(state_converged):
#         _, converged = state_converged
#         return ~converged
#     @jit
#     def body_fn(state_converged):
#         state, _ = state_converged
#         state, converged = step(state)
#         return state, converged
#     # Iterate the FIRE algorithm
#     final_state, converged = jax.lax.while_loop(
#         cond_fn, body_fn, (state, False)
#     )
#     vertTable, heTable, faceTable, velocities, forces, alpha, dt, step_count = final_state
#     return vertTable, heTable, faceTable
