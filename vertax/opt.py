import jax.numpy as jnp 
from jax import jit, jacfwd, lax, grad, hessian, jacrev
import jax

import optax 

from vertax.topo import update_T1
from vertax.geo import update_pbc


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
                tolerance=1e-3,
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
        updates_selected = updates.at[selected_verts].get()
        vertTable = vertTable.at[selected_verts].set(vertTable[selected_verts] + updates_selected)
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
    should_stop = jnp.array(False)

    # Use lax.while_loop for early stopping
    def cond_fn(state):
        # Extract the `should_stop` scalar from the state
        _, _, _, _, _, _, _, should_stop, _ = state
        return jnp.logical_not(should_stop)  # Return a boolean scalar for continuation
    
    final_state = lax.while_loop(cond_fn, update_step, 
                                    (vertTable, heTable, faceTable, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_in_list))

    # Extract the loss values (only valid iterations)
    final_L_list = final_state[-1][:final_state[6]]  # Trim unused part of L_in_list

    return final_state[:3], final_L_list  # Return updated (vertTable, heTable, faceTable)

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

#############################
## EQUILIBTIUM PROPAGATION ##
#############################

def loss_ep(vertTable, 
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
            image_target,
            beta):
    
    loss_inner_value = L_in(vertTable, 
                            heTable, 
                            faceTable, 
                            selected_verts, 
                            selected_hes, 
                            selected_faces, 
                            vert_params, 
                            he_params, 
                            face_params)
    
    loss_outer_value = L_out(vertTable, 
                             heTable, 
                             faceTable, 
                             selected_verts, 
                             selected_hes, 
                             selected_faces, 
                             vertTable_target, 
                             heTable_target, 
                             faceTable_target, 
                             image_target)
    
    loss_ep_value = loss_inner_value + (beta * loss_outer_value)
    
    return loss_ep_value

def forward(vertTable, 
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
            image_target,
            beta):

    @jit
    def update_step(carry):
        vertTable, heTable, faceTable, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_ep_list = carry
        # Compute loss
        L_current = loss_ep(vertTable,heTable,faceTable,selected_verts,selected_hes,selected_faces,vert_params,he_params,face_params,vertTable_target,heTable_target,faceTable_target,L_in,L_out,image_target,beta)
        # Store loss in preallocated array
        L_ep_list = L_ep_list.at[step_count].set(L_current)        
        # Compute relative variation
        rel_variation = jnp.abs((L_current - prev_L_values[-1]) / jnp.where(prev_L_values[-1] != 0, prev_L_values[-1], 1.0))
        # Update stagnation count
        stagnation_count = stagnation_count + jnp.where(rel_variation < tolerance, 1, -stagnation_count)
        # Determine if we should stop
        should_stop = (stagnation_count >= patience) | (step_count >= iterations_max)
        # Compute gradient and update state
        jacforward = jacfwd(loss_ep, argnums=0)(vertTable,heTable,faceTable,selected_verts,selected_hes,selected_faces,vert_params,he_params,face_params,vertTable_target,heTable_target,faceTable_target,L_in,L_out,image_target,beta)
        updates, opt_state = solver_inner.update(jacforward, opt_state)
        # Apply updates
        updates_selected = updates.at[selected_verts].get()
        vertTable = vertTable.at[selected_verts].set(vertTable[selected_verts] + updates_selected)        
        # Apply additional updates
        vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)
        vertTable, heTable, faceTable = update_T1(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params, L_in, min_dist_T1)
        # Update previous loss values
        prev_L_values = prev_L_values.at[1:].set(prev_L_values[:-1])
        prev_L_values = prev_L_values.at[0].set(L_current)
        # Increment step count
        step_count += 1
        return (vertTable, heTable, faceTable, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_ep_list)

    # Initialize optimizer state
    opt_state = solver_inner.init(vertTable)
    # Initialize tracking variables
    initial_L = loss_ep(vertTable,heTable,faceTable,selected_verts,selected_hes,selected_faces,vert_params,he_params,face_params,vertTable_target,heTable_target,faceTable_target,L_in,L_out,image_target,beta)
    L_ep_list = jnp.zeros((iterations_max,))  # Preallocate with max iterations
    L_ep_list = L_ep_list.at[0].set(initial_L)
    prev_L_values = jnp.full((patience,), initial_L)
    stagnation_count = jnp.array(0)
    step_count = jnp.array(0)
    should_stop = jnp.array(False)
    
    def cond_fn(state):
        _, _, _, _, _, _, _, should_stop, _ = state
        return jnp.logical_not(should_stop)  

    # Use lax.while_loop for early stopping
    final_state = lax.while_loop(cond_fn, update_step, 
                                    (vertTable, heTable, faceTable, opt_state, prev_L_values, stagnation_count, step_count, should_stop, L_ep_list))

    # Extract the loss values (only valid iterations)
    final_L_list = final_state[-1][:final_state[6]] 

    # Return updated (vertTable, heTable, faceTable), L_ep
    return final_state[:3], final_L_list 

def outer_eq_prop(vertTable, 
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
                  image_target,
                  beta):
    
    (vertTable_free, heTable_free, faceTable_free), loss_free = forward(vertTable, 
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
                                                                        image_target,
                                                                        beta=0.)
    
    (vertTable_nudged, heTable_nudged, faceTable_nudged), loss_nudged = forward(vertTable, 
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
                                                                                image_target,
                                                                                beta)
    
    grad_loss_ep_free_verts = jacfwd(loss_ep, argnums=6)(vertTable_free, 
                                                         heTable_free, 
                                                         faceTable_free, 
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
                                                         image_target,
                                                         beta=0.)
    
    grad_loss_ep_nudged_verts = jacfwd(loss_ep, argnums=6)(vertTable_nudged, 
                                                           heTable_nudged, 
                                                           faceTable_nudged, 
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
                                                           image_target,
                                                           beta)
    
    grad_loss_ep_free_hes = jacfwd(loss_ep, argnums=7)(vertTable_free, 
                                                       heTable_free, 
                                                       faceTable_free, 
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
                                                       image_target,
                                                       beta=0.)
    grad_loss_ep_nudged_hes = jacfwd(loss_ep, argnums=7)(vertTable_nudged, 
                                                         heTable_nudged, 
                                                         faceTable_nudged, 
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
                                                         image_target,
                                                         beta)
    
    grad_loss_ep_free_faces = jacfwd(loss_ep, argnums=8)(vertTable_free, 
                                                        heTable_free, 
                                                        faceTable_free, 
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
                                                        image_target,
                                                        beta=0.)
    grad_loss_ep_nudged_faces = jacfwd(loss_ep, argnums=8)(vertTable_nudged, 
                                                           heTable_nudged, 
                                                           faceTable_nudged, 
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
                                                           image_target,
                                                           beta)

    grad_verts = (1./beta) * ((grad_loss_ep_nudged_verts) - (grad_loss_ep_free_verts))
    grad_hes = (1./beta) * ((grad_loss_ep_nudged_hes) - (grad_loss_ep_free_hes))
    grad_faces = (1./beta) * ((grad_loss_ep_nudged_faces) - (grad_loss_ep_free_faces))

    params = {'vert_params': vert_params, 'he_params': he_params, 'face_params': face_params}
    grads = {'vert_params': grad_verts, 'he_params': grad_hes, 'face_params': grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params['vert_params']
    he_params = updated_params['he_params']
    face_params = updated_params['face_params']
    
    return vert_params, he_params, face_params


###########################
## IMPLICIT DIFFERENTION ##
###########################

def outer_implicit(vertTable, 
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
                  image_target):
    
    def L_in_flatten(vertTable_flatten, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params):
        vertTable_tmp = jnp.hstack((vertTable_flatten.reshape(len(vertTable_flatten)//2,2),jnp.zeros((len(vertTable_flatten)//2,1))))
        return L_in(vertTable_tmp, heTable, faceTable, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)

    (vertTable_eq, heTable_eq, faceTable_eq), L_in_value = inner_optax(vertTable,
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
    
    # jax.debug.print("{}", jacfwd(L_in_flatten,argnums=0)(vertTable_eq[:,:2].flatten(), heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params).shape)
    # jax.debug.print("{}", jacfwd(jacfwd(L_in_flatten,argnums=0),argnums=0)(vertTable_eq[:,:2].flatten(), heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params).shape)
    # exit()
    
    H=jacfwd(jacfwd(L_in_flatten,argnums=0),argnums=0)(vertTable_eq[:,:2].flatten(), heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)

    crossderivative_verts=jacfwd(jacfwd(L_in_flatten,argnums=0),argnums=6)(vertTable_eq[:,:2].flatten(), heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)
    crossderivative_hes=jacfwd(jacfwd(L_in_flatten,argnums=0),argnums=7)(vertTable_eq[:,:2].flatten(), heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)
    crossderivative_faces=jacfwd(jacfwd(L_in_flatten,argnums=0),argnums=8)(vertTable_eq[:,:2].flatten(), heTable_eq, faceTable_eq, selected_verts, selected_hes, selected_faces, vert_params, he_params, face_params)
    
    # jax.debug.print("{}", crossderivative_verts.shape)
    # jax.debug.print("{}", crossderivative_hes.shape)
    # jax.debug.print("{}", crossderivative_faces.shape)
    # exit()

    L_in_derivative_verts=-jax.numpy.linalg.solve(H,crossderivative_verts)
    L_in_derivative_hes=-jax.numpy.linalg.solve(H,crossderivative_hes)
    L_in_derivative_faces=-jax.numpy.linalg.solve(H,crossderivative_faces)

    grad_verts = L_in_derivative_verts.T @ grad(L_out,argnums=0)(vertTable_eq,heTable_eq,faceTable_eq,selected_verts,selected_hes,selected_faces,vertTable_target,heTable_target,faceTable_target,image_target)[:,:2].flatten()
    grad_hes = L_in_derivative_hes.T @ grad(L_out,argnums=0)(vertTable_eq,heTable_eq,faceTable_eq,selected_verts,selected_hes,selected_faces,vertTable_target,heTable_target,faceTable_target,image_target)[:,:2].flatten()
    grad_faces = L_in_derivative_faces.T @ grad(L_out,argnums=0)(vertTable_eq,heTable_eq,faceTable_eq,selected_verts,selected_hes,selected_faces,vertTable_target,heTable_target,faceTable_target,image_target)[:,:2].flatten()

    params = {'vert_params': vert_params, 'he_params': he_params, 'face_params': face_params}
    grads = {'vert_params': grad_verts, 'he_params': grad_hes, 'face_params': grad_faces}
    opt_state = solver_outer.init(params)
    updates, opt_state = solver_outer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    vert_params = updated_params['vert_params']
    he_params = updated_params['he_params']
    face_params = updated_params['face_params']
    
    return vert_params, he_params, face_params


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
    
        (vertTable, heTable, faceTable), loss = inner_optax(vertTable,
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

    elif method == 'ep': 

        vert_params, he_params, face_params = outer_eq_prop(vertTable, 
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
                                                            image_target,
                                                            beta)

        (vertTable, heTable, faceTable), loss = forward(vertTable, 
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
                                                        image_target,
                                                        beta=0.)

    elif method == 'id': 

        vert_params, he_params, face_params = outer_implicit(vertTable, 
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
        
        (vertTable, heTable, faceTable), loss = inner_optax(vertTable,
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

    return vertTable, heTable, faceTable, vert_params, he_params, face_params
