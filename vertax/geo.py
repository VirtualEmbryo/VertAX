from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd
from jax.lax import while_loop

import vertax

@partial(jit, static_argnums=(3,))
def sum_edges(face, 
              heTable: jnp.array, 
              faceTable: jnp.array, 
              fun
              ):

    start_he = faceTable.at[face].get()
    he = start_he
    res = fun(he, jnp.array([0., 0., 0.]))
    he = heTable.at[he, 1].get()

    # stacked_data = (current_he, current_res)
    # res is the sum of contributions before current_he
    def cond_fun(stacked_data):
        he, _ = stacked_data
        return he != start_he

    def body_fun(stacked_data):
        he, res = stacked_data
        next_he = heTable.at[he, 1].get()
        res += fun(he, res)
        return next_he, res

    _, res = while_loop(cond_fun, body_fun, (he, res))
    return res

@jit
def get_length(he, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array):
    L_box = jnp.sqrt(len(faceTable))
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()

    x0, y0 = vertTable.at[v_source, :2].get()  # source vertex
    he_offset_x1 = heTable.at[he, 6].get() * L_box  # target offset
    he_offset_y1 = heTable.at[he, 7].get() * L_box
    x1, y1 = vertTable.at[v_target, :2].get() + jnp.array([he_offset_x1, he_offset_y1])  # target vertex

    length = jnp.hypot(x1 - x0, y1 - y0)  
    return jnp.array([length, he_offset_x1, he_offset_y1])

@jit
def get_perimeter(face, 
                  vertTable: jnp.array, 
                  heTable: jnp.array, 
                  faceTable: jnp.array
                  ):

    def fun(he, res):
        return get_length(he, vertTable, heTable, faceTable)

    return sum_edges(face, heTable, faceTable, fun)[0]

@jit
def compute_numerator(he, 
                      res, 
                      vertTable: jnp.array, 
                      heTable: jnp.array, 
                      L_box: float
                      ):

    x_offset, y_offset = res.at[1].get(), res.at[2].get()
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    x0 = vertTable.at[v_source, 0].get() + x_offset  # source
    y0 = vertTable.at[v_source, 1].get() + y_offset
    he_offset_x1 = heTable.at[he, 6].get() * L_box  # offset target
    he_offset_y1 = heTable.at[he, 7].get() * L_box
    x1 = vertTable.at[v_target, 0].get() + x_offset + he_offset_x1  # target
    y1 = vertTable.at[v_target, 1].get() + y_offset + he_offset_y1

    return jnp.array([(x0 * y1) - (x1 * y0), he_offset_x1, he_offset_y1])

# computing area for a face using  ## shoelace formula ##
@jit
def get_area(face, 
             vertTable: jnp.array, 
             heTable: jnp.array, 
             faceTable: jnp.array
             ):

    def fun(he, res):
        return compute_numerator(he, res, vertTable, heTable, jnp.sqrt(len(faceTable)))

    return 0.5 * jnp.abs(sum_edges(face, heTable, faceTable, fun)[0])

# select verts, hes, faces for faces with all verts inside L_box_inner 
def select_verts_hes_faces(vertTable: jnp.array, 
                           heTable: jnp.array, 
                           faceTable: jnp.array,
                           L_box_inner: float):
    
    L_box = jnp.sqrt(len(faceTable))

    selected_faces = []
    selected_hes = jnp.array([], dtype=int)
    selected_verts = jnp.array([], dtype=int)
    
    for face in range(len(faceTable)):
        start_he = faceTable.at[face].get()
        he = start_he
        
        hes_idxs = []
        verts_idxs = []
        all_inside = True  # flag to check if all vertices are inside L_box_inner
        
        while True:
            v_source = heTable.at[he, 3].get()
            vert_x, vert_y = vertTable.at[v_source, 0].get(), vertTable.at[v_source, 1].get()
            
            # check if the vertex is outside the inner box
            if not (((L_box - L_box_inner)/2.) <= vert_x <= ((L_box + L_box_inner)/2.) and ((L_box - L_box_inner)/2.) <= vert_y <= ((L_box + L_box_inner)/2.)):
                all_inside = False
                break
            
            hes_idxs.append(he)
            verts_idxs.append(v_source)
            
            he = heTable.at[he, 1].get()
            if he == start_he:
                break
        
        if all_inside:
            selected_faces.append(face)
            selected_hes = jnp.concatenate((selected_hes, jnp.array(hes_idxs)))
            selected_verts = jnp.concatenate((selected_verts, jnp.array(verts_idxs)))
    
    # unique elements in each array
    selected_verts = jnp.unique(selected_verts)
    selected_hes = jnp.unique(selected_hes)
    selected_faces = jnp.unique(jnp.array(selected_faces))
    
    return selected_verts, selected_hes, selected_faces

# select verts, hes, faces for faces with all verts inside a rectangle
def select_verts_hes_faces_rectangle(vertTable, 
                                    heTable, 
                                    faceTable,
                                    L_bottom,
                                    L_top):
    
    L_x = jnp.sqrt(len(faceTable))

    selected_faces = []
    selected_hes = jnp.array([], dtype=int)
    selected_verts = jnp.array([], dtype=int)
    
    for face in range(len(faceTable)):
        start_he = faceTable.at[face].get()
        he = start_he
        
        hes_idxs = []
        verts_idxs = []
        all_inside = True  # flag to check if all vertices are inside L_box_inner
        
        while True:
            v_source = heTable.at[he, 3].get()
            vert_x, vert_y = vertTable.at[v_source, 0].get(), vertTable.at[v_source, 1].get()
            
            # check if the vertex is outside the inner box
            if not (((0.) <= vert_y <= (L_x)) and ((L_bottom) <= vert_x <= (L_top))):
                all_inside = False
                break
            
            hes_idxs.append(he)
            verts_idxs.append(v_source)
            
            he = heTable.at[he, 1].get()
            if he == start_he:
                break
        
        if all_inside:
            selected_faces.append(face)
            selected_hes = jnp.concatenate((selected_hes, jnp.array(hes_idxs)))
            selected_verts = jnp.concatenate((selected_verts, jnp.array(verts_idxs)))
    
    # unique elements in each array
    selected_verts = jnp.unique(selected_verts)
    selected_hes = jnp.unique(selected_hes)
    selected_faces = jnp.unique(jnp.array(selected_faces))
    
    return selected_verts, selected_hes, selected_faces

# (only for id implementation)
# listing vertices of a face 
@jit
def get_vertices_id(face, 
                    vertTable: jnp.array, 
                    heTable: jnp.array, 
                    faceTable: jnp.array
                    ):

    n_cells = len(faceTable)
    L_box = jnp.sqrt(n_cells)

    start_he = faceTable.at[face].get()

    he = start_he
    v_source = heTable.at[he, 3].get()

    verts_sources = jnp.array([vertTable.at[v_source].get()])

    verts_offsets = jnp.array([jnp.array([0, 0])])
    he_offset_x = heTable.at[he, 6].get()
    he_offset_y = heTable.at[he, 7].get()
    sum0_offsets = he_offset_x
    sum1_offsets = he_offset_y

    he = heTable.at[he, 1].get()

    for _ in range(20 - 1):
        v_source = heTable.at[he, 3].get()
        verts_sources = jnp.concatenate((verts_sources, jnp.array([vertTable.at[v_source].get()])), axis=0)

        verts_offsets = jnp.where(he != start_he, jnp.concatenate(
            (verts_offsets, jnp.array([jnp.array([sum0_offsets * L_box, sum1_offsets * L_box])])),
            axis=0), jnp.concatenate((verts_offsets, jnp.array([verts_offsets.at[0].get()]))))

        he_offset_x = heTable.at[he, 6].get()
        he_offset_y = heTable.at[he, 7].get()
        sum0_offsets += he_offset_x
        sum1_offsets += he_offset_y

        he = jnp.where(he != start_he, heTable.at[he, 1].get(), he)

    return jnp.hstack((verts_sources.at[:, :-1].get(), verts_offsets))

# (only for id implementation)
# computing area for a face using  ## shoelace formula ##  
@jit
def get_area_id(face, 
                vertTable: jnp.array, 
                heTable: jnp.array, 
                faceTable: jnp.array
                ):

    vertices = get_vertices_id(face, vertTable, heTable, faceTable)

    numerator = 0.

    for i in range(len(vertices) - 1):
        numerator += ((vertices.at[i, 0].get() + vertices.at[i, 2].get()) * 
                      (vertices.at[i + 1, 1].get() + vertices.at[i + 1, 3].get()) - 
                      (vertices.at[i, 1].get() + vertices.at[i, 3].get()) * 
                      (vertices.at[i + 1, 0].get() + vertices.at[i + 1, 2].get()))

    return jnp.abs(numerator / 2.)

@jit
def get_perimeter_area(face, 
                       vertTable: jnp.array, 
                       heTable: jnp.array, 
                       faceTable: jnp.array
                       ):

    perimeter = get_perimeter(face, vertTable, heTable, faceTable)
    area = get_area(face, vertTable, heTable, faceTable)

    return perimeter, area

@jit
def get_mean_shape_factor(vertTable: jnp.array, 
                          heTable: jnp.array, 
                          faceTable: jnp.array
                          ):

    num_faces = len(faceTable)
    faces = jnp.arange(num_faces)
    mapped_fn = lambda face: get_perimeter(face, vertTable, heTable, faceTable)
    perimeters = vmap(mapped_fn)(faces)

    return (1./num_faces) * jnp.sum(perimeters) 

@jit
def update_he(he, 
              vertTable: jnp.array, 
              heTable: jnp.array, 
              L_box: float):

    v_idx_target = heTable.at[he, 4].get()
    v_x = vertTable.at[v_idx_target, 0].get()
    v_y = vertTable.at[v_idx_target, 1].get()
    offset_x_target = jnp.where(v_x < 0., -1, jnp.where(v_x > L_box, +1, 0))
    offset_y_target = jnp.where(v_y < 0., -1, jnp.where(v_y > L_box, +1, 0))

    v_idx_source = heTable.at[he, 3].get()
    v_x = vertTable.at[v_idx_source, 0].get()
    v_y = vertTable.at[v_idx_source, 1].get()
    offset_x_source = jnp.where(v_x < 0., -1, jnp.where(v_x > L_box, +1, 0))
    offset_y_source = jnp.where(v_y < 0., -1, jnp.where(v_y > L_box, +1, 0))

    return offset_x_target, offset_y_target, offset_x_source, offset_y_source

@jit
def move_vertex_inside(v: jnp.array, 
                       vertTable: jnp.array, 
                       L_box: float):

    v_x = vertTable.at[v, 0].get()
    v_y = vertTable.at[v, 1].get()
    v_x = jnp.where(v_x < 0., v_x + L_box, jnp.where(v_x > L_box, v_x - L_box, v_x))
    v_y = jnp.where(v_y < 0., v_y + L_box, jnp.where(v_y > L_box, v_y - L_box, v_y))

    return v_x, v_y

# updating vertices positions and offsets for periodic boundary conditions
@jit
def update_pbc(vertTable: jnp.array, 
               heTable: jnp.array, 
               faceTable: jnp.array
               ):

    n_cells = len(faceTable)
    L_box = jnp.sqrt(n_cells)

    mapped_offsets = lambda he: update_he(he, vertTable, heTable, L_box)
    offset_x_target, offset_y_target, offset_x_source, offset_y_source = vmap(mapped_offsets)(jnp.arange(len(heTable)))
    heTable = heTable.at[:, 6].add(+offset_x_target-offset_x_source)
    heTable = heTable.at[:, 7].add(+offset_y_target-offset_y_source)

    mapped_vertices = lambda v: move_vertex_inside(v, vertTable, L_box)
    v_x, v_y = vmap(mapped_vertices)(jnp.arange(len(vertTable)))
    vertTable = vertTable.at[:, 0].set(v_x)
    vertTable = vertTable.at[:, 1].set(v_y)

    return vertTable, heTable, faceTable

# lee edwards pbc for shear transformation 
def lee_edwards_pbc(vertTable, 
                    heTable, 
                    faceTable, 
                    gamma, 
                    L_bottom,
                    L_top):

    L_x = jnp.sqrt(len(faceTable))  
    rho = gamma * (L_top-L_bottom)  # shear transformation 

    def body_fun(he, carry):
        vertTable, moved_mask = carry
        cond1 = (vertTable[heTable[he, 3], 0] > L_bottom) & (vertTable[heTable[he, 4], 0] + heTable[he, 6] * L_x < L_bottom)
        cond2 = (vertTable[heTable[he, 3], 0] < L_top) & (vertTable[heTable[he, 4], 0] + heTable[he, 6] * L_x > L_top)

        def update_cond1(vt, mask):
            vt = vt.at[heTable[he, 4], 1].add(-rho)  # apply shear transformation 
            mask = mask.at[heTable[he, 4]].set(True)
            return vt, mask

        def update_cond2(vt, mask):
            vt = vt.at[heTable[he, 4], 1].add(+rho)  # apply shear transformation 
            mask = mask.at[heTable[he, 4]].set(True)
            return vt, mask

        vertTable, moved_mask = jax.lax.cond(cond1, update_cond1, lambda vt, mask: (vt, mask), vertTable, moved_mask)
        vertTable, moved_mask = jax.lax.cond(cond2, update_cond2, lambda vt, mask: (vt, mask), vertTable, moved_mask)

        return vertTable, moved_mask

    moved_mask = jnp.zeros(vertTable.shape[0], dtype=bool)
    vertTable, moved_mask = jax.lax.fori_loop(0, heTable.shape[0], body_fun, (vertTable, moved_mask))

    vertTable, heTable, faceTable = update_pbc(vertTable, heTable, faceTable)

    unmoved_verts = jnp.where(~moved_mask)[0]  # vertices that have not been moved

    return vertTable, heTable, faceTable, unmoved_verts

# computing shear modulus as the second derivative of the energy with respect to the shear strain
def get_shear_modulus(vertTable, 
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
                      iterations_max,
                      tolerance,
                      patience,
                      L_bottom,
                      L_top):

    L_x = jnp.sqrt(len(faceTable))  

    def get_energy(gamma):

        vertTable_shear, heTable_shear, faceTable_shear, unmoved_verts = lee_edwards_pbc(vertTable, 
                                                                                         heTable, 
                                                                                         faceTable,  
                                                                                         gamma, 
                                                                                         L_bottom,
                                                                                         L_top)

        (vertTable_new, heTable_new, faceTable_new), _ = vertax.inner_optax(vertTable_shear, 
                                                                        heTable_shear, 
                                                                        faceTable_shear, 
                                                                        unmoved_verts,
                                                                        selected_hes,
                                                                        selected_faces,
                                                                        vert_params,
                                                                        he_params,
                                                                        face_params,
                                                                        L_in, 
                                                                        solver, 
                                                                        min_dist_T1,
                                                                        iterations_max,
                                                                        tolerance,
                                                                        patience)

        selected_verts_new, selected_hes_new, selected_faces_new = select_verts_hes_faces_rectangle(vertTable, heTable, faceTable, L_bottom, L_top)
        face_params_new = face_params[selected_faces_new]

        energy_value = L_in(vertTable_new, heTable_new, faceTable_new, selected_verts_new, selected_hes_new, selected_faces_new, vert_params, he_params, face_params_new)

        return energy_value
    
    shear_modulus = (jacfwd(jacfwd(get_energy))(0.))/((L_top-L_bottom)*(L_x))

    return shear_modulus
