import jax.numpy as jnp 
from jax import jit, vmap

@jit
def cost_v2v(vertTable: jnp.array,
             heTable: jnp.array,
             faceTable: jnp.array,
             vertTable_target: jnp.array):
    
    L_box = jnp.sqrt(len(faceTable))

    @jit
    def squared_distance(vertTable: jnp.array, 
                         vertTable_target: jnp.array, 
                         v: int):
        return (jnp.min(jnp.array([((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2)**0.5,
                                   ((vertTable[v][0] - (vertTable_target[v][0] + L_box)) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2)**0.5,
                                   ((vertTable[v][0] - (vertTable_target[v][0] - L_box)) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2)**0.5,
                                   ((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] + L_box)) ** 2)**0.5,
                                   ((vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] - L_box)) ** 2)**0.5,
                                   ((vertTable[v][0] - (vertTable_target[v][0] + L_box)) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] + L_box)) ** 2)**0.5,
                                   ((vertTable[v][0] - (vertTable_target[v][0] + L_box)) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] - L_box)) ** 2)**0.5,
                                   ((vertTable[v][0] - (vertTable_target[v][0] - L_box)) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] + L_box)) ** 2)**0.5,
                                   ((vertTable[v][0] - (vertTable_target[v][0] - L_box)) ** 2 + (vertTable[v][1] - (vertTable_target[v][1] - L_box)) ** 2)**0.5])))**2
        
        # return (vertTable[v][0] - vertTable_target[v][0]) ** 2 + (vertTable[v][1] - vertTable_target[v][1]) ** 2

    mapped_fn = lambda vec: (squared_distance(vertTable, vertTable_target, vec))
    distances = vmap(mapped_fn)(jnp.arange(len(faceTable)))
    
    return (1./(2*len(distances))) * jnp.sum(distances)
