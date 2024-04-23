from functools import partial
import jax.numpy as jnp
from jax import jit, jacfwd


class model:

    def __init__(self, energy, params):

        self.energy = energy
        self.params = params

    @partial(jit, static_argnums=(0,))
    def update(self, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, step: float):
        return vertTable - step * jacfwd(self.energy)(vertTable, heTable, faceTable, self.params)


