from functools import partial
import jax.numpy as jnp
from jax import jit, jacfwd


class model_energy:

    def __init__(self, energy, params):

        self.energy = energy
        self.params = params

    @partial(jit, static_argnums=(0,))
    def update(self, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, step: float):
        return vertTable - step * jacfwd(self.energy)(vertTable, heTable, faceTable, self.params)


class model_eq_prop:

    def __init__(self, energy, cost, L_box, params):

        self.energy = energy
        self.cost = cost
        self.params = params
        self.L_box = L_box

    @partial(jit, static_argnums=(0,))
    def update_energy(self, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, step: float):
        return vertTable - step * jacfwd(self.energy, argnums=0)(vertTable, heTable, faceTable, self.params)

    @partial(jit, static_argnums=(0,))
    def lagrangian(self, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, vertTable_target: jnp.array, params: jnp.array, beta: float):
        return self.energy(vertTable, heTable, faceTable, params) + beta * self.cost(vertTable, vertTable_target, self.L_box)

    @partial(jit, static_argnums=(0,))
    def update_lagrangian(self, vertTable: jnp.array, heTable: jnp.array, faceTable: jnp.array, vertTable_target: jnp.array, params: jnp.array, beta: float, step: float):
        return vertTable - step * jacfwd(self.lagrangian, argnums=0)(vertTable, heTable, faceTable, vertTable_target, params, beta)

    @partial(jit, static_argnums=(0,))
    def update_params(self, vertTable_zero: jnp.array, heTable_zero: jnp.array, faceTable_zero: jnp.array, vertTable_beta: jnp.array, heTable_beta: jnp.array, faceTable_beta: jnp.array, vertTable_target: jnp.array, params: jnp.array, beta: float, step: float):
        return params - step * (1/beta) * (jacfwd(self.lagrangian, argnums=4)(vertTable_beta, heTable_beta, faceTable_beta, vertTable_target, params, beta) - jacfwd(self.lagrangian, argnums=4)(vertTable_zero, heTable_zero, faceTable_zero, vertTable_target, params, beta=0.))


