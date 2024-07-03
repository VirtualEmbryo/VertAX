import operator
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from utils_geograph import update_he, move_vertex, length, get_edge_center


class topology:
    """
    Topology class for the Differentiable Vertex Model.

    Attributes:
        heTable: (jnp.array)  8-columns array with (he_prev, he_next, he_twin, v_source, v_target, face, x_he_offset_vert_target, y_he_offset_vert_target)
        faceTable: (jnp.array) 1-column array with (he_inside)

    """

    def __init__(self,
                 heTable: jnp.array,
                 faceTable: jnp.array):

        self.heTable = heTable
        self.faceTable = faceTable

    # checking heTable attribute
    heTable = property(operator.attrgetter('_heTable'))

    @heTable.setter
    def heTable(self, heTable):
        if jnp.isscalar(heTable) or heTable.shape[1] != 8:
            raise Exception("Error: heTable wrong format, jnp.array with size ( _, 8) expected.")
        self._heTable = heTable

    # checking faceTable attribute
    faceTable = property(operator.attrgetter('_faceTable'))

    @faceTable.setter
    def faceTable(self, faceTable):
        if jnp.isscalar(faceTable) or faceTable.ndim != 1:
            raise Exception("Error: faceTable wrong format, jnp.array with size ( _, ) expected.")
        self._faceTable = faceTable


class geometry(topology):
    """
    Geometry class for the Differentiable Vertex Model.

    Attributes:
        vertTable: (jnp.array) 3-columns array with (x_pos, y_pos, he_vert_source)

    """

    def __init__(self,
                 topology,
                 vertTable: jnp.array,
                 L_box: int):

        self.t_heTable = topology.heTable
        self.t_faceTable = topology.faceTable

        self.vertTable = vertTable

        self.L_box = L_box

    # checking the vertTable attribute
    vertTable = property(operator.attrgetter('_vertTable'))

    @vertTable.setter
    def vertTable(self, vertTable):
        if jnp.isscalar(vertTable) or vertTable.shape[1] != 3:
            raise Exception("Error: vertTable wrong format, jnp.array with size ( _, 3) expected.")
        self._vertTable = vertTable

    ###################################
    ##### FUNCTIONS FOR GEOMETRY ######
    ###################################

    # updating vertices positions for periodic boundary conditions
    @partial(jit, static_argnums=(0,))
    def update_vertices_positions_and_offsets(self, vertTable, t_heTable):

        mapped_offsets = lambda he: update_he(he, t_heTable, vertTable, self.L_box)
        offset_x_target, offset_y_target, offset_x_source, offset_y_source = vmap(mapped_offsets)(jnp.arange(len(t_heTable)))
        t_heTable = t_heTable.at[:, 6].add(+offset_x_target-offset_x_source)
        t_heTable = t_heTable.at[:, 7].add(+offset_y_target-offset_y_source)

        mapped_vertices = lambda v: move_vertex(v, vertTable, self.L_box)
        v_x, v_y = vmap(mapped_vertices)(jnp.arange(len(vertTable)))
        vertTable = vertTable.at[:, 0].set(v_x)
        vertTable = vertTable.at[:, 1].set(v_y)

        return vertTable, t_heTable

    # checking for T1 transitions
    # @partial(jit, static_argnums=(0,))
    def update_T1(self, vertTable: jnp.array, t_heTable: jnp.array, t_faceTable: jnp.array, MIN_DISTANCE: float):

        ### C'È UN PROBLEMA QUANDO 2 LINK CONSECUTIVI DI UNA STESSA CELLULA VOGLIONO TRANSIRE T1 ###
        ###            FORSE FARE VARIABILI DI APPOGGIO E AGGIORNARE A OGNI ITERATA?             ###

        t_heTable_new = t_heTable.copy()
        t_faceTable_new = t_faceTable.copy()
        vertTable_new = vertTable.copy()

        for idx in range(len(t_heTable)):

            he = t_heTable.at[idx].get()

            v_idx_source = he[3]
            v_idx_target = he[4]

            v_pos_source = vertTable.at[v_idx_source].get()
            v_pos_target = vertTable.at[v_idx_target].get()

            v_offset_x_target = he[6] * self.L_box
            v_offset_y_target = he[7] * self.L_box

            v1 = jnp.hstack((v_pos_source[:-1], jnp.array([0, 0])))
            v2 = jnp.hstack((v_pos_target[:-1], jnp.array([v_offset_x_target, v_offset_y_target])))

            distance = length(idx, vertTable, t_heTable, self.L_box)[0]

            if distance < MIN_DISTANCE:

                x1 = v_pos_source.at[0].get()
                y1 = v_pos_source.at[1].get()

                # find the he's center
                cx, cy = get_edge_center(v1, v2)

                # rotate of 90 degrees counterclockwise
                angle = -jnp.pi / 2.
                x1_new = cx + np.cos(angle) * (x1 - cx) - np.sin(angle) * (y1 - cy)
                y1_new = cy + np.sin(angle) * (x1 - cx) + np.cos(angle) * (y1 - cy)

                # scale at larger size than minimal distance
                x1 = (x1_new - cx) * (((MIN_DISTANCE + 10 ** (-3)) / 2.) / np.sqrt(((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cx
                y1 = (y1_new - cy) * (((MIN_DISTANCE + 10 ** (-3)) / 2.) / np.sqrt(((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cy

                he_prev = he[0]
                t_heTable_new = t_heTable_new.at[he_prev, 1].set(he[1])  # change prev_he's next_he with he's next_he
                t_heTable_new = t_heTable_new.at[he_prev, 4].set(v_idx_source)

                he_prev_twin = t_heTable.at[he_prev, 2].get()
                t_heTable_new = t_heTable_new.at[he_prev_twin, 0].set(he[2])  # idx  # change prev_twin_he's prev_he with he
                t_heTable_new = t_heTable_new.at[he_prev_twin, 3].set(v_idx_source)

                he_next = he[1]
                t_heTable_new = t_heTable_new.at[he_next, 0].set(he[0])  # change next_he's prev_he with he's prev_he
                t_heTable_new = t_heTable_new.at[he_next, 3].set(v_idx_source)  # change next_he's source_vertex with he's source_vertex

                he_next_twin = t_heTable.at[he_next, 2].get()
                t_heTable_new = t_heTable_new.at[he_next_twin, 1].set(idx)  # change next_twin_he's next_he with he's twin
                t_heTable_new = t_heTable_new.at[he_next_twin, 4].set(v_idx_source)  # change next twin he's target vertex with he's source vertex

                t_heTable_new = t_heTable_new.at[idx, 0].set(he_next_twin)
                t_heTable_new = t_heTable_new.at[idx, 1].set(t_heTable.at[he_next_twin, 1].get())
                t_heTable_new = t_heTable_new.at[idx, 5].set(t_heTable.at[he_next_twin, 5].get())

                t_faceTable_new = t_faceTable_new.at[he[5]].set(he_next)
                t_faceTable_new = t_faceTable_new.at[t_heTable_new[idx][5]].set(idx)

                vertTable_new = vertTable_new.at[he[3], 0].set(x1)
                vertTable_new = vertTable_new.at[he[3], 1].set(y1)
                vertTable_new = vertTable_new.at[he[3], 2].set(he[1])

            else:
                pass

        return vertTable_new, t_heTable_new, t_faceTable_new
