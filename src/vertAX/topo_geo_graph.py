import os
import time

import operator
from functools import partial

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import while_loop, fori_loop


class DVM_topology:
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


class DVM_geometry(DVM_topology):
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

    @partial(jit, static_argnums=(0,4,))
    def sum_edges(self, face: int, t_heTable: jnp.array, t_faceTable: jnp.array, fun):

        start_he = t_faceTable.at[face].get()
        he = start_he
        res = fun(he, jnp.array([0., 0., 0.]))
        he = t_heTable.at[he, 1].get()

        # stacked_data = (current_he, current_res)
        # res is the sum of contributions before current_he
        def cond_fun(stacked_data):
            he, _ = stacked_data
            return he != start_he

        def body_fun(stacked_data):
            he, res = stacked_data
            next_he = t_heTable.at[he, 1].get()
            res += fun(he, res)
            return next_he, res

        _, res = while_loop(cond_fun, body_fun, (he, res))
        return res

    @partial(jit, static_argnums=(0,))
    def length(self, he, vertTable: jnp.array, t_heTable: jnp.array):

        v_source = t_heTable.at[he, 3].get()
        v_target = t_heTable.at[he, 4].get()
        x0 = vertTable.at[v_source, 0].get()  # source
        y0 = vertTable.at[v_source, 1].get()
        he_offset_x1 = t_heTable.at[he, 6].get() * self.L_box  # offset target
        he_offset_y1 = t_heTable.at[he, 7].get() * self.L_box
        x1 = vertTable.at[v_target, 0].get() + he_offset_x1# target
        y1 = vertTable.at[v_target, 1].get() + he_offset_y1

        return jnp.array([jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2), he_offset_x1, he_offset_y1])

    @partial(jit, static_argnums=(0,))
    def get_perimeter(self, face: int, vertTable: jnp.array, t_heTable: jnp.array, t_faceTable: jnp.array):

        def fun(he, res):
            return self.length(he, vertTable, t_heTable)

        return self.sum_edges(face, t_heTable, t_faceTable, fun)[0]

    @partial(jit, static_argnums=(0,))
    def compute_numerator(self, he, res, vertTable: jnp.array, t_heTable: jnp.array):

        x_offset, y_offset = res.at[1].get(), res.at[2].get()
        v_source = t_heTable.at[he, 3].get()
        v_target = t_heTable.at[he, 4].get()
        x0 = vertTable.at[v_source, 0].get() + x_offset  # source
        y0 = vertTable.at[v_source, 1].get() + y_offset
        he_offset_x1 = t_heTable.at[he, 6].get() * self.L_box  # offset target
        he_offset_y1 = t_heTable.at[he, 7].get() * self.L_box
        x1 = vertTable.at[v_target, 0].get() + x_offset + he_offset_x1  # target
        y1 = vertTable.at[v_target, 1].get() + y_offset + he_offset_y1

        return jnp.array([(x0 * y1) - (x1 * y0), he_offset_x1, he_offset_y1])

    @partial(jit, static_argnums=(0,))
    def get_area(self, face: int, vertTable: jnp.array, t_heTable: jnp.array, t_faceTable: jnp.array):

        def fun(he, res):
            return self.compute_numerator(he, res, vertTable, t_heTable)

        return 0.5 * jnp.abs(self.sum_edges(face, t_heTable, t_faceTable, fun)[0])

    @partial(jit, static_argnums=(0,))
    def compute_single_face(self, face, vertTable, t_heTable, t_faceTable):

        perimeter = self.get_perimeter(face, vertTable, t_heTable, t_faceTable)
        area = self.get_area(face, vertTable, t_heTable, t_faceTable)

        return perimeter, area

    @partial(jit, static_argnums=(0,))
    def get_shape_factor(self, vertTable, t_heTable, t_faceTable):

        num_faces = len(t_faceTable)
        faces = jnp.arange(num_faces)
        mapped_fn = lambda face: self.compute_single_face(face, vertTable, t_heTable, t_faceTable)
        perimeters, areas = vmap(mapped_fn)(faces)

        return jnp.sum(perimeters) / jnp.sqrt(num_faces * jnp.sum(areas))

    # computing center given two vertices with their offsets
    # v1 = (v1_x, v1_y, v1_offset_x * L_box, v1_offset_y * L_box)
    # v2 = (v2_x, v2_y, v2_offset_x * L_box, v2_offset_y * L_box)
    @partial(jit, static_argnums=(0,))
    def get_edge_center(self, v1: jnp.array, v2: jnp.array):

        cx = ((v1.at[0].get() + v1.at[2].get()) + (v2.at[0].get() + v2.at[2].get())) / 2
        cy = ((v1.at[1].get() + v1.at[3].get()) + (v2.at[1].get() + v2.at[3].get())) / 2

        return cx, cy

    # updating vertices positions for periodic boundary conditions
    # @partial(jit, static_argnums=(0,))
    def update_vertices_positions_and_offsets(self, vertTable, t_heTable):

        for v in range(len(vertTable)):  # move back each vertex inside the box

            v_x = vertTable.at[v, 0].get()
            v_y = vertTable.at[v, 1].get()

            if v_x < 0.:
                vertTable = vertTable.at[v, 0].set(v_x + self.L_box)
                offset_x = -1
            elif v_x > self.L_box:
                vertTable = vertTable.at[v, 0].set(v_x - self.L_box)
                offset_x = +1
            else:
                vertTable = vertTable.at[v, 0].set(v_x)
                offset_x = 0
            if v_y < 0.:
                vertTable = vertTable.at[v, 1].set(v_y + self.L_box)
                offset_y = -1
            elif v_y > self.L_box:
                vertTable = vertTable.at[v, 1].set(v_y - self.L_box)
                offset_y = +1
            else:
                vertTable = vertTable.at[v, 1].set(v_y)
                offset_y = 0

            # adding offset to all links for which v is target
            v_target_links_indices = jnp.where(t_heTable.at[:, 4].get() == v)
            for t_idx in v_target_links_indices[0]:
                t_heTable = t_heTable.at[t_idx, 6].add(offset_x)
                t_heTable = t_heTable.at[t_idx, 7].add(offset_y)

            # removing offset to all links for which v is source
            v_source_links_indices = jnp.where(t_heTable.at[:, 3].get() == v)
            for s_idx in v_source_links_indices[0]:
                t_heTable = t_heTable.at[s_idx, 6].add(-offset_x)
                t_heTable = t_heTable.at[s_idx, 7].add(-offset_y)

        return vertTable, t_heTable



    ### THIS VERSION FOR SOME REASONS GIVES NAN IN THE ENERGY AT SOME POINT ###

    # # updating vertices positions for periodic boundary conditions
    # # @partial(jit, static_argnums=(0,))
    # def update_vertices_positions_and_offsets(self, vertTable, t_heTable):
    #
    #     for v in range(len(vertTable)):
    #
    #         v_x = vertTable.at[v, 0].get()
    #         v_y = vertTable.at[v, 1].get()
    #
    #         # Update x-coordinate and offset for periodic boundary conditions
    #         mask_x = (v_x < 0) | (v_x > self.L_box)
    #         offset_x = jnp.where(v_x < 0, 1, jnp.where(v_x > self.L_box, -1, 0))
    #         vertTable = vertTable.at[v, 0].set(jnp.where(mask_x, v_x + offset_x * self.L_box, v_x))
    #
    #         # Update y-coordinate and offset for periodic boundary conditions
    #         mask_y = (v_y < 0) | (v_y > self.L_box)
    #         offset_y = jnp.where(v_y < 0, 1, jnp.where(v_y > self.L_box, -1, 0))
    #         vertTable = vertTable.at[v, 1].set(jnp.where(mask_y, v_y + offset_y * self.L_box, v_y))
    #
    #         # adding offset to all links for which v is target
    #         v_target_links_indices = jnp.where(t_heTable[:, 4] == v)
    #         for t_idx in v_target_links_indices[0]:
    #             t_heTable = t_heTable.at[t_idx, 6].add(offset_x)
    #             t_heTable = t_heTable.at[t_idx, 7].add(offset_y)
    #
    #         # removing offset to all links for which v is source
    #         v_source_links_indices = jnp.where(t_heTable[:, 3] == v)
    #         for s_idx in v_source_links_indices[0]:
    #             t_heTable = t_heTable.at[s_idx, 6].add(-offset_x)
    #             t_heTable = t_heTable.at[s_idx, 7].add(-offset_y)
    #
    #     return vertTable, t_heTable



    # checking for T1 transitions
    # @partial(jit, static_argnums=(0,))
    def update_T1(self, MIN_DISTANCE: float):

        ### C'È UN PROBLEMA QUANDO 2 LINK CONSECUTIVI DI UNA STESSA CELLULA VOGLIONO TRANSIRE T1 ###
        ### PER COME È SCRITTO ADESSO NON FUNZIONA. FORSE FARE VARIABILI DI APPOGGIO E CAMBIARLE OGNI ITERATA? ###

        t_heTable_new = self.t_heTable.copy()
        t_faceTable_new = self.t_faceTable.copy()
        vertTable_new = self.vertTable.copy()

        for he in self.t_heTable:

            v_idx_source = he[3]
            v_idx_target = he[4]

            v_pos_source = self.vertTable.at[v_idx_source].get()
            v_pos_target = self.vertTable.at[v_idx_target].get()

            v_offset_x_target = he[6] * self.L_box
            v_offset_y_target = he[7] * self.L_box

            v1 = jnp.hstack((v_pos_source[:-1], jnp.array([0, 0])))
            v2 = jnp.hstack((v_pos_target[:-1], jnp.array([v_offset_x_target, v_offset_y_target])))

            idx = self.t_heTable.tolist().index(he.tolist())

            distance = self.length(idx, self.vertTable, self.t_heTable)[0]

            if distance < MIN_DISTANCE:

                x1 = v_pos_source.at[0].get()
                y1 = v_pos_source.at[1].get()

                # find the he's center
                cx = self.get_edge_center(v1, v2)[0]
                cy = self.get_edge_center(v1, v2)[1]

                # rotate of 90 degrees counterclockwise
                angle = -np.pi / 2
                x1_new = cx + np.cos(angle) * (x1 - cx) - np.sin(angle) * (y1 - cy)
                y1_new = cy + np.sin(angle) * (x1 - cx) + np.cos(angle) * (y1 - cy)

                # scale at larger size than minimal distance
                x1 = (x1_new - cx) * (((MIN_DISTANCE + 10 ** (-3)) / 2) / np.sqrt(
                    ((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cx
                y1 = (y1_new - cy) * (((MIN_DISTANCE + 10 ** (-3)) / 2) / np.sqrt(
                    ((x1_new - cx) ** 2) + ((y1_new - cy) ** 2))) + cy

                he_prev = he[0]
                t_heTable_new = t_heTable_new.at[he_prev, 1].set(he[1])  # change prev_he's next_he with he's next_he
                t_heTable_new = t_heTable_new.at[he_prev, 4].set(v_idx_source)

                he_prev_twin = self.t_heTable.at[he_prev, 2].get()
                t_heTable_new = t_heTable_new.at[he_prev_twin, 0].set(he[2])  # idx  # change prev_twin_he's prev_he with he
                t_heTable_new = t_heTable_new.at[he_prev_twin, 3].set(v_idx_source)

                he_next = he[1]
                t_heTable_new = t_heTable_new.at[he_next, 0].set(he[0])  # change next_he's prev_he with he's prev_he
                t_heTable_new = t_heTable_new.at[he_next, 3].set(v_idx_source)  # change next_he's source_vertex with he's source_vertex

                he_next_twin = self.t_heTable.at[he_next, 2].get()
                t_heTable_new = t_heTable_new.at[he_next_twin, 1].set(idx)  # change next_twin_he's next_he with he's twin
                t_heTable_new = t_heTable_new.at[he_next_twin, 4].set(v_idx_source)  # change next twin he's target vertex with he's source vertex

                t_heTable_new = t_heTable_new.at[idx, 0].set(he_next_twin)
                t_heTable_new = t_heTable_new.at[idx, 1].set(self.t_heTable.at[he_next_twin, 1].get())
                t_heTable_new = t_heTable_new.at[idx, 5].set(self.t_heTable.at[he_next_twin, 5].get())

                t_faceTable_new = t_faceTable_new.at[he[5]].set(he_next)
                t_faceTable_new = t_faceTable_new.at[t_heTable_new[idx][5]].set(idx)

                vertTable_new = vertTable_new.at[he[3], 0].set(x1)
                vertTable_new = vertTable_new.at[he[3], 1].set(y1)
                vertTable_new = vertTable_new.at[he[3], 2].set(he[1])

            else:
                pass

        return vertTable_new, t_heTable_new, t_faceTable_new