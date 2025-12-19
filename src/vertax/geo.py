"""Geometric functions over a 2D mesh."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, jit, vmap


@partial(jit, static_argnums=(3, 4))
def _sum_edges(
    face: Array, heTable: Array, faceTable: Array, fun: Callable[[Array, Array], Array], max_iter: int
) -> Array:
    """Sums edge contributions for a face using lax.scan.

    Running for a fixed number of iterations (max_iter).
    """
    start_he = faceTable.at[face].get()

    # 1. Process the first edge (start_he)
    # This computes the first contribution and sets the initial state
    res_0 = fun(start_he, jnp.array([0.0, 0.0, 0.0]))

    # 2. Setup initial carry for the scan
    # The scan will process the (max_iter - 1) *remaining* edges
    # state = (last_he, cumulative_res, has_stopped_flag)
    initial_carry = (start_he, res_0, False)

    # 3. Create a dummy array for scan to iterate over
    # It will run (max_iter - 1) times
    xs = jnp.arange(max_iter - 1)

    def scan_body(carry: tuple[Array, Array, Array | bool], _: int) -> tuple[tuple[Array, Array, Array | bool], float]:
        """The body of the lax.scan.

        `carry` is the state from the *previous* iteration.
        `i` is the current loop index (from xs).
        """
        previous_he, previous_res, has_stopped = carry

        # Get the *next* half-edge in the face loop
        current_he = heTable.at[previous_he, 1].get()

        # Check if this new edge is the one we started with
        is_start_node = current_he == start_he

        # Latch the 'has_stopped' flag. Once it's True, it stays True.
        has_stopped = has_stopped | is_start_node

        # Calculate the contribution of the *current* edge,
        # using the *previous* cumulative result (for offsets)
        contribution = fun(current_he, previous_res)

        # Accumulate the result,
        # but only if the 'has_stopped' flag is not set.
        new_res = jax.lax.select(
            has_stopped,
            previous_res,  # If stopped, result is unchanged
            previous_res + contribution,  # If running, accumulate
        )

        # The new carry state for the *next* iteration
        new_carry = (current_he, new_res, has_stopped)

        # scan requires a 'y' output, we don't need it
        return new_carry, 0.0

    # Run the scan
    final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)  # type: ignore

    # The final result is the cumulative 'res' from the final state
    _, final_res, _ = final_carry
    return final_res


@jit
def get_length(he: Array, vertTable: Array, heTable: Array, _faceTable: Array, width: float, height: float) -> Array:
    """Get the lengths of given half-edges."""
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()

    x0, y0 = vertTable.at[v_source, :2].get()  # source vertex
    he_offset_x1 = heTable.at[he, 6].get() * width  # target offset
    he_offset_y1 = heTable.at[he, 7].get() * height
    x1, y1 = vertTable.at[v_target, :2].get() + jnp.array([he_offset_x1, he_offset_y1])  # target vertex

    length = jnp.hypot(x1 - x0, y1 - y0)
    return length


@jit
def get_length_with_offset(
    he: Array, vertTable: Array, heTable: Array, _faceTable: Array, width: float, height: float
) -> Array:
    """Get the length and associated offsets for given half-edges."""
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    x0, y0 = vertTable.at[v_source, :2].get()  # source vertex
    he_offset_x1 = heTable.at[he, 6].get() * width  # target offset
    he_offset_y1 = heTable.at[he, 7].get() * height
    x1, y1 = vertTable.at[v_target, :2].get() + jnp.array([he_offset_x1, he_offset_y1])  # target vertex

    length = jnp.hypot(x1 - x0, y1 - y0)
    return jnp.array([length, he_offset_x1, he_offset_y1])


@partial(jit, static_argnums=(4, 5, 6))
def get_perimeter(
    face: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float, max_iter: int
) -> Array:
    """Get the perimeters of given faces."""

    def fun(he: Array, _: Array) -> Array:
        return get_length_with_offset(he, vertTable, heTable, faceTable, width, height)

    return _sum_edges(face, heTable, faceTable, fun, max_iter)[0]


@jit
def _compute_numerator(he: Array, res: Array, vertTable: Array, heTable: Array, width: float, height: float) -> Array:
    x_offset, y_offset = res.at[1].get(), res.at[2].get()
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    x0 = vertTable.at[v_source, 0].get() + x_offset  # source
    y0 = vertTable.at[v_source, 1].get() + y_offset
    he_offset_x1 = heTable.at[he, 6].get() * width  # offset target
    he_offset_y1 = heTable.at[he, 7].get() * height
    x1 = vertTable.at[v_target, 0].get() + x_offset + he_offset_x1  # target
    y1 = vertTable.at[v_target, 1].get() + y_offset + he_offset_y1

    return jnp.array([(x0 * y1) - (x1 * y0), he_offset_x1, he_offset_y1])


@partial(jit, static_argnums=(4, 5, 6))
def get_area(
    face: Array, vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float, max_iter: int
) -> Array:
    """Get area of given faces (using shoelace formula)."""

    def fun(he: Array, res: Array) -> Array:
        return _compute_numerator(he, res, vertTable, heTable, width, height)

    return 0.5 * jnp.abs(_sum_edges(face, heTable, faceTable, fun, max_iter)[0])


@jit
def _update_he(
    he: Array, vertTable: Array, heTable: Array, width: float, height: float
) -> tuple[Array, Array, Array, Array]:
    v_idx_target = heTable.at[he, 4].get()
    v_x = vertTable.at[v_idx_target, 0].get()
    v_y = vertTable.at[v_idx_target, 1].get()
    offset_x_target = jnp.where(v_x < 0.0, -1, jnp.where(v_x > width, +1, 0))
    offset_y_target = jnp.where(v_y < 0.0, -1, jnp.where(v_y > height, +1, 0))

    v_idx_source = heTable.at[he, 3].get()
    v_x = vertTable.at[v_idx_source, 0].get()
    v_y = vertTable.at[v_idx_source, 1].get()
    offset_x_source = jnp.where(v_x < 0.0, -1, jnp.where(v_x > width, +1, 0))
    offset_y_source = jnp.where(v_y < 0.0, -1, jnp.where(v_y > height, +1, 0))

    return offset_x_target, offset_y_target, offset_x_source, offset_y_source


@jit
def _move_vertex_inside(v: Array, vertTable: Array, width: float, height: float) -> tuple[Array, Array]:
    v_x = vertTable.at[v, 0].get()
    v_y = vertTable.at[v, 1].get()
    v_x = jnp.where(v_x < 0.0, v_x + width, jnp.where(v_x > width, v_x - width, v_x))
    v_y = jnp.where(v_y < 0.0, v_y + height, jnp.where(v_y > height, v_y - height, v_y))

    return v_x, v_y


@jit
def update_pbc(
    vertTable: Array, heTable: Array, faceTable: Array, width: float, height: float
) -> tuple[Array, Array, Array]:
    """Updating vertices positions and offsets for periodic boundary conditions."""

    def mapped_offsets(he: Array) -> tuple[Array, Array, Array, Array]:
        return _update_he(he, vertTable, heTable, width, height)

    offset_x_target, offset_y_target, offset_x_source, offset_y_source = vmap(mapped_offsets)(jnp.arange(len(heTable)))
    heTable = heTable.at[:, 6].add(+offset_x_target - offset_x_source)
    heTable = heTable.at[:, 7].add(+offset_y_target - offset_y_source)

    def mapped_vertices(v: Array) -> tuple[Array, Array]:
        return _move_vertex_inside(v, vertTable, width, height)

    v_x, v_y = vmap(mapped_vertices)(jnp.arange(len(vertTable)))
    vertTable = vertTable.at[:, 0].set(v_x)
    vertTable = vertTable.at[:, 1].set(v_y)

    return vertTable, heTable, faceTable


def select_verts_hes_faces_inside(
    vertTable: Array, heTable: Array, faceTable: Array, x_min: float, x_max: float, y_min: float, y_max: float
) -> tuple[Array, Array, Array]:
    """Get a selection of all vertices, edges and faces for faces that lie totally inside the bounds."""
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
            if not ((y_min <= vert_y <= y_max) and (x_min <= vert_x <= x_max)):
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


# ==========
# Bounded
# ==========
@jit
def get_edge_length(he: Array, vertTable: Array, heTable: Array) -> Array:
    """For a bounded mesh, get the length of straight edges, and 0 for boundary edges."""
    v_source = heTable.at[he, 3].get()
    v_target = heTable.at[he, 4].get()
    mask = v_source != 0
    x0 = vertTable.at[v_source, 0].get()
    y0 = vertTable.at[v_source, 1].get()
    x1 = vertTable.at[v_target, 0].get()
    y1 = vertTable.at[v_target, 1].get()

    return jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * mask


@jit
def get_chord_length(he: Array, vertTable: Array, heTable: Array) -> Array:
    """For a bounded mesh, get the chord lengths of boundary edges, and 0 for non-boundary edges."""
    v_source = heTable.at[he, 5].get()
    v_target = heTable.at[he, 6].get()
    mask = v_source != 0
    x0 = vertTable.at[v_source, 0].get()
    y0 = vertTable.at[v_source, 1].get()
    x1 = vertTable.at[v_target, 0].get()
    y1 = vertTable.at[v_target, 1].get()

    return jnp.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * mask


@jit
def get_surface_length(he: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    """For a bounded mesh, get the lengths of boundary edges, and 0 for non-boundary edges."""
    chordlen = get_chord_length(he, vertTable, heTable)
    angle = angTable.at[he].get()

    return chordlen / jnp.sinc(angle / jnp.pi)


@jit
def _area_factor_between_arc_and_chord(angle: Array) -> Array:
    sinang = jnp.sin(angle)
    return (angle - sinang * jnp.cos(angle)) / (2 * sinang**2)


@jit
def _compute_contributions(he: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    v_source = heTable.at[he, 3].get() + heTable.at[he, 5].get()
    v_target = heTable.at[he, 4].get() + heTable.at[he, 6].get() - 1
    x0 = vertTable.at[v_source, 0].get()
    y0 = vertTable.at[v_source, 1].get()
    x1 = vertTable.at[v_target, 0].get()
    y1 = vertTable.at[v_target, 1].get()

    angle = angTable.at[he].get()
    chordlen = get_chord_length(he, vertTable, heTable)

    return jnp.array([x0 * y1 - x1 * y0, (chordlen**2) * _area_factor_between_arc_and_chord(angle)])


@jit
def get_area_bounded(face: Array, vertTable: Array, angTable: Array, heTable: Array, faceTable: Array) -> Array:
    """Get the areas of given faces."""
    start_he = faceTable.at[face, 0].get()
    res_0 = _compute_contributions(start_he, vertTable, angTable, heTable)
    initial_carry = (start_he, res_0, False)
    xs = jnp.arange(11)

    def scan_body(carry: tuple[Array, Array, Array | bool], _: int) -> tuple[tuple[Array, Array, Array], float]:
        previous_he, previous_res, has_stopped = carry
        current_he = heTable.at[previous_he, 1].get()
        is_start_node = current_he == start_he
        has_stopped = has_stopped | is_start_node
        contribution = _compute_contributions(current_he, vertTable, angTable, heTable)
        new_res = jax.lax.select(
            has_stopped,
            previous_res,  # If stopped, result is unchanged
            previous_res + contribution,  # If running, accumulate
        )
        return (current_he, new_res, has_stopped), 0.0

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)  # type: ignore
    _, final_res, _ = final_carry

    return 0.5 * jnp.sum(jnp.abs(final_res))


@jit
def get_any_length(he: Array, vertTable: Array, angTable: Array, heTable: Array) -> Array:
    """Get bounded length for both surface and inside edges."""
    selector = heTable.at[he, 3].get() != 0
    flat_len = get_edge_length(he, vertTable, heTable)
    curved_len = get_surface_length(he, vertTable, angTable, heTable)
    return jnp.where(selector, flat_len, curved_len)


@jit
def get_perimeter_bounded(face: Array, vertTable: Array, angTable: Array, heTable: Array, faceTable: Array) -> Array:
    """Get the perimeters of given faces."""
    start_he = faceTable.at[face, 0].get()
    res_0 = get_any_length(start_he, vertTable, angTable, heTable)
    initial_carry = (start_he, res_0, False)
    xs = jnp.arange(11)

    def scan_body(carry: tuple[Array, Array, Array | bool], _i: int) -> tuple[tuple[Array, Array, Array | bool], float]:
        previous_he, previous_res, has_stopped = carry
        current_he = heTable.at[previous_he, 1].get()
        is_start_node = current_he == start_he
        has_stopped = has_stopped | is_start_node
        contribution = get_any_length(current_he, vertTable, angTable, heTable)
        new_res = jax.lax.select(
            has_stopped,
            previous_res,  # If stopped, result is unchanged
            previous_res + contribution,  # If running, accumulate
        )
        return (current_he, new_res, has_stopped), 0.0

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, xs)  # type: ignore
    _, final_res, _ = final_carry
    return final_res
