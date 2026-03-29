"""Cost functions collection."""

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.lax import fori_loop
from jax.numpy import arange, array, diff, einsum, exp, expand_dims, int32, meshgrid, pi, sqrt, stack
from jax.numpy import sinc as npsinc
from jax.numpy.fft import ifft2

from vertax.geo import get_area

TARGET_RATIO = 2.0

# import matplotlib.pyplot as plt


# Image size, larger than the cube [-1,1]² --> FFT produces artefacts otherwise
s = array([-3, +3], dtype=float)

# Number of pixels along each axis
ns = array([256, 256], dtype=int32)

# Gaussian distribution parameters
## Mean value
mu = array([0, 0], dtype=float)
## Covariance
sigma = array([[5e-5, 0], [0, 5e-5]], dtype=float)


# Fourier grid defined by the Nyquist-Shannon sampling theorem
def _xigrid() -> Array:
    """Computes the Fourier grid.

      The grid is associated the regular grid defined in the box
    [-s₀/2,s₀/2]x[-s₁/2,s₁/2] and defined by the Nyquist-Shannon sampling theorem.
      * s is the box size (centered at 0).
      * ns is an integer vector defining the size of the grid.
    Returns the Fourier grid.
    """
    return stack(
        meshgrid(
            pi / s[0] * (2 * arange(ns[0], dtype=float) - ns[0] + 1),
            pi / s[1] * (2 * arange(ns[1], dtype=float) - ns[1] + 1),
            indexing="ij",
        )
    )


# FFT phase
def _fft_phase() -> Array:
    """Defines the phase term for the FFT computation.

      * ns is an integer vector defining the size of the grid.
    Returns the phase term.
    """
    return stack(
        meshgrid(
            exp(1j * pi * (1 - ns[0]) / ns[0] * arange(ns[0], dtype=float)),
            exp(1j * pi * (1 - ns[1]) / ns[1] * arange(ns[1], dtype=float)),
            indexing="ij",
        )
    ).prod(axis=0)


# Fourier transform of the Gaussian distribution
def _fourier_gaussian(mu: Array, sigma: Array, xi: Array) -> Array:
    """Computes the Fourier transform of the Gaussian distribution.

    With mean value μ and covariance matrix Σ (positive semi-definite)
    at points ξ.
    * μ is the mean value (vector of size (2,)).
    * Σ is the 2x2 covariance matrix (positive semi-definite).
    * ξ is an array of size (2,p₀,p₁,p₂,...)

    Returns:
    * The Fourier transform of the Gaussian distribution at points ξ.
        It has the size (p₀,p₁,p₂,...).
    """
    return exp(-1j * (expand_dims(mu, tuple(arange(1, len(xi.shape), dtype=int32))) * xi).sum(axis=0)) * exp(
        -_xisigmaxi(sigma, xi) / 2
    )


# Compute ξᵀΣξ
def _xisigmaxi(sigma: Array, xi: Array) -> Array:
    """Computes ξᵀΣξ for a matrix Σ and vectors ξ.

    * Σ is the 2x2 matrix.
    * ξ is an array of size (2,p₀,p₁,p₂,...)

    Returns:
    * ξᵀΣξ of size (p₀,p₁,p₂,...).
    """
    return (xi * einsum("ij,j...->i...", sigma, xi)).sum(axis=0)


# Pre-compute xi, xiexpa, phase, fac, fg
xi, phase = _xigrid(), _fft_phase()
xiexpa = expand_dims(xi, tuple(arange(1, 2)))
fac = ns.prod() / s.prod() * exp(1j * pi / 2 * (((ns - 1) ** 2 / ns).sum()))
fg = _fourier_gaussian(mu, sigma, xi)

###################################
# Change numpy definition of sinc #
###################################


@jit
def _sinc(x: Array) -> Array:
    """Sinus cardinal."""
    return npsinc(x / pi)


###############################################
# Fourier Transform Î of a line segment [a,b] #
#  Î(ξ) = |b-a| exp(-iξ(a+b)/2) sinc((b-a)/2) #
###############################################


# FT of all line segments (sum) with precomputed xi
@jit
def _sum_line_segment_fourier_transform(x: Array) -> Array:
    """Computes the Fourier transform of a set of line segments.

    * x is the array of line segments, the size is ((2,2,m)).
        x[:,:.i] is the line segment defined by the points
        (x[0,0,i],x[0,1,i]) and (x[1,0,i],x[1,1,i])
    * Needs a precomputed ξ, array of points where to compute
        the Fourier transform. The size is ((2,n)).
        The first dimension is for the Fourier vector coordinates.

    Returns:
    * The Fourier transform of the set of line segments at points ξ.
        It has the size (n,n).
    """
    # midx,difx=x.sum(axis=0)/2,diff(x,axis=0)[0]
    # return (sqrt((difx**2).sum(axis=0,keepdims=True)).T*exp(-1j*(midx.T@xi))*sinc((difx.T@xi)/2)).sum(axis=0)
    ####
    midx, difx = x.sum(axis=0) / 2, diff(x, axis=0)[0]
    midx, difx = expand_dims(midx, (-1, -2)), expand_dims(difx, (-1, -2))
    nonodifx = sqrt((difx**2).sum(axis=0))  # 1            no weight
    # nonodifx=1/sqrt((difx**2).sum(axis=0))  # (1/x)**2     weight
    # nonodifx*=exp(-nonodifx)                # exp(-x)      weight
    # nonodifx=(difx**2).sum(axis=0)           # x            weight
    # nonodifx=((difx**2).sum(axis=0))**2       # x**2         weight
    # nonodifx*=exp(nonodifx)                 # exp(-x)      weight
    return (nonodifx * exp(-1j * (midx * xiexpa).sum(axis=0)) * _sinc((xiexpa * difx).sum(axis=0) / 2)).sum(axis=0)
    ####


@jit
def _gaussian_blur_line_segments(x: Array) -> Array:
    """Blurs line segments.

    The line segments are defined in the box [-s₀/2,s₀/2]x[-s₁/2,s₁/2] at regularly spaced points
    with the Gaussian distribution of mean value μ and covariance matrix Σ (positive semi-definite).
    * x is the array of line segments: size ((2,2,m₀,m₁,m₂,...))

    Returns:
    * The blurring of the line segments with the Gaussian distribution.
    """
    return fac * ifft2(fg * _sum_line_segment_fourier_transform(x) * phase) * phase


##################
# COST FUNCTIONS #
##################


@partial(jit, static_argnums=(3, 4))
def cost_v2v(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    vertTable_target: Array,
    _heTable_target: Array,
    _faceTable_target: Array,
    selected_verts: Array | None = None,
    selected_hes: Array | None = None,
    selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Example of a cost function."""
    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])
    if selected_hes is None:
        selected_hes = jnp.arange(heTable.shape[0])
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    def squared_distance(v: Array, vertTable: Array, vertTable_target: Array, width: float, height: float) -> Array:
        return (
            jnp.min(
                jnp.array(
                    [
                        (
                            (vertTable[v][0] - vertTable_target[v][0]) ** 2
                            + (vertTable[v][1] - vertTable_target[v][1]) ** 2
                        )
                        ** 0.5,
                        (
                            (vertTable[v][0] - (vertTable_target[v][0] + width)) ** 2
                            + (vertTable[v][1] - vertTable_target[v][1]) ** 2
                        )
                        ** 0.5,
                        (
                            (vertTable[v][0] - (vertTable_target[v][0] - width)) ** 2
                            + (vertTable[v][1] - vertTable_target[v][1]) ** 2
                        )
                        ** 0.5,
                        (
                            (vertTable[v][0] - vertTable_target[v][0]) ** 2
                            + (vertTable[v][1] - (vertTable_target[v][1] + height)) ** 2
                        )
                        ** 0.5,
                        (
                            (vertTable[v][0] - vertTable_target[v][0]) ** 2
                            + (vertTable[v][1] - (vertTable_target[v][1] - height)) ** 2
                        )
                        ** 0.5,
                        (
                            (vertTable[v][0] - (vertTable_target[v][0] + width)) ** 2
                            + (vertTable[v][1] - (vertTable_target[v][1] + height)) ** 2
                        )
                        ** 0.5,
                        (
                            (vertTable[v][0] - (vertTable_target[v][0] + width)) ** 2
                            + (vertTable[v][1] - (vertTable_target[v][1] - height)) ** 2
                        )
                        ** 0.5,
                        (
                            (vertTable[v][0] - (vertTable_target[v][0] - width)) ** 2
                            + (vertTable[v][1] - (vertTable_target[v][1] + height)) ** 2
                        )
                        ** 0.5,
                        (
                            (vertTable[v][0] - (vertTable_target[v][0] - width)) ** 2
                            + (vertTable[v][1] - (vertTable_target[v][1] - height)) ** 2
                        )
                        ** 0.5,
                    ]
                )
            )
        ) ** 2

    def mapped_fn(v: Array) -> Array:
        return squared_distance(v, vertTable, vertTable_target, width, height)

    distances = vmap(mapped_fn)(selected_verts)

    return (1.0 / (2 * len(distances))) * jnp.sum(distances)


@jit
def cost_mesh2image(
    vertTable: Array,
    heTable: Array,
    _faceTable: Array,
    width: float,
    height: float,
    _vertTable_target: Array,
    _heTable_target: Array,
    _faceTable_target: Array,
    _selected_verts: Array,
    selected_hes: Array,
    _selected_faces: Array,
    image_target: Array,
) -> Array:
    """Example of a cost function."""
    starting = (vertTable[heTable[selected_hes, 3], :2]) * 2 / [width, height]  # (M, 2)
    # ending = (vertTable[heTable[selected_hes, 4], :2]) * 2 / L_box  # (M, 2)
    ending = (
        (
            vertTable[heTable[selected_hes, 4], :2]
            + jnp.stack(
                [
                    heTable[selected_hes, 6],
                    heTable[selected_hes, 7],
                ],
                axis=-1,
            )
            * [width, height]
        )
        * 2
        / [width, height]
    )

    he_edges = stack((starting, ending), axis=1)  # (N, 2, 2)
    x = he_edges.transpose(1, 2, 0) - 1  # (2, 2, N)

    # Blur
    image = _gaussian_blur_line_segments(x).real

    # Normalization
    image = image / image.sum()
    image_target = image_target / image_target.sum()

    l2_norm = jnp.linalg.norm(jnp.sqrt(jnp.sum(((image - image_target) ** 2) * (1 - image_target), axis=-1)).flatten())

    return l2_norm


@jit
def cost_areas(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    width: float,
    height: float,
    _selected_verts: Array,
    _selected_hes: Array,
    selected_faces: Array,
    vertTable_target: Array,
    heTable_target: Array,
    faceTable_target: Array,
    _image_target: Array,
) -> Array:
    """Example of a cost function."""

    def mapped_fn(f: Array) -> Array:
        return (
            get_area(f, vertTable, heTable, faceTable, width, height)
            - get_area(f, vertTable_target, heTable_target, faceTable_target, width, height)
        ) ** 2  # + \

    difference = vmap(mapped_fn)(selected_faces)
    return (1.0 / len(difference)) * jnp.sum(difference)


@jit
def cost_ratio(
    vertTable: Array,
    _angTable: Array | None = None,
    _heTable: Array | None = None,
    _faceTable: Array | None = None,
    _vertTable_target: Array | None = None,
    _angTable_target: Array | None = None,
    _heTable_target: Array | None = None,
    _faceTable_target: Array | None = None,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Example of a cost function."""
    # Compute pairwise squared distances by broadcasting
    # diff shape: (n, n, d)
    diff = vertTable[:, None, :] - vertTable[None, :, :]
    sq_dists = jnp.sum(diff**2, axis=-1)  # (n, n)
    distances = jnp.sqrt(sq_dists + 1e-12)  # small eps for numerical stability

    # For each row, get index of farthest point
    max_distances = jnp.max(distances, axis=1)
    max_idx_1 = jnp.argmax(max_distances)  # index of row with largest max distance
    long_axis = max_distances[max_idx_1]  # longest distance (already excludes diagonal)
    max_idx_2 = jnp.argmax(distances[max_idx_1])  # the farthest point from max_idx_1

    max_point_1 = vertTable[max_idx_1]
    max_point_2 = vertTable[max_idx_2]
    long_axis_vector = max_point_2 - max_point_1
    long_axis_vector = long_axis_vector / (jnp.linalg.norm(long_axis_vector) + 1e-12)

    # Short axis (perpendicular)
    short_axis_vector = jnp.array([long_axis_vector[1], -long_axis_vector[0]])

    # Project points onto short axis: signed distances
    # If vertTable is (n,2) this works. If higher dim, modify accordingly.
    signed_distances_to_long_axis = jnp.dot(vertTable - max_point_1, short_axis_vector)

    short_axis = jnp.max(signed_distances_to_long_axis) - jnp.min(signed_distances_to_long_axis)

    # Numerically stabilize division
    ratio = long_axis / (short_axis + 1e-12)
    return (ratio - TARGET_RATIO) ** 2


@jit
def cost_checkerboard(
    vertTable: Array,
    _angTable: Array,
    heTable: Array,
    faceTable: Array,
    _vertTable_target: Array | None = None,
    _angTable_target: Array | None = None,
    _heTable_target: Array | None = None,
    _faceTable_target: Array | None = None,
    _selected_verts: Array | None = None,
    _selected_hes: Array | None = None,
    _selected_faces: Array | None = None,
    _image_target: Array | None = None,
) -> Array:
    """Example of a cost function."""

    def body_fun(i: int, current_len: Array) -> Array:
        idx = 2 * i
        he_twin = heTable[idx, 2]
        he_face = heTable[idx, 7]
        he_fate = faceTable[he_face, 1]
        he_face_twin = heTable[he_twin, 7]
        he_fate_twin = faceTable[he_face_twin, 1]
        v_source = heTable[idx, 3] - 2
        v_target = heTable[idx, 4] - 2
        pos_source = vertTable[v_source]
        pos_target = vertTable[v_target]
        return current_len + jnp.where(
            jnp.logical_and(he_fate == he_fate_twin, heTable[idx, 3] > 0), jnp.sum((pos_target - pos_source) ** 2), 0.0
        )

    n_edges = heTable.shape[0] // 2
    return fori_loop(0, n_edges, body_fun, 0.0)


@partial(jit, static_argnums=(3, 4))
def cost_tem_halfedge(
    vertTable,
    heTable,
    faceTable,
    width,
    height,
    vertTable_target,
    heTable_target,
    faceTable_target,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    image_target=None,
):
    """
    TEM-inspired loss using half-edge dual graph (cells).
    """

    # ---------- helpers ----------
    def pairwise_periodic_distances(x, y):
        dx = x[:, None, 0] - y[None, :, 0]
        dy = x[:, None, 1] - y[None, :, 1]

        dx = jnp.minimum(jnp.abs(dx), width - jnp.abs(dx))
        dy = jnp.minimum(jnp.abs(dy), height - jnp.abs(dy))

        return jnp.sqrt(dx**2 + dy**2 + 1e-12)

    def sinkhorn(a, b, C, eps=5e-2, n_iters=50):
        K = jnp.exp(-C / eps)
        u = jnp.ones_like(a)
        v = jnp.ones_like(b)

        def body(_, state):
            u, v = state
            u = a / (K @ v + 1e-12)
            v = b / (K.T @ u + 1e-12)
            return (u, v)

        u, v = fori_loop(0, n_iters, body, (u, v))
        return jnp.outer(u, v) * K

    # ---------- dual graph ----------
    def build_dual_adj(heTable, n_faces):
        face = heTable[:, 5].astype(jnp.int32)
        twin = heTable[:, 2].astype(jnp.int32)

        face_twin = face[twin]

        valid = face != face_twin
        keep = valid & (face < face_twin)

        weight = keep.astype(jnp.float32)

        A = jnp.zeros((n_faces, n_faces))

        A = A.at[face, face_twin].add(weight)
        A = A.at[face_twin, face].add(weight)

        return A

    def structure_matrix(A):
        A2 = A @ A
        return 2.0 - A - 0.5 * A2

    # ---------- centroids ----------
    # faceTable stores a half-edge index → get its source vertex
    he_idx = faceTable[:, 0].astype(jnp.int32)

    X = vertTable[heTable[he_idx, 3].astype(jnp.int32), :2]
    Y = vertTable_target[heTable_target[faceTable_target[:, 0].astype(jnp.int32), 3].astype(jnp.int32), :2]

    N = X.shape[0]
    M = Y.shape[0]

    a = jnp.ones(N) / N
    b = jnp.ones(M) / M

    # ---------- costs ----------
    C_geom = pairwise_periodic_distances(X, Y)

    A1 = build_dual_adj(heTable, N)
    A2 = build_dual_adj(heTable_target, M)

    S1 = structure_matrix(A1)
    S2 = structure_matrix(A2)

    # structure comparison (non-periodic, abstract space)
    diff = S1[:, None, :] - S2[None, :, :]
    C_struct = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)

    alpha = 0.5
    C_total = alpha * C_geom + (1.0 - alpha) * C_struct

    gamma = sinkhorn(a, b, C_total)

    return jnp.sum(gamma * C_total)


@partial(jit, static_argnums=(3, 4))
def cost_v2v_tem(vertTable, heTable, faceTable, width, height, vertTable_target, heTable_target, faceTable_target, selected_verts=None, selected_hes=None, selected_faces=None, image_target=None):
    return (
        0.99 * cost_v2v(vertTable, heTable, faceTable, width, height, vertTable_target, heTable_target, faceTable_target, selected_verts=None, selected_hes=None, selected_faces=None, image_target=None)
        + 0.01 * cost_tem_halfedge(vertTable, heTable, faceTable, width, height, vertTable_target, heTable_target, faceTable_target, selected_verts=None, selected_hes=None, selected_faces=None, image_target=None)
    )
