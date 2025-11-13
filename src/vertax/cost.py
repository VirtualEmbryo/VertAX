from functools import partial

import jax.numpy as jnp
from jax import jit, vmap, Array
from jax.numpy import arange, array, diff, einsum, exp, expand_dims, int32, meshgrid, pi, sqrt, stack
from jax.numpy import sinc as npsinc
from jax.numpy.fft import ifft2

from vertax.geo import get_area

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


# Fourier grid defined by the Nyquist–Shannon sampling theorem
def xigrid():
    """Computes the Fourier grid associated the regular grid defined in the box
    [-s₀/2,s₀/2]x[-s₁/2,s₁/2] and defined by the Nyquist–Shannon sampling theorem.
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
def fft_phase():
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
def fourier_gaussian(mu, sigma, xi):
    """Computes the Fourier transform of the Gaussian distribution
    with mean value μ and covariance matrix Σ (positive semi-definite)
    at points ξ.
    * μ is the mean value (vector of size (2,)).
    * Σ is the 2x2 covariance matrix (positive semi-definite).
    * ξ is an array of size (2,p₀,p₁,p₂,...)

    Returns:
    * The Fourier transform of the Gaussian distribution at points ξ.
        It has the size (p₀,p₁,p₂,...).
    """
    return exp(-1j * (expand_dims(mu, tuple(arange(1, len(xi.shape), dtype=int32))) * xi).sum(axis=0)) * exp(
        -xisigmaxi(sigma, xi) / 2
    )


# Compute ξᵀΣξ
def xisigmaxi(sigma, xi):
    """Computes ξᵀΣξ for a matrix Σ and vectors ξ.
    * Σ is the 2x2 matrix.
    * ξ is an array of size (2,p₀,p₁,p₂,...)

    Returns:
    * ξᵀΣξ of size (p₀,p₁,p₂,...).
    """
    return (xi * einsum("ij,j...->i...", sigma, xi)).sum(axis=0)


# Pre-compute xi, xiexpa, phase, fac, fg
xi, phase = xigrid(), fft_phase()
xiexpa = expand_dims(xi, tuple(arange(1, 2)))
fac = ns.prod() / s.prod() * exp(1j * pi / 2 * (((ns - 1) ** 2 / ns).sum()))
fg = fourier_gaussian(mu, sigma, xi)

###################################
# Change numpy definition of sinc #
###################################


@jit
def sinc(x):
    return npsinc(x / pi)


###############################################
# Fourier Transform Î of a line segment [a,b] #
#  Î(ξ) = |b-a| exp(-iξ(a+b)/2) sinc((b-a)/2) #
###############################################


# FT of all line segments (sum) with precomputed xi
@jit
def sum_line_segment_fourier_transform(x):
    """Computes the Fourier transform of a set of line segments
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
    return (nonodifx * exp(-1j * (midx * xiexpa).sum(axis=0)) * sinc((xiexpa * difx).sum(axis=0) / 2)).sum(axis=0)
    ####


@jit
def gaussian_blur_line_segments(x):
    """Blurs line segments defined in the box [-s₀/2,s₀/2]x[-s₁/2,s₁/2] at regularly spaced points
    with the Gaussian distribution of mean value μ and covariance matrix Σ (positive semi-definite).
    * x is the array of line segments: size ((2,2,m₀,m₁,m₂,...))

    Returns:
    * The blurring of the line segments with the Gaussian distribution.
    """
    return fac * ifft2(fg * sum_line_segment_fourier_transform(x) * phase) * phase


##################
# COST FUNCTIONS #
##################


@partial(jit, static_argnums=(3, 4))
def cost_v2v(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    vertTable_target,
    heTable_target,
    faceTable_target,
    selected_verts=None,
    selected_hes=None,
    selected_faces=None,
    image_target=None,
):
    if selected_verts is None:
        selected_verts = jnp.arange(vertTable.shape[0])
    if selected_hes is None:
        selected_hes = jnp.arange(heTable.shape[0])
    if selected_faces is None:
        selected_faces = jnp.arange(faceTable.shape[0])

    # L_box = jnp.sqrt(len(faceTable))

    def squared_distance(v, vertTable, vertTable_target, width: float, height: float):
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

    def mapped_fn(v):
        return squared_distance(v, vertTable, vertTable_target, width, height)

    distances = vmap(mapped_fn)(selected_verts)

    return (1.0 / (2 * len(distances))) * jnp.sum(distances)


@jit
def cost_mesh2image(
    vertTable: Array,
    heTable: Array,
    faceTable: Array,
    selected_verts,
    selected_hes,
    selected_faces,
    vertTable_target,
    heTable_target,
    faceTable_target,
    image_target,
):
    L_box = jnp.sqrt(len(faceTable))

    starting = (vertTable[heTable[selected_hes, 3], :2]) * 2 / L_box  # (M, 2)
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
            * L_box
        )
        * 2
        / L_box
    )

    he_edges = stack((starting, ending), axis=1)  # (N, 2, 2)
    x = he_edges.transpose(1, 2, 0) - 1  # (2, 2, N)

    # Blur
    image = gaussian_blur_line_segments(x).real

    # # Crop
    # image = image[int(256/6):int(((256*2/6)*2)+(256/6)),int(256/6):int(((256*2/6)*2)+(256/6))]
    # image_target = image_target[int(256/6):int(((256*2/6)*2)+(256/6)),int(256/6):int(((256*2/6)*2)+(256/6))]

    # image = image[85:169,85:169]
    image = image[50:210, 50:210]
    # image_target = image_target[85:169,85:169]
    image_target = image_target[50:210, 50:210]

    # plt.imshow(image_target, cmap='gray', alpha=1.)
    # plt.imshow(image, cmap='hot', alpha=0.5)
    # plt.show()

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
    selected_verts,
    selected_hes,
    selected_faces,
    vertTable_target,
    heTable_target,
    faceTable_target,
    image_target,
):
    def mapped_fn(f):
        return (
            get_area(f, vertTable, heTable, faceTable, width, height)
            - get_area(f, vertTable_target, heTable_target, faceTable_target, width, height)
        ) ** 2  # + \

    # (get_perimeter(f, vertTable, heTable, faceTable) - get_perimeter(f, vertTable_target, heTable_target, faceTable_target))**2
    difference = vmap(mapped_fn)(selected_faces)
    return (1.0 / len(difference)) * jnp.sum(difference)
