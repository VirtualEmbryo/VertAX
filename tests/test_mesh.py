"""Private test module to test implementation of a new Mesh class."""

import jax.numpy as jnp
import jax.random
import pytest
from numpy.testing import assert_array_equal
from tifffile import imread

from vertax.pbc import PBCMesh
from vertax.start import create_mesh_from_seeds, create_mesh_from_image
from vertax.plot import plot_mesh


def test_private_constructor_mesh() -> None:
    """Check that a PBCMesh has a private constructor."""
    with pytest.raises(TypeError):
        PBCMesh()


def test_mesh_can_be_privately_created() -> None:
    """Check that a PBCMesh has a private _create function."""
    my_mesh = PBCMesh._create()
    assert hasattr(my_mesh, "vertices")
    assert hasattr(my_mesh, "edges")
    assert hasattr(my_mesh, "faces")


def test_compare_pbc_mesh_with_create_mesh_from_seeds() -> None:
    """Compare the two from_seeds functions. They should be the same."""
    # Initial condition
    n_cells = 100
    key = jax.random.PRNGKey(1)
    L_box = jnp.sqrt(n_cells)
    width = float(L_box)
    height = float(L_box)
    seeds = L_box * jax.random.uniform(key, (n_cells, 2))
    vertTable, heTable, faceTable = create_mesh_from_seeds(seeds)

    my_mesh = PBCMesh.periodic_voronoi_from_random_seeds(n_cells, width, height, random_key=1)

    assert_array_equal(my_mesh.vertices, vertTable)
    assert_array_equal(my_mesh.edges, heTable)
    assert_array_equal(my_mesh.faces, faceTable)


# def test_compare_pbc_mesh_with_create_mesh_from_image() -> None:
#     """Compare the two from_seeds functions. They should be the same."""
#     # Initial condition
#     img = imread("tests/test_image.tif")
#     vertTable, heTable, faceTable = create_mesh_from_image(img)

#     my_mesh = PBCMesh.periodic_voronoi_from_square_image(img)

#     assert_array_equal(my_mesh.vertices, vertTable)
#     assert_array_equal(my_mesh.edges, heTable)
#     assert_array_equal(my_mesh.faces, faceTable)

#     plot_mesh(vertTable, heTable, faceTable, L_box=img.shape[0])


if __name__ == "__main__":
    test_private_constructor_mesh()
    test_mesh_can_be_privately_created()
    test_compare_pbc_mesh_with_create_mesh_from_seeds()
    # test_compare_pbc_mesh_with_create_mesh_from_image()
