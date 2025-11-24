"""Private test module to test implementation of a new Mesh class."""

import jax.numpy as jnp
import jax.random
import pytest
from numpy.testing import assert_array_equal
from tifffile import imread

from vertax.pbc import PBCMesh
from vertax.plot import plot_mesh
from vertax.start import create_mesh_from_image, create_mesh_from_seeds


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


@pytest.mark.long
def no_pytest_test_compare_pbc_mesh_with_create_mesh_from_image() -> None:
    """Compare the two from_seeds functions. They should be the same."""
    plot = False
    # imread tiff = Y is the first axis, X the second.
    img = imread("tests/test_image.tif")[:-101, :]  # non rect, odd and pair dimensions
    print("create mesh from image...")
    vertTable, heTable, faceTable = create_mesh_from_image(img)
    if plot:
        plot_mesh(
            vertTable,
            heTable,
            faceTable,
            width=2 * img.shape[1],
            height=2 * img.shape[0],
            path="ref_image",
            save=True,
            show=False,
        )

    print("PBC now...")
    my_mesh = PBCMesh.periodic_from_image(img)
    if plot:
        plot_mesh(
            my_mesh.vertices,
            my_mesh.edges,
            my_mesh.faces,
            width=2 * img.shape[1],
            height=2 * img.shape[0],
            path="pbc_image",
            save=True,
            show=False,
        )

    assert_array_equal(my_mesh.vertices, vertTable)
    assert_array_equal(my_mesh.edges, heTable)
    assert_array_equal(my_mesh.faces, faceTable)


if __name__ == "__main__":
    test_private_constructor_mesh()
    test_mesh_can_be_privately_created()
    test_compare_pbc_mesh_with_create_mesh_from_seeds()
    no_pytest_test_compare_pbc_mesh_with_create_mesh_from_image()
