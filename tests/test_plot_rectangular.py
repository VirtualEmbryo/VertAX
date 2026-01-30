"""Manual check that a rectangular plot is working."""

from vertax import PbcMesh, plot_mesh


def show_rectangular_mesh() -> None:
    """Manual plot of a PBC mesh with a non square domain."""
    # Settings
    n_cells = 100
    width = 15
    height = 10

    # Initial condition
    mesh = PbcMesh.periodic_voronoi_from_random_seeds(n_cells, width, height, random_key=1)

    plot_mesh(mesh)


if __name__ == "__main__":
    show_rectangular_mesh()
