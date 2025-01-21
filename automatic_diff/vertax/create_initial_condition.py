import numpy as np
from utils_initial_condition import periodicVoronoi


n_cells = 20
L_box = np.sqrt(n_cells)


if __name__ == '__main__':

    seeds = L_box * np.random.random_sample((n_cells, 2))

    periodicVoronoi(L_box, n_cells, seeds, show=True)

