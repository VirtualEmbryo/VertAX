import numpy as np
from initial_condition_utils import periodicVoronoi

n_cells = 100
L_box = np.sqrt(n_cells)

if __name__ == '__main__':

    # periodic voronoi

    for i in range(1):

        # random
        seeds = L_box * np.random.random_sample((n_cells, 2))

        periodicVoronoi(L_box, n_cells, seeds, show=True)

