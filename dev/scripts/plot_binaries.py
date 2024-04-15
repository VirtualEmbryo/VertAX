import os

import numpy as np

from plot_utils import plot_periodic_voronoi

path = str(input('path: [./binaries_[zero][beta]_[i]/] '))

for dt in range(len(os.listdir(path))-2):

    vertTable = np.load(path + str(dt) + '_vertTable.npy')
    faceTable = np.load(path + str(dt) + '_faceTable.npy')
    heTable = np.load(path + str(dt) + '_heTable.npy')

    n_cells = len(faceTable)
    L_box = np.sqrt(n_cells)

    plot_periodic_voronoi(vertTable,
                          faceTable,
                          heTable,
                          L_box=L_box, multicolor=True, lines=True, vertices=False, name=dt, save=True, show=False)

