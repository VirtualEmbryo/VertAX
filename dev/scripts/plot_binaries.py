import os

import numpy as np

from utils_binaries import plot_periodic_voronoi

path = str(input('drag/drop folder: '))[:-1] + '/'

# for dt in range(-1, int((len(os.listdir(path[:-1]))-3)/3)):
for dt in range(0, int((len(os.listdir(path[:-1]))-3)/3)):

    # vertTable = np.load(path + str(dt) + '_simulation_vertTable.npy')
    # faceTable = np.load(path + str(dt) + '_simulation_faceTable.npy')
    # heTable = np.load(path + str(dt) + '_simulation_heTable.npy')

    vertTable = np.load(path + str(dt) + '_vertTable.npy')
    faceTable = np.load(path + str(dt) + '_faceTable.npy')
    heTable = np.load(path + str(dt) + '_heTable.npy')

    n_cells = len(faceTable)
    L_box = np.sqrt(n_cells)

    plot_periodic_voronoi(vertTable,
                          faceTable,
                          heTable,
                          L_box=L_box, multicolor=True, lines=True,
                          vertices=False, path=path, name=dt, save=True, show=False)

