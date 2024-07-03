# import matplotlib.pyplot as plt
# import numpy as np
#
# # Sample data
# x = [0, 1, 2, 3, 4]
# y = [0, 1, 0, 1, 0]
#
# # Define the file path (replace with your actual file path)
# file_path = './parameters_to_delete.txt'
#
# # Read the last line from the file
# with open(file_path, 'r') as file:
#     lines = file.readlines()
#     last_line = lines[-1]
#
# # Split the last line into elements separated by tabs
# elements = last_line.strip().split('\t')
#
# # Convert the list of elements to a NumPy array
# array = np.array(elements, dtype=float)  # Adjust dtype if necessary (e.g., int, str)
#
# print(array)
# values = [0, 0.2, 0.4, 0.6, 0.8]  # Associated numerical values
#
# # Normalize values to range [0, 1]
# norm = plt.Normalize(min(values), max(values))
#
# # Create a colormap (from white to red)
# cmap = plt.get_cmap('Reds')
# colors = cmap(norm(values))
#
# # Create a figure and a set of subplots
# fig, ax = plt.subplots()
#
# # Plot each line segment with the corresponding color
# for i in range(len(x)-1):
#     ax.plot(x[i:i+2], y[i:i+2], '-', color=colors[i])
#
# # Create the colorbar and associate it with the axes
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # Add this line to avoid a warning
# cbar = fig.colorbar(sm, ax=ax, label='Value')
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Colored Lines from White to Red')
#
# plt.show()

import os
import numpy as np

import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color. The keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

def plot_periodic_voronoi(vertTable, faceTable, heTable, L_box, multicolor=True, lines=True, vertices=True, path='.', name='-1', save=False, show=True):

    cmap = get_cmap(len(faceTable))

    all_verts = []
    for face in range(len(faceTable)):

        start_he = faceTable[face]
        he = start_he
        v_source = heTable[he][3]
        verts_sources = np.array([vertTable[v_source][:-1]])
        all_verts.append((vertTable[v_source][:-1]))

        he_offset_x = heTable[he][6]
        he_offset_y = heTable[he][7]
        sum0_offsets = he_offset_x
        sum1_offsets = he_offset_y

        he = heTable[he][1]

        while he != start_he:

            v_source = heTable[he][3]

            all_verts.append((vertTable[v_source][:-1]))

            verts_sources = np.concatenate((verts_sources, (np.array([vertTable[v_source][:-1]]) + np.array([sum0_offsets * L_box, sum1_offsets * L_box]))), axis=0)

            he_offset_x = heTable[he][6]
            he_offset_y = heTable[he][7]
            sum0_offsets += he_offset_x
            sum1_offsets += he_offset_y

            he = heTable[he][1]

        v_source = heTable[he][3]
        verts_sources = np.concatenate((verts_sources, (np.array([vertTable[v_source][:-1]]))), axis=0)

        x, y = zip(*verts_sources)

        if multicolor:
            plt.fill(x, y, color=cmap(face), alpha=.5)
            plt.fill(np.add(x, L_box), np.add(y, L_box), color=cmap(face), alpha=.5)
            plt.fill(np.add(x, -L_box), np.add(y, -L_box), color=cmap(face), alpha=.5)
            plt.fill(np.add(x, -L_box), np.add(y, L_box), color=cmap(face), alpha=.5)
            plt.fill(np.add(x, L_box), np.add(y, -L_box), color=cmap(face), alpha=.5)
            plt.fill(x, np.add(y, L_box), color=cmap(face), alpha=.5)
            plt.fill(x, np.add(y, -L_box), color=cmap(face), alpha=.5)
            plt.fill(np.add(x, L_box), y, color=cmap(face), alpha=.5)
            plt.fill(np.add(x,-L_box), y, color=cmap(face), alpha=.5)

        if lines:
            for i in range(0, len(x)-1, 1):
                plt.plot(x[i:i + 2], y[i:i + 2], '-', color='black')
                plt.plot(tuple(np.add(x[i:i + 2],(L_box, L_box))), tuple(np.add(y[i:i + 2],(L_box, L_box))), '-', color='black')
                plt.plot(tuple(np.add(x[i:i + 2],(-L_box, -L_box))), tuple(np.add(y[i:i + 2],(-L_box, -L_box))), '-', color='black')
                plt.plot(tuple(np.add(x[i:i + 2],(-L_box, -L_box))), tuple(np.add(y[i:i + 2],(L_box, L_box))), '-', color='black')
                plt.plot(tuple(np.add(x[i:i + 2],(L_box, L_box))), tuple(np.add(y[i:i + 2],(-L_box, -L_box))), '-', color='black')
                plt.plot(x[i:i + 2], tuple(np.add(y[i:i + 2],(L_box, L_box))), '-', color='black')
                plt.plot(x[i:i + 2], tuple(np.add(y[i:i + 2],(-L_box, -L_box))), '-', color='black')
                plt.plot(tuple(np.add(x[i:i + 2],(L_box, L_box))), y[i:i + 2], '-', color='black')
                plt.plot(tuple(np.add(x[i:i + 2],(-L_box, -L_box))), y[i:i + 2], '-', color='black')

    if vertices:
        x_all, y_all = zip(*all_verts)
        plt.scatter(x_all, y_all, color='black')

    plt.xlim([0, L_box])
    plt.ylim([0, L_box])

    if save:
        os.makedirs(path + '_images/', exist_ok=True)
        plt.savefig(path + '_images/' + str(name) + '.png')

    if show:
        plt.show()

    plt.clf()

