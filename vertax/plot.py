import os

import numpy as np

import jax.numpy as jnp

import matplotlib.pyplot as plt


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color. The keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_geograph(vertTable, heTable, faceTable, L_box, flip_x=False, flip_y=False, multicolor=True, lines=True, vertices=True, path='.', name='-1', save=False, show=True):

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

        y, x = zip(*verts_sources)

        if flip_x:
            x = tuple(np.array((L_box,) * len(x)) - x)
        if flip_y:
            y = tuple(np.array((L_box,) * len(y)) - y)

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

    plt.gca().set_aspect('equal')

    if save:
        os.makedirs(path, exist_ok=True)
        #plt.savefig(path + str(name) + '.svg', format='svg')
        plt.savefig(path + str(name) + '.png', format='png')

    if show:
        plt.show()

    plt.clf()

def plot_geograph_selected(vertTable, heTable, faceTable, selected_verts, selected_hes, selected_faces, L_box, flip_x=False, flip_y=False, multicolor=True, lines=True, vertices=True, path='.', name='-1', save=False, show=True):

    cmap = get_cmap(len(faceTable))

    all_verts = []
    for face in selected_faces:

        face = int(face) 
        
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

        y, x = zip(*verts_sources)

        if flip_x:
            x = tuple(np.array((L_box,) * len(x)) - x)
        if flip_y:
            y = tuple(np.array((L_box,) * len(y)) - y)

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

    plt.gca().set_aspect('equal')

    if save:
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + str(name) + '.svg', format='svg')
        plt.savefig(path + str(name) + '.png', format='png')

    if show:
        plt.show()

    plt.clf()


def plot_line_tens(vertTable, heTable, faceTable, file_txt, L_box, flip_x=False, flip_y=False, multicolor=False, lines=True, vertices=False, path='.', name='-1', save=False, show=True):

    
    def load_last_np_array(filename):
        with open(filename, 'r') as file:
            content = file.read().strip()
        # Split into separate arrays (assuming each starts with '[' and ends with ']')
        arrays = content.split(']\n[')
        last_array_text = arrays[-1].replace('[', '').replace(']', '')
        # Convert string to NumPy array
        last_array = np.fromstring(last_array_text, sep=' ')
        return last_array

    red_intensity = load_last_np_array(file_txt)
    print(red_intensity)
    red_intensity = (red_intensity-np.min(red_intensity))/(np.max(red_intensity)-np.min(red_intensity))


    cmap = get_cmap(len(faceTable))

    all_verts = []
    for face in range(len(faceTable)):
        
        red_intensity_face = []

        start_he = faceTable[face]
        he = start_he

        red_intensity_face.append(red_intensity[he])
        
        v_source = heTable[he][3]

        verts_sources = np.array([vertTable[v_source][:-1]])
        all_verts.append((vertTable[v_source][:-1]))

        he_offset_x = heTable[he][6]
        he_offset_y = heTable[he][7]
        sum0_offsets = he_offset_x
        sum1_offsets = he_offset_y

        he = heTable[he][1]

        while he != start_he:

            red_intensity_face.append(red_intensity[he])

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

        y, x = zip(*verts_sources)

        if flip_x:
            x = tuple(np.array((L_box,) * len(x)) - x)
        if flip_y:
            y = tuple(np.array((L_box,) * len(y)) - y)

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
                r = red_intensity_face[i]  # Assumi che red_intensity sia una lista con valori tra 0 e 1
                color = (r, 0, 0)  # (rosso, verde, blu)                
                plt.plot(x[i:i + 2], y[i:i + 2], '-', color=color, lw=0.5+r*3.)
                plt.plot(tuple(np.add(x[i:i + 2], (L_box, L_box))), tuple(np.add(y[i:i + 2], (L_box, L_box))), '-', color=color, lw=0.5+r*3.)
                plt.plot(tuple(np.add(x[i:i + 2], (-L_box, -L_box))), tuple(np.add(y[i:i + 2], (-L_box, -L_box))), '-', color=color, lw=0.5+r*3.)
                plt.plot(tuple(np.add(x[i:i + 2], (-L_box, -L_box))), tuple(np.add(y[i:i + 2], (L_box, L_box))), '-', color=color, lw=0.5+r*3.)
                plt.plot(tuple(np.add(x[i:i + 2], (L_box, L_box))), tuple(np.add(y[i:i + 2], (-L_box, -L_box))), '-', color=color, lw=0.5+r*3.)
                plt.plot(x[i:i + 2], tuple(np.add(y[i:i + 2], (L_box, L_box))), '-', color=color, lw=0.5+r*3.)
                plt.plot(x[i:i + 2], tuple(np.add(y[i:i + 2], (-L_box, -L_box))), '-', color=color, lw=0.5+r*3.)
                plt.plot(tuple(np.add(x[i:i + 2], (L_box, L_box))), y[i:i + 2], '-', color=color, lw=0.5+r*3.)
                plt.plot(tuple(np.add(x[i:i + 2], (-L_box, -L_box))), y[i:i + 2], '-', color=color, lw=0.5+r*3.)

    if vertices:
        x_all, y_all = zip(*all_verts)
        plt.scatter(x_all, y_all, color='black')

    plt.xlim([0, L_box])
    plt.ylim([0, L_box])

    plt.gca().set_aspect('equal')

    if save:
        os.makedirs(path, exist_ok=True)
        # plt.savefig(path + str(name) + '.svg', format='svg')
        plt.savefig(path + str(name) + '.png', format='png')

    if show:
        plt.show()

    plt.clf()