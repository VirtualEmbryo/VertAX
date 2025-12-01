import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color. The keyword argument name must be a standard mpl colormap name.
    """
    return plt.cm.get_cmap(name, n)


def plot_mesh(
    vertTable,
    heTable,
    faceTable,
    width: float,
    height: float,
    flip_x=False,
    flip_y=False,
    multicolor=True,
    lines=True,
    vertices=True,
    path=".",
    name="-1",
    save=False,
    show=True,
):
    cmap = get_cmap(len(faceTable))
    all_verts = []
    for face in range(len(faceTable)):
        start_he = faceTable[face]
        he = start_he

        v_source = heTable[he][3]
        verts_sources = np.array([vertTable[v_source]])
        all_verts.append(vertTable[v_source])

        he_offset_x = heTable[he][6]
        he_offset_y = heTable[he][7]
        sum0_offsets = he_offset_x
        sum1_offsets = he_offset_y

        he = heTable[he][1]

        while he != start_he:
            v_source = heTable[he][3]

            all_verts.append(vertTable[v_source])

            verts_sources = np.concatenate(
                (
                    verts_sources,
                    (np.array([vertTable[v_source]]) + np.array([sum0_offsets * width, sum1_offsets * height])),
                ),
                axis=0,
            )

            he_offset_x = heTable[he][6]
            he_offset_y = heTable[he][7]
            sum0_offsets += he_offset_x
            sum1_offsets += he_offset_y

            he = heTable[he][1]

        v_source = heTable[he][3]
        verts_sources = np.concatenate((verts_sources, (np.array([vertTable[v_source]]))), axis=0)
        x, y = zip(*verts_sources, strict=False)

        if flip_x:
            x = tuple(np.array((width,) * len(x)) - x)
        if not flip_y:
            y = tuple(np.array((height,) * len(y)) - y)

        if multicolor:
            plt.fill(x, y, color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(x, np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(x, np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), y, color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), y, color=cmap(face), alpha=0.5)

        if lines:
            for i in range(0, len(x) - 1, 1):
                plt.plot(x[i : i + 2], y[i : i + 2], "-", color="black")
                plt.plot(
                    tuple(np.add(x[i : i + 2], (width, width))),
                    tuple(np.add(y[i : i + 2], (height, height))),
                    "-",
                    color="black",
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (-width, -width))),
                    tuple(np.add(y[i : i + 2], (-height, -height))),
                    "-",
                    color="black",
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (-width, -width))),
                    tuple(np.add(y[i : i + 2], (height, height))),
                    "-",
                    color="black",
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (width, width))),
                    tuple(np.add(y[i : i + 2], (-height, -height))),
                    "-",
                    color="black",
                )
                plt.plot(x[i : i + 2], tuple(np.add(y[i : i + 2], (height, height))), "-", color="black")
                plt.plot(x[i : i + 2], tuple(np.add(y[i : i + 2], (-height, -height))), "-", color="black")
                plt.plot(tuple(np.add(x[i : i + 2], (width, width))), y[i : i + 2], "-", color="black")
                plt.plot(tuple(np.add(x[i : i + 2], (-width, -width))), y[i : i + 2], "-", color="black")

    if vertices:
        all_verts = np.array(all_verts)
        plt.scatter(all_verts[:, 0], height - all_verts[:, 1], color="black")

    plt.xlim([0, width])
    plt.ylim([0, height])

    plt.gca().set_aspect("equal")

    if save:
        os.makedirs(path, exist_ok=True)
        # plt.savefig(path + str(name) + '.svg', format='svg')
        plt.savefig(path + str(name) + ".png", format="png")

    if show:
        plt.show()

    plt.clf()


def plot_mesh_selected(
    vertTable,
    heTable,
    faceTable,
    selected_verts,
    selected_hes,
    selected_faces,
    width: float,
    height: float,
    flip_x=False,
    flip_y=False,
    multicolor=True,
    lines=True,
    vertices=True,
    path=".",
    name="-1",
    save=False,
    show=True,
):
    cmap = get_cmap(len(faceTable))

    all_verts = []
    for face in selected_faces:
        face = int(face)

        start_he = faceTable[face]
        he = start_he

        v_source = heTable[he][3]

        verts_sources = np.array([vertTable[v_source]])
        all_verts.append(vertTable[v_source])

        he_offset_x = heTable[he][6]
        he_offset_y = heTable[he][7]
        sum0_offsets = he_offset_x
        sum1_offsets = he_offset_y

        he = heTable[he][1]

        while he != start_he:
            v_source = heTable[he][3]

            all_verts.append(vertTable[v_source])

            verts_sources = np.concatenate(
                (
                    verts_sources,
                    (np.array([vertTable[v_source]]) + np.array([sum0_offsets * width, sum1_offsets * height])),
                ),
                axis=0,
            )

            he_offset_x = heTable[he][6]
            he_offset_y = heTable[he][7]
            sum0_offsets += he_offset_x
            sum1_offsets += he_offset_y

            he = heTable[he][1]

        v_source = heTable[he][3]
        verts_sources = np.concatenate((verts_sources, (np.array([vertTable[v_source]]))), axis=0)

        x, y = zip(*verts_sources, strict=False)

        if flip_x:
            x = tuple(np.array((width,) * len(x)) - x)
        if flip_y:
            y = tuple(np.array((height,) * len(y)) - y)

        if multicolor:
            plt.fill(x, y, color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(x, np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(x, np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), y, color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), y, color=cmap(face), alpha=0.5)

        if lines:
            for i in range(0, len(x) - 1, 1):
                plt.plot(x[i : i + 2], y[i : i + 2], "-", color="black")
                plt.plot(
                    tuple(np.add(x[i : i + 2], (width, width))),
                    tuple(np.add(y[i : i + 2], (height, height))),
                    "-",
                    color="black",
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (-width, -width))),
                    tuple(np.add(y[i : i + 2], (-height, -height))),
                    "-",
                    color="black",
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (-width, -width))),
                    tuple(np.add(y[i : i + 2], (height, height))),
                    "-",
                    color="black",
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (width, width))),
                    tuple(np.add(y[i : i + 2], (-height, -height))),
                    "-",
                    color="black",
                )
                plt.plot(x[i : i + 2], tuple(np.add(y[i : i + 2], (height, height))), "-", color="black")
                plt.plot(x[i : i + 2], tuple(np.add(y[i : i + 2], (-height, -height))), "-", color="black")
                plt.plot(tuple(np.add(x[i : i + 2], (width, width))), y[i : i + 2], "-", color="black")
                plt.plot(tuple(np.add(x[i : i + 2], (-width, -width))), y[i : i + 2], "-", color="black")

    if vertices:
        x_all, y_all = zip(*all_verts, strict=False)
        plt.scatter(x_all, y_all, color="black")

    plt.xlim([0, width])
    plt.ylim([0, height])

    plt.gca().set_aspect("equal")

    if save:
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + str(name) + ".svg", format="svg")
        plt.savefig(path + str(name) + ".png", format="png")

    if show:
        plt.show()

    plt.clf()


def plot_line_tens(
    vertTable,
    heTable,
    faceTable,
    file_txt,
    width: float,
    height: float,
    flip_x=False,
    flip_y=False,
    multicolor=False,
    lines=True,
    vertices=False,
    path=".",
    name="-1",
    save=False,
    show=True,
):
    def load_last_np_array(filename):
        with open(filename) as file:
            content = file.read().strip()
        # Split into separate arrays (assuming each starts with '[' and ends with ']')
        arrays = content.split("]\n[")
        last_array_text = arrays[-1].replace("[", "").replace("]", "")
        # Convert string to NumPy array
        last_array = np.fromstring(last_array_text, sep=" ")
        return last_array

    red_intensity = load_last_np_array(file_txt)
    print(red_intensity)
    red_intensity = (red_intensity - np.min(red_intensity)) / (np.max(red_intensity) - np.min(red_intensity))

    cmap = get_cmap(len(faceTable))

    all_verts = []
    for face in range(len(faceTable)):
        red_intensity_face = []

        start_he = faceTable[face]
        he = start_he

        red_intensity_face.append(red_intensity[he])

        v_source = heTable[he][3]

        verts_sources = np.array([vertTable[v_source]])
        all_verts.append(vertTable[v_source])

        he_offset_x = heTable[he][6]
        he_offset_y = heTable[he][7]
        sum0_offsets = he_offset_x
        sum1_offsets = he_offset_y

        he = heTable[he][1]

        while he != start_he:
            red_intensity_face.append(red_intensity[he])

            v_source = heTable[he][3]

            all_verts.append(vertTable[v_source])

            verts_sources = np.concatenate(
                (
                    verts_sources,
                    (np.array([vertTable[v_source]]) + np.array([sum0_offsets * width, sum1_offsets * height])),
                ),
                axis=0,
            )

            he_offset_x = heTable[he][6]
            he_offset_y = heTable[he][7]
            sum0_offsets += he_offset_x
            sum1_offsets += he_offset_y

            he = heTable[he][1]

        v_source = heTable[he][3]
        verts_sources = np.concatenate((verts_sources, (np.array([vertTable[v_source]]))), axis=0)

        x, y = zip(*verts_sources, strict=False)

        if flip_x:
            x = tuple(np.array((width,) * len(x)) - x)
        if flip_y:
            y = tuple(np.array((height,) * len(y)) - y)

        if multicolor:
            plt.fill(x, y, color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(x, np.add(y, height), color=cmap(face), alpha=0.5)
            plt.fill(x, np.add(y, -height), color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, width), y, color=cmap(face), alpha=0.5)
            plt.fill(np.add(x, -width), y, color=cmap(face), alpha=0.5)

        if lines:
            for i in range(0, len(x) - 1, 1):
                r = red_intensity_face[i]  # Assumi che red_intensity sia una lista con valori tra 0 e 1
                color = (r, 0, 0)  # (rosso, verde, blu)
                plt.plot(x[i : i + 2], y[i : i + 2], "-", color=color, lw=0.5 + r * 3.0)
                plt.plot(
                    tuple(np.add(x[i : i + 2], (width, width))),
                    tuple(np.add(y[i : i + 2], (height, height))),
                    "-",
                    color=color,
                    lw=0.5 + r * 3.0,
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (-width, -width))),
                    tuple(np.add(y[i : i + 2], (-height, -height))),
                    "-",
                    color=color,
                    lw=0.5 + r * 3.0,
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (-width, -width))),
                    tuple(np.add(y[i : i + 2], (height, height))),
                    "-",
                    color=color,
                    lw=0.5 + r * 3.0,
                )
                plt.plot(
                    tuple(np.add(x[i : i + 2], (width, width))),
                    tuple(np.add(y[i : i + 2], (-height, -height))),
                    "-",
                    color=color,
                    lw=0.5 + r * 3.0,
                )
                plt.plot(
                    x[i : i + 2], tuple(np.add(y[i : i + 2], (height, height))), "-", color=color, lw=0.5 + r * 3.0
                )
                plt.plot(
                    x[i : i + 2], tuple(np.add(y[i : i + 2], (-height, -height))), "-", color=color, lw=0.5 + r * 3.0
                )
                plt.plot(tuple(np.add(x[i : i + 2], (width, width))), y[i : i + 2], "-", color=color, lw=0.5 + r * 3.0)
                plt.plot(
                    tuple(np.add(x[i : i + 2], (-width, -width))), y[i : i + 2], "-", color=color, lw=0.5 + r * 3.0
                )

    if vertices:
        x_all, y_all = zip(*all_verts, strict=False)
        plt.scatter(x_all, y_all, color="black")

    plt.xlim([0, width])
    plt.ylim([0, height])

    plt.gca().set_aspect("equal")

    if save:
        os.makedirs(path, exist_ok=True)
        # plt.savefig(path + str(name) + '.svg', format='svg')
        plt.savefig(path + str(name) + ".png", format="png")

    if show:
        plt.show()

    plt.clf()


# ==========
# Bounded
# ==========
def draw_arc_N_get_points(ang, pos_source, pos_target, lines, n=100):
    edge_vector = pos_target - pos_source
    edge_half_length = np.linalg.norm(edge_vector) / 2
    radius = edge_half_length / np.sin(ang)
    diameter = 2 * radius
    d_midpoint_to_center = np.sqrt(radius**2 - edge_half_length**2)
    midpoint = (pos_source + pos_target) / 2
    unit_vector = edge_vector / np.linalg.norm(edge_vector)
    unit_vector_midpoint_to_center = np.array([-unit_vector[1], unit_vector[0]])
    center = midpoint + unit_vector_midpoint_to_center * d_midpoint_to_center
    ang_source = np.angle(np.dot((pos_source - center), np.array([1, 1j])))
    ang_target = np.angle(np.dot((pos_target - center), np.array([1, 1j])))
    rad2deg = 180 / np.pi
    normalized_ang_source_degrees = ang_source * rad2deg
    normalized_ang_target_degrees = ang_target * rad2deg
    if normalized_ang_source_degrees < 0.0:
        normalized_ang_source_degrees += 360
    if normalized_ang_target_degrees < 0.0:
        normalized_ang_target_degrees += 360
    if lines:
        surface_arc = Arc(
            center,
            diameter,
            diameter,
            theta1=normalized_ang_source_degrees,
            theta2=normalized_ang_target_degrees,
            color="black",
            linewidth=2.0,
        )
        plt.gca().add_patch(surface_arc)
    tau = 2 * np.pi
    if abs(ang_target - ang_source) > np.pi:
        if ang_source < ang_target:
            ang_source += tau
        else:
            ang_target += tau
    intermediate_angles = np.linspace(ang_source, ang_target, n, False)[1:]
    points = [pos_source]
    for a in intermediate_angles:
        points.append(radius * np.array([np.cos(a), np.sin(a)]) + center)
    points.append(pos_target)
    points.append(pos_source)
    x, y = zip(*points, strict=False)
    return x, y


def plot_bounded_mesh(
    vertTable,
    angTable,
    heTable,
    faceTable,
    L_box,
    flip_x=False,
    flip_y=False,
    multicolor=True,
    lines=True,
    vertices=False,
    fates=False,
    path=".",
    name="-1",
    save=False,
    show=True,
):
    if fates:
        cmap = get_cmap(np.max(faceTable[:, 1]) + 1, name="viridis")
    else:
        cmap = get_cmap(faceTable.shape[0])
    draw_curve_threshold = 0.01  # radians. Must be above 0 to avoid overcomplicating a simple plot
    num_edges = heTable.shape[0] // 2

    all_verts = []
    for face in range(faceTable.shape[0]):
        surfaces = []
        is_surface = False
        start_he = int(faceTable[face, 0])
        he = start_he
        v_source = int(heTable[he][3] + heTable[he][5])
        pos_source = vertTable[v_source - 2]
        if heTable[he][3] == 0:
            ang = angTable[he // 2]
            if ang > draw_curve_threshold:
                is_surface = True
                v_target = int(heTable[he][4] + heTable[he][6] - 1)
                pos_target = vertTable[v_target - 2]
                arcx, arcy = draw_arc_N_get_points(ang, pos_source, pos_target, lines)
                if multicolor:
                    if fates:
                        plt.fill(arcx, arcy, facecolor=cmap(faceTable[face, 1]), alpha=0.5)
                    else:
                        plt.fill(arcx, arcy, facecolor=cmap(face), alpha=0.5)
        verts_sources = np.array([pos_source])
        all_verts.append(pos_source)
        surfaces.append(is_surface)
        he = int(heTable[he][1])

        while he != start_he:
            is_surface = False
            v_source = int(heTable[he][3] + heTable[he][5])
            pos_source = vertTable[v_source - 2]
            if heTable[he][3] == 0:
                ang = angTable[he // 2]
                if ang > draw_curve_threshold:
                    is_surface = True
                    v_target = int(heTable[he][4] + heTable[he][6] - 1)
                    pos_target = vertTable[v_target - 2]
                    arcx, arcy = draw_arc_N_get_points(ang, pos_source, pos_target, lines)
                    if multicolor:
                        if fates:
                            plt.fill(arcx, arcy, facecolor=cmap(faceTable[face, 1]), alpha=0.5)
                        else:
                            plt.fill(arcx, arcy, facecolor=cmap(face), alpha=0.5)
            all_verts.append(pos_source)
            verts_sources = np.concatenate([verts_sources, pos_source[None]], axis=0)
            surfaces.append(is_surface)
            he = int(heTable[he][1])

        v_source = int(heTable[he][3] + heTable[he][5])
        pos_source = vertTable[v_source - 2]
        verts_sources = np.concatenate([verts_sources, pos_source[None]], axis=0)

        x, y = zip(*verts_sources, strict=False)

        if flip_x:
            x = tuple(np.array((L_box,) * len(x)) - x)
        if flip_y:
            y = tuple(np.array((L_box,) * len(y)) - y)

        if multicolor:
            if fates:
                plt.fill(x, y, facecolor=cmap(faceTable[face, 1]), alpha=0.5)
            else:
                plt.fill(x, y, facecolor=cmap(face), alpha=0.5)

        if lines:
            for i in range(0, len(x) - 1, 1):
                if not surfaces[i]:
                    plt.plot(x[i : i + 2], y[i : i + 2], "-", color="black")

    if vertices:
        x_all, y_all = zip(*all_verts, strict=False)
        plt.scatter(x_all, y_all, color="black")

    # unlike the pbc case, here is not easy to know a priori what the limits of the progressively optimized cell cluster will be (across a stack of images)
    plt.xlim([-1.5, L_box + 0.5])
    plt.ylim([-1.5, L_box + 0.5])
    plt.gca().set_aspect("equal")

    if save:
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + str(name) + ".pdf")  # format maybe should be left as a choice

    if show:
        plt.show()

    plt.clf()
