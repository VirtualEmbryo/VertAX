"""Analyze masks (with no holes) to extracts a primitive mesh with vertices, edges, and faces."""

import numpy as np
import tifffile as tiff
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label


def segment(image: NDArray, save: bool = False, output_path: str = "segmented_image.tiff") -> NDArray:
    """Segment given image using Cellpose.

    The result will be imperfect and it will always be better if you provide directly
    the masks yourself.

    Args:
        image (NDArray): Image to segment.
        save (bool, optional): Whether to save the segmentation on disk. Defaults to False.
        output_path (str, optional): Where to save the segmentation on disk. Defaults to "segmented_image.tiff".

    Returns:
        NDArray: The segmented image.
    """
    from cellpose import models

    # Ensure the image is in the correct format
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)  # Convert to (W, H, 1)
    # Blur the image with gaussian filter sigma 10
    blurred_image = gaussian_filter(image, sigma=10)
    # Segment the image using Cellpose's `cyto` model
    model = models.Cellpose(model_type="cyto", gpu=True)
    mask, _, _, _ = model.eval(blurred_image, channels=[0, 0], diameter=None)
    if save:
        # Save the resulting image
        tiff.imwrite(output_path, mask.astype(np.uint16), imagej=True)

    return mask


def refine(segmented_image: NDArray) -> NDArray:
    """Grow segmented cells to avoid any holes in the mask.

    Args:
        segmented_image (NDArray): The segmentation with holes.

    Returns:
        NDArray: A mask suitable for what comes next, with no holes.
    """
    # Relabel the mask
    labeled_mask: NDArray = label(segmented_image)
    unique_labels, counts = np.unique(labeled_mask, return_counts=True)
    # Create a mask for small labels
    min_size = 9
    small_labels = unique_labels[counts < min_size]
    # Replace small labels with 0
    updated_mask = labeled_mask.copy()
    for label_i in small_labels:
        if label_i != 0:  # Skip the background label
            updated_mask[labeled_mask == label_i] = 0
    # Compute Euclidean Distance Transform (EDT) for background
    _, indices = distance_transform_edt(updated_mask == 0, return_indices=True)
    # Assign background pixels to the nearest label
    nearest_labels = updated_mask[tuple(indices)]
    expanded_mask = updated_mask.copy()
    expanded_mask[updated_mask == 0] = nearest_labels[updated_mask == 0]
    return expanded_mask


def mask_from_image(image: NDArray) -> NDArray:
    """Automatically create a mask from an image.

    Note that the result will be imperfect and the result will always be better if you provide
    your own masks directly.

    Args:
        image (NDArray): The original image.

    Returns:
        NDArray: The corresponding mask.
    """
    return refine(segment(image))


def pad(mask: NDArray, save: bool = False, output_path: str = "padded_image.tiff") -> NDArray:
    """Pad a mask with mirrored reflections all around the mask.

    Args:
        mask (NDArray): Original mask.
        save (bool, optional): Whether or not to save the padded mask on disk. Defaults to False.
        output_path (str, optional): Where to save it. Defaults to "padded_image.tiff".

    Returns:
        NDArray: The mask, padded around with mirrored reflections.
    """
    # Apply reflect padding with the size of the image itself
    height, width = mask.shape  # imread tiff = Y is the first axis, X the second.
    padded_image = np.pad(
        mask,
        ((height, height), (width, width)),  # Reflect padding on top and left only
        mode="symmetric",
    )

    labelled_image: NDArray = label(padded_image)
    if save:
        # Save the resulting image
        tiff.imwrite(output_path, labelled_image.astype(np.uint16), imagej=True)

    return labelled_image


def _find_trijunctions_and_labels(
    padded_mask: NDArray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], set[int]]:
    # imread tiff = Y is the first axis, X the second.
    height = padded_mask.shape[0]
    width = padded_mask.shape[1]

    labels_to_vertices: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    unique_labels: set[int] = set()
    # Traverse the image and find vertices (three-junctions only) -- adapt to the flip -- 9 cases
    for row in range(height - 1):
        for col in range(width - 1):
            # imread tiff = Y is the first axis, X the second.
            # We "correct" that when finding the points.
            point = (col, row)
            # Extract 2x2 neighborhood
            neighborhood = padded_mask[row : row + 2, col : col + 2]
            labels = tuple(sorted(np.unique(neighborhood)))
            # Check if there are exactly 3 unique labels and ensure uniqueness of label set.
            # Check also that the diagonals are different pixels, should be the case if the quality of the mask is good.
            if (
                len(labels) == 3
                and neighborhood[0, 0] != neighborhood[1, 1]
                and neighborhood[0, 1] != neighborhood[1, 0]
            ):
                if labels not in labels_to_vertices:
                    labels_to_vertices[labels] = [point]
                else:
                    labels_to_vertices[labels].append(point)

    # Only keep unique three-junction points.
    trijunctions: list[tuple[int, int]] = []
    trijunctions_labels: list[tuple[int, int, int]] = []
    for labels, points in labels_to_vertices.items():
        if len(points) == 1:
            trijunctions.append(points[0])
            trijunctions_labels.append(labels)
            unique_labels.update(set(labels))

    return (trijunctions, trijunctions_labels, unique_labels)


def _find_edges(
    trijunctions: list[tuple[int, int]], trijunctions_labels: list[tuple[int, int, int]]
) -> list[tuple[int, int]]:
    nb_trijunctions = len(trijunctions)
    edges = []

    for i in range(nb_trijunctions):
        labels1 = set(trijunctions_labels[i])
        for j in range(i + 1, nb_trijunctions):
            labels2 = set(trijunctions_labels[j])

            shared_labels = labels1.intersection(labels2)
            nb_labels_in_common = len(shared_labels)

            if nb_labels_in_common == 2:  # the two trijunctions share an edge together
                edges.append((i, j))
    return edges


def _find_faces(unique_labels: set[int], trijunctions_labels: list[tuple[int, int, int]]) -> list[list[int]]:
    faces = []

    for label_i in unique_labels:
        label_junctions = [idx for idx, labels in enumerate(trijunctions_labels) if label_i in labels]
        if len(label_junctions) > 2:
            # We don't keep faces on the border which would have less than 3 points and be "open".
            faces.append(label_junctions)
    return faces


def find_vertices_edges_faces(padded_mask: NDArray) -> tuple[NDArray, list[tuple[int, int]], list[list[int]]]:
    """Construct rudimentary mesh from a labeled and padded mask.

    Args:
        padded_mask (NDArray): The labeled padded mask.

    Returns:
        tuple[NDArray, list[tuple[int, int]], list[list[int]]]: Vertices (2d positions),
            edges as list of couple of vertices indices, and faces as a list of vertices indices.
    """
    # 1) Find unique three-junction points
    trijunctions, trijunctions_labels, unique_labels = _find_trijunctions_and_labels(padded_mask)

    # 2) Find edges between three-junction points
    edges = _find_edges(trijunctions, trijunctions_labels)

    # 3) Find faces as sets of three-junctions sharing a unique label
    faces = _find_faces(unique_labels, trijunctions_labels)

    return np.array(trijunctions), edges, faces
