from collections import deque

import numpy as np


def _validate_slic_image(image):
    image = np.asarray(image)

    if image.ndim not in (2, 3):
        raise ValueError("A imagem de entrada do SLIC deve ser 2D em cinza ou 3D com canais.")

    if image.ndim == 3 and image.shape[2] == 0:
        raise ValueError("Imagens com múltiplos canais devem ter pelo menos um canal.")

    return image.astype(np.float64, copy=False)


def _cluster_color_at(image, y, x):
    pixel = image[y, x]

    if image.ndim == 2:
        return float(pixel)

    return pixel.astype(np.float64, copy=True)


def initialize_clusters(image, S):
    """Cria os centros iniciais do SLIC em uma grade regular."""
    image = _validate_slic_image(image)
    height, width = image.shape[:2]
    step = max(1, int(S))
    offset = step // 2

    clusters = []
    for y in range(offset, height, step):
        for x in range(offset, width, step):
            clusters.append([float(y), float(x), _cluster_color_at(image, y, x)])

    if not clusters:
        y = height // 2
        x = width // 2
        clusters.append([float(y), float(x), _cluster_color_at(image, y, x)])

    return clusters


def compute_slic_distance(y, x, cluster, image, S, m):
    """Calcula a distância SLIC entre um pixel e o centro de um cluster."""
    image = _validate_slic_image(image)
    center_y, center_x, center_color = cluster

    ds2 = (y - center_y) ** 2 + (x - center_x) ** 2
    pixel = image[y, x]

    if image.ndim == 2:
        dc2 = (float(pixel) - float(center_color)) ** 2
    else:
        dc2 = np.sum((pixel - np.asarray(center_color, dtype=np.float64)) ** 2)

    return float(np.sqrt(dc2 + (m / S) ** 2 * ds2))


def _compute_window_distances(window, row_grid, col_grid, cluster, S, m):
    center_y, center_x, center_color = cluster

    ds2 = (row_grid - center_y) ** 2 + (col_grid - center_x) ** 2

    if window.ndim == 2:
        dc2 = (window - float(center_color)) ** 2
    else:
        color = np.asarray(center_color, dtype=np.float64)
        dc2 = np.sum((window - color) ** 2, axis=2)

    return np.sqrt(dc2 + (m / S) ** 2 * ds2)


def _recompute_clusters(image, labels, clusters):
    new_clusters = []

    for cluster_index, cluster in enumerate(clusters):
        ys, xs = np.where(labels == cluster_index)

        if len(ys) == 0:
            new_clusters.append(cluster)
            continue

        center_y = float(np.mean(ys))
        center_x = float(np.mean(xs))

        if image.ndim == 2:
            center_color = float(np.mean(image[ys, xs]))
        else:
            center_color = np.mean(image[ys, xs], axis=0)

        new_clusters.append([center_y, center_x, center_color])

    return new_clusters


def enforce_superpixel_connectivity(labels, min_size):
    """Separa ilhas desconectadas e incorpora componentes muito pequenos."""
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels deve ser um array 2D.")

    if min_size < 0:
        raise ValueError("min_size deve ser nao negativo.")

    height, width = labels.shape
    connected_labels = -np.ones(labels.shape, dtype=np.int32)
    visited = np.zeros(labels.shape, dtype=bool)
    next_label = 0
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x]:
                continue

            source_label = labels[start_y, start_x]
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            component = []
            adjacent_labels = []

            while queue:
                y, x = queue.popleft()
                component.append((y, x))

                for delta_y, delta_x in offsets:
                    neighbor_y = y + delta_y
                    neighbor_x = x + delta_x

                    if neighbor_y < 0 or neighbor_y >= height:
                        continue
                    if neighbor_x < 0 or neighbor_x >= width:
                        continue

                    if labels[neighbor_y, neighbor_x] == source_label:
                        if not visited[neighbor_y, neighbor_x]:
                            visited[neighbor_y, neighbor_x] = True
                            queue.append((neighbor_y, neighbor_x))
                        continue

                    adjacent_label = connected_labels[neighbor_y, neighbor_x]
                    if adjacent_label >= 0:
                        adjacent_labels.append(int(adjacent_label))

            if len(component) < min_size and adjacent_labels:
                target_label = max(set(adjacent_labels), key=adjacent_labels.count)
            else:
                target_label = next_label
                next_label += 1

            for y, x in component:
                connected_labels[y, x] = target_label

    return connected_labels


def slic(image, num_superpixels=100, m=10, max_iter=10, enforce_connectivity=True):
    """Gera um mapa 2D de rótulos de superpixels usando SLIC do zero."""
    if num_superpixels <= 0:
        raise ValueError("num_superpixels deve ser maior que zero.")

    if m <= 0:
        raise ValueError("m deve ser maior que zero.")

    if max_iter <= 0:
        raise ValueError("max_iter deve ser maior que zero.")

    image = _validate_slic_image(image)
    height, width = image.shape[:2]

    S = max(1, int(np.sqrt((height * width) / num_superpixels)))
    clusters = initialize_clusters(image, S)

    labels = -np.ones((height, width), dtype=np.int32)
    distances = np.full((height, width), np.inf, dtype=np.float64)

    for _ in range(max_iter):
        for cluster_index, cluster in enumerate(clusters):
            center_y, center_x, _ = cluster
            y0 = max(0, int(center_y - S))
            y1 = min(height, int(center_y + S))
            x0 = max(0, int(center_x - S))
            x1 = min(width, int(center_x + S))

            if y0 >= y1 or x0 >= x1:
                continue

            row_grid, col_grid = np.ogrid[y0:y1, x0:x1]
            window = image[y0:y1, x0:x1]
            local_distances = _compute_window_distances(
                window,
                row_grid,
                col_grid,
                cluster,
                S,
                m,
            )

            distance_window = distances[y0:y1, x0:x1]
            label_window = labels[y0:y1, x0:x1]
            better_pixels = local_distances < distance_window

            distance_window[better_pixels] = local_distances[better_pixels]
            label_window[better_pixels] = cluster_index

        clusters = _recompute_clusters(image, labels, clusters)
        distances.fill(np.inf)

    if enforce_connectivity:
        min_size = max(1, int(0.25 * S * S))
        labels = enforce_superpixel_connectivity(labels, min_size=min_size)

    return labels


def create_superpixel_boundary_mask(labels, thickness=1):
    """Retorna uma máscara booleana marcando as bordas dos superpixels."""
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels deve ser um array 2D.")

    if thickness <= 0:
        raise ValueError("thickness deve ser maior que zero.")

    boundaries = np.zeros(labels.shape, dtype=bool)

    horizontal_changes = labels[1:, :] != labels[:-1, :]
    boundaries[1:, :] |= horizontal_changes
    boundaries[:-1, :] |= horizontal_changes

    vertical_changes = labels[:, 1:] != labels[:, :-1]
    boundaries[:, 1:] |= vertical_changes
    boundaries[:, :-1] |= vertical_changes

    for _ in range(thickness - 1):
        expanded = boundaries.copy()
        expanded[1:, :] |= boundaries[:-1, :]
        expanded[:-1, :] |= boundaries[1:, :]
        expanded[:, 1:] |= boundaries[:, :-1]
        expanded[:, :-1] |= boundaries[:, 1:]
        boundaries = expanded

    return boundaries


def _to_uint8_rgb(image):
    image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image deve ser 2D em cinza ou RGB com formato (altura, largura, 3).")

    image = image.astype(np.float64, copy=False)
    if image.max(initial=0) <= 1.0:
        image = image * 255.0

    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def overlay_superpixel_boundaries(
    image,
    labels,
    boundary_color=(255, 0, 0),
    alpha=0.9,
    thickness=1,
):
    """Sobrepõe as bordas dos superpixels em uma imagem em cinza ou RGB."""
    labels = np.asarray(labels)
    rgb_image = _to_uint8_rgb(image)

    if labels.shape != rgb_image.shape[:2]:
        raise ValueError("labels deve ter a mesma altura e largura da imagem.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha deve estar entre 0 e 1.")

    boundaries = create_superpixel_boundary_mask(labels, thickness=thickness)
    color = np.asarray(boundary_color, dtype=np.float64)

    if color.shape != (3,):
        raise ValueError("boundary_color deve ter três valores.")

    overlay = rgb_image.astype(np.float64, copy=True)
    overlay[boundaries] = (1.0 - alpha) * overlay[boundaries] + alpha * color

    return np.clip(np.round(overlay), 0, 255).astype(np.uint8)
