import numpy as np


def build_binary_mask_from_labels(labels, selected_labels):
    """Constrói uma máscara binária a partir dos rótulos selecionados.

    Pixels pertencentes aos rótulos selecionados recebem valor 1.
    Todos os outros pixels recebem valor 0.
    """
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels deve ser um array 2D.")

    selected_labels = set(int(label_id) for label_id in selected_labels)
    mask = np.zeros(labels.shape, dtype=np.uint8)

    for label_id in selected_labels:
        region_mask = labels == label_id
        mask[region_mask] = 1

    return mask


def build_class_masks_from_labels(labels, label_classes):
    """Cria uma máscara binária independente para cada classe de superpixel."""
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels deve ser um array 2D.")

    class_masks = {}
    for label_id, class_name in label_classes.items():
        if class_name not in class_masks:
            class_masks[class_name] = np.zeros(labels.shape, dtype=np.uint8)

        class_masks[class_name][labels == int(label_id)] = 1

    return class_masks


def grow_labels_by_adjacency(
    labels,
    seed_labels,
    candidate_labels,
    max_iterations=1,
    min_selected_neighbors=1,
):
    """Expande rótulos-semente para superpixels candidatos vizinhos.

    A vizinhança é calculada manualmente pelas transições horizontais e
    verticais do mapa de rótulos. O limite de iterações evita que a expansão
    atravesse gradualmente grandes regiões de fundo.
    """
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels deve ser um array 2D.")

    if max_iterations < 0:
        raise ValueError("max_iterations deve ser nao negativo.")

    if min_selected_neighbors <= 0:
        raise ValueError("min_selected_neighbors deve ser maior que zero.")

    selected = set(int(label_id) for label_id in seed_labels)
    candidates = set(int(label_id) for label_id in candidate_labels) - selected
    adjacency = {int(label_id): set() for label_id in np.unique(labels)}

    def connect(first, second):
        first = int(first)
        second = int(second)
        if first == second:
            return
        adjacency[first].add(second)
        adjacency[second].add(first)

    for row in range(labels.shape[0] - 1):
        for col in range(labels.shape[1]):
            connect(labels[row, col], labels[row + 1, col])

    for row in range(labels.shape[0]):
        for col in range(labels.shape[1] - 1):
            connect(labels[row, col], labels[row, col + 1])

    for _ in range(int(max_iterations)):
        added = {
            label_id
            for label_id in candidates
            if len(adjacency[label_id] & selected) >= min_selected_neighbors
        }

        if not added:
            break

        selected.update(added)
        candidates.difference_update(added)

    return sorted(selected)


def mask_to_visual(mask):
    """Converte uma máscara binária 0/1 para 0/255, usada na visualização."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("mask deve ser um array 2D.")

    return (mask > 0).astype(np.uint8) * 255


def overlay_binary_mask(image, mask, color=(255, 0, 0), alpha=0.35):
    """Sobrepõe uma máscara binária em uma imagem RGB para inspeção visual."""
    image = np.asarray(image)
    mask = np.asarray(mask)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image deve ter formato RGB (altura, largura, 3).")

    if mask.shape != image.shape[:2]:
        raise ValueError("mask deve ter a mesma altura e largura da imagem.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha deve estar entre 0 e 1.")

    color = np.asarray(color, dtype=np.float64)
    if color.shape != (3,):
        raise ValueError("color deve ter três valores.")

    image_float = image.astype(np.float64, copy=False)
    overlay = image_float.copy()
    binary_mask = mask > 0

    overlay[binary_mask] = (1.0 - alpha) * image_float[binary_mask] + alpha * color

    return np.clip(np.round(overlay), 0, 255).astype(np.uint8)


def overlay_instance_boundaries(image, instances, color=(255, 255, 0), alpha=0.9):
    """Sobrepõe fronteiras entre instâncias rotuladas em uma imagem RGB."""
    image = np.asarray(image)
    instances = np.asarray(instances)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image deve ter formato RGB (altura, largura, 3).")

    if instances.shape != image.shape[:2]:
        raise ValueError("instances deve ter a mesma altura e largura da imagem.")

    boundaries = instances < 0
    horizontal_changes = (instances[1:, :] != instances[:-1, :]) & (
        (instances[1:, :] > 0) | (instances[:-1, :] > 0)
    )
    vertical_changes = (instances[:, 1:] != instances[:, :-1]) & (
        (instances[:, 1:] > 0) | (instances[:, :-1] > 0)
    )

    boundaries[1:, :] |= horizontal_changes
    boundaries[:-1, :] |= horizontal_changes
    boundaries[:, 1:] |= vertical_changes
    boundaries[:, :-1] |= vertical_changes

    overlay = image.astype(np.float64, copy=True)
    boundary_color = np.asarray(color, dtype=np.float64)
    overlay[boundaries] = (1.0 - alpha) * overlay[boundaries] + alpha * boundary_color

    return np.clip(np.round(overlay), 0, 255).astype(np.uint8)
