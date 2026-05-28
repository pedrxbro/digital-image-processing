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
