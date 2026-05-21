from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_image_files(folder_path, extensions=IMAGE_EXTENSIONS):
    """Retorna os arquivos de imagem de uma pasta em ordem alfabética."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    files = [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ]

    return sorted(files)


def load_color_image(image_path):
    """Carrega uma imagem em RGB para visualização no Matplotlib."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_grayscale_image(image_path):
    """Carrega uma imagem em escala de cinza."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem em escala de cinza: {image_path}")

    return image


def describe_image(image):
    """Retorna metadados básicos de um array de imagem."""
    if image is None:
        raise ValueError("A imagem é nula")

    channels = image.shape[2] if image.ndim == 3 else 1

    return {
        "shape": image.shape,
        "dtype": image.dtype,
        "min": int(np.min(image)),
        "max": int(np.max(image)),
        "channels": channels,
    }


def find_matching_mask(image_path, masks_folder, mask_extensions=IMAGE_EXTENSIONS):
    """Encontra uma máscara com o mesmo nome-base da imagem."""
    image_path = Path(image_path)
    masks_folder = Path(masks_folder)

    if not masks_folder.exists():
        return None

    for extension in mask_extensions:
        candidate = masks_folder / f"{image_path.stem}{extension}"
        if candidate.exists():
            return candidate

    return None


def inspect_mask(mask):
    """Retorna dimensões, tipo, intervalo e valores únicos de uma máscara."""
    if mask is None:
        raise ValueError("A máscara é nula")

    unique_values = np.unique(mask)

    return {
        "shape": mask.shape,
        "dtype": mask.dtype,
        "min": int(mask.min()),
        "max": int(mask.max()),
        "unique_values": unique_values.tolist(),
        "is_binary": set(unique_values.tolist()).issubset({0, 1, 255}),
    }


def validate_image_mask_pair(image, mask):
    """Valida se a imagem e a máscara têm dimensões espaciais compatíveis."""
    if image is None or mask is None:
        return False

    return image.shape[:2] == mask.shape[:2]
