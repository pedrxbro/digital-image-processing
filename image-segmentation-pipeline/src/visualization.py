import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title="Imagem", figsize=(6, 6), cmap=None):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_images_grid(images, titles=None, cols=5, figsize=(15, 5), cmap=None):
    if not images:
        raise ValueError("A lista de imagens está vazia")

    rows = math.ceil(len(images) / cols)

    plt.figure(figsize=figsize)

    for index, image in enumerate(images):
        plt.subplot(rows, cols, index + 1)
        plt.imshow(image, cmap=cmap)
        if titles:
            plt.title(titles[index])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_comparison(images, titles, cols=4, figsize=(16, 6), cmap=None, cmaps=None, save_path=None):
    """Exibe múltiplas imagens em grade e, opcionalmente, salva a figura."""
    if not images:
        raise ValueError("A lista de imagens está vazia")

    if len(images) != len(titles):
        raise ValueError("As listas de imagens e títulos devem ter o mesmo tamanho")

    if cmaps is not None and len(cmaps) != len(images):
        raise ValueError("A lista de mapas de cor deve ter o mesmo tamanho das imagens")

    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for index, image in enumerate(images):
        current_cmap = cmaps[index] if cmaps is not None else cmap
        axes[index].imshow(image, cmap=current_cmap)
        axes[index].set_title(titles[index])
        axes[index].axis("off")

    for axis in axes[len(images):]:
        axis.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if plt.get_backend().lower() == "agg":
        plt.close(fig)
    else:
        plt.show()

    return fig, axes


def show_image_and_mask(image, mask, image_title="Imagem", mask_title="Máscara", figsize=(10, 5)):
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(image_title)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(mask_title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_mask_overlay(image, mask, title="Sobreposição da máscara", alpha=0.35, figsize=(6, 6)):
    binary_mask = mask > 0
    overlay = image.copy()
    overlay[binary_mask] = [255, 0, 0]

    blended = np.clip((1 - alpha) * image + alpha * overlay, 0, 255).astype(image.dtype)

    plt.figure(figsize=figsize)
    plt.imshow(blended)
    plt.title(title)
    plt.axis("off")
    plt.show()
