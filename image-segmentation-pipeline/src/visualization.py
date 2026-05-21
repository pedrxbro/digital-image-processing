import math

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
