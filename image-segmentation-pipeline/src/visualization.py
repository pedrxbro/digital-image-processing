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


def show_channel_comparison(channel_comparison, save_path=None, show=True):
    """Exibe canal, mascara Otsu e overlay para cada alternativa analisada."""
    comparisons = channel_comparison["comparisons"]
    images = [channel_comparison["original"], channel_comparison["grayscale"]]
    titles = ["Original", "Grayscale"]
    cmaps = [None, "gray"]

    for comparison in comparisons:
        label = comparison["channel"]
        threshold = comparison["threshold"]
        images.extend([comparison["image"], comparison["mask"], comparison["overlay"]])
        titles.extend([label, f"{label}: Otsu={threshold:.1f}", f"{label}: overlay"])
        cmaps.extend(["gray", "gray", None])

    rows = math.ceil(len(images) / 3)
    return show_comparison(
        images,
        titles,
        cols=3,
        figsize=(15, 4 * rows),
        cmaps=cmaps,
        save_path=save_path,
        show=show,
    )


def show_pipeline_evolution(result, save_path=None, show=True):
    """Exibe os principais intermediarios da segmentacao de hemacias."""
    morphology = result.get("morphology_history", {})
    images = [
        result["original"],
        result["grayscale"],
        result["selected_channel"],
        result["frequency_filtered"],
        result.get("edge_enhanced", np.zeros_like(result["grayscale"])),
        result["superpixel_overlay"],
        result.get("rbc_candidate_mask", result["initial_mask"]),
        result.get("excluded_mask", np.zeros_like(result["initial_mask"])),
        result["initial_mask"],
        morphology.get("opened", result["morphology_mask"]),
        morphology.get("closed", result["morphology_mask"]),
        result["morphology_mask"],
        result["contour_filled_mask"],
        result["watershed_distance_map"],
        result["watershed_markers"],
        result["watershed_instances"],
        result["instance_overlay"],
    ]
    titles = [
        "Original",
        "Grayscale",
        result["selected_channel_name"],
        "Filtro em frequencia",
        "Magnitude Sobel",
        "Superpixels SLIC",
        "Candidatos a hemacias",
        "Exclusao de celulas azuladas",
        "Mascara inicial de hemacias",
        "Apos abertura",
        "Apos fechamento",
        "Apos filtro de area",
        "Contornos pequenos preenchidos",
        "Mapa de distancia",
        "Marcadores Watershed",
        "Instancias Watershed",
        f"Overlay final: {result['estimated_rbc_count']} hemacias",
    ]
    cmaps = [
        None, "gray", "gray", "gray", "gray", None, "gray", "gray", "gray",
        "gray", "gray", "gray", "gray", "gray", "nipy_spectral", "nipy_spectral", None,
    ]
    return show_comparison(
        images,
        titles,
        cols=4,
        figsize=(18, 18),
        cmaps=cmaps,
        save_path=save_path,
        show=show,
    )


def show_batch_overlays(results, save_path=None, show=True):
    """Compara as contagens finais e os overlays do lote processado."""
    images = [result["instance_overlay"] for result in results]
    titles = [
        f"{Path(result['image_path']).name}\n{result['estimated_rbc_count']} hemacias"
        for result in results
    ]
    return show_comparison(
        images,
        titles,
        cols=min(3, len(images)),
        figsize=(6 * len(images), 5),
        cmaps=[None] * len(images),
        save_path=save_path,
        show=show,
    )


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


def show_comparison(images, titles, cols=4, figsize=(16, 6), cmap=None, cmaps=None, save_path=None, show=True):
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

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)

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
