import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import calculate_histogram, rgb_to_grayscale


def convert_rgb_to_grayscale(image):
    """Converte uma imagem RGB para escala de cinza."""
    return rgb_to_grayscale(image)


def show_rgb_histogram(image, title="Histograma RGB"):
    channels = ("Vermelho", "Verde", "Azul")
    colors = ("red", "green", "blue")

    plt.figure(figsize=(8, 4))

    for channel_index, channel_name in enumerate(channels):
        histogram, _ = calculate_histogram(
            image[:, :, channel_index],
            bins=256,
            value_range=(0, 256),
        )
        plt.plot(histogram, color=colors[channel_index], label=channel_name)

    plt.title(title)
    plt.xlabel("Intensidade do pixel")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_grayscale_histogram(gray_image, title="Histograma em escala de cinza"):
    histogram, bin_edges = calculate_histogram(gray_image, bins=256, value_range=(0, 256))
    bin_starts = bin_edges[:-1]

    plt.figure(figsize=(8, 4))
    plt.bar(bin_starts, histogram, width=1.0, color="gray")
    plt.title(title)
    plt.xlabel("Intensidade do pixel")
    plt.ylabel("Frequência")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def summarize_histogram(gray_image):
    """Retorna estatísticas simples em escala de cinza para a análise exploratória."""
    return {
        "mean": float(np.mean(gray_image)),
        "std": float(np.std(gray_image)),
        "min": int(np.min(gray_image)),
        "max": int(np.max(gray_image)),
    }
