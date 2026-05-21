import cv2
import matplotlib.pyplot as plt
import numpy as np


def convert_rgb_to_grayscale(image):
    """Converte uma imagem RGB para escala de cinza."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def show_rgb_histogram(image, title="Histograma RGB"):
    channels = ("Vermelho", "Verde", "Azul")
    colors = ("red", "green", "blue")

    plt.figure(figsize=(8, 4))

    for channel_index, channel_name in enumerate(channels):
        histogram = cv2.calcHist([image], [channel_index], None, [256], [0, 256])
        plt.plot(histogram, color=colors[channel_index], label=channel_name)

    plt.title(title)
    plt.xlabel("Intensidade do pixel")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_grayscale_histogram(gray_image, title="Histograma em escala de cinza"):
    plt.figure(figsize=(8, 4))
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color="gray")
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
