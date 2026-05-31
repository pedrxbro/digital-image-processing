import numpy as np

from src.preprocessing import normalize_to_uint8


SOBEL_GX = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]
)
SOBEL_GY = np.array(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]
)


def convolve2d_from_scratch(image, kernel, padding="edge"):
    """Aplica convolucao 2D manual sobre uma imagem de intensidade."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("image deve ser um array 2D.")
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel deve ser 2D e ter dimensoes impares.")
    if padding not in ("edge", "constant"):
        raise ValueError("padding deve ser 'edge' ou 'constant'.")

    row_pad = kernel.shape[0] // 2
    col_pad = kernel.shape[1] // 2
    if padding == "edge":
        padded = np.pad(image, ((row_pad, row_pad), (col_pad, col_pad)), mode="edge")
    else:
        padded = np.pad(image, ((row_pad, row_pad), (col_pad, col_pad)), mode="constant")

    output = np.zeros(image.shape, dtype=np.float64)
    flipped_kernel = kernel[::-1, ::-1]

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            window = padded[row:row + kernel.shape[0], col:col + kernel.shape[1]]
            output[row, col] = float(np.sum(window * flipped_kernel))

    return output


def normalize_gradient_magnitude(gradient):
    """Normaliza a magnitude do gradiente para visualizacao e reuso na pipeline."""
    return normalize_to_uint8(gradient)


def sobel_edges(image, padding="edge"):
    """Calcula Gx, Gy e magnitude Sobel com convolucao manual."""
    gx = convolve2d_from_scratch(image, SOBEL_GX, padding=padding)
    gy = convolve2d_from_scratch(image, SOBEL_GY, padding=padding)
    magnitude = np.sqrt(gx**2 + gy**2)

    return {
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "enhanced": normalize_gradient_magnitude(magnitude),
        "method": "sobel",
        "padding": padding,
        "kernels": {
            "gx": SOBEL_GX.astype(int).tolist(),
            "gy": SOBEL_GY.astype(int).tolist(),
        },
    }
