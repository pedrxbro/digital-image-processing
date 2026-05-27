import numpy as np

from src.preprocessing import normalize_to_uint8


def _validate_grayscale_image(image):
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("A imagem em escala de cinza deve ter formato (altura, largura).")

    return image


def compute_fft(gray_image):
    """Calcula a Transformada de Fourier 2D centralizada de uma imagem em escala de cinza."""
    gray_image = _validate_grayscale_image(gray_image).astype(np.float64)
    fft = np.fft.fft2(gray_image)
    return np.fft.fftshift(fft)


def magnitude_spectrum(fft_shifted, log_scale=True):
    """Cria um espectro de magnitude normalizado para visualização."""
    magnitude = np.abs(fft_shifted)

    if log_scale:
        magnitude = 20.0 * np.log(magnitude + 1.0)

    return normalize_to_uint8(magnitude)


def create_low_pass_mask(shape, radius, center=None):
    """Cria uma máscara passa-baixa circular ideal."""
    if radius <= 0:
        raise ValueError("O raio deve ser maior que zero.")

    rows, cols = shape[:2]

    if center is None:
        center = (rows // 2, cols // 2)

    center_row, center_col = center
    row_grid, col_grid = np.ogrid[:rows, :cols]
    distance = np.sqrt((row_grid - center_row) ** 2 + (col_grid - center_col) ** 2)

    return (distance <= radius).astype(np.float64)


def create_high_pass_mask(shape, radius, center=None):
    """Cria uma máscara passa-alta circular ideal como complemento da passa-baixa."""
    low_pass_mask = create_low_pass_mask(shape, radius, center=center)
    return 1.0 - low_pass_mask


def apply_frequency_filter(gray_image, mask):
    """Aplica uma máscara no domínio da frequência e reconstrói a imagem normalizada."""
    gray_image = _validate_grayscale_image(gray_image)
    mask = np.asarray(mask, dtype=np.float64)

    if mask.shape[:2] != gray_image.shape:
        raise ValueError("A máscara deve ter o mesmo formato da imagem em escala de cinza.")

    fft_shifted = compute_fft(gray_image)
    filtered_shifted = fft_shifted * mask
    inverse_shifted = np.fft.ifftshift(filtered_shifted)
    filtered_image = np.fft.ifft2(inverse_shifted)

    return normalize_to_uint8(np.abs(filtered_image))


def apply_low_pass_filter(gray_image, radius):
    """Aplica um filtro passa-baixa circular ideal no domínio da frequência."""
    gray_image = _validate_grayscale_image(gray_image)
    mask = create_low_pass_mask(gray_image.shape, radius)
    return apply_frequency_filter(gray_image, mask)


def apply_high_pass_filter(gray_image, radius):
    """Aplica um filtro passa-alta circular ideal no domínio da frequência."""
    gray_image = _validate_grayscale_image(gray_image)
    mask = create_high_pass_mask(gray_image.shape, radius)
    return apply_frequency_filter(gray_image, mask)
