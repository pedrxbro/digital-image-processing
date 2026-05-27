import numpy as np


def _validate_rgb_image(image):
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A imagem RGB deve ter formato (altura, largura, 3).")

    return image


def _rgb_to_float01(image):
    image = _validate_rgb_image(image)

    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        return image.astype(np.float64) / max_value

    image = image.astype(np.float64)
    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def _to_uint8(image):
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def bgr_to_rgb(image):
    """Converte uma imagem BGR para RGB trocando os canais manualmente."""
    image = _validate_rgb_image(image)
    return image[:, :, ::-1].copy()


def rgb_to_grayscale(image):
    """Converte RGB para escala de cinza pela equação clássica de luminância."""
    rgb = _rgb_to_float01(image)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    gray = (0.299 * red + 0.587 * green + 0.114 * blue) * 255.0

    return _to_uint8(gray)


def rgb_to_hsv(image):
    """Converte RGB para HSV sem rotinas prontas de conversão de cor.

    Os canais retornados são codificados como uint8 para visualização:
    H vai de 0-360 graus para 0-255, e S/V vão de 0-1 para 0-255.
    """
    rgb = _rgb_to_float01(image)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    delta = max_channel - min_channel

    hue = np.zeros_like(max_channel)
    nonzero_delta = delta != 0

    red_is_max = (max_channel == red) & nonzero_delta
    green_is_max = (max_channel == green) & nonzero_delta
    blue_is_max = (max_channel == blue) & nonzero_delta

    hue[red_is_max] = 60.0 * np.mod(
        (green[red_is_max] - blue[red_is_max]) / delta[red_is_max],
        6.0,
    )
    hue[green_is_max] = 60.0 * (
        (blue[green_is_max] - red[green_is_max]) / delta[green_is_max] + 2.0
    )
    hue[blue_is_max] = 60.0 * (
        (red[blue_is_max] - green[blue_is_max]) / delta[blue_is_max] + 4.0
    )

    saturation = np.zeros_like(max_channel)
    nonzero_value = max_channel != 0
    saturation[nonzero_value] = delta[nonzero_value] / max_channel[nonzero_value]
    value = max_channel

    hsv = np.dstack(
        (
            hue / 360.0 * 255.0,
            saturation * 255.0,
            value * 255.0,
        )
    )

    return _to_uint8(hsv)


def rgb_to_lab(image):
    """Converte RGB para CIE LAB usando as equações sRGB/D65.

    Os canais retornados são codificados como uint8 para visualização e reconstrução:
    L vai de 0-100 para 0-255, enquanto A e B são deslocados em +128.
    """
    rgb = _rgb_to_float01(image)
    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )

    rgb_to_xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear_rgb @ rgb_to_xyz_matrix.T

    d65_white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz_normalized = xyz / d65_white

    delta = 6.0 / 29.0
    epsilon = delta**3
    f_xyz = np.where(
        xyz_normalized > epsilon,
        np.cbrt(xyz_normalized),
        xyz_normalized / (3.0 * delta**2) + 4.0 / 29.0,
    )

    l_channel = 116.0 * f_xyz[:, :, 1] - 16.0
    a_channel = 500.0 * (f_xyz[:, :, 0] - f_xyz[:, :, 1])
    b_channel = 200.0 * (f_xyz[:, :, 1] - f_xyz[:, :, 2])

    lab = np.dstack(
        (
            l_channel / 100.0 * 255.0,
            a_channel + 128.0,
            b_channel + 128.0,
        )
    )

    return _to_uint8(lab)


def lab_to_rgb(lab_image):
    """Reconstrói RGB a partir da representação LAB uint8 produzida por rgb_to_lab."""
    lab_image = _validate_rgb_image(lab_image).astype(np.float64)

    l_channel = lab_image[:, :, 0] / 255.0 * 100.0
    a_channel = lab_image[:, :, 1] - 128.0
    b_channel = lab_image[:, :, 2] - 128.0

    fy = (l_channel + 16.0) / 116.0
    fx = fy + a_channel / 500.0
    fz = fy - b_channel / 200.0

    delta = 6.0 / 29.0

    def f_inverse(values):
        return np.where(
            values > delta,
            values**3,
            3.0 * delta**2 * (values - 4.0 / 29.0),
        )

    d65_white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz = np.dstack((f_inverse(fx), f_inverse(fy), f_inverse(fz))) * d65_white

    xyz_to_rgb_matrix = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    linear_rgb = xyz @ xyz_to_rgb_matrix.T
    positive_linear_rgb = np.clip(linear_rgb, 0.0, None)
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(positive_linear_rgb, 1.0 / 2.4) - 0.055,
    )

    return _to_uint8(np.clip(srgb, 0.0, 1.0) * 255.0)


def split_channels(image):
    """Retorna os três canais de uma imagem colorida como arrays separados."""
    image = _validate_rgb_image(image)
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


def normalize_to_uint8(image):
    """Normaliza uma imagem para o intervalo de intensidade 0-255."""
    image = np.asarray(image).astype(np.float64)
    min_value = np.min(image)
    max_value = np.max(image)

    if max_value == min_value:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - min_value) / (max_value - min_value)
    return _to_uint8(normalized * 255.0)


def calculate_histogram(image, bins=256, value_range=(0, 256)):
    """Calcula o histograma manualmente, sem funções prontas de histograma."""
    image = np.asarray(image)

    if bins <= 0:
        raise ValueError("A quantidade de classes do histograma deve ser positiva.")

    range_min, range_max = value_range
    if range_max <= range_min:
        raise ValueError("O intervalo do histograma deve ter valor máximo maior que o mínimo.")

    histogram = np.zeros(bins, dtype=np.int64)
    bin_edges = np.linspace(range_min, range_max, bins + 1)
    bin_width = (range_max - range_min) / bins

    for value in image.ravel():
        value = float(value)

        if value < range_min or value > range_max:
            continue

        if value == range_max:
            bin_index = bins - 1
        else:
            bin_index = int((value - range_min) / bin_width)

        histogram[bin_index] += 1

    return histogram, bin_edges


def equalize_histogram(gray_image):
    """Equaliza uma imagem em escala de cinza usando histograma, PDF e CDF."""
    gray_image = np.asarray(gray_image)

    if gray_image.ndim != 2:
        raise ValueError("A equalização de histograma exige uma imagem em escala de cinza.")

    gray_uint8 = _to_uint8(gray_image)
    histogram, _ = calculate_histogram(gray_uint8, bins=256, value_range=(0, 256))
    cdf = histogram.cumsum()
    nonzero_cdf = cdf[cdf > 0]

    if nonzero_cdf.size == 0 or nonzero_cdf[-1] == nonzero_cdf[0]:
        return gray_uint8.copy()

    cdf_min = nonzero_cdf[0]
    equalization_map = np.round((cdf - cdf_min) / (cdf[-1] - cdf_min) * 255.0)
    equalization_map = np.clip(equalization_map, 0, 255).astype(np.uint8)

    return equalization_map[gray_uint8]


def replace_lab_l_channel(rgb_image, new_l_channel):
    """Substitui a luminosidade LAB por um novo canal L uint8 e reconstrói RGB."""
    lab_image = rgb_to_lab(rgb_image)
    lab_image[:, :, 0] = _to_uint8(new_l_channel)
    return lab_to_rgb(lab_image)


def convert_color_spaces(image):
    """Retorna as principais representações de cor usadas no pré-processamento."""
    return {
        "rgb": _validate_rgb_image(image).copy(),
        "gray": rgb_to_grayscale(image),
        "hsv": rgb_to_hsv(image),
        "lab": rgb_to_lab(image),
    }
