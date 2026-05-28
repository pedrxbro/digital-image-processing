import numpy as np


def _validate_labels(labels):
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels deve ser um array 2D.")

    return labels


def _validate_color_image(image, name):
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} deve ter formato (altura, largura, 3).")

    return image.astype(np.float64, copy=False)


def _validate_gray_image(gray_image, labels):
    if gray_image is None:
        return None

    gray_image = np.asarray(gray_image)

    if gray_image.ndim != 2:
        raise ValueError("gray_image deve ser um array 2D.")

    if gray_image.shape != labels.shape:
        raise ValueError("gray_image deve ter a mesma altura e largura de labels.")

    return gray_image.astype(np.float64, copy=False)


def _validate_feature_inputs(image_hsv, image_lab, labels, gray_image=None):
    labels = _validate_labels(labels)
    image_hsv = _validate_color_image(image_hsv, "image_hsv")
    image_lab = _validate_color_image(image_lab, "image_lab")

    if image_hsv.shape[:2] != labels.shape:
        raise ValueError("image_hsv deve ter a mesma altura e largura de labels.")

    if image_lab.shape[:2] != labels.shape:
        raise ValueError("image_lab deve ter a mesma altura e largura de labels.")

    gray_image = _validate_gray_image(gray_image, labels)

    return image_hsv, image_lab, labels, gray_image


def _mean(values):
    if values.size == 0:
        return 0.0

    return float(np.sum(values) / values.size)


def _std(values, mean_value):
    if values.size == 0:
        return 0.0

    squared_difference = (values - mean_value) ** 2
    return float(np.sqrt(np.sum(squared_difference) / values.size))


def compute_superpixel_features(image_hsv, image_lab, labels, gray_image=None):
    """Calcula atributos de cor e intensidade para cada rótulo de superpixel.

    A implementação usa arrays NumPy, laços explícitos sobre os rótulos e
    máscaras booleanas. Não usa regionprops nem descritor pronto de regiões.
    """
    image_hsv, image_lab, labels, gray_image = _validate_feature_inputs(
        image_hsv,
        image_lab,
        labels,
        gray_image,
    )

    if gray_image is None:
        # Se a imagem em cinza não for fornecida, usamos V como intensidade.
        gray_image = image_hsv[:, :, 2]

    features = []
    unique_labels = np.unique(labels)

    for label_id in unique_labels:
        region_mask = labels == label_id
        ys, xs = np.where(region_mask)
        area = int(ys.size)

        if area == 0:
            continue

        hsv_pixels = image_hsv[region_mask]
        lab_pixels = image_lab[region_mask]
        gray_pixels = gray_image[region_mask]

        h_values = hsv_pixels[:, 0]
        s_values = hsv_pixels[:, 1]
        v_values = hsv_pixels[:, 2]
        l_values = lab_pixels[:, 0]
        a_values = lab_pixels[:, 1]
        b_values = lab_pixels[:, 2]

        mean_h = _mean(h_values)
        mean_s = _mean(s_values)
        mean_v = _mean(v_values)
        mean_l = _mean(l_values)
        mean_a = _mean(a_values)
        mean_b = _mean(b_values)
        mean_gray = _mean(gray_pixels)

        feature = {
            "label_id": int(label_id),
            "area": area,
            "mean_h": mean_h,
            "mean_s": mean_s,
            "mean_v": mean_v,
            "mean_l": mean_l,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_gray": mean_gray,
            "centroid_x": _mean(xs.astype(np.float64)),
            "centroid_y": _mean(ys.astype(np.float64)),
            "std_h": _std(h_values, mean_h),
            "std_s": _std(s_values, mean_s),
            "std_v": _std(v_values, mean_v),
            "std_l": _std(l_values, mean_l),
            "std_a": _std(a_values, mean_a),
            "std_b": _std(b_values, mean_b),
            "min_s": float(np.min(s_values)),
            "max_s": float(np.max(s_values)),
            "min_l": float(np.min(l_values)),
            "max_l": float(np.max(l_values)),
        }

        features.append(feature)

    return features


def get_feature_values(features, feature_name):
    """Retorna um valor numérico por superpixel para o atributo escolhido."""
    values = []

    for feature in features:
        if feature_name not in feature:
            raise KeyError(f"Atributo não encontrado: {feature_name}")

        values.append(float(feature[feature_name]))

    return np.asarray(values, dtype=np.float64)


def feature_values_to_map(labels, features, feature_name, fill_value=0.0):
    """Cria uma imagem em que cada superpixel recebe o valor de um atributo."""
    labels = _validate_labels(labels)
    feature_map = np.full(labels.shape, fill_value, dtype=np.float64)

    for feature in features:
        if feature_name not in feature:
            raise KeyError(f"Atributo não encontrado: {feature_name}")

        label_id = feature["label_id"]
        region_mask = labels == label_id
        feature_map[region_mask] = float(feature[feature_name])

    return feature_map


def feature_map_to_uint8(feature_map, normalize=False):
    """Converte um mapa de atributos para uint8, usado na visualização."""
    feature_map = np.asarray(feature_map, dtype=np.float64)

    if normalize:
        min_value = float(np.min(feature_map))
        max_value = float(np.max(feature_map))

        if max_value == min_value:
            return np.zeros(feature_map.shape, dtype=np.uint8)

        feature_map = (feature_map - min_value) / (max_value - min_value) * 255.0

    return np.clip(np.round(feature_map), 0, 255).astype(np.uint8)
