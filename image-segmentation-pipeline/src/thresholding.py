import numpy as np

from src.superpixel_features import get_feature_values


def _compare(value, threshold, operator):
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    if operator == "==":
        return value == threshold
    if operator == "!=":
        return value != threshold

    raise ValueError(f"Operador inválido: {operator}")


def _direction_to_operator(direction):
    if direction == "greater_equal":
        return ">="
    if direction == "greater":
        return ">"
    if direction == "less_equal":
        return "<="
    if direction == "less":
        return "<"

    raise ValueError("direction deve ser greater_equal, greater, less_equal ou less.")


def classify_superpixels_by_threshold(features, feature_name, threshold, direction="greater_equal"):
    """Classifica superpixels usando um limiar global sobre um atributo."""
    operator = _direction_to_operator(direction)
    selected_labels = []

    for feature in features:
        value = float(feature[feature_name])

        if _compare(value, threshold, operator):
            selected_labels.append(int(feature["label_id"]))

    return selected_labels


def classify_superpixels_by_rules(features, rules, logic="and"):
    """Classifica superpixels usando uma lista de regras definidas manualmente."""
    if logic not in ("and", "or"):
        raise ValueError("logic deve ser 'and' ou 'or'.")

    selected_labels = []

    for feature in features:
        rule_results = []

        for rule in rules:
            feature_name = rule["feature"]
            threshold = float(rule["threshold"])
            operator = rule.get("operator", ">=")
            value = float(feature[feature_name])
            rule_results.append(_compare(value, threshold, operator))

        if logic == "and" and all(rule_results):
            selected_labels.append(int(feature["label_id"]))

        if logic == "or" and any(rule_results):
            selected_labels.append(int(feature["label_id"]))

    return selected_labels


def manual_histogram(values, bins=256, value_range=None):
    """Calcula um histograma unidimensional sem usar np.histogram."""
    values = np.asarray(values, dtype=np.float64).ravel()

    if values.size == 0:
        raise ValueError("values não pode ser vazio.")

    if bins <= 0:
        raise ValueError("bins deve ser maior que zero.")

    if value_range is None:
        min_value = float(np.min(values))
        max_value = float(np.max(values))
    else:
        min_value = float(value_range[0])
        max_value = float(value_range[1])

    if max_value == min_value:
        max_value = min_value + 1.0

    if max_value < min_value:
        raise ValueError("value_range deve ter máximo maior que mínimo.")

    histogram = np.zeros(bins, dtype=np.int64)
    bin_edges = np.linspace(min_value, max_value, bins + 1)
    bin_width = (max_value - min_value) / bins

    for value in values:
        if value < min_value or value > max_value:
            continue

        if value == max_value:
            bin_index = bins - 1
        else:
            bin_index = int((value - min_value) / bin_width)

        histogram[bin_index] += 1

    return histogram, bin_edges


def otsu_threshold_from_values(values, bins=256, value_range=None):
    """Calcula o limiar de Otsu do zero para valores unidimensionais."""
    values = np.asarray(values, dtype=np.float64).ravel()

    if values.size == 0:
        raise ValueError("values não pode ser vazio.")

    histogram, bin_edges = manual_histogram(values, bins=bins, value_range=value_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total_count = int(np.sum(histogram))
    if total_count == 0:
        return float(bin_centers[0])

    total_sum = float(np.sum(histogram * bin_centers))
    background_count = 0.0
    background_sum = 0.0
    best_threshold = float(bin_centers[0])
    best_variance = -1.0

    for index in range(bins):
        background_count += float(histogram[index])
        background_sum += float(histogram[index] * bin_centers[index])

        foreground_count = total_count - background_count

        if background_count == 0 or foreground_count == 0:
            continue

        background_mean = background_sum / background_count
        foreground_mean = (total_sum - background_sum) / foreground_count
        between_variance = (
            background_count
            * foreground_count
            * (background_mean - foreground_mean) ** 2
        )

        if between_variance > best_variance:
            best_variance = float(between_variance)
            best_threshold = float(bin_centers[index])

    return best_threshold


def classify_superpixels_by_otsu(features, feature_name, bins=256, direction="greater_equal"):
    """Aplica Otsu manual sobre atributos de superpixels e classifica rótulos."""
    values = get_feature_values(features, feature_name)
    threshold = otsu_threshold_from_values(values, bins=bins)
    selected_labels = classify_superpixels_by_threshold(
        features,
        feature_name,
        threshold,
        direction=direction,
    )

    return threshold, selected_labels
