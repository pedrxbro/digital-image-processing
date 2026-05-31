import numpy as np

from src.morphology import connected_components
from src.preprocessing import convert_color_spaces
from src.segmentation import overlay_binary_mask
from src.thresholding import otsu_threshold_from_values


CHANNELS_BY_COLOR_SPACE = {
    "RGB": ("R", "G", "B"),
    "HSV": ("H", "S", "V"),
    "LAB": ("L", "A", "B"),
}


def get_channel(color_spaces, source, selected_channel):
    """Seleciona um canal 2D RGB, HSV, LAB ou GRAY."""
    source = source.upper()
    selected_channel = selected_channel.upper()

    if source == "GRAY":
        if selected_channel != "GRAY":
            raise ValueError("Para source='GRAY', selected_channel deve ser 'GRAY'.")
        return color_spaces["gray"].copy()

    channel_names = CHANNELS_BY_COLOR_SPACE.get(source)
    if channel_names is None:
        raise ValueError("source deve ser RGB, HSV, LAB ou GRAY.")
    if selected_channel not in channel_names:
        raise ValueError(f"Canal invalido para {source}: {selected_channel}.")

    return color_spaces[source.lower()][:, :, channel_names.index(selected_channel)].copy()


def channel_key(source, selected_channel):
    source = source.upper()
    selected_channel = selected_channel.upper()
    return "GRAY" if source == "GRAY" else f"{source}.{selected_channel}"


def _mask_from_threshold(channel, threshold, direction):
    if direction == "greater_equal":
        return (channel >= threshold).astype(np.uint8)
    if direction == "less_equal":
        return (channel <= threshold).astype(np.uint8)
    raise ValueError("direction deve ser 'greater_equal' ou 'less_equal'.")


def compare_segmentation_channels(image, config=None):
    """Compara canais 2D usando o mesmo Otsu manual e retorna mascaras e overlays."""
    config = config or {}
    channels_to_compare = config.get(
        "channels",
        ("GRAY", "HSV.S", "HSV.V", "LAB.L", "LAB.A", "LAB.B"),
    )
    directions = config.get("directions", {})
    bins = int(config.get("bins", 256))
    connectivity = int(config.get("connectivity", 8))
    color_spaces = convert_color_spaces(image)
    comparisons = []

    for key in channels_to_compare:
        if key.upper() == "GRAY":
            source, selected_channel = "GRAY", "GRAY"
        else:
            source, selected_channel = key.split(".", maxsplit=1)

        normalized_key = channel_key(source, selected_channel)
        channel = get_channel(color_spaces, source, selected_channel)
        threshold = otsu_threshold_from_values(channel, bins=bins, value_range=(0, 256))
        direction = directions.get(normalized_key, "greater_equal")
        mask = _mask_from_threshold(channel, threshold, direction)
        _, components = connected_components(mask, connectivity=connectivity)

        comparisons.append(
            {
                "channel": normalized_key,
                "image": channel,
                "threshold": float(threshold),
                "direction": direction,
                "mask": mask,
                "overlay": overlay_binary_mask(image, mask),
                "component_count": len(components),
                "foreground_ratio": float(np.mean(mask)),
                "observation": (
                    "Compare separacao das celulas coradas e do fundo claro; "
                    "prefira uma mascara com menos ruido e contornos mais completos."
                ),
            }
        )

    return {
        "original": np.asarray(image).copy(),
        "grayscale": color_spaces["gray"],
        "color_spaces": color_spaces,
        "comparisons": comparisons,
    }
