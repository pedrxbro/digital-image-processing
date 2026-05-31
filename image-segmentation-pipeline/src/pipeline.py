import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.channels import channel_key, compare_segmentation_channels, get_channel
from src.data_loader import load_color_image, load_grayscale_image
from src.edge_enhancement import sobel_edges
from src.frequency import apply_high_pass_filter, apply_low_pass_filter
from src.morphology import (
    binary_distance_transform,
    connected_components,
    ensure_watershed_marker_coverage,
    fill_small_contours,
    marker_controlled_watershed,
    watershed_markers_from_distance,
    watershed_markers_from_erosion,
)
from src.postprocessing import postprocess_cell_mask
from src.preprocessing import convert_color_spaces, equalize_histogram, normalize_to_uint8
from src.segmentation import (
    build_binary_mask_from_labels,
    mask_to_visual,
    overlay_binary_mask,
    overlay_instance_boundaries,
)
from src.superpixel_features import compute_superpixel_features
from src.superpixels import overlay_superpixel_boundaries, slic
from src.thresholding import (
    classify_superpixels_by_rules,
    classify_superpixels_by_threshold,
    otsu_threshold_from_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    "load_image",
    "compare_channels",
    "preprocess",
    "frequency_filter",
    "edge_enhancement",
    "superpixels",
    "segment",
    "morphology",
    "contour_filling",
    "watershed",
    "evaluate",
    "save_outputs",
]

DEFAULT_CONFIG = {
    "input_strategy": {
        "load_rgb": True,
        "load_grayscale": True,
        "primary_segmentation_input": "channel",
        "channel_source": "LAB",
        "selected_channel": "A",
    },
    "channel_comparison": {
        "enabled": True,
        "channels": ("GRAY", "HSV.S", "HSV.V", "LAB.L", "LAB.A", "LAB.B"),
        "directions": {"GRAY": "less_equal", "HSV.V": "less_equal", "LAB.L": "less_equal"},
        "bins": 256,
        "connectivity": 8,
    },
    "preprocessing": {"equalize_histogram": False},
    "frequency_filter": {"enabled": True, "type": "low_pass", "radius": 35},
    "edge_enhancement": {
        "enabled": True,
        "method": "sobel",
        "input": "frequency_filtered",
        "use_as_segmentation_input": False,
        "use_as_watershed_gradient": True,
        "padding": "edge",
    },
    "superpixels": {
        "enabled": True,
        "input": "LAB",
        "num_superpixels": 450,
        "compactness": 10,
        "max_iter": 6,
    },
    "segmentation": {
        "method": "rbc_superpixel_rules",
        "feature": "mean_s",
        "direction": "greater_equal",
        "bins": 128,
        "threshold_offset": 0.0,
        "include_rule_groups": (
            {
                "name": "visible_rbc",
                "description": "Hemacias com saturacao e componente LAB.B suficientes.",
                "rules": (
                    {"feature": "mean_s", "threshold": 20, "operator": ">="},
                    {"feature": "mean_b", "threshold": 115, "operator": ">="},
                ),
            },
            {
                "name": "pale_rbc",
                "description": "Complemento para hemacias palidas preservadas pelo LAB.A.",
                "rules": (
                    {"feature": "mean_a", "threshold": 129, "operator": ">="},
                    {"feature": "mean_b", "threshold": 125, "operator": ">="},
                ),
            },
        ),
        "exclude_rule_groups": (
            {
                "name": "blue_cells",
                "description": "Exclui regioes azuladas mais compativeis com leucocitos.",
                "rules": (
                    {"feature": "mean_s", "threshold": 15, "operator": ">="},
                    {"feature": "mean_b", "threshold": 120, "operator": "<="},
                ),
            },
        ),
    },
    "morphology": {
        "enabled": True,
        "opening_size": 5,
        "closing_size": 3,
        "min_object_area": 700,
        "max_object_area": None,
        "fill_holes": False,
        "connectivity": 8,
    },
    "contour_filling": {
        "enabled": True,
        "input": "morphology",
        "max_contour_area_to_fill": 250,
        "connectivity": 8,
    },
    "watershed": {
        "enabled": True,
        "input": "contour_filled_mask",
        "marker_strategy": "erosion",
        "min_distance": 4,
        "min_separation": 20,
        "erosion_iterations": 12,
        "connectivity": 8,
        "use_edge_gradient": True,
        "ensure_component_coverage": True,
    },
    "output_mask_source": "watershed_mask",
    "output": {
        "save_report": True,
        "save_artifacts": True,
        "save_pipeline_figure": True,
        "output_dir": "outputs/pipeline_final",
    },
}


def _merge_config(base, custom):
    merged = dict(base)
    for key, value in custom.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(overrides=None):
    """Retorna a configuracao final com sobrescritas opcionais."""
    return _merge_config(DEFAULT_CONFIG, overrides or {})


def _resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _select_2d_input(result, source):
    if source == "grayscale":
        return result["grayscale"]
    if source == "selected_channel":
        return result["selected_channel"]
    if source == "frequency_filtered":
        return result.get("frequency_filtered", result["preprocessed"])
    if source == "edge_enhanced":
        return result["edge_enhanced"]
    raise ValueError(f"Entrada 2D desconhecida: {source}")


def _summarize_values(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _array_metrics(image):
    image = np.asarray(image)
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        **_summarize_values(image),
    }


def _mask_metrics(mask, connectivity=8):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    _, components = connected_components(mask, connectivity=connectivity)
    areas = [info["area"] for info in components.values()]
    return {
        "foreground_pixels": int(np.sum(mask)),
        "foreground_ratio": float(np.mean(mask)),
        "component_count": len(components),
        "component_areas": _summarize_values(areas),
    }


def _instance_metrics(instances):
    instances = np.asarray(instances)
    instance_ids = np.unique(instances[instances > 0])
    areas = [int(np.sum(instances == instance_id)) for instance_id in instance_ids]
    return {
        "instance_count": int(instance_ids.size),
        "boundary_pixels": int(np.sum(instances < 0)),
        "instance_areas": _summarize_values(areas),
    }


def _channel_comparison_metrics(channel_comparison):
    if channel_comparison is None:
        return []
    return [
        {
            "channel": comparison["channel"],
            "threshold": float(comparison["threshold"]),
            "direction": comparison["direction"],
            "component_count": int(comparison["component_count"]),
            "foreground_ratio": float(comparison["foreground_ratio"]),
        }
        for comparison in channel_comparison["comparisons"]
    ]


def apply_edge_enhancement_stage(result, config):
    edge_config = config["edge_enhancement"]
    if not edge_config.get("enabled", False):
        return
    if edge_config.get("method", "sobel").lower() != "sobel":
        raise ValueError("O unico metodo de bordas suportado e 'sobel'.")

    edge_input = _select_2d_input(result, edge_config.get("input", "frequency_filtered"))
    edges = sobel_edges(edge_input, padding=edge_config.get("padding", "edge"))
    result["edge_gx"], result["edge_gy"] = edges["gx"], edges["gy"]
    result["edge_magnitude"], result["edge_enhanced"] = edges["magnitude"], edges["enhanced"]
    result["edge_info"] = {key: edges[key] for key in ("method", "padding", "kernels")}


def apply_watershed_stage(result, config):
    watershed_config = config["watershed"]
    if not watershed_config.get("enabled", False):
        result["watershed_mask"] = result.get("contour_filled_mask", result["morphology_mask"])
        result["watershed_info"] = {"enabled": False}
        return

    input_name = watershed_config.get("input", "contour_filled_mask")
    binary_mask = result.get(input_name, result.get("contour_filled_mask", result["morphology_mask"]))
    connectivity = int(watershed_config.get("connectivity", 8))
    distance_map = binary_distance_transform(binary_mask)
    strategy = watershed_config.get("marker_strategy", "erosion")

    if strategy == "distance":
        markers = watershed_markers_from_distance(
            binary_mask,
            distance_map,
            min_distance=int(watershed_config.get("min_distance", 4)),
            min_separation=int(watershed_config.get("min_separation", 20)),
            connectivity=connectivity,
        )
    elif strategy == "erosion":
        markers = watershed_markers_from_erosion(
            binary_mask,
            erosion_iterations=int(watershed_config.get("erosion_iterations", 12)),
            connectivity=connectivity,
        )
    else:
        raise ValueError("watershed.marker_strategy deve ser 'distance' ou 'erosion'.")

    if watershed_config.get("ensure_component_coverage", True):
        markers = ensure_watershed_marker_coverage(
            binary_mask,
            markers,
            distance_map,
            connectivity=connectivity,
        )

    use_edge_gradient = (
        watershed_config.get("use_edge_gradient", True)
        and config["edge_enhancement"].get("use_as_watershed_gradient", True)
    )
    gradient_map = result.get("edge_enhanced") if use_edge_gradient else None
    instances = marker_controlled_watershed(
        binary_mask,
        markers,
        distance_map,
        connectivity=connectivity,
        gradient_map=gradient_map,
    )
    _, components_before = connected_components(binary_mask, connectivity=connectivity)
    marker_count = int(np.unique(markers[markers > 0]).size)
    instance_count = int(np.unique(instances[instances > 0]).size)

    result["watershed_instances"] = instances
    result["watershed_markers"] = markers
    result["watershed_distance_map"] = distance_map
    result["watershed_mask"] = (instances > 0).astype(np.uint8)
    result["watershed_info"] = {
        "enabled": True,
        "input": input_name,
        "marker_strategy": strategy,
        "connectivity": connectivity,
        "use_edge_gradient": gradient_map is not None,
        "ensure_component_coverage": bool(watershed_config.get("ensure_component_coverage", True)),
        "min_distance": int(watershed_config.get("min_distance", 4)),
        "min_separation": int(watershed_config.get("min_separation", 20)),
        "erosion_iterations": int(watershed_config.get("erosion_iterations", 12)),
        "components_before": len(components_before),
        "marker_count": marker_count,
        "instances_after": instance_count,
    }


def collect_stage_metrics(result):
    """Coleta informacoes quantitativas de cada etapa executada."""
    connectivity = int(result["config"]["morphology"].get("connectivity", 8))
    morphology_metrics = {}
    for key, mask in result.get("morphology_history", {}).items():
        morphology_metrics[key] = _mask_metrics(mask, connectivity=connectivity)

    segmentation_info = result["segmentation_info"]
    return {
        "load_image": {
            "original": _array_metrics(result["original"]),
            "grayscale": _array_metrics(result["grayscale"]),
        },
        "compare_channels": _channel_comparison_metrics(result.get("channel_comparison")),
        "preprocess": {
            "selected_channel": result["selected_channel_name"],
            "selected_channel_metrics": _array_metrics(result["selected_channel"]),
            "preprocessed_metrics": _array_metrics(result["preprocessed"]),
        },
        "frequency_filter": {
            **result["config"]["frequency_filter"],
            "output_metrics": _array_metrics(result["frequency_filtered"]),
        },
        "edge_enhancement": {
            **result.get("edge_info", {"enabled": False}),
            "magnitude_metrics": _array_metrics(result.get("edge_magnitude", np.zeros_like(result["grayscale"]))),
        },
        "superpixels": {
            **result["config"]["superpixels"],
            "generated_labels": int(np.unique(result["superpixel_labels"]).size),
        },
        "segment": {
            "method": segmentation_info["method"],
            "threshold": segmentation_info.get("threshold"),
            "selected_labels": len(segmentation_info.get("selected_labels", [])),
            "excluded_labels": len(segmentation_info.get("excluded_labels", [])),
            "rule_groups": segmentation_info.get("rule_groups", []),
            "initial_mask": _mask_metrics(result["initial_mask"], connectivity=connectivity),
        },
        "morphology": morphology_metrics,
        "contour_filling": {
            **result["contour_filling_info"],
            "output_mask": _mask_metrics(result["contour_filled_mask"], connectivity=connectivity),
        },
        "watershed": {
            **result["watershed_info"],
            "mask": _mask_metrics(result["watershed_mask"], connectivity=connectivity),
            "instances": _instance_metrics(result["watershed_instances"]),
        },
        "evaluate": {
            "final_mask_source": result["config"]["output_mask_source"],
            "final_mask": _mask_metrics(result["final_mask"], connectivity=connectivity),
            "estimated_rbc_count": int(result["estimated_rbc_count"]),
        },
        "timings_seconds": result["timings_seconds"],
    }


def collect_execution_report(result):
    """Resume configuracao, metricas por etapa e contagem final de hemacias."""
    return {
        "image": str(result["image_path"]),
        "pipeline_steps": result["executed_steps"],
        "summary": {
            "selected_channel": result["selected_channel_name"],
            "segmentation_method": result["segmentation_info"]["method"],
            "estimated_rbc_count": int(result["estimated_rbc_count"]),
            "components_before_watershed": int(result["watershed_info"].get("components_before", 0)),
            "instances_after_watershed": int(result["watershed_info"].get("instances_after", 0)),
            "filled_small_contours": int(result["contour_filling_info"].get("filled_contours", 0)),
            "final_foreground_ratio": float(np.mean(result["final_mask"])),
            "total_seconds": float(sum(result["timings_seconds"].values())),
            "has_reference_mask": False,
        },
        "stage_metrics": result["stage_metrics"],
        "config": result["config"],
        "notes": (
            "A contagem e estimada a partir dos rotulos positivos do Watershed. "
            "Sem mascara de referencia, a validacao quantitativa usa contagem, "
            "proporcao de foreground e inspecao visual dos overlays."
        ),
    }


def _save_image(path, image, cmap=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    if image.ndim == 2 and cmap is None:
        cmap = "gray"
    plt.imsave(path, image, cmap=cmap)
    return str(path)


def save_pipeline_outputs(result, config):
    """Salva artefatos, configuracao e relatorio detalhado por imagem."""
    output_dir = _resolve_project_path(config["output"].get("output_dir", "outputs/pipeline_final"))
    image_dir = output_dir / Path(result["image_path"]).stem
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    if config["output"].get("save_artifacts", True):
        paths["original"] = _save_image(image_dir / "01_original.png", result["original"])
        paths["grayscale"] = _save_image(image_dir / "02_grayscale.png", result["grayscale"])
        paths["selected_channel"] = _save_image(image_dir / "03_selected_channel.png", result["selected_channel"])
        paths["frequency_filtered"] = _save_image(image_dir / "04_frequency_filtered.png", result["frequency_filtered"])
        paths["edge_magnitude"] = _save_image(image_dir / "05_sobel_magnitude.png", result["edge_enhanced"])
        paths["superpixels"] = _save_image(image_dir / "06_superpixels.png", result["superpixel_overlay"])
        paths["initial_mask"] = _save_image(image_dir / "07_initial_rbc_mask.png", mask_to_visual(result["initial_mask"]))

        for index, (name, mask) in enumerate(result.get("morphology_history", {}).items(), start=8):
            paths[f"morphology_{name}"] = _save_image(
                image_dir / f"{index:02d}_morphology_{name}.png",
                mask_to_visual(mask),
            )

        paths["contour_filled_mask"] = _save_image(
            image_dir / "15_contour_filled_mask.png",
            mask_to_visual(result["contour_filled_mask"]),
        )
        paths["watershed_markers"] = _save_image(
            image_dir / "16_watershed_markers.png",
            normalize_to_uint8(result["watershed_markers"]),
            cmap="nipy_spectral",
        )
        paths["watershed_distance_map"] = _save_image(
            image_dir / "17_watershed_distance_map.png",
            normalize_to_uint8(result["watershed_distance_map"]),
        )
        paths["watershed_instances"] = _save_image(
            image_dir / "18_watershed_instances.png",
            np.maximum(result["watershed_instances"], 0),
            cmap="nipy_spectral",
        )
        paths["final_mask"] = _save_image(image_dir / "19_final_mask.png", mask_to_visual(result["final_mask"]))
        paths["final_overlay"] = _save_image(image_dir / "20_final_overlay.png", result["final_overlay"])
        paths["instance_overlay"] = _save_image(image_dir / "21_instance_overlay.png", result["instance_overlay"])
        instances_path = image_dir / "watershed_instances.npy"
        np.save(instances_path, result["watershed_instances"])
        paths["watershed_instances_array"] = str(instances_path)

    config_path = image_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    paths["config"] = str(config_path)

    report_path = image_dir / "execution_report.json"
    paths["execution_report"] = str(report_path)

    if config["output"].get("save_pipeline_figure", True):
        from src.visualization import show_pipeline_evolution

        figure_path = image_dir / "pipeline_evolution.png"
        show_pipeline_evolution(result, save_path=figure_path, show=False)
        paths["pipeline_evolution"] = str(figure_path)

    result["output_paths"] = paths
    result["execution_report"]["output_paths"] = paths
    report_path.write_text(json.dumps(result["execution_report"], indent=2), encoding="utf-8")
    result["report_path"] = report_path

    return paths


def _segment_rbc_superpixels(result, segmentation_config, source):
    features = compute_superpixel_features(
        result["color_spaces"]["hsv"],
        result["color_spaces"]["lab"],
        result["superpixel_labels"],
        gray_image=source,
    )
    labels = result["superpixel_labels"]
    selected_labels = set()
    excluded_labels = set()
    rule_groups = []
    result["segmentation_rule_masks"] = {}

    for group in segmentation_config.get("include_rule_groups", ()):
        group_labels = set(classify_superpixels_by_rules(features, group["rules"], logic=group.get("logic", "and")))
        selected_labels.update(group_labels)
        result["segmentation_rule_masks"][group["name"]] = build_binary_mask_from_labels(labels, group_labels)
        rule_groups.append({"name": group["name"], "type": "include", "label_count": len(group_labels), "rules": group["rules"]})

    result["rbc_candidate_mask"] = build_binary_mask_from_labels(labels, selected_labels)

    for group in segmentation_config.get("exclude_rule_groups", ()):
        group_labels = set(classify_superpixels_by_rules(features, group["rules"], logic=group.get("logic", "and")))
        excluded_labels.update(group_labels)
        result["segmentation_rule_masks"][group["name"]] = build_binary_mask_from_labels(labels, group_labels)
        rule_groups.append({"name": group["name"], "type": "exclude", "label_count": len(group_labels), "rules": group["rules"]})

    initial_mask = result["rbc_candidate_mask"].copy()
    result["excluded_mask"] = build_binary_mask_from_labels(labels, excluded_labels)
    initial_mask[result["excluded_mask"] > 0] = 0
    selected_labels.difference_update(excluded_labels)

    return initial_mask, {
        "method": "rbc_superpixel_rules",
        "threshold": None,
        "features": features,
        "selected_labels": sorted(selected_labels),
        "excluded_labels": sorted(excluded_labels),
        "rule_groups": rule_groups,
    }


def run_pipeline(image_path, overrides=None, steps=None):
    """Executa a pipeline final e retorna todos os intermediarios."""
    config = build_config(overrides)
    steps = steps or PIPELINE_STEPS
    result = {
        "image_path": Path(image_path),
        "config": config,
        "executed_steps": [],
        "timings_seconds": {},
    }

    for step in steps:
        started_at = time.perf_counter()

        if step == "load_image":
            result["original"] = load_color_image(image_path)
            result["grayscale"] = load_grayscale_image(image_path)
            result["color_spaces"] = convert_color_spaces(result["original"])
            strategy = config["input_strategy"]
            result["selected_channel"] = get_channel(result["color_spaces"], strategy["channel_source"], strategy["selected_channel"])
            result["selected_channel_name"] = channel_key(strategy["channel_source"], strategy["selected_channel"])

        elif step == "compare_channels" and config["channel_comparison"].get("enabled", False):
            result["channel_comparison"] = compare_segmentation_channels(result["original"], config["channel_comparison"])

        elif step == "preprocess":
            result["preprocessed"] = result["selected_channel"].copy()
            if config["preprocessing"].get("equalize_histogram", False):
                result["preprocessed"] = equalize_histogram(result["preprocessed"])

        elif step == "frequency_filter":
            frequency_config = config["frequency_filter"]
            result["frequency_filtered"] = result["preprocessed"].copy()
            if frequency_config.get("enabled", False):
                if frequency_config["type"] == "low_pass":
                    filter_function = apply_low_pass_filter
                elif frequency_config["type"] == "high_pass":
                    filter_function = apply_high_pass_filter
                else:
                    raise ValueError("frequency_filter.type deve ser 'low_pass' ou 'high_pass'.")
                result["frequency_filtered"] = filter_function(result["preprocessed"], int(frequency_config["radius"]))

        elif step == "edge_enhancement":
            apply_edge_enhancement_stage(result, config)

        elif step == "superpixels" and config["superpixels"].get("enabled", False):
            superpixel_config = config["superpixels"]
            labels = slic(
                result["color_spaces"][superpixel_config.get("input", "LAB").lower()],
                num_superpixels=int(superpixel_config["num_superpixels"]),
                m=float(superpixel_config["compactness"]),
                max_iter=int(superpixel_config["max_iter"]),
            )
            result["superpixel_labels"] = labels
            result["superpixel_overlay"] = overlay_superpixel_boundaries(result["original"], labels)

        elif step == "segment":
            segmentation_config = config["segmentation"]
            source_names = {
                "channel": "selected_channel",
                "frequency_filtered": "frequency_filtered",
                "edge_enhanced": "edge_enhanced",
                "grayscale": "grayscale",
            }
            source = _select_2d_input(result, source_names[config["input_strategy"].get("primary_segmentation_input", "channel")])
            if config["edge_enhancement"].get("use_as_segmentation_input", False):
                source = result["edge_enhanced"]
            method = segmentation_config.get("method", "rbc_superpixel_rules")

            if method == "rbc_superpixel_rules":
                result["initial_mask"], result["segmentation_info"] = _segment_rbc_superpixels(result, segmentation_config, source)
            elif method == "otsu_pixel":
                threshold = otsu_threshold_from_values(source, bins=int(segmentation_config["bins"]), value_range=(0, 256))
                threshold += float(segmentation_config.get("threshold_offset", 0.0))
                direction = segmentation_config.get("direction", "greater_equal")
                result["initial_mask"] = ((source >= threshold) if direction == "greater_equal" else (source <= threshold)).astype(np.uint8)
                result["segmentation_info"] = {"method": method, "threshold": float(threshold), "features": None}
            elif method == "otsu_superpixel":
                features = compute_superpixel_features(
                    result["color_spaces"]["hsv"],
                    result["color_spaces"]["lab"],
                    result["superpixel_labels"],
                    gray_image=source,
                )
                feature_name = segmentation_config["feature"]
                threshold = otsu_threshold_from_values([feature[feature_name] for feature in features], bins=int(segmentation_config["bins"]))
                threshold += float(segmentation_config.get("threshold_offset", 0.0))
                labels = classify_superpixels_by_threshold(features, feature_name, threshold, direction=segmentation_config.get("direction", "greater_equal"))
                result["initial_mask"] = build_binary_mask_from_labels(result["superpixel_labels"], labels)
                result["segmentation_info"] = {"method": method, "threshold": float(threshold), "features": features, "selected_labels": labels}
            else:
                raise ValueError("Metodo de segmentacao desconhecido.")

        elif step == "morphology":
            morphology_config = config["morphology"]
            if morphology_config.get("enabled", False):
                history = postprocess_cell_mask(
                    result["initial_mask"],
                    opening_size=int(morphology_config["opening_size"]),
                    closing_size=int(morphology_config["closing_size"]),
                    min_area=int(morphology_config["min_object_area"]),
                    max_area=morphology_config.get("max_object_area"),
                    fill_holes_enabled=bool(morphology_config.get("fill_holes", False)),
                    connectivity=int(morphology_config.get("connectivity", 8)),
                )
                result["morphology_history"], result["morphology_mask"] = history, history["final"]
            else:
                result["morphology_mask"] = result["initial_mask"].copy()

        elif step == "contour_filling":
            contour_config = config["contour_filling"]
            contour_input = result["morphology_mask"] if contour_config.get("input") == "morphology" else result["initial_mask"]
            if contour_config.get("enabled", False):
                filled, info = fill_small_contours(
                    contour_input,
                    max_area=int(contour_config["max_contour_area_to_fill"]),
                    connectivity=int(contour_config.get("connectivity", 8)),
                )
            else:
                filled, info = contour_input.copy(), {"enabled": False, "filled_contours": 0}
            result["contour_filled_mask"], result["contour_filling_info"] = filled, info

        elif step == "watershed":
            apply_watershed_stage(result, config)

        elif step == "evaluate":
            result["final_mask"] = result[config["output_mask_source"]]
            result["estimated_rbc_count"] = int(np.unique(result["watershed_instances"][result["watershed_instances"] > 0]).size)
            result["final_overlay"] = overlay_binary_mask(result["original"], result["final_mask"], color=(0, 255, 0), alpha=0.32)
            result["instance_overlay"] = overlay_instance_boundaries(result["final_overlay"], result["watershed_instances"], color=(255, 255, 0), alpha=0.95)

        elif step == "save_outputs":
            pass

        result["executed_steps"].append(step)
        result["timings_seconds"][step] = float(time.perf_counter() - started_at)

    if "evaluate" in result["executed_steps"]:
        result["stage_metrics"] = collect_stage_metrics(result)
        result["execution_report"] = collect_execution_report(result)

    if "save_outputs" in result["executed_steps"] and config["output"].get("save_report", True):
        result["output_paths"] = save_pipeline_outputs(result, config)

    return result


def run_pipeline_for_images(image_paths, overrides=None, steps=None):
    """Executa a mesma configuracao final para varias imagens e salva um resumo."""
    results = [run_pipeline(image_path, overrides=overrides, steps=steps) for image_path in image_paths]
    config = build_config(overrides)
    output_dir = _resolve_project_path(config["output"].get("output_dir", "outputs/pipeline_final"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for result in results:
        summary = result["execution_report"]["summary"]
        rows.append(
            {
                "image": Path(result["image_path"]).name,
                "estimated_rbc_count": summary["estimated_rbc_count"],
                "components_before_watershed": summary["components_before_watershed"],
                "instances_after_watershed": summary["instances_after_watershed"],
                "filled_small_contours": summary["filled_small_contours"],
                "final_foreground_ratio": summary["final_foreground_ratio"],
                "total_seconds": summary["total_seconds"],
            }
        )

    csv_path = output_dir / "batch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "batch_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return results, rows
