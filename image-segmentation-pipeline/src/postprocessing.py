import numpy as np

from src.morphology import (
    binary_closing,
    binary_opening,
    create_structuring_element,
    extract_boundary,
    fill_holes,
    filter_components_by_area,
)


def _as_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    binary_mask = np.asarray(binary_mask)

    if binary_mask.ndim != 2:
        raise ValueError("binary_mask deve ser um array 2D.")

    if binary_mask.shape[0] == 0 or binary_mask.shape[1] == 0:
        raise ValueError("binary_mask não pode ser vazio.")

    return (binary_mask > 0).astype(np.uint8)


def postprocess_cell_mask(
    binary_mask: np.ndarray,
    opening_size: int = 3,
    closing_size: int = 5,
    min_area: int = 80,
    max_area: int | None = None,
    fill_holes_enabled: bool = True,
    connectivity: int = 8,
) -> dict:
    """Refina uma máscara binária inicial de células.
    """
    input_mask = _as_binary_mask(binary_mask)

    if min_area < 0:
        raise ValueError("min_area deve ser não negativo.")

    if max_area is not None and max_area < min_area:
        raise ValueError("max_area deve ser maior ou igual a min_area.")

    opening_element = create_structuring_element("square", opening_size)
    closing_element = create_structuring_element("square", closing_size)
    boundary_element = create_structuring_element("square", 3)

    opened_mask = binary_opening(input_mask, opening_element)
    closed_mask = binary_closing(opened_mask, closing_element)

    if fill_holes_enabled:
        filled_mask = fill_holes(closed_mask, connectivity=connectivity)
    else:
        filled_mask = closed_mask.copy()

    area_filtered_mask = filter_components_by_area(
        filled_mask,
        min_area=min_area,
        max_area=max_area,
        connectivity=connectivity,
    )
    boundary_mask = extract_boundary(area_filtered_mask, boundary_element)
    final_mask = area_filtered_mask.copy()

    return {
        "input": input_mask,
        "opened": opened_mask,
        "closed": closed_mask,
        "filled": filled_mask,
        "area_filtered": area_filtered_mask,
        "boundary": boundary_mask,
        "final": final_mask,
    }
