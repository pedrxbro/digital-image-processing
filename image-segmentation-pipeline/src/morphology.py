from collections import deque
import heapq

import numpy as np


def _as_binary_image(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Valida uma imagem 2D e converte qualquer valor positivo para 1."""
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(f"{name} deve ser um array 2D.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} não pode ser vazio.")

    return (image > 0).astype(np.uint8)


def _validate_structuring_element(structuring_element: np.ndarray) -> np.ndarray:
    """Garante que o elemento estruturante seja binário e tenha centro definido."""
    structuring_element = np.asarray(structuring_element)

    if structuring_element.ndim != 2:
        raise ValueError("structuring_element deve ser um array 2D.")

    element_height, element_width = structuring_element.shape
    if element_height == 0 or element_width == 0:
        raise ValueError("structuring_element não pode ser vazio.")

    if element_height % 2 == 0 or element_width % 2 == 0:
        raise ValueError("As dimensões do elemento estruturante devem ser ímpares.")

    binary_element = (structuring_element > 0).astype(np.uint8)
    if np.sum(binary_element) == 0:
        raise ValueError("structuring_element deve conter pelo menos um valor 1.")

    return binary_element


def _validate_size(size: int) -> int:
    if not isinstance(size, (int, np.integer)):
        raise TypeError("size deve ser um número inteiro.")

    if size <= 0:
        raise ValueError("size deve ser maior que zero.")

    if size % 2 == 0:
        raise ValueError("O tamanho do elemento estruturante deve ser ímpar.")

    return int(size)


def _validate_connectivity(connectivity: int) -> int:
    if connectivity not in (4, 8):
        raise ValueError("connectivity deve ser 4 ou 8.")

    return int(connectivity)


def _neighbor_offsets(connectivity: int) -> list[tuple[int, int]]:
    connectivity = _validate_connectivity(connectivity)

    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    return offsets


def create_structuring_element(shape: str = "square", size: int = 3) -> np.ndarray:
    """Cria um elemento estruturante binário."""
    size = _validate_size(size)
    shape = shape.lower()
    element = np.zeros((size, size), dtype=np.uint8)
    center = size // 2

    if shape == "square":
        for row in range(size):
            for col in range(size):
                element[row, col] = 1

    elif shape == "cross":
        for row in range(size):
            element[row, center] = 1
        for col in range(size):
            element[center, col] = 1

    elif shape == "disk":
        radius = center
        for row in range(size):
            for col in range(size):
                distance = np.sqrt((row - center) ** 2 + (col - center) ** 2)
                if distance <= radius:
                    element[row, col] = 1

    else:
        raise ValueError("Forma inválida. Use 'square', 'cross' ou 'disk'.")

    return element


def add_binary_padding(
    binary_image: np.ndarray,
    padding_height: int,
    padding_width: int | None = None,
    value: int = 0,
) -> np.ndarray:
    """Adiciona borda constante a uma imagem binária."""
    binary_image = _as_binary_image(binary_image, "binary_image")

    if padding_width is None:
        padding_width = padding_height

    if padding_height < 0 or padding_width < 0:
        raise ValueError("padding_height e padding_width devem ser não negativos.")

    padding_height = int(padding_height)
    padding_width = int(padding_width)
    value = 1 if value > 0 else 0

    image_height, image_width = binary_image.shape
    padded_image = np.full(
        (
            image_height + 2 * padding_height,
            image_width + 2 * padding_width,
        ),
        value,
        dtype=np.uint8,
    )
    padded_image[
        padding_height:padding_height + image_height,
        padding_width:padding_width + image_width,
    ] = binary_image

    return padded_image


def erode_binary(
    binary_image: np.ndarray,
    structuring_element: np.ndarray,
    padding_value: int = 0,
) -> np.ndarray:
    """Aplica erosão binária manual.

    Um pixel só permanece 1 quando todos os pontos cobertos pelo elemento
    estruturante também são 1 na imagem de entrada.
    """
    binary_image = _as_binary_image(binary_image, "binary_image")
    structuring_element = _validate_structuring_element(structuring_element)

    image_height, image_width = binary_image.shape
    element_height, element_width = structuring_element.shape
    row_offset = element_height // 2
    col_offset = element_width // 2

    padded_image = add_binary_padding(
        binary_image,
        row_offset,
        col_offset,
        value=padding_value,
    )
    output = np.zeros((image_height, image_width), dtype=np.uint8)

    for image_row in range(image_height):
        for image_col in range(image_width):
            keep_pixel = True

            for element_row in range(element_height):
                for element_col in range(element_width):
                    if structuring_element[element_row, element_col] == 0:
                        continue

                    padded_row = image_row + element_row
                    padded_col = image_col + element_col
                    if padded_image[padded_row, padded_col] == 0:
                        keep_pixel = False
                        break

                if not keep_pixel:
                    break

            if keep_pixel:
                output[image_row, image_col] = 1

    return output


def dilate_binary(
    binary_image: np.ndarray,
    structuring_element: np.ndarray,
) -> np.ndarray:
    """Aplica dilatação binária manual.

    Um pixel da saída vira 1 quando pelo menos um ponto coberto pelo elemento
    estruturante encontra valor 1 na imagem de entrada.
    """
    binary_image = _as_binary_image(binary_image, "binary_image")
    structuring_element = _validate_structuring_element(structuring_element)

    image_height, image_width = binary_image.shape
    element_height, element_width = structuring_element.shape
    row_offset = element_height // 2
    col_offset = element_width // 2

    padded_image = add_binary_padding(binary_image, row_offset, col_offset, value=0)
    output = np.zeros((image_height, image_width), dtype=np.uint8)

    for image_row in range(image_height):
        for image_col in range(image_width):
            grow_pixel = False

            for element_row in range(element_height):
                for element_col in range(element_width):
                    if structuring_element[element_row, element_col] == 0:
                        continue

                    padded_row = image_row + element_row
                    padded_col = image_col + element_col
                    if padded_image[padded_row, padded_col] == 1:
                        grow_pixel = True
                        break

                if grow_pixel:
                    break

            if grow_pixel:
                output[image_row, image_col] = 1

    return output


def binary_opening(
    binary_image: np.ndarray,
    structuring_element: np.ndarray,
) -> np.ndarray:
    """Executa abertura: erosão seguida de dilatação."""
    eroded_image = erode_binary(binary_image, structuring_element)
    opened_image = dilate_binary(eroded_image, structuring_element)
    return opened_image


def binary_closing(
    binary_image: np.ndarray,
    structuring_element: np.ndarray,
) -> np.ndarray:
    """Executa fechamento: dilatação seguida de erosão."""
    dilated_image = dilate_binary(binary_image, structuring_element)
    # Fora da imagem nao significa fundo conhecido. No fechamento, assumir 1
    # evita apagar objetos parcialmente visiveis nas bordas apos a dilatacao.
    closed_image = erode_binary(dilated_image, structuring_element, padding_value=1)
    return closed_image


def extract_boundary(
    binary_image: np.ndarray,
    structuring_element: np.ndarray | None = None,
) -> np.ndarray:
    """Extrai a fronteira binária pela diferença entre imagem e erosão."""
    binary_image = _as_binary_image(binary_image, "binary_image")

    if structuring_element is None:
        structuring_element = create_structuring_element("square", 3)

    eroded_image = erode_binary(binary_image, structuring_element)
    boundary = binary_image - eroded_image

    return (boundary > 0).astype(np.uint8)


def fill_holes(binary_image: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Preenche buracos internos de objetos binários usando busca em largura.

    A ideia é marcar todo fundo conectado às bordas. Qualquer pixel de fundo
    não visitado por essa busca não tem caminho até a borda, logo é buraco e
    deve ser preenchido.
    """
    binary_image = _as_binary_image(binary_image, "binary_image")
    connectivity = _validate_connectivity(connectivity)
    offsets = _neighbor_offsets(connectivity)

    height, width = binary_image.shape
    visited_background = np.zeros((height, width), dtype=bool)
    queue = deque()

    for col in range(width):
        if binary_image[0, col] == 0:
            queue.append((0, col))
            visited_background[0, col] = True
        if binary_image[height - 1, col] == 0 and not visited_background[height - 1, col]:
            queue.append((height - 1, col))
            visited_background[height - 1, col] = True

    for row in range(height):
        if binary_image[row, 0] == 0 and not visited_background[row, 0]:
            queue.append((row, 0))
            visited_background[row, 0] = True
        if binary_image[row, width - 1] == 0 and not visited_background[row, width - 1]:
            queue.append((row, width - 1))
            visited_background[row, width - 1] = True

    while queue:
        row, col = queue.popleft()

        for row_delta, col_delta in offsets:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta

            if neighbor_row < 0 or neighbor_row >= height:
                continue
            if neighbor_col < 0 or neighbor_col >= width:
                continue
            if visited_background[neighbor_row, neighbor_col]:
                continue
            if binary_image[neighbor_row, neighbor_col] == 1:
                continue

            visited_background[neighbor_row, neighbor_col] = True
            queue.append((neighbor_row, neighbor_col))

    filled_image = binary_image.copy()
    holes = (binary_image == 0) & (~visited_background)
    filled_image[holes] = 1

    return filled_image.astype(np.uint8)


def connected_components(
    binary_image: np.ndarray,
    connectivity: int = 8,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Rotula componentes conexos manualmente com BFS.

    Retorna uma matriz de rótulos, onde 0 é fundo, e um dicionário com área,
    bounding box, centróide, largura, altura e pixels de cada componente.
    """
    binary_image = _as_binary_image(binary_image, "binary_image")
    connectivity = _validate_connectivity(connectivity)
    offsets = _neighbor_offsets(connectivity)

    height, width = binary_image.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components_info = {}
    current_label = 0

    for start_row in range(height):
        for start_col in range(width):
            if binary_image[start_row, start_col] == 0:
                continue
            if labels[start_row, start_col] != 0:
                continue

            current_label += 1
            queue = deque([(start_row, start_col)])
            labels[start_row, start_col] = current_label

            pixels = []
            min_row = start_row
            max_row = start_row
            min_col = start_col
            max_col = start_col
            row_sum = 0
            col_sum = 0

            while queue:
                row, col = queue.popleft()
                pixels.append((row, col))

                min_row = min(min_row, row)
                max_row = max(max_row, row)
                min_col = min(min_col, col)
                max_col = max(max_col, col)
                row_sum += row
                col_sum += col

                for row_delta, col_delta in offsets:
                    neighbor_row = row + row_delta
                    neighbor_col = col + col_delta

                    if neighbor_row < 0 or neighbor_row >= height:
                        continue
                    if neighbor_col < 0 or neighbor_col >= width:
                        continue
                    if binary_image[neighbor_row, neighbor_col] == 0:
                        continue
                    if labels[neighbor_row, neighbor_col] != 0:
                        continue

                    labels[neighbor_row, neighbor_col] = current_label
                    queue.append((neighbor_row, neighbor_col))

            area = len(pixels)
            components_info[current_label] = {
                "area": int(area),
                "bbox": (int(min_row), int(min_col), int(max_row), int(max_col)),
                "centroid": (
                    float(row_sum / area),
                    float(col_sum / area),
                ),
                "pixel_count": int(area),
                "pixels": pixels,
                "width": int(max_col - min_col + 1),
                "height": int(max_row - min_row + 1),
            }

    return labels, components_info


def binary_distance_transform(
    binary_image: np.ndarray,
    structuring_element: np.ndarray | None = None,
) -> np.ndarray:
    """Estima a distância ao fundo contando erosões sucessivas."""
    current = _as_binary_image(binary_image, "binary_image")

    if structuring_element is None:
        structuring_element = create_structuring_element("cross", 3)

    structuring_element = _validate_structuring_element(structuring_element)
    distances = np.zeros(current.shape, dtype=np.int32)
    level = 0

    while np.any(current):
        level += 1
        distances[current > 0] = level
        current = erode_binary(current, structuring_element)

    return distances


def watershed_markers_from_distance(
    binary_mask: np.ndarray,
    distance_map: np.ndarray,
    min_distance: int = 4,
    connectivity: int = 8,
) -> np.ndarray:
    """Cria marcadores em máximos locais da distância interna."""
    binary_mask = _as_binary_image(binary_mask, "binary_mask")
    distance_map = np.asarray(distance_map)
    connectivity = _validate_connectivity(connectivity)
    offsets = _neighbor_offsets(connectivity)

    if distance_map.shape != binary_mask.shape:
        raise ValueError("distance_map deve ter o mesmo formato de binary_mask.")

    if min_distance <= 0:
        raise ValueError("min_distance deve ser maior que zero.")

    height, width = binary_mask.shape
    maxima = np.zeros(binary_mask.shape, dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            value = int(distance_map[row, col])
            if binary_mask[row, col] == 0 or value < min_distance:
                continue

            is_local_maximum = True
            for row_delta, col_delta in offsets:
                neighbor_row = row + row_delta
                neighbor_col = col + col_delta

                if neighbor_row < 0 or neighbor_row >= height:
                    continue
                if neighbor_col < 0 or neighbor_col >= width:
                    continue
                if distance_map[neighbor_row, neighbor_col] > value:
                    is_local_maximum = False
                    break

            if is_local_maximum:
                maxima[row, col] = 1

    markers, _ = connected_components(maxima, connectivity=connectivity)
    return markers


def watershed_markers_from_erosion(
    binary_mask: np.ndarray,
    erosion_iterations: int = 8,
    connectivity: int = 8,
) -> np.ndarray:
    """Cria marcadores erodindo a máscara para separar regiões encostadas."""
    binary_mask = _as_binary_image(binary_mask, "binary_mask")

    if erosion_iterations <= 0:
        raise ValueError("erosion_iterations deve ser maior que zero.")

    current = binary_mask.copy()
    element = create_structuring_element("disk", 3)

    for _ in range(int(erosion_iterations)):
        current = erode_binary(current, element)

    markers, _ = connected_components(current, connectivity=connectivity)
    return markers


def marker_controlled_watershed(
    binary_mask: np.ndarray,
    markers: np.ndarray,
    distance_map: np.ndarray,
    connectivity: int = 8,
) -> np.ndarray:
    """Separa instâncias por inundação manual guiada pela distância interna."""
    binary_mask = _as_binary_image(binary_mask, "binary_mask")
    markers = np.asarray(markers, dtype=np.int32)
    distance_map = np.asarray(distance_map)
    connectivity = _validate_connectivity(connectivity)
    offsets = _neighbor_offsets(connectivity)

    if markers.shape != binary_mask.shape or distance_map.shape != binary_mask.shape:
        raise ValueError("binary_mask, markers e distance_map devem ter o mesmo formato.")

    output = markers.copy()
    queued = output > 0
    height, width = binary_mask.shape
    queue = []

    for row in range(height):
        for col in range(width):
            if output[row, col] > 0:
                heapq.heappush(queue, (-int(distance_map[row, col]), row, col))

    while queue:
        _, row, col = heapq.heappop(queue)
        current_label = int(output[row, col])

        for row_delta, col_delta in offsets:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta

            if neighbor_row < 0 or neighbor_row >= height:
                continue
            if neighbor_col < 0 or neighbor_col >= width:
                continue
            if binary_mask[neighbor_row, neighbor_col] == 0:
                continue

            neighbor_label = int(output[neighbor_row, neighbor_col])
            if neighbor_label == 0 and not queued[neighbor_row, neighbor_col]:
                queued[neighbor_row, neighbor_col] = True
                heapq.heappush(
                    queue,
                    (-int(distance_map[neighbor_row, neighbor_col]), neighbor_row, neighbor_col),
                )

        if output[row, col] != 0:
            continue

        neighbor_labels = set()
        for row_delta, col_delta in offsets:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta

            if neighbor_row < 0 or neighbor_row >= height:
                continue
            if neighbor_col < 0 or neighbor_col >= width:
                continue

            neighbor_label = int(output[neighbor_row, neighbor_col])
            if neighbor_label > 0:
                neighbor_labels.add(neighbor_label)

        if len(neighbor_labels) == 1:
            output[row, col] = next(iter(neighbor_labels))
        elif len(neighbor_labels) > 1:
            output[row, col] = -1

        if output[row, col] != 0:
            for row_delta, col_delta in offsets:
                neighbor_row = row + row_delta
                neighbor_col = col + col_delta

                if neighbor_row < 0 or neighbor_row >= height:
                    continue
                if neighbor_col < 0 or neighbor_col >= width:
                    continue
                if binary_mask[neighbor_row, neighbor_col] == 0:
                    continue
                if output[neighbor_row, neighbor_col] != 0:
                    continue
                if queued[neighbor_row, neighbor_col]:
                    continue

                queued[neighbor_row, neighbor_col] = True
                heapq.heappush(
                    queue,
                    (-int(distance_map[neighbor_row, neighbor_col]), neighbor_row, neighbor_col),
                )

    return output


def watershed_instances(
    binary_mask: np.ndarray,
    erosion_iterations: int = 8,
    connectivity: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executa watershed manual e retorna instâncias, marcadores e distâncias."""
    binary_mask = _as_binary_image(binary_mask, "binary_mask")
    distance_map = binary_distance_transform(binary_mask)
    markers = watershed_markers_from_erosion(
        binary_mask,
        erosion_iterations=erosion_iterations,
        connectivity=connectivity,
    )
    instances = marker_controlled_watershed(
        binary_mask,
        markers,
        distance_map,
        connectivity=connectivity,
    )
    return instances, markers, distance_map


def filter_components_by_area(
    binary_image: np.ndarray,
    min_area: int | None = None,
    max_area: int | None = None,
    connectivity: int = 8,
) -> np.ndarray:
    """Mantém apenas componentes cujo tamanho esteja no intervalo informado."""
    binary_image = _as_binary_image(binary_image, "binary_image")

    if min_area is not None and min_area < 0:
        raise ValueError("min_area deve ser não negativo ou None.")

    if max_area is not None and max_area < 0:
        raise ValueError("max_area deve ser não negativo ou None.")

    if min_area is not None and max_area is not None and max_area < min_area:
        raise ValueError("max_area deve ser maior ou igual a min_area.")

    labels, components_info = connected_components(binary_image, connectivity=connectivity)
    filtered_image = np.zeros(binary_image.shape, dtype=np.uint8)

    for label_id, info in components_info.items():
        area = info["area"]
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        filtered_image[labels == label_id] = 1

    return filtered_image


def remove_small_objects(
    binary_image: np.ndarray,
    min_area: int,
    connectivity: int = 8,
) -> np.ndarray:
    """Remove componentes com área menor que `min_area`."""
    if min_area < 0:
        raise ValueError("min_area deve ser não negativo.")

    return filter_components_by_area(
        binary_image,
        min_area=min_area,
        max_area=None,
        connectivity=connectivity,
    )


def _component_perimeter(
    binary_image: np.ndarray,
    pixels: list[tuple[int, int]],
) -> int:
    """Estima perímetro contando pixels de fronteira do componente."""
    height, width = binary_image.shape
    perimeter = 0

    for row, col in pixels:
        is_boundary = False

        for row_delta, col_delta in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta

            if neighbor_row < 0 or neighbor_row >= height:
                is_boundary = True
                break
            if neighbor_col < 0 or neighbor_col >= width:
                is_boundary = True
                break
            if binary_image[neighbor_row, neighbor_col] == 0:
                is_boundary = True
                break

        if is_boundary:
            perimeter += 1

    return perimeter


def filter_components_by_shape(
    binary_image: np.ndarray,
    min_area: int,
    max_area: int | None,
    min_circularity: float | None = None,
    connectivity: int = 8,
) -> np.ndarray:
    """Filtra componentes por área e circularidade aproximada.

    A circularidade é estimada por `4*pi*area/perimeter²`, usando a quantidade
    de pixels de fronteira como aproximação simples de perímetro.
    """
    binary_image = _as_binary_image(binary_image, "binary_image")

    if min_area < 0:
        raise ValueError("min_area deve ser não negativo.")

    if max_area is not None and max_area < min_area:
        raise ValueError("max_area deve ser maior ou igual a min_area.")

    if min_circularity is not None and min_circularity < 0:
        raise ValueError("min_circularity deve ser não negativa ou None.")

    labels, components_info = connected_components(binary_image, connectivity=connectivity)
    filtered_image = np.zeros(binary_image.shape, dtype=np.uint8)

    for label_id, info in components_info.items():
        area = info["area"]
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        if min_circularity is not None:
            perimeter = _component_perimeter(binary_image, info["pixels"])
            if perimeter == 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < min_circularity:
                continue

        filtered_image[labels == label_id] = 1

    return filtered_image


def morphological_reconstruction(
    marker: np.ndarray,
    mask: np.ndarray,
    structuring_element: np.ndarray,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Reconstrói uma máscara por dilatação geodésica binária.

    A cada iteração o marcador é dilatado manualmente e limitado pela máscara.
    O processo para quando não há mais alteração entre duas iterações ou quando
    `max_iterations` é atingido.
    """
    marker = _as_binary_image(marker, "marker")
    mask = _as_binary_image(mask, "mask")
    structuring_element = _validate_structuring_element(structuring_element)

    if marker.shape != mask.shape:
        raise ValueError("marker e mask devem ter a mesma forma.")

    if max_iterations <= 0:
        raise ValueError("max_iterations deve ser maior que zero.")

    current = (marker & mask).astype(np.uint8)

    for _ in range(int(max_iterations)):
        dilated = dilate_binary(current, structuring_element)
        reconstructed = (dilated & mask).astype(np.uint8)

        if np.array_equal(reconstructed, current):
            return reconstructed

        current = reconstructed

    return current
