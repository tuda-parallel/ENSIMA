"""
Reads and preprocesses 3D point cloud coordinates from stamping geometry files.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import numpy as np

from ensima.classes.logger import Logger


def read_coordinates_from_file(filename, log_level="INFO", prefix="ReadCoords"):
    logger = Logger(__name__, level=log_level, prefix=prefix).get()
    coordinates = []
    in_geometry = False
    in_begin = False

    with open(filename) as file:
        for line in file:
            stripped = line.strip()

            # sometimes strange spaces appear between these strings, better check it this way
            if "BEGINN" in stripped.upper() and (
                "STEMPEL" in stripped.upper()
                or "BINDER" in stripped.upper()
                or "PART" in stripped.upper()
            ):
                in_begin = True
                logger.debug("BEGINN STEMPEL/BINDER found.")
                continue

            if not in_begin:
                continue

            if stripped == "KNOTENGEOMETRIE":
                in_geometry = True
                logger.debug("KNOTENGEOMETRIE section started.")
                continue
            elif stripped.startswith("LISTENGESTEUERT"):
                continue
            elif stripped.upper() in {"ENDE STEMPEL", "INZIDENZTAFEL", "ENDE BINDER"}:
                logger.debug(f"Terminating at: {stripped}")
                break
            elif in_geometry:
                if stripped == "":
                    continue
                parts = stripped.split(",")
                if len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        coordinates.append([x, y, z])
                    except ValueError as e:
                        logger.warning(
                            f"Skipping malformed line: '{line.strip()}'. Reason: {e}"
                        )
                        continue

    coords_array = np.array(coordinates)
    logger.info(f"Parsed {len(coordinates)} coordinates from {filename}")
    return coords_array


def compute_bounding_box_dimensions(coords):
    """
    Compute dimensions (a, b, h) of an axis-aligned bounding box from 3D coordinates.

    Args:
        coords (np.ndarray): Array of shape (N, 3) containing x, y, z coordinates.

    Returns:
        a, b, h (float): Dimensions of the bounding box, rounded to two decimal places.

    Raises:
        ValueError: If the input array is not 2D with 3 columns or is empty.
    """
    import numpy as np

    if not isinstance(coords, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("Coordinate array is empty.")

    x_min, y_min, z_min = np.min(coords, axis=0)
    x_max, y_max, z_max = np.max(coords, axis=0)

    a = round(x_max - x_min, 2)  # length along x
    b = round(y_max - y_min, 2)  # length along y
    h = round(z_max - z_min, 2)  # height (z)

    return a, b, h


def extract_points_for_parts(
    parts, log_level="INFO", prefix="ReadCoords"
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Extracts 3D coordinate points from files for a list of parts.

    This function reads coordinate data from specified file paths for each part
    and logs the process. It extracts the x, y, and z components of the coordinates
    and stores them in a list along with the part name.

    Args:
        parts (list[tuple[str, str]]): A list of tuples where each tuple contains:
            - part (str): The name of the part.
            - path (str): Path to the file containing coordinate data.
        log_level (str, optional): Logging level. Defaults to "INFO".
        prefix (str, optional): Prefix for log messages. Defaults to "ReadCoords".

    Returns:
        list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, str]]:
        A list of tuples containing x, y, z coordinate arrays and the part name
        for each part that had coordinate data.
    """
    logger = Logger(__name__, level=log_level, prefix=prefix).get()
    points = []
    for part, path in parts:
        logger.info(f"Extracting points for {part}")
        coords = read_coordinates_from_file(path, log_level=log_level)
        if len(coords) > 0:
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            points.append((x, y, z, part))
        else:
            logger.warning(f"No coordinates found for {part}")

    return points


def uniform_vector(x, y, z, n=1024, mode="upsample"):
    """
    Adjust a point cloud to have exactly n points.

    Args:
        x, y, z: Arrays of coordinates.
        n: Target number of points.
        mode: How to adjust points:
            - "upsample": repeat points if len < n
            - "downsample": randomly sample points if len > n

    Returns:
        pc: (n, 3) array of points.
    """
    vector = np.vstack([x, y, z]).T
    num_points = len(vector)

    if mode == "upsample":
        if num_points < n:
            # repeat deterministically
            repeats = n // num_points
            remainder = n % num_points
            pc = np.vstack([vector] * repeats + [vector[:remainder]])
        else:
            # downsample randomly to exactly n
            idx = np.random.choice(num_points, n, replace=False)
            pc = vector[idx]

    elif mode == "downsample":
        if num_points > n:
            # downsample randomly to exactly n
            idx = np.random.choice(num_points, n, replace=False)
            pc = vector[idx]
        else:
            # upsample randomly with replacement to exactly n
            idx = np.random.choice(num_points, n, replace=True)
            pc = vector[idx]
    else:
        raise ValueError(f"Invalid mode: {mode}, choose 'upsample' or 'downsample'")

    return pc


def preprocess_point_cloud(pc: np.ndarray, rotate: bool = False) -> np.ndarray:
    """
    Preprocess a point cloud by centering and optionally applying a random rotation.

    Args:
        pc: (N, 3) array of points.
        rotate: If True, apply a random rotation around the Z-axis.

    Returns:
        pc: Preprocessed (N, 3) point cloud array.
    """
    # -------------------------------
    # Centering to remove translation offsets
    # -------------------------------
    pc = pc - pc.mean(axis=0)

    # -------------------------------
    # Optional random rotation for rotation invariance
    # -------------------------------
    if rotate:
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        pc = pc @ rotation_matrix.T

    return pc


if __name__ == "__main__":
    path = "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_04/Einleger.t52"
    coords = read_coordinates_from_file(path, log_level="DEBUG")
    a, b, h = compute_bounding_box_dimensions(coords)
    logger = Logger(__name__).get()
    logger.debug(f"Dimensions: a={a}, b={b}, h={h}")
