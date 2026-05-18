"""
Formats numeric values with appropriate SI units for human-readable output.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import numpy as np


def format_with_units(value: float, unit_type: str) -> str:
    """
    Format a numeric value with an automatically chosen unit based on its magnitude.

    Supports milli- and micro-units for small values.

    Args:
        value: The numeric value to be formatted.
        unit_type: The base unit type, either "J" for energy or "g" for mass.

    Returns:
        A formatted string representing the value with the appropriate unit.

    Raises:
        ValueError: If the unit_type is not supported.
    """
    thresholds = {
        "J": [("MJ", 1e6), ("kJ", 1e3), ("J", 1), ("mJ", 1e-3), ("µJ", 1e-6)],
        "g": [("t", 1e6), ("kg", 1e3), ("g", 1), ("mg", 1e-3), ("µg", 1e-6)],
    }

    if unit_type not in thresholds:
        raise ValueError(f"Unsupported unit type: {unit_type}")

    abs_value = abs(value)  # Handle negative values gracefully

    for unit, factor in thresholds[unit_type]:
        if abs_value >= factor:
            return f"{value / factor:.3f} {unit}"
    # If value is extremely small
    return f"{value:.3e} {unit_type}"


def convert_seconds_to_hms(seconds_array):
    """
    Converts a NumPy array of time values in seconds to a formatted
    hh:mm:ss string
    """
    # Calculate hours, minutes, and seconds using NumPy's vectorized operations
    hours = np.floor_divide(seconds_array, 3600).astype(int)
    minutes = np.floor_divide(np.remainder(seconds_array, 3600), 60).astype(int)
    seconds = np.remainder(seconds_array, 60).astype(int)
    out = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return out


def get_unit(value: float, unit_type: str) -> tuple[float, str]:
    """
    Format a numeric value with an automatically chosen unit based on its magnitude.

    Supports milli- and micro-units for small values.

    Args:
        value: The numeric value to be formatted.
        unit_type: The base unit type, either "J" for energy or "g" for mass.

    Returns:
        factor,unit

    Raises:
        ValueError: If the unit_type is not supported.
    """
    thresholds = {
        "J": [("MJ", 1e6), ("kJ", 1e3), ("J", 1), ("mJ", 1e-3), ("µJ", 1e-6)],
        "g": [("t", 1e6), ("kg", 1e3), ("g", 1), ("mg", 1e-3), ("µg", 1e-6)],
    }

    if unit_type not in thresholds:
        raise ValueError(f"Unsupported unit type: {unit_type}")

    abs_value = abs(value)  # Handle negative values gracefully

    for unit, factor in thresholds[unit_type]:
        if abs_value >= factor:
            return factor, unit

    # If value is extremely small
    return (
        1,
        unit_type,
    )
