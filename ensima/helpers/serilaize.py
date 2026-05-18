"""
Converts objects to JSON-serializable formats for result storage and logging.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import numpy as np


def to_json_serializable(obj):
    """
    Convert an object into a JSON-serializable format.

    This function recursively converts elements in an object to a format suitable
    for JSON serialization. It handles NumPy arrays, lists, dictionaries, and
    specific NumPy numeric types, converting them to standard Python data types
    (e.g., lists, floats, ints).

    Parameters:
    obj
        The object to be converted to a JSON-serializable format. It can be of any
        type, but specific handling is done for NumPy arrays, lists, dictionaries,
        and NumPy numeric types.

    Returns:
    Any
        A JSON-serializable representation of the input object.

    Raises:
    TypeError
        If the input object includes elements that cannot be processed or
        converted into a JSON-serializable representation.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [to_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj
