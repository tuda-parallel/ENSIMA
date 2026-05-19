"""
Functions for reading and parsing simulation result CSV files into NumPy arrays.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import os
import re

import numpy as np
import pandas as pd

from ensima.classes.logger import Logger


def read_simple_data(input: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads input data from a CSV file and returns two numpy arrays:
    - x: A 2D array with the input parameters (thickness, pressure, friction, thinning,
    thickening).
    - y: A 2D array with the output results (feasibility, compute time,
    energy consumption, etc.).

    Args:
        input (str): The path to the CSV file containing the data.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): A 2D numpy array of input parameters.
            - y (numpy.ndarray): A 2D numpy array of output results.

    Example:
        x, y = read_data("csv/CylindricalCup.csv")

        the csv file contains data like:
        0.9,1,0.1,-23.2,43.6,0,30.2,27.8,42,0,0,0,0,0,0,0.0064
        1,3,0.1,-32.2,27.2,0,26.2,25.9,45.2,0.8,1.9,0,0,0,0,0.594
        :
    """

    # Read CSV file
    df = pd.read_csv(
        filepath_or_buffer=input,
        names=[
            "thickness",
            "pressure",
            "friction",
            "thinning",
            "thickening",
            "feasibility_1",
            "feasibility_2",
            "feasibility_3",
            "feasibility_4",
            "feasibility_5",
            "feasibility_6",
            "feasibility_7",
            "compute_time",
            "energy_consumption",
            "spring_coefficient",
            "rand",
        ],
    )

    # Separate x (parameters) from y (results)
    x = df[
        ["thickness", "pressure", "friction", "thinning", "thickening"]
    ].to_numpy()  # Parameters (5 columns)
    y = df.drop(
        columns=["thickness", "pressure", "friction", "thinning", "thickening"]
    ).to_numpy()  # Rest of the columns for y

    return x, y


def read_data(
    input_file: str,
    x_fields: list[str] | None = None,
    y_fields: list[str] | None = None,
    log_level: str = "INFO",
    lines_to_read=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads data from a CSV file containing simulation or experimental results,
    extracting specified input and output fields.

    The function returns two 2D numpy arrays:
      - `x`: Input parameter values such as thickness, pressure, friction, thinning,
      and thickening.
      - `y`: Output result values such as feasibility metrics or other target values.

    It automatically parses the CSV header to locate columns matching the given input (
    `x_fields`) and output (`y_fields`) names, then extracts and converts the
    corresponding data rows to float arrays. Missing or non-numeric values are
    converted to NaN.

    Args:
        input_file (str): Path to the CSV file containing the data.
        x_fields (list[str], optional): List of column names to use as input
        parameters. Defaults to ["Rp", "D", "p", "Fr", "db"].
        y_fields (list[str], optional): List of column names to use as output targets.
        Defaults to ["L1", "L2", "L3", "L4", "L5", "L6"].
        lines_to_read (int, optional): Number of lines to read from the CSV

    Returns:
        tuple:
            - x (numpy.ndarray): 2D array of input data with shape (num_samples,
            num_x_fields).
            - y (numpy.ndarray): 2D array of output data with shape (num_samples,
            num_y_fields).

    Example:
        x_fields = ["Fr", "p"]
        y_fields = ["L1", "L2", "L3", "L4", "L5", "L6"]
        x, y = read_data("/d/github/ENSIMA/test/csv/DataSets-AIandML_20250401.csv", x_fields, y_fields, log_level="INFO")
        print(x.shape, y.shape)
        (1654, 2) (1654, 6)

    Notes:
        - The CSV file is expected to contain headers that include the column names.
        - Time values in columns (e.g., CPU time) are automatically converted from
        strings (HH:MM:SS) to seconds.
        - Non-numeric or missing values are converted to numpy.nan for consistency.
    """
    if x_fields is None:
        x_fields = ["Rp", "D", "p", "Fr", "db"]
    if y_fields is None:
        y_fields = ["L1", "L2", "L3", "L4", "L5", "L6"]
    # Read the content of the file
    x_columns, y_columns, ignored_fields, data_array, _ = parse(
        input_file, x_fields, y_fields, log_level, lines_to_read
    )

    # Extract x and y
    x = data_array[:, x_columns]
    y = data_array[:, y_columns]
    check(log_level, x, y, x_fields, y_fields, ignored_fields, input_file)
    return x, y


def read_data_type(
    input_file: str,
    x_fields: list[str] | None = None,
    y_fields: list[str] | None = None,
    log_level: str = "INFO",
    lines_to_read=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_fields is None:
        x_fields = ["Rp", "D", "p", "Fr", "db"]
    if y_fields is None:
        y_fields = ["L1", "L2", "L3", "L4", "L5", "L6"]

    x_columns, y_columns, ignored_fields, data_array, type = parse(
        input_file, x_fields, y_fields, log_level, lines_to_read
    )

    # Extract x and y
    x = data_array[:, x_columns]
    y = data_array[:, y_columns]
    check(log_level, x, y, x_fields, y_fields, ignored_fields, input_file)
    return x, y, np.array(type)


def parse(
    input_file: str,
    x_fields: list[str] | None = None,
    y_fields: list[str] | None = None,
    log_level: str = "INFO",
    lines_to_read=None,
) -> tuple[list[int], list[int], list[str], np.ndarray, int]:
    """
    Parses a structured CSV-like file with a custom header and START/END markers.

    Extracts the numerical data section between 'START' and 'END', identifies the
    column indices of specified input (x) and output (y) fields, and returns all
    relevant parsing information including the data array and type identifier.

    The file is expected to contain a description of column names using either a
    `$Description:` or `$ Row Format:` line, followed by a data block enclosed by
    START and END lines.

    Args:
        input_file (str): Path to the structured data file.
        x_fields (list[str], optional): Names of columns to be used as input features.
                                         Defaults to ["Rp", "D", "p", "Fr", "db"].
        y_fields (list[str], optional): Names of columns to be used as target outputs.
                                         Defaults to ["L1", "L2", "L3", "L4", "L5", "L6"].
        log_level (str, optional): Logging verbosity level. Defaults to "INFO".
        lines_to_read (int, optional): If provided, only the last N lines in the data
                                       block are parsed.

    Returns:
        tuple:
            - x_columns (list[int]): Indices of input fields in the data array.
            - y_columns (list[int]): Indices of output fields in the data array.
            - ignored_fields (list[str]): Column names that are not part of x or y.
            - data_array (np.ndarray): 2D array of all parsed numeric data.
            - type (int): The "Type" identifier extracted from a header line (e.g., 'Type: 1').

    Example:
        # >>> x_cols, y_cols, ignored, data, type_id = parse("data.csv", ["Fr", "p"], ["L1", "L2"])
        # >>> x = data[:, x_cols]
        # >>> y = data[:, y_cols]
        # >>> print(x.shape, y.shape, type_id)

    Notes:
        - Missing or malformed values are converted to `np.nan`.
        - Time fields (e.g., CPU time in HH:MM:SS format) are automatically converted to seconds.
        - The 'Type' is parsed from lines like:
          "$---- Type: 1, A/B = 1.2, h/B = 0.20,  Complexity: 18 % ----"
    """
    if x_fields is None:
        x_fields = ["Rp", "D", "p", "Fr", "db"]
    if y_fields is None:
        y_fields = ["L1", "L2", "L3", "L4", "L5", "L6"]
    # Read the content of the file
    logger = Logger(__name__, level=log_level).get()
    logger.info(f"Opened {input_file} for reding")
    with open(input_file) as file:
        lines = file.readlines()

    # Find the line containing the description of the columns
    start_index = None
    end_index = None
    column_names = []
    type_index = None
    logger.debug(f"Extracting description line from from {input_file}")
    for i, line in enumerate(lines):
        if "$Description:" in line:
            description_line = line.strip()
            # Extract column names from the description line, split by commas
            column_names = [
                val for val in description_line.split(":")[1].strip().split(",") if val
            ]
        if line.startswith("$ Row Format"):
            raw_fields = lines[i + 1][1:].split(",")
            column_names = [f.strip() for f in raw_fields if f.strip()]
        elif "START" in line:
            start_index = i + 1  # The data starts right after START
        elif "END" in line:
            end_index = i  # The data ends right before END

    if not column_names:
        logger.error("No column names found in input file")
        raise ValueError(
            "Description of entries not found. Make sure you have at least on "
            "description lne like:\n$ Row Format:\n$ Nr , Rp  , D  ,  p , Fr , db,  "
            "dD , dT , L1 , L2 , L3 , L4 , L5 , L6 , L7 , SB ,  CPU   , "
        )
    else:
        if log_level.upper() == "DEBUG":
            logger.debug("Column Names and Field Classification:")
            for idx, col in enumerate(column_names):
                if col in x_fields:
                    pos = x_fields.index(col)
                    logger.debug(f"Column {idx}: '{col}' → X field position {pos}")
                elif col in y_fields:
                    pos = y_fields.index(col)
                    logger.debug(f"Column {idx}: '{col}' → Y field position {pos}")
                else:
                    logger.debug(f"Column {idx}: '{col}' → Ignored")

    # Get indices for x (D, p, Fr, dD, dT) and y (L1, L2, L3, L4, L5, L6, L7)
    x_columns = [column_names.index(name) for name in x_fields]
    y_columns = [column_names.index(name) for name in y_fields]
    ignored_fields = [
        name for name in column_names if name not in x_fields and name not in y_fields
    ]
    if "type" in column_names:
        type_index = column_names.index("type")
        logger.debug(f"Type column found at index {type_index}")
    else:
        logger.debug("Type column not found")
        type_index = None

    not_allowed_nan_columns = sorted(set(x_columns) | set(y_columns))
    if lines_to_read is not None:
        start_index = end_index - lines_to_read
        logger.info(
            f"Reading only last {lines_to_read} lines ([{start_index},{end_index}]) from {input_file}"
        )
    data_array, type = process_lines(
        lines,
        start_index,
        end_index,
        column_names,
        not_allowed_nan_columns,
        log_level,
        type_index,
    )
    return x_columns, y_columns, ignored_fields, data_array, type


def process_lines(
    lines,
    start_index,
    end_index,
    column_names,
    not_allowed_nan_columns,
    log_level,
    type_index=None,
):

    logger = Logger(__name__, level=log_level).get()
    n_cols = len(column_names)
    data = []
    type_list = []
    part_type = 0

    for line in lines[start_index:end_index]:
        if line.startswith("$"):
            if "Type:" in line and type_index is None:
                match = re.search(r"Type:\s*(\d+)", line)
                if match:
                    part_type = int(match.group(1))
                    logger.debug(f"Type number: {part_type} from {lines.index(line)}")
        else:
            # values = [val.strip() for val in line.strip().split(",") if val.strip()]
            values = [
                val.strip() for val in re.split(r"[ ,]+", line.strip()) if val.strip()
            ]

            if not values:
                continue

            # Pad with np.nan if values fewer than columns
            if len(values) < n_cols:
                values += [np.nan] * (n_cols - len(values))
            elif len(values) > n_cols:
                values = values[:n_cols]  # truncate if too long

            converted_values = []
            for v in values:
                if v is np.nan:
                    converted_values.append(np.nan)
                    continue
                converted_values.append(convert(v))
            # if NaN is in the columns where no NaN is allowed, skip these
            if any(np.isnan(converted_values[idx]) for idx in not_allowed_nan_columns):
                logger.warning(f"NaN found in {values}. Skipping this line")
                continue
            if type_index is not None:
                part_type = str(values[type_index])
            data.append(converted_values)
            type_list.append(part_type)

    data_array = np.array(data, dtype=float)
    return data_array, type_list


def convert(value):
    # If value is str and contains '%', remove it first
    if isinstance(value, str) and "%" in value:
        value = value.replace("%", "")

    # Try float conversion
    try:
        return float(value)
    except ValueError:
        # Handle time string
        if isinstance(value, str) and ":" in value:
            return time_str_to_seconds(value)
        # Handle question mark as zero
        if isinstance(value, str) and "?" in value:
            logger = Logger(__name__, level="INFO").get()
            logger.warning("'?' found in CSV file. Treating as zero ")
            return 0
        # Not recognized, return NaN
        return np.nan


def time_str_to_seconds(t):
    """Convert 'HH:MM:SS' or 'MM:SS' string to total seconds."""
    try:
        parts = t.split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            return np.nan
    except Exception:
        return np.nan


def read_last_progress(log_file_path) -> tuple[int, float]:
    """
    Reads the last line in the log file containing progress information.

    Returns:
        A tuple (step_number, progress_percent), or (-1, -1.0) if not found.
    """
    progress_pattern = re.compile(r"STEP\s*:\s*(\d+).*?\(\s*([\d.]+)% DONE\)")

    try:
        with open(log_file_path) as file:
            lines = file.readlines()
    except FileNotFoundError:
        return -1, -1.0

    for line in reversed(lines):
        match = progress_pattern.search(line)
        if match:
            step = int(match.group(1))
            progress = float(match.group(2))
            return step, progress

    return -1, -1.0


def check(log_level, x, y, x_fields, y_fields, ignored_fields, input_file):
    logger = Logger(__name__, level=log_level).get()
    # Check that there are no nans
    nan_x = np.argwhere(np.isnan(x))
    nan_y = np.argwhere(np.isnan(y))
    if len(nan_x) > 0 or len(nan_y) > 0:
        logger.error("NaNs found in x or y")
    if len(nan_x) > 0:
        logger.warning(f"NaNs found in x at indices: {nan_x[: min(len(nan_x), 10)]}")
    if len(nan_y) > 0:
        logger.warning(f"NaNs found in y at indices: {nan_x[: min(len(nan_y), 10)]}")

    logger.debug(f"Closed {os.path.basename(input_file)}. Reading done")
    logger.debug(f"x fields: {x_fields}, x shape: {x.shape}")
    logger.debug(f"y fields: {y_fields}, y shape: {y.shape}")
    logger.debug(f"ignored fields: {ignored_fields}")


if __name__ == "__main__":
    logger = Logger(__name__, level="DEBUG").get()
    # x, y = read_data("/d/github/ENSIMA/test/csv/DataSets.csv")

    x_fields = ["Rp", "Fr", "p"]
    y_fields = ["L1", "L2", "L3", "L4", "L5", "L6"]
    x, y = read_data(
        "/d/github/ENSIMA/test/csv/DataSets-AIandML_20250401.csv",
        x_fields,
        y_fields,
        log_level="INFO",
    )
    logger.info(f"X={x}")
    logger.info(f"Y={y}")
    logger.info(f"Rp = {np.unique(x[:,0])}")

    x, y, type = read_data_type(
        "/d/github/ENSIMA/test/csv/DataSets-AIandML_labeled.csv",
        x_fields,
        y_fields,
        log_level="INFO",
    )
    logger.info(f"X={x}")
    logger.info(f"Y={y}")
    logger.info(f"Rp = {np.unique(type)}")
