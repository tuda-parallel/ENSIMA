"""
Utilities to locate and build the CSV parser binary for processing simulation results.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import os
import subprocess

from ensima_optimize.classes.execute import execute_block
from ensima_optimize.classes.logger import Logger


def csv_parser_path(log_level: str = "WARNING") -> str:
    """
    Ensures that the 'csv_parser' binary exists in the 'tools/' directory.

    This function checks if the 'csv_parser' binary is present at its expected location
    within the project structure. If the binary is missing, it attempts to run 'make' in
    the 'tools/' directory to build it. The operation is logged based on the specified log level.

    Args:
        log_level (str): Logging verbosity level (e.g., 'DEBUG', 'INFO', 'WARNING').
                         Defaults to 'WARNING'.

    Returns:
        str: Absolute path to the 'csv_parser' binary.

    Raises:
        RuntimeError: If the build process fails or the binary is still not found after attempting to build it.
    """
    logger = Logger(__name__, level=log_level).get()

    parser_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "tools", "csv_parser")
    )

    if not os.path.isfile(parser_path):
        logger.warning(
            f"'csv_parser' not found at {parser_path}. Attempting to build it with 'make'..."
        )
        try:
            out, _ = execute_block(
                f"cd {os.path.dirname(parser_path)} && make",
                raise_exception=True,
                options={"check": True},
                log_level=log_level,
            )
            logger.debug(f"Executed: {out}")
            if os.path.isfile(parser_path):
                logger.info("'csv_parser' successfully built.")
            else:
                logger.error("'make' completed but 'csv_parser' is still missing.")
                raise RuntimeError("'csv_parser' is still missing after 'make'.")
        except subprocess.CalledProcessError as e:
            logger.exception(f"'make' failed in 'tools/' directory: {e}")
            raise RuntimeError(f"Make failed with error: {e}") from e
    else:
        logger.debug(f"'csv_parser' already exists at {parser_path}.")

    return parser_path


if __name__ == "__main__":
    parser_path = csv_parser_path("DEBUG")
    print(parser_path)
