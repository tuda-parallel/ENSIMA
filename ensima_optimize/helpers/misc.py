"""
General-purpose file system utilities used across the ENSIMA optimization pipeline.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import glob
import os

from ensima_optimize.classes.logger import Logger


def safe_remove(signal_file_path: str):
    """
    Safely removes a file from the filesystem, handling potential errors.

    Args:
        signal_file_path (str): Path to the file that needs to be removed

    Returns:
        None

    Note:
        - Silently ignores if file does not exist
        - Logs other errors that might occur during removal
    """
    try:
        for file_path in glob.glob(signal_file_path):
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger = Logger(__name__, "INFO").get()
                logger.error(f"An error occurred removing {file_path}: {e}")
    except Exception as e:
        logger = Logger(__name__, "INFO").get()
        logger.error(f"Glob failed for {signal_file_path}: {e}")


def delete_matching(pattern: str, logger=None):
    """
    Delete files and folders matching a full path or wildcard pattern.

    Args:
        pattern (str): Full path or wildcard, e.g., "/path/to/result_*"
        logger: Optional logger for info/warnings.
    """
    for p in glob.glob(pattern):
        if os.path.isfile(p):
            os.remove(p)
            if logger:
                logger.info(f"Deleted file: {p}")
        elif os.path.isdir(p):
            # Delete all files inside
            for file in os.listdir(p):
                file_path = os.path.join(p, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    if logger:
                        logger.info(f"Deleted file: {file_path}")
            # Try removing the folder (only works if empty)
            try:
                os.rmdir(p)
                if logger:
                    logger.info(f"Deleted empty folder: {p}")
            except OSError:
                if logger:
                    logger.warning(f"Folder not empty, skipped: {p}")
