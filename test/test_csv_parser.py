"""
Tests the CSV parser binary location and build process.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import logging

from ensima.helpers.parse_results import csv_parser_path

LOGGER = logging.getLogger(__name__)


def test_progress() -> None:
    parser_path = csv_parser_path("DEBUG")
    print(parser_path)
