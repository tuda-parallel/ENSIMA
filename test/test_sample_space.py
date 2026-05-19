"""
Tests for the SearchSpace class with various constraint configurations.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import logging

import numpy as np
import pytest

from ensima.classes.logger import Logger
from ensima.classes.search_space import SearchSpace
from ensima.helpers.parse_args import parse_arguments

LOGGER = logging.getLogger(__name__)


def test_progress() -> None:
    try:
        constraints = {
            "Rp": [270, 420, 890],  # discrete values
            "D": (0.1, 5.0),
        }
        LOGGER.info(f"Constrains are: {constraints}")
        search_space = SearchSpace(["Rp", "D", "p"], constraints)

        # Assume observed data:
        x = np.array([[270, 1.0, 0.05], [420, 2.0, 0.1], [890, 1.5, 0.07]])

        test_samples = search_space.create_sample_space(
            x=x,
            n_samples=1000,
            method="grid",
            expansion_factor=0.1,
        )
        LOGGER.info(f"Continuous samples are:\n {test_samples}")

        search_space.create_sample_space(
            x=x,
            n_samples=1000,
            method="random",
            expansion_factor=0.1,
        )
        LOGGER.info(f"Discrete samples are:\n {test_samples}")

    except Exception as e:
        LOGGER.exception("Exception during Sample space creation")
        pytest.fail(f"progress monitor failed: {e}")


if __name__ == "__main__":
    args = parse_arguments(
        [
            "--path",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_01_Flat",
            "-l",
            "DEBUG",
            "-j",
            "ASaeule",
            "-ofs",
            "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
        ]
    )
    # Init logger
    logger = Logger(__name__, level=args.log_level).get()

    x_fields = ["Rp", "D", "p"]
    user_constraints = {
        "Rp": [270, 420, 890],  # discrete values
        "D": (0.1, 5.0),
    }
    logger.info(f"Constrains are: {user_constraints}")
    x_space = SearchSpace(x_fields, user_constraints)

    # Assume observed data:
    x_observed = np.array([[270, 1.0, 0.05], [420, 2.0, 0.1], [890, 1.5, 0.07]])

    samples = x_space.create_sample_space(
        x=x_observed,
        n_samples=1000,
        method="grid",
        expansion_factor=0.1,
    )
    logger.info(f"Continuous samples are:\n {samples}")

    random_samples = x_space.create_sample_space(
        x=x_observed,
        n_samples=1000,
        method="random",
        expansion_factor=0.1,
    )
    logger.info(f"Discrete samples are:\n {random_samples}")
