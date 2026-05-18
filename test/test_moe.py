"""
Tests for the MoEPointNetSystem and expert model integration.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import logging
import warnings
from argparse import Namespace

# Use Agg backend to avoid hangs in environments without a display
import matplotlib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")

from ensima_optimize.classes.mixture_of_experts import MoEPointNetSystem

warnings.filterwarnings("ignore", category=ConvergenceWarning)
LOGGER = logging.getLogger(__name__)


def test_moe() -> None:
    """
    Fast test for Mixture of Experts (MoE) system.
    """
    # Generate some random parts geometry with labels
    geometry = []
    labels = ["part_A", "part_B"]
    n_points_per_part = 1024  # Must be 1024 for PointNetAutoEncoder as it is hardcoded
    n_test_points = 1024

    scale = 1.0

    for label in labels:
        geo_x = np.random.rand(n_points_per_part) * scale
        geo_y = np.random.rand(n_points_per_part) * scale
        geo_z = np.random.rand(n_points_per_part) * scale
        geometry.append((geo_x, geo_y, geo_z, label))

    test_geo_x = np.random.rand(n_test_points) * scale
    test_geo_y = np.random.rand(n_test_points) * scale
    test_geo_z = np.random.rand(n_test_points) * scale
    unknown_geometry = (test_geo_x, test_geo_y, test_geo_z)
    new_x = np.random.rand(5, 1)

    functions = {
        "part_A": lambda x: np.sin(x),
        "part_B": lambda x: np.cos(x),
    }

    train_points = {}
    model_settings = {}
    for label in labels:
        x = np.random.rand(10, 1)
        y = functions[label](x)
        train_points[label] = (x, y)
        model_settings[label] = {
            "log_level": "INFO",
            "normalize_y": False,
            "optimizer_restart": 1,
            "epochs": 1,
            "save_and_load": False,
        }

    args = Namespace(
        jobname="test_part",
        new_expert_start=100,
        sample_mode="upsample",
        x_space_precision=2,
        latent_input_dim=None,
        latent_output_dim=None,
        save_and_load=False,
        epochs=1,
    )

    # ---- supervised Mode ----
    LOGGER.info("Starting Supervised Mode test")
    system_sup = MoEPointNetSystem(
        mode="supervised",
        model_settings=model_settings,
        train_points=train_points,
        geo_points=geometry,
        n_points=1024,
        log_level="DEBUG",
        args=args,
    )
    system_sup.predict_with_moe(new_x, unknown_geometry, soft=False)
    system_sup.predict_with_moe(new_x, unknown_geometry, soft=True)

    # ---- Unsupervised Mode ----
    LOGGER.info("Starting Unsupervised Mode test")
    system_unsup = MoEPointNetSystem(
        mode="unsupervised",
        model_settings=model_settings,
        train_points=train_points,
        geo_points=geometry,
        n_points=1024,
        log_level="DEBUG",
        args=args,
    )
    system_unsup.predict_with_moe(new_x, unknown_geometry, soft=False)
    system_unsup.predict_with_moe(new_x, unknown_geometry, soft=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_moe()
    print("All tests passed")
