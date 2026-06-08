"""
Abstract base class for simulation implementations.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

from abc import ABC, abstractmethod
from argparse import Namespace

import numpy as np


class BaseSimulation(ABC):
    """
    Interface that every simulation backend must implement.

    Subclass this, implement ``run()``, and pass the subclass to
    ``main(simulation_class=MySimulation)`` or directly to
    ``BayesianOptimization(simulation_class=MySimulation)``.
    """

    def __init__(
        self,
        args: Namespace,
        next_sample: np.ndarray,
        iteration: int = 0,
        total_iterations: int = 0,
        prefix=None,
    ):
        self.args = args
        self.next_sample = next_sample
        self.iteration = iteration
        self.total_iterations = total_iterations
        self.prefix = prefix

    @abstractmethod
    def run(self, file_modifier, lock, type_filter: bool = False) -> np.ndarray:
        """
        Run one simulation and return the outputs.

        Args:
            file_modifier: ``BaseFileModifier`` instance (may be ``None``).
            lock:          multiprocessing lock from the optimizer.
            type_filter:   whether to tag the result in the CSV by job name.

        Returns:
            np.ndarray of shape ``(1, n_outputs)``, columns matching
            ``--y_fields`` in order.
        """
