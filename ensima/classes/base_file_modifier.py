"""
Abstract base class for file modifier implementations.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

from abc import ABC, abstractmethod


class BaseFileModifier(ABC):
    """
    Interface that every file-modifier backend must implement.

    Subclass this, implement ``set_design_parameters()``, and pass the
    subclass to ``main(file_modifier_class=MyFileModifier)`` or directly to
    ``BayesianOptimization(file_modifier_class=MyFileModifier)``.
    """

    def __init__(self, input_file: str, log_level: str = "info", prefix: str = ""):
        self.input_file = input_file

    @abstractmethod
    def set_design_parameters(self, parameters: dict) -> None:
        """
        Write the proposed parameter values into the simulator's input file.

        Args:
            parameters: mapping of field name → proposed value,
                        keys match ``--x_fields``.
        """

    def print(self) -> None:  # noqa: B027
        """Log or display current parameter values. Override as needed."""
