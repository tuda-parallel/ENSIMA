"""
Extended Rich Console class with verbosity control for ENSIMA output.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

from rich.console import Console


class MyConsole(Console):
    """Console child class that overwrites
    the print method for silent version
    Args:
        Console (_type_): _description_
    """

    def __init__(self, verbose: bool = True, name: str = "optimization"):
        super().__init__()
        self.verbose = verbose
        self.name = name

    def set(self, flag):
        if flag:
            self.verbose = True
        else:
            self.verbose = False

    def print(self, *args, **kwargs):
        if self.verbose:
            super().print(*args, **kwargs)

    def info(self, s):
        Console.print(self, f"[cyan][{self.name}][/]{s}")
