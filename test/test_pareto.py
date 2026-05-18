"""
Tests for Pareto frontier computation.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import numpy as np

from ensima.helpers.pareto import pareto_frontier


def test_pareto_optimization():
    # Two objectives: we want to minimize both
    points = np.array(
        [
            [1, 5],
            [2, 3],
            [3, 1],
            [4, 4],
            [5, 2],
        ]
    )

    pareto_indices = pareto_frontier(points, mode="minimization")
    print("Pareto indices:", pareto_indices)
    print("Pareto points:\n", points[pareto_indices])
    # plot_pareto(points, pareto_indices)


if __name__ == "__main__":
    test_pareto_optimization()
