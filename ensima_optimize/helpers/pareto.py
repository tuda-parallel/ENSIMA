"""
Computes and visualizes Pareto frontiers for multi-objective optimization results.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# Try to import Numba. If it fails, the numba_available flag remains False.
numba_available = False
try:
    import numba

    numba_available = True
except ImportError:
    pass


# The core Pareto frontier logic
def _pareto_frontier(points: np.ndarray, mode: str = "minimization") -> np.ndarray:
    """Core logic for finding the Pareto frontier."""
    if mode == "maximization":
        points = -points

    n_points = points.shape[0]
    is_on_frontier = np.ones(n_points, dtype=np.bool_)

    for i in range(n_points):
        if not is_on_frontier[i]:
            continue

        for j in range(n_points):
            if i == j:
                continue

            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_on_frontier[i] = False
                break

    return np.nonzero(is_on_frontier)[0]


# Conditionally apply Numba's JIT decorator
if numba_available:
    # Overwrite the function with a JIT-compiled version
    _pareto_frontier = numba.jit(_pareto_frontier, nopython=True)


# The public-facing function
def pareto_frontier(points: np.ndarray, mode: str = "minimization") -> np.ndarray:
    """
    Computes the Pareto frontier, using Numba for acceleration if available.

    Args:
        points (np.ndarray): Array of shape (n_points, n_objectives).
        mode (str): "minimization" (default) or "maximization".

    Returns:
        np.ndarray: Indices of points on the Pareto frontier.
    """
    if mode not in ["minimization", "maximization"]:
        raise ValueError("Mode must be 'minimization' or 'maximization'.")

    # Call the core function, which is either JIT-compiled or pure Python
    return _pareto_frontier(points, mode)


def plot_pareto(points: np.ndarray, pareto_indices: np.ndarray, save_path: str = None):
    """
    Plots all 2D projections of a multi-objective dataset,
    highlighting the Pareto front in each subplot.

    Args:
        points (np.ndarray): Array of shape (n_points, n_objectives).
        pareto_indices (np.ndarray, optional): Indices of Pareto-optimal points.
        save_path (str, optional): Path to save the plot. If None, plot is not saved.
    """
    n_obj = points.shape[1]
    n_plots = n_obj * (n_obj - 1) // 2
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    plot_idx = 0
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            ax = axes[plot_idx]
            ax.scatter(points[:, i], points[:, j], label="All Points", alpha=0.4)
            if pareto_indices is not None:
                ax.scatter(
                    points[pareto_indices, i],
                    points[pareto_indices, j],
                    color="red",
                    label="Pareto Front",
                    edgecolor="k",
                )
            ax.set_xlabel(f"Objective {i}")
            ax.set_ylabel(f"Objective {j}")
            ax.legend()
            ax.set_title(f"Obj {i} vs Obj {j}")
            plot_idx += 1

    # Turn off any unused subplots
    for k in range(plot_idx, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ensure folder exists
        plt.savefig(f"{save_path}/pareto.png", bbox_inches="tight")
        with open(
            f"{save_path}/pareto.pkl",
            "wb",
        ) as f:
            pickle.dump(fig, f)
    else:
        plt.show()
