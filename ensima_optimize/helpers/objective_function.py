"""
Dummy objective functions for testing and demonstrating the Bayesian optimization loop.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import numpy as np


def objective_function(x: np.ndarray) -> np.ndarray:
    """Dummy objective function for testing Bayesian optimization.

    Args:
        x (np.ndarray): input array of shape (n_samples, n_dimensions)

    Returns:
        np.ndarray: y values of shape (n_samples, n_outputs) according to the specified function
    """
    result = np.sin(x).sum(axis=1, keepdims=True) + np.cos(2 * x).sum(
        axis=1, keepdims=True
    )
    # result = (
    #     np.sin(x).sum(axis=1, keepdims=True)
    #     + np.cos(2 * x).sum(axis=1, keepdims=True)
    #     + np.sin(-1 * x**2).sum(axis=1, keepdims=True)
    # )
    # return result #
    return np.hstack([result, -result * result])


# def objective_function(x):
#     # Let's assume x is a 2D array of shape (n_samples, n_dimensions)
#     return np.sin(x).sum(axis=1, keepdims=True) + np.cos(2 * x).sum(axis=1, keepdims=True)


def dummy_init(
    dims: int = 4, samples: int = 30
) -> tuple[np.ndarray, np.ndarray, callable]:
    """Initializes the input data for the optimization problem.

    Args:
        dims (int, optional): dimensions. Defaults to 4.
        samples (int, optional): number of initial samples. Defaults to 30.

    Returns:
        tuple[np.ndarray, np.ndarray, callable]: x,y, and objective_function
    """
    x = np.full((samples, dims), np.nan)
    for dim in range(dims):
        # x[:, dim] = np.linspace(-random.random() * 10, random.random() * 10, samples)
        x[:, dim] = np.linspace(-10, 10, samples)

    y = objective_function(x)

    return x, y, objective_function
