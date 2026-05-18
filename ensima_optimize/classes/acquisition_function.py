"""
This module contains the AcquisitionFunction class used for Bayesian Optimization.

The AcquisitionFunction class provides methods to calculate acquisition functions
such as Expected Improvement (EI) based on a Gaussian Process model.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import pickle

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm


class AcquisitionFunction:
    """
    A class used to represent an Acquisition Function for Bayesian Optimization.
    """

    @staticmethod
    def expected_improvement(
        x,
        model,
        y_best,
        mode="minimization",
        full_cov=False,
        n_samples_mc=10000,
        plot=False,
    ):
        """
        Calculate Expected Improvement (EI) at points x based on a Gaussian Process model.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Input points where EI should be evaluated.
        model : Model
            A GP model (supports sklearn or torch backend).
        y_best : np.ndarray, shape (n_outputs,)
            Best known value of the target function.
        mode : str
            "minimization" or "maximization".
        full_cov : bool
            If True, consider full output covariance (multivariate EI using sampling).
        n_samples_mc : int
            Number of Monte Carlo samples if full_cov is True.
        plot : bool
            If True, plot the EI.

        Returns
        -------
        ei : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Expected improvement.
        """

        x = np.atleast_2d(x)
        mu, cov_or_std = model.predict(x, return_cov=full_cov)

        if full_cov:
            # expected improvement, but estimated via Monte Carlo sampling instead of the closed form.
            # mu: (n_samples, m), cov_or_std: (n_samples, m, m)
            ei_list = []
            for i in range(x.shape[0]):
                mean = mu[i]
                cov = cov_or_std[i]
                samples = np.random.multivariate_normal(
                    mean, cov, size=n_samples_mc
                )  # (n_samples_mc, m)

                if mode == "maximization":
                    improvement = samples - y_best
                else:  # minimization
                    improvement = y_best - samples

                improvement = np.maximum(0, improvement)
                ei_point = improvement.mean(axis=0)  # marginal EI per output
                ei_list.append(ei_point)
            ei = np.array(ei_list)  # shape (n_points_mc, m)
        else:
            # marginal variance approach (current code)
            std = np.maximum(cov_or_std, 1e-6)
            if mode == "maximization":
                improvement = mu - y_best
            else:
                improvement = y_best - mu
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)  # shape (n_points, m)

        if plot:
            AcquisitionFunction.plot_ei(x, ei)

        return ei

    @staticmethod
    def upper_confidence_bound(x, model, kappa=2.576, mode="minimization", plot=False):
        """
        Upper Confidence Bound acquisition function.

        kappa > 0 controls exploration-exploitation tradeoff (higher = more exploration).

        Parameters:
        - x: input points (n_samples, n_features)
        - model: GP model with predict() -> mu, std
        - kappa: exploration parameter
        - mode: "minimization" or "maximization"
        """

        x = np.atleast_2d(x)
        mu, std = model.predict(x)
        std = np.maximum(std, 1e-6)

        if mode == "minimization":
            ucb = mu - kappa * std  # For minimization, lower is better
            acquisition = -ucb  # BO usually maximizes acquisition function
        elif mode == "maximization":
            ucb = mu + kappa * std
            acquisition = ucb
        else:
            raise ValueError("Mode must be 'maximization' or 'minimization'.")

        if plot:
            AcquisitionFunction.plot_ei(x, acquisition)  # reuse plot_ei or rename it

        return acquisition

    @staticmethod
    def crowding_distance(points: np.ndarray) -> np.ndarray:
        """
        Calculates the crowding distance for each point in a set of non-dominated solutions.

        Args:
            points (np.ndarray): The set of Pareto points, shape (n_points, m_objectives).

        Returns:
            np.ndarray: The crowding distance for each point.
        """
        n_points, m_objectives = points.shape
        if n_points <= 2:
            return (
                np.ones(n_points) * np.inf
            )  # For small fronts, all points are equally "isolated"

        distances = np.zeros(n_points)

        for j in range(m_objectives):
            # Sort points by the j-th objective
            sorted_indices = np.argsort(points[:, j])
            sorted_points = points[sorted_indices]

            # Boundary points have infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Calculate distance for interior points
            for i in range(1, n_points - 1):
                distances[sorted_indices[i]] += (
                    sorted_points[i + 1, j] - sorted_points[i - 1, j]
                ) / (sorted_points[-1, j] - sorted_points[0, j] + 1e-9)

        return distances

    @staticmethod
    def plot_ei(
        x, ei, x_fields=None, y_fields=None, next_samples=None, save_path: str = None
    ):
        """
        Plot Expected Improvement (EI) vs each input dimension for each output separately,
        and show the sum of EI per input dimension as an additional row.
        If `next_samples` is provided, highlight them in the plots.

        Args:
            x (np.ndarray): Input sample points, shape (n_samples, n_dims).
            ei (np.ndarray): Expected Improvement values, shape (n_samples, n_outputs).
            x_fields (list[str] or None): Names of input dimensions for labeling axes.
            y_fields (list[str] or None): Names of output dimensions for labeling plots.
            next_samples (np.ndarray or None): Array of next sample points to highlight,
                shape (n_points, n_dims). If None, no highlight is shown.
            save_path (str or None): Path to save the plot. If None, plot is not saved.

        Returns:
            None. Displays matplotlib plots.
        """
        n_samples, n_dims = x.shape
        _, n_outputs = ei.shape
        colors = sns.color_palette("YlOrRd", 5).as_hex()
        if next_samples is not None:
            alphas = np.linspace(0.6, 1.0, next_samples.shape[0])[::-1]

        fig, axes = plt.subplots(
            n_outputs + 1,
            n_dims,
            figsize=(4 * n_dims, 3 * (n_outputs + 1)),
            sharex="col",
        )

        # Ensure axes is 2D
        if (n_outputs + 1) == 1:
            axes = axes[np.newaxis, :]
        if n_dims == 1:
            axes = axes[:, np.newaxis]

        # Set default labels if None provided
        if x_fields is None:
            x_fields = [f"X[{i}]" for i in range(n_dims)]
        if y_fields is None:
            y_fields = [f"Output {i+1}" for i in range(n_outputs)]

        # Plot EI per output and input dimension
        for out_idx in range(n_outputs):
            for dim_idx in range(n_dims):
                ax = axes[out_idx, dim_idx]
                ax.scatter(x[:, dim_idx], ei[:, out_idx], s=10, alpha=0.7)
                ax.set_title(f"EI ({y_fields[out_idx]}) vs {x_fields[dim_idx]}")
                ax.set_xlabel(x_fields[dim_idx])
                ax.set_ylabel("EI")
                ax.grid(True)

                # Highlight selected next_samples
                if next_samples is not None:
                    for i in range(next_samples.shape[0]):
                        matches = np.all(
                            np.isclose(x, next_samples[i], atol=1e-12), axis=1
                        )
                        if np.any(matches):
                            idx = np.where(matches)[0][0]
                        else:
                            distances = np.linalg.norm(x - next_samples[i], axis=1)
                            idx = np.argmin(distances)
                        ax.scatter(
                            next_samples[i, dim_idx],
                            ei[idx, out_idx],
                            c=colors[i],
                            marker="o",
                            s=100,
                            edgecolors="black",
                            alpha=alphas[i],
                            label=(f"Next sample {i+1}"),
                        )

        # Plot sum of EI across outputs
        ei_sum = ei.sum(axis=1)
        for dim_idx in range(n_dims):
            ax = axes[-1, dim_idx]
            ax.scatter(x[:, dim_idx], ei_sum, s=10, c="k", alpha=0.8)
            ax.set_title(f"Sum(EI) vs {x_fields[dim_idx]}")
            ax.set_xlabel(x_fields[dim_idx])
            ax.set_ylabel("Sum EI")
            ax.grid(True)

            # Highlight next samples
            if next_samples is not None:
                for i in range(next_samples.shape[0]):
                    matches = np.all(np.isclose(x, next_samples[i], atol=1e-12), axis=1)
                    if np.any(matches):
                        idx = np.where(matches)[0][0]
                    else:
                        distances = np.linalg.norm(x - next_samples[i], axis=1)
                        idx = np.argmin(distances)
                    ax.scatter(
                        next_samples[i, dim_idx],
                        ei_sum[idx] * 1.05,
                        c=colors[i],
                        marker="o",
                        s=100,
                        edgecolors="black",
                        alpha=alphas[i],
                        label=f"Next sample {i+1}" if dim_idx == 0 else None,
                    )

        # Show legend once
        if next_samples is not None:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", fontsize="medium")

        if save_path is not None:
            plt.savefig(f"{save_path}/ei.png", bbox_inches="tight")
            with open(f"{save_path}/ei.pkl", "wb") as f:
                pickle.dump(fig, f)
        else:
            plt.tight_layout()
            plt.show()
