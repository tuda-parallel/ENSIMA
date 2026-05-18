"""
Scatter plot utilities for visualizing multi-output Bayesian optimization results.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import json
import os
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot(
    x_fields,
    y_fields,
    x,
    y,
    attention_coefficients=None,
    highlight_points=None,
    save_path: str = None,
):
    """
    Generates scatter plots for multi-output data visualizations across multiple input dimensions.

    Allows deep customization, including the use of attention coefficients to scale output data
    and highlighting particular points for further emphasis. Supports flexible axis labeling,
    automatic subplots generation, and custom color mappings based on sample size.

    Parameters:
        x_fields: List[str]
            A list of labels for the x-axis corresponding to each dimension in the input data.
        y_fields: List[str]
            A list of labels for the y-axis corresponding to the output data.
        x: np.ndarray
            A 2D numpy array representing input data. Rows are samples, columns are input dimensions.
        y: np.ndarray
            A 2D numpy array representing output data. Rows are samples, columns are output dimensions.
        attention_coefficients: Optional[List[float]]
            A list of scaling factors applied to each output column in `y`. Defaults to ones if not provided.
        highlight_points: Optional[np.ndarray]
            A 2D numpy array representing specific points to highlight. Each row is a point, and columns
            match the inputs' dimensions.
        save_path: Optional[str]
            A path to save the plot as an image file; if None, the plot will be displayed.

    Raises:
        ValueError
            Raised if the length of `attention_coefficients` does not match the number of `y_fields`.

    """
    n_samples, n_dims = x.shape
    n_outputs = y.shape[1]

    if attention_coefficients is None:
        attention_coefficients = [1] * n_outputs
    elif len(attention_coefficients) != n_outputs:
        raise ValueError("attention_coefficients length must match number of y_fields")

    cmap = cm.get_cmap("jet", n_samples)
    colors = [cmap(i) for i in range(n_samples)]

    fig, axes = plt.subplots(
        n_outputs, n_dims, figsize=(4 * n_dims, 3 * n_outputs), sharex="col"
    )
    if n_outputs == 1:
        axes = axes[np.newaxis, :]
    if n_dims == 1:
        axes = axes[:, np.newaxis]

    alphas = None
    if highlight_points is not None:
        alphas = np.linspace(0.6, 1.0, highlight_points.shape[0])[::-1]

    for out_idx, y_field in enumerate(y_fields):
        for dim_idx, x_field in enumerate(x_fields):
            ax = axes[out_idx, dim_idx]
            x_vals = x[:, dim_idx]
            y_vals = y[:, out_idx] * attention_coefficients[out_idx]
            ax.scatter(
                x_vals,
                y_vals,
                s=20,
                alpha=0.7,
                c=colors,  # pass the full list/array of colors
            )

            if highlight_points is not None:
                for i in range(highlight_points.shape[0]):
                    distances = np.linalg.norm(x - highlight_points[i], axis=1)
                    idx = np.argmin(distances)
                    ax.scatter(
                        highlight_points[i, dim_idx],
                        y[idx, out_idx] * attention_coefficients[out_idx],
                        marker="o",
                        edgecolors="black",
                        s=100,
                        color=colors[idx],
                        alpha=alphas[i],
                        label=(f"Highlight {i+1}" if dim_idx == 0 else None),
                    )

            if out_idx == n_outputs - 1:
                ax.set_xlabel(x_field)
            if dim_idx == 0:
                ax.set_ylabel(y_field)

            ax.grid(True)
            if out_idx == 0:
                ax.set_title(x_field)

            if out_idx == 0 and dim_idx == 0 and highlight_points is not None:
                ax.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ensure folder exists
        plt.savefig(
            f"{save_path}/points.png", bbox_inches="tight"
        )  # save figure to file
        with open(f"{save_path}/points.pkl", "wb") as f:
            pickle.dump(fig, f)
    else:
        plt.show()


def plot_simulation_results(json_file="simulation_results.json"):
    # Load JSON
    with open(json_file) as f:
        data = json.load(f)

    # Sort keys by simulation order (sim_01, sim_02, …)
    sims = sorted(data.keys())

    # Extract relevant info
    best_outputs = np.array([data[k]["best_output"] for k in sims], dtype=float)
    jobnames = [data[k]["jobname"] for k in sims]
    targets = [data[k].get("target", None) for k in sims]

    # Find global best
    best_idx = np.argmax(best_outputs)
    best_value = best_outputs[best_idx]

    # --- Plot setup ---
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(len(sims)), best_outputs, marker="o", label="Best Outputs", lw=1.5
    )
    plt.scatter(
        best_idx,
        best_value,
        color="red",
        zorder=5,
        label=f"Global Best ({sims[best_idx]})",
    )

    # Annotate each point with its jobname or target
    for i, (job, tgt) in enumerate(zip(jobnames, targets)):
        label = f"{job}"
        if tgt is not None:
            label += f" | Target={tgt}"
        plt.text(
            i, best_outputs[i], label, fontsize=8, ha="center", va="bottom", rotation=30
        )

    plt.title("Best Output per Simulation")
    plt.xlabel("Simulation Index")
    plt.ylabel("Best Output")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save and show
    plt.savefig("simulation_comparison.png", dpi=300)
    plt.show()
