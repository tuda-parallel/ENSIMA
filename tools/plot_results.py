# Sort simulations by their sim_xx key
import json

import matplotlib.pyplot as plt
import numpy as np

from ensima.helpers.units import get_unit

# Load data
# json_file = "/d/gitlab/ensima-code/optimization/SeatShell.json"
json_file = "/d/gitlab/ensima-code/optimization/paper/MOE/DACH-VWS/DACH-VWS.json"
# json_file = "/d/gitlab/ensima-code/optimization/paper/MOE/Laengstraeger/Laengstraeger_02_2.json"
# json_file = "/d/gitlab/ensima-code/optimization/paper/MOE/Laengstraeger/Laengstraeger_02.json"
# json_file = "/d/gitlab/ensima-code/optimization/paper/MOE/SeatShell/SeatShell.json"
# json_file = "/d/gitlab/ensima-code/optimization/paper/MLVGP/SeatShell.json"

with open(json_file) as f:
    data = json.load(f)

from ensima.helpers.co2 import estimate_co2
from ensima.helpers.energy import estimate_energy

expert = {}
if "DACH" in json_file:
    expert = {
        "jobname": "DACH",
        "x": [
            [140.00, 3.00, 0.00, 0.00, -6.4, 6.9],
            [140.00, 3.00, 0.00, 220.00, -11.1, 2.8],
            [140.00, 6.00, 0.03, 220.00, -15.8, 0.9],
            [140.00, 9.00, 0.06, 220.00, -21.1, 0.4],
        ],
        "y": [
            [38.9, 34.40, 15.90, 10.80, 0.00, 0.00, 0.00],
            [14.4, 13.20, 10.50, 61.90, 0.00, 0.00, 0.00],
            [0.9, 9.90, 7.60, 81.60, 0.00, 0.00, 0.00],
            [0.5, 2.60, 9.10, 87.80, 0.00, 0.00, 0.00],
        ],
        "best_output": [0.5, 2.60, 9.10, 87.80, 0.00, 0.00, 0.00],
        "elapsed_seconds": [
            2 * 3600 + 18 * 60 + 12,  # 8292
            3 * 3600 + 5 * 60 + 39,  # 11139
            2 * 3600 + 49 * 60 + 5,  # 10145
            2 * 3600 + 50 * 60 + 48,  # 10248
        ],
        "args": {
            "x_fields": ["Rp", "p", "Fr", "db", "D", "T"],
            "y_fields": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
        },
    }

if len(expert) > 0:
    cores = 16
    cpu_frequency_Hz = 2.5 * 1e9
    expert["energy"] = [
        estimate_energy(t, cores, frequency=cpu_frequency_Hz)
        for t in expert["elapsed_seconds"]
    ]
    expert["co2"] = [estimate_co2(energy) for energy in expert["energy"]]


def plot_over_iterations(field, name, sims=None, best_points=None, ax=None):
    """
    Plot a list/array of simulations over iterations.
    Automatically uses global 'data' keys for labeling.
    Optionally highlights top points.

    Args:
        field: list/array of shape (num_sims, num_iterations)
        name: str, Y-axis label
        best_points: list of dicts with keys 'sim' (name) and 'iter'
        ax: optional matplotlib axis
    """
    show_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        show_fig = True
    if sims is None:
        sims = sorted(data.keys())

    for sim_idx, sim_name in enumerate(sims):
        vals = np.array(field[sim_idx], dtype=float)
        iterations = np.arange(len(vals))
        ax.plot(iterations, vals, marker="o", label=f"{sim_name}")

    # Highlight top points if provided
    if best_points is not None:
        colors = plt.cm.cool(np.linspace(0, 1, len(best_points)))
        for rank, (info, color) in enumerate(zip(best_points, colors), 1):
            sim_idx = sims.index(info["sim"])
            iter_idx = info["iter"]
            y_val = float(field[sim_idx][iter_idx])
            x_val = iter_idx
            ax.scatter(
                x_val,
                y_val,
                s=120,
                edgecolors="black",
                linewidths=1.5,
                color=color,
                zorder=5,
                label=f"Top {rank}",
            )
            info["rank"] = f"Top {rank}"

    ax.set_xlabel("Iteration")
    ax.set_ylabel(name)
    ax.set_title(f"{name} vs Iterations")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if show_fig:
        plt.show()


def plot_cumulative_with_secondary(
    field, name, secondary_field=None, secondary_name=None
):
    """
    Plot cumulative values over iterations for multiple simulations with fill under the curve.
    Optionally plot a secondary cumulative field on the left Y-axis.

    Args:
        field: list or array, main field (plotted on right Y-axis)
        name: str, label for main field
        secondary_field: list or array, optional, secondary field (left Y-axis)
        secondary_name: str, label for secondary field
    """
    if secondary_field is not None:
        fig, ax_right = plt.subplots(figsize=(12, 3))
    else:
        fig, ax_right = plt.subplots(figsize=(8.7, 3))
    iterations = None

    # Plot main cumulative field on right axis
    for sim_idx, sim_name in enumerate(sims):
        vals = np.array(field[sim_idx], dtype=float)
        cumulative_vals = np.cumsum(vals)
        iterations = np.arange(len(cumulative_vals))
        ax_right.plot(
            iterations,
            cumulative_vals,
            marker="o",
            linestyle="-",
            label=(
                f"{sim_name} ({name})" if secondary_field is not None else f"{sim_name}"
            ),
        )
        ax_right.fill_between(iterations, 0, cumulative_vals, alpha=0.2)

    ax_right.set_xlabel("Iteration")
    ax_right.set_ylabel(name)
    ax_right.grid(True, linestyle="--", alpha=0.6)

    if secondary_field is not None:
        ax_left = ax_right.twinx()
        for sim_idx, sim_name in enumerate(sims):
            sec_vals = np.array(secondary_field[sim_idx], dtype=float)
            cumulative_sec = np.cumsum(sec_vals)
            ax_left.plot(
                iterations,
                cumulative_sec,
                marker="x",
                linestyle="--",
                label=f"{sim_name} ({secondary_name})",
            )
            ax_left.fill_between(iterations, 0, cumulative_sec, alpha=0.1)
        ax_left.set_ylabel(secondary_name)

        # Combine legends
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        ax_right.legend(
            lines_right + lines_left,
            labels_right + labels_left,
            loc="center left",
            bbox_to_anchor=(1.1, 0.6),
        )
    else:
        ax_right.legend(loc="best")

    plt.title(
        f"Cumulative {name}"
        + (f" & {secondary_name}" if secondary_field is not None else "")
        + " vs Iterations"
    )
    plt.tight_layout()
    plt.show()


sims = sorted(data.keys())
energy = [data[k].get("energy", None) for k in data]
co2 = [data[k].get("co2", None) for k in data]
elapsed_seconds = [data[k].get("elapsed_seconds", None) for k in data]

if expert:
    sims.append("expert")
    expert_energy = np.array(expert["energy"], dtype=float)
    expert_elapsed = np.array(expert["elapsed_seconds"], dtype=float)
    expert_co2 = np.array(expert["co2"], dtype=float)
    max_len = max(len(e) for e in energy if e is not None)
    if len(expert_energy) < max_len:
        expert_energy = np.pad(
            expert_energy, (0, max_len - len(expert_energy)), constant_values=np.nan
        )
        expert_elapsed = np.pad(
            expert_elapsed, (0, max_len - len(expert_elapsed)), constant_values=np.nan
        )
        expert_co2 = np.pad(
            expert_co2, (0, max_len - len(expert_co2)), constant_values=np.nan
        )
    energy.append(expert_energy)
    elapsed_seconds.append(expert_elapsed)
    co2.append(expert_co2)

all_energy = np.concatenate([np.array(e).ravel() for e in energy if e is not None])
max_energy = np.max(all_energy)
factor, unit = get_unit(max_energy, "J")
energy = np.array(energy) / factor
elapsed_minutes = [[v / 60 for v in sim] for sim in elapsed_seconds]

plot_over_iterations(energy, f"Energy ({unit})", best_points=best_points, sims=sims)

plot_over_iterations(co2, "CO2 (g)", sims=sims)

# plot_over_iterations(elapsed_seconds, "Time (seconds)", sims=sims)
plot_over_iterations(elapsed_minutes, "Time (minutes)", sims=sims)

plot_cumulative_with_secondary(elapsed_seconds, "Time (seconds)")
plot_cumulative_with_secondary(
    elapsed_seconds, "Time (seconds)", energy, f"Energy ({unit})"
)
