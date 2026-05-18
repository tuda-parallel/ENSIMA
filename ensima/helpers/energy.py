"""
This module provides helper functions for estimating energy consumption.

The `estimate_energy` function calculates the dynamic energy consumption based on
real-time CPU frequency and utilization over a specified duration.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import contextlib
import time

import psutil
import torch

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# Internal statea
_start_time = None
_gpu_energy_start = None
_device = None


def estimate_energy(
    duration: float,
    num_cores=None,
    capacitance: float = 1e-9,
    base_voltage: float = 1.2,
    frequency: float = None,
):
    """
    Estimates dynamic energy consumption based on real-time CPU frequency and utilization.

    Args:
        duration: Measurement duration in seconds.
        capacitance: Effective capacitance (default: 1e-9).
        base_voltage: Approximate CPU voltage in Volts (default: 1.2V).
        frequency: CPU frequency in Hz (optional) for post-simulation, otherwise this is cimputed online.

    Returns:
        Estimated energy consumption in Joules.
    """
    if num_cores is None:
        num_cores = psutil.cpu_count(logical=False)  # Get number of physical cores

    # Measure CPU frequency and utilization dynamically
    if frequency is None:
        # don't use current, this fluctuates
        cpu_freq = psutil.cpu_freq().current * 1e6  # Convert MHz to Hz
        # freq = psutil.cpu_freq()
        # cpu_freq = freq.min * 1e6  # Convert MHz to Hz
    else:
        cpu_freq = frequency

    # CPU usage
    # cpu_usage = psutil.cpu_percent(interval=duration) / 100  # Get CPU utilization (0-1)
    # if cpu_usage == 0:
    #     cpu_usage = psutil.cpu_percent(interval=duration) / 100
    #     if cpu_usage == 0:  # If still 0, assume minimal activity
    #         cpu_usage = 0.01  # Set to 1% to avoid division by zero
    cpu_usage = 1

    # Dynamic power estimation formula
    # P = C×V^2×f
    if cpu_freq and capacitance and base_voltage and num_cores and cpu_usage:
        power = capacitance * (base_voltage**2) * cpu_freq * num_cores * cpu_usage
    else:
        power = 0

    # Energy = Power × Time
    energy = power * duration  # Joules

    # print(f"{capacitance} * ({base_voltage} ** 2) * {cpu_freq} * {num_cores} * {cpu_usage}")

    return energy


def energy_estimation_start(device: str = "cpu"):
    """
    Start energy measurement for training or prediction.
    Args:
        device: "cpu" or "cuda"
    """
    global _start_time, _gpu_energy_start, _device
    _device = device.lower()
    _start_time = time.time()

    if _device == "cuda" and GPU_AVAILABLE:
        _gpu_energy_start = _read_gpu_energy_mj()
    else:
        _gpu_energy_start = 0.0


def energy_estimation_end():
    """
    Stop measurement and return total energy in Joules.
    Includes both CPU and GPU energy if device is 'cuda'.
    Returns:
        float: estimated total energy (J)
    """
    global _start_time, _gpu_energy_start, _device

    if _start_time is None:
        raise RuntimeError(
            "energy_estimation_start() must be called before energy_estimation_end()."
        )

    duration = time.time() - _start_time

    # Always compute CPU energy
    energy = _compute_cpu_energy(duration)

    # Add GPU energy if applicable
    if _device == "cuda" and GPU_AVAILABLE:
        energy += _compute_gpu_energy(_gpu_energy_start)

    # Reset internal state
    _start_time = None
    _gpu_energy_start = None
    _device = None

    return energy


def _read_gpu_energy_mj():
    """Return cumulative GPU energy (mJ) across all GPUs."""
    total = 0.0
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        with contextlib.suppress(pynvml.NVMLError):
            total += pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    return total


def _compute_gpu_energy(start_mj):
    """Compute GPU energy since start (Joules)."""
    end_mj = _read_gpu_energy_mj()
    return max(0.0, (end_mj - start_mj) / 1e3)  # mJ → J


def _compute_cpu_energy(duration_s, capacitance=1e-9, base_voltage=1.2):
    num_cores = torch.get_num_threads()
    energy = estimate_energy(duration_s, num_cores)
    return energy
