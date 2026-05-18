"""
Bayesian Optimization class for optimizing the design parameter selection.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import argparse
import datetime
import json
import multiprocessing
import os
import pickle
import shutil
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from rich.console import Console
from rich.table import Table
from scipy.signal import find_peaks

from ensima_optimize.classes.acquisition_function import AcquisitionFunction
from ensima_optimize.classes.file_modifier import FileModifier
from ensima_optimize.classes.logger import Logger
from ensima_optimize.classes.mixture_of_experts import MoEPointNetSystem
from ensima_optimize.classes.model import Model
from ensima_optimize.classes.search_space import SearchSpace
from ensima_optimize.classes.simulation import Simulation
from ensima_optimize.helpers.co2 import estimate_co2
from ensima_optimize.helpers.energy import (
    energy_estimation_end,
    energy_estimation_start,
    estimate_energy,
)
from ensima_optimize.helpers.misc import delete_matching, safe_remove
from ensima_optimize.helpers.optimum import find_min_or_max
from ensima_optimize.helpers.pareto import pareto_frontier, plot_pareto
from ensima_optimize.helpers.read_data import (
    read_data,
    read_data_type,
    read_last_progress,
)
from ensima_optimize.helpers.read_geometry import read_coordinates_from_file
from ensima_optimize.helpers.serilaize import to_json_serializable
from ensima_optimize.helpers.units import convert_seconds_to_hms, format_with_units


class BayesianOptimization:

    def __init__(
        self,
        args: Namespace = None,
        x: np.ndarray = None,
        y: np.ndarray = None,
        objective_function: callable = None,
        optimizer_restart: int = 5,
        type_filter: bool = False,
        parts: list[tuple[str, str]] = None,
        types: np.ndarray = None,
        device: str = "auto",
        backend: str = "torch",
    ) -> None:
        """
        A Bayesian Optimization framework for optimizing an objective function.

        This class leverages Gaussian Process Regression with a Matern kernel to model
        the objective function and guide the search for optimal solutions.

         Args:
            args (Namespace, optional): Command line arguments passed. May contain:
                - ofsolver (str, optional): Path to the OpenFOAM solver executable. Defaults to None.
                - jobname (str, optional): Name of the optimization job. Defaults to "optimization".
                - path (str, optional): Path to job directory. Defaults to "./".
                - output (str, optional): Directory or filename for storing optimization results. Defaults to "{jobname}_results".
                - cores (int, optional): Number of CPU cores allocated for computations. Defaults to 4.
                - openform (str, optional): Path to the OpenForm software. Defaults to None.
                - session (str, optional): Name of the OpenForm session file. Defaults to "session.ofs".
            x (numpy.ndarray): Initial input samples of shape (n_samples, n_features). Defaults to None.
            y (numpy.ndarray): Corresponding output values of shape (n_samples, n_targets). Defaults to None.
            objective_function (callable, optional): The objective function to optimize. Defaults to None.
            optimizer_restart (int, optional): Number of restarts allowed for optimizer. Defaults to 5.
            type_filter (bool,optional): in case the data was filtered by type (read_data_type was used). Defaults to False.
            parts (list of tuple[str, str]):  list containing the job name and the path to the .t52 geo file
            device (str): Torch device to use (e.g., "cpu", "cuda", or "auto"). Default "auto". Works only if torch is set for training in model.py
            backend (str): Torch backend to use (e.g., "torch" or "sklearn"). Default "torch" as it works better

        Raises:
            ValueError: If `x` or `y` is not provided.

        """
        self.args = args
        self.file_modifier = None
        self.constrains = None
        self.logger_obj = Logger(__name__, level=self.args.log_level)
        self.logger = self.logger_obj.logger
        self.next_samples = None
        self.x_sample_space = None
        self.ei = None
        self.ei_sum = None
        self.type_filter = type_filter
        self.types = types
        self.device = device
        self.training_energy = 0
        self.parts = parts

        # x and y fields
        self.logger.info("Initializing Bayesian Optimization Model")
        self.logger.info(f"Input fields are: {self.args.x_fields}")
        self.logger.info(f"Output fields are: {self.args.y_fields}")

        if args is None:  # in case dummy init
            args = argparse.Namespace()
            self.args.x_fields = [f"x{i+1}" for i in range(x.shape[1])]

        if "label" in args.output and not type_filter:
            self.logger.critical("Data is labeled, the filter will be applied")
            self.type_filter = True

        if self.types is not None and len(self.types) > 0:
            self.type_filter = True

            # Weight the outputs:
        if self.args.attention_coefficients is None:
            self.args.attention_coefficients = np.ones(y.shape[1])
        else:
            arr = np.array(self.args.attention_coefficients, dtype=float)
            if np.any(arr < -1) or np.any(arr > 1):
                raise ValueError("Attention coefficients must be between -1 and 1")
            if arr.shape[0] != y.shape[1]:
                raise ValueError(
                    "Attention coefficients length must match number of y_fields"
                )
            self.args.attention_coefficients = arr
            self.logger.info(
                f"y values are scaled by: {self.args.attention_coefficients}"
            )

        if self.args.approximate_computing_check is not None:
            self.logger.info(
                f"Approximate computing check is set to {self.args.approximate_computing_check}"
            )
            if len(self.args.approximate_computing_limit) != len(self.args.y_fields):
                self.logger.critical(
                    "Approximated computing limit is not equal to number of y_fields"
                )
                exit(1)

        # Set Objective function if exists, otherwise point to input file
        if not self.args.jobname:
            if objective_function:
                self.objective_function = objective_function
            else:
                raise RuntimeError("Objective function must be passed for dummy run")
        else:
            self.objective_function = None

        # Raise error if the initial data set is empty
        if x is None or y is None:
            raise ValueError("Initial samples (x and y) must be provided.")

        self.x = x
        self.y = y * self.args.attention_coefficients
        self.x_dim = x.shape[1] if x is not None else None
        self.y_dim = y.shape[1] if y is not None else None
        self.training_energy = 0
        self.prediction_energy = 0

        # specify a training device:

        if "auto" in self.device:
            if torch.cuda.is_available() and "NVIDIA" in torch.cuda.get_device_name(0):
                self.device = "cuda"
                if "torch" not in backend:
                    self.logger.warning("Setting torch backend as default")
                    backend = "torch"
            else:
                self.device = "cpu"

            self.logger.info(f"Device detect {self.device} in auto mode")

        if args.input is not None:
            self.logger.warning("Input vector provided. BO is skipped ")
            # self.x = []
            # self.y = []
            self.model = None
            self.args.approximate_computing_check = None
            self.args.parallel_samples = 1
            self.training_energy = 0
            return

        if self.args.energy_estimation:
            energy_estimation_start(device=self.device)
        # Standard Model
        if parts is None:
            if len(self.x) < 2:
                self.logger.error(
                    f"Not enough point for optimization ({len(self.x)} points provided. "
                    f"Run MOE to obtain initial data (see example/example_moe_optimization.py)"
                )
                self.logger.info("Setting up randome sampling")
            self.model = Model(
                self.x,
                self.y,
                backend=backend,
                normalize_y=False,
                log_level=args.log_level,
                epochs=self.args.epochs,
                optimizer_restart=optimizer_restart,
                save_and_load=args.save_and_load,
                precision=args.x_space_precision,
                latent_input_dim=args.latent_input_dim,
                latent_output_dim=args.latent_output_dim,
                device=self.device,
            )
        # Mixture of Experts (Gating and Supervised mode are hardcoded)
        else:
            self.logger.info("Using Mixture of experts model")
            self.type_filter = True
            geometry = []
            train_points = {}
            model_settings = {}
            for part, geometry_path in parts:
                # Extract the geometry
                coords = read_coordinates_from_file(
                    geometry_path, log_level=args.log_level
                )
                if len(coords) > 0:
                    geo_x = coords[:, 0]
                    geo_y = coords[:, 1]
                    geo_z = coords[:, 2]
                else:
                    continue
                geometry.append((geo_x, geo_y, geo_z, part))
                # Extract the relevant input output points from the large csv file
                part_x = x[self.types == part]
                part_y = y[self.types == part]
                train_points[part] = (part_x, part_y)

                # Extract Settings
                model_settings[part] = {
                    "backend": "torch",
                    "normalize_y": False,
                    "log_level": args.log_level,
                    "epochs": self.args.epochs,
                    "optimizer_restart": optimizer_restart,
                    "save_and_load": args.save_and_load,
                    "precision": args.x_space_precision,
                    "latent_input_dim": args.latent_input_dim,
                    "latent_output_dim": args.latent_output_dim,
                    "device": self.device,
                }
            # create moe Model
            self.model = MoEPointNetSystem(
                mode="supervised",  # or unsupervised
                model_settings=model_settings,
                train_points=train_points,
                geo_points=geometry,
                # n_points=1024, # let algo decide on min length of parts
                log_level="DEBUG",
                args=args,
                sample_mode=self.args.sample_mode,
            )
            self.model.set_geometry_path(args.geometry_path)
            self.model.set_gating_mode(True)  # True is soft, False is hard gating

        # remove signal file if it exists
        safe_remove(f"{self.args.path}/{self.args.jobname}*.sig")
        if self.args.energy_estimation:
            self.training_energy = energy_estimation_end()
        self.logger.info(f"Training energy: {self.training_energy:.2f} J")

    def optimize(
        self,
        n_iters: int,
        n_parallel: int = 1,
        user_constraints: dict = None,
        epochs: int = 100,
    ) -> None:
        """
        Optimize the objective function using Bayesian Optimization.

        Args:
            n_iters (int): Number of iterations to run the optimization.
            n_parallel (int, optional): Number of parallel samples to evaluate. Defaults to 1.
            user_constraints (dict, optional): Dictionary mapping variable names to (lower_bound, upper_bound) tuples. These
                    override the default bounds specified in constrains.py.
            epochs (int,optional): Number of training epochs for torch backend.
         Example:
            constraints = {"Rp": [270, 420, 890], "D": (0.01, 5.0), "p": (0.01, 0.5)}
            bayes_opt.optimize(n_iters=50, n_parallel=5, user_constraints=constrains)
        """
        # counters to terminate BO after end is reached
        counter = 0
        total_sim_energy = 0
        total_train_energy = self.training_energy
        total_prediction_energy = 0
        total_co2 = 0
        total_start_time = time.time()
        end_condition = False
        old_ei_sum = np.nan
        skip = False
        m = multiprocessing.Manager()
        lock = None
        # weights = np.ones(
        #     self.y.shape[1]
        # )  # can be customized for importance, but y is already scaled
        if n_parallel > 1:
            lock = m.Lock()
        adjusted_cofficents = np.where(
            self.args.attention_coefficients == 0,
            1,
            self.args.attention_coefficients,
        )
        # init
        parallel_msg = f"({n_parallel} parallel samples)" if n_parallel > 1 else ""
        parallel_msg = f"Optimization started: {n_iters} steps targeted {parallel_msg}"
        self.logger.info(parallel_msg)
        console = Console()
        table = Table(title="Samples")
        x_fields_str = " ".join(self.args.x_fields)
        y_fields_str = " ".join(self.args.y_fields)
        table.add_column(f"X ({x_fields_str})", style="bold green", justify="center")
        table.add_column(f"Y ({y_fields_str})", style="bold green", justify="center")
        table.add_column(
            f"Y Mean Predicted ({y_fields_str})", style="bold green", justify="center"
        )
        table.add_column(
            f"Y STD Predicted ({y_fields_str})", style="bold green", justify="center"
        )
        table.add_column("EI", style="bold cyan", justify="center")
        if isinstance(self.model, MoEPointNetSystem):
            table.add_column(
                f"Experts (total {len(self.model.experts)}) ",
                style="bold cyan",
                justify="center",
            )
        if self.args.energy_estimation:
            table.add_column("Sim Energy ", style="bold cyan", justify="right")
            table.add_column("Train Energy ", style="bold cyan", justify="right")
            table.add_column("Pred Energy ", style="bold cyan", justify="right")
            table.add_column("CO₂ ", style="bold cyan", justify="right")

        table.add_column("Progress ", style="bold cyan", justify="right")
        table.add_column("Time ", style="bold cyan", justify="right")

        # save the results in a dict
        result = {
            "best_iteration": None,
            "x": [],
            "y": [],
            "y_model_mean": [],  # mean (μ)
            "y_model_std": [],  # std (σ) or covariance
            "ei_sum": [],
            "experts": [],
            "sim_energy": [],
            "train_energy": [],
            "pred_energy": [],
            "co2": [],
            "progress": [],
            "elapsed_seconds": [],
        }

        # set the constraints:
        search_space = SearchSpace(
            self.args.x_fields, user_constraints, self.args.log_level
        )

        # optimization loop
        #################################
        i = 0
        while i < n_iters:
            # print("\n")
            start_time = time.time()
            self.logger_obj.set_prefix(f"Iteration {i+1}/{n_iters}")
            self.logger.info("Started")
            # Take samples from a provided list (args.input). Skip optimization
            if self.args.input is not None:
                self.logger.warning("Selecting point from input vector")
                self.next_samples = np.array([self.args.input.pop(0)])
                self.x_sample_space = None
                self.ei = np.zeros(len(self.args.y_fields))
                self.ei_sum = 0
                mode = "minimization"
                for j, x_field in enumerate(self.args.x_fields):
                    self.logger.info(
                        f"Specified {x_field} values {self.next_samples[:,j]}"
                    )
            # Optimizations)
            else:
                # 1) set the search space (contains linear combination of variables)
                x_sample_space = search_space.create_sample_space(
                    x=self.x,
                    n_samples=self.args.x_space_n_samples,
                    expansion_factor=self.args.x_space_expansion_factor,
                    precision=self.args.x_space_precision,
                    method=self.args.x_space_structure,
                    point_creation=self.args.x_space_point_creation,
                )
                # 2) find current optimum (min or max)
                #! Decide if minimization or maximization
                # mode = "maximization"
                # y_best = np.max(self.y, axis=0)
                mode = "minimization"
                idx_best = find_min_or_max(self.y, self.x, self.args, self.logger, mode)
                y_best = self.y[idx_best]
                x_best = self.x[idx_best]

                # Find the best point
                self.logger.info("Finding best point")
                self._select_best_points(x_sample_space, y_best, n_parallel, mode=mode)
                # 2) overwrite if user desires (HGAL)
                if any(keyword in self.args.method for keyword in ["hg", "al"]):
                    self.logger.debug(f"Best point is {self.next_samples}")
                    self._human_guided_al(x_sample_space, n_parallel, index=i)
                self.logger.debug(f"Best point is {self.next_samples}")
                for j, x_field in enumerate(self.args.x_fields):
                    self.logger.info(f"Best {x_field} values {self.next_samples[:,j]}")
            # 4) execute simulation
            # Initialize ThreadPoolExecutor or ProcessPoolExecutor to run simulations in parallel
            # with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            with ProcessPoolExecutor(max_workers=n_parallel) as executor:
                futures = {}
                # Refine points in parallel
                if n_parallel > 1:
                    suffix = f"s ({i+1}--{i+n_parallel})"
                else:
                    suffix = ""
                msg = f"Executing {n_parallel} simulation{suffix} -- EI: {float(self.ei_sum):.5}"

                self.logger.info(msg)
                for idx, next_sample in enumerate(self.next_samples):
                    next_sample = next_sample.reshape(1, -1)
                    prefix = f"{i+1}+{idx}" if n_parallel > 1 else f"{i+1}"
                    self.logger_obj.set_prefix(f"Iteration {prefix}/{n_iters}")
                    self.logger.info("Running simulation")
                    # Run the simulation
                    # Submit each simulation task to the executor
                    future = executor.submit(
                        self._run_simulation,
                        next_sample,
                        i,
                        n_iters,
                        idx,
                        n_parallel,
                        lock,
                    )
                    futures[future] = idx
                    # next_sample, new_y, start, end  = self._run_simulation(next_sample, i, n_iters)

                # Wait for all futures to complete and process the results
                tasks = 0
                for future in as_completed(futures):
                    next_sample, new_y, start, end, progress = future.result()
                    idx = futures[future]
                    prefix = f"{i+1}+{idx}" if n_parallel > 1 else f"{i+1}"
                    self.logger_obj.set_prefix(f"Iteration {prefix}/{n_iters}")
                    self.logger.info("Simulation complete")
                    tasks += 1

                    # scale if needed
                    if not np.all(self.args.attention_coefficients == 1):
                        self.logger.debug(
                            f"new y from simulation: {np.round(new_y.flatten(), 2)}"
                        )
                        new_y = new_y * self.args.attention_coefficients
                        self.logger.debug(
                            f"new y inside model (transformed): {np.round(new_y.flatten(), 2)}"
                        )

                    # save prediction
                    # if the model hast not been trained or the input is given, skip prediction
                    if self.x.shape[0] <= n_parallel or self.args.input is not None:
                        mu_pred = np.full(self.y.shape[1], np.nan)
                        std_pred = np.full(self.y.shape[1], np.nan)
                    else:
                        mu_pred, cov_or_std_pred = self.model.predict(
                            next_sample, return_cov=self.args.full_cov
                        )
                        if self.args.full_cov:
                            std_pred = np.sqrt(np.diag(cov_or_std_pred.squeeze()))
                        else:
                            std_pred = cov_or_std_pred
                        self.logger.debug(
                            f"Predicted μ ± σ inside model (transformed): {np.round(mu_pred.flatten(), 2)} ± {np.round(std_pred.flatten(), 2)} "
                        )
                        # Apply attention coefficients if needed
                        if self.args.attention_coefficients is not None:
                            mu_pred = mu_pred / adjusted_cofficents
                            std_pred = std_pred / adjusted_cofficents
                        self.logger.debug(
                            f"Predicted μ ± σ: {np.round(mu_pred.flatten(), 2)} ± {np.round(std_pred.flatten(), 2)} "
                        )

                    # Update the samples
                    self.logger.info("Appending samples")
                    self.x = np.vstack((self.x, next_sample))
                    self.y = np.vstack((self.y, new_y))
                    self.logger.info(
                        f"new x size {self.x.shape}, new y size {self.y.shape}"
                    )
                    # append name
                    if self.type_filter:
                        self.types = np.append(self.types, self.args.jobname)

                    # Track the new minimum after the update
                    # This is basically the same as new_y
                    # 1) sum the weighted y values
                    # Problem if attention coefficients are 1, the sum is always the same
                    # idx_best = np.argmin(self.y @ weights)
                    # 2) compute distance to zero and find the points closest to it
                    # compute L2 norm (distance from zero) for each row
                    # row_norms = np.linalg.norm(self.y, axis=1)
                    # idx_best = np.argmin(row_norms)
                    idx_best = find_min_or_max(
                        self.y, self.x, self.args, self.logger, mode
                    )
                    new_y_best = self.y[idx_best]
                    new_x_best = self.x[idx_best]

                    # if not enough points where provided for training or no optimization was performed,
                    # dummy assigns the best values:
                    if self.x.shape[0] <= n_parallel or self.args.input is not None:
                        y_best = new_y_best
                        x_best = new_x_best
                    # Mixture of experts modifies self.y and self.x, check that the old min
                    # is still in the dataset, else adjust:
                    # check that y_best is in self.y
                    if not np.any(
                        np.all(
                            np.isclose(self.y, y_best, rtol=1e-8, atol=1e-10), axis=1
                        )
                    ):
                        self.logger.warning(
                            "y_best is not found in self.y."
                            "If you are using MOE, ignore this message, as y_best must be rested now"
                        )
                        # rest the best values
                        y_best = new_y_best
                        x_best = new_x_best

                    # estimate energy
                    if self.args.energy_estimation:
                        energy = estimate_energy(end - start, self.args.cores)
                        total_sim_energy += energy
                        total_prediction_energy += self.prediction_energy
                        # CO2 includes sim, train and prediction energy
                        # however training and prediction only happens once per iteration (even if tasks > 1)
                        # so we only add it when tasks == 1 or for the first task
                        current_iter_overhead = 0
                        if tasks == 1:
                            current_iter_overhead = (
                                self.training_energy + self.prediction_energy
                            )
                        co2 = estimate_co2(energy + current_iter_overhead)
                        total_co2 += co2
                    # track time
                    elapsed_time = time.time() - start_time

                    self.logger.info(f"┌─ x = {np.round(next_sample.flatten(), 2)}")
                    adjusted_y = new_y / adjusted_cofficents
                    adjusted_y[:, self.args.attention_coefficients == 0] = 0
                    self.logger.info(f"├─ f(x) = {np.round(adjusted_y.flatten(), 2)}")
                    self.logger.info(
                        f"├─ y predicted = {np.round(mu_pred.flatten(), 2)}"
                    )
                    self.logger.info(f"├─ Expect improvement = {self.ei_sum:.2f}")

                    if self.args.energy_estimation:
                        self.logger.info(
                            f"├─ Sim Energy = {energy:f} J -- Total: {total_sim_energy:f} J"
                        )
                        self.logger.info(
                            f"├─ Train Energy = {self.training_energy if tasks == 1 else 0:f} J -- Total: {total_train_energy:f} J"
                        )
                        self.logger.info(
                            f"├─ Pred Energy = {self.prediction_energy if tasks == 1 else 0:f} J -- Total: {total_prediction_energy:f} J"
                        )
                        self.logger.info(
                            f"├─ Co2 footprint = {co2:f} g of CO2 -- Total: {total_co2:f} g of CO2"
                        )
                    self.logger.info(
                        f"├─ Elapsed time = {str(datetime.timedelta(seconds=int(time.time() - start_time))).zfill(8)}"
                    )
                    # these values should be printed in the space of the of_simulation
                    # -> Adjust with cofficent
                    # for MOE, these points make no sense, as y is computed from other parts
                    # only when prgoress is 100%
                    if (
                        not np.allclose(y_best, new_y_best, atol=1e-8)
                        and progress >= 99
                    ):
                        self.logger.info(
                            f"├─ Old output min = {np.round(y_best/adjusted_cofficents, 2)}"
                        )
                        self.logger.info(f"├─ Old input at min = {np.round(x_best, 2)}")
                        self.logger.info(
                            f"├─ New output min = {np.round(new_y_best/adjusted_cofficents, 2)}"
                        )
                        self.logger.info(f"└─ Input at min = {np.round(new_x_best, 2)}")
                    elif self.x.shape[0] <= n_parallel or self.args.input is not None:
                        "└─ No optimization performed, skipping min calculation"
                    else:
                        self.logger.info(
                            f"├─ Output min (no change) = {np.round(new_y_best/adjusted_cofficents, 2)}"
                        )
                        self.logger.info(
                            f"└─ Input at min (no change) = {np.round(new_x_best, 2)}"
                        )
                        result["best_iteration"] = i + tasks

                    # Save results
                    #################################
                    result["x"].append(next_sample.flatten())
                    result["y"].append(adjusted_y.flatten())
                    result["ei_sum"].append(self.ei_sum)
                    if isinstance(self.model, MoEPointNetSystem):
                        result["experts"].append(dict(self.model.get_experts()))
                    else:
                        result["experts"].append(None)

                    if self.args.energy_estimation:
                        result["sim_energy"].append(energy)
                        result["train_energy"].append(
                            self.training_energy if tasks == 1 else 0
                        )
                        result["pred_energy"].append(
                            self.prediction_energy if tasks == 1 else 0
                        )
                        result["co2"].append(co2)
                    else:
                        result["sim_energy"].append(None)
                        result["train_energy"].append(None)
                        result["pred_energy"].append(None)
                        result["co2"].append(None)

                    result["progress"].append(progress)
                    result["elapsed_seconds"].append(elapsed_time)

                    # Save predictions
                    result["y_model_mean"].append(mu_pred.flatten())
                    result["y_model_std"].append(std_pred.flatten())

                    # save results for an intermediate table
                    #################################
                    row = [
                        f"{np.round(next_sample.flatten(), 2)}",
                        f"{np.round(adjusted_y.flatten(), 2)}",
                        f"{np.round(mu_pred.flatten(), 2)}",
                        f"{np.round(std_pred.flatten(), 2)}",
                        f"{self.ei_sum:.2f}",
                    ]
                    if isinstance(self.model, MoEPointNetSystem):
                        tmp = ", ".join(
                            f"{value*100:.1f}% {label}"
                            for label, value in self.model.get_experts().items()
                        )
                        row.append(f"{tmp}")
                    if self.args.energy_estimation:
                        row.extend(
                            [
                                format_with_units(energy, "J"),
                                format_with_units(
                                    self.training_energy if tasks == 1 else 0, "J"
                                ),
                                format_with_units(
                                    self.prediction_energy if tasks == 1 else 0, "J"
                                ),
                                format_with_units(co2, "g"),
                            ]
                        )
                    row.extend(
                        [
                            f"{progress}%",
                            f"{str(datetime.timedelta(seconds=int(elapsed_time))).zfill(8)}",
                        ]
                    )
                    table.add_row(*row)

                    # Print intermediate results:
                    #################################
                    self.logger.info("Results so far:")
                    console.print(table)
                    with open(
                        f"{self.args.jobname}_intermediate_results.txt",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        console_tmp = Console(
                            file=f,
                            force_terminal=False,
                            color_system=None,
                            markup=False,
                        )
                        console_tmp.print(
                            f"%%%%%%%% Results (Iteration {i+tasks}/{n_iters}) %%%%%%%%%%%%%%%"
                        )
                        console_tmp.print(
                            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n"
                            f"Best Output: {np.round(new_y_best/adjusted_cofficents, 2)} \n"
                            f"Best Input: {np.round(new_x_best, 2)} \n"
                            f"Total Sim Energy: {format_with_units(total_sim_energy, 'J')} \n"
                            f"Total Train Energy: {format_with_units(total_train_energy, 'J')} \n"
                            f"Total Pred Energy: {format_with_units(total_prediction_energy, 'J')} \n"
                            f"Total CO₂: {format_with_units(total_co2, 'g')}"
                        )
                        if self.args.approximate_computing_check is not None:
                            console_tmp.print(
                                f"Approximate Computing check: {self.args.approximate_computing_check} %\n"
                                f"Approximate Computing Limit: {self.args.approximate_computing_limit}"
                            )
                        if self.args.time_limit is not None:
                            console_tmp.print(
                                f"Time Limit: {convert_seconds_to_hms(self.args.time_limit)}\n"
                            )
                        console_tmp.print("\n" + "%" * 40)
                        console_tmp.print(table)

                    # Check end condition
                    #################################
                    if not any(keyword in self.args.method for keyword in ["hg", "al"]):
                        # check end condition only if no human is involved
                        self.logger.debug("Checking condition for no_improvement")
                        if self.args.end_condition == "no_improvement":
                            if not np.isnan(old_ei_sum):
                                self.logger.debug(
                                    f"abs({old_ei_sum} - {self.ei_sum}) < 1 "
                                    f"-- {abs(old_ei_sum - self.ei_sum) < 1}"
                                )
                                if (
                                    abs(old_ei_sum - self.ei_sum) < 1
                                    and n_parallel == tasks
                                ):
                                    counter += 1
                                    self.logger.warning(
                                        f"No improvement {counter}/{int(self.args.end_value)}"
                                    )
                                    if counter == int(self.args.end_value):
                                        end_condition = True
                            old_ei_sum = self.ei_sum
                        elif self.args.end_condition == "constant_min":
                            self.logger.debug("Checking condition for constant_min)")
                            self.logger.debug(
                                f" np.allclose({y_best}, {new_y_best}, atol=1e-8) -- {np.allclose(y_best, new_y_best, atol=1e-8)}"
                            )
                            if np.allclose(y_best, new_y_best, atol=1e-8):
                                counter += 1
                                self.logger.warning(
                                    f"Constant min {counter}/{int(self.args.end_value)}"
                                )
                                if counter == int(self.args.end_value):
                                    end_condition = True
                        elif self.args.end_condition == "energy_budget":
                            self.logger.debug("Checking condition for energy_budget:")
                            if self.args.energy_estimation:
                                if total_sim_energy > self.args.end_value:
                                    end_condition = True
                                    self.logger.warning(
                                        f"Energy budget reached {total_sim_energy:.3f}/{self.args.end_value:.3f}"
                                    )
                            else:
                                self.logger.error("Energy estimation must be enabled")

                    self.logger.info("Ended")
                    if end_condition:
                        self.logger.info(
                            "Solution converged before reaching iteration end\n --- Ending BO ---"
                        )
                        skip = True
                        break
            if skip:
                break
            else:
                # Train again the GPR models
                #################################
                i += tasks
                self.logger_obj.set_prefix(f"Iteration {i}/{n_iters}")
                if self.args.input is not None:
                    # skip the training
                    continue
                # Refit the GP model only once all new x and y are acquired
                if tasks > 1:
                    self.logger.info(f"Updating model with {tasks} new samples")
                else:
                    self.logger.info(f"Updating model with {tasks} new sample")
                if self.args.energy_estimation:
                    energy_estimation_start()
                ###### actual training
                start_time = time.time()
                self.model.train(self.x, self.y, epochs)
                elapsed = time.time() - start_time
                ###########
                if self.args.energy_estimation:
                    self.training_energy = energy_estimation_end()
                    total_train_energy += self.training_energy
                self.logger.info(f"Finished updating model in {elapsed:.2f} seconds")
                # Adjust x and y if moe is no longer needed. Shrink to points belonging to expert
                if (
                    isinstance(self.model, MoEPointNetSystem)
                    and self.model.reset_points
                ):
                    self.x = self.x[-self.args.new_expert_start :, :]
                    self.y = self.y[-self.args.new_expert_start :, :]
                    self.model.reset_points = False
                    self.logger.info(f"Resting x {self.x.shape} and y {self.y.shape}")
                self.logger.info("Ended\n")

        # Create and print summary tables
        #################################
        # 1) summary table
        table_summary = Table(title="Optimization Summary")
        table_summary.add_column("Metric", style="bold cyan")
        table_summary.add_column("Value", style="bold green", justify="right")
        table_summary.add_row(
            "Optimization Job",
            self.args.jobname if self.args.jobname is not None else "dummy",
        )
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        table_summary.add_row("Date", date)
        if self.args.target is not None:
            table_summary.add_row(
                "Target", f"{np.round(np.array(self.args.target), 2)}"
            )
        table_summary.add_row(
            "Best Output", f"{np.round(new_y_best/adjusted_cofficents, 2)}"
        )
        table_summary.add_row("Best Input", f"{np.round(new_x_best, 2)}")
        table_summary.add_row("Total iterations", f"{n_iters}")
        if self.args.energy_estimation:
            table_summary.add_row(
                "Total Sim Energy", format_with_units(total_sim_energy, "J")
            )
            table_summary.add_row(
                "Total Train Energy", format_with_units(total_train_energy, "J")
            )
            table_summary.add_row(
                "Total Pred Energy", format_with_units(total_prediction_energy, "J")
            )
            table_summary.add_row("Total CO₂", format_with_units(total_co2, "g"))
        total_time = time.time() - total_start_time
        table_summary.add_row(
            "Total Time",
            f"{str(datetime.timedelta(seconds=int(total_time))).zfill(8)}",
        )

        # 2) table for args
        table_args = Table(title="Optimization arguments")
        table_args.add_column("Metric", style="bold cyan")
        table_args.add_column("Value", style="bold green", justify="right")
        for key, value in vars(self.args).items():
            if key == "time_limit" and value is not None:
                value = convert_seconds_to_hms(value)
            table_args.add_row(f"{key}", f"{value}")
        if isinstance(self.model, MoEPointNetSystem):
            table_args.add_row("Embdeing dim", f"{self.model.embedding_dim}")
            table_args.add_row("Point cloud size", f"{self.model.n_points} points")
            table_args.add_row("Experts", f"{list(self.model.experts.keys())}")

        console.print(table_summary)
        console.print("\n\n")
        console.print(table)
        console.print("\n\n")
        console.print(table_args)

        # write them to a file
        with open("summary.txt", "a", encoding="utf-8") as f:
            file_console = Console(
                file=f,
                force_terminal=False,  # no colors
                color_system=None,  # no ANSI escapes
                markup=False,  # ignore Rich markup like [bold]
            )
            file_console.print("\n" + "=" * 120)
            file_console.print(table_summary)
            file_console.print("\n\n")
            file_console.print(table)
            file_console.print("\n\n")
            file_console.print(table_args)
            file_console.print("\n\n\n\n")

        # Save/move all files to a directory
        #################################
        files = [
            f"{self.args.jobname}_intermediate_results.txt",
            "summary.txt",
            "log.txt",
        ]
        if self.args.log_file is not None:
            files.append(self.args.log_file)

        for fname in files:
            if os.path.exists(fname):
                try:
                    dest = os.path.join(
                        self.args.result_folder, os.path.basename(fname)
                    )
                    shutil.move(fname, dest)
                except Exception as e:
                    print(f"Failed to move '{fname}': {e}")

        self.dump_simulation_results(
            new_x_best=new_x_best,
            new_y_best=new_y_best,
            adjusted_cofficents=adjusted_cofficents,
            n_iters=n_iters,
            total_sim_energy=total_sim_energy,
            total_train_energy=total_train_energy,
            total_prediction_energy=total_prediction_energy,
            total_co2=total_co2,
            total_time=total_time,
            date=date,
            result=result,
        )
        self.result = result

    def _run_simulation(
        self,
        next_sample: np.ndarray,
        iteration_index: int,
        n_iters: int,
        parallel_index: int,
        n_parallel: int,
        lock=None,
        clean: bool = True,
    ) -> tuple:
        """
        Run a simulation or evaluate an objective function and return the results along with time
        stamps. This method handles both simulation file modification and calling an objective
        function, depending on the provided configuration. Thread safety is ensured using a lock
        if provided.

        Args:
            next_sample (np.ndarray): The configuration for the next simulation or optimization step.
            iteration_index (int): The index of the current iteration in the overall optimization.
            n_iters (int): Total number of iterations in the optimization process.
            parallel_index (int): The index of the parallel process or thread executing this simulation.
            n_parallel (int): Total number of parallel processes or threads being executed simultaneously.
            lock (optional): An optional thread lock to ensure thread-safe operations, particularly for
                file modifications when multiple threads are used.
            clean (bool, optional): Whether to clean up the file after each iteration. Defaults to True.

        Returns:
            tuple: A tuple containing the configuration of the simulation (`next_sample`),
            the result of the simulation or objective function execution (`new_y`), the start
            time of the simulation (`start`), and the end time of the simulation (`end`).
        """
        # Update the file modifier as needed

        prefix = f"{iteration_index+1}"
        suffix = iteration_index + 1
        if n_parallel > 1:
            prefix += f"+{parallel_index}"
            suffix += parallel_index

        self.logger_obj.set_prefix("")
        if self.objective_function is None:
            if lock is not None:
                lock.acquire()  # released later during simulation
                self.logger.debug("Acquired lock")

            self.file_modifier = FileModifier(
                os.path.join(str(self.args.path), str(self.args.jobname) + ".dat"),
                log_level=self.args.log_level,
                prefix=f"Iteration {prefix}/{n_iters}",
            )
            self.logger.debug("File modifier updated\n")

        progress = 100
        # Trigger the simulation or objective function evaluation (dummy version)
        if self.objective_function:
            # If an objective function is provided, use it
            start = time.time()
            new_y = self.objective_function(next_sample)
            end = time.time()
        else:
            # Otherwise, trigger the long-running simulation
            start = time.time()
            sim = Simulation(self.args, next_sample, suffix, n_iters, prefix=prefix)
            new_y = sim.run(
                self.file_modifier,
                lock,
                self.type_filter,
            )
            end = time.time()
            sim_name = f"{self.args.path}/{self.args.jobname}_{suffix}"
            try:
                step, progress = read_last_progress(f"{sim_name}.log")
                self.logger.debug(
                    f"Simulation ended at step {step} with progress {progress}%"
                )
            except Exception as e:
                print(e)
            if clean:
                delete_matching(f"{sim_name}.*", self.logger)

        return next_sample, new_y, start, end, progress

    def _select_best_points(
        self,
        x_sample_space: np.ndarray,
        y_best: np.ndarray,
        n_points: int = 1,
        peak_distance: int = 5,
        selection_strategy: str = "crowding_distance",  # New parameter to switch strategies
        mode="minimization",
    ) -> None:
        """
        Select the best points for multi-objective optimization using different strategies.

        Args:
            x_sample_space (np.ndarray): The input samples where the EI is evaluated.
            y_best (np.ndarray): The best observed value for each objective.
            n_points (int, optional): The number of points to select. Defaults to 1.
            peak_distance (int, optional): Minimum distance between peaks (for 'peak_based' strategy).
            selection_strategy (str): The strategy to use. Options are 'crowding_distance',
                                      'peak_based', or 'highest_sum'.
            mode (str, optional): Optimization mode ("maximization" or "minimization").
        """
        debug = False
        if self.args.energy_estimation:
            energy_estimation_start(device=self.device)

        # --- Case 1: No training data yet → fallback to random sampling ---
        if len(self.x) < 1:
            self.logger.warning(
                "No training data available. Selecting random points from sample space."
            )
            if len(x_sample_space) < n_points:
                n_points = len(x_sample_space)

            random_indices = np.random.choice(
                len(x_sample_space), size=n_points, replace=False
            )
            self.next_samples = x_sample_space[random_indices]
            self.x_sample_space = x_sample_space
            self.ei = np.zeros(self.y.shape[1])
            self.ei_sum = 0
            if self.args.energy_estimation:
                self.prediction_energy = energy_estimation_end()
            return  # skip EI and Pareto selection logic

        # --- Case 2: Normal path when model is trained ---
        mask = np.ones(len(x_sample_space), dtype=bool)
        self.logger.info("1. Applying mask to remove duplicated points in search space")
        if self.x.size > 0:
            for i, candidate in enumerate(x_sample_space):
                # If candidate is close to any existing point, mark as False
                if np.any(
                    np.all(np.isclose(self.x, candidate, atol=1e-6, rtol=1e-5), axis=1)
                ):
                    mask[i] = False

        # Apply the mask
        x_sample_space_filtered = x_sample_space[mask]
        #  fallback if not enough points remain
        if len(x_sample_space_filtered) < n_points:
            x_sample_space_filtered = x_sample_space

        # 1. Compute the Expected Improvement (EI) values for all objectives
        # the size of the x_space can become a problem. To compensate for memory constraints
        # evaluate the model in batches inside the AcquisitionFunction
        self.logger.info("2. Computing expected improvement values")
        ei = AcquisitionFunction.expected_improvement(
            x_sample_space_filtered,
            self.model,
            y_best,
            mode=mode,
            n_samples_mc=self.args.n_samples_mc,
            full_cov=self.args.full_cov,
        )
        if debug:
            AcquisitionFunction.plot_ei(x_sample_space_filtered, ei)

        # 2. Multi-Objective Filtering: Use Pareto frontier on the EI values
        # We always maximize EI, regardless of the original problem's 'mode'.
        self.logger.info("3. Computing pareto frontier on the EI values")
        if len(x_sample_space_filtered) > 10_000:
            self.logger.warning(
                "Too many points to compute the Pareto frontier. Skipping"
            )
            pareto_idx = []
        else:
            pareto_idx = pareto_frontier(ei, mode="maximization")
        if debug:
            plot_pareto(ei, pareto_idx)
        # Candidate points are those on the Pareto front of the EI space
        # This ensures your next samples are promising in all objectives, not just in one.
        if len(pareto_idx) < n_points:
            self.logger.warning(
                "No Pareto points found. Falling back to highest summed EI."
            )
            x_candidates = x_sample_space_filtered
            ei_candidates = ei
        else:
            x_candidates = x_sample_space_filtered[pareto_idx]
            ei_candidates = ei[pareto_idx]

        ei_sum_candidates = ei_candidates.sum(axis=1)

        # --- Selection Logic based on the chosen strategy ---

        self.logger.info(
            f"4. Selecting {n_points} best points using {selection_strategy}"
        )
        selected_indices_in_candidates = []

        if selection_strategy == "crowding_distance":
            # Strategy 1: Balance exploitative (highest sum) and exploration (crowding distance)
            # if n_points = 1, it's simply exploitative, and the exploration comes from
            # BO iterative nature (future ei are suppressed around sampled point).
            # Once n_Points is larger than one exploration happens more significantly

            # Get the index of the best point by summed EI
            best_sum_idx = np.argmax(ei_sum_candidates)
            # append it, if no other point close to it has been selected yet
            if not np.any(
                np.all(
                    np.isclose(
                        self.x, x_candidates[best_sum_idx], atol=1e-2, rtol=1e-5
                    ),
                    axis=1,
                )
            ):
                self.logger.info(
                    f"5. Max EI used for selecting optimal point: {x_candidates[best_sum_idx]}"
                )
                selected_indices_in_candidates.append(best_sum_idx)
            else:
                self.logger.debug(
                    "Best point already selected. Skipping exampling this point. Using crowling distance instead"
                )

            if len(selected_indices_in_candidates) < n_points:
                # Calculate crowding distance for diversity
                self.logger.info(
                    f"6. Finding {n_points - len(selected_indices_in_candidates)} points with crowding distance"
                )
                crowding_distances = AcquisitionFunction.crowding_distance(
                    ei_candidates
                )

                # Sort candidates by crowding distance in descending order
                diverse_indices = np.argsort(crowding_distances)[::-1]

                # Add diverse points until n_points is reached
                for idx in diverse_indices:
                    if len(selected_indices_in_candidates) >= n_points:
                        break
                    if idx not in selected_indices_in_candidates:
                        selected_indices_in_candidates.append(idx)

        elif selection_strategy == "peak_based":
            # Strategy 2: peak-finding approach
            peaks, _ = find_peaks(ei_sum_candidates, distance=peak_distance)

            if len(peaks) == 0:
                # Fallback to selecting the highest sum points if no peaks are found
                top_indices_in_candidates = np.argsort(ei_sum_candidates)[-n_points:]
                selected_indices_in_candidates = list(top_indices_in_candidates)
            else:
                global_extreme_idx = np.argmax(ei_sum_candidates)
                ranked_peaks = np.argsort(ei_sum_candidates[peaks])[::-1]

                # Prioritize the global maximum
                selected_indices_in_candidates.append(global_extreme_idx)

                # Add top ranked peaks
                for peak_idx in ranked_peaks:
                    candidate_idx = peaks[peak_idx]
                    if candidate_idx not in selected_indices_in_candidates:
                        selected_indices_in_candidates.append(candidate_idx)
                    if len(selected_indices_in_candidates) >= n_points:
                        break

        elif selection_strategy == "highest_sum":
            # Strategy 3: A simple, greedy approach
            top_indices_in_candidates = np.argsort(ei_sum_candidates)[-n_points:]
            selected_indices_in_candidates = list(top_indices_in_candidates)

        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")

        # Final selection and assignment
        num_to_select = min(n_points, len(selected_indices_in_candidates))
        next_samples = x_candidates[selected_indices_in_candidates[:num_to_select]]

        self.next_samples = next_samples
        self.x_sample_space = x_sample_space_filtered
        self.ei = ei
        self.ei_sum = ei.sum()  # The sum over the entire sample space
        if self.args.energy_estimation:
            self.prediction_energy = energy_estimation_end()

    def _human_guided_al(
        self,
        x_sample_space: np.ndarray,
        n_points: int = 1,
        index: int = 0,
    ) -> None:
        """
        Select the best points based on the Expected Improvement (EI) criterion.

        Args:
            x_sample_space (np.ndarray): The input samples where the EI is evaluated.
            y_best (np.ndarray): The maximum/minimum observed value from the GPR model.
            n_points (int, optional): The number of points to select. Defaults to 1.
            mode (str, optional): "maximization" or "minimization", determines the objective to optimize.
            index (int, optional): The index of the parallel process or thread executing this simulation.
        Returns:
            None.
        """
        # Initialize a list to store the user-selected points
        user_points = []

        # Let EI show the user first the best option
        if index != 0:
            self.plot()
        else:
            self.plot(False)

        for i in range(n_points):
            # Prompt the user for input, expect a value for each dimension of x_sample_space
            if n_points > 1:
                prompt = f"Enter for the {i+1}-th parallel execution the samples separated by spaces (e.g., "
            else:
                prompt = "Enter the samples separated by spaces (e.g., "

            prompt += (
                " ".join([f"x{j+1}" for j in range(x_sample_space.shape[1])]) + "): "
            )
            user_input = input(prompt)

            # If the input is not empty, convert to a numpy array and append to the user_points list
            if user_input:
                user_point = np.array([float(val) for val in user_input.split()])
                if len(user_point) != x_sample_space.shape[1]:
                    raise ValueError(
                        f"Each input must have {x_sample_space.shape[1]} values corresponding to the dimensions of the x_sample_space."
                    )
                user_points.append(user_point)
            else:
                # Fallback: Use the default method if no input is provided (points have been already calculated)
                return None

        # Convert the list of user points to a numpy array for the final result (overwrites previous results)
        self.ei_sum = 0.0
        self.next_samples = np.array(user_points)

    def plot(
        self,
        show_last_point: bool = True,
        reread: bool = False,
        fix_other_dims: bool = False,
        save_path: str = None,
    ) -> None:
        """
        Plots the Gaussian Process (GP) mean and confidence interval along with the true objective function.
        This method visualizes the GP predictions, including the mean and 95% confidence interval,
        for each dimension of the input and output data. It also optionally highlights the last
        sampled point and overlays the true objective function if available.
        Args:
            show_last_point (bool, optional): Whether to highlight the last sampled point in the plot.
                                            Defaults to True.
            reread (bool, optional): Whether to reread the content of the output file, or plot x and y from memory
            fix_other_dims (bool, optional): if True, fix other input dims to mean to plot clean 1D slice vs single dim
                                    Defaults to False
            save_path (str, optional): The path to save the plot. Defaults to None.
        Raises:
            ValueError: If the dimensions of `self.x` or `self.y` are incompatible with the plotting logic.
        """
        """Plots the GP mean and confidence interval along with the true objective function."""

        # plotting only makes sense, if the x space was generated with a linear point_selection method.
        # In combination there should be a distinct line for each fixed dimension.
        coeffs_safe = np.where(
            self.args.attention_coefficients == 0,
            1e12,
            self.args.attention_coefficients,
        )

        if reread and self.objective_function is None:
            if self.type_filter:
                # filter by type
                x, y, types = read_data_type(
                    self.args.output,
                    self.args.x_fields,
                    self.args.y_fields,
                    log_level="INFO",
                )
                x = x[types == self.args.jobname]
                y = y[types == self.args.jobname]
            else:
                x, y = read_data(
                    self.args.output,
                    self.args.x_fields,
                    self.args.y_fields,
                    log_level="INFO",
                )
        else:
            x = self.x
            y = self.y
            y = y / coeffs_safe  # this Y was not scaled

        if self.x_sample_space is None:  # no optimization was performed, only plotting
            self.logger.info(
                "No optimization was performed, creating x space for plotting"
            )
            search_space = SearchSpace(
                self.args.x_fields, log_level=self.args.log_level
            )
            self.x_sample_space = search_space.create_sample_space(
                x=self.x,
                n_samples=self.args.x_space_n_samples * 10,
                expansion_factor=self.args.x_space_expansion_factor,
                precision=self.args.x_space_precision,
                method=self.args.x_space_structure,
                point_creation=self.args.x_space_point_creation,
            )
            self.ei = AcquisitionFunction.expected_improvement(
                self.x_sample_space,
                self.model,
                np.min(self.y, axis=0),
                mode="minimization",
                n_samples_mc=self.args.n_samples_mc,
                full_cov=self.args.full_cov,
            )
        # 1) plot ei:
        self.logger.debug("Creating EI plot")
        AcquisitionFunction.plot_ei(
            self.x_sample_space,
            self.ei,
            self.args.x_fields,
            self.args.y_fields,
            next_samples=self.next_samples,
            save_path=save_path,
        )

        if len(self.x_sample_space) < 100_000:
            pareto_idx = pareto_frontier(self.ei, mode="maximization")
            plot_pareto(self.ei, pareto_idx, save_path)

        # 2) Calculate the true objective function values
        if self.objective_function:
            y_true = self.objective_function(self.x_sample_space)
        else:
            y_true = np.array([])

        # 3) plot GPR
        self.logger.debug("Creating GPR plot")

        sns.set_theme(style="whitegrid")  # cleaner plots
        # colors = sns.color_palette("YlOrRd", 5).as_hex()[::-1]
        colors = sns.color_palette("RdYlGn", 5).as_hex()
        alphas = np.linspace(0.7, 0.9, 5)[::-1]
        max_rows = 3
        n_cols = int(np.ceil(y.shape[1] / max_rows))

        for dim in range(x.shape[1]):
            self.logger.debug(f"Plotting dimension {dim}")
            fig = plt.figure(figsize=(10 * n_cols, 4 * max_rows))
            if fix_other_dims:
                # Fix all other dims to their mean values except the chosen dim
                fixed_x = np.mean(self.x_sample_space, axis=0)
                n_points = self.x_sample_space.shape[0]
                x_grid = np.tile(fixed_x, (n_points, 1))
                x_grid[:, dim] = self.x_sample_space[:, dim]

                # Predict
                mu_grid, cov_or_std_grid = self.model.predict(
                    x_grid, return_cov=self.args.full_cov
                )
                if self.args.attention_coefficients is not None:
                    mu_grid = mu_grid / coeffs_safe
                    if self.args.full_cov:
                        # scale full covariance
                        # cov_or_std_grid: shape (n_points, m, m)
                        for i in range(cov_or_std_grid.shape[0]):
                            cov_or_std_grid[i] = (
                                cov_or_std_grid[i]
                                / coeffs_safe[:, None]
                                / coeffs_safe[None, :]
                            )
                    else:
                        cov_or_std_grid = cov_or_std_grid / coeffs_safe

                mu_grid = mu_grid.reshape(-1, y.shape[1])
                for y_dim in range(y.shape[1]):
                    plt.subplot(max_rows, n_cols, y_dim + 1)
                    if self.objective_function is not None:
                        sns.lineplot(
                            x=x_grid[:, dim],
                            y=y_true[:, y_dim],
                            linestyle="--",
                            color="red",
                            label=f"True Objective Function (dim {y_dim + 1})",
                        )

                    if self.args.full_cov:
                        # Monte Carlo sampling from multivariate Gaussian
                        samples = np.random.multivariate_normal(
                            mean=mu_grid[:, y_dim],
                            cov=cov_or_std_grid[
                                :, y_dim, y_dim
                            ],  # slice per output if needed
                            size=self.args.n_samples_mc,
                        )  # shape (n_samples_mc, n_points)
                        mu_plot = samples.mean(axis=0)
                        lower = np.quantile(samples, 0.025, axis=0)
                        upper = np.quantile(samples, 0.975, axis=0)
                    else:
                        mu_plot = mu_grid[:, y_dim]
                        lower = mu_plot - 1.96 * cov_or_std_grid[:, y_dim]
                        upper = mu_plot + 1.96 * cov_or_std_grid[:, y_dim]

                    sns.lineplot(
                        x=x_grid[:, dim],
                        y=mu_plot,
                        color="blue",
                        label=f"GP Mean Prediction (dim {y_dim + 1})",
                    )
                    plt.fill_between(
                        x_grid[:, dim],
                        lower,
                        upper,
                        color="blue",
                        alpha=0.2,
                        label="95% Confidence Interval",
                    )

                    plt.scatter(
                        x[:, dim],
                        y[:, y_dim],
                        c="black",
                        marker="x",
                        label="Initial Samples",
                        alpha=0.2,
                    )

                    if show_last_point:
                        num_points = min(5, x.shape[0])
                        start_idx = x.shape[0] - num_points
                        for i in range(num_points):
                            plt.scatter(
                                x[start_idx + i, dim],
                                y[start_idx + i, y_dim],
                                c=colors[i],
                                marker="o",
                                s=80,
                                edgecolors="black",
                                linewidths=0.8,
                                alpha=alphas[i],
                                label=f"Point #{start_idx + i}",
                            )

                    plt.xlabel(f"{self.args.x_fields[dim]}")
                    plt.ylabel(f"{self.args.y_fields[y_dim]}")
                    plt.title(
                        f"{self.args.x_fields[dim]} vs. {self.args.y_fields[y_dim]}"
                    )
                    plt.legend(ncol=2)
            else:
                # Predict the mean and variance/covariance
                mu, cov_or_std = self.model.predict(
                    self.x_sample_space, return_cov=self.args.full_cov
                )

                # Apply attention coefficients scaling if present
                if self.args.attention_coefficients is not None:
                    mu = mu / coeffs_safe
                    if self.args.full_cov:
                        # scale full covariance matrices
                        for i in range(cov_or_std.shape[0]):
                            cov_or_std[i] = (
                                cov_or_std[i]
                                / coeffs_safe[:, None]
                                / coeffs_safe[None, :]
                            )
                    else:
                        cov_or_std = cov_or_std / coeffs_safe

                mu = mu.reshape(-1, y.shape[1])

                for y_dim in range(y.shape[1]):
                    plt.subplot(max_rows, n_cols, y_dim + 1)
                    x_range = self.x_sample_space[:, dim]

                    if self.objective_function is not None:
                        sns.lineplot(
                            x=x_range,
                            y=y_true[:, y_dim],
                            linestyle="--",
                            color="red",
                            label=f"True Objective Function (dim {y_dim + 1})",
                        )

                    if self.args.full_cov:
                        # Monte Carlo sampling for confidence interval
                        samples = np.random.multivariate_normal(
                            mean=mu[:, y_dim],
                            cov=(
                                cov_or_std[:, y_dim, y_dim]
                                if cov_or_std.ndim == 3
                                else cov_or_std
                            ),
                            size=self.args.n_samples_mc,
                        )  # shape: (n_samples_mc, n_points)
                        mu_plot = samples.mean(axis=0)
                        lower = np.quantile(samples, 0.025, axis=0)
                        upper = np.quantile(samples, 0.975, axis=0)
                    else:
                        mu_plot = mu[:, y_dim]
                        sigma = cov_or_std[:, y_dim]
                        lower = mu_plot - 1.96 * sigma
                        upper = mu_plot + 1.96 * sigma

                    sns.lineplot(
                        x=x_range,
                        y=mu_plot,
                        color="blue",
                        label=f"GP Mean Prediction (dim {y_dim + 1})",
                    )
                    plt.fill_between(
                        x_range,
                        lower,
                        upper,
                        color="blue",
                        alpha=0.2,
                        label="95% Confidence Interval",
                    )

                    # Plot initial samples
                    plt.scatter(
                        x[:, dim],
                        y[:, y_dim],
                        c="black",
                        marker="x",
                        label="Initial Samples",
                        alpha=0.2,
                    )

                    # Highlight last points if requested
                    if show_last_point:
                        num_points = min(5, x.shape[0])
                        start_idx = x.shape[0] - num_points
                        for i in range(num_points):
                            plt.scatter(
                                x[start_idx + i, dim],
                                y[start_idx + i, y_dim],
                                c=colors[i],
                                marker="o",
                                s=80,
                                edgecolors="black",
                                linewidths=0.8,
                                alpha=alphas[i],
                                label=f"Point #{start_idx + i}",
                            )

                    plt.xlabel(f"{self.args.x_fields[dim]}")
                    plt.ylabel(f"{self.args.y_fields[y_dim]}")
                    plt.title(
                        f"{self.args.x_fields[dim]} vs. {self.args.y_fields[y_dim]}"
                    )
                    plt.legend(ncol=2)

            # saved figure
            if save_path is not None:
                os.makedirs(
                    os.path.dirname(save_path), exist_ok=True
                )  # ensure folder exists
                name = f"gpr_{dim}" if fix_other_dims else f"gpr_{dim}_fixed"
                plt.savefig(
                    f"{save_path}/{name}.png",
                    bbox_inches="tight",
                )  # save figure to file
                self.logger.info(f"Plot saved to {save_path}")
                with open(
                    f"{save_path}/{name}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(fig, f)

        if save_path is None:
            plt.show()

    def dump_simulation_results(
        self,
        new_x_best,
        new_y_best,
        adjusted_cofficents,
        n_iters,
        total_sim_energy=None,
        total_train_energy=None,
        total_prediction_energy=None,
        total_co2=None,
        total_time=None,
        date=None,
        result=None,
    ):
        """Append simulation results to a JSON log file, consistent with table_args logic."""

        # Build args dict as in table_args
        args_dict = {}
        for key, value in vars(self.args).items():
            if key == "time_limit" and value is not None:
                value = convert_seconds_to_hms(value)
            args_dict[key] = to_json_serializable(value)

        # Add MoEPointNetSystem-specific fields if applicable
        if isinstance(self.model, MoEPointNetSystem):
            args_dict["Embedding dim"] = self.model.embedding_dim
            args_dict["Point cloud size"] = f"{self.model.n_points} points"
            args_dict["All experts"] = f"{list(self.model.experts.keys())}"
            args_dict["parts"] = [list(p) for p in self.parts] if self.parts else None

        jobname = self.args.jobname if self.args.jobname else "dummy"
        # Create a dictionary with the result
        sim_data = {
            "jobname": jobname,
            "date": date if date else "",
            "target": (
                np.array(self.args.target).tolist()
                if self.args.target is not None
                else None
            ),
            "best_output": (new_y_best / adjusted_cofficents).tolist(),
            "best_input": new_x_best.tolist(),
            "total_iterations": int(n_iters),
            "total_sim_energy": (
                float(total_sim_energy) if total_sim_energy is not None else None
            ),
            "total_train_energy": (
                float(total_train_energy) if total_train_energy is not None else None
            ),
            "total_prediction_energy": (
                float(total_prediction_energy)
                if total_prediction_energy is not None
                else None
            ),
            "co2": float(total_co2) if total_co2 is not None else None,
            "total_time": (
                f"{str(datetime.timedelta(seconds=int(total_time))).zfill(8)}"
                if total_time
                else None
            ),
            "total_time_in_seconds": int(total_time) if total_time else None,
            "args": args_dict,
        }

        # Merge results directly into sim_data
        # Merge results safely
        if result is not None:
            result_safe = to_json_serializable(result)
            sim_data.update(result_safe)

        # Append to JSON file
        if self.args.input is not None:
            json_file = f"{jobname}_no_optimization.json"
        else:
            json_file = f"{jobname}.json"
        # Read existing file or initialize empty dict
        if os.path.exists(json_file):
            try:
                with open(json_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                # File is empty or corrupted
                data = {}
        else:
            data = {}

        sim_number = len(data) + 1
        sim_key = f"sim_{sim_number:02d}"
        data[sim_key] = sim_data

        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Saved results as {sim_key} in '{json_file}'")

    def print_args(self):
        for key, value in self.__dict__.items():
            self.logger.debug(f"{key}: {value}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  jobname='{self.args.jobname}',\n"
            f"  path='{self.args.path}',\n"
            f"  output='{self.args.output}',\n"
            f"  ofsolver='{self.args.ofsolver}',\n"
            f"  openform='{self.args.openform}',\n"
            f"  session='{self.args.session}',\n"
            f"  cores={self.args.cores},\n"
            f"  method='{self.args.method}',\n"
            f"  end_condition='{self.args.end_condition}',\n"
            f"  end_value={self.args.end_value},\n"
            f"  x_dim={self.x_dim},\n"
            f"  y_dim={self.y_dim},\n"
            f"  objective_function={'set' if self.objective_function else 'None'},\n"
            f"  Model info={self.model}\n"
            f")"
        )

    def __str__(self):
        out = ["Baysian optimization Parameters:"]
        for key, value in self.__dict__.items():
            out.append(f"{key}: {value}")
        return "\n".join(out)

    def __getstate__(self):
        # exclude model from being pickled (needed for gpytorch)
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
