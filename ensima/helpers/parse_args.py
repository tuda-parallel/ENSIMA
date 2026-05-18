"""
Argument Parser for Bayesian Optimization Execution.

This script provides a function to parse command-line arguments or accept a list of arguments
as input. It is designed to configure and run Bayesian Optimization with various optional
parameters, including OpenFOAM solver paths, job settings, and execution configurations.

If no argument list is provided, the function defaults to using command-line arguments (sys.argv).

Functions:
    parse_arguments(arg_list=None): Parses the given argument list or sys.argv.

Usage:
    1. From the command line:
        python optimize.py --jobname test --cores 8

    2. Within Python:
        args = parse_arguments(["--jobname", "test", "--cores", "8"])

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
import os
import re
import sys
from argparse import Namespace

from ensima.classes.logger import set_log_file


def parse_arguments(arg_list: object = None) -> Namespace:
    """Parses command-line arguments and returns them.

    Adds optional arguments to customize the Bayesian Optimization execution.
    If no arguments are provided, default values will be used.

    Args:
        arg_list (list, optional): A list of arguments to parse. If None, sys.argv[1:] is used.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """

    if arg_list is None:
        arg_list = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization with optional parameters."
    )

    # Add your fields
    parser.add_argument(
        "--x_fields",
        nargs="+",
        default=["Fr", "p", "D"],
        help="Input variable field names. Example: --x_fields Fr p D",
    )
    parser.add_argument(
        "--y_fields",
        nargs="+",
        default=["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
        help="Output variable field names. Example: --y_fields L1 L2 L3 L4 L5 L6 L7",
    )
    parser.add_argument(
        "--attention_coefficients",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Attention coefficients for each output variable. "
            "Example: --attention_coefficients 1 0.8 -0.01 -0.01 0.7 0.8 1"
        ),
    )

    parser.add_argument(
        "--approximate_computing_check",
        nargs="+",
        type=float,
        default=None,
        help=(
            "A list of float values (between 0.0 and 1.0) used for checking "
            "the approximation of a computing step. The check terminates the "
            "execution if the approximation exceeds these limits. "
            "Example: --approximate_computing_check 0.95 0.90"
        ),
    )
    parser.add_argument(
        "--approximate_computing_limit",
        nargs="+",
        type=float,
        default=None,
        help=(
            "A list of integer values (between 0 and 100) that serve as a "
            "threshold for each output. If the measured output exceeds any "
            "of these limits, the simulation will be terminated prematurely. "
            "Note: 100 skips the check for the variable."
            "Example: --approximate_computing_limit 90 85 75"
        ),
    )

    parser.add_argument(
        "--target",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional list of integer target values per column. "
            "Columns with 100 are skipped in distance-based fallback. "
            "Example: --target 0 20 50 100 10 0 0"
        ),
    )

    # parameter constraints
    parser.add_argument(
        "--user_constraints",
        default=None,
        type=json.loads,
        help='Constraints as JSON string, e.g. \'{"Rp": [270, 420, 890], "D": [0.8, 1.8], "Fr": [0.01, 0.2]}\'',
    )

    # Arguments for OpenFOAM solver and related options
    parser.add_argument(
        "--ofsolver",
        "-ofs",
        type=str,
        default="/rwthfs/rz/cluster/home/rwth1453/OFSolv/Software/OFSolv_V2.16.0-E/bin/OFSolv_V2.16.0-E_amd64.exe",
        help="Path to the OpenFOAM solver executable.",
    )
    parser.add_argument(
        "--jobname", "-j", type=str, default=None, help="Name of the optimization job."
    )
    parser.add_argument(
        "--cores", "-c", type=int, default=4, help="Number of CPU cores allocated."
    )
    parser.add_argument(
        "--energy_estimation",
        "-e",
        action="store_true",
        default=False,
        help="Energy estimation if True",
    )
    parser.add_argument(
        "--openform",
        "-ofm",
        type=str,
        default="/home/cg021604/rwth1453/OpenForm_daily_linux64/OpenForm_64_batch",
        help="Path to the OpenForm software.",
    )
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        default="session.ofs",
        help="OpenForm session file.",
    )

    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="./",
        help="Path to job directory. This dir must contain the .dat file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Absolute path to the output (.csv) file that stores the result from all runs.",
    )

    parser.add_argument(
        "--result_folder",
        type=str,
        default=None,
        help="Absolute path to the result folder that stores the result from all iterations.",
    )

    parser.add_argument(
        "--geometry_path",
        "-g",
        type=str,
        default=None,
        help="Absolute path to the geometry_path (.t52) file that stores the geometry.",
    )

    parser.add_argument(
        "--end_condition",
        "-ec",
        type=str,
        default="no_improvement",
        choices=["no_improvement", "constant_min", "energy_budget"],
        help=(
            "Condition to terminate the optimization process. Options:\n"
            "  - no_improvement: Stop if the sum of expected improvement (EI) does not change for "
            "X consecutive iterations.\n"
            "  - constant_min: Stop if the best observed output (y_best) remains unchanged for "
            "X consecutive iterations (may get stuck in local minima).\n"
            "  - energy_budget: Stop if the estimated total energy consumption exceeds X.\n"
            "X is specified via --end_value."
        ),
    )

    parser.add_argument(
        "--end_value",
        "-ev",
        type=float,
        default=5,
        help=(
            "Threshold for the selected --end_condition:\n"
            "  - For 'no_improvement' or 'constant_min', this is the number of consecutive iterations to trigger termination (int).\n"
            "  - For 'energy_budget', this is the total energy limit in joules (float)."
        ),
    )

    parser.add_argument(
        "--time_limit",
        "-tl",
        type=parse_time_limit,
        default=None,
        help=(
            "specify limit of each simulation as HH:MM:SS pattern"
            "It allows for single or double digits for each part."
        ),
    )

    # Argument for the plot flag (default is False)
    parser.add_argument(
        "--plot",
        "-pl",
        action="store_true",
        help="Flag to enable plotting of the results. Default is False.",
    )
    parser.add_argument(
        "--save_and_load",
        "-sal",
        action="store_true",
        help="Flag to save and load the results. Default is False.",
    )

    # Argument for the number of iterations
    parser.add_argument(
        "--iterations",
        "-it",
        type=int,
        default=20,
        help="Number of iterations for the optimization process.",
    )

    # Argument for the number of parallel samples
    parser.add_argument(
        "--parallel_samples",
        "-ps",
        type=int,
        default=1,
        help="Number of parallel samples to evaluate in each iteration.",
    )

    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="bo",
        help=(
            "Optimization method to use. Supported methods are:\n"
            "1) Bayesian Optimization ('bo'): A method where Bayesian optimization "
            "autonomously selects the next simulation point using an acquisition function, and the "
            "simulation result is used to update the Gaussian Process model iteratively.\n"
            "2) Human-Guided Active Learning ('hgal'): A process where an expert actively selects the next simulation point "
            "based on intuition or domain knowledge, and the results are used to train the surrogate model."
        ),
    )

    # X -space options:
    #####################
    parser.add_argument(
        "--x_space_point_creation",
        type=str,
        default="combination",
        choices=["linear", "combination"],
        help=(
            "Strategy to create points in the sample space. "
            "'linear' generates simple grids along each dimension (faster, may miss regions). "
            "'combination' creates Cartesian products of all dimension samples (denser, more memory)."
        ),
    )

    parser.add_argument(
        "--x_space_n_samples",
        type=lambda s: [int(x) for x in s.split(",")] if "," in s else int(s),
        default=10_000,
        help=(
            "Specifies the sampling density of the search space. "
            "For scalar input (e.g., '1000'), this value is used as the number of samples "
            "per dimension (for 'grid' and 'linear') or as total samples (for 'random' and 'combination'). "
            "For 'combination' with 'grid', the total number of points (Cartesian product) is limited to x_dims * this value. "
            "Alternatively, a comma-separated list (e.g., '20,30,40') may be provided to define "
            "per-dimension sample counts explicitly. List mode is only valid when '--x_space_point_creation combination' is selected."
        ),
    )

    parser.add_argument(
        "--x_space_structure",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help=(
            "Method to generate points along each dimension of the sample space. "
            "'grid' creates evenly spaced points, while 'random' selects points randomly "
            "within the expanded bounds. This applies to all point creation strategies "
            "('linear' or 'combination'), controlling only the per-dimension sampling."
        ),
    )

    parser.add_argument(
        "--x_space_expansion_factor",
        type=float,
        default=0.1,
        help=(
            "Fractional expansion of the observed min/max per continuous variable when "
            "creating the sample space. Ensures the sampled space slightly exceeds the observed range."
        ),
    )
    parser.add_argument(
        "--x_space_precision",
        type=int,
        default=3,
        help="Number of decimal places to round continuous variables (default: 2).",
    )

    # MOE expert assignment precision
    ###############################
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="downsample",
        choices=["downsample", "upsample"],
        help=(
            "Strategy to unify the number of points in each point cloud: "
            "'downsample' reduces larger point clouds to match n_points (defaults to min"
            " points over all geometric points, if n_point is not specified). This mode "
            " randomly selects n_points. "
            "'upsample' increases smaller point clouds to match n_points (defaults to max"
            "points over all geometric points, if n_point is not specified). This mode "
            "deterministically increases the points till n_points"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help=(
            "[PyTorch backend only] Number of training epochs for the GP model. "
            "This controls how long the optimizer runs. More epochs may be required "
            "for complex multitask models with latent dimensions."
        ),
    )

    parser.add_argument(
        "--latent_input_dim",
        type=int,
        default=None,
        help=(
            "[PyTorch backend only] Dimensionality of the latent input space for the "
            "multitask GP model. If None, the model uses the original input dimension. "
            "Set this to a positive integer to project inputs into a latent space before "
            "modeling correlations between tasks. Defaults to twice the size of x"
        ),
    )

    parser.add_argument(
        "--latent_output_dim",
        type=int,
        default=None,
        help=(
            "[PyTorch backend only] Dimensionality of the latent output space for the "
            "multitask GP model. If None, defaults to the number of tasks. "
            "Set this to a positive integer to compress/expand the task space into a "
            "latent representation. Defaults to size of y"
        ),
    )

    parser.add_argument(
        "--log_level",
        "-l",
        default=None,
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Environment variable LOG_LEVEL takes precedence.",
    )

    parser.add_argument(
        "--log_file",
        "-lf",
        default=None,
        help="absolute path to log file (default logs/{jobname}.log)",
    )

    parser.add_argument(
        "--watcher_backend",
        "-wb",
        default="rich",
        choices=["logger", "rich"],
        help="Choose the progress watcher backend: 'rich' for Rich progress bars, 'logger' for simple logging output.",
    )

    parser.add_argument(
        "--license_server_service",
        action="store_true",
        help="If the Licence server is installed as a service, pass this flag",
    )

    parser.add_argument(
        "--license_server",
        default="electric",
        help="License server IP address",
    )
    parser.add_argument(
        "--license_port",
        default="5053",
        help="License port IP address",
    )
    parser.add_argument(
        "--license_type",
        default="RLM",
        help="License server type address",
    )
    parser.add_argument(
        "--license_rlm",
        default="/d/gitlab/ensima-code/test_data/gns/gns_rlm_v15.2_R1_install_linux64/rlm ",
        help="Absolute path of license server executable.",
    )

    parser.add_argument(
        "--selection_strategy",
        "-ss",
        type=str,
        default="crowding_distance",
        choices=["crowding_distance", "peak_based", "highest_sum"],
        help=(
            "Which method is used for selecting the next samples from the Pareto front. "
            "'crowding_distance' balances exploitation and exploration on the Pareto front. "
            "'peak_based' finds local maxima in the summed EI. "
            "'highest_sum' greedily selects the points with the highest summed EI."
            "Note:\n"
            "When `n_points` is 1, the strategy is **purely exploitative**. It selects the\n"
            "single point with the highest summed EI, promising the greatest overall\n"
            "improvement. Exploration is naturally handled by the iterative nature of BO;\n"
            "once a point is sampled, the model's uncertainty in that area drops,\n"
            "suppressing the EI and forcing the algorithm to seek a new, promising region in\n"
            "the next iteration.\n\n"
            "When `n_points` is greater than 1, exploration becomes a more significant and\n"
            "explicit part of the selection strategy. The method first selects the point with\n"
            "the highest summed EI (exploitation) and then fills the rest of the batch by\n"
            "choosing points with the highest crowding distance (exploration). This ensures\n"
            "the selected batch is both promising and diverse, preventing samples from\n"
            "clustering together and maximizing the information gained in each parallel\n"
            "evaluation.\n\n"
        ),
    )
    parser.add_argument(
        "--full_cov",
        type=bool,
        default=False,
        help="Whether to use the full covariance matrix for Monte Carlo EI estimation.",
    )

    parser.add_argument(
        "--n_samples_mc",
        type=int,
        default=10000,
        help="Number of Monte Carlo samples to draw when computing EI with full covariance.",
    )

    parser.add_argument(
        "--new_expert_start",
        type=int,
        default=2,
        help=(
            "Specifies the starting iteration to stop using the MOE and training a new expert."
            "This is particularly useful in multi-expert or mixture-of-experts (MoE)"
            "models. The value must be an integer and cannot be less than 2."
        ),
    )

    parser.add_argument(
        "--input",
        type=json.loads,
        default=None,
        help=(
            "List of options x points to executed. Do not use this option if you want to use optimization."
            "This is just a lazy way to execute the simulation at the desired points. "
            "Example: --input '[[0,20,50],[100,10,0],[5,5,5]]'"
        ),
    )

    args = parser.parse_args(arg_list)
    # set log path and result folder
    args = init_save_and_log(args)
    return args


def parse_time_limit(time_string):
    """
    Parses a time string in the format HH:MM:SS or H:M:S into total seconds.

    This function is designed to be used as a type for argparse. It raises
    a ValueError if the format is invalid.
    """
    # Use a regular expression to match the HH:MM:SS pattern
    # It allows for single or double digits for each part.
    pattern = re.compile(r"^(?:(\d{1,2}):)?(?:(\d{1,2}):)?(\d{1,2})$")
    match = pattern.match(time_string)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid time format: '{time_string}'. Expected format is HH:MM:SS."
        )

    # Extract hours, minutes, and seconds from the regex match
    hours, minutes, seconds = [int(x) if x else 0 for x in match.groups()]

    # Validate that the values are within a valid range
    if minutes > 59 or seconds > 59:
        raise argparse.ArgumentTypeError(
            f"Invalid time format: '{time_string}'. Minutes and seconds must be between 0 and 59."
        )

    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    return total_seconds


def init_save_and_log(args: Namespace):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = ""
    if args.jobname:
        name = f"{args.jobname}_"

    if args.log_file is None:
        args.log_file = f"logs/{name}{timestamp}.log"
    if args.result_folder is None:
        args.result_folder = f"{name}{timestamp}"
        if args.input is not None:
            args.result_folder += "_no_optimization"

    args.result_folder = os.path.abspath(args.result_folder)
    args.log_file = os.path.abspath(args.log_file)
    os.makedirs(args.result_folder, exist_ok=True)
    set_log_file(args.log_file)
    return args
