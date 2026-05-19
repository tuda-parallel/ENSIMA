"""
Script used to produce the MLVGP simulation results in artifacts/JIMS/sim_results/MLVGP/.

Runs single-part Bayesian optimization (MLVGP) on SeatShell, filtering the labeled
CSV to the target part. Executed on a GPU cluster — set -ofs and -ofm to your local
OpenForm solver and software paths before running.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

from ensima.helpers.adjust_args_cluster import adjust_args_for_cluster
from ensima.helpers.parse_args import parse_arguments
from ensima.optimize import main

if __name__ == "__main__":
    args = parse_arguments(
        [
            "-ofs",
            "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
            "-ofm",
            "/d/gitlab/ensima-code/test_data/gns/OpenForm_daily_linux64/OpenForm_64_batch",
            "-p",
            # "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_04",
            # "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_01_Flat",
            # "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_01",
            # "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_02",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_03",
            "-j",
            # "Einleger",
            # "ASaeule",
            # "DACH-VWS",
            # "Laengstraeger_02",
            "SeatShell",
            "-s",
            # "ASaeule-Session_01.ofs",
            # "Einleger-Session.ofs",
            # "DACH-VWS-Session.ofs",
            # "Laengstraeger_02-Session.ofs",
            "SeatShell-Session.ofs",
            "-c",
            "48",
            "-e",
            "-o",
            "/d/github/ENSIMA/test/csv/DataSets-AIandML.csv",
            "-l",
            "DEBUG",
            "-pl",
            "--iterations",
            "50",
            "--parallel_samples",
            "1",
            "--time_limit",
            "02:00:00",
            # The x and y fields to use for the optimization
            "--x_fields",
            "Fr",
            "p",
            "D",
            "--y_fields",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L6",
            "L7",
            "--attention_coefficients",
            # L1 (Inadequate Stretch) should be small, must aber not be 0 necessary,
            # L4 (Safe) as large as possible,
            # L6 (Severe Thinning) and L7 (Cracks) must be 0.
            "1",
            "0.01",
            "0.05",
            "-0.18",
            "0.05",
            "1",
            "1",
            # Acquisition function
            "--selection_strategy",
            "crowding_distance",
            "--approximate_computing_check",
            "90",
            "--approximate_computing_limit",
            "15",  # L1 <= 5%
            "100",  # skips L2
            "100",  # skips L3
            "100",  # L4 >= 90%
            "100",  # skips L5
            "0.1",  # L6 <= 0.1%
            "0.01",  # L7 <= 0.01%
            "--x_space_structure",
            "grid",
            "--x_space_point_creation",
            "combination",
            "--latent_input_dim",
            "6",
            "--epochs",
            "300",
            "--target",
            "0",
            "10",
            "50",
            "100",
            "50",
            "0",
            "0",
            "--x_space_precision",
            "3",
            "--watcher_backend",
            "logger",
            "--end_value",
            "100",
        ]
    )

    # Set the constraints
    args.user_constraints = {
        "D": (0.8, 1.8),
        "p": (1.0, 12),
        "Fr": (0.00, 0.15),
        "Rp": [270, 420, 890],  # discrete values
        "db": (000, 800),
    }
    if "DACH-VWS" in args.jobname:
        args.user_constraints["Rp"] = [270, 420, 890]
        args.user_constraints["D"] = (0.5, 0.8)
        args.user_constraints["p"] = (1.0, 10)
    elif "Laengstraeger_02" in args.jobname:
        args.user_constraints["Rp"] = [390, 597, 712]
        args.user_constraints["D"] = (0.5, 1.35)

    # Filter by part using the labeled CSV
    args.output = "/d/github/ENSIMA/test/csv/DataSets-AIandML_labeled.csv"
    args = adjust_args_for_cluster(args)
    main(args, True)
