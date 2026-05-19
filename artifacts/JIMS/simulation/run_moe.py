"""
Script used to produce the MoE simulation results in artifacts/JIMS/sim_results/MOE/.

Two part configurations are covered (select by uncommenting the relevant lines):

  Longitudinal Beam (Laengstraeger_02) — paper sim_1..4
    x_fields: Fr, p, D, Rp  |  selection: highest_sum  |  prec: 2  |  lat_dim: 4
    new_expert_start varies: 1 (sim_1), 3 (sim_2), 5 (sim_3), 3+ps=2 (sim_4 in softLimits)

  Roof (DACH-VWS) — paper sim_1..3
    x_fields: Fr, p, D      |  selection: crowding_distance  |  prec: 1  |  lat_dim: 6
    new_expert_start varies: 3 (sim_1), 1 (sim_2), 5 (sim_3)
    attention_coefficients differ: [..., -0.2, ...] vs [..., -0.18, ...] for Laengstraeger

Lines marked  # <<< DACH-VWS  indicate where the Roof configuration differs.
Executed on a GPU cluster — set -ofs and -ofm to your local solver paths.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

from ensima.helpers.adjust_args_cluster import (
    adjust_args_and_parts_for_cluster,
)
from ensima.helpers.parse_args import parse_arguments
from ensima.optimize import main

if __name__ == "__main__":
    args = parse_arguments(
        [
            "-ofs",
            "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
            "-ofm",
            "/d/gitlab/ensima-code/test_data/gns/OpenForm_daily_linux64/OpenForm_64_batch",
            #
            # --- Active part (Longitudinal Beam) ---
            "-p",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_02",
            # "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_01",  # <<< DACH-VWS
            #
            "-j",
            "Laengstraeger_02",
            # "DACH-VWS",  # <<< DACH-VWS
            #
            "-s",
            "Laengstraeger_02-Session.ofs",
            # "DACH-VWS-Session.ofs",  # <<< DACH-VWS
            #
            "-g",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_02/Laengstraeger_02.t52",
            # "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_01/DACH-VWS.t52",  # <<< DACH-VWS
            #
            "-c",
            "48",
            "-e",
            "-o",
            "/d/github/ENSIMA/test/csv/DataSets-AIandML_labeled.csv",
            "-l",
            "DEBUG",
            "-pl",
            #
            "--new_expert_start",
            "1",  # Laengstraeger: 1 (sim_1), 3 (sim_2), 5 (sim_3), 3 (sim_4/softLimits)
            # "3",   # <<< DACH-VWS sim_1; also Laengstraeger sim_2
            # "5",   # <<< DACH-VWS sim_3; also Laengstraeger sim_3
            #
            "--iterations",
            "10",
            "--parallel_samples",
            "1",
            # "2",  # sim_4 (Laengstraeger softLimits) uses ps=2, cores=24
            "--time_limit",
            "08:00:00",
            #
            # --- x_fields ---
            "--x_fields",
            "Fr",
            "p",
            "D",
            "Rp",  # Laengstraeger
            # "Fr", "p", "D",        # <<< DACH-VWS (no Rp)
            #
            "--y_fields",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L6",
            "L7",
            #
            # --- Attention coefficients ---
            # L1 (Inadequate Stretch): small  |  L4 (Safe): large  |  L6,L7: zero
            "--attention_coefficients",
            "1",
            "0.05",
            "0.05",
            "-0.18",
            "0.05",
            "1",
            "1",  # Laengstraeger
            # "1", "0.01", "0.05", "-0.2",  "0.05", "1", "1", # <<< DACH-VWS (note -0.2)
            #
            # --- Acquisition function ---
            "--selection_strategy",
            "highest_sum",  # Laengstraeger
            # "crowding_distance",  # <<< DACH-VWS
            #
            "--approximate_computing_check",
            "90",
            "--approximate_computing_limit",
            "15",  # L1 <= 15%
            "100",  # skips L2
            "100",  # skips L3
            "100",  # L4 >= 90%
            "100",  # skips L5
            "0.1",  # L6 <= 0.1%
            "0.01",  # L7 <= 0.01%
            #
            # --- Search space ---
            "--x_space_structure",
            "grid",
            "--x_space_point_creation",
            "combination",
            "--x_space_precision",
            "2",  # Laengstraeger (paper: 2; verified in JSON)
            # "1",  # <<< DACH-VWS (paper: 2; actual run used 1)
            "--sample_mode",
            "upsample",
            #
            # --- GP model ---
            "--latent_input_dim",
            "4",  # Laengstraeger
            # "6",  # <<< DACH-VWS
            "--epochs",
            "300",
            #
            "--target",
            "0",  # L1
            "10",  # L2
            "50",  # L3
            "100",  # L4
            "50",  # L5
            "0",  # L6
            "0",  # L7
            "--watcher_backend",
            "logger",
        ]
    )

    # Training part library — one expert is created per entry.
    # Each tuple is (label, path-to-t52-geometry-file).
    parts = [
        (
            "ASaeule",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_01_Flat/ASaeule.t52",
        ),
        (
            "BSaeule",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_02_Beam/BSaeule_DX56D.t52",
        ),
        (
            "RadhausAdapter",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_03_Deep/RadhausAdapter.t52",
        ),
        (
            "Einleger",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_04/Einleger.t52",
        ),
        (
            "Quertraeger",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/T52-PartFiles/CrossBeam.t52",
        ),
        (
            "FrontFender_01",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/T52-PartFiles/FrontFender_A.t52",
        ),
        (
            "Kotfluegel-DC",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/T52-PartFiles/FrontFender_B.t52",
        ),
        (
            "Heckklappe_aussen",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/T52-PartFiles/TailGate.t52",
        ),
        (
            "Tankdeckeleinsatz",
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/T52-PartFiles/TankCapInsert.t52",
        ),
        # (
        #     "Tunnel_Passat",
        #     "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/T52-PartFiles/Tunnel.t52",
        # ),
    ]

    # Parameter constraints
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

    args, parts = adjust_args_and_parts_for_cluster(args, parts)

    main(args, parts=parts)
