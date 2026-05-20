"""
Demonstrates multi-part optimization using a Mixture-of-Experts GP model with geometry-based routing.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

from pathlib import Path

from ensima.helpers.adjust_args_cluster import (
    adjust_args_and_parts_for_cluster,
)
from ensima.helpers.parse_args import parse_arguments
from ensima.optimize import main

REPO_ROOT = Path(__file__).parent.parent
JIMS_TCO = f"{REPO_ROOT}/artifacts/JIMS/TCO-Benchmark"
TEST_CSV = f"{REPO_ROOT}/test/csv"

# INFO: In contrast to `example_optimization.py`, the moe mode needs:
# 1) the geometry of the new part (passed in args with `-g`)
# 2) types and parts passed to BO. This creates the moe inside BO.model. Note that the
#    settings are hardcoded (soft/hard gating and supervised/unsupervised)
if __name__ == "__main__":
    args = parse_arguments(
        [
            "-ofs",
            "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
            "-ofm",
            "/d/gitlab/ensima-code/test_data/gns/OpenForm_daily_linux64/OpenForm_64_batch",
            #
            "-p",
            # f"{JIMS_TCO}/PartType_01_Flat",
            # f"{JIMS_TCO}/PartType_04",
            # f"{JIMS_TCO}/new_parts/PartType_01",
            f"{JIMS_TCO}/new_parts/PartType_02",
            # f"{JIMS_TCO}/new_parts/PartType_03",
            #
            "-j",
            # "ASaeule",
            # "Einleger",
            # "DACH-VWS",
            "Laengstraeger_02",
            # "SeatShell",
            #
            "-s",
            # "ASaeule-Session_01.ofs",
            # "Einleger-Session.ofs",
            # "DACH-VWS-Session.ofs",
            "Laengstraeger_02-Session.ofs",
            # "SeatShell-Session.ofs",
            #
            "-c",
            "48",
            "-e",
            "-o",
            f"{TEST_CSV}/DataSets-AIandML_labeled.csv",
            #
            "-g",
            # f"{JIMS_TCO}/PartType_01_Flat/Aseule.t52",
            # f"{JIMS_TCO}/PartType_04/Einleger.t52",
            # f"{JIMS_TCO}/new_parts/PartType_01/DACH-VWS.t52",
            f"{JIMS_TCO}/new_parts/PartType_02/Laengstraeger_02.t52",
            # f"{JIMS_TCO}/new_parts/PartType_03/SeatShell.t52",
            #
            "-l",
            "DEBUG",
            "-pl",
            # "INFO",
            # "--save_and_load",
            # "-m",
            # "hgal",
            "--new_expert_start",
            "1",
            "--iterations",
            "10",
            "--parallel_samples",
            "1",
            "--time_limit",
            "08:00:00",  # overwritten for some parts
            # The x and y fields to use for the optimization
            "--x_fields",
            "Fr",
            "p",
            "D",
            # "Rp",
            # "db",
            "--y_fields",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L6",
            "L7",
            "--attention_coefficients",
            # Shift L2,L3, and L5 to maximize L4
            # L1 (Inadequate Stretch) should be small, must aber not be 0 necessary,
            # L4 (Safe) as large as possible,
            # L6 (Severe Thinning) and L7 (Cracks) must be 0.
            # 1, 1, -1, -1, 1, 1, 1,
            "1",
            "0.05",
            "0.05",
            "-0.18",
            "0.05",
            "1",
            "1",
            # Acquisition function
            "--selection_strategy",
            # "crowding_distance",  # "crowding_distance", "peak_based", "highest_sum"
            "highest_sum",
            "--approximate_computing_check",
            # "95",
            "90",  # test at 80%
            "--approximate_computing_limit",
            "15",  # L1 <= 5%
            "100",  # skips L2
            "100",  # skips L3
            "100",  # L4 >= 90%
            "100",  # skips L5
            "1",  # L6 <= 1%
            "1",  # L7 <= 1%
            "--x_space_structure",
            "grid",
            "--x_space_point_creation",
            # "linear",
            "combination",
            "--x_space_precision",
            "2",
            "--sample_mode",
            "upsample",
            "--latent_input_dim",
            "4",
            "--epochs",
            "300",
            "--target",
            "0",  # L1 <= 5%
            "10",  # skips L2
            "50",  # skips L3
            "100",  # L4 >= 90%
            "50",  # skips L5
            "0.5",  # L6 <= 1%
            "0.5",  # L7 <= 1%
            "--watcher_backend",
            "logger",
        ]
    )

    # Create point clouds with labels (one array per type)
    parts = [
        (
            "ASaeule",
            f"{JIMS_TCO}/PartType_01_Flat/ASaeule.t52",
        ),
        (
            "BSaeule",
            f"{JIMS_TCO}/PartType_02_Beam/BSaeule_DX56D.t52",
        ),
        (
            "RadhausAdapter",
            f"{JIMS_TCO}/PartType_03_Deep/RadhausAdapter.t52",
        ),
        # (
        #     "Einleger",
        #     f"{JIMS_TCO}/PartType_04/Einleger.t52",
        # ),
        # (
        #     "ASaeule_BSym",
        #     f"{JIMS_TCO}/T52-PartFiles/A-Pilar.t52",
        # ),
        (
            "Quertraeger",
            f"{JIMS_TCO}/T52-PartFiles/CrossBeam.t52",
        ),
        (
            "FrontFender_01",
            f"{JIMS_TCO}/T52-PartFiles/FrontFender_A.t52",
        ),
        (
            "Kotfluegel-DC",
            f"{JIMS_TCO}/T52-PartFiles/FrontFender_B.t52",
        ),
        (
            "Heckklappe_aussen",
            f"{JIMS_TCO}/T52-PartFiles/TailGate.t52",
        ),
        (
            "Tankdeckeleinsatz",
            f"{JIMS_TCO}/T52-PartFiles/TankCapInsert.t52",
        ),
        (
            "Tunnel_Passat",
            f"{JIMS_TCO}/T52-PartFiles/Tunnel.t52",
        ),
        # (
        #     "SeatShell",
        #     f"{JIMS_TCO}/new_parts/PartType_03/SeatShell.t52",
        # ),
    ]

    # Set the constraints
    args.user_constraints = {
        "D": (0.8, 1.8),
        "p": (1.0, 12),
        "Fr": (0.00, 0.15),  # good is max at 0.2
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

    # adjust args for the cluster if needed
    args, parts = adjust_args_and_parts_for_cluster(args, parts)

    main(args, parts=parts)
