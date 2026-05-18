"""
Example script for running filtered Bayesian optimization on an HPC cluster.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

from ensima.helpers.parse_args import parse_arguments
from ensima.optimize import main

# MOE is the best for new part
# Example showing how to create script for cluster
if __name__ == "__main__":
    args = parse_arguments(
        [
            "-ofs",
            # makes errors
            "/rwthfs/rz/cluster/home/cg021604/gns/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
            # better, but does not produce the files needed
            # "/home/rwth1453/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
            "-ofm",
            "/rwthfs/rz/cluster/home/cg021604/gns/OpenForm_daily_linux64/OpenForm_64_batch",
            "-p",
            # "/rwthfs/rz/cluster/home/qfw89470/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_01_Flat",
            "/rwthfs/rz/cluster/home/qfw89470/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_04",
            "-j",
            # "ASaeule",
            "Einleger",
            "-s",
            # "ASaeule-Session_01.ofs",
            "Einleger-Session.ofs",
            "-c",
            "32",
            "-e",
            "-o",
            "/rwthfs/rz/cluster/home/cg021604/ensima-code/optimization/test/csv/DataSets-AIandML_labeled.csv",
            "-l",
            "DEBUG",
            # "INFO",
            "--license_server",
            "license.itc.rwth-aachen.de",
            "--license_port",
            "50141",
            "--license_type",
            "RLM",
            "--license_server_service",
            # "--save_and_load",
            "--iterations",
            "20",
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
            # "--attention_coefficients" ,
            # 1, 1, -1, -1, 1, 1, 1,
            "--attention_coefficients",
            "1",
            "0.9",
            "-0.2",
            "-0.2",
            "0.9",
            "0.9",
            "1",
            # acquisition function
            "--acquisition_function",
            "sum",
        ]
    )
    # Set the constrains
    args.user_constraints = {
        "Rp": [270, 420, 890],  # discrete values
        "D": (0.8, 1.8),  #
        # "D": [1.5],
        "Fr": (0.01, 0.2),  # max value is 1, good is max at 0.5
        "p": (1.0, 12),
    }

    main(args=args)
