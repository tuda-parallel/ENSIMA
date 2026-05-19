"""
Example using filtered or expert-based Bayesian optimization on a single part.

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

# use filtered or MOE
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
            "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_01",
            # "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_02",
            # "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/new_parts/PartType_03",
            "-j",
            # "Einleger",
            # "ASaeule",
            "DACH-VWS",
            # "Laengstraeger_02",
            # "SeatShell",
            "-s",
            # "ASaeule-Session_01.ofs",
            # "Einleger-Session.ofs",
            "DACH-VWS-Session.ofs",
            # "Laengstraeger_02-Session.ofs",
            # "SeatShell-Session.ofs",
            "-c",
            "48",
            "-e",
            "-l",
            "DEBUG",
            "--iterations",
            "5",
            "--parallel_samples",
            "1",
            "--time_limit",
            "04:00:00",
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

    # Avoid optimization and rather use a list input:
    if "DACH-VWS" in args.jobname:
        mode = "unskilled user"
        if mode == "unskilled user":
            args.input = [
                [140.00, 0.70, 3.00, 0.00, 0.00, 6.4, 6.9],
                [140.00, 0.70, 3.00, 0.00, 220.00, 11.1, 2.8],
                [140.00, 0.70, 3.00, 0.06, 800.00, 24.2, 0.9],
                [140.00, 0.70, 6.00, 0.06, 800.00, 24.7, 0.6],
                [140.00, 0.70, 9.00, 0.06, 800.00, 26.0, 0.5],
                [140.00, 0.70, 9.00, 0.06, 220.00, 20.8, 0.4],
            ]
            x_fields = ["Rp", "D", "p", "Fr", "db", "dD", "dT"]
        elif mode == "expert":
            args.input = [
                [140.00, 0.70, 3.00, 0.00, 0.00, 6.4, 6.9],
                [140.00, 0.70, 3.00, 0.00, 220.00, 11.1, 2.8],
                [140.00, 0.70, 6.00, 0.03, 220.00, 16.2, 0.9],
                [140.00, 0.70, 9.00, 0.06, 220.00, 20.8, 0.4],
            ]
            x_fields = ["Rp", "D", "p", "Fr", "db", "dD", "dT"]

    elif "Laengstraeger_02" in args.jobname:
        # expert
        args.input = [
            [597.00, 1.20, 3.00, 0.03, 0.00, 34.8, 18.6],
            [597.00, 1.00, 3.00, 0.12, 0.00, 27.4, 14.9],
            [390.00, 1.00, 3.00, 0.12, 0.00, 26.8, 15.9],
            [390.00, 1.00, 2.00, 0.12, 0.00, 25.2, 14.9],
        ]
        x_fields = ["Rp", "D", "p", "Fr", "db", "dD", "dT"]

    elif "SeatShell" in args.jobname:
        # expert
        args.input = [
            [250.00, 1.10, 3.00, 0.03, 0.00, 35.6, 4.80],
            [250.00, 1.10, 3.00, 0.03, 800.00, 31.6, 5.20],
            [250.00, 1.10, 6.00, 0.03, 800.00, 31.8, 3.90],
            [250.00, 1.10, 9.00, 0.03, 800.00, 29.5, 3.20],
        ]
        x_fields = ["Rp", "D", "p", "Fr", "db", "dD", "dT"]

    # reduce the xfields:
    args.x_fields = ["Rp", "p", "Fr", "db", "D"]
    keep_indices = [x_fields.index(x) for x in args.x_fields if x in x_fields]
    print(
        f"Keeping indices: {keep_indices}, x_fields: {[x_fields[i] for i in keep_indices]}"
    )
    args.input = [[row[i] for i in keep_indices] for row in args.input]
    # adjust the iterations
    args.iterations = len(args.input)

    # 1) no filter, use all data points
    # args = adjust_args_for_cluster(args)
    # main(args, False)

    # 2)filter by type
    # since the CSV is unlabeled, we filter by type.
    # Use ensima/helpers/complexity.py to find the type number.
    # args.output = "/d/github/ENSIMA/test/csv/DataSets-AIandML.csv"
    # args = adjust_args_for_cluster(args)
    # main(args, True, type_number=1)

    # 3) filter by part
    args.output = "/d/github/ENSIMA/test/csv/DataSets-AIandML_expert.csv"
    args = adjust_args_for_cluster(args)
    main(args, True)
