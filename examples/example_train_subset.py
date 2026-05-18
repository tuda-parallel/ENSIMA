"""
Trains a GP model on a filtered subset of the benchmark dataset.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import os

import numpy as np

from ensima_optimize.classes.bayesian_optimization import BayesianOptimization
from ensima_optimize.classes.logger import Logger
from ensima_optimize.helpers.adjust_args_cluster import adjust_args_for_cluster
from ensima_optimize.helpers.parse_args import parse_arguments
from ensima_optimize.helpers.read_data import read_data

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = "/d/gitlab/ensima-code/optimization/test/csv/DataSets-AIandML_20250401.csv"
path = "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_02_Beam"
# Start license server as a service (../test_data/gns)
args = parse_arguments(
    [
        "-ofs",
        "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
        "-ofm",
        "/d/gitlab/ensima-code/test_data/gns/OpenForm_2.22.3_linux64/OpenForm_64/openform",
        "-p",
        path,
        "-j",
        "BSaeule_DX56D",
        "-s",
        "BSaeule_DX56D-Session_01.ofs",
        "-c",
        "4",
        "-e",
        "-o",
        csv_file,
        "-l",
        "DEBUG",
        # "INFO",
    ]
)
logger = Logger(__name__, level=args.log_level).get()

args = adjust_args_for_cluster(args)
text = "args are: \n"
for key, value in args.__dict__.items():
    text += f"{key}: {value}\n"
logger.debug(text)

logger.info("reading data")
x, y = read_data(args.output, log_level=args.log_level)

# merge Y into three values
y_reduced = np.column_stack(
    [
        y[:, 0] + y[:, 1] / 2,
        y[:, 2] + y[:, 3] / 2,
        y[:, 4] + y[:, 5] / 2,
    ]
)
logger.info("Training Model")
bayes_opt = BayesianOptimization(args, x, y_reduced)
logger.info(repr(bayes_opt))
