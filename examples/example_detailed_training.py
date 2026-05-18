"""
Demonstrates detailed GP model training on an existing benchmark dataset.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

from ensima.classes.bayesian_optimization import BayesianOptimization
from ensima.classes.logger import Logger
from ensima.helpers.adjust_args_cluster import adjust_args_for_cluster
from ensima.helpers.parse_args import parse_arguments
from ensima.helpers.read_data import read_data

csv_file = "/d/gitlab/ensima-code/optimization/test/csv/DataSets-AIandML_20250401.csv"
args = parse_arguments(
    [
        "-j",
        "ASaeule",
        "-o",
        csv_file,
        "-l",
        "DEBUG",
        "--x_fields",
        "Fr",
        "p",
        "--y_fields",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        # "--attention_coefficients" ,
        # 1, 1, -1, -1, 1, 1, 1,
        "--attention_coefficients",
        "1",
        "0.8",
        "-0.4",
        "-0.4",
        "0.8",
        "0.8",
        "1",
    ]
)

# Log
logger = Logger(__name__, level=args.log_level).get()
logger.info("reading data")


# adjust args for cluster if needed
args = adjust_args_for_cluster(args)

# Create initial data set
x, y = read_data(args.output, args.x_fields, args.y_fields, log_level=args.log_level)

logger.info("Training Model")
bayes_opt = BayesianOptimization(args, x, y)
logger.info("Plotting results")
bayes_opt.plot()
logger.info("Plotting done")
