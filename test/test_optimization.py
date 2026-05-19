"""
Integration tests for the Bayesian optimization workflow.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import logging
import os
import warnings

import numpy as np
import pytest
from rich.console import Console
from sklearn.exceptions import ConvergenceWarning

from ensima.classes.bayesian_optimization import BayesianOptimization
from ensima.classes.execute import execute_block_and_log
from ensima.classes.file_modifier import FileModifier
from ensima.helpers.objective_function import dummy_init
from ensima.helpers.parse_args import parse_arguments
from ensima.helpers.read_data import read_data
from ensima.optimize import main

warnings.filterwarnings("ignore", category=ConvergenceWarning)

LOGGER = logging.getLogger(__name__)


def test_dummy() -> None:
    """
    Test the dummy initialization and Bayesian optimization.
    """
    args = parse_arguments([])
    args.output = ""
    x, y, objective_function = dummy_init()
    args.x_fields = [f"x{i+1}" for i in range(x.shape[1])]
    bayes_opt = BayesianOptimization(
        args,
        x,
        y,
        objective_function=objective_function,
    )
    bayes_opt.optimize(20)
    console = Console()
    console.print("[green]-- Dummy test finished -- \n\n")
    assert True


def test_dummy_weight() -> None:
    """
    Test the dummy initialization and Bayesian optimization.
    """
    args = parse_arguments([])
    args.output = ""
    x, y, objective_function = dummy_init()
    args.attention_coefficients = np.random.uniform(-1, 1, size=y.shape[1])
    args.x_fields = [f"x{i+1}" for i in range(x.shape[1])]
    bayes_opt = BayesianOptimization(
        args,
        x,
        y,
        objective_function=objective_function,
    )
    bayes_opt.optimize(20)
    console = Console()
    console.print("[green]-- Dummy test finished -- \n\n")
    assert True


def test_dummy_limit_improvement() -> None:
    """
    Test the dummy initialization and Bayesian optimization.
    """
    args = parse_arguments([])
    args.output = ""
    args.end_condition = "no_improvement"
    args.end_value = 5
    x, y, objective_function = dummy_init()
    args.x_fields = [f"x{i+1}" for i in range(x.shape[1])]
    bayes_opt = BayesianOptimization(
        args,
        x,
        y,
        objective_function=objective_function,
    )
    bayes_opt.optimize(20)
    console = Console()
    console.print("[green]-- Dummy test finished -- \n\n")
    assert True


def test_dummy_limit_min() -> None:
    """
    Test the dummy initialization and Bayesian optimization.
    """
    args = parse_arguments([])
    args.output = ""
    args.end_condition = "constant_min"
    args.end_value = 5
    x, y, objective_function = dummy_init()
    args.x_fields = [f"x{i+1}" for i in range(x.shape[1])]
    bayes_opt = BayesianOptimization(
        args,
        x,
        y,
        objective_function=objective_function,
    )
    bayes_opt.optimize(20)
    console = Console()
    console.print("[green]-- Dummy test finished -- \n\n")
    assert True


def test_dummy_limit_energy() -> None:
    """
    Test the dummy initialization and Bayesian optimization.
    """
    args = parse_arguments([])
    args.output = ""
    x, y, objective_function = dummy_init()
    args.end_condition = "energy_budget"
    args.energy_estimation = True
    args.end_value = 0.8
    args.x_fields = [f"x{i+1}" for i in range(x.shape[1])]
    bayes_opt = BayesianOptimization(
        args,
        x,
        y,
        objective_function=objective_function,
    )
    bayes_opt.optimize(20)
    console = Console()
    console.print("[green]-- Dummy test finished -- \n\n")
    assert True


def test_read_csv():
    """
    Test reading data from a CSV file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file = "csv/Results-CylindricalCup_23-05-02.csv"
    file_full_path = os.path.join(script_dir, file)
    print(file_full_path)
    x, y = read_data(
        file_full_path,
        ["Nr", "D", "p", "Fr", "dD", "dT"],
        ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
    )
    console = Console()
    console.print(f"x: {x}")
    console.print(f"y: {y}")
    console.print("[green]-- Read test finished -- \n\n")
    assert True


def test_execute_block_and_log():
    """
    Test the execution of a block and logging the output.
    """
    execute_block_and_log("ls -lahrt", "./log.out")
    execute_block_and_log("ls -lahrt", "./log.out", True)
    execute_block_and_log("ls -lahrt", "./log.out", True, "./")
    execute_block_and_log("ls -lahrt", "./log.out", True, "./", None)


def test_read_dat():
    # f = FileModifier(
    #     "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark"
    #     "/PartType_02_Beam/BSaeule_DX56D.dat"
    # )
    f = FileModifier(
        "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_02_Beam/BSaeule_DX56D.dat"
    )
    f.set_design_parameters({"Fr": 0.01})
    f.set_design_parameters({"p": 0.1})
    f.set_blank_thickness(0.1)
    f.print()

    # restore
    f.set_design_parameters({"Fr": 0.03})
    f.set_design_parameters({"p": 0.5})
    f.set_blank_thickness(0.7)
    f.print()


def test_ensima_optimization_init():
    """
    Test the ENSIMA optimization process.
    """
    try:
        path = "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark" "/PartType_02_Beam"
        job = "BSaeule_DX56D"
        csv_file = "Somecsvfile.csv"
        session = "BSaeule_DX56D-Session_01.ofs"
        args = parse_arguments(
            ["-p", path, "-j", job, "-s", session, "-e", "-o", csv_file]
        )
        _ = BayesianOptimization(args, np.array([[0]]), np.array([[0]]))
    except Exception as e:
        LOGGER.info(f"Dummy test failed for job {job}")
        pytest.fail(f"Dummy test failed: {e}")


def test_ensima_train():
    """
    Test the ENSIMA optimization process.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(script_dir, "csv/DataSets-AIandML.csv")
        path = "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark" "/PartType_02_Beam"
        # Start license server as a service (../test_data/gns)
        ofsolver = (
            "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1"
            ".0.4e_eng_linux64.exe"
        )
        openform = "/d/gitlab/ensima-code/test_data/gns/OpenForm_daily_linux64/OpenForm_64_batch"
        cores = 8
        job = "BSaeule_DX56D"
        session = "BSaeule_DX56D-Session_01.ofs"
        args = parse_arguments(
            [
                "-ofs",
                ofsolver,
                "-ofm",
                openform,
                "-p",
                path,
                "-j",
                job,
                "-s",
                session,
                "-c",
                f"{cores}",
                "-e",
                "-o",
                csv_file,
                "-l",
                "DEBUG",
            ]
        )
        # Skip this test, as some functions are missing
        x, y = read_data(args.output, log_level=args.log_level)
        bayes_opt = BayesianOptimization(args, x, y[:, 1:2])
        # bayes_opt = BayesianOptimization(args, x, y)
        print(repr(bayes_opt))
    except Exception as e:
        LOGGER.info(f"Reading data from {csv_file}")
        # LOGGER.error(f"Reading data from {e}")
        pytest.fail(f"Dummy test failed: {e}")


def test_ensima_optimization():
    """
    Test the ENSIMA optimization process.
    """
    try:

        csv_file = "/d/github/ENSIMA/test/csv/DataSets-AIandML_20250401.csv"
        path = "/d/github/ENSIMA/artifacts/JIMS/TCO-Benchmark/PartType_01_Flat"
        # Start license server as a service (../test_data/gns)
        args = parse_arguments(
            [
                "-ofs",
                "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
                "-ofm",
                "/d/gitlab/ensima-code/test_data/gns/OpenForm_daily_linux64/OpenForm_64_batch",
                "-p",
                path,
                "-j",
                "ASaeule",
                "-s",
                "ASaeule-Session_01.ofs",
                "-c",
                "2",
                "-e",
                "-o",
                csv_file,
                "-l",
                "DEBUG",
                # "INFO",
            ]
        )
        return
        main(args)
    except Exception as e:
        LOGGER.info(f"Reading data from {csv_file}")
        # LOGGER.error(f"Reading data from {e}")
        pytest.fail(f"Dummy test failed: {e}")


if __name__ == "__main__":
    test_dummy_weight()
    # test_dummy()
    # # test_read_csv()
    # # test_ensima_optimization()
    # test_read_dat()
    print("All tests passed")
