"""
Tests for progress monitoring and simulation execution utilities.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import logging
import socket
import time

import pytest

from ensima.classes.execute import execute_background, execute_block
from ensima.classes.license_server import LicenseServer
from ensima.classes.logger import Logger
from ensima.classes.progress_watcher import ProgressWatcher
from ensima.helpers.parse_args import parse_arguments

LOGGER = logging.getLogger(__name__)


def test_progress() -> None:
    if socket.gethostname() != "electric":
        pytest.skip("Test only runs on host 'electric'")

    args = parse_arguments(
        [
            "--path",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_01_Flat",
            "-l",
            "DEBUG",
            "-j",
            "ASaeule",
            "-ofs",
            "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
        ]
    )

    try:
        # Start the license server
        licenser_server = LicenseServer(args, True)
        licenser_server.start()

        # Compose OFSolver execution command
        command = (
            f"export GNS_LICENSE_SERVER={args.license_server} "
            f"GNS_LICENSE_SERVER_PORT_IPv6={args.license_port} "
            f"GNS_LICENSE_SERVER_TYPE={args.license_type} && "
            f"cd {args.path} && {args.ofsolver} -j {args.jobname} -c 1"
        )

        log_path = f"{args.path}/{args.jobname}.log"
        with ProgressWatcher(log_path, log_level=args.log_level, name=__name__):
            process = execute_background(command, "", log_level=args.log_level)
            time.sleep(20)
            LOGGER.debug("Killing OFSolver process after 20 seconds")
            process.kill()

        licenser_server.stop()

    except Exception as e:
        LOGGER.exception("Exception during progress monitoring test")
        pytest.fail(f"progress monitor failed: {e}")


if __name__ == "__main__":
    args = parse_arguments(
        [
            "--path",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_01_Flat",
            "-l",
            "DEBUG",
            "-j",
            "ASaeule",
            "-ofs",
            "/d/gitlab/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
        ]
    )
    # Init logger
    logger = Logger(__name__, level=args.log_level).get()

    # clen the port
    _, _ = execute_block("lsof -ti tcp:5053 | xargs -r kill -9")

    # Start the license server
    licenser_server = LicenseServer(args)
    licenser_server.start()

    # Run OFSolve
    command = (
        f"export GNS_LICENSE_SERVER={args.license_server}"
        f" GNS_LICENSE_SERVER_PORT_IPv6={args.license_port}"
        f" GNS_LICENSE_SERVER_TYPE={args.license_type} && "
        f"cd {args.path} && {args.ofsolver} -j {args.jobname} -c 1"
    )
    logger.debug(f"Executing command: {command}")
    with ProgressWatcher(f"{args.path}/{args.jobname}.log", name=__name__) as watcher:
        _, _ = execute_block(
            command,
            raise_exception=False,
            options={"check": False},
            log_level=args.log_level,
        )

    logger.debug("Process finished")

    # Stop the license server
    licenser_server.stop()

    # print the results
    with open("./license.log") as f:
        print(f.read())
