"""
Tests for the LicenseServer class lifecycle and connectivity.

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

from ensima.classes.license_server import LicenseServer
from ensima.helpers.parse_args import parse_arguments

LOGGER = logging.getLogger(__name__)


def test_license() -> None:
    if socket.gethostname() != "electric":
        pytest.skip("Test only runs on host 'electric'")

    try:
        args = parse_arguments(["-l", "DEBUG"])
        licenser_server = LicenseServer(args)
        licenser_server.start()
        time.sleep(5)
        licenser_server.logger.debug("Content of log file:")

        log_file = "./license.log"
        with open(log_file) as f:
            content = f.read()

        print(content)
        licenser_server.stop()

        assert len(content) > 0

    except Exception as e:
        LOGGER.info(f"Reading data from {log_file} failed")
        # LOGGER.error(f"Error details: {e}")
        pytest.fail(f"Dummy test failed: {e}")


if __name__ == "__main__":
    args = parse_arguments(["-l", "DEBUG"])
    licenser_server = LicenseServer(args)
    licenser_server.start()
    time.sleep(10)
    licenser_server.logger.debug("Content of log file:")
    with open("./license.log") as f:
        print(f.read())
    licenser_server.stop()
    print("All tests passed")
