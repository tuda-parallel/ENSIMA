"""
Manages the RLM license server lifecycle for OpenFOAM and OpenForm simulation software.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import time
from argparse import Namespace

from ensima.classes.execute import execute_background, execute_block
from ensima.classes.logger import Logger
from ensima.helpers.parse_args import parse_arguments


class LicenseServer:
    def __init__(self, args: Namespace, clean=False):
        self.server = args.license_server
        self.port = args.license_port
        self.type = args.license_type
        self.rlm = args.license_rlm
        self.service = args.license_server_service
        self.process = None
        self.log_level = args.log_level
        self.logger = Logger(__name__, level=self.log_level).get()
        if clean and not self.service:
            self.logger.debug("Closing other instances of license server")
            _, _ = execute_block(f"lsof -ti tcp:{args.license_port} | xargs -r kill -9")

    def start(self):
        if not self.service:
            self.logger.info("Starting License server")
            command = (
                f"export GNS_LICENSE_SERVER={self.server}"
                f" GNS_LICENSE_SERVER_PORT_IPv6={self.port}"
                f" GNS_LICENSE_SERVER_TYPE={self.type} && "
                f" {self.rlm} "
            )
            self.logger.debug(f"License server settings: {command}")
            self.process = execute_background(
                command, log_file="./license.log", log_level=self.log_level
            )
            self.logger.debug("License server log saved in ./license.log")
            time.sleep(5)
        else:
            self.logger.debug("License server is executed as a service, skipping start")

    def stop(self):
        if not self.service:
            self.logger.debug("Killing license server process")
            self.process.kill()
            self.logger.debug("License server process killed. Waiting for exit")
            self.process.wait()
            self.logger.debug("License server ended successfully")
        else:
            self.logger.debug("License server is executed as a service, skipping stop")


if __name__ == "__main__":
    args = parse_arguments(["-l", "DEBUG"])
    licenser_server = LicenseServer(args)
    licenser_server.start()
    time.sleep(10)
    licenser_server.logger.debug("Content of log file:")
    with open("./license.log") as f:
        print(f.read())
    licenser_server.stop()
