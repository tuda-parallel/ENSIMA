"""
Configurable logging utilities with optional color support for ENSIMA components.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import importlib.util
import logging
import os

COLORLOG_AVAILABLE = importlib.util.find_spec("colorlog") is not None
if COLORLOG_AVAILABLE:
    from colorlog import ColoredFormatter

LOG_FILE = None


def set_log_file(file):
    global LOG_FILE
    LOG_FILE = file
    # print(f"Logging to file: {LOG_FILE}")


class PrefixFilter(logging.Filter):
    def __init__(self, prefix=""):
        super().__init__()
        self.prefix = prefix

    def set_prefix(self, prefix):
        self.prefix = prefix

    def filter(self, record):
        # Include separator only if prefix is not empty
        if self.prefix:
            record.prefix = f" | {self.prefix}"
        else:
            record.prefix = ""
        return True


class Logger:
    def __init__(self, name=__name__, level=None, prefix="", log_file=None):
        self.prefix_filter = PrefixFilter(prefix)
        self.logger = logging.getLogger(name)
        self._level = self._resolve_level(level)
        if log_file is None:
            log_file = LOG_FILE

        if not self.logger.handlers:
            self.logger.setLevel(self._level)

            # Console handler
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(self._resolve_formatter(level))
            stream_handler.addFilter(self.prefix_filter)
            self.logger.addHandler(stream_handler)

            # file handler
            if log_file:
                self._add_file_handler(log_file)

            self.logger.propagate = False
            self._level = level

    def set_prefix(self, new_prefix: str):
        self.prefix_filter.set_prefix(new_prefix)

    def _add_file_handler(self, log_file):
        """Attach a file handler with the same formatter and prefix filter."""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(self._resolve_formatter())
        file_handler.addFilter(self.prefix_filter)
        self.logger.addHandler(file_handler)

    def _resolve_level(self, cli_level):
        env_level = os.environ.get("LOG_LEVEL")
        level_name = env_level or cli_level or "WARNING"
        return getattr(logging, level_name.upper(), logging.INFO)

    def _resolve_formatter(self, level=None):
        base_format = (
            "[%(asctime)s | %(levelname)-5s | %(name)s:%(lineno)d%(prefix)s]: %(message)s"
            # "[%(asctime)s | %(levelname)-8s | %(prefix)s]: %(message)s"
        )
        if COLORLOG_AVAILABLE:
            return ColoredFormatter(
                fmt=f"%(log_color)s {base_format}",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    # "ERROR": "red",
                    # "CRITICAL": "bold_red",
                    # "WARNING": "light_red",
                    "ERROR": "bold_red",
                    "CRITICAL": "bold_purple",
                },
            )
        else:
            return logging.Formatter(fmt=base_format, datefmt="%Y-%m-%d %H:%M:%S")

    def get(self):
        return self.logger

    def set_level(self, level):
        self._level = self._resolve_level(level)
        self.logger.setLevel(self._level)

    def get_level(self):
        return self._level


if __name__ == "__main__":
    # Initialize logger
    logger_obj = Logger(level="debug", prefix="Initial")
    logger = logger_obj.logger

    print("\n--- Logging with initial prefix ---")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # Change prefix
    logger_obj.set_prefix("Updated")

    print("\n--- Logging with updated prefix ---")
    logger.debug("Another debug message")
    logger.info("Another info message")
    logger.warning("Another warning message")
    logger.error("Another error message")
    logger.critical("Another critical message")
