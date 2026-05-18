"""
Progress monitoring for simulation runs using Rich progress bars or standard logging.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import importlib
import importlib.util
import sys
import threading
import time

from ensima_optimize.classes.execute import execute_block
from ensima_optimize.classes.license_server import LicenseServer
from ensima_optimize.classes.logger import Logger
from ensima_optimize.helpers.parse_args import parse_arguments
from ensima_optimize.helpers.read_data import read_last_progress

RICH_AVAILABLE = importlib.util.find_spec("rich") is not None

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )


class ProgressWatcher:
    def __init__(
        self,
        filename: str,
        interval: float = 1.0,
        log_level="INFO",
        name: str = __name__,
        prefix: str = "",
        lock=None,
        watcher_backend="rich",
        pause_event=None,
        transient=False,
    ):
        self.filename = filename
        self.interval = interval
        self._worker = None
        self._stop_event = None
        self._last_progress = None
        self.logger = Logger(name, level=log_level, prefix=prefix).get()
        self.lock = lock
        self.watcher_backend = watcher_backend
        self.pause_event = pause_event
        self.transient = transient

    # ---------------- Logger Watch Loop ----------------
    def _watch_loop_logger(self, stop_event):
        old_step = 0
        released_lock = False
        while not stop_event.is_set():
            step, prog = read_last_progress(self.filename)
            if prog >= 0.0 and prog != self._last_progress and step >= old_step:
                self.logger.info(f"Step: {step}, Progress: {prog:.2f}%")
                self._last_progress = prog
                if self.lock and not released_lock:
                    self.lock.release()
                    released_lock = True
            old_step = step
            time.sleep(self.interval)

    def _watch_loop_rich(self, stop_event):
        if not RICH_AVAILABLE:
            return self._watch_loop_logger(stop_event)
        # console = Console(force_terminal=True)
        console = Console(force_terminal=True, file=sys.stdout, color_system="auto")
        released_lock = False
        old_step = 0
        current_task_id = None
        current_step = 0
        first_step_seen = False  # ensure we don’t start with a step > 1

        progress_display = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=self.transient,
            # refresh_per_second=self.interval // 2 if self.interval // 2 > 0 else 1,
        )
        progress_display.start()

        try:
            while not stop_event.is_set():

                if self.pause_event and self.pause_event.is_set():
                    time.sleep(self.interval)
                    for task in progress_display.tasks:
                        if (
                            task.description in ["Step 1", "Step 2"]
                            and task.completed >= task.total
                        ):
                            progress_display.remove_task(task.id)
                    continue

                step, prog = read_last_progress(self.filename)

                if step <= 0:
                    time.sleep(self.interval)
                    continue

                # Wait for step 1 if this is the first read
                if not first_step_seen:
                    if step != 1:
                        time.sleep(self.interval)
                        continue
                    first_step_seen = True

                # Only create a new task if step >= old_step
                if step >= old_step:
                    if step != current_step:
                        # Finish previous task
                        if current_task_id is not None:
                            progress_display.update(current_task_id, completed=100)

                        # Start new task
                        current_task_id = progress_display.add_task(
                            f"Step {step}", total=100
                        )
                        current_step = step

                # Update current task (even if step < old_step)
                if (
                    prog >= 0.0
                    and current_task_id is not None
                    and prog != self._last_progress
                    and not (self.pause_event and self.pause_event.is_set())
                ):
                    progress_display.update(current_task_id, completed=prog)
                    self._last_progress = prog

                    if self.lock and not released_lock:
                        self.lock.release()
                        released_lock = True

                old_step = step
                time.sleep(self.interval)

        finally:
            # # finalize last task
            # if current_task_id is not None:
            #     progress_display.update(current_task_id, completed=100)
            progress_display.stop()

    def _watch_loop(self, stop_event):
        if self.watcher_backend == "logger":
            self._watch_loop_logger(stop_event)
        elif self.watcher_backend == "rich":
            self._watch_loop_rich(stop_event)
        else:
            raise ValueError(f"Unknown mode: {self.watcher_backend}")

    # ---------------- Start ----------------
    def start(self):
        if self._worker and self._worker.is_alive():
            return

        self._stop_event = threading.Event()
        # Logger mode: run in a background thread
        self._worker = threading.Thread(
            target=self._watch_loop, args=(self._stop_event,), daemon=True
        )
        self._worker.start()

    # ---------------- Stop ----------------
    def stop(self):
        if self._stop_event:
            self._stop_event.set()
            if self.watcher_backend == "logger" and self._worker:
                self._worker.join()

    # ---------------- Context Manager ----------------
    def __enter__(self):
        self.logger.debug(
            f"Starting progress watcher on file {self.filename} with interval {self.interval} s"
        )
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


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
    log = f"{args.path}/{args.jobname}.log"
    logger.debug(f"Log path is {log}")
    command = (
        f"export GNS_LICENSE_SERVER={args.license_server}"
        f" GNS_LICENSE_SERVER_PORT_IPv6={args.license_port}"
        f" GNS_LICENSE_SERVER_TYPE={args.license_type} && "
        f"cd {args.path} && {args.ofsolver} -j {args.jobname} -c 1"
    )
    logger.debug(f"Executing command: {command}")
    app_log = "/d/gitlab/ensima-code/optimization/ensima_optimize/app.log"
    with ProgressWatcher(log, log_level=args.log_level) as watcher:
        # process = execute_background(command, app_log, options={"check": False})
        # time.sleep(50)
        # process.wait()
        # logger.debug(f"Process finished")
        # process.kill()
        # time.sleep(5)
        #
        #
        _, _ = execute_block(
            command,
            raise_exception=False,
            # options={"check": False},
            options={"check": False},
            log_level=args.log_level,
        )
        #
        #
        # p = (
        #     execute_background(
        #         command,
        #         raise_exception=False,
        #         # options={"check": False},
        #         options={"check": False},
        #         log_level=args.log_level,
        #     ),
        # )
        # p.wait()
        # while True:
        #     time.sleep(1)

    logger.debug("Process finished")

    # Stop the license server
    licenser_server.stop()

    # print the results
    with open("./license.log") as f:
        print(f.read())
    with open(app_log) as f:
        print(f.read())
