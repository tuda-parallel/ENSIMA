"""
Orchestrates the execution of stamping simulations and collects results for the optimization loop.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import datetime
import multiprocessing
import os
import re
import sys
import time
from argparse import Namespace

import numpy as np

from ensima.classes.execute import (
    execute_block,
    monitor_and_create_signal,
    timer_process,
    write_signal,
)
from ensima.classes.file_modifier import FileModifier
from ensima.classes.logger import Logger
from ensima.classes.progress_watcher import ProgressWatcher
from ensima.helpers.misc import safe_remove
from ensima.helpers.parse_results import csv_parser_path
from ensima.helpers.read_data import read_data
from ensima.helpers.units import convert_seconds_to_hms


class Simulation:
    """
    A class to run simulation workflows using OFsolve and OpenForm,
    managing session files, input parameters, and result parsing.

    Attributes:
        args: Command-line arguments including paths, solver config, etc.
        next_sample: A 2D array of input design parameters to evaluate.
        iteration_index: The current iteration index.
        result: A string naming the output result (used as the jobname in OFsolve).
        prefix: A status prefix string for logging.
        logger: Logger instance.
    """

    def __init__(
        self,
        args: Namespace,
        next_sample: np.ndarray,
        iteration: int = 0,
        total_iterations: int = 0,
        prefix=None,
    ):
        """
        Initialize the simulation instance.

        Args:
            args: Parsed command-line arguments (expects .jobname, .path, etc.).
            next_sample: np.ndarray representing design parameters (1 row).
            iteration: Current iteration number.
            total_iterations: Total number of optimization iterations.
        """
        self.args = args
        self.iteration_index = iteration
        self.check_erg = False
        self.result = f"{args.jobname}_{self.iteration_index}"
        self.next_sample = next_sample
        if prefix is not None:
            self.prefix = f"Iteration {prefix}/{total_iterations}"
        else:
            self.prefix = f"Iteration {self.iteration_index + 1}/{total_iterations}"
        self.logger = Logger(__name__, level=args.log_level, prefix=self.prefix).get()
        self.signal_file_path = (
            f"{self.args.path}/{self.args.jobname}_{self.iteration_index}.sig"
        )
        safe_remove(self.signal_file_path)
        self.type_filter = False
        self.args.original_session = self.args.session
        if self.iteration_index == 1:
            self.check_erg = True

    def run(
        self,
        file_modifier: FileModifier,
        lock,
        type_filter: bool = False,
    ) -> np.ndarray:
        """
        Run the complete simulation pipeline:
        - Modify input files
        - Run OFsolve
        - Run OpenForm
        - Parse result
        - Read output

        Args:
            file_modifier: FileModifier instance to update design parameters.
            lock: Multiprocessing lock for file-safe access. The lock is already acquired and released during the OFsolve execution (inside ProgressWatcher).
            type_filter: weather to or not indicate the jobname in the large CSV file

        Returns:
            y: np.ndarray containing the result values.
        """
        self.type_filter = type_filter
        self._run_ofsolve(file_modifier, lock)
        self._run_openform(lock)
        y = self._parse_results(lock)

        return y

    def _setup_file(self, file_modifier=None):
        """
        Modify the input file

        Args:
            file_modifier: FileModifier instance.
        """
        input_prams = {}
        for index, value in enumerate(self.args.x_fields):
            input_prams[value] = float(self.next_sample[-1, index])

        file_modifier.set_design_parameters(input_prams)
        file_modifier.print()

    def _run_ofsolve(self, file_modifier=None, lock=None):
        """
        Modify the input file and execute OFsolve.

        Args:
            file_modifier: FileModifier instance.
            lock: Multiprocessing lock. This lock is already acquired and released during the OFsolve execution (inside ProgressWatcher).
        """
        self._setup_file(file_modifier=file_modifier)
        self.logger.info("Executing OFsolve")
        transient = False
        start_time = time.time()
        pause_event = multiprocessing.Event()
        command = (
            f"export GNS_LICENSE_SERVER={self.args.license_server}"
            f" GNS_LICENSE_SERVER_PORT_IPv6={self.args.license_port}"
            f" GNS_LICENSE_SERVER_TYPE={self.args.license_type} && "
            f"cd {self.args.path} && {self.args.ofsolver} -j {self.args.jobname} -c {self.args.cores} -o {self.result}"
        )
        self.logger.debug(f"Executing command: {command}")

        # Process to monitor time limit is not reached
        if self.args.time_limit is not None:
            self.logger.info(
                f"Time limit set to {convert_seconds_to_hms(self.args.time_limit)}"
            )
            stop_timer_event = multiprocessing.Event()
            timer = multiprocessing.Process(
                target=timer_process,
                args=(self.args, self.signal_file_path, stop_timer_event),
                daemon=True,
            )
            timer.start()
        else:
            self.logger.debug("No time limit specified.")

        if self.args.approximate_computing_check is not None:
            stop_approx_comp_event = multiprocessing.Event()
            approx_comp_proc = multiprocessing.Process(
                target=self._approximate_computing,
                args=(lock, stop_approx_comp_event, pause_event),
                daemon=True,
            )
            approx_comp_proc.start()
            # transient = True  # clear progress bar on exit

        with ProgressWatcher(
            f"{self.args.path}/{self.result}.log",
            log_level=self.args.log_level,
            name=__name__,
            prefix=self.prefix,
            lock=lock,
            watcher_backend=self.args.watcher_backend,
            pause_event=pause_event,
            transient=transient,
        ):
            _, err = execute_block(
                command,
                raise_exception=False,
                options={"check": False},
                log_level=self.args.log_level,
                log_prefix=self.prefix,
            )
        # If a timer was started, signal it to stop and wait for it to finish.
        if self.args.time_limit is not None:
            stop_timer_event.set()
            timer.join()

        if self.args.approximate_computing_check is not None:
            stop_approx_comp_event.set()
            approx_comp_proc.join()

        if "error" in err.lower():
            self.logger.error(
                f"An error occurred. Terminating optimization. Error was:{err}"
            )
            sys.exit(1)
        else:
            self.logger.info(
                f"OFsolve finished in {str(datetime.timedelta(seconds=int(time.time() - start_time))).zfill(8)}"
            )

    def _run_openform(self, lock: multiprocessing.Lock):
        """
        Create a session file and run OpenForm.

        Arguments:
            lock (multiprocessing.Lock): A lock object ensuring process-safe operations
                when accessing shared resources and approximated computing is on.
        """
        # lock the session file to avoid race conditions when multiple processes
        # try to write or before passing it to openForm
        if lock is not None and self.args.approximate_computing_check is not None:
            lock.acquire()
            self.logger.debug(f"Acquired lock to write and read to {self.args.session}")

        self._create_seesion_file()
        self.logger.info(f"Executing Openform on {self.args.session}")
        start_time = time.time()
        lib_path = os.path.dirname(self.args.openform) + "/lib"
        command = (
            f"export GNS_LICENSE_SERVER={self.args.license_server}"
            f" GNS_LICENSE_SERVER_PORT_IPv6={self.args.license_port}"
            f" GNS_LICENSE_SERVER_TYPE={self.args.license_type}"
            f" LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{lib_path}:{lib_path}/egl && "
            f"cd {self.args.path}  && {self.args.openform} -s {self.args.session}"
        )
        out, err = execute_block(
            command,
            raise_exception=False,
            options={"check": False},
            log_level=self.args.log_level,
            log_prefix=self.prefix,
        )

        if lock is not None and self.args.approximate_computing_check is not None:
            lock.release()
            self.logger.debug("Released lock")

        time.sleep(1)
        if "All licenses in use" in err:
            self.logger.critical("An error occurred. Trying to restart")
            _, _ = execute_block(f"pkill -f {os.path.basename(self.args.openform)}")
            out, err = execute_block(
                command,
                raise_exception=False,
                options={"check": False},
                log_level=self.args.log_level,
                log_prefix=self.prefix,
            )

        self._save_session_file()
        if "Error:" in err:
            self.logger.error("An error occurred. Terminating optimization")
            print(err)
            sys.exit(1)
        else:
            self.logger.info(
                f"Openform finished in {str(datetime.timedelta(seconds=int(time.time() - start_time))).zfill(8)}"
            )

    def _parse_results(self, lock=None) -> np.ndarray:
        """
        Read the last output row from the simulation result CSV and parse it.

        Args:
            lock: Multiprocessing lock.

        Returns:
        y: np.ndarray containing the result values.
        """
        self.logger.info("Parsing results ")
        start_time = time.time()
        if lock is not None:
            lock.acquire()
            self.logger.debug(f"Acquired lock to write and read to {self.args.output}")

        self._run_c_parser()
        y = self._read_result(self.args.y_fields)
        if lock is not None:
            lock.release()
            self.logger.debug("Released lock")

        self.logger.info(
            f"Parsing finished in {str(datetime.timedelta(seconds=int(time.time() - start_time))).zfill(8)}"
        )
        return y

    def _run_c_parser(self):
        """
        Run the external CSV parser on the simulation result.
        """
        self.logger.info("Step 1: Parsing result to CSV")
        parser_path = csv_parser_path()
        erg_path = f"{self.args.path}/{self.result}.erg"
        small_csv = f"{erg_path}/{self.result}.csv"

        # check the file is not empty
        if not os.path.isfile(small_csv) or os.path.getsize(small_csv) == 0:
            self.logger.error(f"Error: File '{small_csv}' is missing or empty.")
            sys.exit(1)

        command = f"cd {erg_path}  && {parser_path} {small_csv} {self.args.output}"
        # Indicate the type if needed in the labeled CSV file
        if self.type_filter:
            command += " --append-name"

        _, _ = execute_block(
            command,
            raise_exception=False,
            options={"check": False},
            log_level=self.args.log_level,
            log_prefix=self.prefix,
        )

        self.logger.debug(
            f"Appended results from {os.path.basename(small_csv)} to {os.path.basename(self.args.output)}"
        )
        self.logger.debug(f"Appended results from {small_csv} to {self.args.output}")

    def _read_result(self, y_fields: list[str] = None) -> np.ndarray:
        """
        Read the last output row from the simulation result CSV.

        Args:
            y_fields (list[str]): List of output variable names (str) from the simulation.

        Returns:
            y: np.ndarray containing the result values.
        """
        self.logger.info(f"Step 2: Reading new results from {self.args.output}")
        _, y = read_data(
            self.args.output,
            [],
            y_fields,
            log_level=self.args.log_level,
            lines_to_read=10,
        )
        self.logger.debug(f"Raw output data shape: {y.shape}")
        y = y[-1, :].reshape(1, -1)
        self.logger.debug(f"Extracted result row: {y}")
        return y

    def _read_csv(self, lock: multiprocessing.Lock) -> list[float]:
        """
        Read and parse values from the simulation result CSV file.

        Reads the last line of data from the small CSV file in the result directory.
        If the last line is numeric, parse it directly. Otherwise, falls back to
        finding the data line after the '* Formability' header.

        Args:
            lock: Multiprocessing lock to ensure thread-safe file access.
                 Note: Currently the lock is not used in this method but kept
                 for consistency with other file operations.

        Returns:
            list[float]: List of numeric values parsed from the CSV data line.
            Returns an empty list if no valid data is found.
        """
        small_csv = f"{self.args.path}/{self.result}.erg/{self.result}.csv"
        self.logger.debug(f"Reading CSV file: {small_csv}")
        if not os.path.isfile(small_csv) or os.path.getsize(small_csv) == 0:
            self.logger.error(f"Error: File '{small_csv}' is missing or empty.")
            sys.exit(1)

        lines = []
        if lock is not None:
            lock.acquire()

        with open(small_csv, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if lock is not None:
            lock.release()

        last_line = lines[-1]
        # If the last line looks numeric -> parse it
        try:
            return [float(x) for x in last_line.split(",") if x.strip()]
        except ValueError:
            # Fallback: find the line after *Formability
            for i, line in enumerate(lines):
                if line.startswith("*Formability") and i + 1 < len(lines):
                    return [float(x) for x in lines[i + 1].split(",")]

        return []

    def _create_seesion_file(self):
        """
        Create a modified OpenForm session file by replacing jobname with result name.
        Do this only once per simulation
        """

        old_file = os.path.join(self.args.path, self.args.original_session)
        # check that the template has the right lines
        self._check_and_adjust_session_file()
        self.logger.info(f"Creating session file from template: {old_file}")
        with open(old_file, encoding="utf-8") as f_in:
            content = f_in.read()

        name, ext = os.path.splitext(self.args.original_session)
        self.args.session = f"{name}_{self.iteration_index}{ext}"

        updated_content = content.replace(self.args.jobname, self.result)
        new_file = os.path.join(self.args.path, self.args.session)
        with open(new_file, "w", encoding="utf-8") as f_out:
            f_out.write(updated_content)

        self.logger.debug(f"Modified session file created: {new_file}")
        self.logger.debug(
            f"Jobname '{self.args.jobname}' replaced with '{self.result}'"
        )
        self.crete_session = False

    def _save_session_file(self):
        """
        Move the updated session file into the result folder for archival.

        The file is moved from <path>/<session> to <path>/<result>.erg/<session>
        """
        res_folder = os.path.join(self.args.path, f"{self.result}.erg")
        os.makedirs(res_folder, exist_ok=True)
        self.logger.debug(f"Ensured result folder exists: {res_folder}")

        source = os.path.join(self.args.path, self.args.session)
        destination = os.path.join(res_folder, self.args.session)
        os.replace(source, destination)

        self.logger.info(f"Session file moved to result archive: {destination}")
        self.logger.debug(f"Session file moved from {source} to {destination}")

    def _check_and_adjust_session_file(self):
        """
        Replace everything up to and including '.erg' with a new root path inside the session file,
        but only if it is not already set.
        """
        if self.check_erg:
            session_file = os.path.join(self.args.path, self.args.session)
            self.logger.info(f"Checking {session_file}")
            # first backup the file:
            backup_dir = os.path.join(self.args.path, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            backup_file = os.path.join(backup_dir, self.args.session)
            with open(session_file, "rb") as f_in, open(backup_file, "wb") as f_out:
                f_out.write(f_in.read())
            self.logger.info(f"Backup created: {backup_file}")

            # now replace correct content:
            res_folder = os.path.join(self.args.path, f"{self.args.jobname}.erg")
            lines = []
            with open(session_file, encoding="utf-8") as f_in:
                line = f_in.read()
                if ".erg/" in line:
                    erg_index = line.find(".erg/")
                    if erg_index != -1:
                        # Look backwards from '.erg/' to find the opening quote
                        start_quote_index = line.rfind('"', 0, erg_index)
                        if start_quote_index != -1:
                            # Extract current root path
                            current_root = line[
                                start_quote_index + 1 : erg_index + 4
                            ]  # include '.erg'
                            if not current_root.startswith(res_folder):
                                self.logger.info(
                                    f"Replacing root '{current_root}' with '{res_folder}'"
                                )
                                line = line.replace(current_root, res_folder)
                    # replace csv name with jobname
                    pattern = r'(\.erg/)([^"]+?)((?:\.csv|\.png)")'
                    line, n = re.subn(
                        pattern,
                        lambda m: f"{m.group(1)}{self.args.jobname}{m.group(3)}",
                        line,
                    )
                    if n > 0:
                        self.logger.debug(
                            f"Replaced {n} CSV/PNG reference(s) with jobname '{self.args.jobname}'"
                        )
                lines.append(line)

            with open(session_file, "w", encoding="utf-8") as f_out:
                f_out.writelines(lines)

            self.logger.info("Checking passed")
            self.check_erg = False

    def _approximate_computing(
        self,
        lock: multiprocessing.Lock,
        stop_approx_comp_event: multiprocessing.Event,
        pause_watcher_event: multiprocessing.Event,
    ):
        """
        Approximates computation process by managing parallel processes, monitoring, and
        handling signals. Uses multiprocessing synchronization primitives for thread-safe
        operations and communicates via signal events.

        Arguments:
            lock (multiprocessing.Lock): Ensures thread-safe operations when accessing shared resources.
            stop_approx_comp_event (multiprocessing.Event): Signals to stop the approximate computing process.
            pause_watcher_event (multiprocessing.Event): Signals to pause/resume the progress watcher.


        Raises:
            No explicit exceptions raised directly in this method.
        """
        stop = False

        for limit in self.args.approximate_computing_check:
            self.logger.info(f"Approximated computing check set at {limit}%")

            log_file = f"{self.args.path}/{self.result}.log"
            monitor_and_create_signal(
                self.args,
                self.signal_file_path,
                log_file,
                stop_approx_comp_event,
                limit,
                # self.logger, # for debugging
            )
            if pause_watcher_event and not pause_watcher_event.is_set():
                pause_watcher_event.set()  # signal watcher to pause

            s = (
                f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                f"Approximated computing reached {limit}% -- Checking"
                f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            )
            # if "rich" in self.args.watcher_backend:
            #     print(s)
            # else:
            self.logger.info(s)
            self._run_openform(lock)
            y = self._read_csv(lock)
            self.logger.info(f"y: {y} ")
            self.logger.info(f"Limits: {self.args.approximate_computing_limit}")
            # stop simulation if limits have been exceeded
            if np.any(np.array(y) >= np.array(self.args.approximate_computing_limit)):
                self.logger.info(
                    "Approximated computing terminating current simulation as limit have been reached"
                )
                decision = "Stop Simulation"
                write_signal(self.args, self.signal_file_path, "STOP")
                stop_approx_comp_event.set()
                stop = True
            # continue simulation if limits have not been reached
            else:
                stop_approx_comp_event.clear()
                decision = "Continue Simulation"
            s = (
                f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                f"Approximated computing check ended: {decision} "
                f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            )
            if "rich" in self.args.watcher_backend:
                self.logger.info("\n" + s + "\n")
                # print(s + "\n")
            else:
                self.logger.info("\n" + s)
            # Resume watcher after iteration
            if pause_watcher_event and pause_watcher_event.is_set():
                pause_watcher_event.clear()  # signal watcher to resume
            if stop:
                return
