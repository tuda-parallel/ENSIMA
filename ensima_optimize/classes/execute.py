"""
Utilities for executing shell commands, managing subprocesses, and monitoring simulation runs.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import multiprocessing
import os
import subprocess
import time
from argparse import Namespace

from rich.markup import escape
from rich.panel import Panel
from rich.status import Status

from ensima_optimize.classes.logger import Logger
from ensima_optimize.classes.print import MyConsole
from ensima_optimize.helpers.misc import safe_remove
from ensima_optimize.helpers.read_data import read_last_progress
from ensima_optimize.helpers.units import convert_seconds_to_hms


def execute_block(
    call: str,
    raise_exception: bool = True,
    dry_run: bool = False,
    options: dict = None,
    log_level: str = "WARNING",
    log_prefix="",
) -> tuple[str, str]:
    """Executes a shell command and blocks until it finishes.

    Args:
        call (str): Shell command to execute.
        raise_exception (bool): Whether to raise an exception on failure.
        dry_run (bool): If True, simulate the run without executing.
        options (dict, optional): Additional subprocess.run options.
        log_level (str, optional): Logging level (default = "WARNING")
        log_prefix (str, optional): Logging prefix (default = "")


    Returns:
        str: Combined stdout of the process.
    """
    logger = Logger(__name__, log_level, prefix=log_prefix).get()
    logger.debug(f" Executing {call}")
    if dry_run:
        print(f"[DRY RUN] {call}")
        return ""

    out = ""
    options = options or {}

    # Default subprocess options
    subprocess_options = {
        "shell": True,
        "text": True,
        "capture_output": True,
        "check": True,
        "executable": "/bin/bash",
        "env": os.environ,
    }
    subprocess_options.update(options)
    logger.debug(f"Executing: {call}\n")

    try:
        result = subprocess.run(call, **subprocess_options)
        out = result.stdout
        err = result.stderr
        logger.debug(f"Command stdout: {out}")
        logger.debug(f"Command stderr: {err}")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Command failed: {call}\n"
            f"Exit code: {e.returncode}\n"
            f"Output: {e.stdout.strip()}\n"
            f"Error: {e.stderr.strip()}"
        )
        logger.error(f"{error_message}\n[/]")
        if raise_exception:
            raise

    return out, err


def execute_block_and_log(
    call: str,
    log_file: str,
    dry_run: bool = False,
    path: str = "./",
    status: Status = None,
    log_level: str = "WARNING",
) -> float:
    """Executes a call and logs it's output. This is a blocking call that
    writes the output to the log once finished.

    Args:
        call (str): Bash call to execute
        log_file (str): Absolute location of the log file
        dry_run (bool): Simulate run if true
        path (str): Path to execute the command
        status (Status): Status for logging
        log_level (str, optional): Logging level (defult = "WARNING")


    Returns:
        float: execution time of the call
    """
    logger = Logger(__name__, log_level).get()
    if status:
        status.update(f">> Executing {call} in {path}")
    else:
        logger.debug(f"Executing {call} in {path}")
    call = f"cd {path} && {call}"
    log_message = f">> Executing command: {call}\n"
    start = time.time()
    end = start
    if not dry_run:
        try:
            out = subprocess.run(
                call,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                executable="/bin/bash",
                env=os.environ,
            )
            end = time.time()
            log_message += f"Output:\n{out.stdout}\n"
        except subprocess.CalledProcessError as e:
            log_message += f"Error:\n{e.stderr}\n"
            error_message = (
                f"Command failed:{call}\n"
                f"Exit code:{e.returncode}\n"
                f"Output:{e.stdout.strip()}\n"
                f"Error:{e.stderr.strip()}"
            )
            logger.error(f"{error_message}[/]")
            raise
        finally:
            # Write the log message to the file
            with open(log_file, "a") as file:
                file.write(log_message)
    return end - start


def execute_background(
    call: str,
    log_file: str = "",
    log_err_file: str = "",
    dry_run=False,
    options: dict = None,
    log_level: str = "WARNING",
    raise_exception: bool = True,
    log_prefix: str = "",
):
    """executes a call in the background and sets up a log dir

    Args:
        call (str): call to execute
        log_file (str): Absolute location of the log file
        log_err_file (str): Absolute location of the error log file
        dry_run (bool): Simulate run if true
        options (dict, optional): Additional subprocess.run options.
        log_level (str, optional): Logging level (defult = "WARNING")
        raise_exception (bool): Whether to raise an exception on failure.
        log_prefix (str, optional): Prefix for logger messages.


    Returns:
        _type_: _description_
    """
    options = options or {}
    logger = Logger(__name__, log_level, prefix=log_prefix).get()
    logger.debug(f" >> Executing (background) {call}")

    if dry_run:
        print(f"[DRY RUN] {call}")
        return None

    # Default Popen options
    popen_options = {
        "shell": True,
        "executable": "/bin/bash",
        "env": os.environ,
        **options,
    }

    if log_file and log_err_file:
        log_out = open(log_file, "a")
        log_err = open(log_err_file, "a")
        popen_options.update({"stdout": log_out, "stderr": log_err})
    elif log_file:
        log_out = open(log_file, "a")
        popen_options.update({"stdout": log_out, "stderr": log_out})
    else:
        popen_options.update(
            {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
        )

    try:
        process = subprocess.Popen(call, **popen_options)
    except Exception as e:
        logger.error(f"Failed to start process: {call}\nError: {e}")
        if raise_exception:
            raise
        return None

    return process


def execute_background_and_log(
    call: str, log_file: str, name="", err_file: str = "", dry_run=False
) -> subprocess.Popen:
    """execute call in background and returns process. The output is displayed using a
    thread that reads the log file

    Args:
        call (str): bash call to execute
        log_file (str): absolute location of the log file to monitor
        name (str, optional): The src of the file. If set to daemon, proxy, or ftio, the output
        is colored. Defaults to "".. Defaults to "".
        dry_run (bool): simulate run if true

    Returns:
        subprocess.Popen: process
    """
    process = execute_background(call, log_file, err_file, dry_run)
    _ = monitor_log_file(log_file, name)
    return process


def monitor_log_file(file: str, src: str = "") -> multiprocessing.Process:
    """monitors a file and displays its output on the console. A process is
    in charge of monitoring the file.

    Args:
        file (str): absolute File path
        src (str, optional): The src of the file. If set to daemon, proxy, or ftio, the output
        is colored. Defaults to "".
    """
    monitor_process = multiprocessing.Process(target=print_file, args=(file, src))
    monitor_process.daemon = True
    monitor_process.start()

    return monitor_process


def print_file(file, src=""):
    """Continuously monitor the log file for new lines and print them."""
    color = ""
    close = ""
    newline = True
    console = MyConsole(True)
    wait_time = 0.05
    if src:
        if "open" in src.lower():
            color = "[purple4]"
            wait_time = 0.1
        elif "of" in src.lower():
            color = "[deep_sky_blue1]"
            wait_time = 0.1
        elif "error" in src.lower():
            color = "[red]"
            wait_time = 0.1
        else:
            color = "[gold3]"
            wait_time = 0.1

        if color:
            close = "[/]"
            newline = "\n"

    while not os.path.exists(file):
        if "error" in src.lower():
            time.sleep(0.1)
        else:
            with console.status(f"[bold green]Waiting for {file} to appear ..."):
                time.sleep(0.1)

    with open(file) as file:
        # Go to the end of the file
        file.seek(0, os.SEEK_END)
        buffer = []
        last_print_time = time.time()

        while True:
            line = file.readline()
            if line:
                buffer.append(line.rstrip())
            else:
                # If there's no new line, wait briefly
                time.sleep(wait_time)

            # Group and print the buffered lines every 0.1 seconds
            current_time = time.time()
            if current_time - last_print_time >= wait_time and buffer:
                # Print grouped lines
                content = "\n".join(buffer)
                buffer.clear()
                last_print_time = current_time

                if not src or "cargo" in src:
                    print(content)
                else:
                    if newline:
                        console.print("\n", end="")
                        console.print(
                            Panel.fit(
                                color + escape(content) + close,
                                title=color + src.upper() + close,
                                style="white",
                                border_style="white",
                                title_align="left",
                            )
                        )


def timer_process(
    args: Namespace,
    signal_file_path: str,
    stop_event: multiprocessing.Event,
):
    """
    A separate thread that waits for the specified time limit.

    Args:
        args (Namespace): contains the time limit in seconds and the path to the signal file
        signal_file_path (str): The path to the signal file.
        stop_event (multiprocessing.Event): An event to signal the thread to stop.
    """
    logger = Logger(__name__, args.log_level).get()
    # Define the path to the signal file to check
    # log_path = signal_file_path.replace("sig","log")
    # mod_time = os.path.getmtime(log_path)
    # clean start
    safe_remove(signal_file_path)
    start_time = time.monotonic()
    flag = False
    while not stop_event.is_set():
        elapsed_time = time.monotonic() - start_time
        # if flag and os.path.getmtime(log_path) > mod_time:
        if flag and not os.path.exists(
            signal_file_path
        ):  # signal with OUTPUT is often not deleted immediately
            logger.warning(
                f"Signal file {signal_file_path} was deleted. Output saved successfully. Graceful shutdown initiated."
            )
            # Change the signal to 'stop' and write it to the file
            with multiprocessing.Lock():
                write_signal(args, signal_file_path, "STOP")
            # Signal the main process to stop and exit this process
            stop_event.set()
            return

        if elapsed_time > args.time_limit and not flag:
            logger.warning(
                f"Terminating simulation as time limit was reached after {convert_seconds_to_hms(args.time_limit)}."
            )
            with multiprocessing.Lock():
                write_signal(args, signal_file_path, "OUTPUT")
                flag = True
                time.sleep(1)
        time.sleep(5)


def monitor_and_create_signal(
    args: Namespace,
    signal_file_path: str,
    log_file: str,
    stop_event: multiprocessing.Event,
    limit: float = 100,
    logger=None,
):
    """
    A separate thread that waits for the specified time limit.

    Args:
        args (Namespace): contains the time limit in seconds and the path to the signal file
        signal_file_path (str): The path to the signal file.
        log_file (str): The path to the log file
        stop_event (multiprocessing.Event): An event to signal the thread to stop.
        limit (float, optional): The limit in percent. Defaults to 100.
        logger (Logger, optional): The logger instance. Defaults to None. If provided,
            debug is set to true
    """
    if limit > 100:
        return
    counter = 0
    if logger is None:
        logger = Logger(__name__, args.log_level).get()
        debug = False
    else:
        debug = True
    safe_remove(signal_file_path)
    flag = False
    logger.debug(f"Scheduled intermediate output at {limit}% on log file {log_file}")
    clean_start = False
    sleep_time = 0.1
    while not stop_event.is_set():
        step, progress = read_last_progress(log_file)
        # make sure the results are not from some intermediate simulation
        # When the code is executed fast, it makes sense to check step 2 also and have
        # the counter (5 times in step 3) as a backup
        if (step == 1 or step == 2 or counter == 5) and not clean_start:
            clean_start = True
            sleep_time = 5
            if debug:
                logger.debug("Clean start set to True ")

        # if flag and os.path.getmtime(log_path) > mod_time:
        elif step == 3:  # step 3
            if flag:
                if not os.path.exists(signal_file_path):
                    logger.debug(
                        f"Signal file {signal_file_path} was deleted. Output created successfully."
                    )
                    stop_event.set()
                    return
            else:
                if progress > limit and clean_start:
                    logger.debug(f"Creating intermediate output at {progress}%.")
                    with multiprocessing.Lock():
                        write_signal(args, signal_file_path, "OUTPUT")
                        flag = True
                        time.sleep(1)
            if counter < 6:
                counter += 1
        if debug:
            logger.debug(
                f"step {step} -- progress {progress}%  -- limit {limit}% -- clean_start {clean_start} -- condition {step == 3 and progress > limit and clean_start}"
            )

        time.sleep(sleep_time)


def write_signal(args: Namespace, signal_file_path: str, signal: str = "STOP"):
    """
    Writes a signal string to a specified file located in a directory with a given job name.
    Args:
        args (Namespace): An object containing two attributes:
                          - path (str): The directory where the signal file will be saved.
                          - jobname (str): The name of the job used to create the file name.
        signal_file_path (str): The path to the signal file.
        signal (str): The signal string to be written to the file.

    Returns:
        None
    """
    with open(signal_file_path, "w") as f:
        f.write(f"{signal}\n")
    # Create the file on startup for the timer thread to find.
    logger = Logger(__name__, args.log_level).get()
    logger.debug(f"Signal '{signal}' written to {signal_file_path}")
