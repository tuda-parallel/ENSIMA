"""
Pytest configuration and shared fixtures.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import os
import re
import shutil
import subprocess
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_TIMESTAMP_RE = re.compile(r".*_?\d{8}_\d{6}$")


def _remove_test_outputs(base_dir: str, created_after: float) -> None:
    """Remove result dirs and logs/ created by parse_arguments during this session."""
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        if (
            os.path.isdir(path)
            and _TIMESTAMP_RE.match(entry)
            and os.path.getmtime(path) >= created_after
        ):
            shutil.rmtree(path, ignore_errors=True)
    logs_dir = os.path.join(base_dir, "logs")
    if os.path.isdir(logs_dir) and os.path.getmtime(logs_dir) >= created_after:
        shutil.rmtree(logs_dir, ignore_errors=True)


@pytest.fixture(autouse=True, scope="session")
def cleanup_test_outputs():
    """Remove timestamped result dirs and logs/ created by parse_arguments after the session."""
    session_start = time.time()
    yield
    _remove_test_outputs(REPO_ROOT, created_after=session_start)
    _remove_test_outputs(os.path.join(REPO_ROOT, "test"), created_after=session_start)


@pytest.fixture
def restore_artifacts():
    """Restore artifacts/ to committed state and remove untracked files after tests."""
    yield
    subprocess.run(["git", "restore", "artifacts/"], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "clean", "-fd", "artifacts/"], cwd=REPO_ROOT, check=True)


@pytest.fixture
def cleanup_log():
    """Remove log.out created during tests."""
    yield
    log_path = os.path.join(REPO_ROOT, "log.out")
    if os.path.exists(log_path):
        os.remove(log_path)
