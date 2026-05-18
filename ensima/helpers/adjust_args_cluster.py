"""
Adjusts file paths and execution arguments for HPC cluster environments.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import socket
from argparse import Namespace

from ensima.classes.logger import Logger

local_hostname = "electric"


def adjust_parts_for_cluster(
    args: Namespace,
    parts: list[tuple[str, str]] = None,
    old="/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
    # new="/rwthfs/rz/cluster/home/cg021604/gns/TCO-Benchmark",
    new="/rwthfs/rz/cluster/home/qfw89470/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
):
    hostname = socket.gethostname().lower()
    logger = Logger(__name__, level=args.log_level).get()
    logger.info(f"Adjusting parts for {hostname}")
    if parts is not None:
        adjusted_parts = []
        for name, part_path in parts:
            new_path = part_path.replace(
                old,
                new,
            )
            logger.debug(f"Adjusted part path for {name}: {new_path}")
            adjusted_parts.append((name, new_path))
        parts = adjusted_parts
    return parts


############################## Aachen
def adjust_args_for_aachen_cluster(args: Namespace) -> Namespace:
    """
    Adjusts command line arguments for execution in a cluster environment.

    This function modifies paths and settings in the provided arguments object
    when running on a cluster (specifically when 'electric' is in the hostname).
    It updates solver paths, OpenForm path, data paths, and license settings.

    Args:
        args (Namespace): The command line arguments namespace object.

    Returns:
        Namespace: The modified arguments namespace object with cluster-specific settings.
    """

    # Adjust for cluster
    logger = Logger(__name__, level=args.log_level).get()
    hostname = socket.gethostname().lower()
    logger.info(f"Adjusting args for {hostname}")
    # args.ofsolver = "/rwthfs/rz/cluster/home/cg021604/gns/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe"
    # better, but does not produce the files needed
    # args.ofs = "/home/rwth1453/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe",
    # args.openform = (
    #     "/rwthfs/rz/cluster/home/cg021604/gns/OpenForm_daily_linux64/OpenForm_64_batch"
    # )
    # args.path = args.path.replace(
    #     "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
    #     "/rwthfs/rz/cluster/home/cg021604/gns/TCO-Benchmark",
    # )
    # args.output = args.output.replace(
    #     "/d/gitlab/ensima-code/optimization/test/csv",
    #     "/rwthfs/rz/cluster/home/cg021604/ensima-code/optimization/test/csv",
    # )
    # if args.geometry_path is not None:
    #     args.geometry_path = args.geometry_path.replace(
    #         "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
    #         "/rwthfs/rz/cluster/home/cg021604/gns/TCO-Benchmark",
    #     )
    args.ofsolver = "/rwthfs/rz/cluster/home/qfw89470/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe"
    args.openform = (
        "/rwthfs/rz/cluster/home/qfw89470/gns/OpenForm_daily_linux64/OpenForm_64_batch"
    )
    args.path = args.path.replace(
        "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
        "/rwthfs/rz/cluster/home/qfw89470/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
    )
    args.output = args.output.replace(
        "/d/gitlab/ensima-code/optimization/test/csv",
        "/rwthfs/rz/cluster/home/qfw89470/ensima-code/optimization/test/csv",
    )
    if args.geometry_path is not None:
        args.geometry_path = args.geometry_path.replace(
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
            "/rwthfs/rz/cluster/home/qfw89470/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
        )

    args.cores = 64
    args.license_server = "license.itc.rwth-aachen.de"
    args.license_port = "50141"
    args.license_type = "RLM"
    args.license_server_service = True

    return args


def adjust_args_and_parts_for_aachen_cluster(
    args: Namespace, parts: list[tuple[str, str]] = None
) -> tuple[Namespace, list[tuple[str, str]]]:
    args = adjust_args_for_aachen_cluster(args)
    parts = adjust_parts_for_cluster(args, parts)
    return args, parts


############################### GPU server
def adjust_args_for_gpu_server(args: Namespace) -> Namespace:
    """
    Adjusts command line arguments for execution in a cluster environment.

    This function modifies paths and settings in the provided arguments object
    when running on a cluster (specifically when 'electric' is in the hostname).
    It updates solver paths, OpenForm path, data paths, and license settings.

    Args:
        args (Namespace): The command line arguments namespace object.

    Returns:
        Namespace: The modified arguments namespace object with cluster-specific settings.
    """

    # Adjust for cluster
    hostname = socket.gethostname().lower()
    logger = Logger(__name__, level=args.log_level).get()
    logger.info(f"Adjusting args for {hostname}")
    args.ofsolver = "/home/ahmadtarraf/ensima-code/OpenForm-Solver/OFSolv_V2.16.0-E/bin/OFSolv_1.0.4e_eng_linux64.exe"
    args.openform = "/home/ahmadtarraf/ensima-code/test_data/gns/OpenForm_daily_linux64/OpenForm_64_batch"
    args.path = args.path.replace(
        "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
        "/home/ahmadtarraf/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
    )
    args.output = args.output.replace(
        "/d/gitlab/ensima-code/optimization/test/csv",
        "/home/ahmadtarraf/ensima-code/optimization/test/csv",
    )
    if args.geometry_path is not None:
        args.geometry_path = args.geometry_path.replace(
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
            "/home/ahmadtarraf/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
        )
    args.cores = 48
    args.license_server = "rms14562"
    args.license_port = "5053"
    args.license_type = "RLM"
    args.license_server_service = False
    args.license_rlm = "/home/ahmadtarraf/ensima-code/test_data/gns/gns_rlm_v15.2_R1_install_linux64/rlm "

    return args


def adjust_args_and_parts_for_gpu_server(
    args: Namespace, parts: list[tuple[str, str]] = None
) -> tuple[Namespace, list[tuple[str, str]]]:
    args = adjust_args_for_gpu_server(args)
    parts = adjust_parts_for_cluster(
        args,
        parts,
        new="/home/ahmadtarraf/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark",
    )
    return args, parts


###########################################################


def adjust_args_and_parts_for_cluster(args, parts):
    """
    Adjusts arguments and parts configurations based on the cluster environment.

    This function determines the hostname of the environment in which the code is
    executing and modifies the `args` and `parts` variables depending on the
    cluster type. Specific adjustments are made for GPU server or Aachen cluster
    environments, if applicable.

    Parameters:
    args: dict
        The dictionary of arguments to be adjusted for the cluster environment.
    parts: dict
        The dictionary of parts configuration to be adjusted for the cluster
        environment.

    Returns:
    tuple
        A tuple containing the adjusted `args` and `parts` dictionaries.
    """
    hostname = socket.gethostname().lower()
    if local_hostname not in hostname:
        if "rms14562" in hostname:
            args, parts = adjust_args_and_parts_for_gpu_server(args, parts)
        else:
            args, parts = adjust_args_and_parts_for_aachen_cluster(args, parts)
    return args, parts


def adjust_args_for_cluster(args):
    """
    Adjust arguments for the cluster or server.

    This function modifies the input arguments based on the hostname of the
    machine. If the machine's hostname matches certain patterns, the arguments
    will be adjusted accordingly for the specific server or cluster.

    Parameters:
    args: The input arguments to be adjusted.

    Returns:
    The adjusted arguments.
    """
    hostname = socket.gethostname().lower()
    if local_hostname not in hostname:
        if "rms14562" in hostname:
            args = adjust_args_for_gpu_server(args)
        else:
            args = adjust_args_for_aachen_cluster(args)
    return args
