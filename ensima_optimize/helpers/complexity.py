"""
Computes geometric complexity metrics for stamping parts from 3D coordinate data.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import numpy as np

from ensima_optimize.classes.logger import Logger
from ensima_optimize.helpers.read_geometry import read_coordinates_from_file

logger = Logger(__name__, level="INFO").get()


def compute_complexity(parts):
    logger = Logger(__name__, level="INFO").get()

    for part, geometry_path in parts:
        coords = read_coordinates_from_file(geometry_path, log_level="WARNING")
        if len(coords) == 0:
            logger.warning(f"{part}: No coordinates found")
            continue

        geo_x = coords[:, 0]
        geo_y = coords[:, 1]
        geo_z = coords[:, 2]

        A = np.max(geo_x) - np.min(geo_x)  # in-plane x
        B = np.max(geo_y) - np.min(geo_y)  # in-plane y
        L = np.sqrt(A**2 + B**2)  # diagonal length
        h = np.max(geo_z) - np.min(geo_z)  # depth / punch stroke

        # Classify part and assign number
        if L / B > 3.0 and h / B > 0.3:
            part_type = "Beam"
            part_number = 2
        elif A / B <= 3.0 and h / A <= 0.2:
            part_type = "Flat"
            part_number = 1
        elif A / B <= 3.0 and h / B > 0.2:
            part_type = "Deep"
            part_number = 3
        else:
            part_type = "Test"
            part_number = 0

        logger.info(f"{part}: {part_type} (Type {part_number})")


if __name__ == "__main__":
    parts = [
        (
            "ASaeule",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_01_Flat/ASaeule.t52",
        ),
        (
            "BSaeule",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_02_Beam/BSaeule_DX56D.t52",
        ),
        (
            "RadhausAdapter",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_03_Deep/RadhausAdapter.t52",
        ),
        (
            "Einleger",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_04/Einleger.t52",
        ),
        (
            "ASaeule_BSym",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/T52-PartFiles/A-Pilar.t52",
        ),
        (
            "Quertraeger",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/T52-PartFiles/CrossBeam.t52",
        ),
        (
            "FrontFender_01",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/T52-PartFiles/FrontFender_A.t52",
        ),
        (
            "Kotfluegel-DC",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/T52-PartFiles/FrontFender_B.t52",
        ),
        (
            "Heckklappe_aussen",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/T52-PartFiles/TailGate.t52",
        ),
        (
            "Tankdeckeleinsatz",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/T52-PartFiles/TankCapInsert.t52",
        ),
        (
            "Tunnel_Passat",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/T52-PartFiles/Tunnel.t52",
        ),
        (
            "SeatShell",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/new_parts/PartType_03/SeatShell.t52",
        ),
        (
            "DACH-VWS-Session.ofs",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/new_parts/PartType_01/DACH-VWS.t52",
        ),
        (
            "Laengstraeger_02-Session.ofs",
            "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/new_parts/PartType_02/Laengstraeger_02.t52",
        ),
    ]
    compute_complexity(parts)
