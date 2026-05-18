"""
Estimates CO2 emissions from simulation energy consumption.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

# 400-600g CO2/kWh for natural gas
# 800g CO2/kWh for oil
# and 800-1200g CO2/kWh for coal
# 3,6e+6 J = 1 kWh


def estimate_co2(energy: float) -> float:
    return energy / (3.6e6) * 800
