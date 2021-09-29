# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
from datetime import timedelta
import numpy as np

from tasmania.third_party import cupy, dawn4py, gt4py, numba


# >>> backends
cpu_backend = {"numpy"}

if gt4py:
    gt_cpu_backend = {"gt4py:numpy", "gt4py:gtx86", "gt4py:gtmc"}
    gtc_cpu_backend = {"gt4py:gtc:gt:cpu_ifirst", "gt4py:gtc:gt:cpu_kfirst"}

    if dawn4py:
        # dawn_cpu_backend = {
        #     "gt4py:dawn:naive",
        #     "gt4py:dawn:cxxopt",
        #     "gt4py:dawn:gtx86",
        #     "gt4py:dawn:gtmc",
        # }
        dawn_cpu_backend = set()
    else:
        dawn_cpu_backend = set()
else:
    gt_cpu_backend = gtc_cpu_backend = dawn_cpu_backend = set()

if numba:
    numba_cpu_backend = {"numba:cpu:numpy"}
else:
    numba_cpu_backend = set()

if cupy:
    gpu_backend = {"cupy"}

    if gt4py:
        gt_gpu_backend = {"gt4py:gtcuda"}
        gtc_gpu_backend = {"gt4py:gtc:gt:gpu"}

        if dawn4py:
            dawn_gpu_backend = {
                "gt4py:dawn:cuda",
                "gt4py:dawn:gtcuda",
            }
        else:
            dawn_gpu_backend = set()
    else:
        gt_gpu_backend = gtc_gpu_backend = dawn_gpu_backend = set()

    if numba:
        numba_gpu_backend = {"numba:gpu"}
    else:
        numba_gpu_backend = set()
else:
    gpu_backend = set()
    gt_gpu_backend = gtc_gpu_backend = dawn_gpu_backend = set()
    numba_gpu_backend = set()

gtc_backend = gtc_cpu_backend.union(gtc_gpu_backend)
dawn_backend = dawn_cpu_backend.union(dawn_gpu_backend)
numba_backend = numba_cpu_backend.union(numba_gpu_backend)
cpu_backend = cpu_backend.union(
    gt_cpu_backend,
    gtc_cpu_backend,  # numba_cpu_backend, dawn_cpu_backend
)
gpu_backend = gpu_backend.union(
    gt_gpu_backend,
    gtc_gpu_backend,  # dawn_gpu_backend, numba_gpu_backend
)
backend = cpu_backend  # .union(gpu_backend)
backend_debug = {"numpy"}

# >>> storage info
dtype = (np.float64,)
aligned_index = ((0, 0, 0), (1, 1, 0), (3, 3, 0), (2, 0, 1))

# >>> x-axis
axis_x = {
    "dims": ("x",),
    "units_to_range": {"km": (-100, 100), "m": (0, 2e5)},
    "length": (1, 20),
    "increasing": True,
}

# >>> y-axis
axis_y = {
    "dims": ("y",),
    "units_to_range": {"km": (-100, 100), "m": (0, 2e5)},
    "length": (1, 20),
    "increasing": True,
}

# >>> z-axis
axis_z = {
    "dims": ("air_potential_temperature",),
    "units_to_range": {"K": (270, 400), "degC": (-20, 40)},
    "length": (1, 20),
    "increasing": False,
}

# >>> topography2d
topography = {
    "type": ("flat", "gaussian", "schaer"),  # , "user_defined"
    "time": (timedelta(seconds=1), timedelta(minutes=60)),
    "units_to_max_height": {"km": (0, 2)},
    "units_to_half_width_x": {"m": (1, 50e3)},
    "units_to_half_width_y": {"m": (1, 50e3), "km": (1, 20)},
    "str": ("x + y",),
}

# >>> horizontal boundary
nb = 4
horizontal_boundary_types = ("dirichlet", "identity", "periodic", "relaxed")

# >>> isentropic model
isentropic_state = {
    "air_isentropic_density": {"kg m^-2 K^-1": (10, 1000)},
    "x_velocity_at_u_locations": {
        "m s^-1": (-50, 50),
        # "km hr^-1": (-150, 150),
    },
    "y_velocity_at_v_locations": {
        "m s^-1": (-50, 50),
        # "km hr^-1": (-150, 150),
    },
    "mass_fraction_of_water_vapor_in_air": {
        "g g^-1": (0, 5),
        # "g kg^-1": (0, 5000),
    },
    "mass_fraction_of_cloud_liquid_water_in_air": {
        "g g^-1": (0, 5),
        # "g kg^-1": (0, 5000),
    },
    "mass_fraction_of_precipitation_water_in_air": {
        "g g^-1": (0, 5),
        # "g kg^-1": (0, 5000),
    },
    "number_density_of_precipitation_water": {
        "g^-1": (0, 1e3),
        # "kg^-1": (0, 1e6),
    },
    "precipitation": {"mm hr^-1": (0, 100)},
    "accumulated_precipitation": {"mm": (0, 100)},
}

# >>> burgers model
burgers_state = {
    "x_velocity": {"m s^-1": (-50, 50)},  # , "km hr^-1": (-150, 150)},
    "y_velocity": {"m s^-1": (-50, 50)},  # , "km hr^-1": (-150, 150)},
}
burgers_tendency = {
    "x_velocity": {"m s^-2": (-50, 50)},  # , "km hr^-2": (-150, 150)},
    "y_velocity": {"m s^-2": (-50, 50)},  # , "km hr^-2": (-150, 150)},
}
