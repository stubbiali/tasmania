# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
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


# backend settings
backend = ("numpy",)
datatype = (np.float64,)  # TODO: datatype = (np.float32, no.float64)
halo = ((0, 0, 0), (1, 1, 0), (3, 3, 0), (2, 0, 1))

# x-axis
axis_x = {
    "dims": ("x",),
    "units_to_range": {"km": (-100, 100), "m": (0, 2e5)},
    "length": (1, 20),
    "increasing": True,
}

# y-axis
axis_y = {
    "dims": ("y",),
    "units_to_range": {"km": (-100, 100), "m": (0, 2e5)},
    "length": (1, 20),
    "increasing": True,
}

# z-axis
axis_z = {
    "dims": ("air_potential_temperature",),
    "units_to_range": {"K": (270, 400), "degC": (-20, 40)},
    "length": (1, 20),
    "increasing": False,
}

# topography2d
topography = {
    "type": ("flat_terrain", "gaussian", "schaer", "user_defined"),
    "time": (timedelta(seconds=1), timedelta(minutes=60)),
    "units_to_max_height": {"km": (0, 2)},
    "units_to_half_width_x": {"m": (1, 50e3)},
    "units_to_half_width_y": {"m": (1, 50e3), "km": (1, 20)},
    "str": ("x + y",),
}

# horizontal boundary
nb = 4
horizontal_boundary_types = ("relaxed", "periodic", "dirichlet")  # 'identity'

# isentropic model
isentropic_state = {
    "air_isentropic_density": {"kg m^-2 K^-1": (10, 1000)},
    "x_velocity_at_u_locations": {"m s^-1": (-50, 50), "km hr^-1": (-150, 150)},
    "y_velocity_at_v_locations": {"m s^-1": (-50, 50), "km hr^-1": (-150, 150)},
    "mass_fraction_of_water_vapor_in_air": {"g g^-1": (0, 5), "g kg^-1": (0, 5000)},
    "mass_fraction_of_cloud_liquid_water_in_air": {
        "g g^-1": (0, 5),
        "g kg^-1": (0, 5000),
    },
    "mass_fraction_of_precipitation_water_in_air": {
        "g g^-1": (0, 5),
        "g kg^-1": (0, 5000),
    },
    "number_density_of_precipitation_water": {"g^-1": (0, 1e3), "kg^-1": (0, 1e6)},
    "precipitation": {"mm hr^-1": (0, 100)},
    "accumulated_precipitation": {"mm": (0, 100)},
}

# burgers model
burgers_state = {
    "x_velocity": {"m s^-1": (-50, 50), "km hr^-1": (-150, 150)},
    "y_velocity": {"m s^-1": (-50, 50), "km hr^-1": (-150, 150)},
}
burgers_tendency = {
    "x_velocity": {"m s^-2": (-50, 50), "km hr^-2": (-150, 150)},
    "y_velocity": {"m s^-2": (-50, 50), "km hr^-2": (-150, 150)},
}
