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
import os
from sympl import DataArray
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils


if __name__ == "__main__":
    for _ in range(100):
        out = utils.st_floats().example()
        assert not np.isnan(out)
        assert not np.isinf(out)

        utils.st_interval(axis_name="x").example()
        utils.st_interval(axis_name="y").example()
        utils.st_interval(axis_name="z").example()

        utils.st_length(axis_name="x").example()
        utils.st_length(axis_name="y").example()
        utils.st_length(axis_name="z").example()

        x = DataArray([0, 100], attrs={"units": "m"})
        time = timedelta(seconds=1800)
        utils.st_topography1d_kwargs(x).example()
        utils.st_topography1d_kwargs(x, topo_time=time).example()

        y = DataArray([0, 100], attrs={"units": "m"})
        utils.st_topography2d_kwargs(x, y).example()
        utils.st_topography2d_kwargs(x, y, topo_time=time).example()

        g = utils.st_grid_xyz().example()
        utils.st_grid_xy().example()
        utils.st_grid_xz().example()
        utils.st_grid_yz().example()

        utils.st_isentropic_field(
            g, "isentropic", "air_isentropic_density", (g.nx, g.ny, g.nz)
        )
        utils.st_isentropic_field(
            g, "isentropic", "x_velocity_at_u_locations", (g.nx + 1, g.ny, g.nz)
        )
        utils.st_isentropic_field(
            g, "isentropic", "y_velocity_at_v_locations", (g.nx, g.ny + 1, g.nz)
        )
        utils.st_isentropic_field(
            g, "isentropic", "mass_fraction_of_water_vapor_in_air", (g.nx, g.ny, g.nz)
        )
        utils.st_isentropic_field(
            g,
            "isentropic",
            "mass_fraction_of_cloud_liquid_water_in_air",
            (g.nx, g.ny, g.nz),
        )
        utils.st_isentropic_field(
            g,
            "isentropic",
            "mass_fraction_of_precipitation_water_in_air",
            (g.nx, g.ny, g.nz),
        )
        utils.st_isentropic_field(g, "isentropic", "precipitation", (g.nx, g.ny, 1))
        utils.st_isentropic_field(
            g, "isentropic", "accumulated_precipitation", (g.nx, g.ny, 1)
        )

        utils.st_isentropic_state(g)
        utils.st_isentropic_state(g, moist=True)
        utils.st_isentropic_state(g, moist=True, precipitation=True)
