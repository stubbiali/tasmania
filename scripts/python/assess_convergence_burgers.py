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
from datetime import datetime
import numpy as np
from sympl import DataArray
import tasmania as taz

# ========================================
# user inputs
# ========================================
field_name = "y_velocity"
field_units = "m s^-1"

prefix = "../../data/pdc_paper/burgers/"

datasets = (
    {
        "filename": "burgers_ps_gtcuda_0.nc",
        "init_time": datetime(year=1992, month=2, day=20, hour=0),
        "eps": DataArray(0.1, attrs={"units": "m^2 s^-1"}),
        "xslice": slice(0, 11),
        "yslice": slice(0, 11),
        # "xslice": slice(3, 8),
        # "yslice": slice(3, 8),
        "tlevel": -1,
    },
    {
        "filename": "burgers_ps_gtcuda_1.nc",
        "init_time": datetime(year=1992, month=2, day=20, hour=0),
        "eps": DataArray(0.1, attrs={"units": "m^2 s^-1"}),
        "xslice": slice(0, 21),
        "yslice": slice(0, 21),
        # "xslice": slice(6, 15),
        # "yslice": slice(6, 15),
        "tlevel": -1,
    },
    {
        "filename": "burgers_ps_gtcuda_2.nc",
        "init_time": datetime(year=1992, month=2, day=20, hour=0),
        "eps": DataArray(0.1, attrs={"units": "m^2 s^-1"}),
        "xslice": slice(0, 41),
        "yslice": slice(0, 41),
        # "xslice": slice(12, 29),
        # "yslice": slice(12, 29),
        "tlevel": -1,
    },
    {
        "filename": "burgers_ps_gtcuda_3.nc",
        "init_time": datetime(year=1992, month=2, day=20, hour=0),
        "eps": DataArray(0.1, attrs={"units": "m^2 s^-1"}),
        "xslice": slice(0, 81),
        "yslice": slice(0, 81),
        # "xslice": slice(24, 57),
        # "yslice": slice(24, 57),
        "tlevel": -1,
    },
)

# ========================================
# code
# ========================================
if __name__ == "__main__":
    for ds in datasets:
        # load the numerical solution
        domain, grid_type, states = taz.load_netcdf_dataset(prefix + ds["filename"])
        state = states[ds["tlevel"]]
        raw_field = np.asarray(state[field_name].to_units(field_units).values)
        sol = raw_field[ds["xslice"], ds["yslice"], 0]

        # compute the analytical solution
        grid = domain.numerical_grid if grid_type == "numerical" else domain.physical_grid
        zsf = taz.ZhaoSolutionFactory(ds["init_time"], ds["eps"])
        rsol = zsf(
            datetime(
                year=1992, month=2, day=20, hour=0, minute=0, second=0, microsecond=768000
            ),
            # state["time"],
            grid,
            slice_x=ds["xslice"],
            slice_y=ds["yslice"],
            field_name=field_name,
            field_units=field_units,
        )

        # compute and print the error
        err = np.linalg.norm(sol - rsol[:, :, 0]) / np.sqrt(sol.size)
        print("{}: {:5.5E}".format(ds["filename"], err))
