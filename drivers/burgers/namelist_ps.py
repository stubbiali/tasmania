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
import pandas as pd
import numpy as np
from sympl import DataArray
import tasmania as taz


factor = 1

# initial conditions
init_time = datetime(year=1992, month=2, day=20, hour=0)

# diffusion
diffusion_type = "fourth_order"
diffusion_coeff = DataArray(0.1, attrs={"units": "m^2 s^-1"})
zsof = taz.ZhaoSolutionFactory(init_time, diffusion_coeff)

# domain
domain_x = DataArray([0, 1], dims="x", attrs={"units": "m"})
nx = 10 * (2 ** factor) + 1
domain_y = DataArray([0, 1], dims="y", attrs={"units": "m"})
ny = 10 * (2 ** factor) + 1
hb_type = "dirichlet"
nb = 3
hb_kwargs = {"core": zsof}

# gt4py settings
gt_kwargs = {
    "backend": "gtx86",
    "build_info": None,
    "dtype": np.float64,
    "exec_info": None,
    "default_origin": (nb, nb, 0),
    "rebuild": False,
    "managed_memory": False,
}
gt_kwargs["backend_opts"] = {"verbose": True} if gt_kwargs["backend"] != "numpy" else None

# numerical scheme
time_integration_scheme = "rk3ws"
flux_scheme = "fifth_order"
physics_time_integration_scheme = "rk2"

# dict operator
gt_powered = True

# simulation time
cfl = 1.0
timestep = pd.Timedelta(cfl / (nx - 1) ** 2, unit="s")
niter = 4 ** factor * 100

# output
save = False
save_frequency = 1
filename = "../../data/burgers_ps_{}.nc".format(gt_kwargs["backend"])
print_frequency = 1
