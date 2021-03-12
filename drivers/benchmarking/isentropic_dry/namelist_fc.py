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
from datetime import datetime, timedelta
import numpy as np
import os
import socket
from sympl import DataArray
import tasmania as taz


# computational domain
domain_x = DataArray([-176, 176], dims="x", attrs={"units": "km"}).to_units(
    "m"
)
nx = 161
domain_y = DataArray([-176, 176], dims="y", attrs={"units": "km"}).to_units(
    "m"
)
ny = 161
domain_z = DataArray(
    [400, 280], dims="potential_temperature", attrs={"units": "K"}
)
nz = 60

# horizontal boundary
hb_type = "relaxed"
nb = 3
hb_kwargs = {"nr": 6}

# backend settings
backend = "gt4py:gtx86"
bo = taz.BackendOptions(
    # gt4py
    build_info={},
    exec_info={"__aggregate_data": True},
    rebuild=False,
    validate_args=False,
    # numba
    cache=True,
    check_rebuild=False,
    fastmath=False,
    inline="always",
    nopython=True,
    parallel=True,
)
so = taz.StorageOptions(
    dtype=np.float64, aligned_index=(nb, nb, 0), managed="gt4py"
)

# topography
topo_type = "gaussian"
topo_kwargs = {
    "time": timedelta(seconds=1800),
    "max_height": DataArray(500.0, attrs={"units": "m"}),
    "width_x": DataArray(50.0, attrs={"units": "km"}),
    "width_y": DataArray(50.0, attrs={"units": "km"}),
    "smooth": False,
}

# initial conditions
init_time = datetime(year=1992, month=2, day=20, hour=0)
x_velocity = DataArray(22.5, attrs={"units": "m s^-1"})
y_velocity = DataArray(0.0, attrs={"units": "m s^-1"})
brunt_vaisala = DataArray(0.015, attrs={"units": "s^-1"})

# time stepping
time_integration_scheme = "rk3ws_si"
substeps = 0
eps = 0.5
a = 0.375
b = 0.375
c = 0.25

# advection
horizontal_flux_scheme = "centered"

# damping
damp = True
damp_type = "rayleigh"
damp_depth = 15
damp_max = 0.0002
damp_at_every_stage = False

# horizontal smoothing
smooth = True
smooth_type = "second_order"
smooth_coeff = 1.0
smooth_coeff_max = 1.0
smooth_damp_depth = 0
smooth_at_every_stage = False

# turbulence
smagorinsky_constant = 0.18

# coriolis
coriolis_parameter = None

# simulation length
timestep = timedelta(seconds=10)
niter = 100

# output
hostname = socket.gethostname()
if "nid" in hostname:
    if os.path.exists("/scratch/snx3000"):
        prefix = "/scratch/snx3000/subbiali/timing/oop"
    else:
        prefix = "/scratch/snx3000tds/subbiali/timing/oop"
elif "daint" in hostname:
    prefix = "/scratch/snx3000/subbiali/timing/oop"
elif "dom" in hostname:
    prefix = "/scratch/snx3000tds/subbiali/timing/oop"
else:
    prefix = "../timing/oop"
exec_info_csv = os.path.join(prefix, f"isentropic_dry_exec_fc_{backend}.csv")
run_info_csv = os.path.join(prefix, "isentropic_dry_run_fc.csv")
log_txt = os.path.join(prefix, f"isentropic_dry_log_fc_{backend}.txt")
