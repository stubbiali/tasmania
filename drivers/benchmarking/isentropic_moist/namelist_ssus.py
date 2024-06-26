# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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

from sympl import DataArray

from tasmania.framework.options import BackendOptions, StorageOptions


# computational domain
domain_x = DataArray([-176, 176], dims="x", attrs={"units": "km"}).to_units("m")
nx = 161
domain_y = DataArray([-176, 176], dims="y", attrs={"units": "km"}).to_units("m")
ny = 161
domain_z = DataArray([400, 280], dims="potential_temperature", attrs={"units": "K"})
nz = 120

# horizontal boundary
hb_type = "relaxed"
nb = 3
hb_kwargs = {"nr": 6}

# backend settings
backend = "gt4py:gt:cpu_kfirst"
bo = BackendOptions(
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
so = StorageOptions(dtype=np.float64, aligned_index=(nb, nb, 0), managed="gt4py")
enable_checks = False

# topography
topo_type = "gaussian"
topo_kwargs = {
    "time": timedelta(seconds=1800),
    "max_height": DataArray(0.5, attrs={"units": "km"}),
    "width_x": DataArray(50.0, attrs={"units": "km"}),
    "width_y": DataArray(50.0, attrs={"units": "km"}),
    "smooth": False,
}

# initial conditions
init_time = datetime(year=1992, month=2, day=20, hour=0)
x_velocity = DataArray(22.5, attrs={"units": "m s^-1"})
y_velocity = DataArray(0.0, attrs={"units": "m s^-1"})
brunt_vaisala = DataArray(0.015, attrs={"units": "s^-1"})
relative_humidity = 0.95

# time stepping
time_integration_scheme = "rk3ws_si"
substeps = 0
eps = 0.5
a = 0.375
b = 0.375
c = 0.25
physics_time_integration_scheme = "rk2"

# advection
horizontal_flux_scheme = "fifth_order_upwind"
vertical_advection = True
implicit_vertical_advection = False
vertical_flux_scheme = "third_order_upwind"

# damping
damp = True
damp_type = "rayleigh"
damp_depth = 15
damp_max = 0.0005
damp_at_every_stage = False

# horizontal smoothing
smooth = True
smooth_type = "second_order"
smooth_coeff = 1.0
smooth_coeff_max = 1.0
smooth_damp_depth = 0
smooth_at_every_stage = False
smooth_moist = True
smooth_moist_type = "second_order"
smooth_moist_coeff = 1.0
smooth_moist_coeff_max = 1.0
smooth_moist_damp_depth = 0
smooth_moist_at_every_stage = False

# turbulence
smagorinsky_constant = 0.18

# coriolis
coriolis_parameter = None

# microphysics
sedimentation = True
sedimentation_flux_scheme = "second_order_upwind"
rain_evaporation = True
autoconversion_threshold = DataArray(0.1, attrs={"units": "g kg^-1"})
autoconversion_rate = DataArray(0.001, attrs={"units": "s^-1"})
collection_rate = DataArray(2.2, attrs={"units": "s^-1"})
saturation_rate = DataArray(0.025, attrs={"units": "s^-1"})
update_frequency = 0

# simulation length
timestep = timedelta(seconds=5)
niter = 100

# output
data_dir = os.path.join(os.path.dirname(__file__), "../../../data")
exec_info_csv = os.path.join(data_dir, "isentropic_moist_exec.csv")
run_info_csv = os.path.join(data_dir, "isentropic_moist_run.csv")
log_txt = os.path.join(data_dir, f"isentropic_moist_log_ssus_{backend}.txt")
