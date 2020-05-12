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
from sympl import DataArray


# computational domain
domain_x = DataArray([-176, 176], dims="x", attrs={"units": "km"}).to_units("m")
nx = 161
domain_y = DataArray([-176, 176], dims="y", attrs={"units": "km"}).to_units("m")
ny = 161
domain_z = DataArray([400, 280], dims="potential_temperature", attrs={"units": "K"})
nz = 60

# horizontal boundary
hb_type = "relaxed"
nb = 3
hb_kwargs = {"nr": 6, "nz": nz}

# gt4py settings
gt_powered = False
gt_kwargs = {
    "backend": "numpy",
    "build_info": None,
    "dtype": np.float64,
    "exec_info": None,
    "default_origin": (nb, nb, 0),
    "rebuild": False,
    "managed_memory": False,
}
gt_kwargs["backend_opts"] = (
    {"verbose": True} if gt_kwargs["backend"] in ("gtx86", "gtmc", "gtcuda") else None
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
isothermal = False
brunt_vaisala = DataArray(0.015, attrs={"units": "s^-1"})
temperature = DataArray(250.0, attrs={"units": "K"})

# time stepping
time_integration_scheme = "rk3ws_si"
substeps = 0
eps = 0.5
a = 0.375
b = 0.375
c = 0.25

# advection
horizontal_flux_scheme = "fifth_order_upwind"

# damping
damp = True
damp_type = "rayleigh"
damp_depth = 15
damp_max = 0.0005
damp_at_every_stage = False

# horizontal diffusion
diff = False
diff_type = "second_order"
diff_coeff = DataArray(10, attrs={"units": "s^-1"})
diff_coeff_max = DataArray(12, attrs={"units": "s^-1"})
diff_damp_depth = 30

# horizontal smoothing
smooth = True
smooth_type = "second_order"
smooth_coeff = 1.0
smooth_coeff_max = 1.0
smooth_damp_depth = 0
smooth_at_every_stage = False

# turbulence
turbulence = True
smagorinsky_constant = 0.18

# coriolis
coriolis = True
coriolis_parameter = None

# simulation length
timestep = timedelta(seconds=10)
niter = 10  # int(1 * 60 * 60 / timestep.total_seconds())

# output
save = False
save_frequency = 20
filename = (
    "../../data/benchmarking/isentropic_dry_{}_{}_nx{}_nz{}_dt{}_nt{}_"
    "{}_L{}_H{}_u{}_{}{}{}{}_fc_{}.nc".format(
        time_integration_scheme,
        horizontal_flux_scheme,
        nx,
        nz,
        int(timestep.total_seconds()),
        niter,
        topo_type,
        int(topo_kwargs["width_x"].to_units("m").values.item()),
        int(topo_kwargs["max_height"].to_units("m").values.item()),
        int(x_velocity.to_units("m s^-1").values.item()),
        "T" if isothermal else "bv",
        "_diff" if diff else "",
        "_smooth" if smooth else "",
        "_turb" if turbulence else "",
        gt_kwargs["backend"],
    )
)
store_names = (
    "air_isentropic_density",
    "air_pressure_on_interface_levels",
    "exner_function_on_interface_levels",
    "height_on_interface_levels",
    "montgomery_potential",
    "x_momentum_isentropic",
    "x_velocity_at_u_locations",
    "y_momentum_isentropic",
    "y_velocity_at_v_locations",
)
print_frequency = -1
