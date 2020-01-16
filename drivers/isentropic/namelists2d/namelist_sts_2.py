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
ny = 1
domain_z = DataArray([340, 280], dims="potential_temperature", attrs={"units": "K"})
nz = 60

# horizontal boundary
hb_type = "relaxed"
nb = 3
hb_kwargs = {"nr": 6}

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
gt_kwargs["backend_opts"] = (
    {"verbose": True} if gt_kwargs["backend"] in ("gtx86", "gtmc", "gtcuda") else None
)

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
x_velocity = DataArray(15.0, attrs={"units": "m s^-1"})
y_velocity = DataArray(0.0, attrs={"units": "m s^-1"})
brunt_vaisala = DataArray(0.01, attrs={"units": "s^-1"})
relative_humidity = 0.9

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
vertical_advection = False
vertical_flux_scheme = "third_order_upwind"

# damping
damp = True
damp_type = "rayleigh"
damp_depth = 15
damp_max = 0.0002
damp_at_every_stage = False

# dict operator
gt_powered = True

# horizontal diffusion
diff = False
diff_type = "second_order"
diff_coeff = DataArray(10, attrs={"units": "s^-1"})
diff_coeff_max = DataArray(12, attrs={"units": "s^-1"})
diff_damp_depth = 30
diff_moist = False
diff_moist_type = "second_order"
diff_moist_coeff = DataArray(0.12, attrs={"units": "s^-1"})
diff_moist_coeff_max = DataArray(0.12, attrs={"units": "s^-1"})
diff_moist_damp_depth = 0

# horizontal smoothing
smooth = True
smooth_type = "second_order"
smooth_coeff = 1.0
smooth_coeff_max = 1.0
smooth_damp_depth = 0
smooth_at_every_stage = False
smooth_moist = False
smooth_moist_type = "second_order"
smooth_moist_coeff = 0.12
smooth_moist_coeff_max = 0.12
smooth_moist_damp_depth = 0
smooth_moist_at_every_stage = False

# turbulence
turbulence = True
smagorinsky_constant = 0.18

# coriolis
coriolis = False
coriolis_parameter = None  # DataArray(1e-3, attrs={'units': 'rad s^-1'})

# microphysics
sedimentation = True
sedimentation_flux_scheme = "second_order_upwind"
rain_evaporation = True
autoconversion_threshold = DataArray(0.1, attrs={"units": "g kg^-1"})
autoconversion_rate = DataArray(0.001, attrs={"units": "s^-1"})
collection_rate = DataArray(2.2, attrs={"units": "s^-1"})
saturation_vapor_pressure_formula = "tetens"
saturation_rate = DataArray(0.025, attrs={"units": "s^-1"})
update_frequency = 0

# simulation length
timestep = timedelta(seconds=10)
niter = int(12 * 60 * 60 / timestep.total_seconds())

# output
save = True
save_frequency = -1
filename = (
    "/scratch/snx3000tds/subbiali/data/prognostic-saturation-2d/isentropic_moist_{}_{}_{}_"
    "nx{}_ny{}_nz{}_dt{}_nt{}_{}_L{}_H{}_u{}_rh{}{}{}{}{}{}{}_sts_{}.nc".format(
        time_integration_scheme,
        horizontal_flux_scheme,
        physics_time_integration_scheme,
        nx,
        ny,
        nz,
        int(timestep.total_seconds()),
        niter,
        topo_type,
        int(topo_kwargs["width_x"].to_units("m").values.item()),
        int(topo_kwargs["max_height"].to_units("m").values.item()),
        int(x_velocity.to_units("m s^-1").values.item()),
        int(relative_humidity * 100),
        "_diff" if diff else "",
        "_smooth" if smooth else "",
        "_turb" if turbulence else "",
        "_f" if coriolis else "",
        "_sed" if sedimentation else "",
        "_evap" if rain_evaporation else "",
        gt_kwargs["backend"],
    )
)
store_names = (
    "accumulated_precipitation",
    # "air_density",
    "air_isentropic_density",
    # "air_pressure_on_interface_levels",
    # "air_temperature",
    # "exner_function_on_interface_levels",
    "height_on_interface_levels",
    "mass_fraction_of_water_vapor_in_air",
    "mass_fraction_of_cloud_liquid_water_in_air",
    "mass_fraction_of_precipitation_water_in_air",
    # "montgomery_potential",
    "precipitation",
    "x_momentum_isentropic",
    # "x_velocity_at_u_locations",
    # "y_momentum_isentropic",
    # "y_velocity_at_v_locations",
)
print_dry_frequency = -1
print_moist_frequency = 20
