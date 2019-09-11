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
import gridtools as gt
import numpy as np
from sympl import DataArray

# backend settings
backend = gt.mode.NUMPY
dtype = np.float64

# computational domain
domain_x = DataArray([-400, 400], dims="x", attrs={"units": "km"}).to_units("m")
nx = 81
domain_y = DataArray([-400, 400], dims="y", attrs={"units": "km"}).to_units("m")
ny = 81
domain_z = DataArray([340, 280], dims="potential_temperature", attrs={"units": "K"})
nz = 60

# topography
_width = DataArray(50.0, attrs={"units": "km"})
topo_type = "flat_terrain"
topo_time = timedelta(seconds=0)
topo_kwargs = {
    #'topo_str': '1 * 10000. * 10000. / (x*x + 10000.*10000.)',
    #'topo_str': '3000. * pow(1. + (x*x + y*y) / 25000.*25000., -1.5)',
    "topo_max_height": DataArray(0.0, attrs={"units": "km"}),
    "topo_width_x": DataArray(
        1 * _width.to_units("km").values.item(), attrs={"units": "km"}
    ),
    "topo_width_y": DataArray(
        1 * _width.to_units("km").values.item(), attrs={"units": "km"}
    ),
    "topo_smooth": False,
}

# moist
moist = False
precipitation = False
rain_evaporation = False

# initial conditions
init_time = datetime(year=1992, month=2, day=20, hour=0)
init_x_velocity = DataArray(0.0, attrs={"units": "m s^-1"})
init_y_velocity = DataArray(0.0, attrs={"units": "m s^-1"})
isothermal = False
if isothermal:
    init_temperature = DataArray(250.0, attrs={"units": "K"})
else:
    init_brunt_vaisala = DataArray(0.01, attrs={"units": "s^-1"})

# numerical scheme
time_integration_scheme = "rk3cosmo"
horizontal_flux_scheme = "fifth_order_upwind"
vertical_flux_scheme = "third_order_upwind"
horizontal_boundary_type = "relaxed"
physics_time_integration_scheme = "forward_euler"
substeps = 0

# damping
damp = True
damp_type = "rayleigh"
damp_depth = 15
damp_max = 0.0002
damp_at_every_stage = False

# horizontal smoothing
smooth = True
smooth_type = "second_order"
smooth_damp_depth = 0
smooth_coeff = 0.2
smooth_coeff_max = 0.2
smooth_at_every_stage = False

# horizontal smoothing for water species
smooth_moist = False
smooth_moist_type = "second_order"
smooth_moist_damp_depth = 0
smooth_moist_coeff = 0.2
smooth_moist_coeff_max = 0.2
smooth_moist_at_every_stage = False

# coriolis
coriolis = True
coriolis_parameter = None  # DataArray(1e-3, attrs={'units': 'rad s^-1'})

# prescribed surface heating
tendencies_in_diagnostics = True
amplitude_at_day_sw = DataArray(800.0, attrs={"units": "W m^-2"})
amplitude_at_day_fw = DataArray(0.0, attrs={"units": "W m^-2"})
amplitude_at_night_sw = DataArray(600.0, attrs={"units": "W m^-2"})
amplitude_at_night_fw = DataArray(0.0, attrs={"units": "W m^-2"})
attenuation_coefficient_at_day = DataArray(1.0 / 600.0, attrs={"units": "m^-1"})
attenuation_coefficient_at_night = DataArray(1.0 / 600.0, attrs={"units": "m^-1"})
frequency_sw = DataArray(3 * np.pi, attrs={"units": "h^-1"})
frequency_fw = DataArray(4 * np.pi, attrs={"units": "h^-1"})
characteristic_length = _width
starting_time = init_time  # + timedelta(seconds=6*60*60)

# simulation length
timestep = timedelta(seconds=24)
niter = int(12 * 60 * 60 / timestep.total_seconds())

# output
filename = (
    "../../data/smolarkiewicz_{}_{}_{}_{}_nx{}_ny{}_nz{}_dt{}_nt{}_ns{}_"
    "{}_L{}_H{}_u{}_wf3_f_ssus.nc".format(
        time_integration_scheme,
        horizontal_flux_scheme,
        vertical_flux_scheme,
        physics_time_integration_scheme,
        nx,
        ny,
        nz,
        int(timestep.total_seconds()),
        niter,
        substeps,
        topo_type,
        int(_width.to_units("m").values.item()),
        int(topo_kwargs["topo_max_height"].to_units("m").values.item()),
        int(init_x_velocity.to_units("m s^-1").values.item()),
    )
)
save_frequency = 25
print_frequency = 25
