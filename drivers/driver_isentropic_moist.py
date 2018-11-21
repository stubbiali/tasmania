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
import os
from sympl import DataArray
import tasmania as taz

# Build the underlying grid
domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}).to_units('m'), 101
domain_y, ny = DataArray([-1., 1.], dims='y', attrs={'units': 'km'}).to_units('m'), 1
domain_z, nz = DataArray([340., 280.], dims='air_potential_temperature', attrs={'units': 'K'}), 60
grid = taz.GridXYZ(domain_x, nx, domain_y, ny, domain_z, nz,
                   topo_type='gaussian', topo_time=timedelta(seconds=1800),
                   topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
                                'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})

# Instantiate the initial state
time          = datetime(year=1992, month=2, day=20)
x_velocity    = DataArray(15., attrs={'units': 'm s^-1'})
y_velocity    = DataArray(0., attrs={'units': 'm s^-1'})
brunt_vaisala = DataArray(0.01, attrs={'units': 's^-1'})
state = taz.get_default_isentropic_state(grid, time, x_velocity, y_velocity, brunt_vaisala, 
                                         moist_on=True, dtype=np.float64)

# The component calculating the microphysical sources/sinks
# prescribed by the Kessler scheme; neglect the evaporation of
# rain in the subcloud layers
kessler = taz.Kessler(grid, pressure_on_interface_levels=True,
                      rain_evaporation_on=False, backend=gt.mode.NUMPY)

# The component performing the saturation adjustment
# as prescribed by the Kessler scheme
saturation = taz.SaturationAdjustmentKessler(grid, pressure_on_interface_levels=True,
                                             backend=gt.mode.NUMPY)

# Instantiate the dry isentropic dynamical core
dycore = taz.IsentropicDynamicalCore(grid, moist_on=True,
                                     time_integration_scheme='forward_euler',
                                     horizontal_flux_scheme='upwind',
                                     horizontal_boundary_type='relaxed',
                                     smooth_on=True, smooth_type='first_order',
                                     smooth_coeff=0.20, smooth_at_every_stage=True,
                                     adiabatic_flow=True, sedimentation_on=False,
                                     backend=gt.mode.NUMPY, dtype=np.float64)

# Create a monitor to dump the solution into a NetCDF file
filename = '../tests/baseline_datasets/isentropic_moist.nc'
if os.path.exists(filename):
	os.remove(filename)
netcdf_monitor = taz.NetCDFMonitor(filename, grid)
netcdf_monitor.store(state)

# Simulation settings
timestep = timedelta(seconds=10)
niter = 2160

# Integrate
for i in range(niter):
	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * timestep)

	# Calculate the microphysical tendencies
	tendencies, _ = kessler(state)

	# Step the solution
	state_new = dycore(state, tendencies, timestep)
	state.update(state_new)

	# Perform the saturation adjustment
	state_new = saturation(state)
	state.update(state_new)

	if (i + 1) % 60 == 0:
		# Save the solution
		netcdf_monitor.store(state)

# Write solution to file
netcdf_monitor.write()

print('Simulation successfully completed. HOORAY!')
