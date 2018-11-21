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
import gridtools as gt
import numpy as np
import os
import tasmania as taz

# Load the computational grid and the initial state
grid, states = taz.load_netcdf_dataset('../tests/baseline_datasets/isentropic_dry.nc')
state = states[0]

# Instantiate the dry isentropic dynamical core
dycore = taz.IsentropicDynamicalCore(grid, moist_on=False,
                                     time_integration_scheme='forward_euler',
                                     horizontal_flux_scheme='maccormack',
                                     horizontal_boundary_type='relaxed',
                                     damp_on=True, damp_type='rayleigh', damp_depth=15,
                                     damp_max=0.0002, damp_at_every_stage=False,
                                     smooth_on=True, smooth_type='second_order',
                                     smooth_coeff=0.12, smooth_at_every_stage=False,
                                     backend=gt.mode.NUMPY, dtype=np.float32)

# Create a monitor to dump the solution into a NetCDF file
filename = '../tests/baseline_datasets/isentropic_dry_bis.nc'
if os.path.exists(filename):
	os.remove(filename)
netcdf_monitor = taz.NetCDFMonitor(filename, grid)
netcdf_monitor.store(state)

# Simulation settings
timestep = timedelta(seconds=24)
niter = 1800

# Integrate
for i in range(niter):
	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * timestep)

	# Step the solution
	state_new = dycore(state, {}, timestep)
	state.update(state_new)

	if (i + 1) % 30 == 0:
		# Save the solution
		netcdf_monitor.store(state)

# Write solution to file
netcdf_monitor.write()

print('Simulation successfully completed. HOORAY!')
