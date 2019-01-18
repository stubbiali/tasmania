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

# The component inferring the diagnostic variables
pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(grid, moist_on=False, pt=pt, 
							   backend=gt.mode.NUMPY, dtype=np.float32)
dcc = taz.DiagnosticComponentComposite(dv)

# The component calculating the pressure gradient in isentropic coordinates
pg = taz.ConservativeIsentropicPressureGradient(grid, order=4, horizontal_boundary_type='relaxed',
		  				   					    backend=gt.mode.NUMPY, dtype=np.float32)

# Wrap the physical components in a ConcurrentCoupling object
cc = taz.ConcurrentCoupling(pg, mode='serial')

# Instantiate the dry isentropic dynamical core
dycore = taz.HomogeneousIsentropicDynamicalCore(grid, moist_on=False,
                                     			time_integration_scheme='rk3cosmo',
												horizontal_flux_scheme='fifth_order_upwind',
												horizontal_boundary_type='relaxed',
												intermediate_parameterizations=cc,
												diagnostics=dcc,
												damp_on=True, damp_type='rayleigh', damp_depth=15,
												damp_max=0.0002, damp_at_every_stage=False,
												smooth_on=True, smooth_type='third_order',
												smooth_coeff=0.03, smooth_at_every_stage=False,
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
		u = state['x_velocity_at_u_locations'].to_units('m s^-1').values[...]
		v = state['y_velocity_at_v_locations'].to_units('m s^-1').values[...]

		umax, umin = u.max(), u.min()
		vmax, vmin = v.max(), v.min()
		cfl = max(umax * timestep.total_seconds() / grid.dx.to_units('m').values.item(),
				  vmax * timestep.total_seconds() / grid.dy.to_units('m').values.item())

		# Print useful info
		print('Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, '
			  'vmax = {:8.4f} m/s, vmin = {:8.4f} m/s'.format(i+1, cfl, umax, umin, vmax, vmin))

	if (i + 1) % 30 == 0:
		# Save the solution
		netcdf_monitor.store(state)

# Write solution to file
netcdf_monitor.write()

print('Simulation successfully completed. HOORAY!')
