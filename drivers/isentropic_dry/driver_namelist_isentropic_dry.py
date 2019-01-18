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
import os
import tasmania as taz

import namelist_isentropic_dry as nl


# Create the underlying grid
grid = taz.GridXYZ(nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
				   topo_type=nl.topo_type, topo_time=nl.topo_time, topo_kwargs=nl.topo_kwargs,
				   dtype=nl.dtype)

# Instantiate the initial state
if nl.isothermal_flow:
	state = taz.get_isothermal_isentropic_state(grid, nl.init_time,
												nl.init_x_velocity, nl.init_y_velocity,
												nl.init_temperature, dtype=nl.dtype)
else:
	state = taz.get_default_isentropic_state(grid, nl.init_time
											 nl.init_x_velocity, nl.init_y_velocity,
											 nl.init_brunt_vaisala, moist_on=False, dtype=nl.dtype)

# Instantiate the dry isentropic dynamical core
dycore = taz.IsentropicDynamicalCore(grid, moist_on=False,
									 time_integration_scheme=nl.time_integration_scheme,
									 horizontal_flux_scheme=nl.horizontal_flux_scheme,
									 horizontal_boundary_type=nl.horizontal_boundary_type,
									 damp_on=nl.damp_on, damp_type=nl.damp_type,
									 damp_depth=nl.damp_depth, damp_max=nl.damp_max,
									 damp_at_every_stage=nl.damp_at_every_stage,
									 smooth_on=nl.smooth_on, smooth_type=nl.smooth_type,
									 smooth_coeff=nl.smooth_coeff,
									 smooth_at_every_stage=nl.smooth_at_every_stage,
									 backend=nl.backend, dtype=nl.dtype)

# Create a monitor to dump the solution into a NetCDF file
if nl.filename is not None:
	if os.path.exists(nl.filename):
		os.remove(nl.filename)
	netcdf_monitor = taz.NetCDFMonitor(nl.filename, grid)
	netcdf_monitor.store(state)

# Simulation settings
dt = nl.timestep
nt = nl.niter

# Integrate
for i in range(nt):
	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * dt)

	# Step the solution
	state_new = dycore(state, {}, dt)
	state.update(state_new)

	if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
		u = state['x_velocity_at_u_locations'].to_units('m s^-1').values[...]
		v = state['y_velocity_at_v_locations'].to_units('m s^-1').values[...]

		umax, umin = u.max(), u.min()
		vmax, vmin = v.max(), v.min()
		cfl = max(umax * dt.total_seconds() / grid.dx.to_units('m').values.item(),
				  vmax * dt.total_seconds() / grid.dy.to_units('m').values.item())

		# Print useful info
		print('Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, '
			  'vmax = {:8.4f} m/s, vmin = {:8.4f} m/s'.format(i+1, cfl, umax, umin, vmax, vmin))

	if (nl.filename is not None) and \
	   (((nl.save_frequency > 0) and ((i + 1) % nl.save_frequency == 0)) or i + 1 == nt):
		# Save the solution
		netcdf_monitor.store(state)

# Write solution to file
if nl.filename is not None:
	netcdf_monitor.write()

print('Simulation successfully completed. HOORAY!')
