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
import numpy as np
import os
import tasmania as taz

import namelist_smolarkiewicz_ssus_1 as nl

# Create the underlying grid
grid = taz.GridXYZ(nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
				   topo_type=nl.topo_type, topo_time=nl.topo_time, topo_kwargs=nl.topo_kwargs,
				   dtype=nl.dtype)

# Instantiate the initial state
if nl.isothermal:
	state = taz.get_isothermal_isentropic_state(grid, nl.init_time,
												nl.init_x_velocity, nl.init_y_velocity,
												nl.init_temperature, dtype=nl.dtype)
else:
	state = taz.get_default_isentropic_state(grid, nl.init_time,
											 nl.init_x_velocity, nl.init_y_velocity,
											 nl.init_brunt_vaisala, dtype=nl.dtype)
state['tendency_of_air_potential_temperature_on_interface_levels'] = \
	taz.make_data_array_3d(np.zeros((nl.nx, nl.ny, nl.nz+1), dtype=nl.dtype),
						   grid, 'K s^-1')

# Instantiate the component calculating the pressure gradient in isentropic coordinates
order = 4 if nl.horizontal_flux_scheme == 'fifth_order_upwind' else 2
pg = taz.ConservativeIsentropicPressureGradient(
	grid, order=order,
	horizontal_boundary_type=nl.horizontal_boundary_type,
	backend=nl.backend, dtype=nl.dtype
)

# Instantiate the component providing the thermal forcing
psh = taz.PrescribedSurfaceHeating(
	grid, tendency_of_air_potential_temperature_in_diagnostics=True,
	air_pressure_on_interface_levels=True,
	amplitude_at_day_sw=nl.amplitude_at_day_sw,
	amplitude_at_day_fw=nl.amplitude_at_day_fw,
	amplitude_at_night_sw=nl.amplitude_at_night_sw,
	amplitude_at_night_fw=nl.amplitude_at_night_fw,
	frequency_sw=nl.frequency_sw,
	frequency_fw=nl.frequency_fw,
	attenuation_coefficient_at_day=nl.attenuation_coefficient_at_day,
	attenuation_coefficient_at_night=nl.attenuation_coefficient_at_night,
	characteristic_length=nl.characteristic_length,
	starting_time=nl.starting_time,
	backend=nl.backend
)

# Instantiate the component integrating the vertical flux
vf = taz.VerticalIsentropicAdvection(
	grid, moist_on=False, flux_scheme=nl.vertical_flux_scheme,
	tendency_of_air_potential_temperature_on_interface_levels=True,
	backend=nl.backend
)

# Instantiate the component calculating the Coriolis forcing term
cf = taz.ConservativeIsentropicCoriolis(grid, coriolis_parameter=nl.coriolis_parameter,
										dtype=nl.dtype)

# Instantiate the component retrieving the diagnostic variables
pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(grid, moist_on=False, pt=pt,
							   backend=nl.backend, dtype=nl.dtype)

# Wrap the components in a SequentialUpdateSplitting object
sus_bd = taz.SequentialUpdateSplitting(
	cf, pg, psh, vf, 
	time_integration_scheme=nl.coupling_time_integration_scheme,
	grid=grid, horizontal_boundary_type=None,
)
sus_ad = taz.SequentialUpdateSplitting(
	dv, psh, vf, pg, cf,
	time_integration_scheme=nl.coupling_time_integration_scheme,
	grid=grid, horizontal_boundary_type=None,
)

# Instantiate the dynamical core
dycore = taz.HomogeneousIsentropicDynamicalCore(
	grid, moist_on=False,
	# Numerical scheme
	time_integration_scheme=nl.time_integration_scheme,
	horizontal_flux_scheme=nl.horizontal_flux_scheme,
	horizontal_boundary_type=nl.horizontal_boundary_type,
	# Parameterizations
	#diagnostics=taz.DiagnosticComponentComposite(dv),
	# Damping (wave absorber)
	damp_on=nl.damp_on, damp_type=nl.damp_type,
	damp_depth=nl.damp_depth, damp_max=nl.damp_max,
	damp_at_every_stage=nl.damp_at_every_stage,
	# Smoothing
	smooth_on=nl.smooth_on, smooth_type=nl.smooth_type,
	smooth_moist_damp_depth=nl.smooth_damp_depth,
	smooth_coeff=nl.smooth_coeff,
	smooth_coeff_max=nl.smooth_coeff_max,
	smooth_at_every_stage=nl.smooth_at_every_stage,
	# Implementation details
	backend=nl.backend, dtype=nl.dtype
)

# Create a monitor to dump to the solution into a NetCDF file
if nl.filename is not None and nl.save_frequency > 0:
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

	#if i % nl.save_frequency == 0:
	#	# Compute the physics before the dynamics
	#	_ = sus_bd(state=state, timestep=0.5*dt)

	_ = sus_bd(state=state, timestep=0.5*dt)

	# Compute the dynamics
	state_new = dycore(state, {}, dt)
	state.update(state_new)

	#if (i+1) % nl.save_frequency == 0:
	#	# Ensure the state is still defined at the mid time level
	#	state['time'] = nl.init_time + (i+0.5) * dt
	#
	#	# Calculate the physics, and couple it with the dynamics
	#	_ = sus_ad(state=state, timestep=0.5*dt)
	#else:
	#	# Ensure the state is still defined at the current time level
	#	state['time'] = nl.init_time + i * dt
	#
	#	# Fuse two consecutive calls to sus_ad/sus_bd into a single call to sus_ad
	#	_ = sus_ad(state=state, timestep=dt)

	# Ensure the state is still defined at the mid time level
	state['time'] = nl.init_time + (i+0.5) * dt

	# Calculate the physics, and couple it with the dynamics
	_ = sus_ad(state=state, timestep=0.5*dt)

	# Ensure the state is defined at the next time level
	state['time'] = nl.init_time + (i+1) * dt

	if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
		u = state['x_velocity_at_u_locations'].to_units('m s^-1').values[...]
		v = state['y_velocity_at_v_locations'].to_units('m s^-1').values[...]

		umax, umin = u.max(), u.min()
		vmax, vmin = v.max(), v.min()
		cfl = max(umax * dt.total_seconds() / grid.dx.to_units('m').values.item(),
				  vmax * dt.total_seconds() / grid.dy.to_units('m').values.item())

		# Print useful info
		print('Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, '
			  'vmax = {:8.4f} m/s, vmin = {:8.4f} m/s'.format(i + 1, cfl, umax, umin, vmax, vmin))

	# Shortcuts
	to_save = (nl.filename is not None) and \
			  (((nl.save_frequency > 0) and
				((i + 1) % nl.save_frequency == 0)) or i + 1 == nt)

	if to_save:
		# Save the solution
		netcdf_monitor.store(state)

print('Simulation successfully completed. HOORAY!')

# Dump solution to file
if nl.filename is not None and nl.save_frequency > 0:
	netcdf_monitor.write()
