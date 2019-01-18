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
import time

import namelist_isentropic_moist_cc as nl

#
# Grid
#
grid = taz.GridXYZ(
	nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
	topo_type=nl.topo_type, topo_time=nl.topo_time,
	topo_kwargs=nl.topo_kwargs, dtype=nl.dtype
)

#
# Initial state
#
if nl.isothermal:
	state = taz.get_isothermal_isentropic_state(
		grid, nl.init_time,	nl.init_x_velocity, nl.init_y_velocity,
		nl.init_temperature, dtype=nl.dtype
	)
else:
	state = taz.get_default_isentropic_state(
		grid, nl.init_time, nl.init_x_velocity, nl.init_y_velocity,
		nl.init_brunt_vaisala, moist=True, precipitation=nl.precipitation,
		dtype=nl.dtype
	)

#
# Intermediate tendencies
#
# Component calculating the pressure gradient in isentropic coordinates
order = 4 if nl.horizontal_flux_scheme == 'fifth_order_upwind' else 2
pg = taz.ConservativeIsentropicPressureGradient(
	grid, order=order,
	horizontal_boundary_type=nl.horizontal_boundary_type,
	backend=nl.backend, dtype=nl.dtype
)

# Component integrating the vertical flux
vf = taz.VerticalIsentropicAdvection(
	grid, moist=False, flux_scheme=nl.vertical_flux_scheme,
	tendency_of_air_potential_temperature_on_interface_levels=True,
	backend=nl.backend
)

# Component calculating the Coriolis acceleration
cf = taz.ConservativeIsentropicCoriolis(
	grid, coriolis_parameter=nl.coriolis_parameter, dtype=nl.dtype
)

# Component calculating the microphysics
ke = taz.Kessler(
	grid, air_pressure_on_interface_levels=True,
	rain_evaporation=nl.rain_evaporation, backend=nl.backend
)

# Wrap the components in a ConcurrentCoupling object
inter_tends = taz.ConcurrentCoupling(
	#pg, vf, cf, ke, execution_policy='serial'
	pg, execution_policy='serial'
)

#
# Intermediate diagnostics
#
# Component retrieving the diagnostic variables
pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(
	grid, moist=True, pt=pt, backend=nl.backend, dtype=nl.dtype
)

# Component performing the saturation adjustment
sa = taz.SaturationAdjustmentKessler(
	grid, air_pressure_on_interface_levels=True, backend=nl.backend
)

# Wrap the components in a TasmaniaDiagnosticComponentComposite object
#inter_diags = taz.TasmaniaDiagnosticComponentComposite(dv, sa)
inter_diags = taz.TasmaniaDiagnosticComponentComposite(dv)

#
# Fast tendencies
#
if nl.precipitation:
	# Component estimating the raindrop fall velocity
	rfv = taz.RaindropFallVelocity(grid, backend=nl.backend)

	# Component integrating the sedimentation flux
	sd = taz.Sedimentation(
		grid, sedimentation_flux_scheme='second_order_upwind',
		backend=nl.backend
	)

	# Wrap the components in a ConcurrentCoupling object
	fast_tends = taz.ConcurrentCoupling(rfv, sd, execution_policy='serial')
else:
	fast_tends = None

#
# Dynamical core
#
dycore = taz.HomogeneousIsentropicDynamicalCore(
	grid, time_units='s', moist=True,
	# parameterizations
	intermediate_tendencies=inter_tends, intermediate_diagnostics=inter_diags,
	substeps=nl.substeps, fast_tendencies=fast_tends, fast_diagnostics=None,
	# numerical scheme
	time_integration_scheme=nl.time_integration_scheme,
	horizontal_flux_scheme=nl.horizontal_flux_scheme,
	horizontal_boundary_type=nl.horizontal_boundary_type,
	# vertical damping
	damp=nl.damp, damp_type=nl.damp_type, damp_depth=nl.damp_depth,
	damp_max=nl.damp_max, damp_at_every_stage=nl.damp_at_every_stage,
	# horizontal smoothing
	smooth=nl.smooth, smooth_type=nl.smooth_type,
	smooth_damp_depth=nl.smooth_damp_depth,
	smooth_coeff=nl.smooth_coeff, smooth_coeff_max=nl.smooth_coeff_max,
	smooth_at_every_stage=nl.smooth_at_every_stage,
	# horizontal smoothing of water species
	smooth_moist=nl.smooth_moist, smooth_moist_type=nl.smooth_moist_type,
	smooth_moist_damp_depth=nl.smooth_moist_damp_depth,
	smooth_moist_coeff=nl.smooth_moist_coeff,
	smooth_moist_coeff_max=nl.smooth_moist_coeff_max,
	smooth_moist_at_every_stage=nl.smooth_moist_at_every_stage,
	# backend settings
	backend=nl.backend, dtype=nl.dtype
)

#
# NetCDF monitor
#
if nl.filename is not None and nl.save_frequency > 0:
	if os.path.exists(nl.filename):
		os.remove(nl.filename)
	netcdf_monitor = taz.NetCDFMonitor(nl.filename, grid)
	netcdf_monitor.store(state)

#
# Time-marching
#
dt = nl.timestep
nt = nl.niter

start_time = time.time()
compute_time = 0.0

for i in range(nt):
	start_time_1 = time.time()

	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * dt)

	# Step the solution
	state_new = dycore(state, {}, dt)
	state.update(state_new)

	compute_time += time.time() - start_time_1

	if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
		u = state['x_velocity_at_u_locations'].to_units('m s^-1').values[...]
		v = state['y_velocity_at_v_locations'].to_units('m s^-1').values[...]

		umax, umin = u.max(), u.min()
		vmax, vmin = v.max(), v.min()
		cfl = max(umax * dt.total_seconds() / grid.dx.to_units('m').values.item(),
				  vmax * dt.total_seconds() / grid.dy.to_units('m').values.item())

		# Print useful info
		print(
			'Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, '
			'vmax = {:8.4f} m/s, vmin = {:8.4f} m/s'.format(
				i + 1, cfl, umax, umin, vmax, vmin
			)
		)

	# Shortcuts
	to_save = (nl.filename is not None) and \
		(((nl.save_frequency > 0) and
		  ((i + 1) % nl.save_frequency == 0)) or i + 1 == nt)

	if to_save:
		# Save the solution
		netcdf_monitor.store(state)

elapsed_time = time.time() - start_time

print('Simulation successfully completed. HOORAY!')
print('Total wall time: {}.'.format(taz.get_time_string(elapsed_time)))
print('Compute time: {}.'.format(taz.get_time_string(compute_time)))

#
# Dump solution to file
#
if nl.filename is not None and nl.save_frequency > 0:
	netcdf_monitor.write()
