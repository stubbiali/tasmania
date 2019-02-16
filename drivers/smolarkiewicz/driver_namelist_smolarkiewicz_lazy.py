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

import namelist_smolarkiewicz_lazy as nl

#============================================================
# The underlying grid
#============================================================
grid = taz.GridXYZ(
	nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
	topo_type=nl.topo_type, topo_time=nl.topo_time, topo_kwargs=nl.topo_kwargs,
	dtype=nl.dtype
)

#============================================================
# The initial state
#============================================================
if nl.isothermal:
	state = taz.get_isothermal_isentropic_state(
		grid, nl.init_time, nl.init_x_velocity, nl.init_y_velocity,
		nl.init_temperature, moist=nl.moist, precipitation=nl.precipitation, 
		dtype=nl.dtype
	)
else:
	state = taz.get_default_isentropic_state(
		grid, nl.init_time, nl.init_x_velocity, nl.init_y_velocity,
		nl.init_brunt_vaisala, moist=nl.moist, precipitation=nl.precipitation, 
		dtype=nl.dtype
	)

#============================================================
# Purpose-costumed code
#============================================================
from sympl import DataArray
import numpy as np
state['tendency_of_x_momentum_isentropic'] = taz.make_data_array_3d(
	np.zeros((nl.nx, nl.ny, nl.nz), dtype=nl.dtype), grid, units='kg m^-1 K^-1 s^-2'
)
state['tendency_of_y_momentum_isentropic'] = taz.make_data_array_3d(
	np.zeros((nl.nx, nl.ny, nl.nz), dtype=nl.dtype), grid, units='kg m^-1 K^-1 s^-2'
)
	
#============================================================
# The slow tendencies
#============================================================
args = []

if nl.moist:
	# Component calculating the microphysics
	ke = taz.Kessler(
		grid, air_pressure_on_interface_levels=True,
		rain_evaporation=nl.rain_evaporation, backend=nl.backend
	)
	args.append(ke)

# Component providing the thermal forcing
psh = taz.PrescribedSurfaceHeating(
	grid, 
	tendency_of_air_potential_temperature_in_diagnostics=True,
	tendency_of_air_potential_temperature_on_interface_levels=True,
	air_pressure_on_interface_levels=True,
	amplitude_at_day_sw=nl.amplitude_at_day_sw,
	amplitude_at_day_fw=nl.amplitude_at_day_fw,
	amplitude_at_night_sw=nl.amplitude_at_night_sw,
	amplitude_at_night_fw=nl.amplitude_at_night_fw,
	attenuation_coefficient_at_day=nl.attenuation_coefficient_at_day,
	attenuation_coefficient_at_night=nl.attenuation_coefficient_at_night,
	frequency_sw=nl.frequency_sw,
	frequency_fw=nl.frequency_fw,
	characteristic_length=nl.characteristic_length,
	starting_time=nl.starting_time, backend=nl.backend
)
args.append(psh)

# Component integrating the vertical flux
vf = taz.VerticalIsentropicAdvection(
	grid, moist=nl.moist, flux_scheme=nl.vertical_flux_scheme,
	tendency_of_air_potential_temperature_on_interface_levels=True,
	backend=nl.backend
)
args.append(vf)

# Wrap the components in a ConcurrentCoupling object
slow_tends = taz.ConcurrentCoupling(*args, execution_policy='serial')

#============================================================
# The intermediate tendencies
#============================================================
args = []

# Component calculating the pressure gradient in isentropic coordinates
order = 4 if nl.horizontal_flux_scheme == 'fifth_order_upwind' else 2
pg = taz.ConservativeIsentropicPressureGradient(
	grid, order=order,
	horizontal_boundary_type=nl.horizontal_boundary_type,
	backend=nl.backend, dtype=nl.dtype
)
args.append(pg)

if nl.coriolis:
	# Component calculating the Coriolis acceleration
	cf = taz.ConservativeIsentropicCoriolis(
		grid, coriolis_parameter=nl.coriolis_parameter, dtype=nl.dtype
	)
	args.append(cf)

# Wrap the components in a ConcurrentCoupling object
inter_tends = taz.ConcurrentCoupling(*args, execution_policy='serial')

#============================================================
# The intermediate diagnostics
#============================================================
args = []

# Component retrieving the diagnostic variables
pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(
	grid, moist=nl.moist, pt=pt, backend=nl.backend, dtype=nl.dtype
)
args.append(dv)

if nl.moist:
	# Component performing the saturation adjustment
	sa = taz.SaturationAdjustmentKessler(
		grid, air_pressure_on_interface_levels=True, backend=nl.backend
	)
	args.append(sa)

# Wrap the components in a DiagnosticComponentComposite object
inter_diags = taz.DiagnosticComponentComposite(*args)

#============================================================
# The fast tendencies
#============================================================
args = []

if nl.moist and nl.precipitation:
	# Component estimating the raindrop fall velocity
	rfv = taz.RaindropFallVelocity(grid, backend=nl.backend)
	args.append(rfv)

	# Component integrating the sedimentation flux
	sd = taz.Sedimentation(
		grid, sedimentation_flux_scheme='second_order_upwind',
		backend=nl.backend
	)
	args.append(sd)

# Wrap the components in a ConcurrentCoupling object
fast_tends = taz.ConcurrentCoupling(*args, execution_policy='serial') \
	if len(args) > 0 else None

#============================================================
# The dynamical core
#============================================================
dycore = taz.HomogeneousIsentropicDynamicalCore(
	grid, time_units='s', moist=nl.moist,
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

#============================================================
# A NetCDF monitor
#============================================================
if nl.filename is not None and nl.save_frequency > 0:
	if os.path.exists(nl.filename):
		os.remove(nl.filename)

	netcdf_monitor = taz.NetCDFMonitor(
		nl.filename, grid, store_names=nl.store_names
	)
	netcdf_monitor.store(state)

#============================================================
# Time-marching
#============================================================
dt = nl.timestep
nt = nl.niter

wall_time_start = time.time()
compute_time = 0.0

for i in range(nt):
	compute_time_start = time.time()

	# Update the (time-dependent) topography
	dycore.update_topography((i + 1) * dt)

	# Calculate the slow tendencies
	slow_tendencies, _ = slow_tends(state, dt)

	# Step the solution
	state_new = dycore(state, slow_tendencies, dt)

	# Update the state
	taz.dict_update(state, state_new)

	#========================================================
	# Purpose-costumed code
	#========================================================
	state['tendency_of_x_momentum_isentropic'] = slow_tendencies['x_momentum_isentropic']
	state['tendency_of_y_momentum_isentropic'] = slow_tendencies['y_momentum_isentropic']

	compute_time += time.time() - compute_time_start

	if (nl.print_frequency > 0) and ((i + 1) % nl.print_frequency == 0):
		u = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...] / \
			state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
		v = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...] / \
			state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]

		umax, umin = u.max(), u.min()
		vmax, vmin = v.max(), v.min()
		cfl = max(umax * dt.total_seconds() / grid.dx.to_units('m').values.item(),
				  vmax * dt.total_seconds() / grid.dy.to_units('m').values.item())

		# Print useful info
		print(
			'Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, '
						'vmax = {:8.4f} m/s, vmin = {:8.4f} m/s'.format(
				i + 1, cfl, umax, umin, vmax, vmin, 
			)
		)

	# Shortcuts
	to_save = (nl.filename is not None) and \
		(((nl.save_frequency > 0) and
		  ((i + 1) % nl.save_frequency == 0)) or i + 1 == nt)

	if to_save:
		# Save the solution
		netcdf_monitor.store(state)

print('Simulation successfully completed. HOORAY!')

#============================================================
# Post-processing
#============================================================
# Dump the solution to file
if nl.filename is not None and nl.save_frequency > 0:
	netcdf_monitor.write()

# Stop chronometer
wall_time = time.time() - wall_time_start

# Print logs
print('Total wall time: {}.'.format(taz.get_time_string(wall_time)))
print('Compute time: {}.'.format(taz.get_time_string(compute_time)))