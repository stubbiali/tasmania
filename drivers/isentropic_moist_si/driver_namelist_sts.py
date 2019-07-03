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
import time

import namelist_sts as nl


# ============================================================
# The underlying domain
# ============================================================
domain = taz.Domain(
	nl.domain_x, nl.nx, nl.domain_y, nl.ny, nl.domain_z, nl.nz,
	horizontal_boundary_type=nl.hb_type, nb=nl.nb,
	horizontal_boundary_kwargs=nl.hb_kwargs,
	topography_type=nl.topo_type,
	topography_kwargs=nl.topo_kwargs,
	dtype=nl.dtype
)
pgrid = domain.physical_grid
cgrid = domain.numerical_grid

# ============================================================
# The initial state
# ============================================================
state = taz.get_isentropic_state_from_brunt_vaisala_frequency(
	cgrid, nl.init_time, nl.x_velocity, nl.y_velocity,
	nl.brunt_vaisala, moist=True, precipitation=nl.precipitation,
	relative_humidity=nl.relative_humidity,	dtype=nl.dtype
)
domain.horizontal_boundary.reference_state = state

# ============================================================
# The dynamics
# ============================================================
pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dycore = taz.IsentropicDynamicalCore(
	domain, moist=True,
	# parameterizations
	intermediate_tendencies=None, intermediate_diagnostics=None,
	substeps=nl.substeps, fast_tendencies=None, fast_diagnostics=None,
	# numerical scheme
	time_integration_scheme=nl.time_integration_scheme,
	horizontal_flux_scheme=nl.horizontal_flux_scheme,
	time_integration_properties={'pt': pt, 'eps': nl.eps, 'a': nl.a, 'b': nl.b, 'c': nl.c},
	# vertical damping
	damp=nl.damp, damp_type=nl.damp_type, damp_depth=nl.damp_depth,
	damp_max=nl.damp_max, damp_at_every_stage=nl.damp_at_every_stage,
	# horizontal smoothing
	smooth=False, smooth_moist=False,
	# backend settings
	backend=nl.backend, dtype=nl.dtype
)

# ============================================================
# The physics
# ============================================================
args = []
ptis = nl.physics_time_integration_scheme

# component retrieving the diagnostic variables
idv = taz.IsentropicDiagnostics(
	domain, grid_type='numerical', moist=True, pt=pt,
	backend=nl.backend, dtype=nl.dtype
)
args.append({'component': idv})

if nl.coriolis:
	# component calculating the Coriolis acceleration
	cf = taz.IsentropicConservativeCoriolis(
		domain, grid_type='numerical',
		coriolis_parameter=nl.coriolis_parameter,
		backend=nl.backend, dtype=nl.dtype
	)
	args.append({'component': cf, 'time_integrator': ptis, 'substeps': 1})

if nl.smooth:
	# component performing the horizontal smoothing
	hs = taz.IsentropicHorizontalSmoothing(
		domain, nl.smooth_type, nl.smooth_coeff, nl.smooth_coeff_max,
		nl.smooth_damp_depth, moist=nl.smooth_moist,
		smooth_moist_coeff=nl.smooth_moist_coeff,
		smooth_moist_coeff_max=nl.smooth_moist_coeff_max,
		smooth_moist_damp_depth=nl.smooth_moist_damp_depth,
		backend=nl.backend, dtype=nl.dtype
	)
	args.append({'component': hs})

if nl.diff:
	# component calculating tendencies due to numerical diffusion
	hd = taz.IsentropicHorizontalDiffusion(
		domain, nl.diff_type, nl.diff_coeff, nl.diff_coeff_max, nl.diff_damp_depth,
		moist=nl.diff_moist, diffusion_moist_coeff=nl.diff_moist_coeff,
		diffusion_moist_coeff_max=nl.diff_moist_coeff_max,
		diffusion_moist_damp_depth=nl.diff_moist_damp_depth,
		backend=nl.backend, dtype=nl.dtype
	)
	args.append({'component': hd, 'time_integrator': ptis, 'substeps': 1})

if nl.turbulence:
	# component implementing the Smagorinsky turbulence model
	turb = taz.IsentropicSmagorinsky(
		domain, 'numerical', smagorinsky_constant=nl.smagorinsky_constant,
		backend=nl.backend, dtype=nl.dtype
	)
	args.append({'component': turb, 'time_integrator': ptis, 'substeps': 1})

if nl.coriolis or nl.smooth or nl.diff or nl.turbulence:
	# component retrieving the velocity components
	ivc = taz.IsentropicVelocityComponents(
		domain, backend=nl.backend, dtype=nl.dtype
	)
	args.append({'component': ivc})

# component calculating the microphysics
ke = taz.Kessler(
	domain, 'numerical', air_pressure_on_interface_levels=True,
	tendency_of_air_potential_temperature_in_diagnostics=True,
	rain_evaporation=nl.rain_evaporation,
	autoconversion_threshold=nl.autoconversion_threshold,
	autoconversion_rate=nl.autoconversion_rate,
	collection_rate=nl.collection_rate,
	backend=nl.backend, dtype=nl.dtype,
)
if nl.update_frequency > 0:
	from sympl import UpdateFrequencyWrapper
	args.append({
		'component': UpdateFrequencyWrapper(ke, nl.update_frequency * nl.timestep),
		'time_integrator': ptis, 'substeps': 1
	})
else:
	args.append({
		'component': ke, 'time_integrator': ptis, 'substeps': 1
	})

if nl.rain_evaporation:
	# include tendency_of_air_potential_temperature in the state
	state['tendency_of_air_potential_temperature'] = taz.make_dataarray_3d(
		np.zeros((cgrid.nx, cgrid.ny, cgrid.nz)), cgrid, 'K s^-1'
	)

	# component integrating the vertical flux
	vf = taz.IsentropicVerticalAdvection(
		domain, flux_scheme=nl.vertical_flux_scheme, moist=True,
		tendency_of_air_potential_temperature_on_interface_levels=False,
		backend=nl.backend, dtype=nl.dtype
	)
	args.append({'component': vf, 'time_integrator': 'rk3ws', 'substeps': 1})

if nl.precipitation:
	# component estimating the raindrop fall velocity
	rfv = taz.RaindropFallVelocity(domain, 'numerical', backend=nl.backend, dtype=nl.dtype)

	# component integrating the sedimentation flux
	sd = taz.Sedimentation(
		domain, 'numerical', sedimentation_flux_scheme=nl.sedimentation_flux_scheme,
		backend=nl.backend, dtype=nl.dtype
	)
	args.append({
		'component': taz.ConcurrentCoupling(rfv, sd), 'time_integrator': ptis, 'substeps': 1
	})

	# component calculating the accumulated precipitation
	ap = taz.Precipitation(domain, 'numerical', backend=nl.backend, dtype=nl.dtype)
	args.append({'component': ap})

# component clipping the negative values of the water species
water_species_names = (
	'mass_fraction_of_water_vapor_in_air',
	'mass_fraction_of_cloud_liquid_water_in_air',
	'mass_fraction_of_precipitation_water_in_air'
)
clp = taz.Clipping(domain, 'numerical', water_species_names)
args.append({'component': clp})

# component performing the saturation adjustment
sa = taz.SaturationAdjustmentKessler(
	domain, grid_type='numerical', air_pressure_on_interface_levels=True,
	backend=nl.backend, dtype=nl.dtype
)
args.append({'component': sa})

# wrap the components in a SequentialTendencySplitting object
physics = taz.SequentialTendencySplitting(*args)

# ============================================================
# A NetCDF monitor
# ============================================================
if nl.filename is not None:
	if os.path.exists(nl.filename):
		os.remove(nl.filename)

	netcdf_monitor = taz.NetCDFMonitor(
		nl.filename, domain, 'physical', store_names=nl.store_names
	)
	netcdf_monitor.store(state)

# ============================================================
# A visualization-purpose monitor
# ============================================================
xlim = nl.domain_x.to_units('km').values
ylim = nl.domain_y.to_units('km').values
zlim = nl.domain_z.to_units('K').values

# the drawers and the artist generating the left subplot
drawer1_properties = {
	'fontsize': 16, 'cmap_name': 'BuRd', 'cbar_on': True,
	'cbar_levels': 18, 'cbar_ticks_step': 4, 'cbar_center': 15,
	'cbar_orientation': 'horizontal',
	'cbar_x_label': 'Horizontal velocity [m s$^{-1}$]',
	'draw_vertical_levels': False,
}
drawer1 = taz.Contourf(
	cgrid, 'horizontal_velocity', 'm s^-1', z=-1,
	xaxis_units='km', yaxis_units='km', properties=drawer1_properties,
)
drawer2_properties = {
	'fontsize': 16, 'x_step': 2, 'y_step': 2, 'colors': 'black',
	'draw_vertical_levels': False, 'alpha': 0.5,
}
drawer2 = taz.Quiver(
	cgrid, z=-1, xaxis_units='km', yaxis_units='km',
	xcomp_name='x_velocity', xcomp_units='m s^-1',
	ycomp_name='y_velocity', ycomp_units='m s^-1',
	properties=drawer2_properties
)
axes1_properties = {
	'fontsize': 16, 'title_left': '$\\theta = {}$ K'.format(zlim[1]),
	'x_label': '$x$ [km]', 'x_lim': xlim,
	'y_label': '$y$ [km]', 'y_lim': ylim,
}
topo_drawer = taz.Contour(
	cgrid, 'topography', 'm', z=-1,
	xaxis_units='km', yaxis_units='km', properties={'colors': 'darkgray'}
)
plot1 = taz.Plot(drawer1, drawer2, topo_drawer, axes_properties=axes1_properties)

# The drawer and the artist generating the right subplot
drawer3_properties = {
	'fontsize': 16, 'cmap_name': 'BuRd', 'cbar_on': True,
	'cbar_levels': 18, 'cbar_ticks_step': 4, 'cbar_center': 15,
	'cbar_orientation': 'horizontal',
	'cbar_x_label': '$x$-velocity [m s$^{-1}$]',
	'draw_vertical_levels': True,
}
drawer3 = taz.Contourf(
	cgrid, 'x_velocity', 'm s^-1', y=int(nl.ny/2),
	xaxis_units='km', zaxis_name='z', zaxis_units='K',
	properties=drawer3_properties,
)
axes3_properties = {
	'fontsize': 16, 'title_left': '$y = {}$ km'.format(0.5*(ylim[0] + ylim[1])),
	'x_label': '$x$ [km]', 'x_lim': xlim,
	'y_label': '$\\theta$ [K]', 'y_lim': (zlim[1], zlim[0]),
}
topo_drawer = taz.LineProfile(
	cgrid, 'topography', 'km', y=int(nl.ny/2), z=-1, axis_units='km',
	properties={'linecolor': 'black', 'linewidth': 1.3}
)
plot2 = taz.Plot(drawer3, topo_drawer, axes_properties=axes3_properties)

# The monitor encompassing and coordinating the two artists
figure_properties = {'fontsize': 16, 'figsize': (12, 7), 'tight_layout': True}
plot_monitor = taz.PlotComposite(
	plot1, plot2, nrows=1, ncols=2,
	interactive=True, figure_properties=figure_properties
)

# ============================================================
# Time-marching
# ============================================================
dt = nl.timestep
nt = nl.niter

wall_time_start = time.time()
compute_time = 0.0

for i in range(nt):
	compute_time_start = time.time()

	# update the (time-dependent) topography
	dycore.update_topography((i + 1) * dt)

	# calculate the dynamics
	state_prv = dycore(state, {}, dt)

	# calculate the physics
	physics(state, state_prv, dt)

	# update the state
	taz.dict_update(state, state_prv)

	compute_time += time.time() - compute_time_start

	if (nl.print_dry_frequency > 0) and ((i + 1) % nl.print_dry_frequency == 0):
		u = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...] / \
			state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
		v = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...] / \
			state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]

		umax, umin = u.max(), u.min()
		vmax, vmin = v.max(), v.min()
		cfl = max(
			umax * dt.total_seconds() / pgrid.dx.to_units('m').values.item(),
			vmax * dt.total_seconds() / pgrid.dy.to_units('m').values.item()
		)

		# print useful info
		print(
			'Iteration {:6d}: CFL = {:4f}, umax = {:8.4f} m/s, umin = {:8.4f} m/s, '
			'vmax = {:8.4f} m/s, vmin = {:8.4f} m/s'.format(
				i + 1, cfl, umax, umin, vmax, vmin
			)
		)

	if (nl.print_moist_frequency > 0) and ((i + 1) % nl.print_moist_frequency == 0):
		qv_max = state['mass_fraction_of_water_vapor_in_air'].values[10:-10, 10:-10, 30:].max() * 1e3
		qc_max = state['mass_fraction_of_cloud_liquid_water_in_air'].values[10:-10, 10:-10, 30:].max() * 1e3
		qr_max = state['mass_fraction_of_precipitation_water_in_air'].values[10:-10, 10:-10, 30:].max() * 1e3
		if 'precipitation' in state:
			prec_max = state['precipitation'].to_units('mm hr^-1').values[10:-10, 10:-10].max()
			accprec_max = state['accumulated_precipitation'].to_units('mm').values[10:-10, 10:-10].max()
			print(
				'Iteration {:6d}: qvmax = {:8.4f} g/kg, qcmax = {:8.4f} g/kg, '
				'qrmax = {:8.4f} g/kg, prec_max = {:8.4f} mm/hr, accprec_max = {:8.4f} mm'
				.format(
					i+1, qv_max, qc_max, qr_max, prec_max, accprec_max
				)
			)
		else:
			print(
				'Iteration {:6d}: qvmax = {:8.4f} g/kg, qcmax = {:8.4f} g/kg, '
				'qrmax = {:8.4f} g/kg'.format(i+1, qv_max, qc_max, qr_max)
			)

	# shortcuts
	to_save = (nl.filename is not None) and \
		(((nl.save_frequency > 0) and
		 ((i + 1) % nl.save_frequency == 0)) or i + 1 == nt)
	to_plot = (nl.plot_frequency > 0) and ((i + 1) % nl.plot_frequency == 0)

	if to_save:
		# save the solution
		netcdf_monitor.store(state)

	if to_plot:
		# plot the solution
		plot1.axes_properties['title_right'] = str((i + 1) * dt)
		plot2.axes_properties['title_right'] = str((i + 1) * dt)
		fig = plot_monitor.store(
			((state, state, state), (state, state)), show=True
		)

print('Simulation successfully completed. HOORAY!')

# ============================================================
# Post-processing
# ============================================================
# dump the solution to file
if nl.filename is not None:
	netcdf_monitor.write()

# stop chronometer
wall_time = time.time() - wall_time_start

# print logs
print('Total wall time: {}.'.format(taz.get_time_string(wall_time)))
print('Compute time: {}.'.format(taz.get_time_string(compute_time)))

