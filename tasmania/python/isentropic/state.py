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
"""
This module contains:
	get_default_isentropic_state
	get_isothermal_isentropic_state
"""
import numpy as np
from sympl import DataArray

from tasmania.python.utils.data_utils import \
	get_physical_constants, make_dataarray_2d, make_dataarray_3d

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


_d_physical_constants = {
	'gas_constant_of_dry_air':
		DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
	'gravitational_acceleration':
		DataArray(9.81, attrs={'units': 'm s^-2'}),
	'reference_air_pressure':
		DataArray(1.0e5, attrs={'units': 'Pa'}),
	'specific_heat_of_dry_air_at_constant_pressure':
		DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
}


# Convenient aliases
mf_wv  = 'mass_fraction_of_water_vapor_in_air'
mf_clw = 'mass_fraction_of_cloud_liquid_water_in_air'
mf_pw  = 'mass_fraction_of_precipitation_water_in_air'


def get_default_isentropic_state(
	grid, time, x_velocity, y_velocity, brunt_vaisala,
	moist=False, precipitation=False, dtype=datatype, physical_constants=None
):
	# Shortcuts
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dz = grid.dz.to_units('K').values.item()
	hs = grid.topography_height
	bv = brunt_vaisala.to_units('s^-1').values.item()

	# Get needed physical constants
	pcs  = get_physical_constants(_d_physical_constants, physical_constants)
	Rd   = pcs['gas_constant_of_dry_air']
	g    = pcs['gravitational_acceleration']
	pref = pcs['reference_air_pressure']
	cp   = pcs['specific_heat_of_dry_air_at_constant_pressure']

	# Initialize the velocity components
	u = x_velocity.to_units('m s^-1').values.item() * np.ones((nx+1, ny, nz), dtype=dtype)
	v = y_velocity.to_units('m s^-1').values.item() * np.ones((nx, ny+1, nz), dtype=dtype)

	# Compute the geometric height of the half levels
	theta1d = grid.z.to_units('K').values[np.newaxis, np.newaxis, :]
	theta   = np.tile(theta1d, (nx, ny, 1))
	h = np.zeros((nx, ny, nz+1), dtype=dtype)
	h[:, :, -1] = hs
	for k in range(nz-1, -1, -1):
		h[:, :, k] = h[:, :, k+1] + g * dz / ((bv**2) * theta[:, :, k])

	# Initialize the Exner function
	exn = np.zeros((nx, ny, nz+1), dtype=dtype)
	exn[:, :, -1] = cp
	for k in range(nz-1, -1, -1):
		exn[:, :, k] = exn[:, :, k+1] - dz * (g**2) / ((bv**2) * (theta[:, :, k]**2))

	# Diagnose the air pressure
	p = pref * ((exn / cp) ** (cp / Rd))

	# Diagnose the Montgomery potential
	mtg_s = grid.z_on_interface_levels.to_units('K').values[-1] * exn[:, :, -1] + g * hs
	mtg = np.zeros((nx, ny, nz), dtype=dtype)
	mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
	for k in range(nz-2, -1, -1):
		mtg[:, :, k] = mtg[:, :, k+1] + dz * exn[:, :, k+1]

	# Diagnose the isentropic density and the momentums
	s  = - (p[:, :, :-1] - p[:, :, 1:]) / (g * dz)
	su = 0.5 * s * (u[:-1, :, :] + u[1:, :, :])
	sv = 0.5 * s * (v[:, :-1, :] + v[:, 1:, :])

	# Instantiate the return state
	state = {
		'time': time,
		'air_isentropic_density':
			make_dataarray_3d(
				s, grid, 'kg m^-2 K^-1', name='air_isentropic_density'
			),
		'air_pressure_on_interface_levels':
			make_dataarray_3d(
				p, grid, 'Pa', name='air_pressure_on_interface_levels'
			),
		'exner_function_on_interface_levels':
			make_dataarray_3d(
				exn, grid, 'J K^-1 kg^-1', name='exner_function_on_interface_levels'
			),
		'height_on_interface_levels':
			make_dataarray_3d(
				h, grid, 'm', name='height_on_interface_levels'
			),
		'montgomery_potential':
			make_dataarray_3d(
				mtg, grid, 'J kg^-1', name='montgomery_potential'
			),
		'x_momentum_isentropic':
			make_dataarray_3d(
				su, grid, 'kg m^-1 K^-1 s^-1', name='x_momentum_isentropic'
			),
		'x_velocity_at_u_locations':
			make_dataarray_3d(
				u, grid, 'm s^-1', name='x_velocity_at_u_locations'
			),
		'y_momentum_isentropic':
			make_dataarray_3d(
				sv, grid, 'kg m^-1 K^-1 s^-1', name='y_momentum_isentropic'
			),
		'y_velocity_at_v_locations':
			make_dataarray_3d(
				v, grid, 'm s^-1', name='y_velocity_at_v_locations'
			),
	}

	if moist:
		# Diagnose the air density and temperature
		rho  = s * dz / (h[:, :, :-1] - h[:, :, 1:])
		state['air_density'] = make_dataarray_3d(rho, grid, 'kg m^-3', name='air_density')
		temp = 0.5 * (exn[:, :, :-1] + exn[:, :, 1:]) * theta / cp
		state['air_temperature'] = make_dataarray_3d(temp, grid, 'K', name='air_temperature')

		# Initialize the relative humidity
		rhmax, L, kc = 0.98, 10, 11
		k  = (nz-1) - np.arange(kc-L+1, kc+L)
		rh = np.zeros((nx, ny, nz), dtype=dtype)
		rh[:, :, k] = rhmax * (np.cos(abs(k - kc) * np.pi / (2. * L)))**2
		rh_ = make_dataarray_3d(rh, grid, '1')

		# Interpolate the pressure at the main levels
		p_unstg = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
		p_unstg_ = make_dataarray_3d(p_unstg, grid, 'Pa')

		# Diagnose the mass fraction fo water vapor
		from tasmania.python.utils.meteo_utils import \
			convert_relative_humidity_to_water_vapor
		qv = convert_relative_humidity_to_water_vapor(
			'goff_gratch', p_unstg_, state['air_temperature'], rh_
		)
		state[mf_wv]  = make_dataarray_3d(qv, grid, 'g g^-1', name=mf_wv)

		# Initialize the mass fraction of cloud liquid water and precipitation water
		qc = np.zeros((nx, ny, nz), dtype=dtype)
		state[mf_clw] = make_dataarray_3d(qc, grid, 'g g^-1', name=mf_clw)
		qr = np.zeros((nx, ny, nz), dtype=dtype)
		state[mf_pw]  = make_dataarray_3d(qr, grid, 'g g^-1', name=mf_pw)

		# Precipitation and accumulated precipitation
		if precipitation:
			state['precipitation'] = make_dataarray_2d(
				np.zeros((nx, ny), dtype=dtype), grid, 'mm hr^-1',
				name='precipitation'
			)
			state['accumulated_precipitation'] = make_dataarray_2d(
				np.zeros((nx, ny), dtype=dtype), grid, 'mm',
				name='accumulated_precipitation'
			)

	return state


def get_isothermal_isentropic_state(
	grid, time, x_velocity, y_velocity, temperature,
	moist=False, precipitation=False, dtype=datatype, physical_constants=None
):
	# Shortcuts
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dz = grid.dz.to_units('K').values.item()

	# Get needed physical constants
	pcs  = get_physical_constants(_d_physical_constants, physical_constants)
	Rd   = pcs['gas_constant_of_dry_air']
	g    = pcs['gravitational_acceleration']
	pref = pcs['reference_air_pressure']
	cp   = pcs['specific_heat_of_dry_air_at_constant_pressure']

	# Initialize the velocity components
	u = x_velocity.to_units('m s^-1').values.item() * np.ones((nx+1, ny, nz), dtype=dtype)
	v = y_velocity.to_units('m s^-1').values.item() * np.ones((nx, ny+1, nz), dtype=dtype)

	# Initialize the air pressure
	theta1d = grid.z_on_interface_levels.to_units('K').values[np.newaxis, np.newaxis, :]
	theta   = np.tile(theta1d, (nx, ny, 1))
	temp	= temperature.to_units('K').values.item()
	p       = pref * ((temp / theta) ** (cp / Rd))

	# Initialize the Exner function
	exn = cp * temp / theta

	# Diagnose the Montgomery potential
	hs = grid.topography_height
	mtg_s = cp * temp + g * hs
	mtg = np.zeros((nx, ny, nz), dtype=dtype)
	mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
	for k in range(nz-2, -1, -1):
		mtg[:, :, k] = mtg[:, :, k+1] + dz * exn[:, :, k+1]

	# Diagnose the height of the half levels
	h = np.zeros((nx, ny, nz+1), dtype=dtype)
	h[:, :, -1] = hs
	for k in range(nz-1, -1, -1):
		h[:, :, k] = h[:, :, k+1] - Rd / (cp * g) * \
					 (theta[:, :, k] * exn[:, :, k] + theta[:, :, k+1] * exn[:, :, k+1]) * \
					 (p[:, :, k] - p[:, :, k+1]) / (p[:, :, k] + p[:, :, k+1])

	# Diagnose the isentropic density and the momentums
	s  = - (p[:, :, :-1] - p[:, :, 1:]) / (g * dz)
	su = 0.5 * s * (u[:-1, :, :] + u[1:, :, :])
	sv = 0.5 * s * (v[:, :-1, :] + v[:, 1:, :])

	# Instantiate the return state
	state = {
		'time': time,
		'air_isentropic_density':
			make_dataarray_3d(
				s, grid, 'kg m^-2 K^-1', name='air_isentropic_density'
			),
		'air_pressure_on_interface_levels':
			make_dataarray_3d(
				p, grid, 'Pa', name='air_pressure_on_interface_levels'
			),
		'exner_function_on_interface_levels':
			make_dataarray_3d(
				exn, grid, 'J K^-1 kg^-1', name='exner_function_on_interface_levels'
			),
		'height_on_interface_levels':
			make_dataarray_3d(
				h, grid, 'm', name='height_on_interface_levels'
			),
		'montgomery_potential':
			make_dataarray_3d(
				mtg, grid, 'J kg^-1', name='montgomery_potential'
			),
		'x_momentum_isentropic':
			make_dataarray_3d(
				su, grid, 'kg m^-1 K^-1 s^-1', name='x_momentum_isentropic'
			),
		'x_velocity_at_u_locations':
			make_dataarray_3d(
				u, grid, 'm s^-1', name='x_velocity_at_u_locations'
			),
		'y_momentum_isentropic':
			make_dataarray_3d(
				sv, grid, 'kg m^-1 K^-1 s^-1', name='y_momentum_isentropic'
			),
		'y_velocity_at_v_locations':
			make_dataarray_3d(
				v, grid, 'm s^-1', name='y_velocity_at_v_locations'
			),
	}

	if moist:
		# Diagnose the air density and temperature
		rho  = s * dz / (h[:, :, :-1] - h[:, :, 1:])
		state['air_density'] = make_dataarray_3d(rho, grid, 'kg m^-3', name='air_density')
		temp = 0.5 * (exn[:, :, :-1] + exn[:, :, 1:]) * theta / cp
		state['air_temperature'] = make_dataarray_3d(temp, grid, 'K', name='air_temperature')

		# Initialize the relative humidity
		rhmax, L, kc = 0.98, 10, 11
		k  = (nz-1) - np.arange(kc-L+1, kc+L)
		rh = np.zeros((nx, ny, nz), dtype=dtype)
		rh[:, :, k] = rhmax * (np.cos(abs(k - kc) * np.pi / (2. * L)))**2
		rh_ = make_dataarray_3d(rh, grid, '1')

		# Interpolate the pressure at the main levels
		p_unstg = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
		p_unstg_ = make_dataarray_3d(p_unstg, grid, 'Pa')

		# Diagnose the mass fraction fo water vapor
		from tasmania.python.utils.meteo_utils import \
			convert_relative_humidity_to_water_vapor
		qv = convert_relative_humidity_to_water_vapor(
			'goff_gratch', p_unstg_, state['air_temperature'], rh_
		)
		state[mf_wv]  = make_dataarray_3d(qv, grid, 'g g^-1', name=mf_wv)

		# Initialize the mass fraction of cloud liquid water and precipitation water
		qc = np.zeros((nx, ny, nz), dtype=dtype)
		state[mf_clw] = make_dataarray_3d(qc, grid, 'g g^-1', name=mf_clw)
		qr = np.zeros((nx, ny, nz), dtype=dtype)
		state[mf_pw]  = make_dataarray_3d(qr, grid, 'g g^-1', name=mf_pw)

		# Precipitation and accumulated precipitation
		if precipitation:
			state['precipitation'] = make_dataarray_2d(
				np.zeros((nx, ny), dtype=dtype), grid, 'mm hr^-1',
				name='precipitation'
			)
			state['accumulated_precipitation'] = make_dataarray_2d(
				np.zeros((nx, ny), dtype=dtype), grid, 'mm',
				name='accumulated_precipitation'
			)

	return state
