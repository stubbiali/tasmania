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
	get_isentropic_state_from_brunt_vaisala_frequency
	get_isentropic_state_from_temperature
"""
import numpy as np
from sympl import DataArray

from tasmania.python.utils.data_utils import \
	get_physical_constants, make_dataarray_3d
from tasmania.python.utils.meteo_utils import \
	convert_relative_humidity_to_water_vapor

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


# convenient aliases
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


def get_isentropic_state_from_brunt_vaisala_frequency(
	grid, time, x_velocity, y_velocity, brunt_vaisala,
	moist=False, precipitation=False, relative_humidity=0.5,
	dtype=datatype, physical_constants=None
):
	"""
	Compute a valid state for the isentropic model given
	the Brunt-Vaisala frequency.

	Parameters
	----------
	grid : tasmania.Grid
		The underlying grid.
	time : datetime
		The time instant at which the state is defined.
	x_velocity : sympl.DataArray
		1-item :class:`sympl.DataArray` representing the uniform
		background x-velocity, in units compatible with [m s^-1].
	y_velocity : sympl.DataArray
		1-item :class:`sympl.DataArray` representing the uniform
		background y-velocity, in units compatible with [m s^-1].
	brunt_vaisala : sympl.DataArray
		1-item :class:`sympl.DataArray` representing the uniform
		Brunt-Vaisala frequency, in units compatible with [s^-1].
	moist : `bool`, optional
		:obj:`True` to include some water species in the model state,
		:obj:`False` for a fully dry configuration. Defaults to :obj:`False`.
	precipitation : `bool`, optional
		:obj:`True` if the model takes care of precipitation,
		:obj:`False` otherwise. Defaults to :obj:`False`.
	relative_humidity : `float`, optional
		The relative humidity in decimals. Defaults to 0.5.
	dtype : `numpy.dtype`, optional
		The data type for any :class:`numpy.ndarray` instantiated and
		used within this class.
	physical_constants : `dict`, optional
		Dictionary whose keys are strings indicating physical constants used
		within this object, and whose values are :class:`sympl.DataArray`\s
		storing the values and units of those constants. The constants might be:

			* 'gas_constant_of_dry_air', in units compatible with [J kg^-1 K^-1];
			* 'gravitational_acceleration', in units compatible with [m s^-2];
			* 'reference_air_pressure', in units compatible with [Pa];
			* 'specific_heat_of_dry_air_at_constant_pressure', \
				in units compatible with [J kg^-1 K^-1].

	Return
	------
	dict :
		The model state dictionary.
	"""
	# shortcuts
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dz = grid.dz.to_units('K').values.item()
	hs = grid.topography.profile.to_units('m').values
	bv = brunt_vaisala.to_units('s^-1').values.item()

	# get needed physical constants
	pcs  = get_physical_constants(_d_physical_constants, physical_constants)
	Rd   = pcs['gas_constant_of_dry_air']
	g    = pcs['gravitational_acceleration']
	pref = pcs['reference_air_pressure']
	cp   = pcs['specific_heat_of_dry_air_at_constant_pressure']

	# initialize the velocity components
	u = x_velocity.to_units('m s^-1').values.item() * np.ones((nx+1, ny, nz), dtype=dtype)
	v = y_velocity.to_units('m s^-1').values.item() * np.ones((nx, ny+1, nz), dtype=dtype)

	# compute the geometric height of the half levels
	theta1d = grid.z.to_units('K').values[np.newaxis, np.newaxis, :]
	theta   = np.tile(theta1d, (nx, ny, 1))
	h = np.zeros((nx, ny, nz+1), dtype=dtype)
	h[:, :, -1] = hs
	for k in range(nz-1, -1, -1):
		h[:, :, k] = h[:, :, k+1] + g * dz / ((bv**2) * theta[:, :, k])

	# initialize the Exner function
	exn = np.zeros((nx, ny, nz+1), dtype=dtype)
	exn[:, :, -1] = cp
	for k in range(nz-1, -1, -1):
		exn[:, :, k] = exn[:, :, k+1] - dz * (g**2) / ((bv**2) * (theta[:, :, k]**2))

	# diagnose the air pressure
	p = pref * ((exn / cp) ** (cp / Rd))

	# diagnose the Montgomery potential
	mtg_s = grid.z_on_interface_levels.to_units('K').values[-1] * exn[:, :, -1] + g * hs
	mtg = np.zeros((nx, ny, nz), dtype=dtype)
	mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
	for k in range(nz-2, -1, -1):
		mtg[:, :, k] = mtg[:, :, k+1] + dz * exn[:, :, k+1]

	# diagnose the isentropic density and the momenta
	s  = - (p[:, :, :-1] - p[:, :, 1:]) / (g * dz)
	su = 0.5 * s * (u[:-1, :, :] + u[1:, :, :])
	sv = 0.5 * s * (v[:, :-1, :] + v[:, 1:, :])

	# instantiate the return state
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
		# diagnose the air density and temperature
		rho  = s * dz / (h[:, :, :-1] - h[:, :, 1:])
		state['air_density'] = make_dataarray_3d(
			rho, grid, 'kg m^-3', name='air_density'
		)
		temp = 0.5 * (exn[:, :, :-1] + exn[:, :, 1:]) * theta / cp
		state['air_temperature'] = make_dataarray_3d(
			temp, grid, 'K', name='air_temperature'
		)

		# initialize the relative humidity
		rh = relative_humidity * np.ones((nx, ny, nz))
		rh_ = make_dataarray_3d(rh, grid, '1')

		# interpolate the pressure at the main levels
		p_unstg = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
		p_unstg_ = make_dataarray_3d(p_unstg, grid, 'Pa')

		# diagnose the mass fraction of water vapor
		qv = convert_relative_humidity_to_water_vapor(
			'goff_gratch', p_unstg_, state['air_temperature'], rh_
		)
		state[mfwv] = make_dataarray_3d(qv, grid, 'g g^-1', name=mfwv)

		# initialize the mass fraction of cloud liquid water and precipitation water
		qc = np.zeros((nx, ny, nz), dtype=dtype)
		state[mfcw] = make_dataarray_3d(qc, grid, 'g g^-1', name=mfcw)
		qr = np.zeros((nx, ny, nz), dtype=dtype)
		state[mfpw] = make_dataarray_3d(qr, grid, 'g g^-1', name=mfpw)

		# precipitation and accumulated precipitation
		if precipitation:
			state['precipitation'] = make_dataarray_3d(
				np.zeros((nx, ny, 1), dtype=dtype), grid, 'mm hr^-1',
				name='precipitation'
			)
			state['accumulated_precipitation'] = make_dataarray_3d(
				np.zeros((nx, ny, 1), dtype=dtype), grid, 'mm',
				name='accumulated_precipitation'
			)

	return state


def get_isentropic_state_from_temperature(
	grid, time, x_velocity, y_velocity, background_temperature,
	bubble_center_x=None, bubble_center_y=None, bubble_center_height=None,
	bubble_radius=None, bubble_maximum_perturbation=None,
	moist=False, precipitation=False, dtype=datatype, physical_constants=None
):
	"""
	Compute a valid state for the isentropic model given
	the air temperature.

	Parameters
	----------
	grid : tasmania.Grid
		The underlying grid.
	time : datetime
		The time instant at which the state is defined.
	x_velocity : sympl.DataArray
		1-item :class:`sympl.DataArray` representing the uniform
		background x-velocity, in units compatible with [m s^-1].
	y_velocity : sympl.DataArray
		1-item :class:`sympl.DataArray` representing the uniform
		background y-velocity, in units compatible with [m s^-1].
	background_temperature : sympl.DataArray
		1-item :class:`sympl.DataArray` representing the background
		temperature, in units compatible with [K].
	bubble_center_x : `sympl.DataArray`, optional
		1-item :class:`sympl.DataArray` representing the x-location
		of the center of the warm/cool bubble.
	bubble_center_y : `sympl.DataArray`, optional
		1-item :class:`sympl.DataArray` representing the y-location
		of the center of the warm/cool bubble.
	bubble_center_height : `sympl.DataArray`, optional
		1-item :class:`sympl.DataArray` representing the height
		of the center of the warm/cool bubble.
	bubble_radius : `sympl.DataArray`, optional
		1-item :class:`sympl.DataArray` representing the radius
		of the warm/cool bubble.
	bubble_maximum_perturbation : `sympl.DataArray`, optional
		1-item :class:`sympl.DataArray` representing the temperature
		perturbation in the center of the warm/cool bubble with respect
		to the ambient conditions.
	moist : `bool`, optional
		:obj:`True` to include some water species in the model state,
		:obj:`False` for a fully dry configuration. Defaults to :obj:`False`.
	precipitation : `bool`, optional
		:obj:`True` if the model takes care of precipitation,
		:obj:`False` otherwise. Defaults to :obj:`False`.
	dtype : `numpy.dtype`, optional
		The data type for any :class:`numpy.ndarray` instantiated and
		used within this class.
	physical_constants : `dict`, optional
		Dictionary whose keys are strings indicating physical constants used
		within this object, and whose values are :class:`sympl.DataArray`\s
		storing the values and units of those constants. The constants might be:

			* 'gas_constant_of_dry_air', in units compatible with [J kg^-1 K^-1];
			* 'gravitational_acceleration', in units compatible with [m s^-2];
			* 'reference_air_pressure', in units compatible with [Pa];
			* 'specific_heat_of_dry_air_at_constant_pressure', \
				in units compatible with [J kg^-1 K^-1].

	Return
	------
	dict :
		The model state dictionary.
	"""
	# shortcuts
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dz = grid.dz.to_units('K').values.item()

	# get needed physical constants
	pcs  = get_physical_constants(_d_physical_constants, physical_constants)
	Rd   = pcs['gas_constant_of_dry_air']
	g    = pcs['gravitational_acceleration']
	pref = pcs['reference_air_pressure']
	cp   = pcs['specific_heat_of_dry_air_at_constant_pressure']

	# initialize the air pressure
	theta1d = grid.z_on_interface_levels.to_units('K').values[np.newaxis, np.newaxis, :]
	theta   = np.tile(theta1d, (nx, ny, 1))
	temp	= background_temperature.to_units('K').values.item()
	p       = pref * ((temp / theta) ** (cp / Rd))

	# initialize the Exner function
	exn = cp * temp / theta

	# diagnose the height of the half levels
	hs = grid.topography.profile.to_units('m').values
	h = np.zeros((nx, ny, nz+1), dtype=dtype)
	h[:, :, -1] = hs
	for k in range(nz-1, -1, -1):
		h[:, :, k] = h[:, :, k+1] - Rd / (cp * g) * \
			(theta[:, :, k] * exn[:, :, k] + theta[:, :, k+1] * exn[:, :, k+1]) * \
			(p[:, :, k] - p[:, :, k+1]) / (p[:, :, k] + p[:, :, k+1])

	# warm/cool bubble
	if bubble_maximum_perturbation is not None:
		x = grid.x.to_units('m').values[:, np.newaxis, np.newaxis]
		y = grid.y.to_units('m').values[np.newaxis, :, np.newaxis]
		cx = bubble_center_x.to_units('m').values.item()
		cy = bubble_center_y.to_units('m').values.item()
		ch = bubble_center_height.to_units('m').values.item()
		r = bubble_radius.to_units('m').values.item()
		delta = bubble_maximum_perturbation.to_units('K').values.item()

		d = np.sqrt(((x - cx)**2 + (y - cy)**2 + (h - ch)**2) / r**2)
		t = temp * np.ones((nx, ny, nz+1), dtype=dtype) \
			+ delta * (np.cos(0.5 * np.pi * d))**2 * (d <= 1.0)
	else:
		t = temp * np.ones((nx, ny, nz+1), dtype=dtype)

	# diagnose the air pressure
	p = pref * ((t / theta) ** (cp / Rd))

	# diagnose the Exner function
	exn = cp * temp / theta

	# diagnose the Montgomery potential
	hs = grid.topography.profile.to_units('m').values
	mtg_s = cp * temp + g * hs
	mtg = np.zeros((nx, ny, nz), dtype=dtype)
	mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
	for k in range(nz-2, -1, -1):
		mtg[:, :, k] = mtg[:, :, k+1] + dz * exn[:, :, k+1]

	# initialize the velocity components
	u = x_velocity.to_units('m s^-1').values.item() * np.ones((nx+1, ny, nz), dtype=dtype)
	v = y_velocity.to_units('m s^-1').values.item() * np.ones((nx, ny+1, nz), dtype=dtype)

	# diagnose the isentropic density and the momenta
	s  = - (p[:, :, :-1] - p[:, :, 1:]) / (g * dz)
	su = 0.5 * s * (u[:-1, :, :] + u[1:, :, :])
	sv = 0.5 * s * (v[:, :-1, :] + v[:, 1:, :])

	# instantiate the return state
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
		# diagnose the air density and temperature
		rho  = s * dz / (h[:, :, :-1] - h[:, :, 1:])
		state['air_density'] = make_dataarray_3d(
			rho, grid, 'kg m^-3', name='air_density'
		)
		state['air_temperature'] = make_dataarray_3d(
			0.5*(t[:, :, :-1] + t[:, :, 1:]), grid, 'K', name='air_temperature'
		)

		# initialize the relative humidity
		rhmax, L, kc = 0.98, 10, 11
		k  = (nz-1) - np.arange(kc-L+1, kc+L)
		rh = np.zeros((nx, ny, nz), dtype=dtype)
		rh[:, :, k] = rhmax * (np.cos(abs(k - kc) * np.pi / (2. * L)))**2
		rh_ = make_dataarray_3d(rh, grid, '1')

		# interpolate the pressure at the main levels
		p_unstg = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
		p_unstg_ = make_dataarray_3d(p_unstg, grid, 'Pa')

		# diagnose the mass fraction fo water vapor
		qv = convert_relative_humidity_to_water_vapor(
			'goff_gratch', p_unstg_, state['air_temperature'], rh_
		)
		state[mfwv] = make_dataarray_3d(qv, grid, 'g g^-1', name=mfwv)

		# initialize the mass fraction of cloud liquid water and precipitation water
		qc = np.zeros((nx, ny, nz), dtype=dtype)
		state[mfcw] = make_dataarray_3d(qc, grid, 'g g^-1', name=mfcw)
		qr = np.zeros((nx, ny, nz), dtype=dtype)
		state[mfpw] = make_dataarray_3d(qr, grid, 'g g^-1', name=mfpw)

		# precipitation and accumulated precipitation
		if precipitation:
			state['precipitation'] = make_dataarray_3d(
				np.zeros((nx, ny), dtype=dtype), grid, 'mm hr^-1',
				name='precipitation'
			)
			state['accumulated_precipitation'] = make_dataarray_3d(
				np.zeros((nx, ny), dtype=dtype), grid, 'mm',
				name='accumulated_precipitation'
			)

	return state
