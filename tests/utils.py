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
from hypothesis import assume, strategies as hyp_st
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import os
from sympl import DataArray
from sympl._core.units import clean_units
import sys

import tasmania as taz
from tasmania.python.utils.data_utils import get_physical_constants

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf


default_physical_constants = {
	'gas_constant_of_dry_air':
		DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
	'gravitational_acceleration':
		DataArray(9.81, attrs={'units': 'm s^-2'}),
	'reference_air_pressure':
		DataArray(1.0e5, attrs={'units': 'Pa'}),
	'specific_heat_of_dry_air_at_constant_pressure':
		DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
}


def get_float_width(dtype):
	"""
	Get the number of bits used by `dtype`.
	"""
	if dtype == np.float16:
		return 16
	elif dtype == np.float32:
		return 32
	else:
		return 64


def get_interval(el0, el1, dims, units, increasing):
	"""
	Generate a 2-elements DataArray representing a domain interval.
	"""
	invert = ((el0 > el1) and increasing) or ((el0 < el1) and not increasing)
	return DataArray(
		[el1, el0] if invert else [el0, el1],
		dims=dims, attrs={'units': units}
	)


def st_floats(**kwargs):
	"""
	Strategy drawing a non-nan and non-infinite floats.
	"""
	kwargs.pop('allow_nan', None)
	kwargs.pop('allow_infinite', None)
	return hyp_st.floats(
		allow_nan=False, allow_infinity=False, **kwargs
	)


def st_one_of(seq):
	"""
	Strategy drawing one of the elements of the input sequence.
	"""
	return hyp_st.one_of(hyp_st.just(el) for el in seq)


@hyp_st.composite
def st_interval(draw, *, axis_name='x'):
	"""
	Strategy drawing an interval along the `axis_name`-axis according to the
	specifications defined in `conf.py`.
	"""
	axis_properties = eval('conf.axis_{}'.format(axis_name))
	units = draw(st_one_of(axis_properties['units_to_range'].keys()))

	el0 = draw(
		st_floats(
			min_value=axis_properties['units_to_range'][units][0],
			max_value=axis_properties['units_to_range'][units][1],
		)
	)
	el1 = draw(
		st_floats(
			min_value=axis_properties['units_to_range'][units][0],
			max_value=axis_properties['units_to_range'][units][1],
		)
	)
	assume(el0 != el1)

	return get_interval(
		el0, el1,
		draw(st_one_of(axis_properties['dims'])),
		draw(hyp_st.just(units)),
		draw(hyp_st.just(axis_properties['increasing']))
	)


def st_length(axis_name='x', min_value=None, max_value=None):
	"""
	Strategy drawing the number of grid points in the `axis_name`-direction.
	"""
	axis_properties = eval('conf.axis_{}'.format(axis_name))
	min_val = axis_properties['length'][0] if min_value is None else min_value
	max_val = axis_properties['length'][1] if max_value is None else max_value
	return hyp_st.integers(min_value=min_val, max_value=max_val)


@hyp_st.composite
def st_topography1d_kwargs(
	draw, x, *, topo_type=None, topo_time=None, topo_max_height=None,
	topo_center=None, topo_half_width=None, topo_str=None, topo_smooth=None
):
	"""
	Strategy drawing a set of keyword arguments accepted by the constructor
	of :class:`tasmania.Topography1d`.
	"""
	if topo_type is not None and isinstance(topo_type, str):
		_topo_type = topo_type
	else:
		_topo_type = draw(st_one_of(conf.topography1d['type']))

	if topo_time is not None and isinstance(topo_time, timedelta):
		_topo_time = topo_time
	else:
		_topo_time = draw(
			hyp_st.timedeltas(
				min_value=timedelta(seconds=0),
				max_value=conf.topography1d['time_max'],
			)
		)

	if (
		topo_max_height is not None and
		isinstance(topo_max_height, DataArray) and
		topo_max_height.shape == ()
	):
		_topo_max_height = topo_max_height
	else:
		units = draw(st_one_of(conf.topography1d['units_to_max_height'].keys()))
		val = draw(
			st_floats(
				min_value=conf.topography1d['units_to_max_height'][units][0],
				max_value=conf.topography1d['units_to_max_height'][units][1],
			)
		)
		_topo_max_height = DataArray(val, attrs={'units': units})

	if (
		topo_center is not None and
		isinstance(topo_center, DataArray) and
		topo_center.shape == ()
	):
		_topo_center = topo_center
	else:
		val = draw(
			st_floats(
				min_value=np.min(x.values),
				max_value=np.max(x.values),
			)
		)
		_topo_center = DataArray(val, attrs={'units': x.attrs['units']})

	if (
		topo_half_width is not None and
		isinstance(topo_half_width, DataArray) and
		topo_half_width.shape == ()
	):
		_topo_half_width = topo_half_width
	else:
		units = draw(st_one_of(conf.topography1d['units_to_half_width'].keys()))
		val = draw(
			st_floats(
				min_value=conf.topography1d['units_to_half_width'][units][0],
				max_value=conf.topography1d['units_to_half_width'][units][1],
			)
		)
		_topo_half_width = DataArray(val, attrs={'units': units})

	if topo_str is not None and isinstance(topo_str, str):
		_topo_str = topo_str
	else:
		_topo_str = draw(st_one_of(conf.topography1d['str']))

	if topo_smooth is not None and isinstance(topo_smooth, bool):
		_topo_smooth = topo_smooth
	else:
		_topo_smooth = draw(hyp_st.booleans())

	return {
		'topo_type': _topo_type,
		'topo_time': _topo_time,
		'topo_max_height': _topo_max_height,
		'topo_center_x': _topo_center,
		'topo_width_x': _topo_half_width,
		'topo_str': _topo_str,
		'topo_smooth': _topo_smooth,
	}


@hyp_st.composite
def st_topography2d_kwargs(
	draw, x, y, *, topo_type=None, topo_time=None,
	topo_max_height=None, topo_center_x=None, topo_center_y=None,
	topo_half_width_x=None, topo_half_width_y=None,
	topo_str=None, topo_smooth=None
):
	"""
	Strategy drawing a set of keyword arguments accepted by the constructor
	of :class:`tasmania.Topography1d`.
	"""
	if topo_type is not None and isinstance(topo_type, str):
		_topo_type = topo_type
	else:
		_topo_type = draw(st_one_of(conf.topography2d['type']))

	if topo_time is not None and isinstance(topo_time, timedelta):
		_topo_time = topo_time
	else:
		_topo_time = draw(
			hyp_st.timedeltas(
				min_value=timedelta(seconds=0),
				max_value=conf.topography2d['time_max'],
			)
		)

	if (
		topo_max_height is not None and
		isinstance(topo_max_height, DataArray) and
		topo_max_height.shape == ()
	):
		_topo_max_height = topo_max_height
	else:
		units = draw(st_one_of(conf.topography1d['units_to_max_height'].keys()))
		val = draw(
			st_floats(
				min_value=conf.topography2d['units_to_max_height'][units][0],
				max_value=conf.topography2d['units_to_max_height'][units][1],
			)
		)
		_topo_max_height = DataArray(val, attrs={'units': units})

	if (
		topo_center_x is not None and
		isinstance(topo_center_x, DataArray) and
		topo_center_x.shape == ()
	):
		_topo_center_x = topo_center_x
	else:
		val = draw(
			st_floats(
				min_value=np.min(x.values),
				max_value=np.max(x.values),
			)
		)
		_topo_center_x = DataArray(val, attrs={'units': x.attrs['units']})

	if (
		topo_center_y is not None and
		isinstance(topo_center_y, DataArray) and
		topo_center_y.shape == ()
	):
		_topo_center_y = topo_center_y
	else:
		val = draw(
			st_floats(
				min_value=np.min(y.values),
				max_value=np.max(y.values),
			)
		)
		_topo_center_y = DataArray(val, attrs={'units': y.attrs['units']})

	if (
		topo_half_width_x is not None and
		isinstance(topo_half_width_x, DataArray) and
		topo_half_width_x.shape == ()
	):
		_topo_half_width_x = topo_half_width_x
	else:
		units = draw(st_one_of(conf.topography2d['units_to_half_width_x'].keys()))
		val = draw(
			st_floats(
				min_value=conf.topography2d['units_to_half_width_x'][units][0],
				max_value=conf.topography2d['units_to_half_width_x'][units][1],
			)
		)
		_topo_half_width_x = DataArray(val, attrs={'units': units})

	if (
		topo_half_width_y is not None and
		isinstance(topo_half_width_y, DataArray) and
		topo_half_width_y.shape == ()
	):
		_topo_half_width_y = topo_half_width_y
	else:
		units = draw(st_one_of(conf.topography2d['units_to_half_width_y'].keys()))
		val = draw(
			st_floats(
				min_value=conf.topography2d['units_to_half_width_y'][units][0],
				max_value=conf.topography2d['units_to_half_width_y'][units][1],
			)
		)
		_topo_half_width_y = DataArray(val, attrs={'units': units})

	if topo_str is not None and isinstance(topo_str, str):
		_topo_str = topo_str
	else:
		_topo_str = draw(st_one_of(conf.topography2d['str']))

	if topo_smooth is not None and isinstance(topo_smooth, bool):
		_topo_smooth = topo_smooth
	else:
		_topo_smooth = draw(hyp_st.booleans())

	return {
		'topo_type': _topo_type,
		'topo_time': _topo_time,
		'topo_max_height': _topo_max_height,
		'topo_center_x': _topo_center_x,
		'topo_center_y': _topo_center_y,
		'topo_width_x': _topo_half_width_x,
		'topo_width_y': _topo_half_width_y,
		'topo_str': _topo_str,
		'topo_smooth': _topo_smooth,
	}


@hyp_st.composite
def st_grid_xyz(
	draw, *, xaxis_name='x', xaxis_length=None,
	yaxis_name='y', yaxis_length=None,  zaxis_name='z', zaxis_length=None
):
	"""
	Strategy drawing a :class:`tasmania.GridXYZ` object.
	"""
	domain_x = draw(st_interval(axis_name=xaxis_name))
	nx = draw(
		st_length(axis_name=xaxis_name) if xaxis_length is None else
		st_length(axis_name=xaxis_name, min_value=xaxis_length[0], max_value=xaxis_length[1])
	)

	domain_y = draw(st_interval(axis_name=yaxis_name))
	ny = draw(
		st_length(axis_name=yaxis_name) if yaxis_length is None else
		st_length(axis_name=yaxis_name, min_value=yaxis_length[0], max_value=yaxis_length[1])
	)

	domain_z = draw(st_interval(axis_name=zaxis_name))
	nz = draw(
		st_length(axis_name=zaxis_name) if zaxis_length is None else
		st_length(axis_name=zaxis_name, min_value=zaxis_length[0], max_value=zaxis_length[1])
	)

	topo_kwargs = draw(st_topography2d_kwargs(domain_x, domain_y))
	topo_type = topo_kwargs.pop('topo_type')
	topo_time = topo_kwargs.pop('topo_time')

	dtype = draw(st_one_of(conf.datatype))

	return taz.GridXYZ(
		domain_x, nx, domain_y, ny, domain_z, nz,
		topo_type=topo_type, topo_time=topo_time, topo_kwargs=topo_kwargs,
		dtype=dtype
	)


@hyp_st.composite
def st_grid_xy(
	draw, *, xaxis_name='x', xaxis_length=None, yaxis_name='y', yaxis_length=None
):
	"""
	Strategy drawing a :class:`tasmania.GridXY` object.
	"""
	domain_x = draw(st_interval(axis_name=xaxis_name))
	nx = draw(
		st_length(axis_name=xaxis_name) if xaxis_length is None else
		st_length(axis_name=xaxis_name, min_value=xaxis_length[0], max_value=xaxis_length[1])
	)

	domain_y = draw(st_interval(axis_name=yaxis_name))
	ny = draw(
		st_length(axis_name=yaxis_name) if yaxis_length is None else
		st_length(axis_name=yaxis_name, min_value=yaxis_length[0], max_value=yaxis_length[1])
	)

	dtype = draw(st_one_of(conf.datatype))

	return taz.GridXY(domain_x, nx, domain_y, ny, dtype=dtype)


@hyp_st.composite
def st_grid_xz(
	draw, *, xaxis_name='x', xaxis_length=None, zaxis_name='z', zaxis_length=None
):
	"""
	Strategy drawing a :class:`tasmania.GridXZ` object.
	"""
	domain_x = draw(st_interval(axis_name=xaxis_name))
	nx = draw(
		st_length(axis_name=xaxis_name) if xaxis_length is None else
		st_length(axis_name=xaxis_name, min_value=xaxis_length[0], max_value=xaxis_length[1])
	)

	domain_z = draw(st_interval(axis_name=zaxis_name))
	nz = draw(
		st_length(axis_name=zaxis_name) if zaxis_length is None else
		st_length(axis_name=zaxis_name, min_value=zaxis_length[0], max_value=zaxis_length[1])
	)

	topo_kwargs = draw(st_topography1d_kwargs(domain_x))
	topo_type = topo_kwargs.pop('topo_type')
	topo_time = topo_kwargs.pop('topo_time')

	dtype = draw(st_one_of(conf.datatype))

	return taz.GridXZ(
		domain_x, nx, domain_z, nz,
		topo_type=topo_type, topo_time=topo_time, topo_kwargs=topo_kwargs,
		dtype=dtype
	)


@hyp_st.composite
def st_grid_yz(
	draw, *, yaxis_name='y', yaxis_length=None, zaxis_name='z', zaxis_length=None
):
	"""
	Strategy drawing a :class:`tasmania.GridYZ` object.
	"""
	domain_y = draw(st_interval(axis_name=yaxis_name))
	ny = draw(
		st_length(axis_name=yaxis_name) if yaxis_length is None else
		st_length(axis_name=yaxis_name, min_value=yaxis_length[0], max_value=yaxis_length[1])
	)

	domain_z = draw(st_interval(axis_name=zaxis_name))
	nz = draw(
		st_length(axis_name=zaxis_name) if zaxis_length is None else
		st_length(axis_name=zaxis_name, min_value=zaxis_length[0], max_value=zaxis_length[1])
	)

	topo_kwargs = draw(st_topography1d_kwargs(domain_y))
	topo_type = topo_kwargs.pop('topo_type')
	topo_time = topo_kwargs.pop('topo_time')

	dtype = draw(st_one_of(conf.datatype))

	return taz.GridXZ(
		domain_y, ny, domain_z, nz,
		topo_type=topo_type, topo_time=topo_time, topo_kwargs=topo_kwargs,
		dtype=dtype
	)


@hyp_st.composite
def st_field(draw, grid, properties_name, field_name, shape=None):
	"""
	Strategy drawing a random field for the variable `field_name`.
	"""
	properties_dict = eval('conf.{}'.format(properties_name))
	units = draw(st_one_of(properties_dict[field_name].keys()))

	shape = shape if shape is not None else (grid.nx, grid.ny, grid.nz)

	raw_field = draw(
		st_arrays(
			grid.x.dtype, shape,
			elements=st_floats(
				min_value=properties_dict[field_name][units][0],
				max_value=properties_dict[field_name][units][1],
			),
			fill=hyp_st.nothing(),
		)
	)

	return taz.make_data_array_3d(raw_field, grid, units, name=field_name)


@hyp_st.composite
def st_isentropic_state(draw, grid, *, time=None, moist=False, precipitation=False):
	"""
	Strategy drawing a valid isentropic model state over `grid`.
	"""
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dz = grid.dz.to_units('K').values.item()
	dtype = grid.x.dtype

	return_dict = {}

	# time
	if time is None:
		time = draw(hyp_st.datetimes())
	return_dict['time'] = time

	# air isentropic density
	return_dict['air_isentropic_density'] = draw(
		st_field(grid, 'isentropic_state', 'air_isentropic_density', (nx, ny, nz))
	)

	# x-velocity
	return_dict['x_velocity_at_u_locations'] = draw(
		st_field(grid, 'isentropic_state', 'x_velocity_at_u_locations', (nx+1, ny, nz))
	)

	# x-momentum
	s = return_dict['air_isentropic_density']
	u = return_dict['x_velocity_at_u_locations']
	s_units = s.attrs['units']
	u_units = u.attrs['units']
	su_units = clean_units(s_units + u_units)
	su_raw = s.to_units('kg m^-2 k^-1').values * \
		0.5 * (
			u.to_units('m s^-1').values[:-1, :, :] + u.to_units('m s^-1').values[1:, :, :]
		)
	return_dict['x_momentum_isentropic'] = taz.make_data_array_3d(
		su_raw, grid, 'kg m^-1 k^-1 s^-1', name='x_momentum_isentropic'
	).to_units(su_units)

	# y-velocity
	return_dict['y_velocity_at_v_locations'] = draw(
		st_field(grid, 'isentropic', 'y_velocity_at_v_locations', (nx, ny+1, nz))
	)

	# y-momentum
	v = return_dict['y_velocity_at_v_locations']
	v_units = v.attrs['units']
	sv_units = clean_units(s_units + v_units)
	sv_raw = s.to_units('kg m^-2 k^-1').values * \
		0.5 * (
			v.to_units('m s^-1').values[:, :-1, :] + v.to_units('m s^-1').values[:, 1:, :]
		)
	return_dict['y_momentum_isentropic'] = taz.make_data_array_3d(
		sv_raw, grid, 'kg m^-1 k^-1 s^-1', name='y_momentum_isentropic'
	).to_units(sv_units)

	# physical constants
	pcs  = get_physical_constants(default_physical_constants)
	Rd   = pcs['gas_constant_of_dry_air']
	g    = pcs['gravitational_acceleration']
	pref = pcs['reference_air_pressure']
	cp   = pcs['specific_heat_of_dry_air_at_constant_pressure']

	# air pressure
	p = np.zeros((nx, ny, nz+1), dtype=dtype)
	p[:, :, 0] = 20
	for k in range(0, nz):
		p[:, :, k+1] = p[:, :, k] + g * dz * s.to_units('kg m^-2 K^-1').values[:, :, k]
	return_dict['air_pressure_on_interface_levels'] = taz.make_data_array_3d(
		p, grid, 'Pa', name='air_pressure_on_interface_levels'
	)

	# exner function
	exn = cp * (p / pref) ** (Rd / cp)
	return_dict['exner_function_on_interface_levels'] = taz.make_data_array_3d(
		exn, grid, 'J kg^-1 K^-1', name='exner_function_on_interface_levels'
	)

	# montgomery potential
	mtg = np.zeros((nx, ny, nz), dtype=dtype)
	mtg_s = grid.z_on_interface_levels.to_units('K').values[-1] * exn[:, :, -1] \
		+ g * grid.topography_height
	mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
	for k in range(nz-1, 0, -1):
		mtg[:, :, k-1] = mtg[:, :, k] + dz * exn[:, :, k]
	return_dict['montgomery_potential'] = taz.make_data_array_3d(
		mtg, grid, 'K kg^-1', name='montgomery_potential'
	)

	# height
	theta1d = grid.z_on_interface_levels.to_units('K').values()
	theta = np.tile(theta1d[np.newaxis, np.newaxis, :], (nx, ny, 1))
	h = np.zeros((nx, ny, nz+1), dtype=dtype)
	h[:, :, -1] = grid.topography_height
	for k in range(nz, 0, -1):
		h[:, :, k-1] = h[:, :, k] - \
			Rd * (theta[:, :, k-1] * exn[:, :, k-1] + theta[:, :, k] * exn[:, :, k]) * \
			(p[:, :, k-1] - p[:, :, k]) / (cp * g * (p[:, :, k-1] + p[:, :, k]))
	return_dict['height_on_interface_levels'] = taz.make_data_array_3d(
		h, grid, 'm', name='height_on_interface_levels'
	)

	if moist:
		# air density
		rho = s.to_units('kg m^-2 K^-1').values * \
			(theta[:, :, :-1] - theta[:, :, 1:]) / (h[:, :, :-1] - h[:, :, 1:])
		return_dict['air_density'] = taz.make_data_array_3d(
			rho, grid, 'kg m^-3', name='air_density'
		)

		# air temperature
		temp = 0.5 * (theta[:, :, 1:] * exn[:, :, 1:] +
			theta[:, :, -1:] * exn[:, :, -1:]) / cp
		return_dict['air_temperature'] = taz.make_data_array_3d(
			temp, grid, 'K', name='air_temperature'
		)

		# mass fraction of water vapor
		return_dict['mass_fraction_of_water_vapor_in_air'] = \
			draw(
				st_field(
					grid, 'isentropic_state',
					'mass_fraction_of_water_vapor_in_air', (nx, ny, nz)
				)
			)

		# mass fraction of cloud liquid water
		return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
			draw(
				st_field(
					grid, 'isentropic_state',
					'mass_fraction_of_cloud_liquid_water_in_air', (nx, ny, nz)
				)
			)

		# mass fraction of precipitation water
		return_dict['mass_fraction_of_precipitation_water_in_air'] = \
			draw(
				st_field(
					grid, 'isentropic_state',
					'mass_fraction_of_precipitation_water_in_air', (nx, ny, nz)
				)
			)

		if precipitation:
			# precipitation
			return_dict['precipitation'] = \
				draw(
					st_field(
						grid, 'isentropic_state', 'precipitation', (nx, ny, 1)
					)
				)

			# accumulated precipitation
			return_dict['accumulated_precipitation'] = \
				draw(
					st_field(
						grid, 'isentropic_state', 'accumulated_precipitation', (nx, ny, 1)
					)
				)

	return return_dict


@hyp_st.composite
def st_burgers_state(draw, grid, *, time=None):
	"""
	Strategy drawing a valid Burgers model state over `grid`.
	"""
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	assert nz == 1

	return_dict = {}

	# time
	if time is None:
		time = draw(hyp_st.datetimes())
	return_dict['time'] = time

	# x-velocity
	return_dict['x_velocity'] = draw(
		st_field(grid, 'burgers_state', 'x_velocity', (nx, ny, nz))
	)

	# y-velocity
	return_dict['y_velocity'] = draw(
		st_field(grid, 'burgers_state', 'y_velocity', (nx, ny, nz))
	)

	return return_dict


@hyp_st.composite
def st_burgers_tendency(draw, grid, *, time=None):
	"""
	Strategy drawing a set of tendencies for the variables whose evolution is
	governed by the Burgers equations.
	"""
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	assert nz == 1

	return_dict = {}

	# time
	if time is None:
		time = draw(hyp_st.datetimes())
	return_dict['time'] = time

	# x-velocity
	return_dict['x_velocity'] = draw(
		st_field(grid, 'burgers_tendency', 'x_velocity', (nx, ny, nz))
	)

	# y-velocity
	return_dict['y_velocity'] = draw(
		st_field(grid, 'burgers_tendency', 'y_velocity', (nx, ny, nz))
	)

	return return_dict
