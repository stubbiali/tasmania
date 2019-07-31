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
	get_constant
	get_physical_constants
	make_state
	make_dataarray_2d
	make_dataarray_3d
	make_raw_state
"""
import numpy as np
from sympl import DataArray


def get_constant(name, units, default_value=None):
	"""
	Get the value of a physical constant in the desired units.
	The function first looks for the constant in :mod:`tasmania.namelist`.
	If not found, it then searches in :obj:`sympl._core.constants.constants`.
	If still not found, the function reverts to :obj:`default_value`, which is
	added to :obj:`sympl._core.constants.constants` before returning.

	Parameters
	----------
	name : str
		Name of the physical constant.
	units : str
		Units in which the constant should be expressed.
	default_value : `dataarray_like`, optional
		1-item :class:`sympl.DataArray` representing the default value for the
		physical constant.

	Return
	------
	float :
		Value of the physical constant.

	Raises
	------
	ValueError :
		If the constant cannot be expressed in the desired units.
	ConstantNotFoundError :
		If the constant cannot be found.
	"""
	try:
		exec('from tasmania.namelist import {} as var'.format(name))
		return locals()['var'].to_units(units).values.item()
	except (ImportError, AttributeError):
		try:
			from sympl import get_constant as sympl_get_constant
			return sympl_get_constant(name, units)
		except KeyError:
			if default_value is not None:
				return_value = default_value.to_units(units).values.item()
				from sympl import set_constant as sympl_set_constant
				sympl_set_constant(name, return_value, units)
				return return_value
			else:
				from tasmania.python.utils.exceptions import ConstantNotFoundError
				raise ConstantNotFoundError('{} not found'.format(name))


def get_physical_constants(default_physical_constants, physical_constants=None):
	"""
	Parameters
	----------
	default_physical_constants : dict
		Dictionary whose keys are names of some physical constants,
		and whose values are :class:`sympl.DataArray`\s storing the
		default values and units of those constants.
	physical_constants : `dict`, optional
		Dictionary whose keys are names of some physical constants,
		and whose values are :class:`sympl.DataArray`\s storing the
		values and units of those constants.

	Return
	------
	dict :
		Dictionary whose keys are the names of the physical constants
		contained in :obj:`default_physical_constants`, and whose values
		are the values of those constants in the default units.
		The function first looks for the value of each constant in
		:obj:`physical_constants`. If this is not given, or it does not
		contain that constant, the value is retrieved via
		:func:`tasmania.utils.data_utils.get_constant`, using the corresponding
		value of :obj:`default_physical_constants` as default.
	"""
	raw_physical_constants = {}

	for name, d_const in default_physical_constants.items():
		d_units = d_const.attrs['units']
		const = physical_constants.get(name, None) if physical_constants is not None else None
		raw_physical_constants[name] = \
			const.to_units(d_units).values.item() if const is not None else \
			get_constant(name, d_units, default_value=d_const)

	return raw_physical_constants


def make_dataarray_2d(raw_array, grid, units, name=None):
	"""
	Create a :class:`sympl.DataArray` out of a 2-D :class:`numpy.ndarray`.

	Parameters
	----------
	raw_array : numpy.ndarray
		2-D :class:`numpy.ndarray` storing the field values.
	grid : tasmania.HorizontalGrid
		The underlying horizontal grid.
	units : str
		The variable units.
	name : `str`, optional
		The variable name. Defaults to :obj:`None`.

	Return
	------
	sympl.DataArray :
		The :class:`sympl.DataArray` whose value array is :obj:`raw_array`,
		whose coordinates and dimensions are retrieved from :obj:`grid`,
		and whose units are :obj:`units`.
	"""
	nx, ny = grid.nx, grid.ny
	try:
		ni, nj = raw_array.shape
	except ValueError:
		raise ValueError(
			'Expected a 2-D array, got a {}-D one.'.format(len(raw_array.shape))
		)

	if ni == nx:
		x = grid.x
	elif ni == nx+1:
		x = grid.x_at_u_locations
	else:
		raise ValueError(
			'The array extent in the x-direction is {} but either '
			'{} or {} was expected.'.format(ni, nx, nx+1)
		)

	if nj == ny:
		y = grid.y
	elif nj == ny+1:
		y = grid.y_at_v_locations
	else:
		raise ValueError(
			'The array extent in the y-direction is {} but either '
			'{} or {} was expected.'.format(nj, ny, ny+1)
		)

	return DataArray(
		raw_array,
		coords=(x.coords[x.dims[0]].values, y.coords[y.dims[0]].values),
		dims=(x.dims[0], y.dims[0]), name=name, attrs={'units': units}
	)


def make_dataarray_3d(raw_array, grid, units, name=None):
	"""
	Create a :class:`sympl.DataArray` out of a 3-D :class:`numpy.ndarray`.

	Parameters
	----------
	raw_array : numpy.ndarray
		3-D :class:`numpy.ndarray` storing the field values.
	grid : tasmania.Grid
		The underlying grid.
	units : str
		The variable units.
	name : `str`, optional
		The variable name. Defaults to :obj:`None`.

	Return
	------
	dataarray_like :
		The :class:`sympl.DataArray` whose value array is :obj:`raw_array`,
		whose coordinates and dimensions are retrieved from :obj:`grid`,
		and whose units are :obj:`units`.
	"""
	nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
	try:
		ni, nj, nk = raw_array.shape
	except ValueError:
		raise ValueError(
			'Expected a 3-D array, got a {}-D one.'.format(len(raw_array.shape))
		)

	if ni == 1 and nx != 1:
		x = DataArray(
			np.array([grid.grid_xy.x.values[0]]),
			dims=grid.grid_xy.x.dims[0] + '_gp',
			attrs={'units': grid.grid_xy.x.attrs['units']}
		)
	elif ni == nx:
		x = grid.grid_xy.x
	elif ni == nx+1:
		x = grid.grid_xy.x_at_u_locations
	else:
		raise ValueError(
			'The array extent in the x-direction is {} but either '
			'{}, {} or {} was expected.'.format(ni, 1, nx, nx+1)
		)

	if nj == 1 and ny != 1:
		y = DataArray(
			np.array([grid.grid_xy.y.values[0]]),
			dims=grid.grid_xy.y.dims[0] + '_gp',
			attrs={'units': grid.grid_xy.y.attrs['units']}
		)
	elif nj == ny:
		y = grid.grid_xy.y
	elif nj == ny+1:
		y = grid.grid_xy.y_at_v_locations
	else:
		raise ValueError(
			'The array extent in the y-direction is {} but either '
			'{}, {} or {} was expected.'.format(nj, 1, ny, ny+1)
		)

	if nk == 1:
		if nz > 1:
			z = DataArray(
				np.array([grid.z_on_interface_levels.values[-1]]),
				dims=grid.z.dims[0] + '_at_surface_level',
				attrs={'units': grid.z.attrs['units']}
			)
		else:
			z = DataArray(
				np.array([grid.z.values[-1]]),
			  	dims=grid.z.dims[0],
				attrs={'units': grid.z.attrs['units']}
			)
	elif nk == nz:
		z = grid.z
	elif nk == nz+1:
		z = grid.z_on_interface_levels
	else:
		raise ValueError(
			'The array extent in the z-direction is {} but either '
			'1, {} or {} was expected.'.format(nk, nz, nz+1)
		)

	return DataArray(
		raw_array,
		coords=[
			x.coords[x.dims[0]].values,
			y.coords[y.dims[0]].values,
			z.coords[z.dims[0]].values,
		],
		dims=[x.dims[0], y.dims[0], z.dims[0]],
		name=name, attrs={'units': units}
	)


def make_state(raw_state, grid, units):
	"""
	Parameters
	----------
	raw_state : dict
		Dictionary whose keys are strings indicating the variables
		included in the model state, and values are :class:`numpy.ndarray`\s
		containing the data for those variables.
	grid : tasmania.Grid
		The underlying grid.
	units : dict
		Dictionary whose keys are strings indicating the variables
		included in the model state, and values are strings indicating
		the units in which those variables should be expressed.

	Return
	------
	dict :
		Dictionary whose keys are strings indicating the variables
		included in the model state, and values are :class:`sympl.DataArray`\s
		containing the data for those variables.
	"""
	try:
		state = {'time': raw_state['time']}
	except KeyError:
		state = {}

	for key in raw_state.keys():
		if key != 'time':
			try:
				state[key] = make_dataarray_3d(
					raw_state[key], grid, units[key], name=key
				)
			except ValueError:
				state[key] = make_dataarray_2d(
					raw_state[key], grid.grid_xy, units[key], name=key
				)

	return state


def make_raw_state(state, units=None):
	"""
	Parameters
	----------
	state : dict
		Dictionary whose keys are strings indicating the variables
		included in the model state, and values are :class:`sympl.DataArray`\s
		containing the data for those variables.
	units : `dict`, optional
		Dictionary whose keys are strings indicating the variables
		included in the model state, and values are strings indicating
		the units in which those variables should be expressed.

	Return
	------
	dict :
		Dictionary whose keys are strings indicating the variables
		included in the model state, and values are :class:`numpy.ndarray`\s
		containing the data for those variables.
	"""
	units = {} if units is None else units

	try:
		raw_state = {'time': state['time']}
	except KeyError:
		raw_state = {}

	for key in state.keys():
		if key != 'time':
			try:
				dataarray = state[key].to_units(units.get(key, state[key].attrs['units']))
			except KeyError:
				raise KeyError('Units not specified for {}.'.format(key))

			raw_state[key] = dataarray.values

	return raw_state
