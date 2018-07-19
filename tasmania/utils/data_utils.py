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
from sympl import DataArray

from tasmania.grids.grid_xy import GridXY
from tasmania.grids.grid_xz import GridXZ


def get_numpy_arrays(state, indices, *args):
	"""
	Given a dictionary of :class:`xarray.DataArray`\s and a set of keys,
	extract the corresponding :class:`numpy.ndarray`\s.

	Parameters
	----------
	state : dict
		A dictionary of :class:`xarray.DataArray`\s.
	indices : tuple
		Tuple of indices or slices identifying the portion of each
		:class:`xarray.DataArray` to be extracted.
	*args : `str` or `tuple of str`
		Each optional positional argument may be either a strings,
		specifying the variable to extract, or a tuple of aliases for the variable.

	Return
	------
	array_like or list:
		The desired :class:`numpy.ndarray`, or a list collecting the desired
		:class:`numpy.ndarray`\s.

	Raises
	------
	KeyError :
		If a variable which should be extracted is not included in the input dictionary.
	"""
	raw_state = []

	for key in args:
		if type(key) is str: 				# key represents the name of the variable
			if key not in state.keys():
				raise KeyError('Variable {} not included in the input dictionary.'
							   .format(key))
			raw_state.append(state[key].values[indices])
		elif type(key) in [tuple, list]: 	# key represents a set of aliases
			for i, alias in enumerate(key):
				if state.get(alias) is not None:
					raw_state.append(state[alias].values[indices])
					break
				elif i == len(key)-1:
					raise KeyError('None of the aliases {} is included in the '
								   'input dictionary.'.format(key))

	return raw_state if len(raw_state) > 1 else raw_state[0]


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
				from tasmania.utils.exceptions import ConstantNotFoundError
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
	raw_array : array_like
		2-D :class:`numpy.ndarray` storing the variable data.
	grid : grid
        The underlying grid, as an instance of
        :class:`~tasmania.grids.grid_xy.GridXY`,
        :class:`~tasmania.grids.grid_xz.GridXZ`,
        or one of their derived classes.
    units : str
    	String indicating the variable units.
    name : `str`, optional
    	String indicating the variable name. Defaults to :obj:`None`.

    Return
    ------
    dataarray_like :
    	The :class:`sympl.DataArray` whose value array is :obj:`raw_array`,
    	whose coordinates and dimensions are retrieved from :obj:`grid`,
    	and whose units are :obj:`units`.

    Raises
    ------
    ValueError:
    	If :obj:`raw_array` is not 2-D.
    TypeError :
    	If :obj:`grid` is not an instance of
    	:class:`~tasmania.grids.grid_xy.GridXY`,
    	:class:`~tasmania.grids.grid_xz.GridXZ`,
    	nor one of their derived classes.
	"""
	if len(raw_array.shape) != 2:
		raise ValueError('raw_array should be 2-D.')

	if isinstance(grid, GridXY):
		return _make_dataarray_xy(raw_array, grid, units, name)
	elif isinstance(grid, GridXZ):
		return _make_dataarray_xz(raw_array, grid, units, name)
	else:
		raise TypeError('grid should be an instance of GridXY, GridXZ, '
						'or one of their derived classes.')


def make_dataarray_3d(raw_array, grid, units, name=None):
	"""
	Create a :class:`sympl.DataArray` out of a 3-D :class:`numpy.ndarray`.

	Parameters
	----------
	raw_array : array_like
		3-D :class:`numpy.ndarray` storing the variable data.
	grid : grid
        The underlying grid, as an instance of
        :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
    units : str
    	String indicating the variable units.
    name : `str`, optional
    	String indicating the variable name. Defaults to :obj:`None`.

    Return
    ------
    dataarray_like :
    	The :class:`sympl.DataArray` whose value array is :obj:`raw_array`,
    	whose coordinates and dimensions are retrieved from :obj:`grid`,
    	and whose units are :obj:`units`.

    Raises
    ------
    ValueError:
    	If :obj:`raw_array` is not 2-D.
	"""
	if len(raw_array.shape) != 3:
		raise ValueError('raw_array should be 3-D.')
	return _make_dataarray_xyz(raw_array, grid, units, name)


def _make_dataarray_xy(raw_array, grid, units, name):
	nx, ny = grid.nx, grid.ny
	ni, nj = raw_array.shape

	if ni == nx:
		x = grid.x
	elif ni == nx+1:
		x = grid.x_at_u_locations
	else:
		raise ValueError('The array extent in the x-direction is {} but either '
						 '{} or {} was expected.'.format(ni, nx, nx+1))

	if nj == ny:
		y = grid.y
	elif nj == ny+1:
		y = grid.y_at_v_locations
	else:
		raise ValueError('The array extent in the y-direction is {} but either '
						 '{} or {} was expected.'.format(nj, ny, ny+1))

	return DataArray(raw_array,
					 coords=[x.coords[x.dims[0]].values,
							 y.coords[y.dims[0]].values],
					 dims=[x.dims[0], y.dims[0]],
					 name=name,
					 attrs={'units': units})


def _make_dataarray_xz(raw_array, grid, units, name):
	nx, nz = grid.nx, grid.nz
	ni, nk = raw_array.shape

	if ni == nx:
		x = grid.x
	elif ni == nx+1:
		x = grid.x_at_u_locations
	else:
		raise ValueError('The array extent in the x-direction is {} but either '
						 '{} or {} was expected.'.format(ni, nx, nx+1))

	if nk == nz:
		z = grid.z
	elif nk == nz+1:
		z = grid.z_on_interface_levels
	else:
		raise ValueError('The array extent in the z-direction is {} but either '
						 '{} or {} was expected.'.format(nk, nz, nz+1))

	return DataArray(raw_array,
					 coords=[x.coords[x.dims[0]].values,
							 z.coords[z.dims[0]].values],
					 dims=[x.dims[0], z.dims[0]],
					 name=name,
					 attrs={'units': units})


def _make_dataarray_xyz(raw_array, grid, units, name):
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	ni, nj, nk = raw_array.shape

	if ni == nx:
		x = grid.x
	elif ni == nx+1:
		x = grid.x_at_u_locations
	else:
		raise ValueError('The array extent in the x-direction is {} but either '
						 '{} or {} was expected.'.format(ni, nx, nx+1))

	if nj == ny:
		y = grid.y
	elif nj == ny+1:
		y = grid.y_at_v_locations
	else:
		raise ValueError('The array extent in the y-direction is {} but either '
						 '{} or {} was expected.'.format(nj, ny, ny+1))

	if nk == nz:
		z = grid.z
	elif nk == nz+1:
		z = grid.z_on_interface_levels
	else:
		raise ValueError('The array extent in the z-direction is {} but either '
						 '{} or {} was expected.'.format(nk, nz, nz+1))

	return DataArray(raw_array,
					 coords=[x.coords[x.dims[0]].values,
							 y.coords[y.dims[0]].values,
							 z.coords[z.dims[0]].values],
					 dims=[x.dims[0], y.dims[0], z.dims[0]],
					 name=name,
					 attrs={'units': units})

