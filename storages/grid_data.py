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
import copy
import numpy as np
import xarray as xr

from grids.axis import Axis
import utils.utils as utils

class GridData:
	"""
	Class storing and handling time-dependent variables defined on a grid. Ideally, this class should be used to 
	represent the state, or a sequence of states at different time levels, of a *generic* climate or meteorological model. 
	The model variables, in the shape of :class:`numpy.ndarray`\s, are passed to the constructor as keyword arguments. 
	After conversion to :class:`xarray.DataArray`\s, the variables are packed in a dictionary whose keys are the input keywords. 
	The class attribute :data:`units` lists, for any admissible keyword, the units in which the associated field should 
	be expressed. Any variable can be accessed in read-only mode via the accessor operator by specifying the corresponding 
	keyword. Other methods are provided to update the state, or to create a sequence of states (useful for animation purposes). 
	This class is designed to be as general as possible. Hence, it is not endowed with any method whose
	implementation depends on the variables actually stored by the class. This kind of methods might be provided by some 
	derived classes, each one representing the state of a *specific* model.

	Attributes
	----------
	grid : obj
		The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
	"""
	# Specify the units in which variables should be expressed
	units = {
		'isentropic_density'   : 'kg m-2 K-1'    ,
		'x_velocity'           : 'm s-1'         ,
		'y_velocity'           : 'm s-1'         ,
		'x_momentum_isentropic': 'kg m-1 s-1 K-1',
		'y_momentum_isentropic': 'kg m-1 s-1 K-1',
		'pressure'             : 'Pa'            ,
		'exner_function'       : 'm2 s-2 K-2'    ,
		'montgomery_potential' : 'm2 s-2'        ,
		'height'               : 'm'             ,
		'density'			   : 'kg m-3'		 ,
		'water_vapor'          : 'kg kg-1'       ,
		'cloud_water'          : 'kg kg-1'       ,
		'precipitation_water'  : 'kg kg-1'       ,
	}

	def __init__(self, time, grid, **kwargs):
		"""
		Constructor.

		Parameters
		----------
		time : obj
			:class:`datetime.datetime` representing the time instant at which the variables are defined.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		**kwargs : array_like
			:class:`numpy.ndarray` representing a gridded variable.
		"""
		self.grid = grid

		self._vars = dict()
		for key in kwargs:
			# Distinguish between horizontally staggered and unstaggered fields
			var = kwargs[key]
			x = grid.x if var.shape[0] == grid.nx else grid.x_half_levels
			y = grid.y if var.shape[1] == grid.ny else grid.y_half_levels

			# Properly treat the vertical axis, so that either two- and three-dimensional arrays can be stored
			# A notable example of a two-dimensional field is the accumulated precipitation
			if var.shape == 1:
				z = Axis(np.array([grid.z_half_levels[-1]]), grid.z.dims, attrs = grid.z.attrs)
			elif var.shape[2] == grid.nz:
				z = grid.z 
			elif var.shape[2] == grid.nz + 1:
				z = grid.z_half_levels

			_var = xr.DataArray(np.copy(var[:, :, :, np.newaxis]), 
							    coords = [x.values, y.values, z.values, [time]],
			 				    dims = [x.dims, y.dims, z.dims, 'time'],
								attrs = {'units': GridData.units[key]})
			self._vars[key] = _var

	def __getitem__(self, key):
		"""
		Get a deep copy of a gridded variable.

		Parameters
		----------
		key : str
			The key corresponding to the variable to access.

		Return
		------
		obj :
			Deep copy of the :class:`xarray.DataArray` representing the variable.
		"""
		return copy.deepcopy(self._vars.get(key, None))

	@property
	def time(self):
		"""
		Return the time at which the data are defined.

		Return
		------
		obj :
			:class:`datetime.datetime` representing the time at which the data are defined.
		"""
		akey = list(self._vars.keys())[0]
		return utils.convert_datetime64_to_datetime(self._vars[akey].coords['time'].values[-1])

	def update(self, other):
		"""
		Sync the current object with another :class:`~storages.grid_data.GridData` (or a derived class).
		This implies that, for each variable stored in the input object:

		* if the current object contains a variable with the same name, the field of that variable is updated;
		* if the current object does not contain any variable with the same name, the variable is added to the current object.

		Parameters
		----------
		other : obj 
			Another :class:`~storages.grid_data.GridData` (or a derived class) with which the current object will be synced.
		"""
		for key in other._vars:
			self._vars[key] = copy.deepcopy(other._vars[key])
	
	def append(self, other):
		"""
		Append a new state to the sequence of states.

		Parameters
		----------
		other : obj 
			Another :class:`~storages.grid_data.GridData` (or a derived class), whose :class:`xarray.DataArray`\s 
			will be concatenated along the temporal axis to the corresponding ones in the current object.

		Note
		----
		:data:`other` is supposed to contain exactly the same variables stored by the current object.
		"""
		for key in self._vars:
			self._vars[key] = xr.concat([self._vars[key], copy.deepcopy(other[key])], 'time')

	def get_max(self, key):
		"""
		Get the maximum value of a variable.

		Parameters
		----------
		key : str
			Key identifying the variable of interest.

		Return
		------
		float :
			The maximum value of the variable of interest.
		"""
		if key in GridData.units.keys():
			return np.amax(self._vars[key].values[:,:,:,-1])
		else:
			raise KeyError('The variable {} is not stored within the current object'.format(key))

	def get_min(self, key):
		"""
		Get the minimum value of a variable.

		Parameters
		----------
		key : str
			Key identifying the variable of interest.

		Return
		------
		float :
			The minimum value of the variable of interest.
		"""
		if key in GridData.units.keys():
			return np.amin(self._vars[key].values[:,:,:,-1])
		else:
			raise KeyError('The variable {} is not stored within the current object'.format(key))

