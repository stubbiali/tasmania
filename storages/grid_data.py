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

import utils

class GridData:
	"""
	Class storing and handling time-dependent variables defined on a grid. Ideally, this class should be used to 
	represent the state, or a sequence of states, of a *generic* climate or meteorological model. The model variables, 
	in the shape of :class:`numpy.ndarray`s, are passed to the constructor as keyword arguments. After conversion to 
	:class:`xarray.DataArray`s, the variables are packed in a dictionary whose keys are the input keywords. The 
	class attribute :data:`units` lists, for any admissible keyword, the units in which the associated field should 
	be expressed. Any variable can be accessed in read-only mode via the accessor operator by specifying the 
	corresponding keyword. Other methods are provided to update the state, or to create a sequence of states 
	(useful for animation purposes). 
	This class is designed be as general as possible. Hence, it is not endowed with any method whose
	implementation depends on the variables actually stored by the class. This kind of methods will be 
	provided by the derived classes, each one representing the state of a *specific* model.
	"""
	# Specify the units in which variables should be expressed
	units = {
		'isentropic_density'  : 'kg m-2 K-1',
		'x_velocity'          : 'm s-1'     ,
		'y_velocity'          : 'm s-1'     ,
		'pressure'            : 'Pa'        ,
		'exner_function'      : 'm2 s-2 K-2',
		'montgomery_potential': 'm2 s-2'    ,
		'height'              : 'm'         ,
		'water_vapour'        : 'kg kg-1'   ,
		'cloud_water'         : 'kg kg-1'   ,
		'precipitation_water' : 'kg kg-1'   ,
	}

	def __init__(self, time, grid, **kwargs):
		"""
		Constructor.
		"""
		self._grid = grid

		self._vars = dict()
		for key in kwargs:
			# Distinguish between staggered and unstaggered fields
			var = kwargs[key]
			x = grid.x if var.shape[0] == grid.nx else grid.x_half_levels
			y = grid.y if var.shape[1] == grid.ny else grid.y_half_levels
			z = grid.z if var.shape[2] == grid.nz else grid.z_half_levels

			_var = xr.DataArray(np.copy(var[:, :, :, np.newaxis]), 
							    coords = [x.values, y.values, z.values, [time]],
			 				    dims = [x.dims, y.dims, z.dims, 'time'],
								attrs = {'units': GridData.units[key]})
			self._vars[key] = _var

	def __getitem__(self, key):
		"""
		Access a gridded variable in read-only mode.
		"""
		return self._vars[key]

	@property
	def time(self):
		"""
		Return the time at which the state corresponds.

		Return:
			A :class:`datetime.datetime` object representing the current time.
		"""
		akey = list(self._vars.keys())[0]
		return utils.convert_datetime64_to_datetime(self._vars[akey].coords['time'].values[-1])

	def update(self, state_new):
		"""
		Update (some of) the gridded variables. 

		Args:
			state_new (obj): A :class:`~storages.grid_data.GridData` carrying the updated model variables.

		Note:
			:data:`state_new` is not required to carry *all* the model variables.
		"""
		for key in state_new._vars:
			self._vars[key] = copy.deepcopy(state_new._vars[key])
	
	def append(self, state_new):
		"""
		Append a new state to the sequence of states.

		Args:
			state_new (obj): The new :class:`~storages.grid_data.GridData` to append.
		"""
		for key in self._vars:
			self._vars[key] = xr.concat([self._vars[key], copy.deepcopy(state_new[key])], 'time')

	def get_max(self, field):
		"""
		Get maximum value of a field.
		"""
		if field in GridData.units.keys():
			return np.amax(self._vars[field].values[:,:,:,-1])

	def get_min(self, field):
		"""
		Get minimum value of a field.
		"""
		if field in GridData.units.keys():
			return np.amin(self._vars[field].values[:,:,:,-1])

