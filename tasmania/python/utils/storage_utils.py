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
	NetCDFMonitor
	load_netcdf_dataset
	_load_grid
	_load_states
"""
from copy import deepcopy
from datetime import timedelta
import netCDF4 as nc4
import numpy as np
import sympl
import xarray as xr

from tasmania.python.grids.grid_xyz import GridXYZ
from tasmania.python.utils.utils import convert_datetime64_to_datetime


class NetCDFMonitor(sympl.NetCDFMonitor):
	"""
	Customized version of :class:`sympl.NetCDFMonitor`, which
	caches stored states and then write them to a NetCDF file,
	together with some grid properties.
	"""
	def __init__(
		self, filename, grid, time_units='seconds',
		store_names=None, write_on_store=False, aliases=None
	):
		"""
		The constructor.

		Parameters
		----------
		filename : str
			The file to which the NetCDF file will be written.
		grid : grid
			Instance of :class:`~tasmania.dynamics.grids.grid_xyz.GridXYZ`
			representing the underlying computational grid.
		time_units : str, optional
			The units in which time will be
			stored in the NetCDF file. Time is stored as an integer
			number of these units. Default is seconds.
		store_names : iterable of str, optional
			Names of quantities to store. If not given,
			all quantities are stored.
		write_on_store : bool, optional
			If True, stored changes are immediately written to file.
			This can result in many file open/close operations.
			Default is to write only when the write() method is
			called directly.
		aliases : dict
			A dictionary of string replacements to apply to state variable
			names before saving them in netCDF files.
		"""
		super().__init__(
			filename, time_units, store_names, write_on_store, aliases
		)
		self._grid = grid

	def store(self, state):
		"""
		Make a deep copy of the input state before calling the parent's method.
		"""
		state_dc = deepcopy(state)
		super().store(state_dc)

	def write(self):
		"""
		Write grid properties and all cached states to the NetCDF file,
		and clear the cache. This will append to any existing NetCDF file.
		"""
		super().write()

		with nc4.Dataset(self._filename, 'a') as dataset:
			g = self._grid

			dataset.createDimension('bool_dim', 1)
			dataset.createDimension('scalar_dim', 1)
			dataset.createDimension('str_dim', 1)

			# List of model state variable names
			names = [var for var in dataset.variables if var != 'time']
			dataset.createDimension('strvec1_dim', len(names))
			state_variable_names = dataset.createVariable(
				'state_variable_names', str, ('strvec1_dim',)
			)
			state_variable_names[:] = np.array(names, dtype='object')

			# x-axis
			dim1_name = dataset.createVariable('dim1_name', str, ('str_dim',))
			dim1_name[:] = np.array([g.x.dims[0]], dtype='object')
			dim1 = dataset.createVariable(g.x.dims[0], g.x.values.dtype, (g.x.dims[0],))
			dim1[:] = g.x.values[:]
			dim1.setncattr('units', g.x.attrs['units'])
			try:
				dim1_u = dataset.createVariable(
						g.x_at_u_locations.dims[0], 
						g.x_at_u_locations.values.dtype,
						(g.x_at_u_locations.dims[0],)
				)
				dim1_u[:] = g.x_at_u_locations.values[:]
				dim1_u.setncattr('units', g.x_at_u_locations.attrs['units'])
			except ValueError:
				pass

			# y-axis
			dim2_name = dataset.createVariable('dim2_name', str, ('str_dim',))
			dim2_name[:] = np.array([g.y.dims[0]], dtype='object')
			dim2 = dataset.createVariable(g.y.dims[0], g.y.values.dtype, (g.y.dims[0],))
			dim2[:] = g.y.values[:]
			dim2.setncattr('units', g.y.attrs['units'])
			try:
				dim2_v = dataset.createVariable(
						g.y_at_v_locations.dims[0],
						g.y_at_v_locations.values.dtype,
						(g.y_at_v_locations.dims[0],)
				)
				dim2_v[:] = g.y_at_v_locations.values[:]
				dim2_v.setncattr('units', g.y_at_v_locations.attrs['units'])
			except ValueError:
				pass

			# z-axis
			dim3_name = dataset.createVariable('dim3_name', str, ('str_dim',))
			dim3_name[:] = np.array([g.z.dims[0]], dtype='object')
			dim3 = dataset.createVariable(g.z.dims[0], g.z.values.dtype, (g.z.dims[0],))
			dim3[:] = g.z.values[:]
			dim3.setncattr('units', g.z.attrs['units'])
			try:
				dim3_hl = dataset.createVariable(
						g.z_on_interface_levels.dims[0],
						g.z_on_interface_levels.values.dtype,
						(g.z_on_interface_levels.dims[0],)
				)
				dim3_hl[:] = g.z_on_interface_levels.values[:]
				dim3_hl.setncattr('units', g.z_on_interface_levels.attrs['units'])
			except ValueError:
				pass

			# Interface level
			z_interface = dataset.createVariable(
				'z_interface', g.z_interface.values.dtype, ('scalar_dim',)
			)
			z_interface[:] = g.z_interface.values.item()
			z_interface.setncattr('units', g.z_interface.attrs['units'])

			# Topography type
			topo         = self._grid.topography
			topo_type    = dataset.createVariable('topo_type', str, ('str_dim',))
			topo_type[:] = np.array([topo.topo_type], dtype='object')

			# Characteristic time scale of the topography
			topo_time    = dataset.createVariable('topo_time', float, ('scalar_dim',))
			topo_time[:] = np.array([topo.topo_time.total_seconds()], dtype=float)
			topo_time.setncattr('units', 's')

			# Topography properties
			keys = []
			for key, value in topo.topo_kwargs.items():
				if isinstance(value, sympl.DataArray):
					var    = dataset.createVariable(key, value.values.dtype, ('scalar_dim',))
					var[:] = value.values.item()
					var.setncattr('units', value.attrs['units'])
				elif isinstance(value, str):
					var    = dataset.createVariable(key, str, ('str_dim',))
					var[:] = np.array([value], dtype='object')
				elif isinstance(value, bool):
					var    = dataset.createVariable(key, int, ('bool_dim',))
					var[:] = np.array([1 if value else 0], dtype=bool)

				keys.append(key)

			# List of keyword arguments to pass to the topography
			dataset.createDimension('strvec2_dim', len(keys))
			topo_kwargs    = dataset.createVariable('topo_kwargs', str, ('strvec2_dim',))
			topo_kwargs[:] = np.array(keys, dtype='object')


def load_netcdf_dataset(filename):
	"""
	Load the sequence of states stored in a NetCDF dataset,
	and build the underlying grid.

	Parameters
	----------
	filename : str
		Path to the NetCDF dataset.

	Returns
	-------
	grid : grid
		Instance of :class:`~tasmania.dynamics.grids.grid_xyz.GridXYZ`
		representing the underlying computational grid.
	states : list of dict
		List of state dictionaries stored in the NetCDF file.
	"""
	with xr.open_dataset(filename) as dataset:
		return _load_grid(dataset), _load_states(dataset)


def _load_grid(dataset):
	dims_x = dataset.data_vars['dim1_name'].values.item()
	x = dataset.coords[dims_x]
	domain_x = sympl.DataArray(
		[x.values[0], x.values[-1]],
		dims=[dims_x], attrs={'units': x.attrs['units']}
	)
	nx = x.shape[0]

	dims_y = dataset.data_vars['dim2_name'].values.item()
	y = dataset.coords[dims_y]
	domain_y = sympl.DataArray(
		[y.values[0], y.values[-1]],
		dims=[dims_y], attrs={'units': y.attrs['units']}
	)
	ny = y.shape[0]

	dims_z = dataset.data_vars['dim3_name'].values.item()
	z_hl = dataset.coords[dims_z + '_on_interface_levels']
	domain_z = sympl.DataArray(
		[z_hl.values[0], z_hl.values[-1]],
		dims=[dims_z], attrs={'units': z_hl.attrs['units']}
	)
	nz = z_hl.shape[0]-1

	z_interface = sympl.DataArray(dataset.data_vars['z_interface'])

	topo_type = dataset.data_vars['topo_type'].values.item()

	topo_time = timedelta(seconds=dataset.data_vars['topo_time'].values.item())

	keys = dataset.data_vars['topo_kwargs'].values[:]
	topo_kwargs = {}
	for key in keys:
		val = dataset.data_vars[key]
		if isinstance(val.values.item(), (str, bool)):
			topo_kwargs[key] = val.values.item()
		elif isinstance(val.values.item(), int):
			topo_kwargs[key] = bool(val.values.item())
		else:
			topo_kwargs[key] = sympl.DataArray(val, attrs={'units': val.attrs['units']})

	return GridXYZ(
		domain_x, nx, domain_y, ny, domain_z, nz, z_interface,
		topo_type, topo_time, topo_kwargs, dtype=domain_z.values.dtype
	)


def _load_states(dataset):
	names = dataset.data_vars['state_variable_names'].values
	nt = dataset.data_vars[names[0]].shape[0]

	states = []
	for n in range(nt):
		state = {'time': convert_datetime64_to_datetime(dataset['time'][n])}
		for name in names:
			state[name] = sympl.DataArray(dataset.data_vars[name][n, :, :, :])
		states.append(state)

	return states
