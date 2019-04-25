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
	IsentropicDiagnostics
	IsentropicVelocityComponents
"""
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.dwarfs.diagnostics import HorizontalVelocity
from tasmania.python.framework.base_components import DiagnosticComponent
from tasmania.python.isentropic.dynamics.diagnostics import \
	IsentropicDiagnostics as Core

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class IsentropicDiagnostics(DiagnosticComponent):
	"""
	With the help of the isentropic density, this class diagnosed

		* the pressure,
		* the Exner function,
		* the Montgomery potential and
		* the height of the interface levels.

	Optionally,

		* the air density and
		* the air temperature

	are calculated as well.
	"""
	# Default values for the physical constants used in the class
	_d_physical_constants = {
		'air_pressure_at_sea_level':
			DataArray(1e5, attrs={'units': 'Pa'}),
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gravitational_acceleration':
			DataArray(9.80665, attrs={'units': 'm s^-2'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
	}

	def __init__(
		self, domain, grid_type, moist, pt, backend=gt.mode.NUMPY,
		dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		moist : bool
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise.
		pt : sympl.DataArray
			One-item :class:`sympl.DataArray` representing the air pressure
			at the top edge of the domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gravitational_acceleration', in units compatible with [m s^-2];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.physics.isentropic.IsentropicDiagnostics._d_physical_constants`
			for the default values.
		"""
		# Keep track of some input parameters
		self._moist = moist
		self._pt = pt.to_units('Pa').values.item()

		# Call parent's constructor
		super().__init__(domain, grid_type)

		# Instantiate the class computing the diagnostic variables
		self._core = Core(self.grid, backend, dtype, physical_constants)

		# Allocate the output arrays
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._out_p   = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._out_exn = np.zeros((nx, ny, nz+1), dtype=dtype)
		self._out_mtg = np.zeros((nx, ny, nz  ), dtype=dtype)
		self._out_h   = np.zeros((nx, ny, nz+1), dtype=dtype)
		if moist:
			self._out_r = np.zeros((nx, ny, nz), dtype=dtype)
			self._out_t = np.zeros((nx, ny, nz), dtype=dtype)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'air_isentropic_density':
				{'dims': dims, 'units': 'kg m^-2 K^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
		dims_stgz = (dims[0], dims[1], self.grid.z_on_interface_levels.dims[0])

		return_dict = {
			'air_pressure_on_interface_levels':
				{'dims': dims_stgz, 'units': 'Pa'},
			'exner_function_on_interface_levels':
				{'dims': dims_stgz, 'units': 'J K^-1 kg^-1'},
			'height_on_interface_levels':
				{'dims': dims_stgz, 'units': 'm'},
			'montgomery_potential':
				{'dims': dims, 'units': 'm^2 s^-2'},
		}

		if self._moist:
			return_dict['air_density'] = {'dims': dims, 'units': 'kg m^-3'}
			return_dict['air_temperature'] = {'dims': dims, 'units': 'K'}

		return return_dict

	def array_call(self, state):
		# Extract useful variables from input state dictionary
		s  = state['air_isentropic_density']

		# Calculate all diagnostics, and initialize the output dictionary
		self._core.get_diagnostic_variables(
			s, self._pt, self._out_p, self._out_exn, self._out_mtg, self._out_h
		)
		diagnostics = {
			'air_pressure_on_interface_levels': self._out_p,
			'exner_function_on_interface_levels': self._out_exn,
			'montgomery_potential': self._out_mtg,
			'height_on_interface_levels': self._out_h,
		}
		if self._moist:
			self._core.get_air_density(s, self._out_h, self._out_r)
			diagnostics['air_density'] = self._out_r
			self._core.get_air_temperature(self._out_exn, self._out_t)
			diagnostics['air_temperature'] = self._out_t

		return diagnostics


class IsentropicVelocityComponents(DiagnosticComponent):
	"""
	Retrieve the horizontal velocity components with the help of the
	isentropic momenta and the isentropic density.
	The class is instantiated over the *numerical* grid of the
	underlying domain.
	"""
	def __init__(self, domain, backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		"""
		super().__init__(domain, 'numerical')

		self._core = HorizontalVelocity(
			self.grid, staggering=True, backend=backend, dtype=dtype
		)

		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		self._out_u = np.zeros((nx+1, ny, nz), dtype=dtype)
		self._out_v = np.zeros((nx, ny+1, nz), dtype=dtype)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])
		dims_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

		return_dict = {
			'x_velocity_at_u_locations': {'dims': dims_x, 'units': 'm s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_y, 'units': 'm s^-1'},
		}

		return return_dict

	def array_call(self, state):
		# Extract the required model variables from the input state
		s  = state['air_isentropic_density']
		su = state['x_momentum_isentropic']
		sv = state['y_momentum_isentropic']

		# Diagnose the velocity components
		self._core.get_velocity_components(s, su, sv, self._out_u, self._out_v)

		# Enforce the boundary conditions
		hb = self.horizontal_boundary
		hb.dmn_set_outermost_layers_x(
			self._out_u, field_name='x_velocity_at_u_locations',
			field_units='m s^-1', time=state['time']
		)
		hb.dmn_set_outermost_layers_y(
			self._out_v, field_name='y_velocity_at_v_locations',
			field_units='m s^-1', time=state['time']
		)

		# Instantiate the output dictionary
		diagnostics = {
			'x_velocity_at_u_locations': self._out_u,
			'y_velocity_at_v_locations': self._out_v,
		}

		return diagnostics
