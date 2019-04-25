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
"""
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.utils.data_utils import get_physical_constants

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


class IsentropicDiagnostics:
	"""
	Class implementing the diagnostic steps of the three-dimensional
	isentropic dynamical core using GT4Py stencils.
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
		self, grid, backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None
	):
		"""
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
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
				* 'gravitational acceleration', in units compatible with [m s^-2].
				* 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
					with [J K^-1 kg^-1].

			Please refer to
			:func:`tasmania.utils.data_utils.get_physical_constants` and
			:obj:`tasmania.IsentropicDiagnostics._d_physical_constants`
			for the default values.
		"""
		# Store the input arguments
		self._grid    = grid
		self._backend = backend
		self._dtype	  = dtype

		# Set physical constants values
		self._physical_constants = get_physical_constants(
			self._d_physical_constants, physical_constants
		)

		# Assign to each grid point on the interface levels
		# the corresponding z-quota; this is required to diagnose
		# the geometrical height at the interface levels
		theta_1d = grid.z_on_interface_levels.to_units('K').values[np.newaxis, np.newaxis, :]
		self._theta = np.tile(theta_1d, (grid.nx, grid.ny, 1))

		# Initialize the pointers to the underlying GT4Py stencils
		# These will be properly re-directed the first time the corresponding
		# entry-point method is invoked
		self._stencil_diagnosing_air_pressure = None
		self._stencil_diagnosing_montgomery = None
		self._stencil_diagnosing_height = None
		self._stencil_diagnosing_air_density = None
		self._stencil_diagnosing_air_temperature = None

	def get_diagnostic_variables(self, s, pt, p, exn, mtg, h):
		"""
		With the help of the isentropic density and the upper boundary
		condition on the pressure distribution, diagnose the pressure,
		the Exner function, the Montgomery potential, and the geometric
		height of the half-levels.

		Parameters
		----------
		s : numpy.ndarray
			The isentropic density, in units of [kg m^-2 K^-1].
		pt : float
			The upper boundary condition on the pressure distribution,
			in units of [Pa].
		p : numpy.ndarray
			The buffer for the pressure at the interface levels, in units of [Pa].
		exn : numpy.ndarray
			The buffer for the Exner function at the interface levels,
			in units of [J K^-1 kg^-1].
		mtg : numpy.ndarray
			The buffer for the Montgomery potential, in units of [J kg^-1].
		h : numpy.ndarray
			The buffer for the geometric height of the interface levels, in units of [m].
		"""
		# Instantiate the underlying stencils
		if self._stencil_diagnosing_air_pressure is None:
			self._stencil_diagnosing_air_pressure_initialize()
		if self._stencil_diagnosing_montgomery is None:
			self._stencil_diagnosing_montgomery_initialize()
		if self._stencil_diagnosing_height is None:
			self._stencil_diagnosing_height_initialize()

		# Shortcuts
		dz    = self._grid.dz.to_units('K').values.item()
		cp    = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
		p_ref = self._physical_constants['air_pressure_at_sea_level']
		rd    = self._physical_constants['gas_constant_of_dry_air']
		g	  = self._physical_constants['gravitational_acceleration']

		# Update the attributes which serve as inputs to the GT4Py stencils
		self._in_s[...] = s[...]

		# Apply upper boundary condition on pressure
		self._out_p[:, :, 0] = pt

		# Compute pressure at all other locations
		self._stencil_diagnosing_air_pressure.compute()

		# Compute the Exner function (not via a GT4Py stencils)
		self._in_exn[...] = cp * (self._out_p[...] / p_ref) ** (rd / cp)

		# Compute Montgomery potential at the lower main level
		mtg_s = self._theta[:, :, -1] * self._in_exn[:, :, -1] \
			+ g * self._grid.topography.profile.to_units('m').values
		self._out_mtg[:, :, -1] = mtg_s + 0.5 * dz * self._in_exn[:, :, -1]

		# Compute Montgomery potential at all other locations
		self._stencil_diagnosing_montgomery.compute()

		# Compute geometrical height of the isentropes
		self._out_h[:, :, -1] = self._grid.topography.profile.to_units('m').values[...]
		self._stencil_diagnosing_height.compute()

		# Write the output into the provided buffers
		p[...]   = self._out_p[...]
		exn[...] = self._in_exn[...]
		mtg[...] = self._out_mtg[...]
		h[...]   = self._out_h[...]

	def get_height(self, s, pt, h):
		"""
		With the help of the isentropic density and the upper boundary
		condition on the pressure distribution, diagnose the geometric
		height of the half-levels.

		Parameters
		----------
		s : numpy.ndarray
			The isentropic density, in units of [kg m^-2 K^-1].
		pt : float
			The upper boundary condition on the pressure distribution,
			in units of [Pa].
		h : numpy.ndarray
			The buffer for the geometric height of the interface levels, in units of [m].
		"""
		# Instantiate the underlying stencils
		if self._stencil_diagnosing_air_pressure is None:
			self._stencil_diagnosing_air_pressure_initialize()
		if self._stencil_diagnosing_height is None:
			self._stencil_diagnosing_height_initialize()

		# Get the values for the physical constants to use
		cp    = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
		p_ref = self._physical_constants['air_pressure_at_sea_level']
		rd    = self._physical_constants['gas_constant_of_dry_air']

		# Update the attributes which serve as inputs to the GT4Py stencils
		self._in_s[...] = s[...]

		# Apply upper boundary condition on pressure
		self._out_p[:, :, 0] = pt

		# Compute pressure at all other locations
		self._stencil_diagnosing_air_pressure.compute()

		# Compute the Exner function (not via a GT4Py stencils)
		self._in_exn[...] = cp * (self._out_p[...] / p_ref) ** (rd / cp)

		# Compute geometrical height of the isentropes
		self._out_h[:, :, -1] = self._grid.topography.profile.to_units('m').values[...]
		self._stencil_diagnosing_height.compute()

		# Write the output into the provided buffer
		h[...] = self._out_h[...]

	def get_air_density(self, s, h, rho):
		"""
		With the help of the isentropic density and the geometric height
		of the interface levels, diagnose the air density.

		Parameters
		----------
		s : numpy.ndarray
			The isentropic density, in units of [kg m^-2 K^-1].
		h : numpy.ndarray
			The geometric height of the interface levels, in units of [m].
		rho : numpy.ndarray
			The buffer for the air density, in units of [kg m^-3].
		"""
		# Instantiate the underlying stencil
		if self._stencil_diagnosing_air_density is None:
			self._stencil_diagnosing_air_density_initialize()

		# Update the attributes which serve as inputs to the stencil
		self._in_s[...] = s[...]
		self._in_h[...] = h[...]

		# Run the stencil's compute function
		self._stencil_diagnosing_air_density.compute()

		# Write the output into the provided buffer
		rho[...] = self._out_rho[...]

	def get_air_temperature(self, exn, temp):
		"""
		With the help of the Exner function, diagnose the air temperature.

		Parameters
		----------
		exn : numpy.ndarray
			The Exner function at the interface levels, in units of [J K^-1 kg^-1].
		temp : numpy.ndarray
			The buffer for the temperature, in units of [K].
		"""
		# Instantiate the underlying stencil
		if self._stencil_diagnosing_air_temperature is None:
			self._stencil_diagnosing_air_temperature_initialize()

		# Update the attributes which serve as inputs to the stencil
		self._in_exn[...] = exn[...]

		# Run the stencil's compute function
		self._stencil_diagnosing_air_temperature.compute()

		# Write the output into the provided buffer
		temp[...] = self._out_temp[...]

	def _stencil_diagnosing_air_pressure_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the pressure.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's inputs and outputs
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype=self._dtype)
		self._out_p = np.zeros((nx, ny, nz+1), dtype=self._dtype)
		self._in_p = self._out_p

		# Instantiate the stencil
		self._stencil_diagnosing_air_pressure = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_air_pressure_defs,
			inputs={'in_s': self._in_s, 'in_p': self._in_p},
			outputs={'out_p': self._out_p},
			domain=gt.domain.Rectangle((0, 0, 1), (nx-1, ny-1, nz)),
			mode=self._backend,
			vertical_direction=gt.vertical_direction.FORWARD
		)

	def _stencil_diagnosing_air_pressure_defs(self, in_s, in_p):
		"""
		GT4Py stencil diagnosing the pressure.

		Parameters
		----------
		in_s : gridtools.Equation
			The isentropic density, in units of [kg m^-2 K^-1].
		in_p : gridtools.Equation
			The pressure, in units of [Pa].

		Returns
		-------
		gridtools.Equation :
			The diagnosed pressure, in units of [Pa].
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_p = gt.Equation()

		# Shortcuts
		dz = self._grid.dz.to_units('K').values.item()
		g  = self._physical_constants['gravitational_acceleration']

		# Computations
		out_p[k] = in_p[k-1] + g * dz * in_s[k-1]

		return out_p

	def _stencil_diagnosing_montgomery_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the Montgomery potential.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's inputs and outputs
		if not hasattr(self, '_in_exn'):
			self._in_exn = np.zeros((nx, ny, nz+1), dtype=self._dtype)
		self._out_mtg = np.zeros((nx, ny, nz), dtype=self._dtype)
		self._in_mtg = self._out_mtg

		# Instantiate the stencil
		self._stencil_diagnosing_montgomery = gt.NGStencil( 
			definitions_func=self._stencil_diagnosing_montgomery_defs,
			inputs={'in_exn': self._in_exn, 'in_mtg': self._in_mtg},
			outputs={'out_mtg': self._out_mtg},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-2)),
			mode=self._backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

	def _stencil_diagnosing_montgomery_defs(self, in_exn, in_mtg):
		"""
		GT4Py stencil diagnosing the Exner function.

		Parameters
		----------
		in_exn : gridtools.Equation
			The Exner function, in units of [J K^-1 kg^-1]
		in_mtg : gridtools.Equation
			The Montgomery potential, in units of [J kg^-1].

		Return
		-------
		gridtools.Equation :
			The diagnosed Montgomery potential, in units of [J kg^-1].
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_mtg = gt.Equation()

		# Shortcuts
		dz = self._grid.dz.to_units('K').values.item()

		# Computations
		out_mtg[k] = in_mtg[k+1] + dz * in_exn[k+1]

		return out_mtg

	def _stencil_diagnosing_height_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the geometric
		height of the half-level isentropes.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's inputs and outputs
		if not hasattr(self, '_in_exn'):
			self._in_exn = np.zeros((nx, ny, nz+1), dtype=self._dtype)
		self._out_h = np.zeros((nx, ny, nz+1), dtype=self._dtype)
		self._in_h = self._out_h

		# Instantiate the stencil
		self._stencil_diagnosing_height = gt.NGStencil( 
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={
				'in_theta': self._theta, 'in_exn': self._in_exn,
				'in_p': self._out_p, 'in_h': self._in_h
			},
			outputs={'out_h': self._out_h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend,
			vertical_direction=gt.vertical_direction.BACKWARD
		)

	def _stencil_diagnosing_height_defs(self, in_theta, in_exn, in_p, in_h):
		"""
		GT4Py stencil diagnosing the geometric height of the isentropes.

		Parameters
		----------
		in_theta : gridtools.Equation
			The :math:`\\theta`-quota of the vertical half levels, in units of [K].
		in_exn : gridtools.Equation
			The Exner function, in units of [J K^-1 kg^-1].
		in_p : gridtools.Equation
			The pressure, in units of [Pa].
		in_h : gridtools.Equation
			The geometric height of the half levels, in units of [m].

		Return
		-------
		gridtools.Equation :
			The diagnosed geometric height of the half levels, in units of [m].
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_h = gt.Equation()

		# Get the values for the physical constants to use
		cp = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
		rd = self._physical_constants['gas_constant_of_dry_air']
		g  = self._physical_constants['gravitational_acceleration']

		# Computations
		out_h[k] = in_h[k+1] - \
			rd * (in_theta[k] * in_exn[k] + in_theta[k+1] * in_exn[k+1]) * \
			(in_p[k] - in_p[k+1]) / (cp * g * (in_p[k] + in_p[k+1]))

		return out_h

	def _stencil_diagnosing_air_density_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the density.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's inputs
		if not hasattr(self, '_in_s'):
			self._in_s = np.zeros((nx, ny, nz), dtype=self._dtype)
		self._in_h = np.zeros((nx, ny, nz+1), dtype=self._dtype)

		# Allocate the Numpy array which will serve as stencil's output
		self._out_rho = np.zeros((nx, ny, nz), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil_diagnosing_air_density = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_air_density_defs,
			inputs={'in_theta': self._theta, 'in_s': self._in_s, 'in_h': self._in_h},
			outputs={'out_rho': self._out_rho},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

	@staticmethod
	def _stencil_diagnosing_air_density_defs(in_theta, in_s, in_h):
		"""
		GT4Py stencil diagnosing the density.

		Parameters
		----------
		in_theta : gridtools.Equation
			The :math:`\\theta`-quota of the vertical half levels, in units of [K].
		in_s : gridtools.Equation
			The isentropic density, in units of [kg m^-2 K^-1].
		in_h : gridtools.Equation
			The geometric height at the half-levels, in units of [m].

		Return
		-------
		gridtools.Equation :
			The diagnosed density, in units of [kg m^-3].
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_rho = gt.Equation()

		# Computations
		out_rho[k] = in_s[k] * (in_theta[k] - in_theta[k+1]) / (in_h[k] - in_h[k+1])

		return out_rho

	def _stencil_diagnosing_air_temperature_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the temperature.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's inputs
		if not hasattr(self, '_in_exn'):
			self._in_exn = np.zeros((nx, ny, nz+1), dtype=self._dtype)

		# Allocate the Numpy array which will serve as stencil's output
		self._out_temp = np.zeros((nx, ny, nz), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil_diagnosing_air_temperature = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_air_temperature_defs,
			inputs={'in_theta': self._theta, 'in_exn': self._in_exn},
			outputs={'out_temp': self._out_temp},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend
		)

	def _stencil_diagnosing_air_temperature_defs(self, in_theta, in_exn):
		"""
		GT4Py stencil diagnosing the temperature.

		Parameters
		----------
		in_theta : gridtools.Equation
			The :math:`\\theta`-quota of the vertical half levels, in units of [K].
		in_exn : gridtools.Equation
			The Exner function, in units of [J K^-1 kg^-1].

		Return
		-------
		gridtools.Equation :
			The diagnosed temperature, in units of [K].
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_temp = gt.Equation()

		# Shortcuts
		cp = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']

		# Computations
		out_temp[k] = .5 * (in_theta[k] * in_exn[k] + in_theta[k+1] * in_exn[k+1]) / cp

		return out_temp
