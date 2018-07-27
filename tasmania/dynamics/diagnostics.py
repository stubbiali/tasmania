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
Classes:
	IsentropicDiagnostics
	VelocityComponents
	WaterConstituents
"""
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.utils.data_utils import get_physical_constants

try:
	from tasmania.namelist import datatype
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

	def __init__(self, grid, backend=gt.mode.NUMPY, dtype=dtype, physical_constants=None):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
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
			:obj:`tasmania.dynamics.diagnostics.IsentropicDiagnostics_d_physical_constants`
			for the default values.
		"""
		# Store the input arguments
		self._grid    = grid
		self._backend = backend
		self._dtype	  = dtype

		# Set physical constants values
		self._physical_constants = get_physical_constants(self._d_physical_constants,
														  physical_constants)

		# Assign to each grid point on the interface levels
		# the corresponding z-quota; this is required to diagnose
		# the geometrical height at the interface levels
		theta_1d = grid.z_on_interface_levels.values[np.newaxis, np.newaxis, :]
		self._theta = np.tile(theta_1d, (grid.nx, grid.ny, 1))

		# Initialize the pointers to the underlying GT4Py stencils
		# These will be properly re-directed the first time the corresponding
		# entry-point method is invoked
		self._stencil_diagnosing_air_pressure = None
		self._stencil_diagnosing_montgomery = None
		self._stencil_diagnosing_height = None
		self._stencil_diagnosing_air_density = None
		self._stencil_diagnosing_air_temperature = None

	def get_diagnostic_variables(self, s, pt):
		"""
		With the help of the isentropic density and the upper boundary
		condition on the pressure distribution, diagnose the pressure,
		the Exner function, the Montgomery potential, and the geometric
		height of the half-levels.

		Parameters
		----------
		s : array_like
			3-D :class:`numpy.ndarray` representing the isentropic density,
			in units of [kg m^-2 K^-1].
		pt : float
			The upper boundary condition on the pressure distribution,
			in units of [Pa].

		Return
		------
		p : array_like
			3-D :class:`numpy.ndarray` representing the pressure at the
			interface levels, in units of [Pa].
		exn : array_like
			3-D :class:`numpy.ndarray` representing the Exner function at the
			interface levels, in units of [J K^-1 kg^-1].
		mtg : array_like
			3-D :class:`numpy.ndarray` representing the Montgomery potential,
			in units of [J kg^-1].
		h : array_like
			3-D :class:`numpy.ndarray` representing the geometric height of the
			interface levels, in units of [m].
		"""
		# Instantiate the underlying stencils
		if self._stencil_diagnosing_air_pressure is None:
			self._stencil_diagnosing_air_pressure_initialize()
		if self._stencil_diagnosing_montgomery is None:
			self._stencil_diagnosing_montgomery_initialize()
		if self._stencil_diagnosing_height is None:
			self._stencil_diagnosing_height_initialize()

		# Shortcuts
		dz    = self._grid.dz.values.item()
		cp    = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
		p_ref = self._physical_constants['air_pressure_at_sea_level']
		Rd    = self._physical_constants['gas_constant_of_dry_air']
		g	  = self._physical_constants['gravitational_acceleration']

		# Update the attributes which serve as inputs to the GT4Py stencils
		self._in_s[:, :, :] = s[:, :, :]

		# Apply upper boundary condition on pressure
		self._out_p[:, :, 0] = pt

		# Compute pressure at all other locations
		self._stencil_diagnosing_air_pressure.compute()
	
		# Compute the Exner function (not via a GT4Py stencils)
		self._in_exn[:, :, :] = cp * (self._out_p[:, :, :] / p_ref) ** (Rd / cp)

		# Compute Montgomery potential at the lower main level
		mtg_s = self._grid.z_on_interface_levels.values[-1] * self._in_exn[:, :, -1] \
				+ g * self._grid.topography_height
		self._out_mtg[:, :, -1] = mtg_s + 0.5 * dz * self._in_exn[:, :, -1]

		# Compute Montgomery potential at all other locations
		self._stencil_diagnosing_montgomery.compute()

		# Compute geometrical height of the isentropes
		self._out_h[:, :, -1] = self._grid.topography_height
		self._stencil_diagnosing_height.compute()

		return self._out_p, self._in_exn, self._out_mtg, self._out_h

	def get_height(self, s, pt):
		"""
		With the help of the isentropic density and the upper boundary
		condition on the pressure distribution, diagnose the geometric
		height of the half-levels.

		Parameters
		----------
		s : array_like
			3-D :class:`numpy.ndarray` representing the isentropic density,
			in units of [kg m^-2 K^-1].
		pt : float
			The upper boundary condition on the pressure distribution,
			in units of [Pa].

		Return
		------
		array_like :
			3-D :class:`numpy.ndarray` representing the geometric height of the
			interface levels, in units of [m].
		"""
		# Instantiate the underlying stencils
		if self._stencil_diagnosing_air_pressure is None:
			self._stencil_diagnosing_air_pressure_initialize()
		if self._stencil_diagnosing_height is None:
			self._stencil_diagnosing_height_initialize()

		# Get the values for the physical constants to use
		cp    = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
		p_ref = self._physical_constants['air_pressure_at_sea_level']
		Rd    = self._physical_constants['gas_constant_of_dry_air']

		# Update the attributes which serve as inputs to the GT4Py stencils
		self._in_s[:, :, :] = s[:, :, :]

		# Apply upper boundary condition on pressure
		self._out_p[:, :, 0] = pt

		# Compute pressure at all other locations
		self._stencil_diagnosing_air_pressure.compute()

		# Compute the Exner function (not via a GT4Py stencils)
		self._in_exn[:, :, :] = cp * (self._out_p[:, :, :] / p_ref) ** (Rd / cp)

		# Compute geometrical height of the isentropes
		self._out_h[:, :, -1] = self._grid.topography_height
		self._stencil_diagnosing_height.compute()

		return self._out_h

	def get_air_density(self, s, h):
		"""
		With the help of the isentropic density and the geometric height
		of the interface levels, diagnose the air density.

		Parameters
		----------
		s : array_like
			3-D :class:`numpy.ndarray` representing the isentropic density,
			in units of [kg m^-2 K^-1].
		h : array_like
			3-D :class:`numpy.ndarray` representing the geometric height of the
			interface levels, in units of [m].

		Return
		------
		array_like :
			3-D :class:`numpy.ndarray` representing the air density,
			in units of [kg m^-3].
		"""
		# Instantiate the underlying stencil
		if self._stencil_diagnosing_air_density is None:
			self._stencil_diagnosing_air_density_initialize()

		# Update the attributes which serve as inputs to the stencil
		self._in_s[:, :, :] = s[:, :, :]
		self._in_h[:, :, :] = h[:, :, :]

		# Run the stencil's compute function
		self._stencil_diagnosing_air_density.compute()

		return self._out_rho

	def get_air_temperature(self, exn):
		"""
		With the help of the Exner function, diagnose the air temperature.

		Parameters
		----------
		exn : array_like
			3-D :class:`numpy.ndarray` representing the Exner function at the
			interface levels, in units of [J K^-1 kg^-1].

		Return
		------
			3-D :class:`numpy.ndarray` representing the Exner function at the
			interface levels, in units of [J K^-1 kg^-1].
		"""
		# Instantiate the underlying stencil
		if self._stencil_diagnosing_air_temperature is None:
			self._stencil_diagnosing_air_temperature_initialize()

		# Update the attributes which serve as inputs to the stencil
		self._in_exn[:, :, :] = exn[:, :, :]

		# Run the stencil's compute function
		self._stencil_diagnosing_air_temperature.compute()

		return self._out_temp

	def _stencil_diagnosing_air_pressure_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the pressure.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
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
			vertical_direction=gt.vertical_direction.FORWARD)

	def _stencil_diagnosing_air_pressure_defs(self, in_s, in_p):
		"""
		GT4Py stencil diagnosing the pressure.

		Parameters
		----------
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_p : obj
			:class:`gridtools.Equation` representing the pressure.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed pressure.
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_p = gt.Equation()

		# Shortcuts
		dz = self._grid.dz.values.item()
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

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		if not hasattr(self, 'in_exn'):
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
			vertical_direction=gt.vertical_direction.BACKWARD)

	def _stencil_diagnosing_montgomery_defs(self, in_exn, in_mtg):
		"""
		GT4Py stencil diagnosing the Exner function.

		Parameters
		----------
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed Montgomery potential.
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_mtg = gt.Equation()

		# Shortcuts
		dz = self._grid.dz.values.item()

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

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		if not hasattr(self, '_in_exn'):
			self._in_exn = np.zeros((nx, ny, nz+1), dtype=self._dtype)
		self._out_h = np.zeros((nx, ny, nz+1), dtype=self._dtype)
		self._in_h = self._out_h

		# Instantiate the stencil
		self._stencil_diagnosing_height = gt.NGStencil( 
			definitions_func=self._stencil_diagnosing_height_defs,
			inputs={'in_theta': self._theta, 'in_exn': self._in_exn,
					'in_p': self._out_p, 'in_h': self._in_h},
			outputs={'out_h': self._out_h},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend,
			vertical_direction=gt.vertical_direction.BACKWARD)

	def _stencil_diagnosing_height_defs(self, in_theta, in_exn, in_p, in_h):
		"""
		GT4Py stencil diagnosing the geometric height of the isentropes.

		Parameters
		----------
		in_theta : obj
			:class:`gridtools.Equation` representing the
			:math:`\\theta`-quota of the vertical half levels.
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner function.
		in_p : obj
			:class:`gridtools.Equation` representing the pressure.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height
			of the half levels.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed geometric
			height of the half levels.
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_h = gt.Equation()

		# Get the values for the physical constants to use
		cp    = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
		Rd    = self._physical_constants['gas_constant_of_dry_air']
		g	  = self._physical_constants['gravitational_acceleration']

		# Computations
		out_h[k] = in_h[k+1] - \
				   Rd * (in_theta[k] * in_exn[k] + in_theta[k+1] * in_exn[k+1]) * \
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
			mode=self._backend)

	@staticmethod
	def _stencil_diagnosing_air_density_defs(in_theta, in_s, in_h):
		"""
		GT4Py stencil diagnosing the density.

		Parameters
		----------
		in_theta : obj
			:class:`gridtools.Equation` representing the
			:math:`\\theta`-quota of the vertical half levels.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height
			at the half-levels.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed density.
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_rho = gt.Equation()

		# Computations
		out_rho[k] = in_s[k] * (in_theta[k] - in_theta[k+1]) / \
					 (in_h[k] - in_h[k+1])

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
			mode=self._backend)

	def _stencil_diagnosing_air_temperature_defs(self, in_theta, in_exn):
		"""
		GT4Py stencil diagnosing the temperature.

		Parameters
		----------
		in_theta : obj
			:class:`gridtools.Equation` representing the
			:math:`\\theta`-quota of the vertical half levels.
		in_exn : obj
			:class:`gridtools.Equation` representing the Exner
			function defined at the vertical half levels

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed temperature.
		"""
		# Index scanning the vertical axis
		k = gt.Index(axis=2)

		# Output field
		out_temp = gt.Equation()

		# Shortcuts
		cp = self._physical_constants['specific_heat_of_dry_air_at_constant_pressure']

		# Computations
		out_temp[k] = .5 * (in_theta[k  ] * in_exn[k  ] +
					 		in_theta[k+1] * in_exn[k+1]) / cp

		return out_temp


class WaterConstituent:
	"""
	This class diagnoses the density (respectively, mass fraction) of any water
	constituent with the help of the air density and the mass fraction (resp.,
	the density) of that water constituent.
	"""
	def __init__(self, grid, backend=gt.mode.NUMPY):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		"""
		# Store the grid
		self._grid    = grid
		self._backend = backend

		# Initialize the pointers to the underlying stencils
		# These will be properly re-directed the first time the corresponding
		# entry-point method is invoked
		self._stencil_diagnosing_density = None
		self._stencil_diagnosing_and_clipping_density = None
		self._stencil_diagnosing_mass_fraction = None
		self._stencil_diagnosing_and_clipping_mass_fraction = None

	def get_density_of_water_constituent(self, d, q, dq, clipping=False):
		"""
		Diagnose the density of a water constituent.

		Parameters
		----------
		d : array_like
			3-D :class:`numpy.ndarray` representing the air density.
		q : array_like
			3-D :class:`numpy.ndarray` representing the mass fraction
			of the water constituent, in units of [g g^-1].
		dq : array_like
			3-D :class:`numpy.ndarray` which will store the output density
			of the water constituent, in the same units of the input
			air density.
		clipping : `bool`, optional
			:obj:`True` to clip the negative values of the output field,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		"""
		# Initialize the underlying GT4Py stencils
		if self._stencil_diagnosing_density is None:
			self._stencil_diagnosing_density_initialize(dq.dtype)

		# Update the arrays which serve as stencil's inputs
		self._d[:, :, :] = d[:, :, :]
		self._q[:, :, :] = q[:, :, :]

		# Set pointer to correct stencil
		stencil = self._stencil_diagnosing_and_clipping_density if clipping \
				  else self._stencil_diagnosing_density

		# Run the stencil's compute function
		stencil.compute()

		# Set the output array
		dq[:, :, :] = self._dq[:, :, :]

	def get_mass_fraction_of_water_constituent_in_air(self, d, dq, q, clipping=False):
		"""
		Diagnose the mass fraction of a water constituent.

		Parameters
		----------
		d : array_like
			3-D :class:`numpy.ndarray` representing the air density.
		dq : array_like
			3-D :class:`numpy.ndarray` representing the density
			of the water constituent, in the same units of the input
			air density.
		q : array_like
			3-D :class:`numpy.ndarray` which will store the output mass fraction
			of the water constituent, in the same units of the input
			air density.
		clipping : `bool`, optional
			:obj:`True` to clip the negative values of the output field,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		"""
		# Initialize the underlying GT4Py stencils
		if self._stencil_diagnosing_mass_fraction is None:
			self._stencil_diagnosing_mass_fraction_initialize(q.dtype)

		# Update the arrays which serve as stencil's inputs
		self._d[:, :, :]  = d[:, :, :]
		self._dq[:, :, :] = dq[:, :, :]

		# Set pointer to correct stencil
		stencil = self._stencil_diagnosing_and_clipping_mass_fraction if clipping \
				  else self._stencil_diagnosing_mass_fraction

		# Run the stencil's compute function
		stencil.compute()

		# Update the output array
		q[:, :, :] = self._q[:, :, :]

	def _stencil_diagnosing_density_initialize(self, dtype):
		"""
		Initialize the GT4Py stencils in charge of diagnosing the density
		of the water constituent.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=dtype)
		if not hasattr(self, '_q'):
			self._q = np.zeros((nx, ny, nz), dtype=dtype)
		if not hasattr(self, '_dq'):
			self._dq = np.zeros((nx, ny, nz), dtype=dtype)

		# Instantiate the stencil which does not clip the negative values
		self._stencil_diagnosing_density = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_density_defs,
			inputs={'in_d': self._d, 'in_q': self._q},
			outputs={'out_dq': self._dq},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend)

		# Instantiate the stencil which does clip the negative values
		self._stencil_diagnosing_and_clipping_density = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_and_clipping_density_defs,
			inputs={'in_d': self._d, 'in_q': self._q},
			outputs={'out_dq': self._dq},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend)

	@staticmethod
	def _stencil_diagnosing_density_defs(in_d, in_q):
		"""
		GT4Py stencil diagnosing the density of the water constituent.

		Parameters
		----------
		in_d : obj
			:class:`gridtools.Equation` representing the air density.
		in_q : obj
			:class:`gridtools.Equation` representing the mass fraction
			of the water constituent.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed density
			of the water constituent.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_dq = gt.Equation()

		# Computations
		out_dq[i, j, k] = in_d[i, j, k] * in_q[i, j, k]

		return out_dq

	@staticmethod
	def _stencil_diagnosing_and_clipping_density_defs(in_d, in_q):
		"""
		GT4Py stencil diagnosing the density of the water constituent,
		and then clipping the negative values.

		Parameters
		----------
		in_d : obj
			:class:`gridtools.Equation` representing the air density.
		in_q : obj
			:class:`gridtools.Equation` representing the mass fraction
			of the water constituent.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed density
			of the water constituent.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Temporary and output field
		tmp_dq = gt.Equation()
		out_dq = gt.Equation()

		# Computations
		tmp_dq[i, j, k] = in_d[i, j, k] * in_q[i, j, k]
		out_dq[i, j, k] = (tmp_dq[i, j, k] > 0.) * tmp_dq[i, j, k]

		return out_dq

	def _stencil_diagnosing_mass_fraction_initialize(self, dtype):
		"""
		Initialize the GT4Py stencils in charge of diagnosing the mass fraction
		of the water constituent.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's
		# inputs and outputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=dtype)
		if not hasattr(self, '_dq'):
			self._dq = np.zeros((nx, ny, nz), dtype=dtype)
		if not hasattr(self, '_q'):
			self._q = np.zeros((nx, ny, nz), dtype=dtype)

		# Instantiate the stencil which does not clip the negative values
		self._stencil_diagnosing_mass_fraction = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_mass_fraction_defs,
			inputs={'in_d': self._d, 'in_dq': self._dq},
			outputs={'out_q': self._q},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend)

		# Instantiate the stencil which does clip the negative values
		self._stencil_diagnosing_and_clipping_mass_fraction = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_and_clipping_mass_fraction_defs,
			inputs={'in_d': self._d, 'in_dq': self._dq},
			outputs={'out_q': self._q},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend)

	@staticmethod
	def _stencil_diagnosing_mass_fraction_defs(in_d, in_dq):
		"""
		GT4Py stencil diagnosing the mass fraction of the water constituent.

		Parameters
		----------
		in_d : obj
			:class:`gridtools.Equation` representing the air density.
		in_dq : obj
			:class:`gridtools.Equation` representing the density
			of the water constituent.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed mass fraction
			of the water constituent.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_q = gt.Equation()

		# Computations
		out_q[i, j, k] = in_dq[i, j, k] / in_d[i, j, k]

		return out_q

	@staticmethod
	def _stencil_diagnosing_and_clipping_mass_fraction_defs(in_d, in_dq):
		"""
		GT4Py stencil diagnosing the mass fraction of the water constituent,
		and then clipping the negative values.

		Parameters
		----------
		in_d : obj
			:class:`gridtools.Equation` representing the air density.
		in_dq : obj
			:class:`gridtools.Equation` representing the density
			of the water constituent.

		Return
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed mass fraction
			of the water constituent.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Temporary and output field
		tmp_q = gt.Equation()
		out_q = gt.Equation()

		# Computations
		tmp_q[i, j, k] = in_dq[i, j, k] / in_d[i, j, k]
		out_q[i, j, k] = (tmp_q[i, j, k] > 0.) * tmp_q[i, j, k]

		return out_q


class HorizontalVelocity:
	"""
	This class diagnoses the horizontal momenta (respectively, velocity
	components) with the help of the air density and the horizontal
	velocity components (resp., momenta).
	"""
	def __init__(self, grid, backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			representing the underlying grid.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
			Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		# Store the grid
		self._grid    = grid
		self._backend = backend
		self._dtype   = dtype

		# Initialize the pointers to the underlying stencils
		# These will be properly re-directed the first time the corresponding
		# entry-point method is invoked
		self._stencil_diagnosing_momenta    = None
		self._stencil_diagnosing_velocity_x = None
		self._stencil_diagnosing_velocity_y = None

	def get_momenta(self, d, u, v):
		"""
		Diagnose the horizontal momenta.

		Parameters
		----------
		d : array_like
			3-D :class:`class.ndarray` representing the air density.
		u : array_like
			3-D :class:`class.ndarray` representing the :math:`x`-staggered
			:math:`x`-velocity field.
		v : array_like
			3-D :class:`class.ndarray` representing the :math:`y`-staggered
			:math:`y`-velocity field.

		Returns
		-------
		du : array_like
			3-D :class:`class.ndarray` representing the :math:`x`-momentum.
		dv : array_like
			3-D :class:`class.ndarray` representing the :math:`y`-momentum.
		"""
		# Initialize the underlying stencil
		if self._stencil_diagnosing_momenta is None:
			self._stencil_diagnosing_momenta_initialize()

		# Update the arrays which serve as stencil's inputs
		self._d[:, :, :] = d[:, :, :]
		self._u[:, :, :] = u[:, :, :]
		self._v[:, :, :] = v[:, :, :]

		# Call the stencil's compute function
		self._stencil_diagnosing_momenta.compute()

		return self._du, self._dv

	def get_velocity_components(self, d, du, dv):
		"""
		Diagnose the horizontal velocity components.

		Parameters
		----------
		d : array_like
			3-D :class:`class.ndarray` representing the air density.
		du : array_like
			3-D :class:`class.ndarray` representing the :math:`x`-momentum.
		dv : array_like
			3-D :class:`class.ndarray` representing the :math:`y`-momentum.

		Returns
		-------
		u : array_like
			3-D :class:`class.ndarray` representing the :math:`x`-staggered
			:math:`x`-velocity field.
		v : array_like
			3-D :class:`class.ndarray` representing the :math:`y`-staggered
			:math:`y`-velocity field.

		Note
		----
		The first and last rows (respectively, columns) of the staggered
		:math:`x`-velocity (resp., :math:`y`-velocity) are not set.
		"""
		# Initialize the underlying stencil
		if self._stencil_diagnosing_velocity_x is None:
			self._stencil_diagnosing_velocity_x_initialize()
		if self._stencil_diagnosing_velocity_y is None:
			self._stencil_diagnosing_velocity_y_initialize()

		# Update the arrays which serve as stencils' inputs
		self._d[:, :, :]  = d[:, :, :]
		self._du[:, :, :] = du[:, :, :]
		self._dv[:, :, :] = dv[:, :, :]

		# Call the stencils' compute function
		self._stencil_diagnosing_velocity_x.compute()
		self._stencil_diagnosing_velocity_y.compute()

		return self._u, self._v

	def _stencil_diagnosing_momenta_initialize(self):
		"""
		Initialize the GT4Py stencil diagnosing the horizontal momenta.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's inputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_u'):
			self._u = np.zeros((nx+1, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_v'):
			self._v = np.zeros((nx, ny+1, nz), dtype=self._dtype)

		# Allocate the Numpy arrays which will serve as stencil's outputs
		if not hasattr(self, '_du'):
			self._du = np.zeros((nx, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_dv'):
			self._dv = np.zeros((nx, ny, nz), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil_diagnosing_momenta = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_momenta_defs,
			inputs={'in_d': self._d, 'in_u': self._u, 'in_v': self._v},
			outputs={'out_du': self._du, 'out_dv': self._dv},
			domain=gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend)

	@staticmethod
	def _stencil_diagnosing_momenta_defs(in_d, in_u, in_v):
		"""
		GT4Py stencil diagnosing the horizontal momenta.

		Parameters
		----------
		in_d : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_u : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`x`-velocity.
		in_v : obj
			:class:`gridtools.Equation` representing the diagnosed :math:`y`-velocity.

		Returns
		-------
		out_du : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		out_dv : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_du = gt.Equation()
		out_dv = gt.Equation()

		# Computations
		out_du[i, j, k] = 0.5 * in_d[i, j, k] * (in_u[i, j, k] + in_u[i+1, j, k])
		out_dv[i, j, k] = 0.5 * in_d[i, j, k] * (in_v[i, j, k] + in_v[i, j+1, k])

		return out_du, out_dv

	def _stencil_diagnosing_velocity_x_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the
		:math:`x`-component of the velocity.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's inputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_du'):
			self._du = np.zeros((nx, ny, nz), dtype=self._dtype)

		# Allocate the Numpy array which will serve as stencil's outputs
		if not hasattr(self, '_u'):
			self._u = np.zeros((nx+1, ny, nz), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil_diagnosing_velocity_x = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_velocity_x_defs,
			inputs={'in_d': self._d, 'in_du': self._du},
			outputs={'out_u': self._u},
			domain=gt.domain.Rectangle((1, 0, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend)

	def _stencil_diagnosing_velocity_y_initialize(self):
		"""
		Initialize the GT4Py stencil in charge of diagnosing the
		:math:`y`-component of the velocity.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will serve as stencil's inputs
		if not hasattr(self, '_d'):
			self._d = np.zeros((nx, ny, nz), dtype=self._dtype)
		if not hasattr(self, '_dv'):
			self._dv = np.zeros((nx, ny, nz), dtype=self._dtype)

		# Allocate the Numpy array which will serve as stencil's outputs
		if not hasattr(self, '_v'):
			self._v = np.zeros((nx, ny+1, nz), dtype=self._dtype)

		# Instantiate the stencil
		self._stencil_diagnosing_velocity_y = gt.NGStencil(
			definitions_func=self._stencil_diagnosing_velocity_y_defs,
			inputs={'in_d': self._d, 'in_dv': self._dv},
			outputs={'out_v': self._v},
			domain=gt.domain.Rectangle((0, 1, 0), (nx-1, ny-1, nz-1)),
			mode=self._backend)

	@staticmethod
	def _stencil_diagnosing_velocity_x_defs(in_d, in_du):
		"""
		GT4Py stencil diagnosing the :math:`x`-component of the velocity.

		Parameters
		----------
		in_d : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_du : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed :math:`x`-velocity.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_u = gt.Equation()

		# Computations
		out_u[i, j, k] = (in_du[i-1, j, k] + in_du[i, j, k]) / \
						 (in_d[i-1, j, k] + in_d[i, j, k])

		return out_u

	@staticmethod
	def _stencil_diagnosing_velocity_y_defs(in_d, in_dv):
		"""
		GT4Py stencil diagnosing the :math:`y`-component of the velocity.

		Parameters
		----------
		in_d : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_dv : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Returns
		-------
		obj :
			:class:`gridtools.Equation` representing the diagnosed :math:`y`-velocity.
		"""
		# Indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_v = gt.Equation()

		# Computations
		out_v[i, j, k] = (in_dv[i, j-1, k] + in_dv[i, j, k]) / \
						 (in_d[i, j-1, k] + in_d[i, j, k])

		return out_v
