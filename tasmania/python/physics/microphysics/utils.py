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
	Clipping(DiagnosticComponent)
	Precipitation(ImplicitTendencyComponent)
	SedimentationFlux
	_{First, Second}OrderUpwind(SedimentationFlux)
	Sedimentation(ImplicitTendencyComponent)
"""
import abc
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.framework.base_components import \
	DiagnosticComponent, ImplicitTendencyComponent
from tasmania.python.utils.data_utils import get_physical_constants

try:
	from tasmania.conf import datatype
except ImportError:
	from numpy import float32 as datatype


class Clipping(DiagnosticComponent):
	"""
	Clipping negative values of water species.
	"""

	def __init__(self, domain, grid_type, tracers=None):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : str
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical'.

		tracers : `dict`, optional
			Dictionary whose keys are the names of the tracers to clip,
			and whose values are dictionaries specifying fundamental
			properties ('units') for those tracers.
		"""
		self._tracers = tracers
		super().__init__(domain, grid_type)

	@property
	def input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {}
		for name, props in self._tracers.items():
			return_dict[name] = {'dims': dims, 'units': props['units']}

		return return_dict

	@property
	def diagnostic_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {}
		for name, props in self._tracers.items():
			return_dict[name] = {'dims': dims, 'units': props['units']}

		return return_dict

	def array_call(self, state):
		diagnostics = {}

		for name in self._tracers:
			q = state[name]
			q[q < 0.0] = 0.0
			diagnostics[name] = q

		return diagnostics


class Precipitation(ImplicitTendencyComponent):
	"""
	Update the (accumulated) precipitation.
	"""
	_d_physical_constants = {
		'density_of_liquid_water':
			DataArray(1e3, attrs={'units': 'kg m^-3'}),
	}

	def __init__(
		self, domain, grid_type='numerical', backend=gt.mode.NUMPY,
		dtype=datatype,	physical_constants=None, **kwargs
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'density_of_liquid_water', in units compatible with [kg m^-3].

		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.ImplicitTendencyComponent`.
		"""
		super().__init__(domain, grid_type, **kwargs)

		pcs = get_physical_constants(
			self._d_physical_constants, physical_constants
		)
		self._rhow = gt.Global(pcs['density_of_liquid_water'])

		nx, ny = self.grid.nx, self.grid.ny
		self._dt = gt.Global()
		self._in_rho = np.zeros((nx, ny, 1), dtype=dtype)
		self._in_qr = np.zeros((nx, ny, 1), dtype=dtype)
		self._in_vt = np.zeros((nx, ny, 1), dtype=dtype)
		self._in_accprec = np.zeros((nx, ny, 1), dtype=dtype)
		self._out_prec = np.zeros((nx, ny, 1), dtype=dtype)
		self._out_accprec = np.zeros((nx, ny, 1), dtype=dtype)

		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs={
				'in_rho': self._in_rho, 'in_qr': self._in_qr,
				'in_vt': self._in_vt, 'in_accprec': self._in_accprec
			},
			global_inputs={'rhow': self._rhow, 'dt': self._dt},
			outputs={'out_prec': self._out_prec, 'out_accprec': self._out_accprec},
			domain=gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, 0)),
			mode=backend
		)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims2d = (g.x.dims[0], g.y.dims[0], g.z.dims[0] + '_at_surface_level') \
			if g.nz > 1 else (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return {
			'air_density': {'dims': dims, 'units': 'kg m^-3'},
			'mass_fraction_of_precipitation_water_in_air':
				{'dims': dims, 'units': 'g g^-1'},
			'raindrop_fall_velocity': {'dims': dims, 'units': 'm s^-1'},
			'accumulated_precipitation': {'dims': dims2d, 'units': 'mm'}
		}

	@property
	def tendency_properties(self):
		return {}

	@property
	def diagnostic_properties(self):
		g = self.grid
		dims2d = (g.x.dims[0], g.y.dims[0], g.z.dims[0] + '_at_surface_level') \
			if g.nz > 1 else (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return {
			'precipitation': {'dims': dims2d, 'units': 'mm hr^-1'},
			'accumulated_precipitation': {'dims': dims2d, 'units': 'mm'}
		}

	def array_call(self, state, timestep):
		self._dt.value = timestep.total_seconds()
		self._in_rho[...] = state['air_density'][:, :, -1:]
		self._in_qr[...] = \
			state['mass_fraction_of_precipitation_water_in_air'][:, :, -1:]
		self._in_vt[...] = state['raindrop_fall_velocity'][:, :, -1:]
		self._in_accprec[...] = state['accumulated_precipitation'][...]

		self._stencil.compute()

		tendencies = {}
		diagnostics = {
			'precipitation': self._out_prec,
			'accumulated_precipitation': self._out_accprec
		}

		return tendencies, diagnostics

	@staticmethod
	def _stencil_defs(rhow, dt, in_rho, in_qr, in_vt, in_accprec):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		out_prec = gt.Equation()
		out_accprec = gt.Equation()

		out_prec[i, j] = 3.6e6 * in_rho[i, j] * in_qr[i, j] * in_vt[i, j] / rhow
		out_accprec[i, j] = in_accprec[i, j] + dt * out_prec[i, j] / 3.6e3

		return out_prec, out_accprec


class SedimentationFlux:
	"""
	Abstract base class whose derived classes discretize the
	vertical derivative of the sedimentation flux with different
	orders of accuracy.
	"""
	__metaclass__ = abc.ABCMeta

	# the vertical extent of the stencil
	nb = None

	@staticmethod
	@abc.abstractmethod
	def __call__(k, rho, h_on_interface_levels, q, vt, dfdz):
		"""
		Get the vertical derivative of the sedimentation flux.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		k : gridtools.Index
			The index running along the vertical axis.
		rho : gridtools.Equation
			The air density, in units of [kg m^-3].
		h_on_interface_levels : gridtools.Equation
			The geometric height of the model half-levels, in units of [m].
		q : gridtools.Equation
			The precipitating water species.
		vt : gridtools.Equation
			The raindrop fall velocity, in units of [m s^-1].
		dfdz : gridtools.Equation
			The vertical derivative of the sedimentation flux.
		"""

	@staticmethod
	def factory(sedimentation_flux_type):
		"""
		Static method returning an instance of the derived class
		which discretizes the vertical derivative of the
		sedimentation flux with the desired level of accuracy.

		Parameters
		----------
		sedimentation_flux_type : str
			String specifying the method used to compute the numerical
			sedimentation flux. Available options are:

			- 'first_order_upwind', for the first-order upwind scheme;
			- 'second_order_upwind', for the second-order upwind scheme.

		Return
		------
			Instance of the derived class implementing the desired method.
		"""
		if sedimentation_flux_type == 'first_order_upwind':
			return _FirstOrderUpwind()
		elif sedimentation_flux_type == 'second_order_upwind':
			return _SecondOrderUpwind()
		else:
			raise ValueError(
				'Only first- and second-order upwind methods have been implemented.'
			)


class _FirstOrderUpwind(SedimentationFlux):
	"""
	Implementation of the standard, first-order accurate upwind method
	to discretize the vertical derivative of the sedimentation flux.
	"""
	nb = 1

	def __init__(self):
		super().__init__()

	@staticmethod
	def __call__(k, rho, h_on_interface_levels, q, vt, dfdz):
		# interpolate the geometric height at the model main levels
		tmp_h = gt.Equation()
		tmp_h[k] = 0.5 * (
			h_on_interface_levels[k] + h_on_interface_levels[k + 1]
		)

		# calculate the vertical derivative of the sedimentation flux
		dfdz[k] = (rho[k - 1] * q[k - 1] * vt[k - 1] - rho[k] * q[k] * vt[k]) / \
			(tmp_h[k - 1] - tmp_h[k])


class _SecondOrderUpwind(SedimentationFlux):
	"""
	Implementation of the second-order accurate upwind method to discretize
	the vertical derivative of the sedimentation flux.
	"""
	nb = 2

	def __init__(self):
		super().__init__()

	@staticmethod
	def __call__(k, rho, h_on_interface_levels, q, vt, dfdz):
		# instantiate temporary and output fields
		tmp_h = gt.Equation()
		tmp_a = gt.Equation()
		tmp_b = gt.Equation()
		tmp_c = gt.Equation()

		# interpolate the geometric height at the model main levels
		tmp_h[k] = 0.5 * (
			h_on_interface_levels[k] + h_on_interface_levels[k + 1])

		# evaluate the space-dependent coefficients occurring in the
		# second-order, upwind finite difference approximation of the
		# vertical derivative of the flux
		tmp_a[k] = (2. * tmp_h[k] - tmp_h[k - 1] - tmp_h[k - 2]) / \
			((tmp_h[k - 1] - tmp_h[k]) * (tmp_h[k - 2] - tmp_h[k]))
		tmp_b[k] = (tmp_h[k - 2] - tmp_h[k]) / \
			((tmp_h[k - 1] - tmp_h[k]) * (tmp_h[k - 2] - tmp_h[k - 1]))
		tmp_c[k] = (tmp_h[k] - tmp_h[k - 1]) / \
			((tmp_h[k - 2] - tmp_h[k]) * (tmp_h[k - 2] - tmp_h[k - 1]))

		# calculate the vertical derivative of the sedimentation flux
		dfdz[k] = \
			tmp_a[k] * rho[k] * q[k] * vt[k] + \
			tmp_b[k] * rho[k - 1] * q[k - 1] * vt[k - 1] + \
			tmp_c[k] * rho[k - 2] * q[k - 2] * vt[k - 2]


class Sedimentation(ImplicitTendencyComponent):
	"""
	Calculate the vertical derivative of the sedimentation flux for multiple
	precipitating tracers.
	"""
	def __init__(
		self, domain, grid_type, tracers,
		sedimentation_flux_scheme='first_order_upwind',
		maximum_vertical_cfl=0.975,
		backend=gt.mode.NUMPY, dtype=datatype, **kwargs
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

		tracers : dict
			Dictionary whose keys are the names of the precipitating tracers to
			consider, and whose values are dictionaries specifying 'units' and
			'sedimentation_velocity' for those tracers.
		sedimentation_flux_scheme : `str`, optional
			The numerical sedimentation flux scheme. Please refer to
			:class:`~tasmania.SedimentationFlux` for the available options.
			Defaults to 'first_order_upwind'.
		maximum_vertical_cfl : `float`, optional
			Maximum allowed vertical CFL number. Defaults to 0.975.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.ImplicitTendencyComponent`.
		"""
		self._tracer_units = {}
		self._velocities = {}
		for tracer in tracers:
			try:
				self._tracer_units[tracer] = tracers[tracer]['units']
			except KeyError:
				raise KeyError(
					'Dictionary for ''{}'' misses the key ''units''.'.format(tracer)
				)

			try:
				self._velocities[tracer] = tracers[tracer]['sedimentation_velocity']
			except KeyError:
				raise KeyError(
					'Dictionary for ''{}'' misses the key ''sedimentation_velocity''.'
						.format(tracer)
				)

		super().__init__(domain, grid_type, **kwargs)

		self._sflux = SedimentationFlux.factory(sedimentation_flux_scheme)
		self._max_cfl = maximum_vertical_cfl
		self._stencil_initialize(backend, dtype)

	@property
	def input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

		return_dict = {
			'air_density': {'dims': dims, 'units': 'kg m^-3'},
			'height_on_interface_levels': {'dims': dims_z, 'units': 'm'}
		}

		for tracer in self._tracer_units:
			return_dict[tracer] = {'dims': dims, 'units': self._tracer_units[tracer]}
			return_dict[self._velocities[tracer]] = {'dims': dims, 'units': 'm s^-1'}

		return return_dict

	@property
	def tendency_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {}
		for tracer, units in self._tracer_units.items():
			return_dict[tracer] = {'dims': dims, 'units': units + ' s^-1'}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state, timestep):
		self._stencil_set_inputs(state, timestep)

		self._stencil.compute()

		tendencies = {
			name: self._outputs['out_' + name] for name in self._tracer_units
		}
		diagnostics = {}

		return tendencies, diagnostics

	def _stencil_initialize(self, backend, dtype):
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

		self._dt = gt.Global()
		self._maxcfl = gt.Global(self._max_cfl)

		self._inputs = {
			'in_rho': np.zeros((nx, ny, nz), dtype=dtype),
			'in_h': np.zeros((nx, ny, nz+1), dtype=dtype)
		}
		self._outputs = {}
		for tracer in self._tracer_units:
			self._inputs['in_' + tracer] = np.zeros((nx, ny, nz), dtype=dtype)
			self._inputs['in_' + self._velocities[tracer]] = \
				np.zeros((nx, ny, nz), dtype=dtype)
			self._outputs['out_' + tracer] = np.zeros((nx, ny, nz), dtype=dtype)

		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=self._inputs,
			global_inputs={'dt': self._dt, 'max_cfl': self._maxcfl},
			outputs=self._outputs,
			domain=gt.domain.Rectangle((0, 0, self._sflux.nb), (nx-1, ny-1, nz-1)),
			mode=backend
		)

	def _stencil_set_inputs(self, state, timestep):
		self._dt.value = timestep.total_seconds()
		self._inputs['in_rho'][...] = state['air_density'][...]
		self._inputs['in_h'][...] = state['height_on_interface_levels'][...]
		for tracer in self._tracer_units:
			self._inputs['in_' + tracer][...] = state[tracer][...]
			velocity = self._velocities[tracer]
			self._inputs['in_' + velocity] = state[velocity][...]

	def _stencil_defs(self, dt, max_cfl, in_rho, in_h, **kwargs):
		k = gt.Index(axis=2)

		tmp_dh = gt.Equation()
		tmp_dh[k] = in_h[k] - in_h[k+1]

		outs = []

		for tracer in self._tracer_units:
			in_q  = kwargs['in_' + tracer]
			in_vt = kwargs['in_' + self._velocities[tracer]]

			tmp_vt = gt.Equation(name='tmp_' + self._velocities[tracer])
			tmp_vt[k] = in_vt[k]
			# 	(vt[k] >  max_cfl * tmp_dh[k] / dt) * max_cfl * tmp_dh[k] / dt + \
			# 	(vt[k] <= max_cfl * tmp_dh[k] / dt) * vt[k]

			tmp_dfdz = gt.Equation(name='tmp_dfdz_' + tracer)
			self._sflux(k, in_rho, in_h, in_q, tmp_vt, tmp_dfdz)

			out_q = gt.Equation(name='out_' + tracer)
			out_q[k] = tmp_dfdz[k] / in_rho[k]

			outs.append(out_q)

		return outs
