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
	IsentropicPrognostic
"""
import abc
import numpy as np

import gridtools as gt

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


# convenient aliases
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


class IsentropicPrognostic:
	"""
	Abstract base class whose derived classes implement different
	schemes to carry out the prognostic steps of the three-dimensional
	moist, isentropic dynamical core. The schemes might be *semi-implicit* -
	they treat horizontal advection explicitly and the pressure gradient
	implicitly. The vertical advection, the Coriolis acceleration and
	the sedimentation motion are not included in the dynamics, but rather
	parameterized. The conservative form of the governing equations is used.
	"""
	# make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(
		self, horizontal_flux_class, horizontal_flux_scheme, grid, hb,
		moist, backend, dtype=datatype
	):
		"""
		Parameters
		----------
		horizontal_flux_class : IsentropicHorizontalFlux, IsentropicMinimal
			Either :class:`~tasmania.IsentropicHorizontalFlux`
			or :class:`~tasmania.IsentropicMinimalHorizontalFlux`.
		horizontal_flux_scheme : str
			The numerical horizontal flux scheme to implement.
			See :class:`~tasmania.IsentropicHorizontalFlux` and
			:class:`~tasmania.IsentropicMinimalHorizontalFlux`
			for the complete list of the available options.
		grid : tasmania.Grid
			The underlying grid.
		hb : tasmania.HorizontalBoundary
			The object handling the lateral boundary conditions.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		"""
		# keep track of the input parameters
		self._hflux_scheme	= horizontal_flux_scheme
		self._grid          = grid
		self._hb            = hb
		self._moist      	= moist
		self._backend		= backend
		self._dtype			= dtype

		# instantiate the class computing the numerical horizontal fluxes
		self._hflux = horizontal_flux_class.factory(
			self._hflux_scheme, grid, moist,
		)
		assert hb.nb >= self._hflux.extent, \
			"The number of lateral boundary layers is {}, but should be " \
			"greater or equal than {}.".format(hb.nb, self._hflux.extent)
		assert grid.nx >= 2*hb.nb+1, \
			"The number of grid points along the first horizontal " \
			"dimension is {}, but should be greater or equal than {}.".format(
				grid.nx, 2*hb.nb+1
			)
		assert grid.ny >= 2*hb.nb+1, \
			"The number of grid points along the second horizontal " \
			"dimension is {}, but should be greater or equal than {}.".format(
				grid.ny, 2*hb.nb+1
			)

		dx = grid.dx.to_units('m').values.item()
		self._dx = gt.Global(dx)
		dy = grid.dy.to_units('m').values.item()
		self._dy = gt.Global(dy)

	@property
	@abc.abstractmethod
	def stages(self):
		"""
		Return
		------
		int :
			The number of stages performed by the time-integration scheme.
		"""

	@abc.abstractmethod
	def stage_call(self, stage, timestep, state, tendencies=None):
		"""
		Perform a stage.

		Parameters
		----------
		stage : int
			The stage to perform.
		timestep : timedelta
			:class:`datetime.timedelta` representing the time step.
		state : dict
			Dictionary whose keys are strings indicating model variables,
			and values are :class:`numpy.ndarray`\s representing the values
			for those variables.
		tendencies : dict
			Dictionary whose keys are strings indicating model variables,
			and values are :class:`numpy.ndarray`\s representing (slow and
			intermediate) physical tendencies for those variables.

		Return
		------
		dict :
			Dictionary whose keys are strings indicating the conservative
			prognostic model variables, and values are :class:`numpy.ndarray`\s
			containing new values for those variables.
		"""
		pass

	@staticmethod
	def factory(
		time_integration_scheme, horizontal_flux_scheme, grid, hb,
		moist=False, backend=gt.mode.NUMPY, dtype=datatype, **kwargs
	):
		"""
		Static method returning an instance of the derived class implementing
		the time stepping scheme specified by ``time_scheme``.

		Parameters
		----------
		time_integration_scheme : str
			The time stepping method to implement. Available options are:

				* 'forward_euler_si', for the semi-implicit forward Euler scheme;
				* 'centered', for the semi-implicit centered scheme;
				* 'rk3ws', for the semi-implicit three-stages RK scheme.

		horizontal_flux_scheme : str
			The numerical horizontal flux scheme to implement.
			See :class:`~tasmania.IsentropicHorizontalFlux` and
			:class:`~tasmania.IsentropicMinimalHorizontalFlux`
			for the complete list of the available options.
		grid : tasmania.Grid
			The underlying grid.
		hb : tasmania.HorizontalBoundary
			The object handling the lateral boundary conditions.
		moist : `bool`, optional
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		backend : obj
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.

		Return
		------
		obj :
			An instance of the derived class implementing ``time_integration_scheme``.
		"""
		from .implementations.prognostic import \
			ForwardEulerSI, CenteredSI, RK3WSSI
		args = (horizontal_flux_scheme,	grid, hb, moist, backend, dtype)

		if time_integration_scheme == 'forward_euler_si':
			return ForwardEulerSI(*args, **kwargs)
		elif time_integration_scheme == 'centered_si':
			return CenteredSI(*args, **kwargs)
		elif time_integration_scheme == 'rk3ws_si':
			return RK3WSSI(*args, **kwargs)
		else:
			raise ValueError(
				"Unknown time integration scheme {}. Available options are: "
				"forward_euler_si, centered_si, rk3ws_si.".format(
					time_integration_scheme
				)
			)

	def _stencils_allocate(self, tendencies):
		"""
		Allocate the arrays and globals which serve as inputs and outputs to
		the underlying GT4Py stencils.
		"""
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype
		tendency_names = () if tendencies is None else tendencies.keys()

		# instantiate a GT4Py Global representing the timestep
		self._dt = gt.Global()

		# allocate the Numpy arrays which will store the current values
		# for the model variables
		self._s_now	  = np.zeros((  nx,	ny, nz), dtype=dtype)
		self._u_now	  = np.zeros((nx+1,	ny, nz), dtype=dtype)
		self._v_now	  = np.zeros((  nx, ny+1, nz), dtype=dtype)
		self._mtg_now = np.zeros((  nx,  ny, nz), dtype=dtype)
		self._su_now  = np.zeros((  nx,	ny, nz), dtype=dtype)
		self._sv_now  = np.zeros((  nx,	ny, nz), dtype=dtype)
		if self._moist:
			self._sqv_now = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqc_now = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqr_now = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the input Numpy arrays which will store the tendencies
		# for the model variables
		if tendency_names is not None:
			if 'air_isentropic_density' in tendency_names:
				self._s_tnd = np.zeros((nx, ny, nz), dtype=dtype)
			if 'x_momentum_isentropic' in tendency_names:
				self._su_tnd = np.zeros((nx, ny, nz), dtype=dtype)
			if 'y_momentum_isentropic' in tendency_names:
				self._sv_tnd = np.zeros((nx, ny, nz), dtype=dtype)
			if mfwv in tendency_names:
				self._qv_tnd = np.zeros((nx, ny, nz), dtype=dtype)
			if mfcw in tendency_names:
				self._qc_tnd = np.zeros((nx, ny, nz), dtype=dtype)
			if mfpw in tendency_names:
				self._qr_tnd = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the Numpy arrays which will store the output values for
		# the model variables
		self._s_new  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su_new = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv_new = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._sqv_new = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqc_new = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqr_new = np.zeros((nx, ny, nz), dtype=dtype)

	def _stencils_set_inputs(self, stage, timestep, state, tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which perform the stages.
		"""
		# shortcuts
		if tendencies is not None:
			s_tnd_on  = tendencies.get('air_isentropic_density', None) is not None
			su_tnd_on = tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = tendencies.get('y_momentum_isentropic', None) is not None
			qv_tnd_on = tendencies.get(mfwv, None) is not None
			qc_tnd_on = tendencies.get(mfcw, None) is not None
			qr_tnd_on = tendencies.get(mfpw, None) is not None
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# update the local time step
		self._dt.value = timestep.total_seconds()

		# update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._s_now[...]   = state['air_isentropic_density'][...]
		self._u_now[...]   = state['x_velocity_at_u_locations'][...]
		self._v_now[...]   = state['y_velocity_at_v_locations'][...]
		self._mtg_now[...] = state['montgomery_potential'][...]
		self._su_now[...]  = state['x_momentum_isentropic'][...]
		self._sv_now[...]  = state['y_momentum_isentropic'][...]
		if self._moist:
			self._sqv_now[...] = state['isentropic_density_of_water_vapor'][...]
			self._sqc_now[...] = state['isentropic_density_of_cloud_liquid_water'][...]
			self._sqr_now[...] = state['isentropic_density_of_precipitation_water'][...]
		if s_tnd_on:
			self._s_tnd[...]  = tendencies['air_isentropic_density'][...]
		if su_tnd_on:
			self._su_tnd[...] = tendencies['x_momentum_isentropic'][...]
		if sv_tnd_on:
			self._sv_tnd[...] = tendencies['y_momentum_isentropic'][...]
		if qv_tnd_on:
			self._qv_tnd[...] = tendencies[mfwv][...]
		if qc_tnd_on:
			self._qc_tnd[...] = tendencies[mfcw][...]
		if qr_tnd_on:
			self._qr_tnd[...] = tendencies[mfpw][...]
