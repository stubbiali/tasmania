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
	_Centered(HomogeneousIsentropicPrognostic)
	_ForwardEuler(HomogeneousIsentropicPrognostic)
	_RK2(HomogeneousIsentropicPrognostic)
	_RK3COSMO(HomogeneousIsentropicPrognostic)
	_RK3(HomogeneousIsentropicPrognostic)
"""
import numpy as np

import gridtools as gt
from tasmania.dynamics.homogeneous_isentropic_prognostic \
	import HomogeneousIsentropicPrognostic

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


# Convenient aliases
mf_wv  = 'mass_fraction_of_water_vapor_in_air'
mf_clw = 'mass_fraction_of_cloud_liquid_water_in_air'
mf_pw  = 'mass_fraction_of_precipitation_water_in_air'


class _Centered(HomogeneousIsentropicPrognostic):
	"""
	This class inherits
	:class:`~tasmania.dynamics.homogeneous_prognostic_isentropic.HomogeneousPrognosticIsentropic`
	to implement a centered time-integration scheme which takes over the
	prognostic part of the three-dimensional, moist, homogeneous, isentropic
	dynamical core.
	"""
	def __init__(self, grid, moist_on, horizontal_boundary_conditions,
				 horizontal_flux_scheme, backend, dtype=datatype):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, horizontal_boundary_conditions,
						 horizontal_flux_scheme, backend, dtype)

		# Initialize the pointer to the underlying GT4Py stencil; it will
		# be properly re-directed the first time the call operator is invoked
		self._stencil = None

		# Boolean flag to quickly assess whether we are within the first time step
		self._is_first_timestep = True

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 1

	def __call__(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time. Only horizontal derivatives are considered;
		possible vertical derivatives are disregarded.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(raw_state, raw_tendencies)

		# Update the attributes which serve as inputs to the GT4Py stencil
		self._stencil_set_inputs(stage, dt, raw_state, raw_tendencies)

		# Run the stencil's compute function
		self._stencil.compute()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Bring the updated isentropic density back to the original dimensions
		s_new = self._hboundary.from_computational_to_physical_domain(
			self._out_s[:mi, :mj, :], (nx, ny, nz))

		# Bring the updated momenta back to the original dimensions
		# Note: let's forget about symmetric conditions...
		su_new = self._hboundary.from_computational_to_physical_domain(
			self._out_su[:mi, :mj, :], (nx, ny, nz))
		sv_new = self._hboundary.from_computational_to_physical_domain(
			self._out_sv[:mi, :mj, :], (nx, ny, nz))

		# Bring the updated water constituents back to the original dimensions
		if self._moist_on:
			sqv_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqv[:mi, :mj, :], (nx, ny, nz))
			sqc_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqc[:mi, :mj, :], (nx, ny, nz))
			sqr_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqr[:mi, :mj, :], (nx, ny, nz))

		# Apply the boundary conditions
		self._hboundary.enforce(s_new,	raw_state['air_isentropic_density'])
		self._hboundary.enforce(su_new, raw_state['x_momentum_isentropic'])
		self._hboundary.enforce(sv_new, raw_state['y_momentum_isentropic'])
		if self._moist_on:
			self._hboundary.enforce(sqv_new,
									raw_state['isentropic_density_of_water_vapor'])
			self._hboundary.enforce(sqc_new,
									raw_state['isentropic_density_of_cloud_liquid_water'])
			self._hboundary.enforce(sqr_new,
									raw_state['isentropic_density_of_precipitation_water'])

		# Instantiate the output state
		raw_state_new = {
			'time': raw_state['time'] + dt,
			'air_isentropic_density': s_new,
			'x_momentum_isentropic': su_new,
			'y_momentum_isentropic': sv_new,
		}
		if self._moist_on:
			raw_state_new['isentropic_density_of_water_vapor'] = sqv_new
			raw_state_new['isentropic_density_of_cloud_liquid_water'] = sqc_new
			raw_state_new['isentropic_density_of_precipitation_water'] = sqr_new

		# Keep track of the current state for the next time step
		self._in_s_old[:mi, :mj, :]  = self._in_s[:mi, :mj, :]
		self._in_su_old[:mi, :mj, :] = self._in_su[:mi, :mj, :]
		self._in_sv_old[:mi, :mj, :] = self._in_sv[:mi, :mj, :]
		if self._moist_on:
			self._in_sqv_old[:mi, :mj, :] = self._in_sqv[:mi, :mj, :]
			self._in_sqc_old[:mi, :mj, :] = self._in_sqc[:mi, :mj, :]
			self._in_sqr_old[:mi, :mj, :] = self._in_sqr[:mi, :mj, :]

		# At this point, the first time step is surely over
		self._is_first_timestep = False

		return raw_state_new

	def _stencil_initialize(self, raw_state, raw_tendencies):
		"""
		Initialize the GT4Py stencil implementing a time-integration centered scheme
		to step the solution by neglecting vertical advection.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencil_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store the output fields
		self._stencil_allocate_outputs()

		# Set the stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
				   'in_su': self._in_su, 'in_sv': self._in_sv,
				   'in_s_old': self._in_s_old, 'in_su_old': self._in_su_old,
				   'in_sv_old': self._in_sv_old}
		_outputs = {'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_old': self._in_sqv_old,
							'in_sqc': self._in_sqc, 'in_sqc_old': self._in_sqc_old,
							'in_sqr': self._in_sqr, 'in_sqr_old': self._in_sqr_old})
			_outputs.update({'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
							 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('x_velocity', None) is not None:
				_inputs['in_u_tnd'] = self._in_u_tnd
			if raw_tendencies.get('y_velocity', None) is not None:
				_inputs['in_v_tnd'] = self._in_v_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# Set the stencil's computational domain and backend
		nz, nb = self._grid.nz, self.nb
		mi, mj = self._hboundary.mi, self._hboundary.mj
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0),
									  (nb + ni - 1, nb + nj - 1, nk - 1))
		_mode = self._backend

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

	def _stencil_allocate_inputs(self, raw_tendencies):
		"""
		Allocate the attributes which will serve as inputs to the GT4Py
		stencil stepping the solution by neglecting vertical advection.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Instantiate a GT4Py Global representing the time step,
		# and the Numpy arrays which represent the solution and
		# the tendencies at the current time level
		super()._stencil_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which represent the solution
		# at the previous time level
		self._in_s_old	= np.zeros((mi, mj, nz), dtype=dtype)
		self._in_su_old = np.zeros((mi, mj, nz), dtype=dtype)
		self._in_sv_old = np.zeros((mi, mj, nz), dtype=dtype)
		if self._moist_on:
			self._in_sqv_old = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqc_old = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqr_old = np.zeros((mi, mj, nz), dtype=dtype)

	def _stencil_set_inputs(self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding the vertical advection.
		"""
		# Update the time step, and the Numpy arrays representing
		# the solution and the tendencies at the current time level
		super()._stencil_set_inputs(stage, dt, raw_state, raw_tendencies)

		# At the first iteration, update the Numpy arrays representing
		# the solution at the previous time step
		if self._is_first_timestep:
			self._in_s_old[...] = self._hboundary.from_physical_to_computational_domain(
				raw_state['air_isentropic_density'])
			self._in_su_old[...] = self._hboundary.from_physical_to_computational_domain(
				raw_state['x_momentum_isentropic'])
			self._in_sv_old[...] = self._hboundary.from_physical_to_computational_domain(
				raw_state['y_momentum_isentropic'])

			if self._moist_on:
				self._in_sqv_old[...] = self._hboundary.from_physical_to_computational_domain(
					raw_state['isentropic_density_of_water_vapor'])
				self._in_sqc_old[...] = self._hboundary.from_physical_to_computational_domain(
					raw_state['isentropic_density_of_cloud_liquid_water'])
				self._in_sqr_old[...] = self._hboundary.from_physical_to_computational_domain(
					raw_state['isentropic_density_of_precipitation_water'])

	def _stencil_defs(self, dt, in_s, in_u, in_v, in_su, in_sv, in_s_old,
					  in_su_old, in_sv_old, in_sqv=None, in_sqc=None, in_sqr=None,
					  in_sqv_old=None, in_sqc_old=None, in_sqr_old=None,
					  in_u_tnd=None, in_v_tnd=None,
					  in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil implementing the centered time-integration scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density
			at the current time.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity
			at the current time.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity
			at the current time.
		in_su : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum
			at the current time.
		in_sv : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum
			at the current time.
		in_s_old : obj
			:class:`gridtools.Equation` representing the isentropic density
			at the previous time level.
		in_su_old : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum
			at the previous time level.
		in_sv_old : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum
			at the previous time level.
		in_sqv : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			water vapor at the current time.
		in_sqc : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			cloud water at the current time.
		in_sqr : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			precipitation water at the current time.
		in_sqv_old : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			water vapor at the previous time level.
		in_sqc_old : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			cloud water at the previous time level.
		in_sqr_old : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of
			precipitation water at the previous time level.
		in_u_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the :math:`x`-velocity.
		in_v_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the :math:`y`-velocity.
		in_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the mass fraction of water vapor.
		in_qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the mass fraction of cloud liquid water.
		in_qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency of
			the mass fraction of precipitation water.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		out_su : obj
			:class:`gridtools.Equation` representing the stepped :math:`x`-momentum.
		out_sv : obj
			:class:`gridtools.Equation` representing the stepped :math:`y`-momentum.
		out_sqv : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of water vapor.
		out_sqc : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of cloud water.
		out_sqr : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of precipitation water.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)
		k = gt.Index(axis=2)

		# Instantiate the output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the horizontal fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_su, in_sv,
							u_tnd=in_u_tnd, v_tnd=in_v_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_su, in_sv, in_sqv, in_sqc, in_sqr,
							in_u_tnd, in_v_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		out_s[i, j] = in_s_old[i, j] \
					  - 2. * dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
								   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)

		# Advance the x-momentum
		if in_u_tnd is None:
			out_su[i, j] = in_su_old[i, j] \
						   - 2. * dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
										(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy)
		else:
			out_su[i, j] = in_su_old[i, j] \
						   - 2. * dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
										(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
										in_s[i, j] * in_u_tnd[i, j])

		# Advance the y-momentum
		if in_v_tnd is None:
			out_sv[i, j] = in_sv_old[i, j] \
						   - 2. * dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
										(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy)
		else:
			out_sv[i, j] = in_sv_old[i, j] \
						   - 2. * dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
										(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
										in_s[i, j] * in_v_tnd[i, j])

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv_old[i, j] \
								- 2. * dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
											 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv[i, j] = in_sqv_old[i, j] \
								- 2. * dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
											 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
											 in_s[i, j] * in_qv_tnd[i, j])

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc_old[i, j] \
								- 2. * dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
											 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc[i, j] = in_sqc_old[i, j] \
								- 2. * dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
											 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
											 in_s[i, j] * in_qc_tnd[i, j])

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr_old[i, j] \
								- 2. * dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
											 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr[i, j] = in_sqr_old[i, j] \
								- 2. * dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
											 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
											 in_s[i, j] * in_qr_tnd[i, j])

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


class _ForwardEuler(HomogeneousIsentropicPrognostic):
	"""
	This class inherits
	:class:`~tasmania.dynamics.homogeneous_prognostic_isentropic.HomogeneousPrognosticIsentropic`
	to implement the forward Euler time-integration scheme which takes over the
	prognostic part of the three-dimensional, moist, homogeneous, isentropic
	dynamical core.
	"""
	def __init__(self, grid, moist_on, horizontal_boundary_conditions,
				 horizontal_flux_scheme, backend, dtype=datatype):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, horizontal_boundary_conditions,
						 horizontal_flux_scheme, backend, dtype)

		# Initialize the pointer to the underlying GT4Py stencil; it will
		# be properly re-directed the first time the call operator is invoked
		self._stencil = None

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 1

	def __call__(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time. Only horizontal derivatives are considered;
		possible vertical derivatives are disregarded.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(raw_state, raw_tendencies)

		# Update the attributes which serve as inputs to the GT4Py stencil
		self._stencil_set_inputs(stage, dt, raw_state, raw_tendencies)

		# Step the prognostic variables
		self._stencil.compute()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Bring the updated isentropic density back to the original dimensions,
		# and enforce the boundary conditions
		s_new = self._hboundary.from_computational_to_physical_domain(
			self._out_s[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(s_new,	raw_state['air_isentropic_density'])

		# Bring the updated momenta back to the original dimensions,
		# and enforce the boundary conditions
		# Note: let's forget about symmetric conditions...
		su_new = self._hboundary.from_computational_to_physical_domain(
			self._out_su[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(su_new, raw_state['x_momentum_isentropic'])
		sv_new = self._hboundary.from_computational_to_physical_domain(
			self._out_sv[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(sv_new, raw_state['y_momentum_isentropic'])

		if self._moist_on:
			# Bring the updated water constituents back to the original dimensions,
			# and enforce the boundary conditions
			sqv_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqv[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqv_new,
									raw_state['isentropic_density_of_water_vapor'])

			sqc_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqc[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqc_new,
									raw_state['isentropic_density_of_cloud_liquid_water'])

			sqr_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqr[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqr_new,
									raw_state['isentropic_density_of_precipitation_water'])

		# Instantiate the output state
		raw_state_new = {
			'time': raw_state['time'] + dt,
			'air_isentropic_density': s_new,
			'x_momentum_isentropic': su_new,
			'y_momentum_isentropic': sv_new,
		}
		if self._moist_on:
			raw_state_new['isentropic_density_of_water_vapor'] = sqv_new
			raw_state_new['isentropic_density_of_cloud_liquid_water'] = sqc_new
			raw_state_new['isentropic_density_of_precipitation_water'] = sqr_new

		return raw_state_new

	def _stencil_initialize(self, raw_state, raw_tendencies):
		"""
		Initialize the GT4Py stencil implementing a time-integration centered scheme
		to step the solution by neglecting vertical advection.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencil_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store the output fields
		self._stencil_allocate_outputs()

		# Set the stencil's computational domain and backend
		nz, nb = self._grid.nz, self.nb
		mi, mj = self._hboundary.mi, self._hboundary.mj
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0),
									  (nb + ni - 1, nb + nj - 1, nk - 1))
		_mode = self._backend

		# Set the stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
				   'in_su': self._in_su, 'in_sv': self._in_sv}
		_outputs = {'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqc': self._in_sqc,
							'in_sqr': self._in_sqr})
			_outputs.update({'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
							 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('x_velocity', None) is not None:
				_inputs['in_u_tnd'] = self._in_u_tnd
			if raw_tendencies.get('y_velocity', None) is not None:
				_inputs['in_v_tnd'] = self._in_v_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

	def _stencil_defs(self, dt, in_s, in_u, in_v, in_su, in_sv,
					  in_sqv=None, in_sqc=None, in_sqr=None,
					  in_u_tnd=None, in_v_tnd=None,
					  in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the prognostic conservative variables
		via the forward Euler scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density
			at the current time level.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity
			at the current time level.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity
			at the current time level.
		in_su : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum
			at the current time level.
		in_sv : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum
			at the current time level.
		in_sqv : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			of water vapor at the current time level.
		in_sqc : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			mass of cloud water at the current time level.
		in_sqr : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density
			mass of precipitation water at the current time level.
		in_u_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the :math:`x`-velocity.
		in_v_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the :math:`y`-velocity.
		in_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the mass fraction of water vapor.
		in_qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the mass fraction of cloud liquid water.
		in_qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the parameterized tendency
			of the mass fraction of precipitation water.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped
			isentropic density.
		out_su : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`x`-momentum.
		out_sv : obj
			:class:`gridtools.Equation` representing the provisional
			:math:`y`-momentum.
		out_sqv : `obj`, optional
			:class:`gridtools.Equation` representing the stepped
			isentropic density of water vapour.
		out_sqc : `obj`, optional
			:class:`gridtools.Equation` representing the stepped
			isentropic density of cloud water.
		out_sqr : `obj`, optional
			:class:`gridtools.Equation` representing the stepped
			isentropic density of precipitation water.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_su, in_sv,
							u_tnd=in_u_tnd, v_tnd=in_v_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v, in_su, in_sv, in_sqv, in_sqc, in_sqr,
							in_u_tnd, in_v_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		out_s[i, j] = in_s[i, j] - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
										 (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)

		# Advance the x-momentum
		if in_u_tnd is None:
			out_su[i, j] = in_su[i, j] - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
											   (flux_su_y[i, j] - flux_su_y[i, j-1]) / dy)
		else:
			out_su[i, j] = in_su[i, j] - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
											   (flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
											   in_s[i, j] * in_u_tnd[i, j])

		# Advance the y-momentum
		if in_v_tnd is None:
			out_sv[i, j] = in_sv[i, j] - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
											   (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy)
		else:
			out_sv[i, j] = in_sv[i, j] - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
											   (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
											   in_s[i, j] * in_u_tnd[i, j])

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv[i, j] - \
								dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv[i, j] = in_sqv[i, j] - \
								dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
									  in_s[i, j] * in_qv_tnd[i, j])

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc[i, j] - \
								dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc[i, j] = in_sqc[i, j] - \
								dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
									  in_s[i, j] * in_qc_tnd[i, j])

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr[i, j] - \
								dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr[i, j] = in_sqr[i, j] - \
								dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
									  in_s[i, j] * in_qr_tnd[i, j])

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


class _RK2(HomogeneousIsentropicPrognostic):
	"""
	This class inherits
	:class:`~tasmania.dynamics.homogeneous_prognostic_isentropic.HomogeneousPrognosticIsentropic`
	to implement the two-stages, second-order Runge-Kutta scheme which takes
	over the prognostic part of the three-dimensional, moist, homogeneous,
	isentropic dynamical core.
	"""
	def __init__(self, grid, moist_on, horizontal_boundary_conditions,
				 horizontal_flux_scheme, backend, dtype=datatype):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, horizontal_boundary_conditions,
						 horizontal_flux_scheme, backend, dtype)

		# Initialize the pointer to the underlying GT4Py stencil; it will
		# be properly re-directed the first time the call operator is invoked
		self._stencil = None

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 2

	def __call__(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil is None:
			self._stencil_initialize(raw_state, raw_tendencies)

		# Update the attributes which serve as inputs to the GT4Py stencil
		self._stencil_set_inputs(stage, dt, raw_state, raw_tendencies)

		# Step the prognostic variables
		self._stencil.compute()

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Bring the updated isentropic density back to the original dimensions,
		# and enforce the boundary conditions
		s_new = self._hboundary.from_computational_to_physical_domain(
			self._out_s[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(s_new,	raw_state['air_isentropic_density'])

		# Bring the updated momenta back to the original dimensions,
		# and enforce the boundary conditions
		# Note: let's forget about symmetric conditions...
		su_new = self._hboundary.from_computational_to_physical_domain(
			self._out_su[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(su_new, raw_state['x_momentum_isentropic'])
		sv_new = self._hboundary.from_computational_to_physical_domain(
			self._out_sv[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(sv_new, raw_state['y_momentum_isentropic'])

		# Bring the updated water constituents back to the original dimensions,
		# and enforce the boundary conditions
		if self._moist_on:
			sqv_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqv[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqv_new,
									raw_state['isentropic_density_of_water_vapor'])

			sqc_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqc[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqc_new,
									raw_state['isentropic_density_of_cloud_liquid_water'])

			sqr_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqr[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqr_new,
									raw_state['isentropic_density_of_precipitation_water'])

		# Instantiate the output state
		raw_state_new = {
			'air_isentropic_density': s_new,
			'x_momentum_isentropic': su_new,
			'y_momentum_isentropic': sv_new,
		}
		if self._moist_on:
			raw_state_new['isentropic_density_of_water_vapor'] = sqv_new
			raw_state_new['isentropic_density_of_cloud_liquid_water'] = sqc_new
			raw_state_new['isentropic_density_of_precipitation_water'] = sqr_new

		return raw_state_new

	def _stencil_initialize(self, raw_state, raw_tendencies):
		"""
		Initialize the GT4Py stencil implementing any stage of the three-steps
		Runge-Kutta time-integration scheme	to step the solution by neglecting
		vertical advection.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencil_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store the output fields
		self._stencil_allocate_outputs()

		# Set the stencil's computational domain and backend
		nz, nb = self._grid.nz, self.nb
		mi, mj = self._hboundary.mi, self._hboundary.mj
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0),
									  (nb + ni - 1, nb + nj - 1, nk - 1))
		_mode = self._backend

		# Set the stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_s_int': self._in_s_int,
				   'in_u_int': self._in_u, 'in_v_int': self._in_v,
				   'in_su': self._in_su, 'in_su_int': self._in_su_int,
				   'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int}
		_outputs = {'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
							'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
							'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int})
			_outputs.update({'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
							 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('x_velocity', None) is not None:
				_inputs['in_u_tnd'] = self._in_u_tnd
			if raw_tendencies.get('y_velocity', None) is not None:
				_inputs['in_v_tnd'] = self._in_v_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

	def _stencil_allocate_inputs(self, raw_tendencies):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Call parent method
		super()._stencil_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store the intermediate values
		# for the prognostic variables
		self._in_s_int	= np.zeros((mi, mj, nz), dtype=dtype)
		self._in_su_int = np.zeros((mi, mj, nz), dtype=dtype)
		self._in_sv_int = np.zeros((mi, mj, nz), dtype=dtype)
		if self._moist_on:
			self._in_sqv_int = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqc_int = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqr_int = np.zeros((mi, mj, nz), dtype=dtype)

	def _stencil_set_inputs(self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		mi, mj = self._hboundary.mi, self._hboundary.mj
		if raw_tendencies is not None:
			u_tnd_on  = raw_tendencies.get('x_velocity', None) is not None
			v_tnd_on  = raw_tendencies.get('y_velocity', None) is not None
			qv_tnd_on = raw_tendencies.get(mf_wv, None) is not None
			qc_tnd_on = raw_tendencies.get(mf_clw, None) is not None
			qr_tnd_on = raw_tendencies.get(mf_pw, None) is not None
		else:
			u_tnd_on = v_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# Update the local time step
		self._dt.value = (1./2. + 1./2.*(stage > 0)) * dt.total_seconds()

		if stage == 0:
			# Extract the Numpy arrays representing the current solution
			s	= raw_state['air_isentropic_density']
			su	= raw_state['x_momentum_isentropic']
			sv	= raw_state['y_momentum_isentropic']
			if self._moist_on:
				sqv = raw_state['isentropic_density_of_water_vapor']
				sqc = raw_state['isentropic_density_of_cloud_liquid_water']
				sqr = raw_state['isentropic_density_of_precipitation_water']

			# Update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s)
			self._in_su[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su)
			self._in_sv[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv)
			if self._moist_on:
				self._in_sqv[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqv)
				self._in_sqc[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqc)
				self._in_sqr[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqr)

		# Extract the Numpy arrays representing the intermediate solution
		s	= raw_state['air_isentropic_density']
		u	= raw_state['x_velocity_at_u_locations']
		v	= raw_state['y_velocity_at_v_locations']
		su	= raw_state['x_momentum_isentropic']
		sv	= raw_state['y_momentum_isentropic']
		if self._moist_on:
			sqv = raw_state['isentropic_density_of_water_vapor']
			sqc = raw_state['isentropic_density_of_cloud_liquid_water']
			sqr = raw_state['isentropic_density_of_precipitation_water']

		# Update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._in_s_int[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(s)
		self._in_u[:mi+1, :mj, :] = self._hboundary.from_physical_to_computational_domain(u)
		self._in_v[:mi, :mj+1, :] = self._hboundary.from_physical_to_computational_domain(v)
		self._in_su_int[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(su)
		self._in_sv_int[:mi, :mj, :] = self._hboundary.from_physical_to_computational_domain(sv)
		if self._moist_on:
			self._in_sqv_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqv)
			self._in_sqc_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqc)
			self._in_sqr_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqr)

		# Extract the Numpy arrays representing the provided tendencies,
		# and update the Numpy arrays which serve as inputs to the GT4Py stencils
		if u_tnd_on:
			u_tnd = raw_tendencies['x_velocity']
			self._in_u_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(u_tnd)
		if v_tnd_on:
			v_tnd = raw_tendencies['y_velocity']
			self._in_v_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(v_tnd)
		if qv_tnd_on:
			qv_tnd = raw_tendencies[mf_wv]
			self._in_qv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qv_tnd)
		if qc_tnd_on:
			qc_tnd = raw_tendencies[mf_clw]
			self._in_qc_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qc_tnd)
		if qr_tnd_on:
			qr_tnd = raw_tendencies[mf_pw]
			self._in_qr_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qr_tnd)

	def _stencil_defs(self, dt, in_s, in_s_int, in_u_int, in_v_int,
					  in_su, in_su_int, in_sv, in_sv_int,
					  in_sqv=None, in_sqv_int=None,
					  in_sqc=None, in_sqc_int=None,
					  in_sqr=None, in_sqr_int=None,
					  in_u_tnd=None, in_v_tnd=None,
					  in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the conservative prognostic variables
		via a stage of the Runge-Kutta scheme.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int,
							in_su_int, in_sv_int, u_tnd=in_u_tnd, v_tnd=in_v_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int,
							in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
							in_u_tnd, in_v_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		out_s[i, j] = in_s[i, j] - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
										 (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)

		# Advance the x-momentum
		if in_u_tnd is None:
			out_su[i, j] = in_su[i, j] - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
											   (flux_su_y[i, j] - flux_su_y[i, j-1]) / dy)
		else:
			out_su[i, j] = in_su[i, j] - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
											   (flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
											   in_s_int[i, j] * in_u_tnd[i, j])

		# Advance the y-momentum
		if in_v_tnd is None:
			out_sv[i, j] = in_sv[i, j] - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
											   (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy)
		else:
			out_sv[i, j] = in_sv[i, j] - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
											   (flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
											   in_s_int[i, j] * in_v_tnd[i, j])

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv[i, j] - \
								dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv[i, j] = in_sqv[i, j] - \
								dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
									  (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
									  in_s_int[i, j] * in_qv_tnd[i, j])

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc[i, j] - \
								dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc[i, j] = in_sqc[i, j] - \
								dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
									  (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
									  in_s_int[i, j] * in_qc_tnd[i, j])

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr[i, j] - \
								dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr[i, j] = in_sqr[i, j] - \
								dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
									  (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
									  in_s_int[i, j] * in_qr_tnd[i, j])

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


class _RK3COSMO(_RK2):
	"""
	This class inherits
	:class:`~tasmania.dynamics._homogeneous_prognostic_isentropic._RK2`
	to implement the three-stages, second-order Runge-Kutta scheme
	which takes over the prognostic part of the three-dimensional,
	moist, homogeneous, isentropic dynamical core.
	"""
	def __init__(self, grid, moist_on, horizontal_boundary_conditions,
				 horizontal_flux_scheme, backend, dtype=datatype):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, horizontal_boundary_conditions,
						 horizontal_flux_scheme, backend, dtype)

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 3

	def _stencil_set_inputs(self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Call parent's method
		super()._stencil_set_inputs(stage, dt, raw_state, raw_tendencies)

		# Update the local time step
		self._dt.value = (1./3. + 1./6.*(stage > 0) + 1./2.*(stage > 1)) * dt.total_seconds()


class _RK3(HomogeneousIsentropicPrognostic):
	"""
	This class inherits
	:class:`~tasmania.dynamics.homogeneous_prognostic_isentropic.HomogeneousPrognosticIsentropic`
	to implement the three-stages, third-order Runge-Kutta scheme
	which takes over the prognostic part of the three-dimensional,
	moist, homogeneous, isentropic dynamical core.
	"""
	def __init__(self, grid, moist_on, horizontal_boundary_conditions,
				 horizontal_flux_scheme, backend=gt.mode.NUMPY, dtype=datatype):
		"""
		Constructor.
		"""
		super().__init__(grid, moist_on, horizontal_boundary_conditions,
						 horizontal_flux_scheme, backend, dtype)

		# Initialize the pointers to the underlying GT4Py stencils; they
		# will be re-directed when the call operator is invoked for the first time
		self._stencil_first = None
		self._stencil_second = None
		self._stencil_third = None

		# Free parameters for RK3
		self._alpha1 = 1./2.
		self._alpha2 = 3./4.

		# Set the other parameters yielding a third-order method
		self._gamma1 = (3.*self._alpha2 - 2.) / \
					   (6. * self._alpha1 * (self._alpha2 - self._alpha1))
		self._gamma2 = (3.*self._alpha1 - 2.) / \
					   (6. * self._alpha2 * (self._alpha1 - self._alpha2))
		self._gamma0 = 1. - self._gamma1 - self._gamma2
		self._beta21 = self._alpha2 - 1. / (6. * self._alpha1 * self._gamma2)

	@property
	def stages(self):
		"""
		Get the number of stages carried out by the time-integration scheme.
		"""
		return 3

	def __call__(self, stage, dt, raw_state, raw_tendencies=None):
		"""
		Method advancing the conservative, prognostic model variables
		one stage forward in time. Only horizontal derivatives are considered;
		possible vertical derivatives are disregarded.
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil_first is None:
			self._stencils_initialize(raw_state, raw_tendencies)

		# Update the attributes which serve as inputs to the GT4Py stencil
		self._stencil_set_inputs(stage, dt, raw_state, raw_tendencies)

		# Run the compute function of the stencil stepping the solution
		self._stencil_compute(stage)

		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj

		# Bring the updated isentropic density back to the original dimensions,
		# and enforce the boundary conditions
		s_new = self._hboundary.from_computational_to_physical_domain(
			self._out_s[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(s_new,	raw_state['air_isentropic_density'])

		# Bring the updated x-momentum back to the original dimensions,
		# and enforce the boundary conditions
		# Note: let's forget about symmetric conditions...
		su_new = self._hboundary.from_computational_to_physical_domain(
			self._out_su[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(su_new, raw_state['x_momentum_isentropic'])

		# Bring the updated y-momentum back to the original dimensions,
		# and enforce the boundary conditions
		# Note: let's forget about symmetric conditions...
		sv_new = self._hboundary.from_computational_to_physical_domain(
			self._out_sv[:mi, :mj, :], (nx, ny, nz))
		self._hboundary.enforce(sv_new, raw_state['y_momentum_isentropic'])

		# Bring the updated water constituents back to the original dimensions,
		# and enforce the boundary conditions
		if self._moist_on:
			sqv_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqv[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqv_new,
									raw_state['isentropic_density_of_water_vapor'])

			sqc_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqc[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqc_new,
									raw_state['isentropic_density_of_cloud_liquid_water'])

			sqr_new = self._hboundary.from_computational_to_physical_domain(
				self._out_sqr[:mi, :mj, :], (nx, ny, nz))
			self._hboundary.enforce(sqr_new,
									raw_state['isentropic_density_of_precipitation_water'])

		# Instantiate the output state
		raw_state_new = {
			'air_isentropic_density': s_new,
			'x_momentum_isentropic': su_new,
			'y_momentum_isentropic': sv_new,
		}
		if self._moist_on:
			raw_state_new['isentropic_density_of_water_vapor'] = sqv_new
			raw_state_new['isentropic_density_of_cloud_liquid_water'] = sqc_new
			raw_state_new['isentropic_density_of_precipitation_water'] = sqr_new

		return raw_state_new

	def _stencils_initialize(self, raw_state, raw_tendencies):
		"""
		Initialize the GT4Py stencils implementing the RK3 scheme.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencils_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store temporary fields to be
		# shared across the different stencils which step the solution
		# disregarding any vertical motion
		self._stencils_allocate_temporaries()

		# Allocate the Numpy arrays which will store the output fields
		self._stencil_allocate_outputs()

		# Set the stencils' computational domain and backend
		nz, nb = self._grid.nz, self.nb
		mi, mj = self._hboundary.mi, self._hboundary.mj
		ni, nj, nk = mi - 2 * nb, mj - 2 * nb, nz
		_domain = gt.domain.Rectangle((nb, nb, 0),
									  (nb + ni - 1, nb + nj - 1, nk - 1))
		_mode = self._backend

		# Set the first stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
				   'in_su': self._in_su, 'in_sv': self._in_sv}
		_outputs = {'out_s0': self._s0, 'out_s': self._out_s,
					'out_su0': self._su0, 'out_su': self._out_su,
					'out_sv0': self._sv0, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
							'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
							'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int})
			_outputs.update({'out_sqv0': self._sqv0, 'out_sqv': self._out_sqv,
							 'out_sqc0': self._sqc0, 'out_sqc': self._out_sqc,
							 'out_sqr0': self._sqr0, 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('x_velocity', None) is not None:
				_inputs['in_u_tnd'] = self._in_u_tnd
			if raw_tendencies.get('y_velocity', None) is not None:
				_inputs['in_v_tnd'] = self._in_v_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# Instantiate the first stencil
		self._stencil_first = gt.NGStencil(
			definitions_func = self._stencil_first_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

		# Set the second stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_s_int': self._in_s_int,
				   'in_u_int': self._in_u, 'in_v_int': self._in_v,
				   'in_su': self._in_su, 'in_su_int': self._in_su_int,
				   'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int,
				   'in_s0': self._s0, 'in_su0': self._su0, 'in_sv0': self._sv0}
		_outputs = {'out_s1': self._s1, 'out_s': self._out_s,
					'out_su1': self._su1, 'out_su': self._out_su,
					'out_sv1': self._sv1, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
							'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
							'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int,
							'in_sqv0': self._sqv0, 'in_sqc0': self._sqc0,
							'in_sqr0': self._sqr0})
			_outputs.update({'out_sqv1': self._sqv1, 'out_sqv': self._out_sqv,
							 'out_sqc1': self._sqc1, 'out_sqc': self._out_sqc,
							 'out_sqr1': self._sqr1, 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('x_velocity', None) is not None:
				_inputs['in_u_tnd'] = self._in_u_tnd
			if raw_tendencies.get('y_velocity', None) is not None:
				_inputs['in_v_tnd'] = self._in_v_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# Instantiate the second stencil
		self._stencil_second = gt.NGStencil(
			definitions_func = self._stencil_second_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

		# Set the third stencil's inputs and outputs
		_inputs = {'in_s': self._in_s, 'in_s_int': self._in_s_int,
				   'in_u_int': self._in_u, 'in_v_int': self._in_v,
				   'in_su': self._in_su, 'in_su_int': self._in_su_int,
				   'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int,
				   'in_s0': self._s0, 'in_su0': self._su0, 'in_sv0': self._sv0,
				   'in_s1': self._s1, 'in_su1': self._su1, 'in_sv1': self._sv1}
		_outputs = {'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv}
		if self._moist_on:
			_inputs.update({'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
							'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
							'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int,
							'in_sqv0': self._sqv0, 'in_sqc0': self._sqc0,
							'in_sqr0': self._sqr0, 'in_sqv1': self._sqv1,
							'in_sqc1': self._sqc1, 'in_sqr1': self._sqr1})
			_outputs.update({'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
							 'out_sqr': self._out_sqr})
		if raw_tendencies is not None:
			if raw_tendencies.get('x_velocity', None) is not None:
				_inputs['in_u_tnd'] = self._in_u_tnd
			if raw_tendencies.get('y_velocity', None) is not None:
				_inputs['in_v_tnd'] = self._in_v_tnd
			if raw_tendencies.get(mf_wv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if raw_tendencies.get(mf_clw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if raw_tendencies.get(mf_pw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# Instantiate the third stencil
		self._stencil_third = gt.NGStencil(
			definitions_func = self._stencil_third_defs,
			inputs			 = _inputs,
			global_inputs	 = {'dt': self._dt},
			outputs			 = _outputs,
			domain			 = _domain,
			mode			 = _mode,
		)

	def _stencils_allocate_inputs(self, raw_tendencies):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencils.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Call parent method
		super()._stencil_allocate_inputs(raw_tendencies)

		# Allocate the Numpy arrays which will store the intermediate values
		# for the prognostic variables
		self._in_s_int	= np.zeros((mi, mj, nz), dtype=dtype)
		self._in_su_int = np.zeros((mi, mj, nz), dtype=dtype)
		self._in_sv_int = np.zeros((mi, mj, nz), dtype=dtype)
		if self._moist_on:
			self._in_sqv_int = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqc_int = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqr_int = np.zeros((mi, mj, nz), dtype=dtype)

	def _stencils_allocate_temporaries(self):
		"""
		Allocate the Numpy arrays which store temporary fields to be shared
		across the different GT4Py.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Allocate the Numpy arrays which will store the first increment
		# for all prognostic variables
		self._s0  = np.zeros((mi, mj, nz), dtype=dtype)
		self._su0 = np.zeros((mi, mj, nz), dtype=dtype)
		self._sv0 = np.zeros((mi, mj, nz), dtype=dtype)
		if self._moist_on:
			self._sqv0 = np.zeros((mi, mj, nz), dtype=dtype)
			self._sqc0 = np.zeros((mi, mj, nz), dtype=dtype)
			self._sqr0 = np.zeros((mi, mj, nz), dtype=dtype)

		# Allocate the Numpy arrays which will store the second increment
		# for all prognostic variables
		self._s1  = np.zeros((mi, mj, nz), dtype=dtype)
		self._su1 = np.zeros((mi, mj, nz), dtype=dtype)
		self._sv1 = np.zeros((mi, mj, nz), dtype=dtype)
		if self._moist_on:
			self._sqv1 = np.zeros((mi, mj, nz), dtype=dtype)
			self._sqc1 = np.zeros((mi, mj, nz), dtype=dtype)
			self._sqr1 = np.zeros((mi, mj, nz), dtype=dtype)

	def _stencil_set_inputs(self, stage, dt, raw_state, raw_tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which step the solution disregarding any vertical motion.
		"""
		# Shortcuts
		mi, mj = self._hboundary.mi, self._hboundary.mj
		if raw_tendencies is not None:
			u_tnd_on  = raw_tendencies.get('x_velocity', None) is not None
			v_tnd_on  = raw_tendencies.get('y_velocity', None) is not None
			qv_tnd_on = raw_tendencies.get(mf_wv, None) is not None
			qc_tnd_on = raw_tendencies.get(mf_clw, None) is not None
			qr_tnd_on = raw_tendencies.get(mf_pw, None) is not None
		else:
			u_tnd_on = v_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# Update the local time step
		self._dt.value = dt.total_seconds()

		# Extract the Numpy arrays representing the intermediate solution
		s	= raw_state['air_isentropic_density']
		u	= raw_state['x_velocity_at_u_locations']
		v	= raw_state['y_velocity_at_v_locations']
		su	= raw_state['x_momentum_isentropic']
		sv	= raw_state['y_momentum_isentropic']
		if self._moist_on:
			sqv = raw_state['isentropic_density_of_water_vapor']
			sqc = raw_state['isentropic_density_of_cloud_liquid_water']
			sqr = raw_state['isentropic_density_of_precipitation_water']

		if stage == 0:
			# Update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s)
			self._in_u[:mi+1, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(u)
			self._in_v[:mi, :mj+1, :] = \
				self._hboundary.from_physical_to_computational_domain(v)
			self._in_su[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su)
			self._in_sv[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv)
			if self._moist_on:
				self._in_sqv[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqv)
				self._in_sqc[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqc)
				self._in_sqr[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqr)
		else:

			# Update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s)
			self._in_u[:mi+1, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(u)
			self._in_v[:mi, :mj+1, :] = \
				self._hboundary.from_physical_to_computational_domain(v)
			self._in_su_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su)
			self._in_sv_int[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv)
			if self._moist_on:
				self._in_sqv_int[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqv)
				self._in_sqc_int[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqc)
				self._in_sqr_int[:mi, :mj, :] = \
					self._hboundary.from_physical_to_computational_domain(sqr)

		# Extract the Numpy arrays representing the provided tendencies,
		# and update the Numpy arrays which serve as inputs to the GT4Py stencils
		if u_tnd_on:
			u_tnd = raw_tendencies['x_velocity']
			self._in_u_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(u_tnd)
		if v_tnd_on:
			v_tnd = raw_tendencies['y_velocity']
			self._in_v_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(v_tnd)
		if qv_tnd_on:
			qv_tnd = raw_tendencies[mf_wv]
			self._in_qv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qv_tnd)
		if qc_tnd_on:
			qc_tnd = raw_tendencies[mf_clw]
			self._in_qc_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qc_tnd)
		if qr_tnd_on:
			qr_tnd = raw_tendencies[mf_pw]
			self._in_qr_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qr_tnd)

	def _stencil_compute(self, stage):
		if stage == 0:
			self._stencil_first.compute()
		elif stage == 1:
			self._stencil_second.compute()
		else:
			self._stencil_third.compute()

	def _stencil_first_defs(
		self, dt, in_s, in_u, in_v, in_su, in_sv, in_sqv=None, in_sqc=None, in_sqr=None,
		in_u_tnd=None, in_v_tnd=None, in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the prognostic variables via the first stage
		of the RK3 scheme.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		a1 = self._alpha1

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate fields representing the first increments
		out_s0  = gt.Equation()
		out_su0 = gt.Equation()
		out_sv0 = gt.Equation()
		if self._moist_on:
			out_sqv0 = gt.Equation()
			out_sqc0 = gt.Equation()
			out_sqr0 = gt.Equation()

		# Instantiate fields representing the output state
		out_s  = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v,
							in_su, in_sv, u_tnd=in_u_tnd, v_tnd=in_v_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s, in_u, in_v,
							in_su, in_sv, in_sqv, in_sqc, in_sqr,
							in_u_tnd, in_v_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		out_s0[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
							   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		out_s[i, j] = in_s[i, j] + a1 * out_s0[i, j]

		# Advance the x-momentum
		if in_u_tnd is None:
			out_su0[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy)
		else:
			out_su0[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
									in_s[i, j] * in_u_tnd[i, j])
		out_su[i, j] = in_su[i, j] + a1 * out_su0[i, j]

		# Advance the y-momentum
		if in_v_tnd is None:
			out_sv0[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy)
		else:
			out_sv0[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
									in_s[i, j] * in_v_tnd[i, j])
		out_sv[i, j] = in_sv[i, j] + a1 * out_sv0[i, j]

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv0[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv0[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
										 in_s[i, j] * in_qv_tnd[i, j])
			out_sqv[i, j] = in_sqv[i, j] + a1 * out_sqv0[i, j]

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc0[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc0[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
										 in_s[i, j] * in_qc_tnd[i, j])
			out_sqc[i, j] = in_sqc[i, j] + a1 * out_sqc0[i, j]

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr0[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr0[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
										 in_s[i, j] * in_qr_tnd[i, j])
			out_sqr[i, j] = in_sqr[i, j] + a1 * out_sqr0[i, j]

		if not self._moist_on:
			return out_s0, out_s, out_su0, out_su, out_sv0, out_sv
		else:
			return out_s0, out_s, out_su0, out_su, out_sv0, out_sv, \
				   out_sqv0, out_sqv, out_sqc0, out_sqc, out_sqr0, out_sqr

	def _stencil_second_defs(
		self, dt, in_s, in_s_int, in_s0, in_u_int, in_v_int,
		in_su, in_su_int, in_su0, in_sv, in_sv_int, in_sv0,
		in_sqv=None, in_sqv_int=None, in_sqv0=None,
		in_sqc=None, in_sqc_int=None, in_sqc0=None,
		in_sqr=None, in_sqr_int=None, in_sqr0=None,
		in_u_tnd=None, in_v_tnd=None, in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the prognostic variables via the second stage
		of the RK3 scheme.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		a2, b21 = self._alpha2, self._beta21

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate fields representing the first increments
		out_s1  = gt.Equation()
		out_su1 = gt.Equation()
		out_sv1 = gt.Equation()
		if self._moist_on:
			out_sqv1 = gt.Equation()
			out_sqc1 = gt.Equation()
			out_sqr1 = gt.Equation()

		# Instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int,
							in_su_int, in_sv_int, u_tnd=in_u_tnd, v_tnd=in_v_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int,
							in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
							in_u_tnd, in_v_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		out_s1[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
							   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		out_s[i, j] = in_s[i, j] + b21 * in_s0[i, j] + (a2 - b21) * out_s1[i, j]

		# Advance the x-momentum
		if in_u_tnd is None:
			out_su1[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy)
		else:
			out_su1[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
									in_s_int[i, j] * in_u_tnd[i, j])
		out_su[i, j] = in_su[i, j] + b21 * in_su0[i, j] + (a2 - b21) * out_su1[i, j]

		# Advance the y-momentum
		if in_v_tnd is None:
			out_sv1[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy)
		else:
			out_sv1[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
									in_s_int[i, j] * in_v_tnd[i, j])
		out_sv[i, j] = in_sv[i, j] + b21 * in_sv0[i, j] + (a2 - b21) * out_sv1[i, j]

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv1[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				out_sqv1[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
										 in_s_int[i, j] * in_qv_tnd[i, j])
			out_sqv[i, j] = in_sqv[i, j] + b21 * in_sqv0[i, j] + (a2 - b21) * out_sqv1[i, j]

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc1[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				out_sqc1[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
										 in_s_int[i, j] * in_qc_tnd[i, j])
			out_sqc[i, j] = in_sqc[i, j] + b21 * in_sqc0[i, j] + (a2 - b21) * out_sqc1[i, j]

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr1[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				out_sqr1[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
										 in_s[i, j] * in_qr_tnd[i, j])
			out_sqr[i, j] = in_sqr[i, j] + b21 * in_sqr0[i, j] + (a2 - b21) * out_sqr1[i, j]

		if not self._moist_on:
			return out_s1, out_s, out_su1, out_su, out_sv1, out_sv
		else:
			return out_s1, out_s, out_su1, out_su, out_sv1, out_sv, \
				   out_sqv1, out_sqv, out_sqc1, out_sqc, out_sqr1, out_sqr

	def _stencil_third_defs(
		self, dt, in_s, in_s_int, in_s0, in_s1, in_u_int, in_v_int,
		in_su, in_su_int, in_su0, in_su1, in_sv, in_sv_int, in_sv0, in_sv1,
		in_sqv=None, in_sqv_int=None, in_sqv0=None, in_sqv1=None,
		in_sqc=None, in_sqc_int=None, in_sqc0=None, in_sqc1=None,
		in_sqr=None, in_sqr_int=None, in_sqr0=None, in_sqr1=None,
		in_u_tnd=None, in_v_tnd=None, in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None):
		"""
		GT4Py stencil stepping the prognostic variables via the second stage
		of the RK3 scheme.
		"""
		# Shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		g0, g1, g2 = self._gamma0, self._gamma1, self._gamma2

		# Declare indices
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Instantiate fields representing the first increments
		tmp_s2 = gt.Equation()
		tmp_su2 = gt.Equation()
		tmp_sv2 = gt.Equation()
		if self._moist_on:
			tmp_sqv2 = gt.Equation()
			tmp_sqc2 = gt.Equation()
			tmp_sqr2 = gt.Equation()

		# Instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist_on:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# Calculate the fluxes
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int,
							in_su_int, in_sv_int, u_tnd=in_u_tnd, v_tnd=in_v_tnd)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
			flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(i, j, k, dt, in_s_int, in_u_int, in_v_int,
							in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
							in_u_tnd, in_v_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd)

		# Advance the isentropic density
		tmp_s2[i, j] = - dt * ((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
							   (flux_s_y[i, j] - flux_s_y[i, j-1]) / dy)
		out_s[i, j] = in_s[i, j] + g0 * in_s0[i, j] \
					  + g1 * in_s1[i, j] \
					  + g2 * tmp_s2[i, j]

		# Advance the x-momentum
		if in_u_tnd is None:
			tmp_su2[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy)
		else:
			tmp_su2[i, j] = - dt * ((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
									(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
									in_s_int[i, j] * in_u_tnd[i, j])
		out_su[i, j] = in_su[i, j] + g0 * in_su0[i, j] \
					   + g1 * in_su1[i, j] \
					   + g2 * tmp_su2[i, j]

		# Advance the y-momentum
		if in_v_tnd is None:
			tmp_sv2[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy)
		else:
			tmp_sv2[i, j] = - dt * ((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
									(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
									in_s_int[i, j] * in_v_tnd[i, j])
		out_sv[i, j] = in_sv[i, j] + g0 * in_sv0[i, j] \
					   + g1 * in_sv1[i, j] \
					   + g2 * tmp_sv2[i, j]

		if self._moist_on:
			# Advance the isentropic density of water vapor
			if in_qv_tnd is None:
				tmp_sqv2[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy)
			else:
				tmp_sqv2[i, j] = - dt * ((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
										 (flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
										 in_s_int[i, j] * in_qv_tnd[i, j])
			out_sqv[i, j] = in_sqv[i, j] + g0 * in_sqv0[i, j] \
							+ g1 * in_sqv1[i, j] \
							+ g2 * tmp_sqv2[i, j]

			# Advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				tmp_sqc2[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy)
			else:
				tmp_sqc2[i, j] = - dt * ((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
										 (flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
										 in_s_int[i, j] * in_qc_tnd[i, j])
			out_sqc[i, j] = in_sqc[i, j] + g0 * in_sqc0[i, j] \
							+ g1 * in_sqc1[i, j] \
							+ g2 * tmp_sqc2[i, j]

			# Advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				tmp_sqr2[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy)
			else:
				tmp_sqr2[i, j] = - dt * ((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
										 (flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
										 in_s[i, j] * in_qr_tnd[i, j])
			out_sqr[i, j] = in_sqr[i, j] + g0 * in_sqr0[i, j] \
							+ g1 * in_sqr1[i, j] \
							+ g2 * tmp_sqr2[i, j]

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr
