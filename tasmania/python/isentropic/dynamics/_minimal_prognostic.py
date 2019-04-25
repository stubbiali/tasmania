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
	Centered(IsentropicMinimalPrognostic)
	ForwardEuler(IsentropicMinimalPrognostic)
	RK2(IsentropicMinimalPrognostic)
	RK3WS(RK2)
	RK3(IsentropicMinimalPrognostic)
"""
import numpy as np
import warnings

import gridtools as gt
from tasmania.python.isentropic.dynamics.minimal_prognostic \
	import IsentropicMinimalPrognostic

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


# convenient aliases
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


class Centered(IsentropicMinimalPrognostic):
	"""
	Implementation of a centered time-integration scheme which
	takes over the prognostic part of the three-dimensional, moist,
	*minimal*, isentropic dynamical core.
	"""
	def __init__(
		self, horizontal_flux_scheme, mode, grid, hb, moist,
		substeps, backend, dtype=datatype
	):
		super().__init__(
			horizontal_flux_scheme, mode, grid, hb, moist,
			substeps, backend, dtype
		)

		# initialize the pointer to the underlying GT4Py stencil in charge
		# of carrying out the stages
		self._stage_stencil = None

		# boolean flag to quickly assess whether we are within the first time step
		self._is_first_timestep = True

	@property
	def stages(self):
		return 1

	@property
	def substep_fractions(self):
		return 1.0

	def stage_call(self, stage, timestep, state, tendencies=None):
		# the first time this method is invoked, initialize the GT4Py stencil
		if self._stage_stencil is None:
			self._stage_stencil_initialize(tendencies)

		# update the attributes which serve as inputs to the GT4Py stencil
		self._stage_stencil_set_inputs(stage, timestep, state, tendencies)

		# step the prognostic variables
		self._stage_stencil.compute()

		# instantiate the output state
		state_new = {
			'time': state['time'] + timestep,
			'air_isentropic_density': self._out_s,
			'x_momentum_isentropic': self._out_su,
			'y_momentum_isentropic': self._out_sv,
		}
		if self._moist:
			state_new.update({
				'isentropic_density_of_water_vapor': self._out_sqv,
				'isentropic_density_of_cloud_liquid_water': self._out_sqc,
				'isentropic_density_of_precipitation_water': self._out_sqr,
			})

		# keep track of the current state for the next time step
		self._in_s_old[...]  = self._in_s[...]
		self._in_su_old[...] = self._in_su[...]
		self._in_sv_old[...] = self._in_sv[...]
		if self._moist:
			self._in_sqv_old[...] = self._in_sqv[...]
			self._in_sqc_old[...] = self._in_sqc[...]
			self._in_sqr_old[...] = self._in_sqr[...]

		# at this point, the first time step is surely over...
		self._is_first_timestep = False

		return state_new

	def _stage_stencil_initialize(self, tendencies):
		# allocate the attributes which will serve as inputs to the stencil
		self._stage_stencil_allocate_inputs(tendencies)

		# allocate the Numpy arrays which will store the output fields
		self._stage_stencil_allocate_outputs()

		# set the stencil's inputs and outputs
		_inputs = {
			'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
			'in_su': self._in_su, 'in_sv': self._in_sv,
			'in_s_old': self._in_s_old, 'in_su_old': self._in_su_old,
			'in_sv_old': self._in_sv_old
		}
		_outputs = {
			'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv
		}
		if self._moist:
			_inputs.update({
				'in_sqv': self._in_sqv, 'in_sqv_old': self._in_sqv_old,
				'in_sqc': self._in_sqc, 'in_sqc_old': self._in_sqc_old,
				'in_sqr': self._in_sqr, 'in_sqr_old': self._in_sqr_old,
			})
			_outputs.update({
				'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
				'out_sqr': self._out_sqr,
			})
		if tendencies is not None:
			if tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd
			if tendencies.get(mfwv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if tendencies.get(mfcw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if tendencies.get(mfpw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# set the stencil's computational domain
		# note: here, _mode attribute is ineffective
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._hb.nb
		_domain = gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1))

		# instantiate the stencil
		self._stage_stencil = gt.NGStencil(
			definitions_func=self._stage_stencil_defs,
			inputs=_inputs,
			global_inputs={'dt': self._dt},
			outputs=_outputs,
			domain=_domain,
			mode=self._backend,
		)

	def _stage_stencil_allocate_inputs(self, tendencies):
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# instantiate a GT4Py Global representing the time step,
		# and the Numpy arrays which represent the solution and
		# the tendencies at the current time level
		super()._stage_stencil_allocate_inputs(tendencies)

		# allocate the Numpy arrays which represent the solution
		# at the previous time level
		self._in_s_old	= np.zeros((nx, ny, nz), dtype=dtype)
		self._in_su_old = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_sv_old = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._in_sqv_old = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_sqc_old = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_sqr_old = np.zeros((nx, ny, nz), dtype=dtype)

	def _stage_stencil_set_inputs(self, stage, timestep, state, tendencies):
		# update the time step, and the Numpy arrays representing
		# the solution and the tendencies at the current time level
		super()._stage_stencil_set_inputs(stage, timestep, state, tendencies)

		# at the first iteration, update the Numpy arrays representing
		# the solution at the previous time step
		if self._is_first_timestep:
			self._in_s_old[...]  = state['air_isentropic_density']
			self._in_su_old[...] = state['x_momentum_isentropic']
			self._in_sv_old[...] = state['y_momentum_isentropic']
			if self._moist:
				self._in_sqv_old[...] = state['isentropic_density_of_water_vapor']
				self._in_sqc_old[...] =	state['isentropic_density_of_cloud_liquid_water']
				self._in_sqr_old[...] = state['isentropic_density_of_precipitation_water']

	def _stage_stencil_defs(
		self, dt, in_s, in_u, in_v, in_su, in_sv,
		in_s_old, in_su_old, in_sv_old,
		in_sqv=None, in_sqc=None, in_sqr=None,
		in_sqv_old=None, in_sqc_old=None, in_sqr_old=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None
	):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()

		# declare the indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate the output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# calculate the horizontal fluxes
		if not self._moist:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(
					i, j, dt, in_s, in_u, in_v, in_su, in_sv,
					s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd
				)
		else:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(
					i, j, dt, in_s, in_u, in_v, in_su, in_sv, in_sqv, in_sqc, in_sqr,
					in_s_tnd, in_su_tnd, in_sv_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd
				)

		# advance the isentropic density
		if in_s_tnd is None:
			out_s[i, j] = in_s_old[i, j] - 2. * dt * (
				(flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
				(flux_s_y[i, j] - flux_s_y[i, j-1]) / dy
			)
		else:
			out_s[i, j] = in_s_old[i, j] - 2. * dt * (
				(flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
				(flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
				in_s_tnd[i, j]
			)

		# advance the x-momentum
		if in_su_tnd is None:
			out_su[i, j] = in_su_old[i, j] - 2. * dt * (
				(flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
				(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy
			)
		else:
			out_su[i, j] = in_su_old[i, j] - 2. * dt * (
				(flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
				(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy -
				in_su_tnd[i, j]
			)

		# advance the y-momentum
		if in_sv_tnd is None:
			out_sv[i, j] = in_sv_old[i, j] - 2. * dt * (
				(flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
				(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy
			)
		else:
			out_sv[i, j] = in_sv_old[i, j] - 2. * dt * (
				(flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
				(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy -
				in_sv_tnd[i, j]
			)

		if self._moist:
			# advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv_old[i, j] - 2. * dt * (
					(flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
					(flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy
				)
			else:
				out_sqv[i, j] = in_sqv_old[i, j] - 2. * dt * (
					(flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
					(flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
					in_s[i, j] * in_qv_tnd[i, j]
				)

			# advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc_old[i, j] - 2. * dt * (
					(flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
					(flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy
				)
			else:
				out_sqc[i, j] = in_sqc_old[i, j] - 2. * dt * (
					(flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
					(flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
					in_s[i, j] * in_qc_tnd[i, j]
				)

			# advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr_old[i, j] - 2. * dt * (
					(flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
					(flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy
				)
			else:
				out_sqr[i, j] = in_sqr_old[i, j] - 2. * dt * (
					(flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
					(flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
					in_s[i, j] * in_qr_tnd[i, j]
				)

		if not self._moist:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr

	def substep_call(
		self, stage, substep, dt, state, stage_state, tmp_state, tendencies=None
	):
		raise NotImplementedError()


class ForwardEuler(IsentropicMinimalPrognostic):
	"""
	Implementation of the forward Euler time-integration scheme which
	takes over the prognostic part of the three-dimensional, moist,
	*minimal*, isentropic dynamical core.
	"""
	def __init__(
		self, horizontal_flux_scheme, mode, grid, hb, moist,
		substeps, backend, dtype=datatype
	):
		super().__init__(
			horizontal_flux_scheme, mode, grid, hb, moist,
			substeps, backend, dtype
		)

		# initialize the pointer to the underlying GT4Py stencil in charge
		# of carrying out the stages
		self._stage_stencil = None

	@property
	def stages(self):
		return 1

	@property
	def substep_fractions(self):
		return 1.0

	def stage_call(self, stage, timestep, state, tendencies=None):
		# the first time this method is invoked, initialize the GT4Py stencil
		if self._stage_stencil is None:
			self._stage_stencil_initialize(tendencies)

		# update the attributes which serve as inputs to the GT4Py stencil
		self._stage_stencil_set_inputs(stage, timestep, state, tendencies)

		# step the prognostic variables
		self._stage_stencil.compute()

		# instantiate the output state
		state_new = {
			'time': state['time'] + timestep,
			'air_isentropic_density': self._out_s,
			'x_momentum_isentropic': self._out_su,
			'y_momentum_isentropic': self._out_sv,
		}
		if self._moist:
			state_new.update({
				'isentropic_density_of_water_vapor': self._out_sqv,
				'isentropic_density_of_cloud_liquid_water': self._out_sqc,
				'isentropic_density_of_precipitation_water': self._out_sqr,
			})

		return state_new

	def _stage_stencil_initialize(self, tendencies):
		# allocate the attributes which will serve as inputs to the stencil
		self._stage_stencil_allocate_inputs(tendencies)

		# allocate the Numpy arrays which will store the output fields
		self._stage_stencil_allocate_outputs()

		# set the stencil's computational domain
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._hb.nb
		if self._mode == 'x':
			_domain = gt.domain.Rectangle(
				(nb, 0, 0), (nx-nb-1, ny-1, nz-1)
			)
		elif self._mode == 'y':
			_domain = gt.domain.Rectangle(
				(0, nb, 0), (nx-1, ny-nb-1, nz-1)
			)
		else:
			_domain = gt.domain.Rectangle(
				(nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)
			)

		# set the stencil's inputs and outputs
		_inputs = {
			'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
			'in_su': self._in_su, 'in_sv': self._in_sv,
		}
		_outputs = {
			'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv
		}
		if self._moist:
			_inputs.update({
				'in_sqv': self._in_sqv, 'in_sqc': self._in_sqc, 'in_sqr': self._in_sqr,
			})
			_outputs.update({
				'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
				'out_sqr': self._out_sqr,
			})
		if tendencies is not None:
			if tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd
			if tendencies.get(mfwv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if tendencies.get(mfcw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if tendencies.get(mfpw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# instantiate the stencil
		self._stage_stencil = gt.NGStencil(
			definitions_func=self._stage_stencil_defs,
			inputs=_inputs,
			global_inputs={'dt': self._dt},
			outputs=_outputs,
			domain=_domain,
			mode=self._backend,
		)

	def _stage_stencil_defs(
		self, dt, in_s, in_u, in_v, in_su, in_sv,
		in_sqv=None, in_sqc=None, in_sqr=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None
	):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		md = self._mode

		# declare indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# calculate the fluxes
		if not self._moist:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(
					i, j, dt, in_s, in_u, in_v, in_su, in_sv,
					s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd
				)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(
					i, j, dt, in_s, in_u, in_v, in_su, in_sv, in_sqv, in_sqc, in_sqr,
					in_s_tnd, in_su_tnd, in_sv_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd
				)

		# advance the isentropic density
		if in_s_tnd is None:
			out_s[i, j] = in_s[i, j] - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_s[i, j] = in_s[i, j] - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0) -
				in_s_tnd[i, j]
			)

		# advance the x-momentum
		if in_su_tnd is None:
			out_su[i, j] = in_su[i, j] - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_su[i, j] = in_su[i, j] - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0) -
				in_su_tnd[i, j]
			)

		# advance the y-momentum
		if in_sv_tnd is None:
			out_sv[i, j] = in_sv[i, j] - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_sv[i, j] = in_sv[i, j] - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0) -
				in_sv_tnd[i, j]
			)

		if self._moist:
			# advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv[i, j] - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqv[i, j] = in_sqv[i, j] - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s[i, j] * in_qv_tnd[i, j]
				)

			# advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc[i, j] - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqc[i, j] = in_sqc[i, j] - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s[i, j] * in_qc_tnd[i, j]
				)

			# advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr[i, j] - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqr[i, j] = in_sqr[i, j] - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s[i, j] * in_qr_tnd[i, j]
				)

		if not self._moist:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


class RK2(IsentropicMinimalPrognostic):
	"""
	Implementation of the two-stages, second-order Runge-Kutta scheme
	which takes over the prognostic part of the three-dimensional, moist,
	*minimal*, isentropic dynamical core.
	"""
	def __init__(
		self, horizontal_flux_scheme, mode, grid, hb, moist,
		substeps, backend, dtype=datatype
	):
		substeps_ = substeps if substeps % 2 == 0 else substeps+1
		if substeps_ != substeps:
			warnings.warn(
				'Number of substeps increased from {} to {}.'.format(
					substeps, substeps_
				)
			)

		super().__init__(
			horizontal_flux_scheme, mode, grid, hb, moist,
			substeps, backend, dtype
		)

		# initialize the pointer to the underlying GT4Py stencil in charge
		# of carrying out the stages
		self._stage_stencil = None

	@property
	def stages(self):
		return 2

	@property
	def substep_fractions(self):
		return 0.5, 1.0

	def stage_call(self, stage, timestep, state, tendencies=None):
		# the first time this method is invoked, initialize the GT4Py stencil
		if self._stage_stencil is None:
			self._stage_stencil_initialize(tendencies)

		# update the attributes which serve as inputs to the GT4Py stencil
		self._stage_stencil_set_inputs(stage, timestep, state, tendencies)

		# step the prognostic variables
		self._stage_stencil.compute()

		# instantiate the output state
		state_new = {
			'time': state['time'] + 0.5*timestep,
			'air_isentropic_density': self._out_s,
			'x_momentum_isentropic': self._out_su,
			'y_momentum_isentropic': self._out_sv,
		}
		if self._moist:
			state_new.update({
				'isentropic_density_of_water_vapor': self._out_sqv,
				'isentropic_density_of_cloud_liquid_water': self._out_sqc,
				'isentropic_density_of_precipitation_water': self._out_sqr,
			})

		return state_new

	def _stage_stencil_initialize(self, tendencies):
		# allocate the attributes which will serve as inputs to the stencil
		self._stage_stencil_allocate_inputs(tendencies)

		# allocate the Numpy arrays which will store the output fields
		self._stage_stencil_allocate_outputs()

		# set the stencil's computational domain
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._hb.nb
		if self._mode == 'x':
			_domain = gt.domain.Rectangle(
				(nb, 0, 0), (nx-nb-1, ny-1, nz-1)
			)
		elif self._mode == 'y':
			_domain = gt.domain.Rectangle(
				(0, nb, 0), (nx-1, ny-nb-1, nz-1)
			)
		else:
			_domain = gt.domain.Rectangle(
				(nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)
			)

		# set the stencil's inputs and outputs
		_inputs = {
			'in_s': self._in_s, 'in_s_int': self._in_s_int,
			'in_u_int': self._in_u, 'in_v_int': self._in_v,
			'in_su': self._in_su, 'in_su_int': self._in_su_int,
			'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int,
		}
		_outputs = {
			'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv
		}
		if self._moist:
			_inputs.update({
				'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
				'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
				'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int,
			})
			_outputs.update({
				'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
				'out_sqr': self._out_sqr,
			})
		if tendencies is not None:
			if tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd
			if tendencies.get(mfwv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if tendencies.get(mfcw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if tendencies.get(mfpw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# instantiate the stencil
		self._stage_stencil = gt.NGStencil(
			definitions_func=self._stage_stencil_defs,
			inputs=_inputs,
			global_inputs={'dt': self._dt},
			outputs=_outputs,
			domain=_domain,
			mode=self._backend,
		)

	def _stage_stencil_allocate_inputs(self, tendencies):
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# call parent method
		super()._stage_stencil_allocate_inputs(tendencies)

		# allocate the Numpy arrays which will store the intermediate values
		# for the prognostic variables
		self._in_s_int	= np.zeros((nx, ny, nz), dtype=dtype)
		self._in_su_int = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_sv_int = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._in_sqv_int = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_sqc_int = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_sqr_int = np.zeros((nx, ny, nz), dtype=dtype)

	def _stage_stencil_set_inputs(self, stage, timestep, state, tendencies):
		# shortcuts
		if tendencies is not None:
			s_tnd_on  = tendencies.get('air_isentropic_density', None) is not None
			qv_tnd_on = tendencies.get(mfwv, None) is not None
			qc_tnd_on = tendencies.get(mfcw, None) is not None
			qr_tnd_on = tendencies.get(mfpw, None) is not None
			su_tnd_on = tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = tendencies.get('y_momentum_isentropic', None) is not None
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# update the local time step
		self._dt.value = (1./2. + 1./2.*(stage > 0)) * timestep.total_seconds()

		if stage == 0:
			# update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s[...]  = state['air_isentropic_density'][...]
			self._in_su[...] = state['x_momentum_isentropic'][...]
			self._in_sv[...] = state['y_momentum_isentropic'][...]
			if self._moist:
				self._in_sqv[...] = state['isentropic_density_of_water_vapor']
				self._in_sqc[...] = state['isentropic_density_of_cloud_liquid_water']
				self._in_sqr[...] = state['isentropic_density_of_precipitation_water']

		# update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._in_s_int[...]  = state['air_isentropic_density'][...]
		self._in_u[...]      = state['x_velocity_at_u_locations'][...]
		self._in_v[...]      = state['y_velocity_at_v_locations'][...]
		self._in_su_int[...] = state['x_momentum_isentropic'][...]
		self._in_sv_int[...] = state['y_momentum_isentropic'][...]
		if self._moist:
			self._in_sqv_int[...] = state['isentropic_density_of_water_vapor']
			self._in_sqc_int[...] = state['isentropic_density_of_cloud_liquid_water']
			self._in_sqr_int[...] = state['isentropic_density_of_precipitation_water']

		# extract the Numpy arrays representing the provided tendencies,
		# and update the Numpy arrays which serve as inputs to the GT4Py stencils
		if s_tnd_on:
			self._in_s_tnd[...]  = tendencies['air_isentropic_density'][...]
		if su_tnd_on:
			self._in_su_tnd[...] = tendencies['x_momentum_isentropic'][...]
		if sv_tnd_on:
			self._in_sv_tnd[...] = tendencies['y_momentum_isentropic'][...]
		if qv_tnd_on:
			self._in_qv_tnd[...] = tendencies[mfwv][...]
		if qc_tnd_on:
			self._in_qc_tnd[...] = tendencies[mfcw][...]
		if qr_tnd_on:
			self._in_qr_tnd[...] = tendencies[mfpw][...]

	def _stage_stencil_defs(
		self, dt, in_s, in_s_int, in_u_int, in_v_int,
		in_su, in_su_int, in_sv, in_sv_int,
		in_sqv=None, in_sqv_int=None,
		in_sqc=None, in_sqc_int=None,
		in_sqr=None, in_sqr_int=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None
	):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		md = self._mode

		# declare indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# calculate the fluxes
		if not self._moist:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(
					i, j, dt, in_s_int, in_u_int, in_v_int, in_su_int, in_sv_int,
					s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd
				)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(
					i, j, dt, in_s_int, in_u_int, in_v_int,
					in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
					in_s_tnd, in_su_tnd, in_sv_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd
				)

		# advance the isentropic density
		if in_s_tnd is None:
			out_s[i, j] = in_s[i, j] - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_s[i, j] = in_s[i, j] - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0) -
				in_s_tnd[i, j]
			)

		# advance the x-momentum
		if in_su_tnd is None:
			out_su[i, j] = in_su[i, j] - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_su[i, j] = in_su[i, j] - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0) -
				in_su_tnd[i, j]
			)

		# advance the y-momentum
		if in_sv_tnd is None:
			out_sv[i, j] = in_sv[i, j] - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_sv[i, j] = in_sv[i, j] - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0) -
				in_sv_tnd[i, j]
			)

		if self._moist:
			# advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv[i, j] = in_sqv[i, j] - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqv[i, j] = in_sqv[i, j] - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qv_tnd[i, j]
				)

			# advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc[i, j] = in_sqc[i, j] - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqc[i, j] = in_sqc[i, j] - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qc_tnd[i, j]
				)

			# advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr[i, j] = in_sqr[i, j] - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqr[i, j] = in_sqr[i, j] - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qr_tnd[i, j]
				)

		if not self._moist:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


class RK3WS(RK2):
	"""
	Implementation of the three-stages, second-order Runge-Kutta scheme
	which takes over the prognostic part of the three-dimensional, moist,
	*minimal*, isentropic dynamical core.
	"""
	def __init__(
		self, horizontal_flux_scheme, mode, grid, hb, moist,
		substeps, backend, dtype=datatype
	):
		substeps_ = substeps if substeps % 6 == 0 else substeps + (6 - substeps % 6)
		if substeps_ != substeps:
			warnings.warn(
				'Number of substeps increased from {} to {}.'.format(
					substeps, substeps_
				)
			)

		super().__init__(
			horizontal_flux_scheme, mode, grid, hb, moist,
			substeps, backend, dtype
		)

	@property
	def stages(self):
		return 3

	@property
	def substep_fractions(self):
		return 1.0/3.0, 0.5, 1.0

	def stage_call(self, stage, timestep, state, tendencies=None):
		state_new = super().stage_call(stage, timestep, state, tendencies)

		if stage == 0:
			state_new['time'] = state['time'] + 1.0/3.0 * timestep
		elif stage == 1:
			state_new['time'] = state['time'] + 1.0/6.0 * timestep
		else:
			state_new['time'] = state['time'] + 1.0/2.0 * timestep

		return state_new

	def _stage_stencil_set_inputs(self, stage, timestep, state, tendencies):
		# call parent's method
		super()._stage_stencil_set_inputs(stage, timestep, state, tendencies)

		# update the local time step
		self._dt.value = \
			(1./3. + 1./6.*(stage > 0) + 1./2.*(stage > 1)) * timestep.total_seconds()


class RK3(IsentropicMinimalPrognostic):
	"""
	Implementation of the three-stages, third-order Runge-Kutta scheme
	which takes over the prognostic part of the three-dimensional, moist,
	*minimal*, isentropic dynamical core.
	"""
	def __init__(
		self, horizontal_flux_scheme, mode, grid, hb, moist,
		substeps, backend, dtype=datatype
	):
		super().__init__(
			horizontal_flux_scheme, mode, grid, hb, moist,
			substeps, backend, dtype
		)

		# initialize the pointers to the underlying GT4Py stencils
		# in charge of carrying out the stages
		self._stage_stencil_first  = None
		self._stage_stencil_second = None
		self._stage_stencil_third  = None

		# free parameters for RK3
		self._alpha1 = 1./2.
		self._alpha2 = 3./4.

		# set the other parameters so to yield a third-order method
		self._gamma1 = (3.*self._alpha2 - 2.) / \
			(6. * self._alpha1 * (self._alpha2 - self._alpha1))
		self._gamma2 = (3.*self._alpha1 - 2.) / \
			(6. * self._alpha2 * (self._alpha1 - self._alpha2))
		self._gamma0 = 1. - self._gamma1 - self._gamma2
		self._beta21 = self._alpha2 - 1. / (6. * self._alpha1 * self._gamma2)

	@property
	def stages(self):
		return 3

	@property
	def substep_fractions(self):
		return self._alpha1, self._alpha2, 1.0

	def stage_call(self, stage, timestep, state, tendencies=None):
		# the first time this method is invoked, initialize the GT4Py stencil
		if self._stage_stencil_first is None:
			self._stage_stencils_initialize(tendencies)

		# update the attributes which serve as inputs to the GT4Py stencil
		self._stage_stencil_set_inputs(stage, timestep, state, tendencies)

		# run the compute function of the stencil stepping the solution
		self._stage_stencil_compute(stage)

		# instantiate the output state
		state_new = {
			'air_isentropic_density': self._out_s,
			'x_momentum_isentropic': self._out_su,
			'y_momentum_isentropic': self._out_sv,
		}
		if self._moist:
			state_new['isentropic_density_of_water_vapor'] = self._out_sqv
			state_new['isentropic_density_of_cloud_liquid_water'] = self._out_sqc
			state_new['isentropic_density_of_precipitation_water'] = self._out_sqr
		if stage == 0:
			state_new['time'] = state['time'] + self._alpha1*timestep
		elif stage == 1:
			state_new['time'] = state['time'] + (self._alpha2 - self._alpha1)*timestep
		else:
			state_new['time'] = state['time'] + (1 - self._alpha2 - self._alpha1)*timestep

		return state_new

	def _stage_stencils_initialize(self, tendencies):
		# allocate the attributes which will serve as inputs to the stencil
		self._stage_stencils_allocate_inputs(tendencies)

		# allocate the Numpy arrays which will store temporary fields to be
		# shared across the different stencils which step the solution
		# disregarding any vertical motion
		self._stage_stencils_allocate_temporaries()

		# allocate the Numpy arrays which will store the output fields
		self._stage_stencil_allocate_outputs()

		# set the stencils' computational domain
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._hb.nb
		if self._mode == 'x':
			_domain = gt.domain.Rectangle(
				(nb, 0, 0), (nx-nb-1, ny-1, nz-1)
			)
		elif self._mode == 'y':
			_domain = gt.domain.Rectangle(
				(0, nb, 0), (nx-1, ny-nb-1, nz-1)
			)
		else:
			_domain = gt.domain.Rectangle(
				(nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)
			)

		# set the first stencil's inputs and outputs
		_inputs = {
			'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v,
			'in_su': self._in_su, 'in_sv': self._in_sv,
		}
		_outputs = {
			'out_s0': self._s0, 'out_s': self._out_s,
			'out_su0': self._su0, 'out_su': self._out_su,
			'out_sv0': self._sv0, 'out_sv': self._out_sv,
		}
		if self._moist:
			_inputs.update({
				'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
				'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
				'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int,
			})
			_outputs.update({
				'out_sqv0': self._sqv0, 'out_sqv': self._out_sqv,
				'out_sqc0': self._sqc0, 'out_sqc': self._out_sqc,
				'out_sqr0': self._sqr0, 'out_sqr': self._out_sqr,
			})
		if tendencies is not None:
			if tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd
			if tendencies.get(mfwv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if tendencies.get(mfcw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if tendencies.get(mfpw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# instantiate the first stencil
		self._stage_stencil_first = gt.NGStencil(
			definitions_func=self._stage_stencil_first_defs,
			inputs=_inputs,
			global_inputs={'dt': self._dt},
			outputs=_outputs,
			domain=_domain,
			mode=self._backend,
		)

		# set the second stencil's inputs and outputs
		_inputs = {
			'in_s': self._in_s, 'in_s_int': self._in_s_int,
			'in_u_int': self._in_u, 'in_v_int': self._in_v,
			'in_su': self._in_su, 'in_su_int': self._in_su_int,
			'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int,
			'in_s0': self._s0, 'in_su0': self._su0, 'in_sv0': self._sv0,
		}
		_outputs = {
			'out_s1': self._s1, 'out_s': self._out_s,
			'out_su1': self._su1, 'out_su': self._out_su,
			'out_sv1': self._sv1, 'out_sv': self._out_sv,
		}
		if self._moist:
			_inputs.update({
				'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
				'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
				'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int,
				'in_sqv0': self._sqv0, 'in_sqc0': self._sqc0,
				'in_sqr0': self._sqr0,
			})
			_outputs.update({
				'out_sqv1': self._sqv1, 'out_sqv': self._out_sqv,
				'out_sqc1': self._sqc1, 'out_sqc': self._out_sqc,
				'out_sqr1': self._sqr1, 'out_sqr': self._out_sqr,
			})
		if tendencies is not None:
			if tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd
			if tendencies.get(mfwv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if tendencies.get(mfcw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if tendencies.get(mfpw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# instantiate the second stencil
		self._stage_stencil_second = gt.NGStencil(
			definitions_func=self._stage_stencil_second_defs,
			inputs=_inputs,
			global_inputs={'dt': self._dt},
			outputs=_outputs,
			domain=_domain,
			mode=self._backend,
		)

		# set the third stencil's inputs and outputs
		_inputs = {
			'in_s': self._in_s, 'in_s_int': self._in_s_int,
			'in_u_int': self._in_u, 'in_v_int': self._in_v,
			'in_su': self._in_su, 'in_su_int': self._in_su_int,
			'in_sv': self._in_sv, 'in_sv_int': self._in_sv_int,
			'in_s0': self._s0, 'in_su0': self._su0, 'in_sv0': self._sv0,
			'in_s1': self._s1, 'in_su1': self._su1, 'in_sv1': self._sv1,
		}
		_outputs = {
			'out_s': self._out_s, 'out_su': self._out_su, 'out_sv': self._out_sv,
		}
		if self._moist:
			_inputs.update({
				'in_sqv': self._in_sqv, 'in_sqv_int': self._in_sqv_int,
				'in_sqc': self._in_sqc, 'in_sqc_int': self._in_sqc_int,
				'in_sqr': self._in_sqr, 'in_sqr_int': self._in_sqr_int,
				'in_sqv0': self._sqv0, 'in_sqc0': self._sqc0,
				'in_sqr0': self._sqr0, 'in_sqv1': self._sqv1,
				'in_sqc1': self._sqc1, 'in_sqr1': self._sqr1,
			})
			_outputs.update({
				'out_sqv': self._out_sqv, 'out_sqc': self._out_sqc,
				'out_sqr': self._out_sqr,
			})
		if tendencies is not None:
			if tendencies.get('air_isentropic_density', None) is not None:
				_inputs['in_s_tnd'] = self._in_s_tnd
			if tendencies.get('x_momentum_isentropic', None) is not None:
				_inputs['in_su_tnd'] = self._in_su_tnd
			if tendencies.get('y_momentum_isentropic', None) is not None:
				_inputs['in_sv_tnd'] = self._in_sv_tnd
			if tendencies.get(mfwv, None) is not None:
				_inputs['in_qv_tnd'] = self._in_qv_tnd
			if tendencies.get(mfcw, None) is not None:
				_inputs['in_qc_tnd'] = self._in_qc_tnd
			if tendencies.get(mfpw, None) is not None:
				_inputs['in_qr_tnd'] = self._in_qr_tnd

		# instantiate the third stencil
		self._stage_stencil_third = gt.NGStencil(
			definitions_func=self._stage_stencil_third_defs,
			inputs=_inputs,
			global_inputs={'dt': self._dt},
			outputs=_outputs,
			domain=_domain,
			mode=self._backend,
		)

	def _stage_stencils_allocate_inputs(self, tendencies):
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# call parent's method
		super()._stage_stencil_allocate_inputs(tendencies)

		# allocate the Numpy arrays which will store the intermediate values
		# for the prognostic variables
		self._in_s_int	= np.zeros((nx, ny, nz), dtype=dtype)
		self._in_su_int = np.zeros((nx, ny, nz), dtype=dtype)
		self._in_sv_int = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._in_sqv_int = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_sqc_int = np.zeros((nx, ny, nz), dtype=dtype)
			self._in_sqr_int = np.zeros((nx, ny, nz), dtype=dtype)

	def _stage_stencils_allocate_temporaries(self):
		"""
		Allocate the Numpy arrays which store temporary fields to be shared
		among the different GT4Py stencils.
		"""
		# shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# allocate the Numpy arrays which will store the first increment
		# for all prognostic variables
		self._s0  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su0 = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv0 = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._sqv0 = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqc0 = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqr0 = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the Numpy arrays which will store the second increment
		# for all prognostic variables
		self._s1  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su1 = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv1 = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._sqv1 = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqc1 = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqr1 = np.zeros((nx, ny, nz), dtype=dtype)

	def _stage_stencil_set_inputs(self, stage, dt, state, tendencies):
		# shortcuts
		if tendencies is not None:
			s_tnd_on  = tendencies.get('air_isentropic_density', None) is not None
			qv_tnd_on = tendencies.get(mfwv, None) is not None
			qc_tnd_on = tendencies.get(mfcw, None) is not None
			qr_tnd_on = tendencies.get(mfpw, None) is not None
			su_tnd_on = tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = tendencies.get('y_momentum_isentropic', None) is not None
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# update the local time step
		self._dt.value = dt.total_seconds()

		if stage == 0:
			# update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s[...]  = state['air_isentropic_density'][...]
			self._in_u[...]  = state['x_velocity_at_u_locations'][...]
			self._in_v[...]  = state['y_velocity_at_v_locations'][...]
			self._in_su[...] = state['x_momentum_isentropic'][...]
			self._in_sv[...] = state['y_momentum_isentropic'][...]
			if self._moist:
				self._in_sqv[...] = state['isentropic_density_of_water_vapor'][...]
				self._in_sqc[...] = state['isentropic_density_of_cloud_liquid_water'][...]
				self._in_sqr[...] = state['isentropic_density_of_precipitation_water'][...]
		else:
			# update the Numpy arrays which serve as inputs to the GT4Py stencils
			self._in_s_int[...]  = state['air_isentropic_density'][...]
			self._in_u[...]      = state['x_velocity_at_u_locations'][...]
			self._in_v[...]      = state['y_velocity_at_v_locations'][...]
			self._in_su_int[...] = state['x_momentum_isentropic'][...]
			self._in_sv_int[...] = state['y_momentum_isentropic'][...]
			if self._moist:
				self._in_sqv_int[...] = state['isentropic_density_of_water_vapor'][...]
				self._in_sqc_int[...] = state['isentropic_density_of_cloud_liquid_water'][...]
				self._in_sqr_int[...] = state['isentropic_density_of_precipitation_water'][...]

		# extract the Numpy arrays representing the provided tendencies,
		# and update the Numpy arrays which serve as inputs to the GT4Py stencils
		if s_tnd_on:
			self._in_s_tnd[...]  = tendencies['air_isentropic_density'][...]
		if su_tnd_on:
			self._in_su_tnd[...] = tendencies['x_momentum_isentropic'][...]
		if sv_tnd_on:
			self._in_sv_tnd[...] = tendencies['y_momentum_isentropic'][...]
		if qv_tnd_on:
			self._in_qv_tnd[...] = tendencies[mfwv][...]
		if qc_tnd_on:
			self._in_qc_tnd[...] = tendencies[mfcw][...]
		if qr_tnd_on:
			self._in_qr_tnd[...] = tendencies[mfpw][...]

	def _stage_stencil_compute(self, stage):
		if stage == 0:
			self._stage_stencil_first.compute()
		elif stage == 1:
			self._stage_stencil_second.compute()
		else:
			self._stage_stencil_third.compute()

	def _stage_stencil_first_defs(
		self, dt, in_s, in_u, in_v, in_su, in_sv,
		in_sqv=None, in_sqc=None, in_sqr=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None
	):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		md = self._mode
		a1 = self._alpha1

		# declare indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate fields representing the first increments
		out_s0	= gt.Equation()
		out_su0 = gt.Equation()
		out_sv0 = gt.Equation()
		if self._moist:
			out_sqv0 = gt.Equation()
			out_sqc0 = gt.Equation()
			out_sqr0 = gt.Equation()

		# instantiate fields representing the output state
		out_s  = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# calculate the fluxes
		if not self._moist:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(
					i, j, dt, in_s, in_u, in_v, in_su, in_sv,
					s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd
				)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(
					i, j, dt, in_s, in_u, in_v, in_su, in_sv,
					in_sqv, in_sqc, in_sqr, in_s_tnd, in_su_tnd, in_sv_tnd,
					in_qv_tnd, in_qc_tnd, in_qr_tnd
				)

		# advance the isentropic density
		if in_s_tnd is None:
			out_s0[i, j] = - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_s0[i, j] = - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0) -
				in_s_tnd[i, j]
			)
		out_s[i, j] = in_s[i, j] + a1 * out_s0[i, j]

		# advance the x-momentum
		if in_su_tnd is None:
			out_su0[i, j] = - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_su0[i, j] = - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0) -
				in_su_tnd[i, j]
			)
		out_su[i, j] = in_su[i, j] + a1 * out_su0[i, j]

		# advance the y-momentum
		if in_sv_tnd is None:
			out_sv0[i, j] = - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_sv0[i, j] = - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0) -
				in_sv_tnd[i, j]
			)
		out_sv[i, j] = in_sv[i, j] + a1 * out_sv0[i, j]

		if self._moist:
			# advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv0[i, j] = - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqv0[i, j] = - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s[i, j] * in_qv_tnd[i, j]
				)
			out_sqv[i, j] = in_sqv[i, j] + a1 * out_sqv0[i, j]

			# advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc0[i, j] = - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqc0[i, j] = - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s[i, j] * in_qc_tnd[i, j]
				)
			out_sqc[i, j] = in_sqc[i, j] + a1 * out_sqc0[i, j]

			# advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr0[i, j] = - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqr0[i, j] = - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s[i, j] * in_qr_tnd[i, j]
				)
			out_sqr[i, j] = in_sqr[i, j] + a1 * out_sqr0[i, j]

		if not self._moist:
			return out_s0, out_s, out_su0, out_su, out_sv0, out_sv
		else:
			return out_s0, out_s, out_su0, out_su, out_sv0, out_sv, \
				out_sqv0, out_sqv, out_sqc0, out_sqc, out_sqr0, out_sqr

	def _stage_stencil_second_defs(
		self, dt, in_s, in_s_int, in_s0, in_u_int, in_v_int,
		in_su, in_su_int, in_su0, in_sv, in_sv_int, in_sv0,
		in_sqv=None, in_sqv_int=None, in_sqv0=None,
		in_sqc=None, in_sqc_int=None, in_sqc0=None,
		in_sqr=None, in_sqr_int=None, in_sqr0=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None
	):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		md = self._mode
		a2, b21 = self._alpha2, self._beta21

		# declare indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate fields representing the first increments
		out_s1	= gt.Equation()
		out_su1 = gt.Equation()
		out_sv1 = gt.Equation()
		if self._moist:
			out_sqv1 = gt.Equation()
			out_sqc1 = gt.Equation()
			out_sqr1 = gt.Equation()

		# instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# calculate the fluxes
		if not self._moist:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(
					i, j, dt, in_s_int, in_u_int, in_v_int, in_su_int, in_sv_int,
					s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd
				)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(
					i, j, dt, in_s_int, in_u_int, in_v_int,
					in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
					in_s_tnd, in_su_tnd, in_sv_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd
				)

		# advance the isentropic density
		if in_s_tnd is None:
			out_s1[i, j] = - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_s1[i, j] = - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0) -
				in_s_tnd[i, j]
			)
		out_s[i, j] = in_s[i, j] + b21 * in_s0[i, j] + (a2 - b21) * out_s1[i, j]

		# advance the x-momentum
		if in_su_tnd is None:
			out_su1[i, j] = - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_su1[i, j] = - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0) -
				in_su_tnd[i, j]
			)
		out_su[i, j] = in_su[i, j] + b21 * in_su0[i, j] + (a2 - b21) * out_su1[i, j]

		# advance the y-momentum
		if in_sv_tnd is None:
			out_sv1[i, j] = - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			out_sv1[i, j] = - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0) -
				in_sv_tnd[i, j]
			)
		out_sv[i, j] = in_sv[i, j] + b21 * in_sv0[i, j] + (a2 - b21) * out_sv1[i, j]

		if self._moist:
			# advance the isentropic density of water vapor
			if in_qv_tnd is None:
				out_sqv1[i, j] = - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqv1[i, j] = - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qv_tnd[i, j]
				)
			out_sqv[i, j] = in_sqv[i, j] + b21 * in_sqv0[i, j] + (a2 - b21) * out_sqv1[i, j]

			# advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				out_sqc1[i, j] = - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqc1[i, j] = - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qc_tnd[i, j]
				)
			out_sqc[i, j] = in_sqc[i, j] + b21 * in_sqc0[i, j] + (a2 - b21) * out_sqc1[i, j]

			# advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				out_sqr1[i, j] = - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				out_sqr1[i, j] = - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qr_tnd[i, j]
				)
			out_sqr[i, j] = in_sqr[i, j] + b21 * in_sqr0[i, j] + (a2 - b21) * out_sqr1[i, j]

		if not self._moist:
			return out_s1, out_s, out_su1, out_su, out_sv1, out_sv
		else:
			return out_s1, out_s, out_su1, out_su, out_sv1, out_sv, \
				out_sqv1, out_sqv, out_sqc1, out_sqc, out_sqr1, out_sqr

	def _stage_stencil_third_defs(
		self, dt, in_s, in_s_int, in_s0, in_s1, in_u_int, in_v_int,
		in_su, in_su_int, in_su0, in_su1, in_sv, in_sv_int, in_sv0, in_sv1,
		in_sqv=None, in_sqv_int=None, in_sqv0=None, in_sqv1=None,
		in_sqc=None, in_sqc_int=None, in_sqc0=None, in_sqc1=None,
		in_sqr=None, in_sqr_int=None, in_sqr0=None, in_sqr1=None,
		in_s_tnd=None, in_su_tnd=None, in_sv_tnd=None,
		in_qv_tnd=None, in_qc_tnd=None, in_qr_tnd=None
	):
		# shortcuts
		dx = self._grid.dx.to_units('m').values.item()
		dy = self._grid.dy.to_units('m').values.item()
		md = self._mode
		g0, g1, g2 = self._gamma0, self._gamma1, self._gamma2

		# declare indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# instantiate fields representing the first increments
		tmp_s2 = gt.Equation()
		tmp_su2 = gt.Equation()
		tmp_sv2 = gt.Equation()
		if self._moist:
			tmp_sqv2 = gt.Equation()
			tmp_sqc2 = gt.Equation()
			tmp_sqr2 = gt.Equation()

		# instantiate output fields
		out_s = gt.Equation()
		out_su = gt.Equation()
		out_sv = gt.Equation()
		if self._moist:
			out_sqv = gt.Equation()
			out_sqc = gt.Equation()
			out_sqr = gt.Equation()

		# calculate the fluxes
		if not self._moist:
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = \
				self._hflux(
					i, j, dt, in_s_int, in_u_int, in_v_int, in_su_int, in_sv_int,
					s_tnd=in_s_tnd, su_tnd=in_su_tnd, sv_tnd=in_sv_tnd
				)
		else:
			flux_s_x,  flux_s_y, flux_su_x,  flux_su_y, flux_sv_x,	flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y = \
				self._hflux(
					i, j, dt, in_s_int, in_u_int, in_v_int,
					in_su_int, in_sv_int, in_sqv_int, in_sqc_int, in_sqr_int,
					in_s_tnd, in_su_tnd, in_sv_tnd, in_qv_tnd, in_qc_tnd, in_qr_tnd
				)

		# advance the isentropic density
		if in_s_tnd is None:
			tmp_s2[i, j] = - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			tmp_s2[i, j] = - dt * (
				((flux_s_x[i, j] - flux_s_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_s_y[i, j] - flux_s_y[i, j-1]) / dy if md != 'x' else 0) -
				in_s_tnd[i, j]
			)
		out_s[i, j] = in_s[i, j] + \
			g0 * in_s0[i, j] + g1 * in_s1[i, j] + g2 * tmp_s2[i, j]

		# advance the x-momentum
		if in_su_tnd is None:
			tmp_su2[i, j] = - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			tmp_su2[i, j] = - dt * (
				((flux_su_x[i, j] - flux_su_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_su_y[i, j] - flux_su_y[i, j-1]) / dy if md != 'x' else 0) -
				in_su_tnd[i, j]
			)
		out_su[i, j] = in_su[i, j] + \
			g0 * in_su0[i, j] + g1 * in_su1[i, j] + g2 * tmp_su2[i, j]

		# advance the y-momentum
		if in_sv_tnd is None:
			tmp_sv2[i, j] = - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0)
			)
		else:
			tmp_sv2[i, j] = - dt * (
				((flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx if md != 'y' else 0) +
				((flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy if md != 'x' else 0) -
				in_sv_tnd[i, j]
			)
		out_sv[i, j] = in_sv[i, j] + \
			g0 * in_sv0[i, j] + g1 * in_sv1[i, j] + g2 * tmp_sv2[i, j]

		if self._moist:
			# advance the isentropic density of water vapor
			if in_qv_tnd is None:
				tmp_sqv2[i, j] = - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				tmp_sqv2[i, j] = - dt * (
					((flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qv_tnd[i, j]
				)
			out_sqv[i, j] = in_sqv[i, j] + \
				g0 * in_sqv0[i, j] + g1 * in_sqv1[i, j] + g2 * tmp_sqv2[i, j]

			# advance the isentropic density of cloud liquid water
			if in_qc_tnd is None:
				tmp_sqc2[i, j] = - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				tmp_sqc2[i, j] = - dt * (
					((flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qc_tnd[i, j]
				)
			out_sqc[i, j] = in_sqc[i, j] + \
				g0 * in_sqc0[i, j] + g1 * in_sqc1[i, j] + g2 * tmp_sqc2[i, j]

			# advance the isentropic density of precipitation water
			if in_qr_tnd is None:
				tmp_sqr2[i, j] = - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0)
				)
			else:
				tmp_sqr2[i, j] = - dt * (
					((flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx if md != 'y' else 0) +
					((flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy if md != 'x' else 0) -
					in_s_int[i, j] * in_qr_tnd[i, j]
				)
			out_sqr[i, j] = in_sqr[i, j] + \
				g0 * in_sqr0[i, j] + g1 * in_sqr1[i, j] + g2 * tmp_sqr2[i, j]

		if not self._moist:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr
