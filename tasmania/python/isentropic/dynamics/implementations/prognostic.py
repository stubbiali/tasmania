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
	ForwardEulerSI(IsentropicPrognostic)
	CenteredSI(IsentropicPrognostic)
	RK3WSSI(IsentropicPrognostic)
"""
import numpy as np

import gridtools as gt
from tasmania.python.isentropic.dynamics.diagnostics import \
	IsentropicDiagnostics
from tasmania.python.isentropic.dynamics.horizontal_fluxes import \
	IsentropicHorizontalFlux, NGIsentropicMinimalHorizontalFlux
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic


def step_forward_euler(
	fluxer, i, j, dt, dx, dy, s_now, s_int, s_new,
	u_int, v_int, mtg_int, su_int, sv_int,
	sq_now=None, sq_int=None, sq_new=None, s_tnd=None, q_tnd=None
):
	q_tnd = {} if q_tnd is None else q_tnd
	q_tnd = {
		tracer: tendency for tracer, tendency in q_tnd.items()
		if tendency is not None
	}

	if isinstance(fluxer, NGIsentropicMinimalHorizontalFlux):
		fluxes = fluxer(
			i, j, dt, s_int, u_int, v_int, su_int, sv_int,
			s_tnd=s_tnd, **sq_int, **q_tnd
		)
	else:
		fluxes = fluxer(
			i, j, dt, s_int, u_int, v_int, mtg_int, su_int, sv_int,
			s_tnd=s_tnd, **sq_int, **q_tnd
		)

	flux_s_x, flux_s_y = fluxes[0], fluxes[1]
	flux_sq_x = {
		tracer: fluxes[6 + 2*idx] for idx, tracer in enumerate(sq_now.keys())
	}
	flux_sq_y = {
		tracer: fluxes[6 + 2*idx + 1] for idx, tracer in enumerate(sq_now.keys())
	}

	s_new[i, j] = s_now[i, j] - dt * (
		(flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
		(flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
		(s_tnd[i, j] if s_tnd is not None else 0.0)
	)

	for tracer in sq_now:
		sq_new[tracer][i, j] = sq_now[tracer][i, j] - dt * (
			(flux_sq_x[tracer][i, j] - flux_sq_x[tracer][i-1, j]) / dx +
			(flux_sq_y[tracer][i, j] - flux_sq_y[tracer][i, j-1]) / dy -
			(s_int[i, j] * q_tnd[tracer][i, j] if tracer in q_tnd else 0.0)
		)


def step_forward_euler_momentum(
	fluxer, eps, i, j, dt, dx, dy, s_now, s_int, s_new, u_int, v_int,
	mtg_now, mtg_int, mtg_new, su_now, su_int, su_new, sv_now, sv_int, sv_new,
	su_tnd=None, sv_tnd=None
):
	sq = {'s' + tracer: gt.Equation() for tracer in fluxer._tracers}

	if isinstance(fluxer, NGIsentropicMinimalHorizontalFlux):
		fluxes = fluxer(
			i, j, dt, s_int, u_int, v_int, su_int, sv_int,
			su_tnd=su_tnd, sv_tnd=sv_tnd, **sq
		)
	else:
		fluxes = fluxer(
			i, j, dt, s_int, u_int, v_int, mtg_int, su_int, sv_int,
			su_tnd=su_tnd, sv_tnd=sv_tnd, **sq
		)

	flux_su_x, flux_su_y = fluxes[2], fluxes[3]
	flux_sv_x, flux_sv_y = fluxes[4], fluxes[5]

	su_new[i, j] = su_now[i, j] - dt * (
		(flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
		(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
		(1.0 - eps) * s_now[i, j] * (mtg_now[i+1, j] - mtg_now[i-1, j]) / (2.0 * dx) +
		eps * s_new[i, j] * (mtg_new[i+1, j] - mtg_new[i-1, j]) / (2.0 * dx) -
		(su_tnd[i, j] if su_tnd is not None else 0.0)
	)
	sv_new[i, j] = sv_now[i, j] - dt * (
		(flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
		(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
		(1.0 - eps) * s_now[i, j] * (mtg_now[i, j+1] - mtg_now[i, j-1]) / (2.0 * dy) +
		eps * s_new[i, j] * (mtg_new[i, j+1] - mtg_new[i, j-1]) / (2.0 * dy) -
		(sv_tnd[i, j] if sv_tnd is not None else 0.0)
	)


class ForwardEulerSI(IsentropicPrognostic):
	""" The semi-implicit upwind scheme. """
	def __init__(
		self, horizontal_flux_scheme, grid, hb, tracers, backend, dtype, **kwargs
	):
		# call parent's constructor
		super().__init__(
			NGIsentropicMinimalHorizontalFlux, horizontal_flux_scheme,
			grid, hb, tracers, backend, dtype
		)

		# extract the upper boundary conditions on the pressure field and
		# the off-centering parameter for the semi-implicit integrator
		self._pt = kwargs['pt'].to_units('Pa').values.item() if 'pt' in kwargs else 0.0
		self._eps = gt.Global(kwargs.get('eps', 0.5))
		assert 0.0 <= self._eps.value <= 1.0, \
			'The off-centering parameter should be between 0 and 1.'

		# instantiate the component retrieving the diagnostic variables
		self._diagnostics = IsentropicDiagnostics(grid, backend, dtype)

		# initialize the pointers to the stencils
		self._stencil = None
		self._stencil_momentum = None

	@property
	def stages(self):
		return 1

	def stage_call(self, stage, timestep, state, tendencies=None):
		tendencies = {} if tendencies is None else tendencies

		if self._stencil is None:
			# initialize the stencils
			self._stencils_initialize(tendencies)

		# set stencils' inputs
		self._stencils_set_inputs(stage, timestep, state, tendencies)

		# step the isentropic density and the water species
		self._stencil.compute()

		# apply the boundary conditions on the stepped isentropic density
		try:
			self._hb.dmn_enforce_field(
				self._s_new, 'air_isentropic_density', 'kg m^-2 K^-1',
				time=state['time']+timestep
			)
		except AttributeError:
			self._hb.enforce_field(
				self._s_new, 'air_isentropic_density', 'kg m^-2 K^-1',
				time=state['time']+timestep
			)

		# diagnose the Montgomery potential from the stepped isentropic density
		self._diagnostics.get_montgomery_potential(
			self._s_new, self._pt, self._mtg_new
		)

		# step the momenta
		self._stencil_momentum.compute()

		# collect the outputs
		out_state = {
			'time': state['time'] + timestep,
			'air_isentropic_density': self._s_new,
			'x_momentum_isentropic': self._su_new,
			'y_momentum_isentropic': self._sv_new
		}
		out_state.update({
			's_' + tracer: self._sq_new[tracer] for tracer in self._tracers
		})

		return out_state

	def _stencils_allocate(self, tendencies):
		super()._stencils_allocate(tendencies)

		# allocate the array which will store the Montgomery potential
		# retrieved from the updated isentropic density
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype
		self._mtg_new = np.zeros((nx, ny, nz), dtype=dtype)

	def _stencils_initialize(self, tendencies):
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._hb.nb
		tracer_symbols = {
			tracer: self._tracers[tracer]['stencil_symbol']
			for tracer in self._tracers
		}

		s_tnd_on  = 'air_isentropic_density' in tendencies
		su_tnd_on = 'x_momentum_isentropic' in tendencies
		sv_tnd_on = 'y_momentum_isentropic' in tendencies
		q_tnd_on = {tracer: tracer in tendencies for tracer in self._tracers}

		# allocate inputs and outputs
		self._stencils_allocate(tendencies)

		# set inputs and outputs for the stencil advancing the isentropic
		# density and the water constituents, then instantiate the stencil
		inputs = {
			's_now': self._s_now, 'u_now': self._u_now, 'v_now': self._v_now,
			'su_now': self._su_now, 'sv_now': self._sv_now
		}
		inputs.update({
			's' + tracer_symbols[tracer] + '_now': self._sq_now[tracer]
			for tracer in self._tracers
		})
		if s_tnd_on:
			inputs['s_tnd'] = self._s_tnd
		inputs.update({
			tracer_symbols[tracer] + '_tnd': self._q_tnd[tracer]
			for tracer in self._tracers if q_tnd_on[tracer]
		})

		outputs = {'s_new': self._s_new}
		outputs.update({
			's' + tracer_symbols[tracer] + '_new': self._sq_new[tracer]
			for tracer in self._tracers
		})

		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=inputs,
			global_inputs={'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

		# set inputs and outputs for the stencil advancing the momenta,
		# then instantiate the stencil
		inputs = {
			's_now': self._s_now, 's_new': self._s_new,
			'u_now': self._u_now, 'v_now': self._v_now,
			'mtg_now': self._mtg_now, 'mtg_new': self._mtg_new,
			'su_now': self._su_now, 'sv_now': self._sv_now
		}
		if su_tnd_on:
			inputs['su_tnd'] = self._su_tnd
		if sv_tnd_on:
			inputs['sv_tnd'] = self._sv_tnd
		outputs = {'su_new': self._su_new, 'sv_new': self._sv_new}
		self._stencil_momentum = gt.NGStencil(
			definitions_func=self._stencil_momentum_defs,
			inputs=inputs,
			global_inputs={'eps': self._eps, 'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

	def _stencil_defs(
		self, dt, dx, dy, s_now, u_now, v_now, su_now, sv_now,
		s_tnd=None, **tracer_kwargs
	):
		tracer_symbols = {
			tracer: self._tracers[tracer]['stencil_symbol']
			for tracer in self._tracers
		}

		# horizontal indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output fields
		mtg_now = gt.Equation()
		s_new = gt.Equation()
		sq_new = {
			tracer: gt.Equation(name='s' + tracer_symbols[tracer] + '_new')
			for tracer in self._tracers
		}

		# retrieve tracers
		sq_now = {
			tracer: tracer_kwargs['s' + tracer_symbols[tracer] + '_now']
			for tracer in self._tracers
		}
		sq_int = {
			's' + tracer_symbols[tracer]: sq_now[tracer]
			for tracer in self._tracers
		}
		q_tnd = {
			tracer: tracer_kwargs.get(tracer_symbols[tracer] + '_tnd', None)
			for tracer in self._tracers
		}

		# calculations
		step_forward_euler(
			self._hflux, i, j, dt, dx, dy, s_now, s_now, s_new,
			u_now, v_now, mtg_now, su_now, sv_now, s_tnd=s_tnd,
			sq_now=sq_now, sq_int=sq_int, sq_new=sq_new, q_tnd=q_tnd
		)

		return_list = [s_new, ] + [sq_new[tracer] for tracer in self._tracers]

		return return_list

	def _stencil_momentum_defs(
		self, eps, dt, dx, dy, s_now, s_new, u_now, v_now, mtg_now, mtg_new,
		su_now, sv_now, su_tnd=None, sv_tnd=None
	):
		# horizontal indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output fields
		su_new = gt.Equation()
		sv_new = gt.Equation()

		# calculations
		step_forward_euler_momentum(
			self._hflux, eps, i, j, dt, dx, dy, s_now, s_now, s_new, u_now, v_now,
			mtg_now, mtg_now, mtg_new, su_now, su_now, su_new, sv_now, sv_now, sv_new,
			su_tnd=su_tnd, sv_tnd=sv_tnd
		)

		return su_new, sv_new


class CenteredSI(IsentropicPrognostic):
	pass


class RK3WSSI(IsentropicPrognostic):
	""" The semi-implicit three-stages Runge-Kutta scheme. """
	def __init__(
		self, horizontal_flux_scheme, grid, hb, tracers, backend, dtype, **kwargs
	):
		# call parent's constructor
		super().__init__(
			NGIsentropicMinimalHorizontalFlux, horizontal_flux_scheme,
			grid, hb, tracers, backend, dtype
		)

		# extract the upper boundary conditions on the pressure field and
		# the off-centering parameter for the semi-implicit integrator
		self._pt = kwargs['pt'].to_units('Pa').values.item() if 'pt' in kwargs else 0.0
		self._eps = gt.Global(kwargs.get('eps', 0.5))
		assert 0.0 <= self._eps.value <= 1.0, \
			'The off-centering parameter should be between 0 and 1.'

		# instantiate the component retrieving the diagnostic variables
		self._diagnostics = IsentropicDiagnostics(grid, backend, dtype)

		# initialize the pointers to the stencils
		self._stencil = None
		self._stencil_momentum = None

	@property
	def stages(self):
		return 3

	def stage_call(self, stage, timestep, state, tendencies=None):
		tendencies = {} if tendencies is None else tendencies

		if self._stencil is None:
			# initialize the stencils
			self._stencils_initialize(tendencies)

		# set stencils' inputs
		self._stencils_set_inputs(stage, timestep, state, tendencies)

		# step the isentropic density and the water species
		self._stencil.compute()

		# apply the boundary conditions on the stepped isentropic density
		if stage == 0:
			dt = timestep/3.0
		elif stage == 1:
			dt = timestep/6.0
		else:
			dt = 0.5*timestep
		try:
			self._hb.dmn_enforce_field(
				self._s_new, 'air_isentropic_density', 'kg m^-2 K^-1',
				time=state['time']+dt
			)
		except AttributeError:
			self._hb.enforce_field(
				self._s_new, 'air_isentropic_density', 'kg m^-2 K^-1',
				time=state['time']+dt
			)

		# diagnose the Montgomery potential from the stepped isentropic density
		self._diagnostics.get_montgomery_potential(
			self._s_new, self._pt, self._mtg_new
		)

		# step the momenta
		self._stencil_momentum.compute()

		# collect the outputs
		out_state = {
			'time': state['time'] + dt,
			'air_isentropic_density': self._s_new,
			'x_momentum_isentropic': self._su_new,
			'y_momentum_isentropic': self._sv_new
		}
		out_state.update({
			's_' + tracer: self._sq_new[tracer] for tracer in self._tracers
		})

		return out_state

	def _stencils_allocate(self, tendencies):
		super()._stencils_allocate(tendencies)

		# allocate the array which will store the Montgomery potential
		# retrieved from the updated isentropic density
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype
		self._mtg_new = np.zeros((nx, ny, nz), dtype=dtype)

		# allocate the arrays which will store the intermediate values
		# of the model variables
		self._s_int  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su_int = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv_int = np.zeros((nx, ny, nz), dtype=dtype)
		self._sq_int = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._tracers
		}

	def _stencils_initialize(self, tendencies):
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._hb.nb
		tracer_symbols = {
			tracer: self._tracers[tracer]['stencil_symbol']
			for tracer in self._tracers
		}

		s_tnd_on  = 'air_isentropic_density' in tendencies
		su_tnd_on = 'x_momentum_isentropic' in tendencies
		sv_tnd_on = 'y_momentum_isentropic' in tendencies
		q_tnd_on = {tracer: tracer in tendencies for tracer in self._tracers}

		# allocate inputs and outputs
		self._stencils_allocate(tendencies)

		# set inputs and outputs for the stencil advancing the isentropic
		# density and the water constituents, then instantiate the stencil
		inputs = {
			's_now': self._s_now, 's_int': self._s_int,
			'u_int': self._u_now, 'v_int': self._v_now,
			'su_int': self._su_int, 'sv_int': self._sv_int
		}
		if s_tnd_on:
			inputs['s_tnd'] = self._s_tnd
		inputs.update({
			's' + tracer_symbols[tracer] + '_now': self._sq_now[tracer]
			for tracer in self._tracers
		})
		inputs.update({
			's' + tracer_symbols[tracer] + '_int': self._sq_int[tracer]
			for tracer in self._tracers
		})
		inputs.update({
			tracer_symbols[tracer] + '_tnd': self._q_tnd[tracer]
			for tracer in self._tracers if q_tnd_on[tracer]
		})

		outputs = {'s_new': self._s_new}
		outputs.update({
			's' + tracer_symbols[tracer] + '_new': self._sq_new[tracer]
			for tracer in self._tracers
		})

		self._stencil = gt.NGStencil(
			definitions_func=self._stencil_defs,
			inputs=inputs,
			global_inputs={'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

		# set inputs and outputs for the stencil advancing the momenta,
		# then instantiate the stencil
		inputs = {
			's_now': self._s_now, 's_int': self._s_int, 's_new': self._s_new,
			'u_int': self._u_now, 'v_int': self._v_now,
			'mtg_now': self._mtg_now, 'mtg_new': self._mtg_new,
			'su_now': self._su_now, 'su_int': self._su_int,
			'sv_now': self._sv_now, 'sv_int': self._sv_int
		}
		if su_tnd_on:
			inputs['su_tnd'] = self._su_tnd
		if sv_tnd_on:
			inputs['sv_tnd'] = self._sv_tnd

		outputs = {'su_new': self._su_new, 'sv_new': self._sv_new}

		self._stencil_momentum = gt.NGStencil(
			definitions_func=self._stencil_momentum_defs,
			inputs=inputs,
			global_inputs={'eps': self._eps, 'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

	def _stencils_set_inputs(self, stage, timestep, state, tendencies):
		# shortcuts
		if tendencies is not None:
			s_tnd_on  = tendencies.get('air_isentropic_density', None) is not None
			su_tnd_on = tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = tendencies.get('y_momentum_isentropic', None) is not None
			q_tnd_on  = {tracer: tracer in tendencies for tracer in self._tracers}
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = False
			q_tnd_on = {tracer: False for tracer in self._tracers}

		# update the local time step
		if stage == 0:
			self._dt.value = (timestep / 3.0).total_seconds()
		elif stage == 1:
			self._dt.value = (0.5 * timestep).total_seconds()
		else:
			self._dt.value = timestep.total_seconds()

		# update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._s_int[...]  = state['air_isentropic_density'][...]
		self._u_now[...]  = state['x_velocity_at_u_locations'][...]
		self._v_now[...]  = state['y_velocity_at_v_locations'][...]
		self._su_int[...] = state['x_momentum_isentropic'][...]
		self._sv_int[...] = state['y_momentum_isentropic'][...]
		for tracer in self._tracers:
			self._sq_int[tracer][...] = state['s_' + tracer][...]
		if s_tnd_on:
			self._s_tnd[...]  = tendencies['air_isentropic_density'][...]
		if su_tnd_on:
			self._su_tnd[...] = tendencies['x_momentum_isentropic'][...]
		if sv_tnd_on:
			self._sv_tnd[...] = tendencies['y_momentum_isentropic'][...]
		for tracer in self._tracers:
			if q_tnd_on[tracer]:
				self._q_tnd[tracer][...] = tendencies[tracer][...]

		if stage == 0:
			# update the Numpy arrays storing the current solution
			self._s_now[...]   = state['air_isentropic_density'][...]
			self._mtg_now[...] = state['montgomery_potential'][...]
			self._su_now[...]  = state['x_momentum_isentropic'][...]
			self._sv_now[...]  = state['y_momentum_isentropic'][...]
			for tracer in self._tracers:
				self._sq_now[tracer][...] = state['s_' + tracer][...]

	def _stencil_defs(
		self, dt, dx, dy, s_now, s_int, u_int, v_int, su_int, sv_int,
		s_tnd=None, **tracer_kwargs
	):
		tracer_symbols = {
			tracer: self._tracers[tracer]['stencil_symbol']
			for tracer in self._tracers
		}

		# horizontal indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output fields
		mtg_int = gt.Equation()
		s_new = gt.Equation()
		sq_new = {
			tracer: gt.Equation(name='s' + tracer_symbols[tracer] + '_new')
			for tracer in self._tracers
		}

		# retrieve tracers
		sq_now = {
			tracer: tracer_kwargs['s' + tracer_symbols[tracer] + '_now']
			for tracer in self._tracers
		}
		sq_int = {
			's' + tracer_symbols[tracer]:
				tracer_kwargs['s' + tracer_symbols[tracer] + '_int']
			for tracer in self._tracers
		}
		q_tnd = {
			tracer: tracer_kwargs.get(tracer_symbols[tracer] + '_tnd', None)
			for tracer in self._tracers
		}

		# calculations
		step_forward_euler(
			self._hflux, i, j, dt, dx, dy, s_now, s_int, s_new,
			u_int, v_int, mtg_int, su_int, sv_int, s_tnd=s_tnd,
			sq_now=sq_now, sq_int=sq_int, sq_new=sq_new, q_tnd=q_tnd
		)

		return_list = [s_new, ] + [sq_new[tracer] for tracer in self._tracers]

		return return_list

	def _stencil_momentum_defs(
		self, eps, dt, dx, dy, s_now, s_int, s_new, u_int, v_int, mtg_now, mtg_new,
		su_now, su_int, sv_now, sv_int, su_tnd=None, sv_tnd=None
	):
		# horizontal indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output fields
		mtg_int = gt.Equation()
		su_new = gt.Equation()
		sv_new = gt.Equation()

		# calculations
		step_forward_euler_momentum(
			self._hflux, eps, i, j, dt, dx, dy, s_now, s_int, s_new, u_int, v_int,
			mtg_now, mtg_int, mtg_new, su_now, su_int, su_new, sv_now, sv_int, sv_new,
			su_tnd=su_tnd, sv_tnd=sv_tnd
		)

		return su_new, sv_new


class SIL3(IsentropicPrognostic):
	""" The semi-implicit Lorenz three cycle scheme. """
	def __init__(
		self, horizontal_flux_scheme, grid, hb, tracers, backend, dtype, **kwargs
	):
		# call parent's constructor
		super().__init__(
			NGIsentropicMinimalHorizontalFlux, horizontal_flux_scheme,
			grid, hb, tracers, backend, dtype
		)

		# extract the upper boundary conditions on the pressure field and
		# the free coefficients of the scheme
		self._pt = kwargs['pt'].to_units('Pa').values.item() if 'pt' in kwargs else 0.0
		self._a = gt.Global(kwargs.get('a', 0.375))
		self._b = gt.Global(kwargs.get('b', 0.375))
		self._c = gt.Global(kwargs.get('c', 0.25))

		# instantiate the component retrieving the diagnostic variables
		self._diagnostics = IsentropicDiagnostics(grid, backend, dtype)

		# initialize the pointers to the stencils
		self._stencil_first_stage_slow = None
		self._stencil_first_stage_fast = None
		self._stencil_second_stage_slow = None
		self._stencil_second_stage_fast = None
		self._stencil_third_stage_slow = None
		self._stencil_third_stage_fast = None

	@property
	def stages(self):
		return 3

	def stage_call(self, stage, timestep, state, tendencies=None):
		tendencies = {} if tendencies is None else tendencies

		if self._stencil_first_stage_slow is None:
			# initialize the stencils
			self._stencils_initialize(tendencies)

		# set stencils' inputs
		self._stencils_set_inputs(stage, timestep, state, tendencies)

		# step the isentropic density and the water species
		if stage == 0:
			self._stencil_first_stage_slow.compute()
		elif stage == 1:
			self._stencil_second_stage_slow.compute()
		else:
			self._stencil_third_stage_slow.compute()

		# apply the boundary conditions on the stepped isentropic density
		dt = timestep / 3.0
		if stage == 0:
			try:
				self._hb.dmn_enforce_field(
					self._s1, 'air_isentropic_density', 'kg m^-2 K^-1',
					time=state['time']+dt
				)
			except AttributeError:
				self._hb.enforce_field(
					self._s1, 'air_isentropic_density', 'kg m^-2 K^-1',
					time=state['time']+dt
				)
		elif stage == 1:
			try:
				self._hb.dmn_enforce_field(
					self._s2, 'air_isentropic_density', 'kg m^-2 K^-1',
					time=state['time']+dt
				)
			except AttributeError:
				self._hb.enforce_field(
					self._s2, 'air_isentropic_density', 'kg m^-2 K^-1',
					time=state['time']+dt
				)
		else:
			try:
				self._hb.dmn_enforce_field(
					self._s_new, 'air_isentropic_density', 'kg m^-2 K^-1',
					time=state['time']+dt
				)
			except AttributeError:
				self._hb.enforce_field(
					self._s_new, 'air_isentropic_density', 'kg m^-2 K^-1',
					time=state['time']+dt
				)

		# diagnose the Montgomery potential from the stepped isentropic density,
		# then step the momenta
		if stage == 0:
			self._diagnostics.get_montgomery_potential(self._s1, self._pt, self._mtg1)
			self._stencil_first_stage_fast.compute()
		elif stage == 1:
			self._diagnostics.get_montgomery_potential(self._s2, self._pt, self._mtg2)
			self._stencil_second_stage_fast.compute()
		else:
			self._diagnostics.get_montgomery_potential(self._s_new, self._pt, self._mtg_new)
			self._stencil_third_stage_fast.compute()

		# collect the outputs
		out_state = {'time': state['time'] + dt}
		if stage == 0:
			out_state['air_isentropic_density'] = self._s1
			out_state['x_momentum_isentropic'] = self._su1
			out_state['y_momentum_isentropic'] = self._sv1
			out_state.update({
				's_' + tracer: self._sq1[tracer] for tracer in self._tracers
			})
		elif stage == 1:
			out_state['air_isentropic_density'] = self._s2
			out_state['x_momentum_isentropic'] = self._su2
			out_state['y_momentum_isentropic'] = self._sv2
			out_state.update({
				's_' + tracer: self._sq2[tracer] for tracer in self._tracers
			})
		else:
			out_state['air_isentropic_density'] = self._s_new
			out_state['x_momentum_isentropic'] = self._su_new
			out_state['y_momentum_isentropic'] = self._sv_new
			out_state.update({
				's_' + tracer: self._sq_new[tracer] for tracer in self._tracers
			})

		return out_state

	def _stencils_allocate(self, tendencies):
		super()._stencils_allocate(tendencies)

		# allocate the arrays which will store the intermediate values
		# of the model variables
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype
		self._s1   = np.zeros((nx, ny, nz), dtype=dtype)
		self._s2   = np.zeros((nx, ny, nz), dtype=dtype)
		self._u1   = np.zeros((nx+1, ny, nz), dtype=dtype)
		self._u2   = np.zeros((nx+1, ny, nz), dtype=dtype)
		self._v1   = np.zeros((nx, ny+1, nz), dtype=dtype)
		self._v2   = np.zeros((nx, ny+1, nz), dtype=dtype)
		self._mtg1 = np.zeros((nx, ny, nz), dtype=dtype)
		self._mtg2 = self._mtg1
		self._mtg_new = np.zeros((nx, ny, nz), dtype=dtype)
		self._su1  = np.zeros((nx, ny, nz), dtype=dtype)
		self._su2  = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv1  = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv2  = np.zeros((nx, ny, nz), dtype=dtype)
		self._sq1  = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._tracers
		}
		self._sq2  = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._tracers
		}

		# allocate the arrays which will store the physical tendencies
		if hasattr(self, '_s_tnd'):
			self._s_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
			self._s_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
		if hasattr(self, '_su_tnd'):
			self._su_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
			self._su_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
		if hasattr(self, '_sv_tnd'):
			self._sv_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
			self._sv_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
		self._q_tnd_1 = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._q_tnd
		}
		self._q_tnd_2 = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._q_tnd
		}

	def _stencils_initialize(self, tendencies):
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._hb.nb
		ts = {
			tracer: self._tracers[tracer]['stencil_symbol']
			for tracer in self._tracers
		}

		s_tnd_on  = 'air_isentropic_density' in tendencies
		su_tnd_on = 'x_momentum_isentropic' in tendencies
		sv_tnd_on = 'y_momentum_isentropic' in tendencies
		q_tnd_on = {tracer: tracer in tendencies for tracer in self._tracers}

		# allocate inputs and outputs
		self._stencils_allocate(tendencies)

		# initialize the stencil performing the slow mode of the first stage
		inputs = {
			's0': self._s_now,
			'u0': self._u_now,
			'v0': self._v_now,
			'su0': self._su_now,
			'sv0': self._sv_now,
		}
		if s_tnd_on:
			inputs['s_tnd_0'] = self._s_tnd
		inputs.update({
			's' + ts[tracer] + '0': self._sq_now[tracer]
			for tracer in self._tracers
		})
		inputs.update({
			ts[tracer] + '_tnd_0': self._q_tnd[tracer]
			for tracer in self._tracers if q_tnd_on[tracer]
		})

		outputs = {'s1': self._s1}
		outputs.update({
			's' + ts[tracer] + '1': self._sq1[tracer]
			for tracer in self._tracers
		})

		self._stencil_first_stage_slow = gt.NGStencil(
			definitions_func=self._stencil_first_stage_slow_defs,
			inputs=inputs,
			global_inputs={'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

		# initialize the stencil performing the fast mode of the first stage
		inputs = {
			's0': self._s_now, 's1': self._s1,
			'u0': self._u_now,
			'v0': self._v_now,
			'mtg0': self._mtg_now, 'mtg1': self._mtg1,
			'su0': self._su_now,
			'sv0': self._sv_now
		}
		if su_tnd_on:
			inputs['su_tnd_0'] = self._su_tnd
		if sv_tnd_on:
			inputs['sv_tnd_0'] = self._sv_tnd

		outputs = {'su1': self._su1, 'sv1': self._sv1}

		self._stencil_first_stage_fast = gt.NGStencil(
			definitions_func=self._stencil_first_stage_fast_defs,
			inputs=inputs,
			global_inputs={'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

		# initialize the stencil performing the slow mode of the second stage
		inputs = {
			's0': self._s_now, 's1': self._s1,
			'u0': self._u_now, 'u1': self._u1,
			'v0': self._v_now, 'v1': self._v1,
			'su0': self._su_now, 'su1': self._su1,
			'sv0': self._sv_now, 'sv1': self._sv1
		}
		if s_tnd_on:
			inputs['s_tnd_0'] = self._s_tnd
			inputs['s_tnd_1'] = self._s_tnd_1
		inputs.update({
			's' + ts[tracer] + '0': self._sq_now[tracer]
			for tracer in self._tracers
		})
		inputs.update({
			's' + ts[tracer] + '1': self._sq1[tracer]
			for tracer in self._tracers
		})
		inputs.update({
			ts[tracer] + '_tnd_0': self._q_tnd[tracer]
			for tracer in self._tracers if q_tnd_on[tracer]
		})
		inputs.update({
			ts[tracer] + '_tnd_1': self._q_tnd_1[tracer]
			for tracer in self._tracers if q_tnd_on[tracer]
		})

		outputs = {'s2': self._s2}
		outputs.update({
			's' + ts[tracer] + '2': self._sq2[tracer]
			for tracer in self._tracers
		})

		self._stencil_second_stage_slow = gt.NGStencil(
			definitions_func=self._stencil_second_stage_slow_defs,
			inputs=inputs,
			global_inputs={'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

		# initialize the stencil performing the fast mode of the second stage
		inputs = {
			's0': self._s_now, 's1': self._s1, 's2': self._s2,
			'u0': self._u_now, 'u1': self._u1,
			'v0': self._v_now, 'v1': self._v1,
			'mtg0': self._mtg_now, 'mtg2': self._mtg2,
			'su0': self._su_now, 'su1': self._su1,
			'sv0': self._sv_now, 'sv1': self._sv1
		}
		if su_tnd_on:
			inputs['su_tnd_0'] = self._su_tnd
			inputs['su_tnd_1'] = self._su_tnd_1
		if sv_tnd_on:
			inputs['sv_tnd_0'] = self._sv_tnd
			inputs['sv_tnd_1'] = self._sv_tnd_1
		outputs = {'su2': self._su2, 'sv2': self._sv2}
		self._stencil_second_stage_fast = gt.NGStencil(
			definitions_func=self._stencil_second_stage_fast_defs,
			inputs=inputs,
			global_inputs={'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

		# initialize the stencil performing the slow mode of the third stage
		inputs = {
			's0': self._s_now, 's1': self._s1, 's2': self._s2,
			'u0': self._u_now, 'u1': self._u1, 'u2': self._u2,
			'v0': self._v_now, 'v1': self._v1, 'v2': self._v2,
			'su0': self._su_now, 'su1': self._su1, 'su2': self._su2,
			'sv0': self._sv_now, 'sv1': self._sv1, 'sv2': self._sv2
		}
		if s_tnd_on:
			inputs['s_tnd_0'] = self._s_tnd
			inputs['s_tnd_1'] = self._s_tnd_1
			inputs['s_tnd_2'] = self._s_tnd_2
		inputs.update({
			's' + ts[tracer] + '0': self._sq_now[tracer]
			for tracer in self._tracers
		})
		inputs.update({
			's' + ts[tracer] + '1': self._sq1[tracer]
			for tracer in self._tracers
		})
		inputs.update({
			's' + ts[tracer] + '2': self._sq2[tracer]
			for tracer in self._tracers
		})
		inputs.update({
			ts[tracer] + '_tnd_0': self._q_tnd[tracer]
			for tracer in self._tracers if q_tnd_on[tracer]
		})
		inputs.update({
			ts[tracer] + '_tnd_1': self._q_tnd_1[tracer]
			for tracer in self._tracers if q_tnd_on[tracer]
		})
		inputs.update({
			ts[tracer] + '_tnd_2': self._q_tnd_2[tracer]
			for tracer in self._tracers if q_tnd_on[tracer]
		})

		outputs = {'s3': self._s_new}
		outputs.update({
			's' + ts[tracer] + '3': self._sq_new[tracer]
			for tracer in self._tracers
		})

		self._stencil_third_stage_slow = gt.NGStencil(
			definitions_func=self._stencil_third_stage_slow_defs,
			inputs=inputs,
			global_inputs={'dt': self._dt, 'dx': self._dx, 'dy': self._dy},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

		# initialize the stencil performing the fast mode of the third stage
		inputs = {
			's0': self._s_now, 's1': self._s1, 's2': self._s2, 's3': self._s_new,
			'u0': self._u_now, 'u1': self._u1, 'u2': self._u2,
			'v0': self._v_now, 'v1': self._v1, 'v2': self._v2,
			'mtg0': self._mtg_now, 'mtg2': self._mtg2, 'mtg3': self._mtg_new,
			'su0': self._su_now, 'su1': self._su1, 'su2': self._su2,
			'sv0': self._sv_now, 'sv1': self._sv1, 'sv2': self._sv2
		}
		if su_tnd_on:
			inputs['su_tnd_0'] = self._su_tnd
			inputs['su_tnd_1'] = self._su_tnd_1
			inputs['su_tnd_2'] = self._su_tnd_2
		if sv_tnd_on:
			inputs['sv_tnd_0'] = self._sv_tnd
			inputs['sv_tnd_1'] = self._sv_tnd_1
			inputs['sv_tnd_2'] = self._sv_tnd_2
		outputs = {'su3': self._su_new, 'sv3': self._sv_new}
		self._stencil_third_stage_fast = gt.NGStencil(
			definitions_func=self._stencil_third_stage_fast_defs,
			inputs=inputs,
			global_inputs={
				'dt': self._dt, 'dx': self._dx, 'dy': self._dy,
				'a': self._a, 'b': self._b, 'c': self._c
			},
			outputs=outputs,
			domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
			mode=self._backend
		)

	def _stencils_set_inputs(self, stage, timestep, state, tendencies):
		# shortcuts
		if tendencies is not None:
			s_tnd_on  = tendencies.get('air_isentropic_density', None) is not None
			su_tnd_on = tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = tendencies.get('y_momentum_isentropic', None) is not None
			q_tnd_on  = {tracer: tracer in tendencies for tracer in self._tracers}
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = False
			q_tnd_on = {tracer: False for tracer in self._tracers}

		# update the local time step
		self._dt.value = timestep.total_seconds()

		# update the Numpy arrays which serve as inputs to the GT4Py stencils
		if stage == 0:
			self._s_now[...]   = state['air_isentropic_density'][...]
			self._u_now[...]   = state['x_velocity_at_u_locations'][...]
			self._v_now[...]   = state['y_velocity_at_v_locations'][...]
			self._mtg_now[...] = state['montgomery_potential'][...]
			self._su_now[...]  = state['x_momentum_isentropic'][...]
			self._sv_now[...]  = state['y_momentum_isentropic'][...]
			for tracer in self._tracers:
				self._sq_now[tracer][...] = state['s_' + tracer][...]
			if s_tnd_on:
				self._s_tnd[...]  = tendencies['air_isentropic_density'][...]
			if su_tnd_on:
				self._su_tnd[...] = tendencies['x_momentum_isentropic'][...]
			if sv_tnd_on:
				self._sv_tnd[...] = tendencies['y_momentum_isentropic'][...]
			for tracer in self._tracers:
				if q_tnd_on[tracer]:
					self._q_tnd[tracer][...] = tendencies[tracer][...]
		elif stage == 1:
			self._s1[...]  = state['air_isentropic_density'][...]
			self._u1[...]  = state['x_velocity_at_u_locations'][...]
			self._v1[...]  = state['y_velocity_at_v_locations'][...]
			if 'montgomery_potential' in state:
				self._mtg1[...] = state['montgomery_potential'][...]
			self._su1[...] = state['x_momentum_isentropic'][...]
			self._sv1[...] = state['y_momentum_isentropic'][...]
			for tracer in self._tracers:
				self._sq1[tracer][...] = state['s_' + tracer][...]
			if s_tnd_on:
				self._s_tnd_1[...]  = tendencies['air_isentropic_density'][...]
			if su_tnd_on:
				self._su_tnd_1[...] = tendencies['x_momentum_isentropic'][...]
			if sv_tnd_on:
				self._sv_tnd_1[...] = tendencies['y_momentum_isentropic'][...]
			for tracer in self._tracers:
				if q_tnd_on[tracer]:
					self._q_tnd_1[tracer][...] = tendencies[tracer][...]
		else:
			self._s2[...]  = state['air_isentropic_density'][...]
			self._u2[...]  = state['x_velocity_at_u_locations'][...]
			self._v2[...]  = state['y_velocity_at_v_locations'][...]
			if 'montgomery_potential' in state:
				self._mtg2[...] = state['montgomery_potential'][...]
			self._su2[...] = state['x_momentum_isentropic'][...]
			self._sv2[...] = state['y_momentum_isentropic'][...]
			for tracer in self._tracers:
				self._sq2[tracer][...] = state['s_' + tracer][...]
			if s_tnd_on:
				self._s_tnd_2[...]  = tendencies['air_isentropic_density'][...]
			if su_tnd_on:
				self._su_tnd_2[...] = tendencies['x_momentum_isentropic'][...]
			if sv_tnd_on:
				self._sv_tnd_2[...] = tendencies['y_momentum_isentropic'][...]
			for tracer in self._tracers:
				if q_tnd_on[tracer]:
					self._q_tnd_2[tracer][...] = tendencies[tracer][...]

	def _stencil_first_stage_slow_defs(
		self, dt, dx, dy, s0, u0, v0, su0, sv0, s_tnd_0=None, **tracer_kwargs
	):
		ts = {
			tracer: self._tracers[tracer]['stencil_symbol']
			for tracer in self._tracers
		}

		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		sq0 = {
			's' + ts[tracer]: tracer_kwargs['s' + ts[tracer] + '0']
			for tracer in ts
		}
		q_tnd_0 = {
			ts[tracer] + '_tnd': tracer_kwargs.get(ts[tracer] + '_tnd_0', None)
			for tracer in ts
		}

		s1 = gt.Equation()
		sq1 = {
			tracer: gt.Equation(name='s' + ts[tracer] + '1')
			for tracer in ts
		}

		fluxes = self._hflux(
			i, j, dt, s0, u0, v0, su0, sv0, s_tnd=s_tnd_0,
			**sq0, **q_tnd_0
		)

		flux_s_x, flux_s_y = fluxes[0], fluxes[1]
		s1[i, j] = s0[i, j] - dt / 3.0 * (
			(flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
			(flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
			(s_tnd_0[i, j] if s_tnd_0 is not None else 0.0)
		)

		for idx, tracer in enumerate(ts.keys()):
			flux_sq_x, flux_sq_y = fluxes[6 + 2*idx], fluxes[6 + 2*idx + 1]
			sq1[tracer][i, j] = sq0['s' + ts[tracer]][i, j] - dt / 3.0 * (
				(flux_sq_x[i, j] - flux_sq_x[i-1, j]) / dx +
				(flux_sq_y[i, j] - flux_sq_y[i, j-1]) / dy -
				(s0[i, j] * q_tnd_0[ts[tracer] + '_tnd'][i, j]
				 if q_tnd_0[ts[tracer] + '_tnd'] is not None else 0.0)
			)

		return_list = [s1, ] + [sq1[tracer] for tracer in ts]

		return return_list

	def _stencil_first_stage_fast_defs(
		self, dt, dx, dy, s0, s1, u0, v0, mtg0, mtg1, su0, sv0,
		su_tnd_0=None, sv_tnd_0=None
	):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		sq = {
			's' + self._tracers[tracer]['stencil_symbol']: gt.Equation()
			for tracer in self._tracers
		}

		su1 = gt.Equation()
		sv1 = gt.Equation()

		fluxes = self._hflux(
			i, j, dt, s0, u0, v0, su0, sv0,
			su_tnd=su_tnd_0, sv_tnd=sv_tnd_0, **sq
		)

		flux_su_x, flux_su_y = fluxes[2], fluxes[3]
		flux_sv_x, flux_sv_y = fluxes[4], fluxes[5]

		su1[i, j] = su0[i, j] - dt / 3.0 * (
			(flux_su_x[i, j] - flux_su_x[i-1, j]) / dx +
			(flux_su_y[i, j] - flux_su_y[i, j-1]) / dy +
			0.5 * s0[i, j] * (mtg0[i+1, j] - mtg0[i-1, j]) / (2.0 * dx) +
			0.5 * s1[i, j] * (mtg1[i+1, j] - mtg1[i-1, j]) / (2.0 * dx) -
			(su_tnd_0[i, j] if su_tnd_0 is not None else 0.0)
		)
		sv1[i, j] = sv0[i, j] - dt / 3.0 * (
			(flux_sv_x[i, j] - flux_sv_x[i-1, j]) / dx +
			(flux_sv_y[i, j] - flux_sv_y[i, j-1]) / dy +
			0.5 * s0[i, j] * (mtg0[i, j+1] - mtg0[i, j-1]) / (2.0 * dy) +
			0.5 * s1[i, j] * (mtg1[i, j+1] - mtg1[i, j-1]) / (2.0 * dy) -
			(sv_tnd_0[i, j] if sv_tnd_0 is not None else 0.0)
		)

		return su1, sv1

	def _stencil_second_stage_slow_defs(
		self, dt, dx, dy, s0, s1, u0, u1, v0, v1, su0, su1, sv0, sv1,
		s_tnd_0=None, s_tnd_1=None, **tracer_kwargs
	):
		ts = {
			tracer: self._tracers[tracer]['stencil_symbol']
			for tracer in self._tracers
		}

		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		sq0 = {
			's' + ts[tracer]: tracer_kwargs['s' + ts[tracer] + '0']
			for tracer in ts
		}
		sq1 = {
			's' + ts[tracer]: tracer_kwargs['s' + ts[tracer] + '1']
			for tracer in ts
		}
		q_tnd_0 = {
			ts[tracer] + '_tnd': tracer_kwargs.get(ts[tracer] + '_tnd_0', None)
			for tracer in ts
		}
		q_tnd_1 = {
			ts[tracer] + '_tnd': tracer_kwargs.get(ts[tracer] + '_tnd_1', None)
			for tracer in ts
		}

		s2 = gt.Equation()
		sq2 = {
			tracer: gt.Equation(name='s' + ts[tracer] + '2')
			for tracer in ts
		}

		fluxes0 = self._hflux(
			i, j, dt, s0, u0, v0, su0, sv0, s_tnd=s_tnd_0,
			**sq0, **q_tnd_0
		)
		fluxes1 = self._hflux(
			i, j, dt, s1, u1, v1, su1, sv1, s_tnd=s_tnd_1,
			**sq1, **q_tnd_1
		)

		flux_s_x_0, flux_s_y_0 = fluxes0[0], fluxes0[1]
		flux_s_x_1, flux_s_y_1 = fluxes1[0], fluxes1[1]
		s2[i, j] = s0[i, j] - dt * (
			1.0 / 6.0 * (
				(flux_s_x_0[i, j] - flux_s_x_0[i-1, j]) / dx +
				(flux_s_y_0[i, j] - flux_s_y_0[i, j-1]) / dy -
				(s_tnd_0[i, j] if s_tnd_0 is not None else 0.0)
			) + 0.5 * (
				(flux_s_x_1[i, j] - flux_s_x_1[i-1, j]) / dx +
				(flux_s_y_1[i, j] - flux_s_y_1[i, j-1]) / dy -
				(s_tnd_1[i, j] if s_tnd_1 is not None else 0.0)
			)
		)

		for idx, tracer in enumerate(ts.keys()):
			flux_sq_x_0, flux_sq_y_0 = fluxes0[6 + 2*idx], fluxes0[6 + 2*idx + 1]
			flux_sq_x_1, flux_sq_y_1 = fluxes1[6 + 2*idx], fluxes1[6 + 2*idx + 1]
			sq2[tracer][i, j] = sq0['s' + ts[tracer]][i, j] - dt * (
				1.0 / 6.0 * (
					(flux_sq_x_0[i, j] - flux_sq_x_0[i-1, j]) / dx +
					(flux_sq_y_0[i, j] - flux_sq_y_0[i, j-1]) / dy -
					(s0[i, j] * q_tnd_0[ts[tracer] + '_tnd'][i, j]
					 if q_tnd_0[ts[tracer] + '_tnd'] is not None else 0.0)
				) + 0.5 * (
					(flux_sq_x_1[i, j] - flux_sq_x_1[i-1, j]) / dx +
					(flux_sq_y_1[i, j] - flux_sq_y_1[i, j-1]) / dy -
					(s1[i, j] * q_tnd_1[ts[tracer] + '_tnd'][i, j]
					 if q_tnd_1[ts[tracer] + '_tnd'] is not None else 0.0)
				)
			)

		return_list = [s2, ] + [sq2[tracer] for tracer in self._tracers]

		return return_list

	def _stencil_second_stage_fast_defs(
		self, dt, dx, dy, s0, s1, s2, u0, u1, v0, v1, mtg0, mtg2, su0, su1, sv0, sv1,
		su_tnd_0=None, su_tnd_1=None, sv_tnd_0=None, sv_tnd_1=None
	):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		su2 = gt.Equation()
		sv2 = gt.Equation()

		sq = {
			's' + self._tracers[tracer]['stencil_symbol']: gt.Equation()
			for tracer in self._tracers
		}

		fluxes0 = self._hflux(
			i, j, dt, s0, u0, v0, su0, sv0, su_tnd=su_tnd_0, sv_tnd=sv_tnd_0, **sq
		)
		fluxes1 = self._hflux(
			i, j, dt, s1, u1, v1, su1, sv1, su_tnd=su_tnd_1, sv_tnd=sv_tnd_1, **sq
		)

		flux_su_x_0, flux_su_y_0 = fluxes0[2], fluxes0[3]
		flux_su_x_1, flux_su_y_1 = fluxes1[2], fluxes1[3]
		flux_sv_x_0, flux_sv_y_0 = fluxes0[4], fluxes0[5]
		flux_sv_x_1, flux_sv_y_1 = fluxes1[4], fluxes1[5]

		su2[i, j] = su0[i, j] - dt * (
			1.0 / 6.0 * (
				(flux_su_x_0[i, j] - flux_su_x_0[i-1, j]) / dx +
				(flux_su_y_0[i, j] - flux_su_y_0[i, j-1]) / dy -
				(su_tnd_0[i, j] if su_tnd_0 is not None else 0.0)
			) + 0.5 * (
				(flux_su_x_1[i, j] - flux_su_x_1[i-1, j]) / dx +
				(flux_su_y_1[i, j] - flux_su_y_1[i, j-1]) / dy -
				(su_tnd_1[i, j] if su_tnd_1 is not None else 0.0)
			) +
			1.0 / 3.0 * s0[i, j] * (mtg0[i+1, j] - mtg0[i-1, j]) / (2.0 * dx) +
			1.0 / 3.0 * s2[i, j] * (mtg2[i+1, j] - mtg2[i-1, j]) / (2.0 * dx)

		)
		sv2[i, j] = sv0[i, j] - dt * (
			1.0 / 6.0 * (
				(flux_sv_x_0[i, j] - flux_sv_x_0[i-1, j]) / dx +
				(flux_sv_y_0[i, j] - flux_sv_y_0[i, j-1]) / dy -
				(sv_tnd_0[i, j] if sv_tnd_0 is not None else 0.0)
			) + 0.5 * (
				(flux_sv_x_1[i, j] - flux_sv_x_1[i-1, j]) / dx +
				(flux_sv_y_1[i, j] - flux_sv_y_1[i, j-1]) / dy -
				(sv_tnd_1[i, j] if sv_tnd_1 is not None else 0.0)
			) +
			1.0 / 3.0 * s0[i, j] * (mtg0[i, j+1] - mtg0[i, j-1]) / (2.0 * dy) +
			1.0 / 3.0 * s2[i, j] * (mtg2[i, j+1] - mtg2[i, j-1]) / (2.0 * dy)

		)

		return su2, sv2

	def _stencil_third_stage_slow_defs(
		self, dt, dx, dy, s0, s1, s2, u0, u1, u2, v0, v1, v2,
		su0, su1, su2, sv0, sv1, sv2,
		s_tnd_0=None, s_tnd_1=None, s_tnd_2=None, **tracer_kwargs
	):
		ts = {
			tracer: self._tracers[tracer]['stencil_symbol']
			for tracer in self._tracers
		}

		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		sq0 = {
			's' + ts[tracer]: tracer_kwargs['s' + ts[tracer] + '0']
			for tracer in ts
		}
		sq1 = {
			's' + ts[tracer]: tracer_kwargs['s' + ts[tracer] + '1']
			for tracer in ts
		}
		sq2 = {
			's' + ts[tracer]: tracer_kwargs['s' + ts[tracer] + '2']
			for tracer in ts
		}
		q_tnd_0 = {
			ts[tracer] + '_tnd': tracer_kwargs.get(ts[tracer] + '_tnd_0', None)
			for tracer in ts
		}
		q_tnd_1 = {
			ts[tracer] + '_tnd': tracer_kwargs.get(ts[tracer] + '_tnd_1', None)
			for tracer in ts
		}
		q_tnd_2 = {
			ts[tracer] + '_tnd': tracer_kwargs.get(ts[tracer] + '_tnd_2', None)
			for tracer in ts
		}

		s3 = gt.Equation()
		sq3 = {
			tracer: gt.Equation(name='s' + ts[tracer] + '3')
			for tracer in self._tracers
		}

		fluxes0 = self._hflux(
			i, j, dt, s0, u0, v0, su0, sv0, s_tnd=s_tnd_0, **sq0, **q_tnd_0
		)
		fluxes1 = self._hflux(
			i, j, dt, s1, u1, v1, su1, sv1, s_tnd=s_tnd_1, **sq1, **q_tnd_1
		)
		fluxes2 = self._hflux(
			i, j, dt, s2, u2, v2, su2, sv2, s_tnd=s_tnd_2, **sq2, **q_tnd_2
		)

		flux_s_x_0, flux_s_y_0 = fluxes0[0], fluxes0[1]
		flux_s_x_1, flux_s_y_1 = fluxes1[0], fluxes1[1]
		flux_s_x_2, flux_s_y_2 = fluxes2[0], fluxes2[1]
		s3[i, j] = s0[i, j] - dt * (
			0.5 * (
				(flux_s_x_0[i, j] - flux_s_x_0[i-1, j]) / dx +
				(flux_s_y_0[i, j] - flux_s_y_0[i, j-1]) / dy -
				(s_tnd_0[i, j] if s_tnd_0 is not None else 0.0)
			) - 0.5 * (
				(flux_s_x_1[i, j] - flux_s_x_1[i-1, j]) / dx +
				(flux_s_y_1[i, j] - flux_s_y_1[i, j-1]) / dy -
				(s_tnd_1[i, j] if s_tnd_1 is not None else 0.0)
			) + (
				(flux_s_x_2[i, j] - flux_s_x_2[i-1, j]) / dx +
				(flux_s_y_2[i, j] - flux_s_y_2[i, j-1]) / dy -
				(s_tnd_2[i, j] if s_tnd_2 is not None else 0.0)
			)
		)

		for idx, tracer in enumerate(ts.keys()):
			flux_sq_x_0, flux_sq_y_0 = fluxes0[6 + 2*idx], fluxes0[6 + 2*idx + 1]
			flux_sq_x_1, flux_sq_y_1 = fluxes1[6 + 2*idx], fluxes1[6 + 2*idx + 1]
			flux_sq_x_2, flux_sq_y_2 = fluxes2[6 + 2*idx], fluxes2[6 + 2*idx + 1]
			sq3[tracer][i, j] = sq0['s' + ts[tracer]][i, j] - dt * (
				0.5 * (
					(flux_sq_x_0[i, j] - flux_sq_x_0[i-1, j]) / dx +
					(flux_sq_y_0[i, j] - flux_sq_y_0[i, j-1]) / dy -
					(s0[i, j] * q_tnd_0[ts[tracer] + '_tnd'][i, j]
					 if q_tnd_0[ts[tracer] + '_tnd'] is not None else 0.0)
				) - 0.5 * (
					(flux_sq_x_1[i, j] - flux_sq_x_1[i-1, j]) / dx +
					(flux_sq_y_1[i, j] - flux_sq_y_1[i, j-1]) / dy -
					(s1[i, j] * q_tnd_1[ts[tracer] + '_tnd'][i, j]
					 if q_tnd_1[ts[tracer] + '_tnd'] is not None else 0.0)
				) + (
					(flux_sq_x_2[i, j] - flux_sq_x_2[i-1, j]) / dx +
					(flux_sq_y_2[i, j] - flux_sq_y_2[i, j-1]) / dy -
					(s2[i, j] * q_tnd_2[ts[tracer] + '_tnd'][i, j]
					 if q_tnd_2[ts[tracer] + '_tnd'] is not None else 0.0)
				)
			)

		return_list = [s3, ] + [sq3[tracer] for tracer in ts]

		return return_list

	def _stencil_third_stage_fast_defs(
		self, dt, dx, dy, a, b, c, s0, s1, s2, s3, u0, u1, u2, v0, v1, v2,
		mtg0, mtg2, mtg3, su0, su1, su2, sv0, sv1, sv2,
		su_tnd_0=None, su_tnd_1=None, su_tnd_2=None,
		sv_tnd_0=None, sv_tnd_1=None, sv_tnd_2=None
	):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		su3 = gt.Equation()
		sv3 = gt.Equation()

		sq = {
			's' + self._tracers[tracer]['stencil_symbol']: gt.Equation()
			for tracer in self._tracers
		}

		fluxes0 = self._hflux(
			i, j, dt, s0, u0, v0, su0, sv0, su_tnd=su_tnd_0, sv_tnd=sv_tnd_0, **sq
		)
		fluxes1 = self._hflux(
			i, j, dt, s1, u1, v1, su1, sv1, su_tnd=su_tnd_1, sv_tnd=sv_tnd_1, **sq
		)
		fluxes2 = self._hflux(
			i, j, dt, s2, u2, v2, su2, sv2, su_tnd=su_tnd_2, sv_tnd=sv_tnd_2, **sq
		)

		flux_su_x_0, flux_su_y_0 = fluxes0[2], fluxes0[3]
		flux_su_x_1, flux_su_y_1 = fluxes1[2], fluxes1[3]
		flux_su_x_2, flux_su_y_2 = fluxes2[2], fluxes2[3]
		flux_sv_x_0, flux_sv_y_0 = fluxes0[4], fluxes0[5]
		flux_sv_x_1, flux_sv_y_1 = fluxes1[4], fluxes1[5]
		flux_sv_x_2, flux_sv_y_2 = fluxes2[4], fluxes2[5]

		su3[i, j] = su0[i, j] - dt * (
			0.5 * (
				(flux_su_x_0[i, j] - flux_su_x_0[i-1, j]) / dx +
				(flux_su_y_0[i, j] - flux_su_y_0[i, j-1]) / dy -
				(su_tnd_0[i, j] if su_tnd_0 is not None else 0.0)
			) - 0.5 * (
				(flux_su_x_1[i, j] - flux_su_x_1[i-1, j]) / dx +
				(flux_su_y_1[i, j] - flux_su_y_1[i, j-1]) / dy -
				(su_tnd_1[i, j] if su_tnd_1 is not None else 0.0)
			) + (
				(flux_su_x_2[i, j] - flux_su_x_2[i-1, j]) / dx +
				(flux_su_y_2[i, j] - flux_su_y_2[i, j-1]) / dy -
				(su_tnd_2[i, j] if su_tnd_2 is not None else 0.0)
			) +
			a * s0[i, j] * (mtg0[i+1, j] - mtg0[i-1, j]) / (2.0 * dx) +
			b * s2[i, j] * (mtg2[i+1, j] - mtg2[i-1, j]) / (2.0 * dx) +
			c * s3[i, j] * (mtg3[i+1, j] - mtg3[i-1, j]) / (2.0 * dx)

		)
		sv3[i, j] = sv0[i, j] - dt * (
			0.5 * (
				(flux_sv_x_0[i, j] - flux_sv_x_0[i-1, j]) / dx +
				(flux_sv_y_0[i, j] - flux_sv_y_0[i, j-1]) / dy -
				(sv_tnd_0[i, j] if sv_tnd_0 is not None else 0.0)
			) - 0.5 * (
				(flux_sv_x_1[i, j] - flux_sv_x_1[i-1, j]) / dx +
				(flux_sv_y_1[i, j] - flux_sv_y_1[i, j-1]) / dy -
				(sv_tnd_1[i, j] if sv_tnd_1 is not None else 0.0)
			) + (
				(flux_sv_x_2[i, j] - flux_sv_x_2[i-1, j]) / dx +
				(flux_sv_y_2[i, j] - flux_sv_y_2[i, j-1]) / dy -
				(sv_tnd_2[i, j] if sv_tnd_2 is not None else 0.0)
			) +
			a * s0[i, j] * (mtg0[i, j+1] - mtg0[i, j-1]) / (2.0 * dy) +
			b * s2[i, j] * (mtg2[i, j+1] - mtg2[i, j-1]) / (2.0 * dy) +
			c * s3[i, j] * (mtg3[i, j+1] - mtg3[i, j-1]) / (2.0 * dy)
		)

		return su3, sv3

