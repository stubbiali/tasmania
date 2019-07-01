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
from tasmania.python.isentropic.dynamics.fluxes import \
	IsentropicHorizontalFlux, IsentropicMinimalHorizontalFlux
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic


# convenient aliases
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


def step_forward_euler(
	fluxer, i, j, dt, dx, dy, s_now, s_int, s_new,
	u_int, v_int, mtg_int, su_int, sv_int,
	sqv_now=None, sqv_int=None, sqv_new=None,
	sqc_now=None, sqc_int=None, sqc_new=None,
	sqr_now=None, sqr_int=None, sqr_new=None,
	s_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
):
	if isinstance(fluxer, IsentropicMinimalHorizontalFlux):
		fluxes = fluxer(
			i, j, dt, s_int, u_int, v_int, su_int, sv_int,
			sqv=sqv_int, sqc=sqc_int, sqr=sqr_int,
			s_tnd=s_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd
		)
	else:
		fluxes = fluxer(
			i, j, dt, s_int, u_int, v_int, mtg_int, su_int, sv_int,
			sqv=sqv_int, sqc=sqc_int, sqr=sqr_int,
			s_tnd=s_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd
		)

	flux_s_x, flux_s_y = fluxes[0], fluxes[1]
	if sqv_now is not None:
		flux_sqv_x, flux_sqv_y = fluxes[6], fluxes[7]
		flux_sqc_x, flux_sqc_y = fluxes[8], fluxes[9]
		flux_sqr_x, flux_sqr_y = fluxes[10], fluxes[11]

	s_new[i, j] = s_now[i, j] - dt * (
		(flux_s_x[i, j] - flux_s_x[i-1, j]) / dx +
		(flux_s_y[i, j] - flux_s_y[i, j-1]) / dy -
		(s_tnd[i, j] if s_tnd is not None else 0.0)
	)

	if sqv_now is not None:
		sqv_new[i, j] = sqv_now[i, j] - dt * (
			(flux_sqv_x[i, j] - flux_sqv_x[i-1, j]) / dx +
			(flux_sqv_y[i, j] - flux_sqv_y[i, j-1]) / dy -
			(s_int[i, j] * qv_tnd[i, j] if qv_tnd is not None else 0.0)
		)

		sqc_new[i, j] = sqc_now[i, j] - dt * (
			(flux_sqc_x[i, j] - flux_sqc_x[i-1, j]) / dx +
			(flux_sqc_y[i, j] - flux_sqc_y[i, j-1]) / dy -
			(s_int[i, j] * qc_tnd[i, j] if qc_tnd is not None else 0.0)
		)

		sqr_new[i, j] = sqr_now[i, j] - dt * (
			(flux_sqr_x[i, j] - flux_sqr_x[i-1, j]) / dx +
			(flux_sqr_y[i, j] - flux_sqr_y[i, j-1]) / dy -
			(s_int[i, j] * qr_tnd[i, j] if qr_tnd is not None else 0.0)
		)


def step_forward_euler_momentum(
	fluxer, eps, i, j, dt, dx, dy, s_now, s_int, s_new, u_int, v_int,
	mtg_now, mtg_int, mtg_new, su_now, su_int, su_new, sv_now, sv_int, sv_new,
	su_tnd=None, sv_tnd=None
):
	sqv_int = gt.Equation()
	sqc_int = gt.Equation()
	sqr_int = gt.Equation()

	if isinstance(fluxer, IsentropicMinimalHorizontalFlux):
		fluxes = fluxer(
			i, j, dt, s_int, u_int, v_int, su_int, sv_int,
			sqv=sqv_int, sqc=sqc_int, sqr=sqr_int, su_tnd=su_tnd, sv_tnd=sv_tnd
		)
	else:
		fluxes = fluxer(
			i, j, dt, s_int, u_int, v_int, mtg_int, su_int, sv_int,
			sqv=sqv_int, sqc=sqc_int, sqr=sqr_int, su_tnd=su_tnd, sv_tnd=sv_tnd
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
		self, horizontal_flux_scheme, grid, hb, moist, backend, dtype, **kwargs
	):
		# call parent's constructor
		super().__init__(
			IsentropicMinimalHorizontalFlux, horizontal_flux_scheme,
			grid, hb, moist, backend, dtype
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
		if self._moist:
			out_state.update({
				'isentropic_density_of_water_vapor': self._sqv_new,
				'isentropic_density_of_cloud_liquid_water': self._sqc_new,
				'isentropic_density_of_precipitation_water': self._sqr_new
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

		s_tnd_on  = 'air_isentropic_density' in tendencies
		su_tnd_on = 'x_momentum_isentropic' in tendencies
		sv_tnd_on = 'y_momentum_isentropic' in tendencies
		qv_tnd_on = mfwv in tendencies
		qc_tnd_on = mfcw in tendencies
		qr_tnd_on = mfpw in tendencies

		# allocate inputs and outputs
		self._stencils_allocate(tendencies)

		# set inputs and outputs for the stencil advancing the isentropic
		# density and the water constituents, then instantiate the stencil
		inputs = {
			's_now': self._s_now, 'u_now': self._u_now, 'v_now': self._v_now,
			'su_now': self._su_now, 'sv_now': self._sv_now
		}
		if s_tnd_on:
			inputs['s_tnd'] = self._s_tnd
		outputs = {'s_new': self._s_new}
		if self._moist:
			inputs.update({
				'sqv_now': self._sqv_now, 'sqc_now': self._sqc_now,
				'sqr_now': self._sqr_now
			})
			if qv_tnd_on:
				inputs['qv_tnd'] = self._qv_tnd
			if qc_tnd_on:
				inputs['qc_tnd'] = self._qc_tnd
			if qr_tnd_on:
				inputs['qr_tnd'] = self._qr_tnd
			outputs.update({
				'sqv_new': self._sqv_new, 'sqc_new': self._sqc_new,
				'sqr_new': self._sqr_new
			})
		# instantiate the stencil advancing
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
		sqv_now=None, sqc_now=None, sqr_now=None,
		s_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		# horizontal indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output fields
		mtg_now = gt.Equation()
		s_new = gt.Equation()
		sqv_new = gt.Equation() if sqv_now is not None else None
		sqc_new = gt.Equation() if sqv_now is not None else None
		sqr_new = gt.Equation() if sqv_now is not None else None

		# calculations
		step_forward_euler(
			self._hflux, i, j, dt, dx, dy, s_now, s_now, s_new,
			u_now, v_now, mtg_now, su_now, sv_now,
			sqv_now=sqv_now, sqv_int=sqv_now, sqv_new=sqv_new,
			sqc_now=sqc_now, sqc_int=sqc_now, sqc_new=sqc_new,
			sqr_now=sqr_now, sqr_int=sqr_now, sqr_new=sqr_new,
			s_tnd=s_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd
		)

		if sqv_new is None:
			return s_new
		else:
			return s_new, sqv_new, sqc_new, sqr_new

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
		self, horizontal_flux_scheme, grid, hb, moist, backend, dtype, **kwargs
	):
		# call parent's constructor
		super().__init__(
			IsentropicMinimalHorizontalFlux, horizontal_flux_scheme,
			grid, hb, moist, backend, dtype
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
		if self._moist:
			out_state.update({
				'isentropic_density_of_water_vapor': self._sqv_new,
				'isentropic_density_of_cloud_liquid_water': self._sqc_new,
				'isentropic_density_of_precipitation_water': self._sqr_new
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
		self._s_int   = np.zeros((nx, ny, nz), dtype=dtype)
		self._su_int  = np.zeros((nx, ny, nz), dtype=dtype)
		self._sv_int  = np.zeros((nx, ny, nz), dtype=dtype)
		if self._moist:
			self._sqv_int = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqc_int = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqr_int = np.zeros((nx, ny, nz), dtype=dtype)

	def _stencils_initialize(self, tendencies):
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		nb = self._hb.nb

		s_tnd_on  = 'air_isentropic_density' in tendencies
		su_tnd_on = 'x_momentum_isentropic' in tendencies
		sv_tnd_on = 'y_momentum_isentropic' in tendencies
		qv_tnd_on = mfwv in tendencies
		qc_tnd_on = mfcw in tendencies
		qr_tnd_on = mfpw in tendencies

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
		outputs = {'s_new': self._s_new}
		if self._moist:
			inputs.update({
				'sqv_now': self._sqv_now, 'sqv_int': self._sqv_int,
				'sqc_now': self._sqc_now, 'sqc_int': self._sqc_int,
				'sqr_now': self._sqr_now, 'sqr_int': self._sqr_int,
			})
			if qv_tnd_on:
				inputs['qv_tnd'] = self._qv_tnd
			if qc_tnd_on:
				inputs['qc_tnd'] = self._qc_tnd
			if qr_tnd_on:
				inputs['qr_tnd'] = self._qr_tnd
			outputs.update({
				'sqv_new': self._sqv_new, 'sqc_new': self._sqc_new,
				'sqr_new': self._sqr_new
			})
		# instantiate the stencil advancing
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
			qv_tnd_on = tendencies.get(mfwv, None) is not None
			qc_tnd_on = tendencies.get(mfcw, None) is not None
			qr_tnd_on = tendencies.get(mfpw, None) is not None
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

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
		if self._moist:
			self._sqv_int[...] = state['isentropic_density_of_water_vapor'][...]
			self._sqc_int[...] = state['isentropic_density_of_cloud_liquid_water'][...]
			self._sqr_int[...] = state['isentropic_density_of_precipitation_water'][...]
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

		if stage == 0:
			# update the Numpy arrays storing the current solution
			self._s_now[...]   = state['air_isentropic_density'][...]
			self._mtg_now[...] = state['montgomery_potential'][...]
			self._su_now[...]  = state['x_momentum_isentropic'][...]
			self._sv_now[...]  = state['y_momentum_isentropic'][...]
			if self._moist:
				self._sqv_now[...] = state['isentropic_density_of_water_vapor'][...]
				self._sqc_now[...] = state['isentropic_density_of_cloud_liquid_water'][...]
				self._sqr_now[...] = state['isentropic_density_of_precipitation_water'][...]

	def _stencil_defs(
		self, dt, dx, dy, s_now, s_int, u_int, v_int, su_int, sv_int,
		sqv_now=None, sqv_int=None, sqc_now=None, sqc_int=None, sqr_now=None, sqr_int=None,
		s_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		# horizontal indices
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		# temporary and output fields
		mtg_int = gt.Equation()
		s_new = gt.Equation()
		sqv_new = gt.Equation() if sqv_now is not None else None
		sqc_new = gt.Equation() if sqv_now is not None else None
		sqr_new = gt.Equation() if sqv_now is not None else None

		# calculations
		step_forward_euler(
			self._hflux, i, j, dt, dx, dy, s_now, s_int, s_new,
			u_int, v_int, mtg_int, su_int, sv_int,
			sqv_now=sqv_now, sqv_int=sqv_int, sqv_new=sqv_new,
			sqc_now=sqc_now, sqc_int=sqc_int, sqc_new=sqc_new,
			sqr_now=sqr_now, sqr_int=sqr_int, sqr_new=sqr_new,
			s_tnd=s_tnd, qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd
		)

		if sqv_new is None:
			return s_new
		else:
			return s_new, sqv_new, sqc_new, sqr_new

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

