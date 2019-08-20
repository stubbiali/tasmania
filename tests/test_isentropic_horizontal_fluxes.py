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
from hypothesis import \
	given, HealthCheck, settings, strategies as hyp_st
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.isentropic.dynamics.horizontal_fluxes import \
	IsentropicHorizontalFlux
from tasmania.python.isentropic.dynamics.implementations.horizontal_fluxes import \
	Upwind, Centered, ThirdOrderUpwind, FifthOrderUpwind

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .utils import st_domain, st_floats, st_one_of, compare_arrays
except (ImportError, ModuleNotFoundError):
	from conf import backend as conf_backend  # nb as conf_nb
	from utils import st_domain, st_floats, st_one_of


class WrappingStencil:
	def __init__(self, core, nb, backend):
		self.core = core
		self.nb = nb
		self.backend = backend

	def __call__(
		self, dt, s, u, v, mtg, su, sv, sqv=None, sqc=None, sqr=None,
		s_tnd=None, su_tnd=None, sv_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		mi, mj, mk = s.shape

		__dt = gt.Global()
		__dt.value = dt
		global_inputs = {'dt': __dt}

		inputs = {'s': s, 'u': u, 'v': v, 'mtg': mtg, 'su': su, 'sv': sv}
		if sqv is not None:
			inputs['sqv'] = sqv
			inputs['sqc'] = sqc
			inputs['sqr'] = sqr
		if s_tnd is not None:
			inputs['s_tnd'] = s_tnd
		if su_tnd is not None:
			inputs['su_tnd'] = su_tnd
		if sv_tnd is not None:
			inputs['sv_tnd'] = sv_tnd
		if qv_tnd is not None:
			inputs['qv_tnd'] = qv_tnd
		if qc_tnd is not None:
			inputs['qc_tnd'] = qc_tnd
		if qr_tnd is not None:
			inputs['qr_tnd'] = qr_tnd

		self.flux_s_x  = np.zeros_like(s, dtype=s.dtype)
		self.flux_s_y  = np.zeros_like(s, dtype=s.dtype)
		self.flux_su_x = np.zeros_like(s, dtype=s.dtype)
		self.flux_su_y = np.zeros_like(s, dtype=s.dtype)
		self.flux_sv_x = np.zeros_like(s, dtype=s.dtype)
		self.flux_sv_y = np.zeros_like(s, dtype=s.dtype)
		outputs = {
			'flux_s_x' : self.flux_s_x,  'flux_s_y' : self.flux_s_y,
			'flux_su_x': self.flux_su_x, 'flux_su_y': self.flux_su_y,
			'flux_sv_x': self.flux_sv_x, 'flux_sv_y': self.flux_sv_y,
		}
		if sqv is not None:
			self.flux_sqv_x = np.zeros_like(s, dtype=s.dtype)
			self.flux_sqv_y = np.zeros_like(s, dtype=s.dtype)
			self.flux_sqc_x = np.zeros_like(s, dtype=s.dtype)
			self.flux_sqc_y = np.zeros_like(s, dtype=s.dtype)
			self.flux_sqr_x = np.zeros_like(s, dtype=s.dtype)
			self.flux_sqr_y = np.zeros_like(s, dtype=s.dtype)
			outputs.update({
				'flux_sqv_x': self.flux_sqv_x, 'flux_sqv_y': self.flux_sqv_y,
				'flux_sqc_x': self.flux_sqc_x, 'flux_sqc_y': self.flux_sqc_y,
				'flux_sqr_x': self.flux_sqr_x, 'flux_sqr_y': self.flux_sqr_y,
			})

		stencil = gt.NGStencil(
			definitions_func=self.stencil_defs,
			inputs=inputs,
			global_inputs=global_inputs,
			outputs=outputs,
			domain=gt.domain.Rectangle(
				(self.nb-1, self.nb-1, 0), (mi-self.nb-1, mj-self.nb-1, mk-1)),
			mode=self.backend,
		)

		stencil.compute()

	def stencil_defs(
		self, dt, s, u, v, mtg, su, sv, sqv=None, sqc=None, sqr=None,
		s_tnd=None, su_tnd=None, sv_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		fs = self.core(
			i, j, dt, s, u, v, mtg, su, sv, sqv, sqc, sqr,
			s_tnd, su_tnd, sv_tnd, qv_tnd, qc_tnd, qr_tnd
		)

		if len(fs) == 6:
			return fs[0], fs[1], fs[2], fs[3], fs[4], fs[5]
		else:
			return fs[0], fs[1], fs[2], fs[3], fs[4], fs[5], \
				fs[6], fs[7], fs[8], fs[9], fs[10], fs[11]


def get_upwind_fluxes(u, v, phi):
	nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]

	fx = np.zeros_like(phi, dtype=phi.dtype)
	fy = np.zeros_like(phi, dtype=phi.dtype)

	for i in range(0, nx-1):
		for j in range(0, ny-1):
			for k in range(0, nz):
				fx[i, j, k] = u[i+1, j, k] * (phi[i, j, k] if u[i+1, j, k] > 0 else phi[i+1, j, k])
				fy[i, j, k] = v[i, j+1, k] * (phi[i, j, k] if v[i, j+1, k] > 0 else phi[i, j+1, k])

	return fx, fy


def get_centered_fluxes(u, v, phi):
	fx = np.zeros_like(phi, dtype=phi.dtype)
	fy = np.zeros_like(phi, dtype=phi.dtype)

	fx[:-1, :] = u[1:-1, :] * 0.5 * (phi[:-1, :] + phi[1:, :])
	fy[:, :-1] = v[:, 1:-1] * 0.5 * (phi[:, :-1] + phi[:, 1:])

	return fx, fy


def get_maccormack_fluxes():
	### TODO ###
	pass


def get_third_order_upwind_fluxes(u, v, phi):
	f4x = np.zeros_like(phi, dtype=phi.dtype)
	f4y = np.zeros_like(phi, dtype=phi.dtype)

	f4x[1:-2, :] = u[2:-2, :] / 12.0 * (
		7.0 * (phi[2:-1, :] + phi[1:-2, :]) -
		(phi[3:, :] + phi[:-3, :])
	)
	f4y[:, 1:-2] = v[:, 2:-2] / 12.0 * (
		7.0 * (phi[:, 2:-1] + phi[:, 1:-2]) -
		(phi[:, 3:] + phi[:, :-3])
	)

	fx = np.zeros_like(phi, dtype=phi.dtype)
	fy = np.zeros_like(phi, dtype=phi.dtype)

	fx[1:-2, :] = f4x[1:-2, :] - np.abs(u[2:-2, :]) / 12.0 * (
		3.0 * (phi[2:-1, :] - phi[1:-2, :]) -
		(phi[3:, :] - phi[:-3, :])
	)
	fy[:, 1:-2] = f4y[:, 1:-2] - np.abs(v[:, 2:-2]) / 12.0 * (
		3.0 * (phi[:, 2:-1] - phi[:, 1:-2]) -
		(phi[:, 3:] - phi[:, :-3])
	)

	return fx, fy


def get_fifth_order_upwind_fluxes(u, v, phi):
	f6x = np.zeros_like(phi, dtype=phi.dtype)
	f6y = np.zeros_like(phi, dtype=phi.dtype)

	f6x[2:-3, :] = u[3:-3, :] / 60.0 * (
		37.0 * (phi[3:-2, :] + phi[2:-3, :]) -
		8.0 * (phi[4:-1, :] + phi[1:-4, :]) +
		(phi[5:, :] + phi[:-5, :])
	)
	f6y[:, 2:-3] = v[:, 3:-3] / 60.0 * (
		37.0 * (phi[:, 3:-2] + phi[:, 2:-3]) -
		8.0 * (phi[:, 4:-1] + phi[:, 1:-4]) +
		(phi[:, 5:] + phi[:, :-5])
	)

	fx = np.zeros_like(phi, dtype=phi.dtype)
	fy = np.zeros_like(phi, dtype=phi.dtype)

	fx[2:-3, :] = f6x[2:-3, :] - np.abs(u[3:-3, :]) / 60.0 * (
		10.0 * (phi[3:-2, :] - phi[2:-3, :]) -
		5.0 * (phi[4:-1, :] - phi[1:-4, :]) +
		(phi[5:, :] - phi[:-5, :])
	)
	fy[:, 2:-3] = f6y[:, 2:-3] - np.abs(v[:, 3:-3]) / 60.0 * (
		10.0 * (phi[:, 3:-2] - phi[:, 2:-3]) -
		5.0 * (phi[:, 4:-1] - phi[:, 1:-4]) +
		(phi[:, 5:] - phi[:, :-5])
	)

	return fx, fy


flux_properties = {
	'upwind': {'type': Upwind, 'get_fluxes': get_upwind_fluxes},
	'centered': {'type': Centered, 'get_fluxes': get_centered_fluxes},
	'third_order_upwind': {'type': ThirdOrderUpwind, 'get_fluxes': get_third_order_upwind_fluxes},
	'fifth_order_upwind': {'type': FifthOrderUpwind, 'get_fluxes': get_fifth_order_upwind_fluxes},
}


def validation(flux_scheme, domain, field, timestep, backend):
	grid = domain.numerical_grid
	nb = domain.horizontal_boundary.nb
	flux_type = flux_properties[flux_scheme]['type']
	get_fluxes = flux_properties[flux_scheme]['get_fluxes']

	# ========================================
	# test interface
	# ========================================
	i = gt.Index()
	j = gt.Index()

	dt = gt.Global()

	s_eq   = gt.Equation(name='s')
	u_eq   = gt.Equation(name='u')
	v_eq   = gt.Equation(name='v')
	mtg_eq = gt.Equation(name='mtg')
	su_eq  = gt.Equation(name='su')
	sv_eq  = gt.Equation(name='sv')
	sqv_eq = gt.Equation(name='sqv')
	sqc_eq = gt.Equation(name='sqc')
	sqr_eq = gt.Equation(name='sqr')

	#
	# dry
	#
	fluxer_dry = IsentropicHorizontalFlux.factory(flux_scheme, grid, False)

	assert isinstance(fluxer_dry, flux_type)

	out = fluxer_dry(i, j, dt, s_eq, u_eq, v_eq, mtg_eq, su_eq, sv_eq)

	assert len(out) == 6
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'flux_s_x'
	assert out[1].get_name() == 'flux_s_y'
	assert out[2].get_name() == 'flux_su_x'
	assert out[3].get_name() == 'flux_su_y'
	assert out[4].get_name() == 'flux_sv_x'
	assert out[5].get_name() == 'flux_sv_y'

	#
	# moist
	#
	fluxer_moist = IsentropicHorizontalFlux.factory(flux_scheme, grid, True)

	assert isinstance(fluxer_moist, flux_type)

	out = fluxer_moist(
		i, j, dt, s_eq, u_eq, v_eq, mtg_eq, su_eq, sv_eq,
		sqv=sqv_eq, sqc=sqc_eq, sqr=sqr_eq
	)

	assert len(out) == 12
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'flux_s_x'
	assert out[1].get_name() == 'flux_s_y'
	assert out[2].get_name() == 'flux_su_x'
	assert out[3].get_name() == 'flux_su_y'
	assert out[4].get_name() == 'flux_sv_x'
	assert out[5].get_name() == 'flux_sv_y'
	assert out[6].get_name() == 'flux_sqv_x'
	assert out[7].get_name() == 'flux_sqv_y'
	assert out[8].get_name() == 'flux_sqc_x'
	assert out[9].get_name() == 'flux_sqc_y'
	assert out[10].get_name() == 'flux_sqr_x'
	assert out[11].get_name() == 'flux_sqr_y'

	# ========================================
	# test_numerics
	# ========================================
	s = field[:-1, :-1, :-1]
	u = field[:, :-1, :-1]
	v = field[:-1, :, :-1]
	mtg = field[:-1, 1:, :-1]
	su = field[1:, :-1, :-1]
	sv = field[:-1, :-1, :-1]
	sqv = field[:-1, :-1, 1:]
	sqc = field[1:, :-1, 1:]
	sqr = field[:-1, :-1, 1:]

	#
	# dry
	#
	ws = WrappingStencil(fluxer_dry, nb, backend)
	ws(timestep, s, u, v, mtg, su, sv)

	flux_s_x, flux_s_y = get_fluxes(u, v, s)
	compare_arrays(ws.flux_s_x[nb-1:-nb, nb-1:-nb], flux_s_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_s_y[nb-1:-nb, nb-1:-nb], flux_s_y[nb-1:-nb, nb-1:-nb])

	flux_su_x, flux_su_y = get_fluxes(u, v, su)
	compare_arrays(ws.flux_su_x[nb-1:-nb, nb-1:-nb], flux_su_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_su_y[nb-1:-nb, nb-1:-nb], flux_su_y[nb-1:-nb, nb-1:-nb])

	flux_sv_x, flux_sv_y = get_fluxes(u, v, sv)
	compare_arrays(ws.flux_sv_x[nb-1:-nb, nb-1:-nb], flux_sv_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_sv_y[nb-1:-nb, nb-1:-nb], flux_sv_y[nb-1:-nb, nb-1:-nb])

	#
	# moist
	#
	ws = WrappingStencil(fluxer_moist, nb, backend)
	ws(timestep, s, u, v, mtg, su, sv, sqv=sqv, sqc=sqc, sqr=sqr)

	compare_arrays(ws.flux_s_x[nb-1:-nb, nb-1:-nb], flux_s_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_s_y[nb-1:-nb, nb-1:-nb], flux_s_y[nb-1:-nb, nb-1:-nb])

	compare_arrays(ws.flux_su_x[nb-1:-nb, nb-1:-nb], flux_su_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_su_y[nb-1:-nb, nb-1:-nb], flux_su_y[nb-1:-nb, nb-1:-nb])

	compare_arrays(ws.flux_sv_x[nb-1:-nb, nb-1:-nb], flux_sv_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_sv_y[nb-1:-nb, nb-1:-nb], flux_sv_y[nb-1:-nb, nb-1:-nb])

	flux_sqv_x, flux_sqv_y = get_fluxes(u, v, sqv)
	compare_arrays(ws.flux_sqv_x[nb-1:-nb, nb-1:-nb], flux_sqv_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_sqv_y[nb-1:-nb, nb-1:-nb], flux_sqv_y[nb-1:-nb, nb-1:-nb])

	flux_sqc_x, flux_sqc_y = get_fluxes(u, v, sqc)
	compare_arrays(ws.flux_sqc_x[nb-1:-nb, nb-1:-nb], flux_sqc_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_sqc_y[nb-1:-nb, nb-1:-nb], flux_sqc_y[nb-1:-nb, nb-1:-nb])

	flux_sqr_x, flux_sqr_y = get_fluxes(u, v, sqr)
	compare_arrays(ws.flux_sqr_x[nb-1:-nb, nb-1:-nb], flux_sqr_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_sqr_y[nb-1:-nb, nb-1:-nb], flux_sqr_y[nb-1:-nb, nb-1:-nb])


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_upwind(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 1  # nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf.nb))
	domain = data.draw(st_domain(nb=nb), label="domain")
	grid = domain.numerical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		)
	)
	timestep = data.draw(st_floats(min_value=0, max_value=3600))
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	validation('upwind', domain, field, timestep, backend)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_centered(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 1  # nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf.nb))
	domain = data.draw(st_domain(nb=nb), label="domain")
	grid = domain.numerical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		)
	)
	timestep = data.draw(st_floats(min_value=0, max_value=3600))
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	validation('centered', domain, field, timestep, backend)


def test_maccormack():
	### TODO ###
	pass


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_third_order_upwind(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 2  # nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf.nb))
	domain = data.draw(st_domain(nb=nb), label="domain")
	grid = domain.numerical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		)
	)
	timestep = data.draw(st_floats(min_value=0, max_value=3600))
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	validation('third_order_upwind', domain, field, timestep, backend)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_fifth_order_upwind(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 3  # nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf.nb))
	domain = data.draw(st_domain(nb=nb), label="domain")
	grid = domain.numerical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		)
	)
	timestep = data.draw(st_floats(min_value=0, max_value=3600))
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	validation('fifth_order_upwind', domain, field, timestep, backend)


if __name__ == '__main__':
	pytest.main([__file__])
