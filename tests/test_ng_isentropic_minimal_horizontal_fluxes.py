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
	given, HealthCheck, settings, strategies as hyp_st, reproduce_failure
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.isentropic.dynamics.fluxes import NGIsentropicMinimalHorizontalFlux
from tasmania.python.isentropic.dynamics.implementations.ng_minimal_horizontal_fluxes import \
	Upwind, Centered, ThirdOrderUpwind, FifthOrderUpwind

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .test_isentropic_minimal_horizontal_fluxes import \
		get_upwind_fluxes, get_centered_fluxes, get_maccormack_fluxes, \
		get_third_order_upwind_fluxes, get_fifth_order_upwind_fluxes
	from .utils import compare_arrays, st_domain, st_floats, st_one_of
except ModuleNotFoundError:
	from conf import backend as conf_backend  # nb as conf_nb
	from test_isentropic_minimal_horizontal_fluxes import \
		get_upwind_fluxes, get_centered_fluxes, get_maccormack_fluxes, \
		get_third_order_upwind_fluxes, get_fifth_order_upwind_fluxes
	from utils import compare_arrays, st_domain, st_floats, st_one_of

import sys
python_version = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
if python_version <= '3.5':
	import collections
	Dict = collections.OrderedDict
else:
	Dict = dict


class WrappingStencil:
	def __init__(self, core, nb, backend):
		self.core = core
		self.nb = nb
		self.backend = backend

	def __call__(
		self, dt, s, u, v, su, sv, s_tnd=None, su_tnd=None, sv_tnd=None,
		sq0=None, q0_tnd=None, sq1=None, q1_tnd=None,
		sq2=None, q2_tnd=None, sq3=None, q3_tnd=None
	):
		mi, mj, mk = s.shape

		__dt = gt.Global()
		__dt.value = dt
		global_inputs = {'dt': __dt}

		inputs = {'s': s, 'u': u, 'v': v, 'su': su, 'sv': sv}
		if s_tnd is not None:
			inputs['s_tnd'] = s_tnd
		if su_tnd is not None:
			inputs['su_tnd'] = su_tnd
		if sv_tnd is not None:
			inputs['sv_tnd'] = sv_tnd
		if sq0 is not None:
			inputs['sq0'] = sq0
		if sq1 is not None:
			inputs['sq1'] = sq1
		if sq2 is not None:
			inputs['sq2'] = sq2
		if sq3 is not None:
			inputs['sq3'] = sq3
		if q0_tnd is not None:
			inputs['q0_tnd'] = q0_tnd
		if q1_tnd is not None:
			inputs['q1_tnd'] = q1_tnd
		if q2_tnd is not None:
			inputs['q2_tnd'] = q2_tnd
		if q3_tnd is not None:
			inputs['q3_tnd'] = q3_tnd

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
		if sq0 is not None:
			self.flux_sq0_x = np.zeros_like(s, dtype=s.dtype)
			self.flux_sq0_y = np.zeros_like(s, dtype=s.dtype)
			outputs.update({
				'flux_sq0_x': self.flux_sq0_x, 'flux_sq0_y': self.flux_sq0_y,
			})
		if sq1 is not None:
			self.flux_sq1_x = np.zeros_like(s, dtype=s.dtype)
			self.flux_sq1_y = np.zeros_like(s, dtype=s.dtype)
			outputs.update({
				'flux_sq1_x': self.flux_sq1_x, 'flux_sq1_y': self.flux_sq1_y,
			})
		if sq2 is not None:
			self.flux_sq2_x = np.zeros_like(s, dtype=s.dtype)
			self.flux_sq2_y = np.zeros_like(s, dtype=s.dtype)
			outputs.update({
				'flux_sq2_x': self.flux_sq2_x, 'flux_sq2_y': self.flux_sq2_y,
			})
		if sq3 is not None:
			self.flux_sq3_x = np.zeros_like(s, dtype=s.dtype)
			self.flux_sq3_y = np.zeros_like(s, dtype=s.dtype)
			outputs.update({
				'flux_sq3_x': self.flux_sq3_x, 'flux_sq3_y': self.flux_sq3_y,
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
		self, dt, s, u, v, su, sv, s_tnd=None, su_tnd=None, sv_tnd=None,
		sq0=None, q0_tnd=None, sq1=None, q1_tnd=None,
		sq2=None, q2_tnd=None, sq3=None, q3_tnd=None
	):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		return self.core(
			i, j, dt, s, u, v, su, sv,
			s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd,
			sq0=sq0, q0_tnd=q0_tnd, sq1=sq1, q1_tnd=q1_tnd,
			sq2=sq2, q2_tnd=q2_tnd, sq3=sq3, q3_tnd=q3_tnd,
		)


def validation(tracers, flux_scheme, domain, field, timestep, backend):
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
	su_eq  = gt.Equation(name='su')
	sv_eq  = gt.Equation(name='sv')
	sq0_eq = gt.Equation(name='sq0')
	sq1_eq = gt.Equation(name='sq1')
	sq2_eq = gt.Equation(name='sq2')
	sq3_eq = gt.Equation(name='sq3')

	fluxer = NGIsentropicMinimalHorizontalFlux.factory(flux_scheme, grid, tracers)

	assert isinstance(fluxer, flux_type)

	out = fluxer(
		i, j, dt, s_eq, u_eq, v_eq, su_eq, sv_eq,
		sq0=sq0_eq, sq1=sq1_eq, sq2=sq2_eq, sq3=sq3_eq
	)

	assert len(out) == 6 + 2*len(tracers)
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'flux_s_x'
	assert out[1].get_name() == 'flux_s_y'
	assert out[2].get_name() == 'flux_su_x'
	assert out[3].get_name() == 'flux_su_y'
	assert out[4].get_name() == 'flux_sv_x'
	assert out[5].get_name() == 'flux_sv_y'
	for k, tracer in enumerate(tracers.keys()):
		assert out[6+2*k].get_name() == \
			'flux_s' + tracers[tracer]['stencil_symbol'] + '_x'
		assert out[6+2*k+1].get_name() == \
			'flux_s' + tracers[tracer]['stencil_symbol'] + '_y'

	# ========================================
	# test_numerics
	# ========================================
	s = field[:-1, :-1, :-1]
	u = field[:, :-1, :-1]
	v = field[:-1, :, :-1]
	su = field[1:, :-1, :-1]
	sv = field[:-1, :-1, :-1]
	sq0 = field[:-1, :-1, 1:]
	sq1 = field[1:, :-1, 1:]
	sq2 = field[:-1, :-1, 1:]
	sq3 = field[:-1, 1:, 1:]

	ws = WrappingStencil(fluxer, nb, backend)
	ws(timestep, s, u, v, su, sv, sq0=sq0, sq1=sq1, sq2=sq2, sq3=sq3)

	flux_s_x, flux_s_y = get_fluxes(u, v, s)
	compare_arrays(ws.flux_s_x[nb-1:-nb, nb-1:-nb], flux_s_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_s_y[nb-1:-nb, nb-1:-nb], flux_s_y[nb-1:-nb, nb-1:-nb])

	flux_su_x, flux_su_y = get_fluxes(u, v, su)
	compare_arrays(ws.flux_su_x[nb-1:-nb, nb-1:-nb], flux_su_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_su_y[nb-1:-nb, nb-1:-nb], flux_su_y[nb-1:-nb, nb-1:-nb])

	flux_sv_x, flux_sv_y = get_fluxes(u, v, sv)
	compare_arrays(ws.flux_sv_x[nb-1:-nb, nb-1:-nb], flux_sv_x[nb-1:-nb, nb-1:-nb])
	compare_arrays(ws.flux_sv_y[nb-1:-nb, nb-1:-nb], flux_sv_y[nb-1:-nb, nb-1:-nb])

	if 'tracer0' in tracers:
		flux_sq0_x, flux_sq0_y = get_fluxes(u, v, sq0)
		compare_arrays(ws.flux_sq0_x[nb-1:-nb, nb-1:-nb], flux_sq0_x[nb-1:-nb, nb-1:-nb])
		compare_arrays(ws.flux_sq0_y[nb-1:-nb, nb-1:-nb], flux_sq0_y[nb-1:-nb, nb-1:-nb])

	if 'tracer1' in tracers:
		flux_sq1_x, flux_sq1_y = get_fluxes(u, v, sq1)
		compare_arrays(ws.flux_sq1_x[nb-1:-nb, nb-1:-nb], flux_sq1_x[nb-1:-nb, nb-1:-nb])
		compare_arrays(ws.flux_sq1_y[nb-1:-nb, nb-1:-nb], flux_sq1_y[nb-1:-nb, nb-1:-nb])

	if 'tracer2' in tracers:
		flux_sq2_x, flux_sq2_y = get_fluxes(u, v, sq2)
		compare_arrays(ws.flux_sq2_x[nb-1:-nb, nb-1:-nb], flux_sq2_x[nb-1:-nb, nb-1:-nb])
		compare_arrays(ws.flux_sq2_y[nb-1:-nb, nb-1:-nb], flux_sq2_y[nb-1:-nb, nb-1:-nb])

	if 'tracer3' in tracers:
		flux_sq3_x, flux_sq3_y = get_fluxes(u, v, sq3)
		compare_arrays(ws.flux_sq3_x[nb-1:-nb, nb-1:-nb], flux_sq3_x[nb-1:-nb, nb-1:-nb])
		compare_arrays(ws.flux_sq3_y[nb-1:-nb, nb-1:-nb], flux_sq3_y[nb-1:-nb, nb-1:-nb])


flux_properties = {
	'upwind': {'type': Upwind, 'get_fluxes': get_upwind_fluxes},
	'centered': {'type': Centered, 'get_fluxes': get_centered_fluxes},
	'third_order_upwind': {'type': ThirdOrderUpwind, 'get_fluxes': get_third_order_upwind_fluxes},
	'fifth_order_upwind': {'type': FifthOrderUpwind, 'get_fluxes': get_fifth_order_upwind_fluxes},
}


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
	nb = 1  # nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb))
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
	q0_on = data.draw(hyp_st.booleans(), label="q0_on")
	q1_on = data.draw(hyp_st.booleans(), label="q1_on")
	q2_on = data.draw(hyp_st.booleans(), label="q2_on")
	q3_on = data.draw(hyp_st.booleans(), label="q3_on")
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	tracers = Dict()
	if q0_on:
		tracers['tracer0'] = {'stencil_symbol': 'q0'}
	if q1_on:
		tracers['tracer1'] = {'stencil_symbol': 'q1'}
	if q2_on:
		tracers['tracer2'] = {'stencil_symbol': 'q2'}
	if q3_on:
		tracers['tracer3'] = {'stencil_symbol': 'q3'}

	validation(tracers, 'upwind', domain, field, timestep, backend)


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
	nb = 1  # nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb))
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
	q0_on = data.draw(hyp_st.booleans(), label="q0_on")
	q1_on = data.draw(hyp_st.booleans(), label="q1_on")
	q2_on = data.draw(hyp_st.booleans(), label="q2_on")
	q3_on = data.draw(hyp_st.booleans(), label="q3_on")
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	tracers = Dict()
	if q0_on:
		tracers['tracer0'] = {'stencil_symbol': 'q0'}
	if q1_on:
		tracers['tracer1'] = {'stencil_symbol': 'q1'}
	if q2_on:
		tracers['tracer2'] = {'stencil_symbol': 'q2'}
	if q3_on:
		tracers['tracer3'] = {'stencil_symbol': 'q3'}

	validation(tracers, 'centered', domain, field, timestep, backend)


def _test_maccormack():
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
	nb = 2  # nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb))
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
	q0_on = data.draw(hyp_st.booleans(), label="q0_on")
	q1_on = data.draw(hyp_st.booleans(), label="q1_on")
	q2_on = data.draw(hyp_st.booleans(), label="q2_on")
	q3_on = data.draw(hyp_st.booleans(), label="q3_on")
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	tracers = Dict()
	if q0_on:
		tracers['tracer0'] = {'stencil_symbol': 'q0'}
	if q1_on:
		tracers['tracer1'] = {'stencil_symbol': 'q1'}
	if q2_on:
		tracers['tracer2'] = {'stencil_symbol': 'q2'}
	if q3_on:
		tracers['tracer3'] = {'stencil_symbol': 'q3'}

	validation(tracers, 'third_order_upwind', domain, field, timestep, backend)


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
	nb = 3  # nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb))
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
	q0_on = data.draw(hyp_st.booleans(), label="q0_on")
	q1_on = data.draw(hyp_st.booleans(), label="q1_on")
	q2_on = data.draw(hyp_st.booleans(), label="q2_on")
	q3_on = data.draw(hyp_st.booleans(), label="q3_on")
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	tracers = Dict()
	if q0_on:
		tracers['tracer0'] = {'stencil_symbol': 'q0'}
	if q1_on:
		tracers['tracer1'] = {'stencil_symbol': 'q1'}
	if q2_on:
		tracers['tracer2'] = {'stencil_symbol': 'q2'}
	if q3_on:
		tracers['tracer3'] = {'stencil_symbol': 'q3'}

	validation(tracers, 'fifth_order_upwind', domain, field, timestep, backend)


if __name__ == '__main__':
	pytest.main([__file__])
