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
	assume, given, HealthCheck, settings, strategies as hyp_st
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.isentropic.dynamics.vertical_fluxes import \
	NGIsentropicMinimalVerticalFlux
from tasmania.python.isentropic.dynamics.implementations.ng_minimal_vertical_fluxes import \
	Upwind, Centered, ThirdOrderUpwind, FifthOrderUpwind

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .utils import st_domain, st_floats, st_one_of, compare_arrays
except ModuleNotFoundError:
	from conf import backend as conf_backend  # nb as conf_nb
	from utils import st_domain, st_floats, st_one_of, compare_arrays

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

	def __call__(self, w, s, su, sv, sq0=None, sq1=None, sq2=None, sq3=None):
		mi, mj, mk = s.shape

		inputs = {'w': w, 's': s, 'su': su, 'sv': sv}
		if sq0 is not None:
			inputs['sq0'] = sq0
		if sq1 is not None:
			inputs['sq1'] = sq1
		if sq2 is not None:
			inputs['sq2'] = sq2
		if sq3 is not None:
			inputs['sq3'] = sq3

		self.flux_s  = np.zeros_like(s, dtype=s.dtype)
		self.flux_su = np.zeros_like(s, dtype=s.dtype)
		self.flux_sv = np.zeros_like(s, dtype=s.dtype)
		outputs = {
			'flux_s_z' : self.flux_s,
			'flux_su_z': self.flux_su,
			'flux_sv_z': self.flux_sv,
		}
		if sq0 is not None:
			self.flux_sq0 = np.zeros_like(s, dtype=s.dtype)
			outputs['flux_sq0_z'] = self.flux_sq0
		if sq1 is not None:
			self.flux_sq1 = np.zeros_like(s, dtype=s.dtype)
			outputs['flux_sq1_z'] = self.flux_sq1
		if sq2 is not None:
			self.flux_sq2 = np.zeros_like(s, dtype=s.dtype)
			outputs['flux_sq2_z'] = self.flux_sq2
		if sq3 is not None:
			self.flux_sq3 = np.zeros_like(s, dtype=s.dtype)
			outputs['flux_sq3_z'] = self.flux_sq3

		stencil = gt.NGStencil(
			definitions_func=self.stencil_defs,
			inputs=inputs,
			outputs=outputs,
			domain=gt.domain.Rectangle(
				(0, 0, self.nb), (mi-1, mj-1, mk-self.nb)),
			mode=self.backend,
		)

		stencil.compute()

	def stencil_defs(self, w, s, su, sv, sq0=None, sq1=None, sq2=None, sq3=None):
		k = gt.Index(axis=2)
		return self.core(k, w, s, su, sv, sq0=sq0, sq1=sq1, sq2=sq2, sq3=sq3)


def get_upwind_flux(w, phi):
	nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]

	f = np.zeros_like(phi, dtype=phi.dtype)

	for i in range(0, nx):
		for j in range(0, ny):
			for k in range(1, nz):
				f[i, j, k] = w[i, j, k] * (phi[i, j, k] if w[i, j, k] > 0 else phi[i, j, k-1])

	return f


def get_centered_flux(w, phi):
	f = np.zeros_like(phi, dtype=phi.dtype)

	f[:, :, 1:] = w[:, :, 1:-1] * 0.5 * (phi[:, :, :-1] + phi[:, :, 1:])

	return f


def get_third_order_upwind_flux(w, phi):
	f4 = np.zeros_like(phi, dtype=phi.dtype)

	f4[:, :, 2:-1] = w[:, :, 2:-2] / 12.0 * (
		7.0 * (phi[:, :, 1:-2] + phi[:, :, 2:-1]) -
		(phi[:, :, :-3] + phi[:, :, 3:])
	)

	f = np.zeros_like(phi, dtype=phi.dtype)

	f[:, :, 2:-1] = f4[:, :, 2:-1] - np.abs(w[:, :, 2:-2]) / 12.0 * (
		3.0 * (phi[:, :, 1:-2] - phi[:, :, 2:-1]) -
		(phi[:, :, :-3] - phi[:, :, 3:])
	)

	return f


def get_fifth_order_upwind_flux(w, phi):
	f6 = np.zeros_like(phi, dtype=phi.dtype)

	f6[:, :, 3:-2] = w[:, :, 3:-3] / 60.0 * (
		37.0 * (phi[:, :, 2:-3] + phi[:, :, 3:-2]) -
		8.0 * (phi[:, :, 1:-4] + phi[:, :, 4:-1]) +
		(phi[:, :, :-5] + phi[:, :, 5:])
	)

	f = np.zeros_like(phi, dtype=phi.dtype)

	f[:, :, 3:-2] = f6[:, :, 3:-2] - np.abs(w[:, :, 3:-3]) / 60.0 * (
		10.0 * (phi[:, :, 2:-3] - phi[:, :, 3:-2]) -
		5.0 * (phi[:, :, 1:-4] - phi[:, :, 4:-1]) +
		(phi[:, :, :-5] - phi[:, :, 5:])
	)

	return f


flux_properties = {
	'upwind': {'type': Upwind, 'get_fluxes': get_upwind_flux},
	'centered': {'type': Centered, 'get_fluxes': get_centered_flux},
	'third_order_upwind': {'type': ThirdOrderUpwind, 'get_fluxes': get_third_order_upwind_flux},
	'fifth_order_upwind': {'type': FifthOrderUpwind, 'get_fluxes': get_fifth_order_upwind_flux},
}


def validation(tracers, flux_scheme, domain, field, backend):
	grid = domain.numerical_grid
	flux_type = flux_properties[flux_scheme]['type']
	nb = flux_type.extent
	get_fluxes = flux_properties[flux_scheme]['get_fluxes']

	# ========================================
	# test interface
	# ========================================
	k = gt.Index(axis=2)

	w_eq   = gt.Equation(name='w')
	s_eq   = gt.Equation(name='s')
	su_eq  = gt.Equation(name='su')
	sv_eq  = gt.Equation(name='sv')
	sq0_eq = gt.Equation(name='sq0')
	sq1_eq = gt.Equation(name='sq1')
	sq2_eq = gt.Equation(name='sq2')
	sq3_eq = gt.Equation(name='sq3')

	fluxer = NGIsentropicMinimalVerticalFlux.factory(flux_scheme, grid, tracers)

	assert isinstance(fluxer, flux_type)

	out = fluxer(
		k, w_eq, s_eq, su_eq, sv_eq, sq0=sq0_eq, sq1=sq1_eq, sq2=sq2_eq, sq3=sq3_eq
	)

	assert len(out) == 3 + len(tracers)
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'flux_s_z'
	assert out[1].get_name() == 'flux_su_z'
	assert out[2].get_name() == 'flux_sv_z'
	for idx, tracer in enumerate(tracers.keys()):
		assert out[idx + 3].get_name() == \
			'flux_s' + tracers[tracer]['stencil_symbol'] + '_z'

	# ========================================
	# test numerics
	# ========================================
	w = field[1:, 1:, :]
	s = field[:-1, :-1, :-1]
	su = field[1:, :-1, :-1]
	sv = field[:-1, :-1, :-1]
	sq0 = field[:-1, :-1, 1:]
	sq1 = field[1:, :-1, 1:]
	sq2 = field[:-1, :-1, 1:]
	sq3 = field[1:, 1:, 1:]

	z = slice(nb, grid.nz-nb+1)

	ws = WrappingStencil(fluxer, nb, backend)
	ws(
		w, s, su, sv,
		sq0=sq0 if 'tracer0' in tracers else None,
		sq1=sq1 if 'tracer1' in tracers else None,
		sq2=sq2 if 'tracer2' in tracers else None,
		sq3=sq3 if 'tracer3' in tracers else None,
	)

	flux_s = get_fluxes(w, s)
	assert np.allclose(ws.flux_s[:, :, z], flux_s[:, :, z], equal_nan=True)

	flux_su = get_fluxes(w, su)
	assert np.allclose(ws.flux_su[:, :, z], flux_su[:, :, z], equal_nan=True)

	flux_sv = get_fluxes(w, sv)
	assert np.allclose(ws.flux_sv[:, :, z], flux_sv[:, :, z], equal_nan=True)

	if 'tracer0' in tracers:
		flux_sq0 = get_fluxes(w, sq0)
		compare_arrays(ws.flux_sq0[:, :, z], flux_sq0[:, :, z])

	if 'tracer1' in tracers:
		flux_sq1 = get_fluxes(w, sq1)
		compare_arrays(ws.flux_sq1[:, :, z], flux_sq1[:, :, z])

	if 'tracer2' in tracers:
		flux_sq2 = get_fluxes(w, sq2)
		compare_arrays(ws.flux_sq2[:, :, z], flux_sq2[:, :, z])

	if 'tracer3' in tracers:
		flux_sq3 = get_fluxes(w, sq3)
		compare_arrays(ws.flux_sq3[:, :, z], flux_sq3[:, :, z])


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
	domain = data.draw(st_domain(), label="domain")
	grid = domain.physical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		)
	)
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

	validation(tracers, 'upwind', domain, field, backend)


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
	domain = data.draw(st_domain(), label="domain")
	grid = domain.physical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		)
	)
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

	validation(tracers, 'centered', domain, field, backend)


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
	domain = data.draw(st_domain(), label="domain")
	grid = domain.physical_grid
	assume(grid.nz >= 5)
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		)
	)
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

	validation(tracers, 'third_order_upwind', domain, field, backend)


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
	domain = data.draw(st_domain(), label="domain")
	grid = domain.physical_grid
	assume(grid.nz >= 7)
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		)
	)
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

	validation(tracers, 'fifth_order_upwind', domain, field, backend)


if __name__ == '__main__':
	pytest.main([__file__])
