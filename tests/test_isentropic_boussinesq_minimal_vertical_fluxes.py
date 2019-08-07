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
from tasmania.python.isentropic.dynamics.horizontal_fluxes import \
	IsentropicBoussinesqMinimalVerticalFlux
from tasmania.python.isentropic.dynamics.implementations.boussinesq_minimal_vertical_fluxes import \
	Upwind, Centered, ThirdOrderUpwind, FifthOrderUpwind

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .test_ng_isentropic_minimal_vertical_fluxes import \
		get_upwind_flux, get_centered_flux, \
		get_third_order_upwind_flux, get_fifth_order_upwind_flux
	from .utils import compare_arrays, st_domain, st_floats, st_one_of
except ModuleNotFoundError:
	from conf import backend as conf_backend  # nb as conf_nb
	from test_isentropic_minimal_vertical_fluxes import \
		get_upwind_flux, get_centered_flux, \
		get_third_order_upwind_flux, get_fifth_order_upwind_flux
	from utils import compare_arrays, st_domain, st_floats, st_one_of


class WrappingStencil:
	def __init__(self, core, nb, backend):
		self.core = core
		self.nb = nb
		self.backend = backend

	def __call__(self, w, s, su, sv, ddmtg, sqv=None, sqc=None, sqr=None):
		mi, mj, mk = s.shape

		inputs = {'w': w, 's': s, 'su': su, 'sv': sv, 'ddmtg': ddmtg}
		if sqv is not None:
			inputs['sqv'] = sqv
			inputs['sqc'] = sqc
			inputs['sqr'] = sqr

		self.flux_s     = np.zeros_like(s, dtype=s.dtype)
		self.flux_su    = np.zeros_like(s, dtype=s.dtype)
		self.flux_sv    = np.zeros_like(s, dtype=s.dtype)
		self.flux_ddmtg = np.zeros_like(s, dtype=s.dtype)
		outputs = {
			'flux_s_z'    : self.flux_s,
			'flux_su_z'   : self.flux_su,
			'flux_sv_z'   : self.flux_sv,
			'flux_ddmtg_z': self.flux_ddmtg
		}
		if sqv is not None:
			self.flux_sqv = np.zeros_like(s, dtype=s.dtype)
			self.flux_sqc = np.zeros_like(s, dtype=s.dtype)
			self.flux_sqr = np.zeros_like(s, dtype=s.dtype)
			outputs.update({
				'flux_sqv_z': self.flux_sqv,
				'flux_sqc_z': self.flux_sqc,
				'flux_sqr_z': self.flux_sqr,
			})

		stencil = gt.NGStencil(
			definitions_func=self.stencil_defs,
			inputs=inputs,
			outputs=outputs,
			domain=gt.domain.Rectangle(
				(0, 0, self.nb), (mi-1, mj-1, mk-self.nb)),
			mode=self.backend,
		)

		stencil.compute()

	def stencil_defs(self, w, s, su, sv, ddmtg, sqv=None, sqc=None, sqr=None):
		k = gt.Index(axis=2)

		fs = self.core(k, w, s, su, sv, ddmtg, sqv, sqc, sqr)

		if len(fs) == 4:
			return fs[0], fs[1], fs[2], fs[3]
		else:
			return fs[0], fs[1], fs[2], fs[3], fs[4], fs[5], fs[6]


flux_properties = {
	'upwind':
		{'type': Upwind, 'get_fluxes': get_upwind_flux},
	'centered':
		{'type': Centered, 'get_fluxes': get_centered_flux},
	'third_order_upwind':
		{'type': ThirdOrderUpwind, 'get_fluxes': get_third_order_upwind_flux},
	'fifth_order_upwind':
		{'type': FifthOrderUpwind, 'get_fluxes': get_fifth_order_upwind_flux},
}


def validation(flux_scheme, domain, field, backend):
	grid = domain.numerical_grid
	flux_type = flux_properties[flux_scheme]['type']
	nb = flux_type.extent
	get_fluxes = flux_properties[flux_scheme]['get_fluxes']

	# ========================================
	# test interface
	# ========================================
	k = gt.Index(axis=2)

	w_eq     = gt.Equation(name='w')
	s_eq     = gt.Equation(name='s')
	su_eq    = gt.Equation(name='su')
	sv_eq    = gt.Equation(name='sv')
	ddmtg_eq = gt.Equation(name='ddmtg')
	sqv_eq   = gt.Equation(name='sqv')
	sqc_eq   = gt.Equation(name='sqc')
	sqr_eq   = gt.Equation(name='sqr')

	#
	# dry
	#
	fluxer_dry = IsentropicBoussinesqMinimalVerticalFlux.factory(
		flux_scheme, grid, False
	)

	assert isinstance(fluxer_dry, flux_type)

	out = fluxer_dry(k, w_eq, s_eq, su_eq, sv_eq, ddmtg_eq)

	assert len(out) == 4
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'flux_s_z'
	assert out[1].get_name() == 'flux_su_z'
	assert out[2].get_name() == 'flux_sv_z'
	assert out[3].get_name() == 'flux_ddmtg_z'

	#
	# moist
	#
	fluxer_moist = IsentropicBoussinesqMinimalVerticalFlux.factory(
		flux_scheme, grid, True
	)

	assert isinstance(fluxer_moist, flux_type)

	out = fluxer_moist(
		k, w_eq, s_eq, su_eq, sv_eq, ddmtg_eq, sqv=sqv_eq, sqc=sqc_eq, sqr=sqr_eq
	)

	assert len(out) == 7
	assert all(isinstance(obj, gt.Equation) for obj in out)
	assert out[0].get_name() == 'flux_s_z'
	assert out[1].get_name() == 'flux_su_z'
	assert out[2].get_name() == 'flux_sv_z'
	assert out[3].get_name() == 'flux_ddmtg_z'
	assert out[4].get_name() == 'flux_sqv_z'
	assert out[5].get_name() == 'flux_sqc_z'
	assert out[6].get_name() == 'flux_sqr_z'

	# ========================================
	# test_numerics
	# ========================================
	w = field[1:, 1:, :]
	s = field[:-1, :-1, :-1]
	su = field[1:, :-1, :-1]
	sv = field[:-1, :-1, :-1]
	ddmtg = field[1:, 1:, 1:]
	sqv = field[:-1, :-1, 1:]
	sqc = field[1:, :-1, 1:]
	sqr = field[:-1, :-1, 1:]

	z = slice(nb, grid.nz-nb+1)

	#
	# dry
	#
	ws = WrappingStencil(fluxer_dry, nb, backend)
	ws(w, s, su, sv, ddmtg)

	flux_s = get_fluxes(w, s)
	compare_arrays(ws.flux_s[:, :, z], flux_s[:, :, z])

	flux_su = get_fluxes(w, su)
	compare_arrays(ws.flux_su[:, :, z], flux_su[:, :, z])

	flux_sv = get_fluxes(w, sv)
	compare_arrays(ws.flux_sv[:, :, z], flux_sv[:, :, z])

	flux_ddmtg = get_fluxes(w, ddmtg)
	compare_arrays(ws.flux_ddmtg[:, :, z], flux_ddmtg[:, :, z])

	#
	# moist
	#
	ws = WrappingStencil(fluxer_moist, nb, backend)
	ws(w, s, su, sv, ddmtg, sqv=sqv, sqc=sqc, sqr=sqr)

	compare_arrays(ws.flux_s[:, :, z], flux_s[:, :, z])

	compare_arrays(ws.flux_su[:, :, z], flux_su[:, :, z])

	compare_arrays(ws.flux_sv[:, :, z], flux_sv[:, :, z])

	compare_arrays(ws.flux_ddmtg[:, :, z], flux_ddmtg[:, :, z])

	flux_sqv = get_fluxes(w, sqv)
	compare_arrays(ws.flux_sqv[:, :, z], flux_sqv[:, :, z])

	flux_sqc = get_fluxes(w, sqc)
	compare_arrays(ws.flux_sqc[:, :, z], flux_sqc[:, :, z])

	flux_sqr = get_fluxes(w, sqr)
	compare_arrays(ws.flux_sqr[:, :, z], flux_sqr[:, :, z])


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
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	validation('upwind', domain, field, backend)


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
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	validation('centered', domain, field, backend)


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
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	validation('third_order_upwind', domain, field, backend)


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
	backend = data.draw(st_one_of(conf_backend), label="backend")

	# ========================================
	# test bed
	# ========================================
	validation('fifth_order_upwind', domain, field, backend)


if __name__ == '__main__':
	pytest.main([__file__])
