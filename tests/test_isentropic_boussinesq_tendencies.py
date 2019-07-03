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
	assume, given, HealthCheck, reproduce_failure, settings, strategies as hyp_st
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.isentropic.physics.boussinesq_tendencies import \
	NonconservativeFlux, Upwind, Centered, ThirdOrderUpwind, FifthOrderUpwind, \
	IsentropicBoussinesqTendency
from tasmania.python.utils.data_utils import make_dataarray_3d

try:
	from .conf import backend as conf_backend
	from .utils import compare_arrays, compare_dataarrays, st_floats, \
		st_one_of, st_domain, st_isentropic_boussinesq_state_ff
except ModuleNotFoundError:
	from conf import backend as conf_backend
	from utils import compare_arrays, compare_dataarrays, st_floats, \
		st_one_of, st_domain, st_isentropic_boussinesq_state_ff


mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


def get_upwind_fluxes(dx, dy, u, v, h):
	dtype = u.dtype
	slice_u = u[1:-1, 1:-1, 1:-1]
	slice_v = v[1:-1, 1:-1, 1:-1]

	flux_x = np.zeros(u.shape, dtype=dtype)
	flux_x[1:-1, 1:-1, 1:-1] = slice_u / dx * (
		(slice_u > 0) * (h[1:-1, 1:-1, 1:-1] - h[:-2, 1:-1, 1:-1]) +
		(slice_u < 0) * (h[2:, 1:-1, 1:-1] - h[1:-1, 1:-1, 1:-1])
	)

	flux_y = np.zeros(u.shape, dtype=dtype)
	flux_y[1:-1, 1:-1, 1:-1] = slice_v / dy * (
		(slice_v > 0) * (h[1:-1, 1:-1, 1:-1] - h[1:-1, :-2, 1:-1]) +
		(slice_v < 0) * (h[1:-1, 2:, 1:-1] - h[1:-1, 1:-1, 1:-1])
	)

	return flux_x, flux_y


def get_centered_fluxes(dx, dy, u, v, h):
	dtype = u.dtype
	slice_u = u[1:-1, 1:-1, 1:-1]
	slice_v = v[1:-1, 1:-1, 1:-1]

	flux_x = np.zeros(u.shape, dtype=dtype)
	flux_x[1:-1, 1:-1, 1:-1] = \
		slice_u * (h[2:, 1:-1, 1:-1] - h[:-2, 1:-1, 1:-1]) / (2.0 * dx)

	flux_y = np.zeros(u.shape, dtype=dtype)
	flux_y[1:-1, 1:-1, 1:-1] = \
		slice_v * (h[1:-1, 2:, 1:-1] - h[1:-1, :-2, 1:-1]) / (2.0 * dy)

	return flux_x, flux_y


def get_third_order_upwind_fluxes(dx, dy, u, v, h):
	dtype = u.dtype
	slice_u = u[2:-2, 2:-2, 1:-1]
	slice_v = v[2:-2, 2:-2, 1:-1]

	flux_x = np.zeros(u.shape, dtype=dtype)
	flux_x[2:-2, 2:-2, 1:-1] = \
		slice_u / (12.0 * dx) * (
			8.0 * (h[3:-1, 2:-2, 1:-1] - h[1:-3, 2:-2, 1:-1])
			- (h[4:, 2:-2, 1:-1] - h[:-4, 2:-2, 1:-1])
		) + \
		np.abs(slice_u) / (12.0 * dx) * (
			h[4:, 2:-2, 1:-1]
			- 4.0 * h[3:-1, 2:-2, 1:-1]
			+ 6.0 * h[2:-2, 2:-2, 1:-1]
			- 4.0 * h[1:-3, 2:-2, 1:-1]
			+ h[:-4, 2:-2, 1:-1]
		)

	flux_y = np.zeros(u.shape, dtype=dtype)
	flux_y[2:-2, 2:-2, 1:-1] = \
		slice_v / (12.0 * dy) * (
			8.0 * (h[2:-2, 3:-1, 1:-1] - h[2:-2, 1:-3, 1:-1])
			- (h[2:-2, 4:, 1:-1] - h[2:-2, :-4, 1:-1])
		) + \
		np.abs(slice_v) / (12.0 * dy) * (
			h[2:-2, 4:, 1:-1]
			- 4.0 * h[2:-2, 3:-1, 1:-1]
			+ 6.0 * h[2:-2, 2:-2, 1:-1]
			- 4.0 * h[2:-2, 1:-3, 1:-1]
			+ h[2:-2, :-4, 1:-1]
		)

	return flux_x, flux_y


def get_fifth_order_upwind_fluxes(dx, dy, u, v, h):
	dtype = u.dtype
	slice_u = u[3:-3, 3:-3, 1:-1]
	slice_v = v[3:-3, 3:-3, 1:-1]

	flux_x = np.zeros(u.shape, dtype=dtype)
	flux_x[3:-3, 3:-3, 1:-1] = \
		slice_u / (60.0 * dx) * (
			45.0 * (h[4:-2, 3:-3, 1:-1] - h[2:-4, 3:-3, 1:-1])
			- 9.0 * (h[5:-1, 3:-3, 1:-1] - h[1:-5, 3:-3, 1:-1])
			+ (h[6:, 3:-3, 1:-1] - h[:-6, 3:-3, 1:-1])
		) - \
		np.abs(slice_u) / (60.0 * dx) * (
			(h[6:, 3:-3, 1:-1] + h[:-6, 3:-3, 1:-1])
			- 6.0 * (h[5:-1, 3:-3, 1:-1] + h[1:-5, 3:-3, 1:-1])
			+ 15.0 * (h[4:-2, 3:-3, 1:-1] + h[2:-4, 3:-3, 1:-1])
			- 20.0 * h[3:-3, 3:-3, 1:-1]
		)

	flux_y = np.zeros(u.shape, dtype=dtype)
	flux_y[3:-3, 3:-3, 1:-1] = \
		slice_v / (60.0 * dy) * (
			45.0 * (h[3:-3, 4:-2, 1:-1] - h[3:-3, 2:-4, 1:-1])
			- 9.0 * (h[3:-3, 5:-1, 1:-1] - h[3:-3, 1:-5, 1:-1])
			+ (h[3:-3, 6:, 1:-1] - h[3:-3, :-6, 1:-1])
		) - \
		np.abs(slice_v) / (60.0 * dy) * (
			(h[3:-3, 6:, 1:-1] + h[3:-3, :-6, 1:-1])
			- 6.0 * (h[3:-3, 5:-1, 1:-1] + h[3:-3, 1:-5, 1:-1])
			+ 15.0 * (h[3:-3, 4:-2, 1:-1] + h[3:-3, 2:-4, 1:-1])
			- 20.0 * h[3:-3, 3:-3, 1:-1]
		)

	return flux_x, flux_y


class WrappingStencil:
	def __init__(self, flux_name, dx, dy, u, v, h, backend):
		dx_g = gt.Global(dx)
		dy_g = gt.Global(dy)

		self.core = NonconservativeFlux.factory(flux_name)
		nb = self.core.nb

		self.flux_x = np.zeros_like(u)
		self.flux_y = np.zeros_like(v)

		nx, ny, nz = u.shape[0], u.shape[1], u.shape[2]-1

		self.stencil = gt.NGStencil(
			definitions_func=self.stencil_defs,
			inputs={'u': u, 'v': v, 'h': h},
			global_inputs={'dx': dx_g, 'dy': dy_g},
			outputs={'flux_x': self.flux_x, 'flux_y': self.flux_y},
			domain=gt.domain.Rectangle((nb, nb, 1), (nx-nb-1, ny-nb-1, nz-1)),
			mode=backend
		)

		self.stencil.compute()

	def stencil_defs(self, dx, dy, u, v, h):
		i = gt.Index(axis=0)
		j = gt.Index(axis=1)

		flux_x = self.core.get_flux_x(i, j, dx, u, h)
		flux_y = self.core.get_flux_y(i, j, dy, v, h)

		return flux_x, flux_y


flux_providers = {
	'upwind': get_upwind_fluxes,
	'centered': get_centered_fluxes,
	'third_order_upwind': get_third_order_upwind_fluxes,
	'fifth_order_upwind': get_fifth_order_upwind_fluxes
}


def validation(flux_name, grid, field, backend):
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = field[:-1, :-1]
	v = field[:-1, 1:]
	h = field[1:, :-1]

	ws = WrappingStencil(flux_name, dx, dy, u, v, h, backend)
	flux_x_val, flux_y_val = flux_providers[flux_name](dx, dy, u, v, h)

	assert type(ws.core) == flux_classes[flux_name]
	compare_arrays(ws.flux_x, flux_x_val)
	compare_arrays(ws.flux_y, flux_y_val)


flux_classes = {
	'upwind': Upwind, 'centered': Centered,
	'third_order_upwind': ThirdOrderUpwind,
	'fifth_order_upwind': FifthOrderUpwind
}


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_upwind_flux(data):
	# ========================================
	# random data generation
	# ========================================
	flux_class = flux_classes['upwind']
	nb = flux_class.nb  # TODO: nb = hyp_st.integers(min_value=flux_class.nb, max_value=max(flux_class.nb, conf_nb))
	domain = data.draw(st_domain(nb=nb), label='domain')
	grid = domain.numerical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		), label='field'
	)
	backend = data.draw(st_one_of(conf_backend), label='backend')

	# ========================================
	# test bed
	# ========================================
	validation('upwind', grid, field, backend)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_centered_flux(data):
	# ========================================
	# random data generation
	# ========================================
	flux_class = flux_classes['centered']
	nb = flux_class.nb  # TODO: nb = hyp_st.integers(min_value=flux_class.nb, max_value=max(flux_class.nb, conf_nb))
	domain = data.draw(st_domain(nb=nb), label='domain')
	grid = domain.numerical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		), label='field'
	)
	backend = data.draw(st_one_of(conf_backend), label='backend')

	# ========================================
	# test bed
	# ========================================
	validation('centered', grid, field, backend)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_third_order_upwind_flux(data):
	# ========================================
	# random data generation
	# ========================================
	flux_class = flux_classes['third_order_upwind']
	nb = flux_class.nb  # TODO: nb = hyp_st.integers(min_value=flux_class.nb, max_value=max(flux_class.nb, conf_nb))
	domain = data.draw(st_domain(nb=nb), label='domain')
	grid = domain.numerical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		), label='field'
	)
	backend = data.draw(st_one_of(conf_backend), label='backend')

	# ========================================
	# test bed
	# ========================================
	validation('third_order_upwind', grid, field, backend)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_fifth_order_upwind_flux(data):
	# ========================================
	# random data generation
	# ========================================
	flux_class = flux_classes['fifth_order_upwind']
	nb = flux_class.nb  # TODO: nb = hyp_st.integers(min_value=flux_class.nb, max_value=max(flux_class.nb, conf_nb))
	domain = data.draw(st_domain(nb=nb), label='domain')
	grid = domain.numerical_grid
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=st_floats(),
			fill=hyp_st.nothing(),
		), label='field'
	)
	backend = data.draw(st_one_of(conf_backend), label='backend')

	# ========================================
	# test bed
	# ========================================
	validation('fifth_order_upwind', grid, field, backend)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_boussinesq_tendency(data):
	# ========================================
	# random data generation
	# ========================================
	flux_name = data.draw(st_one_of(flux_classes.keys()), label='flux_name')
	nb = flux_classes[flux_name].nb
	# TODO: nb = hyp_st.integers(min_value=flux_class.nb, max_value=max(flux_class.nb, conf_nb))
	domain = data.draw(st_domain(nb=nb), label='domain')
	grid = domain.numerical_grid
	moist = data.draw(hyp_st.booleans(), label='moist')
	state = data.draw(
		st_isentropic_boussinesq_state_ff(grid, moist=moist), label='state'
	)
	backend = data.draw(st_one_of(conf_backend), label='backend')
	dtype = grid.x.dtype

	# ========================================
	# test bed
	# ========================================
	ibt = IsentropicBoussinesqTendency(
		domain, advection_scheme=flux_name, moist=moist, backend=backend, dtype=dtype
	)

	assert 'air_isentropic_density' in ibt.input_properties
	assert 'dd_montgomery_potential' in ibt.input_properties
	assert 'height_on_interface_levels' in ibt.input_properties
	assert 'x_momentum_isentropic' in ibt.input_properties
	assert 'y_momentum_isentropic' in ibt.input_properties
	if moist:
		assert mfwv in ibt.input_properties
		assert mfcw in ibt.input_properties
		assert mfpw in ibt.input_properties
		assert len(ibt.input_properties) == 8
	else:
		assert len(ibt.input_properties) == 5

	assert 'air_isentropic_density' in ibt.tendency_properties
	assert 'dd_montgomery_potential' in ibt.tendency_properties
	assert 'x_momentum_isentropic' in ibt.tendency_properties
	assert 'y_momentum_isentropic' in ibt.tendency_properties
	if moist:
		assert mfwv in ibt.tendency_properties
		assert mfcw in ibt.tendency_properties
		assert mfpw in ibt.tendency_properties
		assert len(ibt.tendency_properties) == 7
	else:
		assert len(ibt.tendency_properties) == 4

	assert 'metric_term' in ibt.diagnostic_properties
	assert len(ibt.diagnostic_properties) == 1

	tendencies, diagnostics = ibt(state)

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	# u
	s = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values[...]
	su = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	u_val = np.zeros((nx, ny, nz+1), dtype=dtype)
	u_val[:, :, 1:-1] = 0.5 * (su[:, :, :-1] / s[:, :, :-1] + su[:, :, 1:] / s[:, :, 1:])
	compare_arrays(ibt._tmp_u, u_val)

	# v
	sv = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values[...]
	v_val = np.zeros((nx, ny, nz+1), dtype=dtype)
	v_val[:, :, 1:-1] = 0.5 * (sv[:, :, :-1] / s[:, :, :-1] + sv[:, :, 1:] / s[:, :, 1:])
	compare_arrays(ibt._tmp_v, v_val)

	# b
	h = state['height_on_interface_levels'].to_units('m').values[...]
	flux_x, flux_y = flux_providers[flux_name](dx, dy, u_val, v_val, h)
	b = flux_x + flux_y
	b_val = make_dataarray_3d(b, grid, 'm s^-1')
	assert 'metric_term' in diagnostics
	compare_dataarrays(
		diagnostics['metric_term'], b_val, compare_coordinate_values=False
	)
	assert len(diagnostics) == 1

	# s
	s_tnd = s * (b[:, :, :-1] - b[:, :, 1:]) / (h[:, :, :-1] - h[:, :, 1:])
	s_tnd_val = make_dataarray_3d(s_tnd, grid, 'kg m^-2 K^-1 s^-1')
	assert 'air_isentropic_density' in tendencies
	compare_dataarrays(
		tendencies['air_isentropic_density'], s_tnd_val,
		compare_coordinate_values=False
	)

	# ddmtg
	ddmtg = state['dd_montgomery_potential'].to_units('m^2 K^-2 s^-2').values
	ddmtg_tnd = ddmtg * (b[:, :, :-1] - b[:, :, 1:]) / (h[:, :, :-1] - h[:, :, 1:])
	ddmtg_tnd_val = make_dataarray_3d(ddmtg_tnd, grid, 'm^2 K^-2 s^-3')
	assert 'dd_montgomery_potential' in tendencies
	compare_dataarrays(
		tendencies['dd_montgomery_potential'], ddmtg_tnd_val,
		compare_coordinate_values=False
	)

	# su
	su_tnd = su * (b[:, :, :-1] - b[:, :, 1:]) / (h[:, :, :-1] - h[:, :, 1:])
	su_tnd_val = make_dataarray_3d(su_tnd, grid, 'kg m^-1 K^-1 s^-2')
	assert 'x_momentum_isentropic' in tendencies
	compare_dataarrays(
		tendencies['x_momentum_isentropic'], su_tnd_val,
		compare_coordinate_values=False
	)

	# sv
	sv_tnd = sv * (b[:, :, :-1] - b[:, :, 1:]) / (h[:, :, :-1] - h[:, :, 1:])
	sv_tnd_val = make_dataarray_3d(sv_tnd, grid, 'kg m^-1 K^-1 s^-2')
	assert 'y_momentum_isentropic' in tendencies
	compare_dataarrays(
		tendencies['y_momentum_isentropic'], sv_tnd_val,
		compare_coordinate_values=False
	)

	if moist:
		names = (mfwv, mfcw, mfpw)

		for name in names:
			q = state[name].to_units('g g^-1').values[...]
			q_tnd = q * (b[:, :, :-1] - b[:, :, 1:]) / (h[:, :, :-1] - h[:, :, 1:])
			q_tnd_val = make_dataarray_3d(q_tnd, grid, 'g g^-1 s^-1')
			assert name in tendencies
			compare_dataarrays(
				tendencies[name], q_tnd_val, compare_coordinate_values=False
			)

		assert len(tendencies) == 7
	else:
		assert len(tendencies) == 4


if __name__ == '__main__':
	pytest.main([__file__])
