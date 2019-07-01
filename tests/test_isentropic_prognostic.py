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
from copy import deepcopy
from datetime import timedelta
from hypothesis import \
	assume, given, HealthCheck, reproduce_failure, settings, strategies as hyp_st
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.isentropic.dynamics.diagnostics import \
	IsentropicDiagnostics
from tasmania.python.isentropic.dynamics.fluxes import \
	IsentropicMinimalHorizontalFlux
from tasmania.python.isentropic.dynamics.prognostic \
	import IsentropicPrognostic
from tasmania.python.isentropic.dynamics.implementations.prognostic \
	import ForwardEulerSI, CenteredSI, RK3WSSI
from tasmania.python.utils.data_utils import make_raw_state

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .test_isentropic_minimal_horizontal_fluxes import \
		get_upwind_fluxes, get_centered_fluxes, \
		get_third_order_upwind_fluxes, get_fifth_order_upwind_fluxes
	from .test_isentropic_minimal_prognostic import \
		forward_euler_step
	from .utils import compare_arrays, compare_datetimes, \
		st_domain, st_floats, st_one_of, st_isentropic_state_f
except ModuleNotFoundError:
	from conf import backend as conf_backend  # nb as conf_nb
	from test_isentropic_minimal_horizontal_fluxes import \
		get_upwind_fluxes, get_centered_fluxes, \
		get_third_order_upwind_fluxes, get_fifth_order_upwind_fluxes
	from test_isentropic_minimal_prognostic import \
		forward_euler_step
	from utils import compare_arrays, compare_datetimes, \
		st_domain, st_floats, st_one_of, st_isentropic_state_f


mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_factory(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(
		st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=3),
		label='domain'
	)
	moist = data.draw(hyp_st.booleans(), label='moist')
	backend = data.draw(st_one_of(conf_backend), label='backend')
	pt = DataArray(
		data.draw(st_floats(min_value=0, max_value=100), label='pt'), attrs={'units': 'Pa'}
	)
	eps = data.draw(st_floats(min_value=0, max_value=1), label='eps')

	# ========================================
	# test bed
	# ========================================
	grid = domain.numerical_grid
	hb = domain.horizontal_boundary
	dtype = grid.x.dtype

	imp_euler_si = IsentropicPrognostic.factory(
		'forward_euler_si', 'upwind', grid, hb, moist,
		backend=backend, dtype=dtype, pt=pt, eps=eps
	)
	assert isinstance(imp_euler_si, ForwardEulerSI)
	assert isinstance(imp_euler_si._hflux, IsentropicMinimalHorizontalFlux)

	imp_rk3ws_si = IsentropicPrognostic.factory(
		'rk3ws_si', 'fifth_order_upwind', grid, hb, moist,
		backend=backend, dtype=dtype, pt=pt, eps=eps
	)
	assert isinstance(imp_rk3ws_si, RK3WSSI)
	assert isinstance(imp_rk3ws_si._hflux, IsentropicMinimalHorizontalFlux)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_upwind_si(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 1  # TODO: nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb))
	domain = data.draw(
		st_domain(xaxis_length=(3, 30), yaxis_length=(3, 30), nb=nb),
		label='domain'
	)
	hb = domain.horizontal_boundary
	assume(hb.type != 'dirichlet')
	grid = domain.numerical_grid
	moist = data.draw(hyp_st.booleans(), label='moist')
	state = data.draw(st_isentropic_state_f(grid, moist=moist), label='state')
	tendencies = {}
	if data.draw(hyp_st.booleans(), label='s_tnd'):
		tendencies['air_isentropic_density'] = state['air_isentropic_density']
	if data.draw(hyp_st.booleans(), label='su_tnd'):
		tendencies['x_momentum_isentropic'] = state['x_momentum_isentropic']
	if data.draw(hyp_st.booleans(), label='sv_tn:'):
		tendencies['y_momentum_isentropic'] = state['y_momentum_isentropic']
	if moist:
		if data.draw(hyp_st.booleans(), label='qv_tnd'):
			tendencies[mfwv] = state[mfwv]
		if data.draw(hyp_st.booleans(), label='qc_tnd'):
			tendencies[mfcw] = state[mfcw]
		if data.draw(hyp_st.booleans(), label='qr_tnd'):
			tendencies[mfpw] = state[mfpw]

	backend = data.draw(st_one_of(conf_backend), label='backend')
	pt = state['air_pressure_on_interface_levels'][0, 0, 0]
	eps = data.draw(st_floats(min_value=0, max_value=1), label='eps')

	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(minutes=60)
		),
		label='timestep'
	)

	# ========================================
	# test bed
	# ========================================
	hb.reference_state = state
	dtype = grid.x.dtype

	imp = IsentropicPrognostic.factory(
		'forward_euler_si', 'upwind', grid, hb, moist,
		backend=backend, dtype=dtype, pt=pt, eps=eps
	)

	raw_state = make_raw_state(state)
	if moist:
		raw_state['isentropic_density_of_water_vapor'] = \
			raw_state['air_isentropic_density'] * raw_state[mfwv]
		raw_state['isentropic_density_of_cloud_liquid_water'] = \
			raw_state['air_isentropic_density'] * raw_state[mfcw]
		raw_state['isentropic_density_of_precipitation_water'] = \
			raw_state['air_isentropic_density'] * raw_state[mfpw]

	raw_tendencies = make_raw_state(tendencies)
	if moist:
		if mfwv in raw_tendencies:
			raw_tendencies['isentropic_density_of_water_vapor'] = \
				raw_state['air_isentropic_density'] * raw_tendencies[mfwv]
		if mfcw in raw_tendencies:
			raw_tendencies['isentropic_density_of_cloud_liquid_water'] = \
				raw_state['air_isentropic_density'] * raw_tendencies[mfcw]
		if mfpw in raw_tendencies:
			raw_tendencies['isentropic_density_of_precipitation_water'] = \
				raw_state['air_isentropic_density'] * raw_tendencies[mfpw]

	raw_state_new = imp.stage_call(0, timestep, raw_state, raw_tendencies)

	assert 'time' in raw_state_new.keys()
	compare_datetimes(raw_state_new['time'], raw_state['time'] + timestep)

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	dt = timestep.total_seconds()
	u_now = raw_state['x_velocity_at_u_locations']
	v_now = raw_state['y_velocity_at_v_locations']

	# isentropic density
	s_now = raw_state['air_isentropic_density']
	s_tnd = raw_tendencies.get('air_isentropic_density', None)
	s_new = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_upwind_fluxes, 'xy', dx, dy, dt, u_now, v_now, s_now, s_now, s_tnd, s_new
	)
	hb.dmn_enforce_field(
		s_new, 'air_isentropic_density', 'kg m^-2 K^-1', time=state['time']+timestep
	)
	assert 'air_isentropic_density' in raw_state_new
	compare_arrays(
		s_new[nb:-nb, nb:-nb], raw_state_new['air_isentropic_density'][nb:-nb, nb:-nb]
	)

	if moist:
		# water species
		names = [
			'isentropic_density_of_water_vapor',
			'isentropic_density_of_cloud_liquid_water',
			'isentropic_density_of_precipitation_water',
		]
		sq_new = np.zeros((nx, ny, nz), dtype=dtype)
		for name in names:
			sq_now = raw_state[name]
			sq_tnd = raw_tendencies.get(name, None)
			forward_euler_step(
				get_upwind_fluxes, 'xy', dx, dy, dt, u_now, v_now,
				sq_now, sq_now, sq_tnd, sq_new
			)
			assert name in raw_state_new
			compare_arrays(sq_new[nb:-nb, nb:-nb], raw_state_new[name][nb:-nb, nb:-nb])

	# montgomery potential
	ids = IsentropicDiagnostics(grid, backend, dtype)
	mtg_new = np.zeros((nx, ny, nz), dtype=dtype)
	ids.get_montgomery_potential(s_new, pt.values.item(), mtg_new)
	compare_arrays(mtg_new, imp._mtg_new)

	# x-momentum
	mtg_now = raw_state['montgomery_potential']
	su_now = raw_state['x_momentum_isentropic']
	su_tnd = raw_tendencies.get('x_momentum_isentropic', None)
	su_new = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_upwind_fluxes, 'xy', dx, dy, dt, u_now, v_now, su_now, su_now, su_tnd, su_new
	)
	su_new[1:-1, 1:-1] -= dt * (
		(1-eps) * s_now[1:-1, 1:-1] *
			(mtg_now[2:, 1:-1] - mtg_now[:-2, 1:-1]) / (2.0 * dx) +
		eps * s_new[1:-1, 1:-1] *
			(mtg_new[2:, 1:-1] - mtg_new[:-2, 1:-1]) / (2.0 * dx)
	)
	assert 'x_momentum_isentropic' in raw_state_new
	compare_arrays(
		su_new[nb:-nb, nb:-nb], raw_state_new['x_momentum_isentropic'][nb:-nb, nb:-nb]
	)

	# y-momentum
	sv_now = raw_state['y_momentum_isentropic']
	sv_tnd = raw_tendencies.get('y_momentum_isentropic', None)
	sv_new = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_upwind_fluxes, 'xy', dx, dy, dt, u_now, v_now, sv_now, sv_now, sv_tnd, sv_new
	)
	sv_new[1:-1, 1:-1] -= dt * (
		(1-eps) * s_now[1:-1, 1:-1] *
			(mtg_now[1:-1, 2:] - mtg_now[1:-1, :-2]) / (2.0 * dy) +
		eps * s_new[1:-1, 1:-1] *
			(mtg_new[1:-1, 2:] - mtg_new[1:-1, :-2]) / (2.0 * dy)
	)
	assert 'y_momentum_isentropic' in raw_state_new
	compare_arrays(
		sv_new[nb:-nb, nb:-nb], raw_state_new['y_momentum_isentropic'][nb:-nb, nb:-nb]
	)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_rk3ws_si(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 3  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb))
	domain = data.draw(
		st_domain(xaxis_length=(7, 50), yaxis_length=(7, 50), nb=nb),
		label='domain'
	)
	hb = domain.horizontal_boundary
	assume(hb.type != 'dirichlet')
	grid = domain.numerical_grid
	moist = data.draw(hyp_st.booleans(), label='moist')
	state = data.draw(st_isentropic_state_f(grid, moist=moist), label='state')
	tendencies = {}
	if data.draw(hyp_st.booleans(), label='s_tnd'):
		tendencies['air_isentropic_density'] = state['air_isentropic_density']
	if data.draw(hyp_st.booleans(), label='su_tnd'):
		tendencies['x_momentum_isentropic'] = state['x_momentum_isentropic']
	if data.draw(hyp_st.booleans(), label='sv_tn:'):
		tendencies['y_momentum_isentropic'] = state['y_momentum_isentropic']
	if moist:
		if data.draw(hyp_st.booleans(), label='qv_tnd'):
			tendencies[mfwv] = state[mfwv]
		if data.draw(hyp_st.booleans(), label='qc_tnd'):
			tendencies[mfcw] = state[mfcw]
		if data.draw(hyp_st.booleans(), label='qr_tnd'):
			tendencies[mfpw] = state[mfpw]

	backend = data.draw(st_one_of(conf_backend), label='backend')
	pt = state['air_pressure_on_interface_levels'][0, 0, 0]
	eps = data.draw(st_floats(min_value=0, max_value=1), label='eps')

	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=1),
			max_value=timedelta(minutes=60)
		),
		label='timestep'
	)

	# ========================================
	# test bed
	# ========================================
	hb.reference_state = state
	dtype = grid.x.dtype

	imp = IsentropicPrognostic.factory(
		'rk3ws_si', 'fifth_order_upwind', grid, hb, moist,
		backend=backend, dtype=dtype, pt=pt, eps=eps
	)

	raw_state = make_raw_state(state)
	if moist:
		raw_state['isentropic_density_of_water_vapor'] = \
			raw_state['air_isentropic_density'] * raw_state[mfwv]
		raw_state['isentropic_density_of_cloud_liquid_water'] = \
			raw_state['air_isentropic_density'] * raw_state[mfcw]
		raw_state['isentropic_density_of_precipitation_water'] = \
			raw_state['air_isentropic_density'] * raw_state[mfpw]

	raw_tendencies = make_raw_state(tendencies)
	if moist:
		if mfwv in raw_tendencies:
			raw_tendencies['isentropic_density_of_water_vapor'] = \
				raw_state['air_isentropic_density'] * raw_tendencies[mfwv]
		if mfcw in raw_tendencies:
			raw_tendencies['isentropic_density_of_cloud_liquid_water'] = \
				raw_state['air_isentropic_density'] * raw_tendencies[mfcw]
		if mfpw in raw_tendencies:
			raw_tendencies['isentropic_density_of_precipitation_water'] = \
				raw_state['air_isentropic_density'] * raw_tendencies[mfpw]

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = raw_state['x_velocity_at_u_locations']
	v = raw_state['y_velocity_at_v_locations']

	names = [
		'isentropic_density_of_water_vapor',
		'isentropic_density_of_cloud_liquid_water',
		'isentropic_density_of_precipitation_water'
	]
	sq_new = np.zeros((nx, ny, nz), dtype=dtype)

	#
	# stage 0
	#
	dts = (timestep/3.0).total_seconds()

	raw_state_1 = imp.stage_call(0, timestep, raw_state, raw_tendencies)

	assert 'time' in raw_state_1.keys()
	compare_datetimes(raw_state_1['time'], raw_state['time'] + 1.0/3.0*timestep)

	# isentropic density
	s0 = raw_state['air_isentropic_density']
	s_tnd = raw_tendencies.get('air_isentropic_density', None)
	s1 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, s0, s0, s_tnd, s1
	)
	hb.dmn_enforce_field(
		s1, 'air_isentropic_density', 'kg m^-2 K^-1', time=state['time']+1.0/3.0*timestep
	)
	assert 'air_isentropic_density' in raw_state_1
	compare_arrays(s1, raw_state_1['air_isentropic_density'])

	if moist:
		# water species
		for name in names:
			sq0 = raw_state[name]
			sq_tnd = raw_tendencies.get(name, None)
			forward_euler_step(
				get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v,
				sq0, sq0, sq_tnd, sq_new
			)
			assert name in raw_state_1
			compare_arrays(sq_new[nb:-nb, nb:-nb], raw_state_1[name][nb:-nb, nb:-nb])

	# montgomery potential
	ids = IsentropicDiagnostics(grid, backend, dtype)
	mtg1 = np.zeros((nx, ny, nz), dtype=dtype)
	ids.get_montgomery_potential(s1, pt.to_units('Pa').values.item(), mtg1)
	compare_arrays(mtg1, imp._mtg_new)

	# x-momentum
	mtg0 = raw_state['montgomery_potential']
	compare_arrays(mtg0, imp._mtg_now)
	su0 = raw_state['x_momentum_isentropic']
	su_tnd = raw_tendencies.get('x_momentum_isentropic', None)
	su1 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, su0, su0, su_tnd, su1
	)
	su1[nb:-nb, nb:-nb] -= dts * (
		(1-eps) * s0[nb:-nb, nb:-nb] *
			(mtg0[nb+1:-nb+1, nb:-nb] - mtg0[nb-1:-nb-1, nb:-nb]) / (2.0 * dx) +
		eps * s1[nb:-nb, nb:-nb] *
			(mtg1[nb+1:-nb+1, nb:-nb] - mtg1[nb-1:-nb-1, nb:-nb]) / (2.0 * dx)
	)
	assert 'x_momentum_isentropic' in raw_state_1
	compare_arrays(
		su1[nb:-nb, nb:-nb], raw_state_1['x_momentum_isentropic'][nb:-nb, nb:-nb]
	)

	# y-momentum
	sv0 = raw_state['y_momentum_isentropic']
	sv_tnd = raw_tendencies.get('y_momentum_isentropic', None)
	sv1 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, sv0, sv0, sv_tnd, sv1
	)
	sv1[nb:-nb, nb:-nb] -= dts * (
		(1-eps) * s0[nb:-nb, nb:-nb] *
			(mtg0[nb:-nb, nb+1:-nb+1] - mtg0[nb:-nb, nb-1:-nb-1]) / (2.0 * dy) +
		eps * s1[nb:-nb, nb:-nb] *
			(mtg1[nb:-nb, nb+1:-nb+1] - mtg1[nb:-nb, nb-1:-nb-1]) / (2.0 * dy)
	)
	assert 'y_momentum_isentropic' in raw_state_1
	compare_arrays(
		sv1[nb:-nb, nb:-nb], raw_state_1['y_momentum_isentropic'][nb:-nb, nb:-nb]
	)

	#
	# stage 1
	#
	raw_state_1['x_velocity_at_u_locations'] = raw_state['x_velocity_at_u_locations']
	raw_state_1['y_velocity_at_v_locations'] = raw_state['y_velocity_at_v_locations']
	raw_state_1_dc = deepcopy(raw_state_1)

	if moist:
		if mfwv in raw_tendencies:
			raw_tendencies['isentropic_density_of_water_vapor'] = \
				raw_state_1['air_isentropic_density'] * raw_tendencies[mfwv]
		if mfcw in raw_tendencies:
			raw_tendencies['isentropic_density_of_cloud_liquid_water'] = \
				raw_state_1['air_isentropic_density'] * raw_tendencies[mfcw]
		if mfpw in raw_tendencies:
			raw_tendencies['isentropic_density_of_precipitation_water'] = \
				raw_state_1['air_isentropic_density'] * raw_tendencies[mfpw]

	dts = (0.5*timestep).total_seconds()

	raw_state_2 = imp.stage_call(1, timestep, raw_state_1, raw_tendencies)

	assert 'time' in raw_state_2.keys()
	compare_datetimes(raw_state_2['time'], raw_state['time'] + 0.5*timestep)

	# isentropic density
	s2 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, s0, s1, s_tnd, s2
	)
	hb.dmn_enforce_field(
		s2, 'air_isentropic_density', 'kg m^-2 K^-1', time=state['time']+0.5*timestep
	)
	assert 'air_isentropic_density' in raw_state_2
	compare_arrays(s2, raw_state_2['air_isentropic_density'])

	if moist:
		# water species
		for name in names:
			sq0 = raw_state[name]
			sq1 = raw_state_1_dc[name]
			sq_tnd = raw_tendencies.get(name, None)
			forward_euler_step(
				get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v,
				sq0, sq1, sq_tnd, sq_new
			)
			assert name in raw_state_2
			compare_arrays(sq_new[nb:-nb, nb:-nb], raw_state_2[name][nb:-nb, nb:-nb])

	# montgomery potential
	mtg2 = np.zeros((nx, ny, nz), dtype=dtype)
	ids.get_montgomery_potential(s2, pt.to_units('Pa').values.item(), mtg2)
	compare_arrays(mtg2, imp._mtg_new)

	# x-momentum
	su1 = raw_state_1_dc['x_momentum_isentropic']
	su2 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, su0, su1, su_tnd, su2
	)
	su2[nb:-nb, nb:-nb] -= dts * (
		(1-eps) * s0[nb:-nb, nb:-nb] *
			(mtg0[nb+1:-nb+1, nb:-nb] - mtg0[nb-1:-nb-1, nb:-nb]) / (2.0 * dx) +
		eps * s2[nb:-nb, nb:-nb] *
		(mtg2[nb+1:-nb+1, nb:-nb] - mtg2[nb-1:-nb-1, nb:-nb]) / (2.0 * dx)
	)
	assert 'x_momentum_isentropic' in raw_state_2
	compare_arrays(
		su2[nb:-nb, nb:-nb], raw_state_2['x_momentum_isentropic'][nb:-nb, nb:-nb]
	)

	# y-momentum
	sv1 = raw_state_1_dc['y_momentum_isentropic']
	sv2 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, sv0, sv1, sv_tnd, sv2
	)
	sv2[nb:-nb, nb:-nb] -= dts * (
		(1-eps) * s0[nb:-nb, nb:-nb] *
			(mtg0[nb:-nb, nb+1:-nb+1] - mtg0[nb:-nb, nb-1:-nb-1]) / (2.0 * dy) +
		eps * s2[nb:-nb, nb:-nb] *
			(mtg2[nb:-nb, nb+1:-nb+1] - mtg2[nb:-nb, nb-1:-nb-1]) / (2.0 * dy)
	)
	assert 'y_momentum_isentropic' in raw_state_2
	compare_arrays(
		sv2[nb:-nb, nb:-nb], raw_state_2['y_momentum_isentropic'][nb:-nb, nb:-nb]
	)

	#
	# stage 2
	#
	raw_state_2['x_velocity_at_u_locations'] = raw_state['x_velocity_at_u_locations']
	raw_state_2['y_velocity_at_v_locations'] = raw_state['y_velocity_at_v_locations']
	raw_state_2_dc = deepcopy(raw_state_2)

	if moist:
		if mfwv in raw_tendencies:
			raw_tendencies['isentropic_density_of_water_vapor'] = \
				raw_state_2['air_isentropic_density'] * raw_tendencies[mfwv]
		if mfcw in raw_tendencies:
			raw_tendencies['isentropic_density_of_cloud_liquid_water'] = \
				raw_state_2['air_isentropic_density'] * raw_tendencies[mfcw]
		if mfpw in raw_tendencies:
			raw_tendencies['isentropic_density_of_precipitation_water'] = \
				raw_state_2['air_isentropic_density'] * raw_tendencies[mfpw]

	dts = timestep.total_seconds()

	raw_state_3 = imp.stage_call(2, timestep, raw_state_2, raw_tendencies)

	assert 'time' in raw_state_3.keys()
	compare_datetimes(raw_state_3['time'], raw_state['time'] + timestep)

	# isentropic density
	s3 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, s0, s2, s_tnd, s3
	)
	hb.dmn_enforce_field(
		s3, 'air_isentropic_density', 'kg m^-2 K^-1', time=state['time']+timestep
	)
	assert 'air_isentropic_density' in raw_state_3
	compare_arrays(s3, raw_state_3['air_isentropic_density'])

	if moist:
		# water species
		for name in names:
			sq0 = raw_state[name]
			sq2 = raw_state_2_dc[name]
			sq_tnd = raw_tendencies.get(name, None)
			forward_euler_step(
				get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v,
				sq0, sq2, sq_tnd, sq_new
			)
			assert name in raw_state_3
			compare_arrays(sq_new[nb:-nb, nb:-nb], raw_state_3[name][nb:-nb, nb:-nb])

	# montgomery potential
	mtg3 = np.zeros((nx, ny, nz), dtype=dtype)
	ids.get_montgomery_potential(s3, pt.to_units('Pa').values.item(), mtg3)
	compare_arrays(mtg3, imp._mtg_new)

	# x-momentum
	su2 = raw_state_2_dc['x_momentum_isentropic']
	su3 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, su0, su2, su_tnd, su3
	)
	su3[nb:-nb, nb:-nb] -= dts * (
		(1-eps) * s0[nb:-nb, nb:-nb] *
			(mtg0[nb+1:-nb+1, nb:-nb] - mtg0[nb-1:-nb-1, nb:-nb]) / (2.0 * dx) +
		eps * s3[nb:-nb, nb:-nb] *
			(mtg3[nb+1:-nb+1, nb:-nb] - mtg3[nb-1:-nb-1, nb:-nb]) / (2.0 * dx)
	)
	assert 'x_momentum_isentropic' in raw_state_3
	compare_arrays(
		su3[nb:-nb, nb:-nb], raw_state_3['x_momentum_isentropic'][nb:-nb, nb:-nb]
	)

	# y-momentum
	sv2 = raw_state_2_dc['y_momentum_isentropic']
	sv3 = np.zeros((nx, ny, nz), dtype=dtype)
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dts, u, v, sv0, sv2, sv_tnd, sv3
	)
	sv3[nb:-nb, nb:-nb] -= dts * (
		(1-eps) * s0[nb:-nb, nb:-nb] *
			(mtg0[nb:-nb, nb+1:-nb+1] - mtg0[nb:-nb, nb-1:-nb-1]) / (2.0 * dy) +
		eps * s3[nb:-nb, nb:-nb] *
			(mtg3[nb:-nb, nb+1:-nb+1] - mtg3[nb:-nb, nb-1:-nb-1]) / (2.0 * dy)
	)
	assert 'y_momentum_isentropic' in raw_state_3
	compare_arrays(
		sv3[nb:-nb, nb:-nb], raw_state_3['y_momentum_isentropic'][nb:-nb, nb:-nb]
	)


if __name__ == '__main__':
	pytest.main([__file__])
