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

from tasmania.python.dwarfs.diagnostics import \
	HorizontalVelocity, WaterConstituent
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.isentropic.dynamics.diagnostics import \
	IsentropicDiagnostics as RawIsentropicDiagnostics
from tasmania.python.isentropic.dynamics.ng_dycore import NGIsentropicDynamicalCore
from tasmania.python.isentropic.physics.coriolis import \
	IsentropicConservativeCoriolis
from tasmania.python.isentropic.physics.diagnostics import IsentropicDiagnostics

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .test_isentropic_minimal_horizontal_fluxes import \
		get_fifth_order_upwind_fluxes
	from .test_isentropic_minimal_prognostic import \
		forward_euler_step
	from .utils import compare_arrays, compare_datetimes, \
		st_floats, st_one_of, st_domain, \
		st_isentropic_state_f
except ModuleNotFoundError:
	from conf import backend as conf_backend  # nb as conf_nb
	from test_isentropic_minimal_horizontal_fluxes import \
		get_fifth_order_upwind_fluxes
	from test_isentropic_minimal_prognostic import \
		forward_euler_step
	from utils import compare_arrays, compare_datetimes, \
		st_floats, st_one_of, st_domain, \
		st_isentropic_state_f

import sys
python_version = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
if python_version <= '3.5':
	import collections
	Dict = collections.OrderedDict
else:
	Dict = dict


__tracers = Dict(
	tracer0={'units': 'g g^-1', 'stencil_symbol': 'q0'},
	tracer1={'units': 'g g^-1', 'stencil_symbol': 'q1'},
	tracer2={'units': 'g kg^-1', 'stencil_symbol': 'q2'},
	tracer3={'units': 'kg^-1', 'stencil_symbol': 'q3'},
)


def rk3wssi_stage(
	stage, timestep, raw_state_now, raw_state_int, raw_state_ref, raw_tendencies,
	raw_state_new, field_properties, dx, dy, hb, tracers, hv, wc, damp, vd,
	smooth, hs, diagnostics, eps
):
	u_int = raw_state_int['x_velocity_at_u_locations']
	v_int = raw_state_int['y_velocity_at_v_locations']

	for tracer in tracers:
		wc.get_density_of_water_constituent(
			raw_state_int['air_isentropic_density'],
			raw_state_int[tracer],
			raw_state_int['s_' + tracer]
		)

		if tracer in raw_tendencies:
			raw_tendencies['s_' + tracer] = \
				raw_state_int['air_isentropic_density'] * raw_tendencies[tracer]

	if stage == 0:
		fraction = 1.0/3.0
	elif stage == 1:
		fraction = 0.5
	else:
		fraction = 1.0

	raw_state_new['time'] = raw_state_now['time'] + fraction*timestep

	dt = (fraction*timestep).total_seconds()

	# isentropic density
	s_now = raw_state_now['air_isentropic_density']
	s_int = raw_state_int['air_isentropic_density']
	s_tnd = raw_tendencies.get('air_isentropic_density', None)
	s_new = raw_state_new['air_isentropic_density']
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dt,
		u_int, v_int, s_now, s_int, s_tnd, s_new
	)
	hb.dmn_enforce_field(
		s_new, field_name='air_isentropic_density',
		field_units='kg m^-2 K^-1', time=raw_state_new['time']
	)

	# water species
	for tracer in tracers:
		sq_now = raw_state_now['s_' + tracer]
		sq_int = raw_state_int['s_' + tracer]
		sq_tnd = raw_tendencies.get('s_' + tracer, None)
		sq_new = raw_state_new['s_' + tracer]
		forward_euler_step(
			get_fifth_order_upwind_fluxes, 'xy', dx, dy, dt,
			u_int, v_int, sq_now, sq_int, sq_tnd, sq_new
		)

	# montgomery potential
	pt = raw_state_now['air_pressure_on_interface_levels'][0, 0, 0]
	mtg_new = raw_state_new['montgomery_potential']
	diagnostics.get_montgomery_potential(s_new, pt, mtg_new)

	# x-momentum
	nb = hb.nb
	mtg_now = raw_state_now['montgomery_potential']
	su_now = raw_state_now['x_momentum_isentropic']
	su_int = raw_state_int['x_momentum_isentropic']
	su_tnd = raw_tendencies.get('x_momentum_isentropic', None)
	su_new = raw_state_new['x_momentum_isentropic']
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dt,
		u_int, v_int, su_now, su_int, su_tnd, su_new
	)
	su_new[nb:-nb, nb:-nb] -= dt * (
		(1 - eps) * s_now[nb:-nb, nb:-nb] *
			(mtg_now[nb+1:-nb+1, nb:-nb] - mtg_now[nb-1:-nb-1, nb:-nb]) / (2.0 * dx) +
		eps * s_new[nb:-nb, nb:-nb] *
			(mtg_new[nb+1:-nb+1, nb:-nb] - mtg_new[nb-1:-nb-1, nb:-nb]) / (2.0 * dx)
	)

	# y-momentum
	sv_now = raw_state_now['y_momentum_isentropic']
	sv_int = raw_state_int['y_momentum_isentropic']
	sv_tnd = raw_tendencies.get('y_momentum_isentropic', None)
	sv_new = raw_state_new['y_momentum_isentropic']
	forward_euler_step(
		get_fifth_order_upwind_fluxes, 'xy', dx, dy, dt,
		u_int, v_int, sv_now, sv_int, sv_tnd, sv_new
	)
	sv_new[nb:-nb, nb:-nb] -= dt * (
		(1 - eps) * s_now[nb:-nb, nb:-nb] *
			(mtg_now[nb:-nb, nb+1:-nb+1] - mtg_now[nb:-nb, nb-1:-nb-1]) / (2.0 * dy) +
		eps * s_new[nb:-nb, nb:-nb] *
			(mtg_new[nb:-nb, nb+1:-nb+1] - mtg_new[nb:-nb, nb-1:-nb-1]) / (2.0 * dy)
	)

	for tracer in tracers:
		wc.get_mass_fraction_of_water_constituent_in_air(
			raw_state_new['air_isentropic_density'],
			raw_state_new['s_' + tracer],
			raw_state_new[tracer], clipping=True
		)

	hb.dmn_enforce_raw(raw_state_new, field_properties=field_properties)

	if damp:
		names = [
			'air_isentropic_density',
			'x_momentum_isentropic',
			'y_momentum_isentropic'
		]
		for name in names:
			phi_now = raw_state_now[name]
			phi_new = raw_state_new[name]
			phi_ref = raw_state_ref[name]
			phi_out = raw_state_new[name]
			vd(timestep, phi_now, phi_new, phi_ref, phi_out)

	if smooth:
		for name in field_properties:
			phi = raw_state_new[name]
			phi_out = raw_state_new[name]
			hs(phi, phi_out)
			hb.dmn_enforce_field(
				phi_out, field_name=name,
				field_units=field_properties[name]['units'],
				time=raw_state_new['time']
			)

	hv.get_velocity_components(
		raw_state_new['air_isentropic_density'],
		raw_state_new['x_momentum_isentropic'],
		raw_state_new['y_momentum_isentropic'],
		raw_state_new['x_velocity_at_u_locations'],
		raw_state_new['y_velocity_at_v_locations'],
	)
	hb.dmn_set_outermost_layers_x(
		raw_state_new['x_velocity_at_u_locations'],
		field_name='x_velocity_at_u_locations', field_units='m s^-1',
		time=raw_state_new['time']
	)
	hb.dmn_set_outermost_layers_y(
		raw_state_new['y_velocity_at_v_locations'],
		field_name='y_velocity_at_v_locations', field_units='m s^-1',
		time=raw_state_new['time']
	)


def rk3ws_step(
	domain, tracers, timestep, raw_state_0, raw_tendencies, hv, wc,
	damp, damp_at_every_stage, vd, smooth, smooth_at_every_stage, hs,
	diagnostics, eps
):
	grid, hb = domain.numerical_grid, domain.horizontal_boundary
	nx, ny, nz, nb = grid.nx, grid.ny, grid.nz, hb.nb
	dx, dy = grid.dx.to_units('m').values.item(), grid.dy.to_units('m').values.item()
	dtype = grid.x.dtype

	for tracer in tracers:
		raw_state_0['s_' + tracer] = np.zeros((nx, ny, nz), dtype=dtype)

	raw_state_1 = deepcopy(raw_state_0)
	raw_state_2 = deepcopy(raw_state_0)
	raw_state_3 = deepcopy(raw_state_0)

	field_properties = {
		'air_isentropic_density': {'units': 'kg m^-2 K^-1'},
		'x_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
		'y_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
	}
	field_properties.update({
		tracer: {'units': tracers[tracer]['units']} for tracer in tracers
	})

	state_ref = hb.reference_state
	raw_state_ref = {}
	for name in field_properties:
		raw_state_ref[name] = \
			state_ref[name].to_units(field_properties[name]['units']).values

	# stage 0
	_damp = damp and damp_at_every_stage
	_smooth = smooth and smooth_at_every_stage
	rk3wssi_stage(
		0, timestep, raw_state_0, raw_state_0, raw_state_ref, raw_tendencies,
		raw_state_1, field_properties, dx, dy, hb, tracers, hv, wc, _damp, vd,
		_smooth, hs, diagnostics, eps
	)

	# stage 1
	rk3wssi_stage(
		1, timestep, raw_state_0, raw_state_1, raw_state_ref, raw_tendencies,
		raw_state_2, field_properties, dx, dy, hb, tracers, hv, wc, _damp, vd,
		_smooth, hs, diagnostics, eps
	)

	# stage 2
	rk3wssi_stage(
		2, timestep, raw_state_0, raw_state_2, raw_state_ref, raw_tendencies,
		raw_state_3, field_properties, dx, dy, hb, tracers, hv, wc, damp, vd,
		smooth, hs, diagnostics, eps
	)

	return raw_state_3


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test1(data):
	"""
	- Slow tendencies: no
	- Intermediate tendencies: no
	- Intermediate diagnostics: no
	- Sub-stepping: no
	"""
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(
		st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=3),
		label='domain'
	)
	grid = domain.numerical_grid
	hb = domain.horizontal_boundary
	assume(hb.type != 'dirichlet')

	q_on = {
		tracer: data.draw(hyp_st.booleans(), label=tracer+'_on')
		for tracer in __tracers
	}
	tracers = Dict()
	for tracer in __tracers:
		if q_on[tracer]:
			tracers[tracer] = __tracers[tracer]
	state = data.draw(st_isentropic_state_f(grid, tracers=tracers), label='state')

	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(hours=1)),
		label='timestep'
	)

	eps = data.draw(st_floats(min_value=0, max_value=1), label='eps')

	damp = data.draw(hyp_st.booleans(), label='damp')
	damp_depth = data.draw(
		hyp_st.integers(min_value=0, max_value=grid.nz), label='damp_depth'
	)
	damp_at_every_stage = data.draw(hyp_st.booleans(), label='damp_at_every_stage')

	smooth = data.draw(hyp_st.booleans(), label='smooth')
	smooth_damp_depth = data.draw(
		hyp_st.integers(min_value=0, max_value=grid.nz), label='smooth_damp_depth'
	)
	smooth_at_every_stage = data.draw(hyp_st.booleans(), label='smooth_at_every_stage')

	backend = data.draw(st_one_of(conf_backend), label='backend')
	dtype = grid.x.dtype

	# ========================================
	# test bed
	# ========================================
	domain.horizontal_boundary.reference_state = state

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	hv = HorizontalVelocity(grid, True, backend, dtype)
	wc = WaterConstituent(grid, backend, dtype)
	vd = VerticalDamping.factory(
		'rayleigh', (nx, ny, nz), grid, damp_depth, 0.0002, 's', backend, dtype
	)
	hs = HorizontalSmoothing.factory(
		'second_order', (nx, ny, nz), .03, .24, smooth_damp_depth,
		hb.nb, backend, dtype
	)
	diagnostics = RawIsentropicDiagnostics(grid, backend, dtype)

	dycore = NGIsentropicDynamicalCore(
		domain,
		intermediate_tendencies=None,
		intermediate_diagnostics=None,
		fast_tendencies=None,
		fast_diagnostics=None,
		time_integration_scheme='rk3ws_si',
		horizontal_flux_scheme='fifth_order_upwind',
		time_integration_properties={
			'pt': state['air_pressure_on_interface_levels'][0, 0, 0], 'eps': eps
		},
		damp=damp,
		damp_type='rayleigh',
		damp_depth=damp_depth,
		damp_max=0.0002,
		damp_at_every_stage=damp_at_every_stage,
		smooth=smooth,
		smooth_type='second_order',
		smooth_coeff=.03,
		smooth_coeff_max=.24,
		smooth_damp_depth=smooth_damp_depth,
		smooth_at_every_stage=smooth_at_every_stage,
		tracers=tracers,
		smooth_tracer=smooth,
		smooth_tracer_type='second_order',
		smooth_tracer_coeff=.03,
		smooth_tracer_coeff_max=.24,
		smooth_tracer_damp_depth=smooth_damp_depth,
		smooth_tracer_at_every_stage=smooth_at_every_stage,
		backend=backend,
		dtype=dtype
	)

	#
	# test properties
	#
	assert 'air_isentropic_density' in dycore.input_properties
	assert 'montgomery_potential' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	for tracer in tracers:
		assert tracer in dycore.input_properties
	assert len(dycore.input_properties) == 6 + len(tracers)

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	for tracer in tracers:
		assert tracer in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 3 + len(tracers)

	assert 'air_isentropic_density' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	for tracer in tracers:
		assert tracer in dycore.output_properties
	assert len(dycore.output_properties) == 5 + len(tracers)

	#
	# test numerics
	#
	state_dc = deepcopy(state)

	state_new = dycore(state, {}, timestep)

	for key in state:
		if key == 'time':
			compare_datetimes(state['time'], state_dc['time'])
		else:
			compare_arrays(state[key], state_dc[key])

	assert 'time' in state_new
	compare_datetimes(state_new['time'], state['time'] + timestep)

	assert 'air_isentropic_density' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	for tracer in tracers:
		assert tracer in state_new
	assert len(state_new) == 6 + len(tracers)

	raw_state_now = {'time': state['time']}
	for name, props in dycore.input_properties.items():
		raw_state_now[name] = state[name].to_units(props['units']).values
	raw_state_now['air_pressure_on_interface_levels'] = \
		state['air_pressure_on_interface_levels'].to_units('Pa').values

	raw_state_new_val = rk3ws_step(
		domain, tracers, timestep, raw_state_now, {}, hv, wc,
		damp, damp_at_every_stage, vd, smooth, smooth_at_every_stage, hs,
		diagnostics, eps
	)

	for name in state_new:
		if name != 'time':
			compare_arrays(state_new[name].values, raw_state_new_val[name])


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test2(data):
	"""
	- Slow tendencies: yes
	- Intermediate tendencies: no
	- Intermediate diagnostics: no
	- Sub-stepping: no
	"""
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(
		st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=3),
		label='domain'
	)
	grid = domain.numerical_grid
	hb = domain.horizontal_boundary
	assume(hb.type != 'dirichlet')

	q_on = {
		tracer: data.draw(hyp_st.booleans(), label=tracer+'_on')
		for tracer in __tracers
	}
	tracers = Dict()
	for tracer in __tracers:
		if q_on[tracer]:
			tracers[tracer] = __tracers[tracer]
	state = data.draw(st_isentropic_state_f(grid, tracers=tracers), label='state')

	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(hours=1)),
		label='timestep'
	)

	eps = data.draw(st_floats(min_value=0, max_value=1), label='eps')

	tendencies = {}
	if data.draw(hyp_st.booleans(), label='s_tnd'):
		tendencies['air_isentropic_density'] = deepcopy(state['air_isentropic_density'])
		tendencies['air_isentropic_density'].attrs['units'] = 'kg m^-2 K^-1 s^-1'
	if data.draw(hyp_st.booleans(), label='su_tnd'):
		tendencies['x_momentum_isentropic'] = deepcopy(state['x_momentum_isentropic'])
		tendencies['x_momentum_isentropic'].attrs['units'] = 'kg m^-1 K^-1 s^-2'
	if data.draw(hyp_st.booleans(), label='sv_tnd'):
		tendencies['y_momentum_isentropic'] = deepcopy(state['y_momentum_isentropic'])
		tendencies['y_momentum_isentropic'].attrs['units'] = 'kg m^-1 K^-1 s^-2'
	for tracer in tracers:
		if data.draw(hyp_st.booleans(), label=tracer+'_tnd'):
			tendencies[tracer] = deepcopy(state[tracer])
			tendencies[tracer].attrs['units'] += ' s^-1'

	damp = data.draw(hyp_st.booleans(), label='damp')
	damp_depth = data.draw(
		hyp_st.integers(min_value=0, max_value=grid.nz), label='damp_depth'
	)
	damp_at_every_stage = data.draw(hyp_st.booleans(), label='damp_at_every_stage')

	smooth = data.draw(hyp_st.booleans(), label='smooth')
	smooth_damp_depth = data.draw(
		hyp_st.integers(min_value=0, max_value=grid.nz), label='smooth_damp_depth'
	)
	smooth_at_every_stage = data.draw(hyp_st.booleans(), label='smooth_at_every_stage')

	backend = data.draw(st_one_of(conf_backend), label='backend')
	dtype = grid.x.dtype

	# ========================================
	# test bed
	# ========================================
	domain.horizontal_boundary.reference_state = state

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	hv = HorizontalVelocity(grid, True, backend, dtype)
	wc = WaterConstituent(grid, backend, dtype)
	vd = VerticalDamping.factory(
		'rayleigh', (nx, ny, nz), grid, damp_depth, 0.0002, 's', backend, dtype
	)
	hs = HorizontalSmoothing.factory(
		'second_order', (nx, ny, nz), .03, .24, smooth_damp_depth,
		hb.nb, backend, dtype
	)
	diagnostics = RawIsentropicDiagnostics(grid, backend, dtype)

	dycore = NGIsentropicDynamicalCore(
		domain,
		intermediate_tendencies=None,
		intermediate_diagnostics=None,
		fast_tendencies=None,
		fast_diagnostics=None,
		time_integration_scheme='rk3ws_si',
		horizontal_flux_scheme='fifth_order_upwind',
		time_integration_properties={
			'pt': state['air_pressure_on_interface_levels'][0, 0, 0], 'eps': eps
		},
		damp=damp,
		damp_type='rayleigh',
		damp_depth=damp_depth,
		damp_max=0.0002,
		damp_at_every_stage=damp_at_every_stage,
		smooth=smooth,
		smooth_type='second_order',
		smooth_coeff=.03,
		smooth_coeff_max=.24,
		smooth_damp_depth=smooth_damp_depth,
		smooth_at_every_stage=smooth_at_every_stage,
		tracers=tracers,
		smooth_tracer=smooth,
		smooth_tracer_type='second_order',
		smooth_tracer_coeff=.03,
		smooth_tracer_coeff_max=.24,
		smooth_tracer_damp_depth=smooth_damp_depth,
		smooth_tracer_at_every_stage=smooth_at_every_stage,
		backend=backend,
		dtype=dtype
	)

	#
	# test properties
	#
	assert 'air_isentropic_density' in dycore.input_properties
	assert 'montgomery_potential' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	for tracer in tracers:
		assert tracer in dycore.input_properties
	assert len(dycore.input_properties) == 6 + len(tracers)

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	for tracer in tracers:
		assert tracer in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 3 + len(tracers)

	assert 'air_isentropic_density' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	for tracer in tracers:
		assert tracer in dycore.output_properties
	assert len(dycore.output_properties) == 5 + len(tracers)

	#
	# test numerics
	#
	state_dc = deepcopy(state)

	state_new = dycore(state, tendencies, timestep)

	for key in state:
		if key == 'time':
			compare_datetimes(state['time'], state_dc['time'])
		else:
			compare_arrays(state[key], state_dc[key])

	assert 'time' in state_new
	compare_datetimes(state_new['time'], state['time'] + timestep)

	assert 'air_isentropic_density' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	for tracer in tracers:
		assert tracer in state_new
	assert len(state_new) == 6 + len(tracers)

	raw_state_now = {'time': state['time']}
	for name, props in dycore.input_properties.items():
		raw_state_now[name] = state[name].to_units(props['units']).values
	raw_state_now['air_pressure_on_interface_levels'] = \
		state['air_pressure_on_interface_levels'].to_units('Pa').values

	raw_tendencies = {}
	for name, props in dycore.tendency_properties.items():
		if name in tendencies:
			raw_tendencies[name] = tendencies[name].to_units(props['units']).values

	raw_state_new_val = rk3ws_step(
		domain, tracers, timestep, raw_state_now, raw_tendencies, hv, wc,
		damp, damp_at_every_stage, vd, smooth, smooth_at_every_stage, hs,
		diagnostics, eps
	)

	for name in state_new:
		if name != 'time':
			compare_arrays(state_new[name].values, raw_state_new_val[name])


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test3(data):
	"""
	- Slow tendencies: yes
	- Intermediate tendencies: yes
	- Intermediate diagnostics: yes
	- Sub-stepping: no
	"""
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(
		st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=3),
		label='domain'
	)
	grid = domain.numerical_grid
	hb = domain.horizontal_boundary
	assume(hb.type != 'dirichlet')

	q_on = {
		tracer: data.draw(hyp_st.booleans(), label=tracer+'_on')
		for tracer in __tracers
	}
	tracers = Dict()
	for tracer in __tracers:
		if q_on[tracer]:
			tracers[tracer] = __tracers[tracer]
	state = data.draw(st_isentropic_state_f(grid, tracers=tracers), label='state')

	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(hours=1)),
		label='timestep'
	)

	eps = data.draw(st_floats(min_value=0, max_value=1), label='eps')

	tendencies = {}
	if data.draw(hyp_st.booleans(), label='s_tnd'):
		tendencies['air_isentropic_density'] = deepcopy(state['air_isentropic_density'])
		tendencies['air_isentropic_density'].attrs['units'] = 'kg m^-2 K^-1 s^-1'
	if data.draw(hyp_st.booleans(), label='su_tnd'):
		tendencies['x_momentum_isentropic'] = deepcopy(state['x_momentum_isentropic'])
		tendencies['x_momentum_isentropic'].attrs['units'] = 'kg m^-1 K^-1 s^-2'
	if data.draw(hyp_st.booleans(), label='sv_tnd'):
		tendencies['y_momentum_isentropic'] = deepcopy(state['y_momentum_isentropic'])
		tendencies['y_momentum_isentropic'].attrs['units'] = 'kg m^-1 K^-1 s^-2'
	for tracer in tracers:
		if data.draw(hyp_st.booleans(), label=tracer+'_tnd'):
			tendencies[tracer] = deepcopy(state[tracer])
			tendencies[tracer].attrs['units'] += ' s^-1'

	damp = data.draw(hyp_st.booleans(), label='damp')
	damp_depth = data.draw(
		hyp_st.integers(min_value=0, max_value=grid.nz), label='damp_depth'
	)
	damp_at_every_stage = data.draw(hyp_st.booleans(), label='damp_at_every_stage')

	smooth = data.draw(hyp_st.booleans(), label='smooth')
	smooth_damp_depth = data.draw(
		hyp_st.integers(min_value=0, max_value=grid.nz), label='smooth_damp_depth'
	)
	smooth_at_every_stage = data.draw(hyp_st.booleans(), label='smooth_at_every_stage')

	backend = data.draw(st_one_of(conf_backend), label='backend')
	dtype = grid.x.dtype

	# ========================================
	# test bed
	# ========================================
	domain.horizontal_boundary.reference_state = state

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	hv = HorizontalVelocity(grid, True, backend, dtype)
	wc = WaterConstituent(grid, backend, dtype)
	vd = VerticalDamping.factory(
		'rayleigh', (nx, ny, nz), grid, damp_depth, 0.0002, 's', backend, dtype
	)
	hs = HorizontalSmoothing.factory(
		'second_order', (nx, ny, nz), .03, .24, smooth_damp_depth,
		hb.nb, backend, dtype
	)
	diagnostics = RawIsentropicDiagnostics(grid, backend, dtype)

	cf = IsentropicConservativeCoriolis(
		domain, grid_type='numerical', backend=backend, dtype=dtype
	)
	cfv = cf._f.value

	moist = len(tracers) > 0
	dv = IsentropicDiagnostics(
		domain, 'numerical', moist,
		state['air_pressure_on_interface_levels'][0, 0, 0],
		backend, dtype,
	)

	dycore = NGIsentropicDynamicalCore(
		domain,
		intermediate_tendencies=cf,
		intermediate_diagnostics=dv,
		fast_tendencies=None,
		fast_diagnostics=None,
		time_integration_scheme='rk3ws_si',
		horizontal_flux_scheme='fifth_order_upwind',
		time_integration_properties={
			'pt': state['air_pressure_on_interface_levels'][0, 0, 0], 'eps': eps
		},
		damp=damp,
		damp_type='rayleigh',
		damp_depth=damp_depth,
		damp_max=0.0002,
		damp_at_every_stage=damp_at_every_stage,
		smooth=smooth,
		smooth_type='second_order',
		smooth_coeff=.03,
		smooth_coeff_max=.24,
		smooth_damp_depth=smooth_damp_depth,
		smooth_at_every_stage=smooth_at_every_stage,
		tracers=tracers,
		smooth_tracer=smooth,
		smooth_tracer_type='second_order',
		smooth_tracer_coeff=.03,
		smooth_tracer_coeff_max=.24,
		smooth_tracer_damp_depth=smooth_damp_depth,
		smooth_tracer_at_every_stage=smooth_at_every_stage,
		backend=backend,
		dtype=dtype
	)

	#
	# test properties
	#
	assert 'air_isentropic_density' in dycore.input_properties
	assert 'montgomery_potential' in dycore.input_properties
	assert 'x_momentum_isentropic' in dycore.input_properties
	assert 'x_velocity_at_u_locations' in dycore.input_properties
	assert 'y_momentum_isentropic' in dycore.input_properties
	assert 'y_velocity_at_v_locations' in dycore.input_properties
	for tracer in tracers:
		assert tracer in dycore.input_properties
	assert len(dycore.input_properties) == 6 + len(tracers)

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	for tracer in tracers:
		assert tracer in dycore.tendency_properties
	assert len(dycore.tendency_properties) == 3 + len(tracers)

	assert 'air_isentropic_density' in dycore.output_properties
	assert 'air_pressure_on_interface_levels' in dycore.output_properties
	assert 'exner_function_on_interface_levels' in dycore.output_properties
	assert 'height_on_interface_levels' in dycore.output_properties
	assert 'montgomery_potential' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	if moist:
		assert 'air_density' in dycore.output_properties
		assert 'air_temperature' in dycore.output_properties
		for tracer in tracers:
			assert tracer in dycore.output_properties

		assert len(dycore.output_properties) == 11 + len(tracers)
	else:
		assert len(dycore.output_properties) == 9

	#
	# test numerics
	#
	state_dc = deepcopy(state)

	state_new = dycore(state, tendencies, timestep)

	for key in state:
		if key == 'time':
			compare_datetimes(state['time'], state_dc['time'])
		else:
			compare_arrays(state[key], state_dc[key])

	assert 'time' in state_new
	compare_datetimes(state_new['time'], state['time'] + timestep)

	assert 'air_isentropic_density' in state_new
	assert 'air_pressure_on_interface_levels' in state_new
	assert 'exner_function_on_interface_levels' in state_new
	assert 'height_on_interface_levels' in state_new
	assert 'montgomery_potential' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	if moist:
		assert 'air_density' in state_new
		assert 'air_temperature' in state_new
		for tracer in tracers:
			assert tracer in state_new

		assert len(state_new) == 12 + len(tracers)
	else:
		assert len(state_new) == 10

	raw_state_0 = {'time': state['time']}
	for name, props in dycore.input_properties.items():
		raw_state_0[name] = state[name].to_units(props['units']).values
	for name, props in dycore.output_properties.items():
		if name not in dycore.input_properties:
			raw_state_0[name] = state[name].to_units(props['units']).values
	for tracer in tracers:
		raw_state_0['s_' + tracer] = np.zeros((nx, ny, nz), dtype=dtype)

	raw_tendencies = {}
	for name, props in dycore.tendency_properties.items():
		if name in tendencies:
			raw_tendencies[name] = tendencies[name].to_units(props['units']).values

	if 'x_momentum_isentropic' not in raw_tendencies:
		raw_tendencies['x_momentum_isentropic'] = np.zeros((nx, ny, nz), dtype=dtype)
	if 'y_momentum_isentropic' not in raw_tendencies:
		raw_tendencies['y_momentum_isentropic'] = np.zeros((nx, ny, nz), dtype=dtype)

	raw_tendencies_dc = deepcopy(raw_tendencies)

	dx, dy = grid.dx.to_units('m').values.item(), grid.dy.to_units('m').values.item()

	raw_state_1 = deepcopy(raw_state_0)
	raw_state_2 = deepcopy(raw_state_0)
	raw_state_3 = deepcopy(raw_state_0)

	field_properties = {
		'air_isentropic_density': {'units': 'kg m^-2 K^-1'},
		'x_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
		'y_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
	}
	field_properties.update({
		tracer: {'units': props['units']} for tracer, props in tracers.items()
	})

	state_ref = hb.reference_state
	raw_state_ref = {}
	for name in field_properties:
		raw_state_ref[name] = \
			state_ref[name].to_units(field_properties[name]['units']).values

	#
	# stage 0
	#
	su0 = raw_state_0['x_momentum_isentropic']
	sv0 = raw_state_0['y_momentum_isentropic']
	raw_tendencies['x_momentum_isentropic'][...] = \
		raw_tendencies_dc['x_momentum_isentropic'] + cfv * sv0[...]
	raw_tendencies['y_momentum_isentropic'][...] = \
		raw_tendencies_dc['y_momentum_isentropic'] - cfv * su0[...]

	_damp = damp and damp_at_every_stage
	_smooth = smooth and smooth_at_every_stage
	rk3wssi_stage(
		0, timestep, raw_state_0, raw_state_0, raw_state_ref, raw_tendencies,
		raw_state_1, field_properties, dx, dy, hb, tracers, hv, wc, _damp, vd,
		_smooth, hs, diagnostics, eps
	)

	#
	# stage 1
	#
	su1 = raw_state_1['x_momentum_isentropic']
	sv1 = raw_state_1['y_momentum_isentropic']
	raw_tendencies['x_momentum_isentropic'][...] = \
		raw_tendencies_dc['x_momentum_isentropic'] + cfv * sv1[...]
	raw_tendencies['y_momentum_isentropic'][...] = \
		raw_tendencies_dc['y_momentum_isentropic'] - cfv * su1[...]

	_damp = damp and damp_at_every_stage
	_smooth = smooth and smooth_at_every_stage
	rk3wssi_stage(
		1, timestep, raw_state_0, raw_state_1, raw_state_ref, raw_tendencies,
		raw_state_2, field_properties, dx, dy, hb, tracers, hv, wc, _damp, vd,
		_smooth, hs, diagnostics, eps
	)

	#
	# stage 2
	#
	su2 = raw_state_2['x_momentum_isentropic']
	sv2 = raw_state_2['y_momentum_isentropic']
	raw_tendencies['x_momentum_isentropic'][...] = \
		raw_tendencies_dc['x_momentum_isentropic'] + cfv * sv2[...]
	raw_tendencies['y_momentum_isentropic'][...] = \
		raw_tendencies_dc['y_momentum_isentropic'] - cfv * su2[...]

	rk3wssi_stage(
		2, timestep, raw_state_0, raw_state_2, raw_state_ref, raw_tendencies,
		raw_state_3, field_properties, dx, dy, hb, tracers, hv, wc, damp, vd,
		smooth, hs, diagnostics, eps
	)

	diagnostics.get_diagnostic_variables(
		raw_state_3['air_isentropic_density'],
		raw_state_3['air_pressure_on_interface_levels'][0, 0, 0],
		raw_state_3['air_pressure_on_interface_levels'],
		raw_state_3['exner_function_on_interface_levels'],
		raw_state_3['montgomery_potential'],
		raw_state_3['height_on_interface_levels']
	)
	if moist:
		diagnostics.get_air_density(
			raw_state_3['air_isentropic_density'],
			raw_state_3['height_on_interface_levels'],
			raw_state_3['air_density']
		)
		diagnostics.get_air_temperature(
			raw_state_3['exner_function_on_interface_levels'],
			raw_state_3['air_temperature']
		)

	for name in state_new:
		if name != 'time':
			compare_arrays(state_new[name].values, raw_state_3[name])


if __name__ == '__main__':
	pytest.main([__file__])
