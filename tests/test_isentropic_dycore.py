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
	assume, given, HealthCheck, reproduce_failure, settings, strategies as hyp_st, \
	Verbosity
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.dwarfs.diagnostics import \
	HorizontalVelocity, WaterConstituent
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.grids.domain import Domain
from tasmania.python.isentropic.dynamics.diagnostics import \
	IsentropicDiagnostics as RawIsentropicDiagnostics
from tasmania.python.isentropic.dynamics.dycore import IsentropicDynamicalCore
# from tasmania.python.isentropic.physics.coriolis import \
#	IsentropicConservativeCoriolis
from tasmania.python.isentropic.physics.diagnostics import IsentropicDiagnostics

try:
	from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
	from .test_isentropic_horizontal_fluxes import get_fifth_order_upwind_fluxes
	from .test_isentropic_prognostic import forward_euler_step
	from .utils import compare_arrays, compare_datetimes, \
		st_floats, st_one_of, st_domain, st_isentropic_state_f
except ModuleNotFoundError:
	from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
	from test_isentropic_horizontal_fluxes import get_fifth_order_upwind_fluxes
	from test_isentropic_prognostic import forward_euler_step
	from utils import compare_arrays, compare_datetimes, \
		st_floats, st_one_of, st_domain, st_isentropic_state_f


mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


dummy_domain = Domain(
	DataArray([0, 1], dims='x', attrs={'units': 'm'}), 10,
	DataArray([0, 1], dims='y', attrs={'units': 'm'}), 10,
	DataArray([0, 1], dims='z', attrs={'units': 'K'}), 10,
)
dummy_grid = dummy_domain.numerical_grid
rid = RawIsentropicDiagnostics(dummy_grid)


def get_density_of_water_constituent(s, q, sq, clipping=True):
	sq[...] = s[...] * q[...]
	if clipping:
		sq[sq < 0.0] = 0.0


def get_mass_fraction_of_water_constituent_in_air(s, sq, q, clipping=True):
	q[...] = sq[...] / s[...]
	if clipping:
		q[q < 0.0] = 0.0


def get_montgomery_potential(grid, s, pt, mtg):
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dz = grid.dz.to_units('K').values.item()
	theta_s = grid.z_on_interface_levels.to_units('K').values[-1]
	topo = grid.topography.profile.to_units('m').values
	dtype = s.dtype

	pref = rid._pcs["air_pressure_at_sea_level"]
	rd = rid._pcs["gas_constant_of_dry_air"]
	g = rid._pcs["gravitational_acceleration"]
	cp = rid._pcs["specific_heat_of_dry_air_at_constant_pressure"]

	p = np.zeros((nx, ny, nz+1), dtype=dtype)
	p[:, :, 0] = pt
	for k in range(1, nz+1):
		p[:, :, k] = p[:, :, k-1] + g * dz * s[:, :, k-1]

	exn = cp * (p / pref) ** (rd / cp)

	mtg_s = theta_s * exn[:, :, -1] + g * topo
	mtg[:, :, -1] = mtg_s + 0.5 * dz * exn[:, :, -1]
	for k in range(nz-2, -1, -1):
		mtg[:, :, k] = mtg[:, :, k+1] + dz * exn[:, :, k+1]


def get_velocity_components(s, su, sv, u, v):
	u[1:-1, :] = (su[:-1, :] + su[1:, :]) / (s[:-1, :] + s[1:, :])
	v[:, 1:-1] = (sv[:, :-1] + sv[:, 1:]) / (s[:, :-1] + s[:, 1:])


def apply_rayleigh_damping(vd, dt, phi_now, phi_new, phi_ref, phi_out):
	ni, nj, nk = phi_now.shape
	rmat = vd._rmat.data[:ni, :nj, :nk]
	phi_out[...] = phi_new - dt.total_seconds() * rmat * (phi_now - phi_ref)


def apply_second_order_smoothing(hs, phi, phi_out):
	ni, nj, nk = phi.shape

	g = hs._gamma.data[:ni, :nj, :nk]

	if ni < 5:
		i, j, k = slice(0, ni), slice(2, nj-2), slice(0, nk)
		jm1, jp1 = slice(1, nj-3), slice(3, nj-1)
		jm2, jp2 = slice(0, nj-4), slice(4, nj)

		phi_out[i, j, k] = (1 - 0.375 * g[i, j, k]) * phi[i, j, k] + \
			0.0625 * g[i, j, k] * (
				- phi[i, jm2, k] + 4.0 * phi[i, jm1, k]
				- phi[i, jp2, k] + 4.0 * phi[i, jp1, k]
			)
	elif nj < 5:
		i, j, k = slice(2, ni-2), slice(0, nj), slice(0, nk)
		im1, ip1 = slice(1, ni-3), slice(3, ni-1)
		im2, ip2 = slice(0, ni-4), slice(4, ni)

		phi_out[i, j, k] = (1 - 0.375 * g[i, j, k]) * phi[i, j, k] + \
			0.0625 * g[i, j, k] * (
				- phi[im2, j, k] + 4.0 * phi[im1, j, k]
				- phi[ip2, j, k] + 4.0 * phi[ip1, j, k]
			)
	else:
		i, j, k = slice(2, ni-2), slice(2, nj-2), slice(0, nk)
		im1, ip1 = slice(1, ni-3), slice(3, ni-1)
		im2, ip2 = slice(0, ni-4), slice(4, ni)
		jm1, jp1 = slice(1, nj-3), slice(3, nj-1)
		jm2, jp2 = slice(0, nj-4), slice(4, nj)

		phi_out[i, j, k] = (1 - 0.75 * g[i, j, k]) * phi[i, j, k] + \
			0.0625 * g[i, j, k] * (
				- phi[im2, j, k] + 4.0 * phi[im1, j, k]
				- phi[ip2, j, k] + 4.0 * phi[ip1, j, k]
				- phi[i, jm2, k] + 4.0 * phi[i, jm1, k]
				- phi[i, jp2, k] + 4.0 * phi[i, jp1, k]
			)


def rk3wssi_stage(
	stage, timestep, grid, raw_state_now, raw_state_int, raw_state_ref, raw_tendencies,
	raw_state_new, field_properties, hb, moist, damp, vd, smooth, hs, eps
):
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	u_int = raw_state_int['x_velocity_at_u_locations']
	v_int = raw_state_int['y_velocity_at_v_locations']

	if moist:
		get_density_of_water_constituent(
			raw_state_int['air_isentropic_density'],
			raw_state_int[mfwv],
			raw_state_int['isentropic_density_of_water_vapor'],
			clipping=True
		)
		get_density_of_water_constituent(
			raw_state_int['air_isentropic_density'],
			raw_state_int[mfcw],
			raw_state_int['isentropic_density_of_cloud_liquid_water'],
			clipping=True
		)
		get_density_of_water_constituent(
			raw_state_int['air_isentropic_density'],
			raw_state_int[mfpw],
			raw_state_int['isentropic_density_of_precipitation_water'],
			clipping=True
		)

		if mfwv in raw_tendencies:
			raw_tendencies['isentropic_density_of_water_vapor'] = \
				raw_state_int['air_isentropic_density'] * raw_tendencies[mfwv]
		if mfcw in raw_tendencies:
			raw_tendencies['isentropic_density_of_cloud_liquid_water'] = \
				raw_state_int['air_isentropic_density'] * raw_tendencies[mfcw]
		if mfpw in raw_tendencies:
			raw_tendencies['isentropic_density_of_precipitation_water'] = \
				raw_state_int['air_isentropic_density'] * raw_tendencies[mfpw]

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

	if moist:
		# water species
		names = [
			'isentropic_density_of_water_vapor',
			'isentropic_density_of_cloud_liquid_water',
			'isentropic_density_of_precipitation_water'
		]
		for name in names:
			sq_now = raw_state_now[name]
			sq_int = raw_state_int[name]
			sq_tnd = raw_tendencies.get(name, None)
			sq_new = raw_state_new[name]
			forward_euler_step(
				get_fifth_order_upwind_fluxes, 'xy', dx, dy, dt,
				u_int, v_int, sq_now, sq_int, sq_tnd, sq_new
			)

	# montgomery potential
	pt = raw_state_now['air_pressure_on_interface_levels'][0, 0, 0]
	mtg_new = raw_state_new['montgomery_potential']
	get_montgomery_potential(grid, s_new, pt, mtg_new)

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

	if moist:
		get_mass_fraction_of_water_constituent_in_air(
			raw_state_new['air_isentropic_density'],
			raw_state_new['isentropic_density_of_water_vapor'],
			raw_state_new[mfwv], clipping=True
		)
		get_mass_fraction_of_water_constituent_in_air(
			raw_state_new['air_isentropic_density'],
			raw_state_new['isentropic_density_of_cloud_liquid_water'],
			raw_state_new[mfcw], clipping=True
		)
		get_mass_fraction_of_water_constituent_in_air(
			raw_state_new['air_isentropic_density'],
			raw_state_new['isentropic_density_of_precipitation_water'],
			raw_state_new[mfpw], clipping=True
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
			apply_rayleigh_damping(vd, timestep, phi_now, phi_new, phi_ref, phi_out)

	if smooth:
		for name in field_properties:
			phi = raw_state_new[name]
			phi_out = raw_state_new[name]
			apply_second_order_smoothing(hs, phi, phi_out)
			hb.dmn_enforce_field(
				phi_out, field_name=name,
				field_units=field_properties[name]['units'],
				time=raw_state_new['time']
			)

	get_velocity_components(
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
	domain, moist, timestep, raw_state_0, raw_tendencies,
	damp, damp_at_every_stage, vd, smooth, smooth_at_every_stage, hs, eps
):
	grid, hb = domain.numerical_grid, domain.horizontal_boundary
	nx, ny, nz, nb = grid.nx, grid.ny, grid.nz, hb.nb
	dtype = grid.x.dtype

	if moist:
		raw_state_0['isentropic_density_of_water_vapor'] = \
			np.zeros((nx, ny, nz), dtype=dtype)
		raw_state_0['isentropic_density_of_cloud_liquid_water'] = \
			np.zeros((nx, ny, nz), dtype=dtype)
		raw_state_0['isentropic_density_of_precipitation_water'] = \
			np.zeros((nx, ny, nz), dtype=dtype)

	raw_state_1 = deepcopy(raw_state_0)
	raw_state_2 = deepcopy(raw_state_0)
	raw_state_3 = deepcopy(raw_state_0)

	field_properties = {
		'air_isentropic_density': {'units': 'kg m^-2 K^-1'},
		'x_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
		'y_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
	}
	if moist:
		field_properties.update({
			mfwv: {'units': 'g g^-1'},
			mfcw: {'units': 'g g^-1'},
			mfpw: {'units': 'g g^-1'},
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
		0, timestep, grid, raw_state_0, raw_state_0, raw_state_ref, raw_tendencies,
		raw_state_1, field_properties, hb, moist, _damp, vd, _smooth, hs, eps
	)

	# stage 1
	rk3wssi_stage(
		1, timestep, grid, raw_state_0, raw_state_1, raw_state_ref, raw_tendencies,
		raw_state_2, field_properties, hb, moist, _damp, vd, _smooth, hs, eps
	)

	# stage 2
	rk3wssi_stage(
		2, timestep, grid, raw_state_0, raw_state_2, raw_state_ref, raw_tendencies,
		raw_state_3, field_properties, hb, moist, damp, vd,	smooth, hs, eps
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
	nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label='nb')
	domain = data.draw(
		st_domain(
			xaxis_length=(1, 20),
			yaxis_length=(1, 20),
			zaxis_length=(2, 10),
			nb=nb
		),
		label='domain'
	)
	grid = domain.numerical_grid
	hb = domain.horizontal_boundary
	assume(hb.type != 'dirichlet')

	moist = True  # data.draw(hyp_st.booleans(), label='moist')
	state = data.draw(st_isentropic_state_f(grid, moist=moist), label='state')
	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(hours=1)),
		label='timestep'
	)

	eps = data.draw(st_floats(min_value=0, max_value=1), label='eps')

	damp = False
	damp_depth = data.draw(
		hyp_st.integers(min_value=0, max_value=grid.nz), label='damp_depth'
	)
	damp_at_every_stage = False  # data.draw(hyp_st.booleans(), label='damp_at_every_stage')

	smooth = False  # data.draw(hyp_st.booleans(), label='smooth')
	smooth_damp_depth = data.draw(
		hyp_st.integers(min_value=0, max_value=grid.nz), label='smooth_damp_depth'
	)
	smooth_at_every_stage = False  # data.draw(hyp_st.booleans(), label='smooth_at_every_stage')

	backend = data.draw(st_one_of(conf_backend), label='backend')
	dtype = grid.x.dtype
	halo = data.draw(st_one_of(conf_halo), label='halo')

	# ========================================
	# test bed
	# ========================================
	domain.horizontal_boundary.reference_state = state

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	vd = VerticalDamping.factory(
		'rayleigh', grid, (nx+1, ny+1, nz+1), damp_depth, 0.0002, 's',
		backend=backend, dtype=dtype, halo=halo
	)
	hs = HorizontalSmoothing.factory(
		'second_order', (nx+1, ny+1, nz+1), .03, .24, smooth_damp_depth,
		hb.nb, backend=backend, dtype=dtype, halo=halo
	)

	dycore = IsentropicDynamicalCore(
		domain,
		intermediate_tendencies=None,
		intermediate_diagnostics=None,
		fast_tendencies=None,
		fast_diagnostics=None,
		moist=moist,
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
		smooth_moist=smooth,
		smooth_moist_type='second_order',
		smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24,
		smooth_moist_damp_depth=smooth_damp_depth,
		smooth_moist_at_every_stage=smooth_at_every_stage,
		backend=backend,
		dtype=dtype,
		halo=halo,
		rebuild=True
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
	if moist:
		assert mfwv in dycore.input_properties
		assert mfcw in dycore.input_properties
		assert mfpw in dycore.input_properties
		assert len(dycore.input_properties) == 9
	else:
		assert len(dycore.input_properties) == 6

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	if moist:
		assert mfwv in dycore.tendency_properties
		assert mfwv in dycore.tendency_properties
		assert mfpw in dycore.tendency_properties
		assert len(dycore.tendency_properties) == 6
	else:
		assert len(dycore.tendency_properties) == 3

	assert 'air_isentropic_density' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	if moist:
		assert mfwv in dycore.output_properties
		assert mfcw in dycore.output_properties
		assert mfpw in dycore.output_properties
		assert len(dycore.output_properties) == 8
	else:
		assert len(dycore.output_properties) == 5

	#
	# test numerics
	#
	state_dc = deepcopy(state)

	state_new = dycore(state, {}, timestep)

	for key in state:
		if key == 'time':
			compare_datetimes(state['time'], state_dc['time'])
		else:
			compare_arrays(state[key].values, state_dc[key].values)

	assert 'time' in state_new
	compare_datetimes(state_new['time'], state['time'] + timestep)

	assert 'air_isentropic_density' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	if moist:
		assert mfwv in state_new
		assert mfcw in state_new
		assert mfpw in state_new
		assert len(state_new) == 9
	else:
		assert len(state_new) == 6

	raw_state_now = {'time': state['time']}
	for name, props in dycore.input_properties.items():
		raw_state_now[name] = state[name].to_units(props['units']).values
	raw_state_now['air_pressure_on_interface_levels'] = \
		state['air_pressure_on_interface_levels'].to_units('Pa').values

	raw_state_new_val = rk3ws_step(
		domain, moist, timestep, raw_state_now, {},
		damp, damp_at_every_stage, vd, smooth, smooth_at_every_stage, hs, eps
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
def _test2(data):
	"""
	- Slow tendencies: yes
	- Intermediate tendencies: no
	- Intermediate diagnostics: no
	- Sub-stepping: no
	"""
	# ========================================
	# random data generation
	# ========================================
	nb = data.draw(hyp_st.integers(min_value=1, max_value=max(3, conf_nb)), label='nb')
	domain = data.draw(
		st_domain(
			xaxis_length=(1, 30),
			yaxis_length=(1, 30),
			zaxis_length=(2, 20),
			nb=nb
		),
		label='domain'
	)
	grid = domain.numerical_grid
	hb = domain.horizontal_boundary
	assume(hb.type != 'dirichlet')

	moist = data.draw(hyp_st.booleans(), label='moist')
	state = data.draw(st_isentropic_state_f(grid, moist=moist), label='state')
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
	if moist:
		if data.draw(hyp_st.booleans(), label='qv_tnd'):
			tendencies[mfwv] = deepcopy(state[mfwv])
			tendencies[mfwv].attrs['units'] = 'g g^-1 s^-1'
		if data.draw(hyp_st.booleans(), label='qc_tnd'):
			tendencies[mfcw] = deepcopy(state[mfcw])
			tendencies[mfcw].attrs['units'] = 'g g^-1 s^-1'
		if data.draw(hyp_st.booleans(), label='qr_tnd'):
			tendencies[mfpw] = deepcopy(state[mfpw])
			tendencies[mfpw].attrs['units'] = 'g g^-1 s^-1'

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
	halo = data.draw(st_one_of(conf_halo), label='halo')

	# ========================================
	# test bed
	# ========================================
	domain.horizontal_boundary.reference_state = state

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	vd = VerticalDamping.factory(
		'rayleigh', grid, (nx, ny, nz), damp_depth, 0.0002, 's',
		backend=backend, dtype=dtype, halo=halo
	)
	hs = HorizontalSmoothing.factory(
		'second_order', (nx, ny, nz), .03, .24, smooth_damp_depth,
		hb.nb, backend=backend, dtype=dtype, halo=halo
	)

	dycore = IsentropicDynamicalCore(
		domain,
		intermediate_tendencies=None,
		intermediate_diagnostics=None,
		fast_tendencies=None,
		fast_diagnostics=None,
		moist=moist,
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
		smooth_moist=smooth,
		smooth_moist_type='second_order',
		smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24,
		smooth_moist_damp_depth=smooth_damp_depth,
		smooth_moist_at_every_stage=smooth_at_every_stage,
		backend=backend,
		dtype=dtype,
		halo=halo
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
	if moist:
		assert mfwv in dycore.input_properties
		assert mfcw in dycore.input_properties
		assert mfpw in dycore.input_properties
		assert len(dycore.input_properties) == 9
	else:
		assert len(dycore.input_properties) == 6

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	if moist:
		assert mfwv in dycore.tendency_properties
		assert mfwv in dycore.tendency_properties
		assert mfpw in dycore.tendency_properties
		assert len(dycore.tendency_properties) == 6
	else:
		assert len(dycore.tendency_properties) == 3

	assert 'air_isentropic_density' in dycore.output_properties
	assert 'x_momentum_isentropic' in dycore.output_properties
	assert 'x_velocity_at_u_locations' in dycore.output_properties
	assert 'y_momentum_isentropic' in dycore.output_properties
	assert 'y_velocity_at_v_locations' in dycore.output_properties
	if moist:
		assert mfwv in dycore.output_properties
		assert mfcw in dycore.output_properties
		assert mfpw in dycore.output_properties
		assert len(dycore.output_properties) == 8
	else:
		assert len(dycore.output_properties) == 5

	#
	# test numerics
	#
	state_dc = deepcopy(state)

	state_new = dycore(state, tendencies, timestep)

	for key in state:
		if key == 'time':
			compare_datetimes(state['time'], state_dc['time'])
		else:
			compare_arrays(state[key].values, state_dc[key].values)

	assert 'time' in state_new
	compare_datetimes(state_new['time'], state['time'] + timestep)

	assert 'air_isentropic_density' in state_new
	assert 'x_momentum_isentropic' in state_new
	assert 'x_velocity_at_u_locations' in state_new
	assert 'y_momentum_isentropic' in state_new
	assert 'y_velocity_at_v_locations' in state_new
	if moist:
		assert mfwv in state_new
		assert mfcw in state_new
		assert mfpw in state_new
		assert len(state_new) == 9
	else:
		assert len(state_new) == 6

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
		domain, moist, timestep, raw_state_now, raw_tendencies,
		damp, damp_at_every_stage, vd, smooth, smooth_at_every_stage, hs, eps
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
def _test3(data):
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

	moist = data.draw(hyp_st.booleans(), label='moist')
	state = data.draw(st_isentropic_state_f(grid, moist=moist), label='state')
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
	if moist:
		if data.draw(hyp_st.booleans(), label='qv_tnd'):
			tendencies[mfwv] = deepcopy(state[mfwv])
			tendencies[mfwv].attrs['units'] = 'g g^-1 s^-1'
		if data.draw(hyp_st.booleans(), label='qc_tnd'):
			tendencies[mfcw] = deepcopy(state[mfcw])
			tendencies[mfcw].attrs['units'] = 'g g^-1 s^-1'
		if data.draw(hyp_st.booleans(), label='qr_tnd'):
			tendencies[mfpw] = deepcopy(state[mfpw])
			tendencies[mfpw].attrs['units'] = 'g g^-1 s^-1'

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

	dv = IsentropicDiagnostics(
		domain, 'numerical', moist,
		state['air_pressure_on_interface_levels'][0, 0, 0],
		backend, dtype,
	)

	dycore = IsentropicDynamicalCore(
		domain,
		intermediate_tendencies=cf,
		intermediate_diagnostics=dv,
		fast_tendencies=None,
		fast_diagnostics=None,
		moist=moist,
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
		smooth_moist=smooth,
		smooth_moist_type='second_order',
		smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24,
		smooth_moist_damp_depth=smooth_damp_depth,
		smooth_moist_at_every_stage=smooth_at_every_stage,
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
	if moist:
		assert mfwv in dycore.input_properties
		assert mfcw in dycore.input_properties
		assert mfpw in dycore.input_properties
		assert len(dycore.input_properties) == 9
	else:
		assert len(dycore.input_properties) == 6

	assert 'air_isentropic_density' in dycore.tendency_properties
	assert 'x_momentum_isentropic' in dycore.tendency_properties
	assert 'y_momentum_isentropic' in dycore.tendency_properties
	if moist:
		assert mfwv in dycore.tendency_properties
		assert mfwv in dycore.tendency_properties
		assert mfpw in dycore.tendency_properties
		assert len(dycore.tendency_properties) == 6
	else:
		assert len(dycore.tendency_properties) == 3

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
		assert mfwv in dycore.output_properties
		assert mfcw in dycore.output_properties
		assert mfpw in dycore.output_properties
		assert len(dycore.output_properties) == 14
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
		assert mfwv in state_new
		assert mfcw in state_new
		assert mfpw in state_new
		assert len(state_new) == 15
	else:
		assert len(state_new) == 10

	raw_state_0 = {'time': state['time']}
	for name, props in dycore.input_properties.items():
		raw_state_0[name] = state[name].to_units(props['units']).values
	for name, props in dycore.output_properties.items():
		if name not in dycore.input_properties:
			raw_state_0[name] = state[name].to_units(props['units']).values
	if moist:
		raw_state_0['isentropic_density_of_water_vapor'] = \
			np.zeros((nx, ny, nz), dtype=dtype)
		raw_state_0['isentropic_density_of_cloud_liquid_water'] = \
			np.zeros((nx, ny, nz), dtype=dtype)
		raw_state_0['isentropic_density_of_precipitation_water'] = \
			np.zeros((nx, ny, nz), dtype=dtype)

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

	names = [
		'air_isentropic_density',
		'x_momentum_isentropic',
		'y_momentum_isentropic',
	]
	if moist:
		names.append('isentropic_density_of_water_vapor')
		names.append('isentropic_density_of_cloud_liquid_water')
		names.append('isentropic_density_of_precipitation_water')

	field_properties = {
		'air_isentropic_density': {'units': 'kg m^-2 K^-1'},
		'x_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
		'y_momentum_isentropic': {'units': 'kg m^-1 K^-1 s^-1'},
	}
	if moist:
		field_properties.update({
			mfwv: {'units': 'g g^-1'},
			mfcw: {'units': 'g g^-1'},
			mfpw: {'units': 'g g^-1'},
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
		raw_state_1, field_properties, dx, dy, hb, moist, hv, wc, _damp, vd,
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
		raw_state_2, field_properties, dx, dy, hb, moist, hv, wc, _damp, vd,
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
		raw_state_3, field_properties, dx, dy, hb, moist, hv, wc, damp, vd,
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
	# pytest.main([__file__])
	test1()
