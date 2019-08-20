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
from pint import UnitRegistry
import pytest

from tasmania.python.isentropic.dynamics.diagnostics import \
	IsentropicDiagnostics as DynamicsIsentropicDiagnostics
from tasmania.python.isentropic.physics.diagnostics import \
	IsentropicDiagnostics, IsentropicVelocityComponents
from tasmania.python.utils.data_utils import make_dataarray_3d

try:
	from .conf import backend as conf_backend
	from .utils import compare_datetimes, compare_arrays, compare_dataarrays, \
		st_floats, st_one_of, st_domain, st_physical_grid, st_isentropic_state
except ModuleNotFoundError:
	from conf import backend as conf_backend
	from utils import compare_datetimes, compare_arrays, compare_dataarrays, \
		st_floats, st_one_of, st_domain, st_physical_grid, st_isentropic_state


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_diagnostic_variables(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(st_physical_grid(zaxis_name='z'), label='grid')
	dtype = grid.x.dtype

	s = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label='r'
	)
	pt = data.draw(st_floats(min_value=1, max_value=1e5), label='pt')

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	did = DynamicsIsentropicDiagnostics(grid, backend=backend, dtype=dtype)

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	p   = np.zeros((nx, ny, nz+1), dtype=dtype)
	exn = np.zeros((nx, ny, nz+1), dtype=dtype)
	mtg = np.zeros((nx, ny, nz), dtype=dtype)
	h   = np.zeros((nx, ny, nz+1), dtype=dtype)

	did.get_diagnostic_variables(s, pt, p, exn, mtg, h)

	cp    = did._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
	p_ref = did._physical_constants['air_pressure_at_sea_level']
	rd    = did._physical_constants['gas_constant_of_dry_air']
	g	  = did._physical_constants['gravitational_acceleration']

	dz   = grid.dz.to_units('K').values.item()
	topo = grid.topography.profile.to_units('m').values

	# pressure
	p_val = np.zeros((nx, ny, nz+1), dtype=dtype)
	p_val[:, :, 0] = pt
	for k in range(1, nz+1):
		p_val[:, :, k] = p_val[:, :, k-1] + g * dz * s[:, :, k-1]
	assert np.allclose(p, p_val)

	# exner
	exn_val = cp * (p_val / p_ref) ** (rd / cp)
	assert np.allclose(exn, exn_val)

	# montgomery
	mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
	theta_s = grid.z_on_interface_levels.to_units('K').values[-1]
	mtg_s = theta_s * exn_val[:, :, -1] + g * topo
	mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
	for k in range(nz-2, -1, -1):
		mtg_val[:, :, k] = mtg_val[:, :, k+1] + dz * exn_val[:, :, k+1]
	assert np.allclose(mtg, mtg_val)

	# height
	theta = grid.z_on_interface_levels.to_units('K').values
	h_val = np.zeros((nx, ny, nz+1), dtype=dtype)
	h_val[:, :, -1] = topo
	for k in range(nz-1, -1, -1):
		h_val[:, :, k] = h_val[:, :, k+1] - (rd / (cp * g)) * \
			(theta[k] * exn_val[:, :, k] + theta[k+1] * exn_val[:, :, k+1]) * \
			(p_val[:, :, k] - p_val[:, :, k+1]) / (p_val[:, :, k] + p_val[:, :, k+1])
	assert np.allclose(h, h_val)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_height(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(st_physical_grid(zaxis_name='z'), label='grid')
	dtype = grid.x.dtype

	s = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label='r'
	)
	pt = data.draw(st_floats(min_value=1, max_value=1e5), label='pt')

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	did = DynamicsIsentropicDiagnostics(grid, backend=backend, dtype=dtype)

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	h = np.zeros((nx, ny, nz+1), dtype=dtype)

	did.get_height(s, pt, h)

	cp    = did._physical_constants['specific_heat_of_dry_air_at_constant_pressure']
	p_ref = did._physical_constants['air_pressure_at_sea_level']
	rd    = did._physical_constants['gas_constant_of_dry_air']
	g	  = did._physical_constants['gravitational_acceleration']

	dz   = grid.dz.to_units('K').values.item()
	topo = grid.topography.profile.to_units('m').values

	# pressure
	p_val = np.zeros((nx, ny, nz+1), dtype=dtype)
	p_val[:, :, 0] = pt
	for k in range(1, nz+1):
		p_val[:, :, k] = p_val[:, :, k-1] + g * dz * s[:, :, k-1]

	# exner
	exn_val = cp * (p_val / p_ref) ** (rd / cp)

	# height
	theta = grid.z_on_interface_levels.to_units('K').values
	h_val = np.zeros((nx, ny, nz+1), dtype=dtype)
	h_val[:, :, -1] = topo
	for k in range(nz-1, -1, -1):
		h_val[:, :, k] = h_val[:, :, k+1] - (rd / (cp * g)) * \
			(theta[k] * exn_val[:, :, k] + theta[k+1] * exn_val[:, :, k+1]) * \
			(p_val[:, :, k] - p_val[:, :, k+1]) / (p_val[:, :, k] + p_val[:, :, k+1])
	assert np.allclose(h, h_val)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_air_density(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(st_physical_grid(zaxis_name='z'), label='grid')
	dtype = grid.x.dtype

	s = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label='r'
	)
	h = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz+1),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label='r'
	)

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	did = DynamicsIsentropicDiagnostics(grid, backend=backend, dtype=dtype)

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	r = np.zeros((nx, ny, nz), dtype=dtype)

	did.get_air_density(s, h, r)

	theta = grid.z_on_interface_levels.to_units('K').values[np.newaxis, np.newaxis, :]

	r_val = s * (theta[:, :, :-1] - theta[:, :, 1:]) / (h[:, :, :-1] - h[:, :, 1:])

	assert np.allclose(r, r_val, equal_nan=True)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_air_temperature(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(st_physical_grid(zaxis_name='z'), label='grid')
	dtype = grid.x.dtype

	exn = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz+1),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label='r'
	)

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	did = DynamicsIsentropicDiagnostics(grid, backend=backend, dtype=dtype)

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	temp = np.zeros((nx, ny, nz), dtype=dtype)

	did.get_air_temperature(exn, temp)

	theta = grid.z_on_interface_levels.to_units('K').values[np.newaxis, np.newaxis, :]
	cp = did._physical_constants['specific_heat_of_dry_air_at_constant_pressure']

	temp_val = 0.5 * (theta[:, :, :-1] * exn[:, :, :-1] + theta[:, :, 1:] * exn[:, :, 1:]) / cp

	assert np.allclose(temp, temp_val, equal_nan=True)


unit_registry = UnitRegistry()


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_isentropic_diagnostics(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(st_domain(), label='domain')

	grid = domain.numerical_grid
	dtype = grid.x.dtype

	state = data.draw(st_isentropic_state(grid, moist=True), label='state')

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	#
	# validation data
	#
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	p   = np.zeros((nx, ny, nz+1), dtype=dtype)
	exn = np.zeros((nx, ny, nz+1), dtype=dtype)
	mtg = np.zeros((nx, ny, nz  ), dtype=dtype)
	h   = np.zeros((nx, ny, nz+1), dtype=dtype)
	r   = np.zeros((nx, ny, nz  ), dtype=dtype)
	t   = np.zeros((nx, ny, nz  ), dtype=dtype)

	did = DynamicsIsentropicDiagnostics(grid, backend, dtype)

	s = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values
	pt = state['air_pressure_on_interface_levels'].to_units('Pa').values[0, 0, 0]

	did.get_diagnostic_variables(s, pt, p, exn, mtg, h)
	did.get_air_density(s, h, r)
	did.get_air_temperature(exn, t)

	#
	# dry
	#
	pid = IsentropicDiagnostics(
		domain, 'numerical', moist=False,
		pt=state['air_pressure_on_interface_levels'][0, 0, 0],
		backend=backend, dtype=dtype
	)

	diags = pid(state)

	names = (
		'air_pressure_on_interface_levels',
		'exner_function_on_interface_levels',
		'montgomery_potential',
		'height_on_interface_levels'
	)
	units = ('Pa', 'J kg^-1 K^-1', 'm^2 s^-2', 'm')
	raw_val = (p, exn, mtg, h)

	for i in range(len(names)):
		assert names[i] in diags
		val = make_dataarray_3d(raw_val[i], grid, units[i], name=names[i])
		compare_dataarrays(diags[names[i]], val)

	assert len(diags) == len(names)

	#
	# moist
	#
	pid = IsentropicDiagnostics(
		domain, 'numerical', moist=True,
		pt=state['air_pressure_on_interface_levels'][0, 0, 0],
		backend=backend, dtype=dtype
	)

	diags = pid(state)

	names = (
		'air_pressure_on_interface_levels',
		'exner_function_on_interface_levels',
		'montgomery_potential',
		'height_on_interface_levels',
		'air_density',
		'air_temperature'
	)
	units = ('Pa', 'J kg^-1 K^-1', 'm^2 s^-2', 'm', 'kg m^-3', 'K')
	raw_val = (p, exn, mtg, h, r, t)

	for i in range(len(names)):
		assert names[i] in diags
		val = make_dataarray_3d(raw_val[i], grid, units[i], name=names[i])
		compare_dataarrays(diags[names[i]], val)

	assert len(diags) == len(names)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_horizontal_velocity(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(st_domain(), label='domain')

	hb = domain.horizontal_boundary
	assume(hb.type != 'dirichlet')

	grid = domain.numerical_grid
	dtype = grid.x.dtype

	state = data.draw(st_isentropic_state(grid, moist=True), label='state')

	hb.reference_state = state

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	ivc = IsentropicVelocityComponents(domain, backend, dtype)

	diags = ivc(state)

	s  = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values
	su = state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values
	sv = state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	u = np.zeros((nx+1, ny, nz), dtype=dtype)
	v = np.zeros((nx, ny+1, nz), dtype=dtype)

	assert 'x_velocity_at_u_locations' in diags
	u[1:-1, :] = (su[:-1, :] + su[1:, :]) / (s[:-1, :] + s[1:, :])
	hb.dmn_set_outermost_layers_x(
		u, field_name='x_velocity_at_u_locations', field_units='m s^-1', time=state['time']
	)
	u_val = make_dataarray_3d(u, grid, 'm s^-1', name='x_velocity_at_u_locations')
	compare_dataarrays(diags['x_velocity_at_u_locations'], u_val)

	assert 'y_velocity_at_v_locations' in diags
	v[:, 1:-1] = (sv[:, :-1] + sv[:, 1:]) / (s[:, :-1] + s[:, 1:])
	hb.dmn_set_outermost_layers_y(
		v, field_name='y_velocity_at_v_locations', field_units='m s^-1', time=state['time']
	)
	v_val = make_dataarray_3d(v, grid, 'm s^-1', name='y_velocity_at_u_locations')
	compare_dataarrays(diags['y_velocity_at_v_locations'], v_val)


if __name__ == '__main__':
	pytest.main([__file__])
