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
from datetime import datetime, timedelta
from hypothesis import \
	given, HealthCheck, settings, strategies as hyp_st
import numpy as np
import pytest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils
from test_burgers_advection import \
	first_order_advection, third_order_advection, fifth_order_advection

import tasmania.conf as taz_conf
from tasmania.python.burgers.dynamics.stepper import \
	BurgersStepper, _ForwardEuler, _RK2, _RK3WS
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary
from tasmania.python.grids.grid import NumericalGrid


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_forward_euler(data):
	# ========================================
	# random data generation
	# ========================================
	pgrid = data.draw(
		utils.st_physical_grid(
			xaxis_length=(2*taz_conf.nb+1, 40),
			yaxis_length=(2*taz_conf.nb+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)

	nx, ny = pgrid.grid_xy.nx, pgrid.grid_xy.ny
	hb_type = data.draw(utils.st_horizontal_boundary_type(), label='hb_type')
	nb = 1  # TODO: nb = data.draw(utils.st_horizontal_boundary_layers(nx, ny))
	hb_kwargs = data.draw(
		utils.st_horizontal_boundary_kwargs(hb_type, nx, ny, nb), label='hb_kwargs'
	)
	hb = HorizontalBoundary.factory(hb_type, nx, ny, nb, **hb_kwargs)

	grid = NumericalGrid(pgrid, hb)

	state = data.draw(
		utils.st_burgers_state(grid, time=datetime(year=1992, month=2, day=20)),
		label='in_state'
	)

	if_tendency = data.draw(hyp_st.booleans(), label='if_tendency')
	tendency = {} if not if_tendency else \
		data.draw(
			utils.st_burgers_tendency(grid, time=state['time']), label='tendency'
		)

	timestep = data.draw(
		utils.st_timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)

	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = grid.grid_xy.x.dtype

	# ========================================
	# test
	# ========================================
	bs = BurgersStepper.factory(
		'forward_euler', grid.grid_xy, nb, 'first_order', backend, dtype
	)

	assert isinstance(bs, _ForwardEuler)

	raw_state = {
		'time': state['time'],
		'x_velocity': state['x_velocity'].to_units('m s^-1').values,
		'y_velocity': state['y_velocity'].to_units('m s^-1').values,
	}
	if if_tendency:
		raw_tendency = {
			'time': state['time'],
			'x_velocity': tendency['x_velocity'].to_units('m s^-2').values,
			'y_velocity': tendency['y_velocity'].to_units('m s^-2').values,
		}
	else:
		raw_tendency = {}

	out_state = bs(0, raw_state, raw_tendency, timestep)

	nb = hb.nb
	dx = grid.grid_xy.dx.to_units('m').values.item()
	dy = grid.grid_xy.dy.to_units('m').values.item()
	u, v = raw_state['x_velocity'], raw_state['y_velocity']
	if if_tendency:
		tnd_u, tnd_v = raw_tendency['x_velocity'], raw_tendency['y_velocity']

	adv_u_x, adv_u_y = first_order_advection(dx, dy, u, v, u)
	adv_v_x, adv_v_y = first_order_advection(dx, dy, u, v, v)
	out_u = u[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
			adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
		)
	out_v = v[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
			adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
		)
	if if_tendency:
		out_u += timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
		out_v += timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

	assert np.allclose(out_u, out_state['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(out_v, out_state['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_rk2(data):
	# ========================================
	# random data generation
	# ========================================
	pgrid = data.draw(
		utils.st_physical_grid(
			xaxis_length=(2*taz_conf.nb+1, 40),
			yaxis_length=(2*taz_conf.nb+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)

	nx, ny = pgrid.grid_xy.nx, pgrid.grid_xy.ny
	hb_type = data.draw(utils.st_horizontal_boundary_type(), label='hb_type')
	nb = 2  # TODO: nb = data.draw(utils.st_horizontal_boundary_layers(nx, ny))
	hb_kwargs = data.draw(
		utils.st_horizontal_boundary_kwargs(hb_type, nx, ny, nb), label='hb_kwargs'
	)
	hb = HorizontalBoundary.factory(hb_type, nx, ny, nb, **hb_kwargs)

	grid = NumericalGrid(pgrid, hb)

	state = data.draw(
		utils.st_burgers_state(grid, time=datetime(year=1992, month=2, day=20)),
		label='in_state'
	)

	if_tendency = data.draw(hyp_st.booleans(), label='if_tendency')
	tendency = {} if not if_tendency else \
		data.draw(
			utils.st_burgers_tendency(grid, time=state['time']), label='tendency'
		)

	timestep = data.draw(
		utils.st_timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)

	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = grid.grid_xy.x.dtype

	# ========================================
	# test
	# ========================================
	bs = BurgersStepper.factory(
		'rk2', grid.grid_xy, nb, 'third_order', backend, dtype
	)

	assert isinstance(bs, _RK2)

	raw_state_0 = {
		'time': state['time'],
		'x_velocity': state['x_velocity'].to_units('m s^-1').values,
		'y_velocity': state['y_velocity'].to_units('m s^-1').values,
	}
	if if_tendency:
		raw_tendency = {
			'time': state['time'],
			'x_velocity': tendency['x_velocity'].to_units('m s^-2').values,
			'y_velocity': tendency['y_velocity'].to_units('m s^-2').values,
		}
	else:
		raw_tendency = {}

	# ========================================
	# stage 0
	# ========================================
	raw_state_1 = bs(0, raw_state_0, raw_tendency, timestep)

	nb = hb.nb
	dx = grid.grid_xy.dx.to_units('m').values.item()
	dy = grid.grid_xy.dy.to_units('m').values.item()
	u0, v0 = raw_state_0['x_velocity'], raw_state_0['y_velocity']
	if if_tendency:
		tnd_u, tnd_v = raw_tendency['x_velocity'], raw_tendency['y_velocity']

	adv_u_x, adv_u_y = third_order_advection(dx, dy, u0, v0, u0)
	adv_v_x, adv_v_y = third_order_advection(dx, dy, u0, v0, v0)
	u1 = u0[nb:-nb, nb:-nb, :] - 0.5 * timestep.total_seconds() * (
		adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
	)
	v1 = v0[nb:-nb, nb:-nb, :] - 0.5 * timestep.total_seconds() * (
		adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
	)
	if if_tendency:
		u1 += 0.5 * timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
		v1 += 0.5 * timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

	#assert raw_state_1['time'] == state['time'] + 0.5*timestep
	assert np.allclose(u1, raw_state_1['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v1, raw_state_1['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)

	# ========================================
	# stage 1
	# ========================================
	raw_state_1 = deepcopy(raw_state_1)
	raw_state_2 = bs(1, raw_state_1, raw_tendency, timestep)

	u1, v1 = raw_state_1['x_velocity'], raw_state_1['y_velocity']

	adv_u_x, adv_u_y = third_order_advection(dx, dy, u1, v1, u1)
	adv_v_x, adv_v_y = third_order_advection(dx, dy, u1, v1, v1)
	u2 = u0[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
		adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
	)
	v2 = v0[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
		adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
	)
	if if_tendency:
		u2 += timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
		v2 += timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

	#assert raw_state_2['time'] == state['time'] + timestep
	assert np.allclose(u2, raw_state_2['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v2, raw_state_2['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_rk3ws(data):
	# ========================================
	# random data generation
	# ========================================
	pgrid = data.draw(
		utils.st_physical_grid(
			xaxis_length=(2*taz_conf.nb+1, 40),
			yaxis_length=(2*taz_conf.nb+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)

	nx, ny = pgrid.grid_xy.nx, pgrid.grid_xy.ny
	hb_type = data.draw(utils.st_horizontal_boundary_type(), label='hb_type')
	nb = 3  # TODO: nb = data.draw(utils.st_horizontal_boundary_layers(nx, ny))
	hb_kwargs = data.draw(
		utils.st_horizontal_boundary_kwargs(hb_type, nx, ny, nb), label='hb_kwargs'
	)
	hb = HorizontalBoundary.factory(hb_type, nx, ny, nb, **hb_kwargs)

	grid = NumericalGrid(pgrid, hb)

	state = data.draw(
		utils.st_burgers_state(grid, time=datetime(year=1992, month=2, day=20)),
		label='in_state'
	)

	if_tendency = data.draw(hyp_st.booleans(), label='if_tendency')
	tendency = {} if not if_tendency else \
		data.draw(
			utils.st_burgers_tendency(grid, time=state['time']), label='tendency'
		)

	timestep = data.draw(
		utils.st_timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)

	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = grid.grid_xy.x.dtype

	# ========================================
	# test
	# ========================================
	bs = BurgersStepper.factory(
		'rk3ws', grid.grid_xy, nb, 'fifth_order', backend, dtype
	)

	assert isinstance(bs, _RK2)
	assert isinstance(bs, _RK3WS)

	raw_state_0 = {
		'time': state['time'],
		'x_velocity': state['x_velocity'].to_units('m s^-1').values,
		'y_velocity': state['y_velocity'].to_units('m s^-1').values,
	}
	if if_tendency:
		raw_tendency = {
			'time': state['time'],
			'x_velocity': tendency['x_velocity'].to_units('m s^-2').values,
			'y_velocity': tendency['y_velocity'].to_units('m s^-2').values,
		}
	else:
		raw_tendency = {}

	# ========================================
	# stage 0
	# ========================================
	raw_state_1 = bs(0, raw_state_0, raw_tendency, timestep)

	nb = hb.nb
	dx = grid.grid_xy.dx.to_units('m').values.item()
	dy = grid.grid_xy.dy.to_units('m').values.item()
	u0, v0 = raw_state_0['x_velocity'], raw_state_0['y_velocity']
	if if_tendency:
		tnd_u, tnd_v = raw_tendency['x_velocity'], raw_tendency['y_velocity']

	adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u0, v0, u0)
	adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u0, v0, v0)
	u1 = u0[nb:-nb, nb:-nb, :] - 1.0/3.0 * timestep.total_seconds() * (
		adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
	)
	v1 = v0[nb:-nb, nb:-nb, :] - 1.0/3.0 * timestep.total_seconds() * (
		adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
	)
	if if_tendency:
		u1 += 1.0/3.0 * timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
		v1 += 1.0/3.0 * timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

	#assert raw_state_1['time'] == state['time'] + 1.0/3.0*timestep
	assert np.allclose(u1, raw_state_1['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v1, raw_state_1['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)

	# ========================================
	# stage 1
	# ========================================
	raw_state_1 = deepcopy(raw_state_1)
	raw_state_2 = bs(1, raw_state_1, raw_tendency, timestep)

	u1, v1 = raw_state_1['x_velocity'], raw_state_1['y_velocity']

	adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u1, v1, u1)
	adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u1, v1, v1)
	u2 = u0[nb:-nb, nb:-nb, :] - 0.5 * timestep.total_seconds() * (
		adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
	)
	v2 = v0[nb:-nb, nb:-nb, :] - 0.5 * timestep.total_seconds() * (
		adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
	)
	if if_tendency:
		u2 += 0.5 * timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
		v2 += 0.5 * timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

	#assert raw_state_2['time'] == state['time'] + timestep
	assert np.allclose(u2, raw_state_2['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v2, raw_state_2['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)

	# ========================================
	# stage 1
	# ========================================
	raw_state_2 = deepcopy(raw_state_2)
	raw_state_3 = bs(2, raw_state_2, raw_tendency, timestep)

	u2, v2 = raw_state_2['x_velocity'], raw_state_2['y_velocity']

	adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u2, v2, u2)
	adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u2, v2, v2)
	u3 = u0[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
		adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
	)
	v3 = v0[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
		adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
	)
	if if_tendency:
		u3 += timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
		v3 += timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

	#assert raw_state_2['time'] == state['time'] + timestep
	assert np.allclose(u3, raw_state_3['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v3, raw_state_3['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)


if __name__ == '__main__':
	pytest.main([__file__])
