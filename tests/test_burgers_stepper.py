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
	given, HealthCheck, reproduce_failure, settings, strategies as hyp_st
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
from tasmania.python.dwarfs.horizontal_boundary import HorizontalBoundary


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_forward_euler(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(2*taz_conf.nb+1, 40),
			yaxis_length=(2*taz_conf.nb+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)
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
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)
	hb_type = data.draw(utils.st_one_of(conf.horizontal_boundary), label='hb_type')
	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = data.draw(utils.st_one_of(conf.datatype), label='dtype')

	# ========================================
	# test
	# ========================================
	hb = HorizontalBoundary.factory(hb_type, grid, 1)  # TODO: 1 ==> taz_conf.nb
	bs = BurgersStepper.factory(
		'forward_euler', grid, 'first_order', hb, backend, dtype
	)

	assert isinstance(bs, _ForwardEuler)

	state_cd = {
		'time': state['time'],
		'x_velocity': hb.from_physical_to_computational_domain(
				state['x_velocity'].to_units('m s^-1').values
			),
		'y_velocity': hb.from_physical_to_computational_domain(
				state['y_velocity'].to_units('m s^-1').values
			),
	}
	if if_tendency:
		tendency_cd = {
			'time': state['time'],
			'x_velocity': hb.from_physical_to_computational_domain(
				tendency['x_velocity'].to_units('m s^-2').values
			),
			'y_velocity': hb.from_physical_to_computational_domain(
				tendency['y_velocity'].to_units('m s^-2').values
			),
		}
	else:
		tendency_cd = {}

	out_state = bs(0, state_cd, tendency_cd, timestep)

	nb = hb.nb
	dx, dy = grid.dx.to_units('m').values.item(), grid.dy.to_units('m').values.item()
	u, v = state_cd['x_velocity'], state_cd['y_velocity']
	if if_tendency:
		tnd_u, tnd_v = tendency_cd['x_velocity'], tendency_cd['y_velocity']

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
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(2*taz_conf.nb+1, 40),
			yaxis_length=(2*taz_conf.nb+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)
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
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)
	hb_type = data.draw(utils.st_one_of(conf.horizontal_boundary), label='hb_type')
	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = data.draw(utils.st_one_of(conf.datatype), label='dtype')

	# ========================================
	# test
	# ========================================
	hb = HorizontalBoundary.factory(hb_type, grid, 2)  # TODO: 2 ==> taz_conf.nb
	bs = BurgersStepper.factory(
		'rk2', grid, 'third_order', hb, backend, dtype
	)

	assert isinstance(bs, _RK2)

	state_0_cd = {
		'time': state['time'],
		'x_velocity': hb.from_physical_to_computational_domain(
			state['x_velocity'].to_units('m s^-1').values
		),
		'y_velocity': hb.from_physical_to_computational_domain(
			state['y_velocity'].to_units('m s^-1').values
		),
	}
	if if_tendency:
		tendency_cd = {
			'time': state['time'],
			'x_velocity': hb.from_physical_to_computational_domain(
				tendency['x_velocity'].to_units('m s^-2').values
			),
			'y_velocity': hb.from_physical_to_computational_domain(
				tendency['y_velocity'].to_units('m s^-2').values
			),
		}
	else:
		tendency_cd = {}

	# ========================================
	# stage 0
	# ========================================
	state_1_cd = bs(0, state_0_cd, tendency_cd, timestep)

	nb = hb.nb
	dx, dy = grid.dx.to_units('m').values.item(), grid.dy.to_units('m').values.item()
	u0, v0 = state_0_cd['x_velocity'], state_0_cd['y_velocity']
	if if_tendency:
		tnd_u, tnd_v = tendency_cd['x_velocity'], tendency_cd['y_velocity']

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

	#assert state_1_cd['time'] == state['time'] + 0.5*timestep
	assert np.allclose(u1, state_1_cd['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v1, state_1_cd['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)

	# ========================================
	# stage 1
	# ========================================
	state_1_cd = deepcopy(state_1_cd)
	state_2_cd = bs(1, state_1_cd, tendency_cd, timestep)

	u1, v1 = state_1_cd['x_velocity'], state_1_cd['y_velocity']

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

	#assert state_2_cd['time'] == state['time'] + timestep
	assert np.allclose(u2, state_2_cd['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v2, state_2_cd['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_rk3ws(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(2*taz_conf.nb+1, 40),
			yaxis_length=(2*taz_conf.nb+1, 40),
			zaxis_length=(1, 1)
		),
		label='grid'
	)
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
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)
	hb_type = data.draw(utils.st_one_of(conf.horizontal_boundary), label='hb_type')
	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = data.draw(utils.st_one_of(conf.datatype), label='dtype')

	# ========================================
	# test
	# ========================================
	hb = HorizontalBoundary.factory(hb_type, grid, 3)  # TODO: 3 ==> taz_conf.nb
	bs = BurgersStepper.factory(
		'rk3ws', grid, 'fifth_order', hb, backend, dtype
	)

	assert isinstance(bs, _RK2)
	assert isinstance(bs, _RK3WS)

	state_0_cd = {
		'time': state['time'],
		'x_velocity': hb.from_physical_to_computational_domain(
			state['x_velocity'].to_units('m s^-1').values
		),
		'y_velocity': hb.from_physical_to_computational_domain(
			state['y_velocity'].to_units('m s^-1').values
		),
	}
	if if_tendency:
		tendency_cd = {
			'time': state['time'],
			'x_velocity': hb.from_physical_to_computational_domain(
				tendency['x_velocity'].to_units('m s^-2').values
			),
			'y_velocity': hb.from_physical_to_computational_domain(
				tendency['y_velocity'].to_units('m s^-2').values
			),
		}
	else:
		tendency_cd = {}

	# ========================================
	# stage 0
	# ========================================
	state_1_cd = bs(0, state_0_cd, tendency_cd, timestep)

	nb = hb.nb
	dx, dy = grid.dx.to_units('m').values.item(), grid.dy.to_units('m').values.item()
	u0, v0 = state_0_cd['x_velocity'], state_0_cd['y_velocity']
	if if_tendency:
		tnd_u, tnd_v = tendency_cd['x_velocity'], tendency_cd['y_velocity']

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

	#assert state_1_cd['time'] == state['time'] + 1.0/3.0*timestep
	assert np.allclose(u1, state_1_cd['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v1, state_1_cd['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)

	# ========================================
	# stage 1
	# ========================================
	state_1_cd = deepcopy(state_1_cd)
	state_2_cd = bs(1, state_1_cd, tendency_cd, timestep)

	u1, v1 = state_1_cd['x_velocity'], state_1_cd['y_velocity']

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

	#assert state_2_cd['time'] == state['time'] + timestep
	assert np.allclose(u2, state_2_cd['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v2, state_2_cd['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)

	# ========================================
	# stage 1
	# ========================================
	state_2_cd = deepcopy(state_2_cd)
	state_3_cd = bs(2, state_2_cd, tendency_cd, timestep)

	u2, v2 = state_2_cd['x_velocity'], state_2_cd['y_velocity']

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

	#assert state_2_cd['time'] == state['time'] + timestep
	assert np.allclose(u3, state_3_cd['x_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)
	assert np.allclose(v3, state_3_cd['y_velocity'][nb:-nb, nb:-nb, :], atol=1e-6)


if __name__ == '__main__':
	pytest.main([__file__])
	#test_rk3ws()
