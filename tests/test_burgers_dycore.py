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
from tasmania.python.burgers.dynamics.dycore import BurgersDynamicalCore
from tasmania.python.burgers.state import ZhaoSolutionFactory
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
	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)
	hb_type = data.draw(
		utils.st_one_of(('periodic', 'relaxed', 'zhao')), label='hb_type'
	)
	eps = data.draw(utils.st_floats(min_value=-1e3, max_value=1e3))
	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = data.draw(utils.st_one_of(conf.datatype), label='dtype')

	# ========================================
	# test
	# ========================================
	zsf = ZhaoSolutionFactory(eps)
	hb = HorizontalBoundary.factory(
		hb_type, grid, 1, init_time=state['time'], solution_factory=zsf
	)
	dycore = BurgersDynamicalCore(
		grid, time_units='s', intermediate_tendencies=None,
		time_integration_scheme='forward_euler', flux_scheme='first_order',
		boundary=hb, backend=backend, dtype=dtype
	)

	new_state = dycore(state, {}, timestep)

	assert 'time' in new_state
	assert 'x_velocity' in new_state
	assert 'y_velocity' in new_state
	assert len(new_state) == 3

	assert new_state['time'] == state['time'] + timestep

	dt = timestep.total_seconds()
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = state['x_velocity'].to_units('m s^-1').values
	v = state['y_velocity'].to_units('m s^-1').values

	u0 = hb.from_physical_to_computational_domain(u)
	v0 = hb.from_physical_to_computational_domain(v)

	adv_u_x, adv_u_y = first_order_advection(dx, dy, u0, v0, u0)
	adv_v_x, adv_v_y = first_order_advection(dx, dy, u0, v0, v0)

	u1 = u0 - dt * (adv_u_x + adv_u_y)
	v1 = v0 - dt * (adv_v_x + adv_v_y)

	u1 = hb.from_computational_to_physical_domain(u1, out_dims=(grid.nx, grid.ny, grid.nz))
	v1 = hb.from_computational_to_physical_domain(v1, out_dims=(grid.nx, grid.ny, grid.nz))

	hb.enforce(u1, u0, field_name='x_velocity', time=new_state['time'])
	hb.enforce(v1, v0, field_name='y_velocity', time=new_state['time'])

	assert new_state['x_velocity'].attrs['units'] == 'm s^-1'
	assert np.allclose(u1, new_state['x_velocity'], equal_nan=True)

	assert new_state['y_velocity'].attrs['units'] == 'm s^-1'
	assert np.allclose(v1, new_state['y_velocity'], equal_nan=True)


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
	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)
	hb_type = data.draw(
		utils.st_one_of(('periodic', 'relaxed', 'zhao')), label='hb_type'
	)
	eps = data.draw(utils.st_floats(min_value=-1e3, max_value=1e3))
	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = data.draw(utils.st_one_of(conf.datatype), label='dtype')

	# ========================================
	# test
	# ========================================
	zsf = ZhaoSolutionFactory(eps)
	hb = HorizontalBoundary.factory(
		hb_type, grid, 2, init_time=state['time'], solution_factory=zsf
	)
	dycore = BurgersDynamicalCore(
		grid, time_units='s', intermediate_tendencies=None,
		time_integration_scheme='rk2', flux_scheme='third_order',
		boundary=hb, backend=backend, dtype=dtype
	)

	new_state = dycore(state, {}, timestep)

	assert 'time' in new_state
	assert 'x_velocity' in new_state
	assert 'y_velocity' in new_state
	assert len(new_state) == 3

	assert new_state['time'] == state['time'] + timestep

	dt = timestep.total_seconds()
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = state['x_velocity'].to_units('m s^-1').values
	v = state['y_velocity'].to_units('m s^-1').values

	u0 = hb.from_physical_to_computational_domain(u)
	v0 = hb.from_physical_to_computational_domain(v)

	adv_u_x, adv_u_y = third_order_advection(dx, dy, u0, v0, u0)
	adv_v_x, adv_v_y = third_order_advection(dx, dy, u0, v0, v0)

	u1 = u0 - 0.5 * dt * (adv_u_x + adv_u_y)
	v1 = v0 - 0.5 * dt * (adv_v_x + adv_v_y)

	u1 = hb.from_computational_to_physical_domain(u1, out_dims=(grid.nx, grid.ny, grid.nz))
	v1 = hb.from_computational_to_physical_domain(v1, out_dims=(grid.nx, grid.ny, grid.nz))

	hb.enforce(u1, u0, field_name='x_velocity', time=state['time']+0.5*timestep)
	hb.enforce(v1, v0, field_name='y_velocity', time=state['time']+0.5*timestep)

	u1 = hb.from_physical_to_computational_domain(u1)
	v1 = hb.from_physical_to_computational_domain(v1)

	adv_u_x, adv_u_y = third_order_advection(dx, dy, u1, v1, u1)
	adv_v_x, adv_v_y = third_order_advection(dx, dy, u1, v1, v1)

	u2 = u0 - dt * (adv_u_x + adv_u_y)
	v2 = v0 - dt * (adv_v_x + adv_v_y)

	u2 = hb.from_computational_to_physical_domain(u2, out_dims=(grid.nx, grid.ny, grid.nz))
	v2 = hb.from_computational_to_physical_domain(v2, out_dims=(grid.nx, grid.ny, grid.nz))

	hb.enforce(u2, u0, field_name='x_velocity', time=state['time']+timestep)
	hb.enforce(v2, v0, field_name='y_velocity', time=state['time']+timestep)

	assert new_state['x_velocity'].attrs['units'] == 'm s^-1'
	assert np.allclose(u2, new_state['x_velocity'], equal_nan=True)

	assert new_state['y_velocity'].attrs['units'] == 'm s^-1'
	assert np.allclose(v2, new_state['y_velocity'], equal_nan=True)


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
	timestep = data.draw(
		hyp_st.timedeltas(
			min_value=timedelta(seconds=0),
			max_value=timedelta(seconds=120)
		),
		label='timestep'
	)
	hb_type = data.draw(
		utils.st_one_of(('periodic', 'relaxed', 'zhao')), label='hb_type'
	)
	eps = data.draw(utils.st_floats(min_value=-1e3, max_value=1e3))
	backend = data.draw(utils.st_one_of(conf.backend), label='backend')
	dtype = data.draw(utils.st_one_of(conf.datatype), label='dtype')

	# ========================================
	# test
	# ========================================
	zsf = ZhaoSolutionFactory(eps)
	hb = HorizontalBoundary.factory(
		hb_type, grid, 3, init_time=state['time'], solution_factory=zsf
	)
	dycore = BurgersDynamicalCore(
		grid, time_units='s', intermediate_tendencies=None,
		time_integration_scheme='rk3ws', flux_scheme='fifth_order',
		boundary=hb, backend=backend, dtype=dtype
	)

	new_state = dycore(state, {}, timestep)

	assert 'time' in new_state
	assert 'x_velocity' in new_state
	assert 'y_velocity' in new_state
	assert len(new_state) == 3

	assert new_state['time'] == state['time'] + timestep

	dt = timestep.total_seconds()
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	u = state['x_velocity'].to_units('m s^-1').values
	v = state['y_velocity'].to_units('m s^-1').values

	u0 = hb.from_physical_to_computational_domain(u)
	v0 = hb.from_physical_to_computational_domain(v)

	adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u0, v0, u0)
	adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u0, v0, v0)

	u1 = u0 - 1.0/3.0 * dt * (adv_u_x + adv_u_y)
	v1 = v0 - 1.0/3.0 * dt * (adv_v_x + adv_v_y)

	u1 = hb.from_computational_to_physical_domain(u1, out_dims=(grid.nx, grid.ny, grid.nz))
	v1 = hb.from_computational_to_physical_domain(v1, out_dims=(grid.nx, grid.ny, grid.nz))

	hb.enforce(u1, u0, field_name='x_velocity', time=state['time']+1.0/3.0*timestep)
	hb.enforce(v1, v0, field_name='y_velocity', time=state['time']+1.0/3.0*timestep)

	u1 = hb.from_physical_to_computational_domain(u1)
	v1 = hb.from_physical_to_computational_domain(v1)

	adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u1, v1, u1)
	adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u1, v1, v1)

	u2 = u0 - 0.5 * dt * (adv_u_x + adv_u_y)
	v2 = v0 - 0.5 * dt * (adv_v_x + adv_v_y)

	u2 = hb.from_computational_to_physical_domain(u2, out_dims=(grid.nx, grid.ny, grid.nz))
	v2 = hb.from_computational_to_physical_domain(v2, out_dims=(grid.nx, grid.ny, grid.nz))

	hb.enforce(u2, u0, field_name='x_velocity', time=state['time']+0.5*timestep)
	hb.enforce(v2, v0, field_name='y_velocity', time=state['time']+0.5*timestep)

	u2 = hb.from_physical_to_computational_domain(u2)
	v2 = hb.from_physical_to_computational_domain(v2)

	adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u2, v2, u2)
	adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u2, v2, v2)

	u3 = u0 - dt * (adv_u_x + adv_u_y)
	v3 = v0 - dt * (adv_v_x + adv_v_y)

	u3 = hb.from_computational_to_physical_domain(u3, out_dims=(grid.nx, grid.ny, grid.nz))
	v3 = hb.from_computational_to_physical_domain(v3, out_dims=(grid.nx, grid.ny, grid.nz))

	hb.enforce(u3, u0, field_name='x_velocity', time=state['time']+timestep)
	hb.enforce(v3, v0, field_name='y_velocity', time=state['time']+timestep)

	assert new_state['x_velocity'].attrs['units'] == 'm s^-1'
	assert np.allclose(u3, new_state['x_velocity'], equal_nan=True)

	assert new_state['y_velocity'].attrs['units'] == 'm s^-1'
	assert np.allclose(v3, new_state['y_velocity'], equal_nan=True)


if __name__ == '__main__':
	pytest.main([__file__])
	#test_rk3ws()
