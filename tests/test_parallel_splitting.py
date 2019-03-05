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
import numpy as np
import pytest
from sympl._core.exceptions import InvalidStateError

import gridtools as gt
from tasmania.python.framework.parallel_splitting import ParallelSplitting
from tasmania.python.isentropic.dynamics.homogeneous_dycore import \
	HomogeneousIsentropicDynamicalCore


def test_compatibility(
	isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	timestep = timedelta(seconds=10)

	tendency1 = make_fake_tendency_1(grid)
	tendency2 = make_fake_tendency_2(grid)

	#
	# failing
	#
	state_dc = deepcopy(state)
	ps1 = ParallelSplitting(
		{'component': tendency1, 'time_integrator': 'forward_euler', 'substeps': 1},
		{'component': tendency2, 'time_integrator': 'forward_euler'},
		execution_policy='as_parallel'
	)
	try:
		ps1(state=state_dc, state_prv=state_dc, timestep=timestep)
		assert False
	except InvalidStateError:
		assert True

	#
	# failing
	#
	state_dc = deepcopy(state)
	ps2 = ParallelSplitting(
		{'component': tendency2, 'time_integrator': 'forward_euler', 'substeps': 1},
		{'component': tendency1, 'time_integrator': 'forward_euler'},
		execution_policy='serial'
	)
	try:
		ps2(state=state_dc, state_prv=state_dc, timestep=timestep)
		assert False
	except InvalidStateError:
		assert True

	#
	# successful
	#
	state_dc = deepcopy(state)
	ps3 = ParallelSplitting(
		{'component': tendency1, 'time_integrator': 'forward_euler', 'substeps': 1},
		{'component': tendency2, 'time_integrator': 'forward_euler'},
		execution_policy='serial'
	)
	try:
		ps3(state=state_dc, state_prv=state_dc, timestep=timestep)
		assert True
	except InvalidStateError:
		assert False

	#
	# successful
	#
	state_dc = deepcopy(state)
	ps4 = ParallelSplitting(
		{'component': tendency1, 'time_integrator': 'forward_euler', 'substeps': 2},
		{'component': tendency2, 'time_integrator': 'forward_euler', 'substeps': 3},
		execution_policy='serial'
	)
	try:
		ps4(state=state_dc, state_prv=state_dc, timestep=timestep)
		assert True
	except InvalidStateError:
		assert False


def test_numerics_forward_euler(
	isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	timestep = timedelta(seconds=10)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, moist=False, time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind', horizontal_boundary_type='periodic',
		intermediate_tendencies=None, intermediate_diagnostics=None,
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=False,
		smooth=True, smooth_type='second_order', smooth_damp_depth=0,
		smooth_coeff=.03, smooth_at_every_stage=False,
		backend=gt.mode.NUMPY, dtype=state['air_isentropic_density'].dtype,
	)

	tendency1 = make_fake_tendency_1(grid)
	tendency2 = make_fake_tendency_2(grid)

	state_dc = deepcopy(state)
	state_prv = dycore(state_dc, {}, timestep)
	state_prv_dc = deepcopy(state_prv)

	ps = ParallelSplitting(
		{'component': tendency1, 'time_integrator': 'forward_euler'},
		{'component': tendency2, 'time_integrator': 'forward_euler'},
		execution_policy='serial'
	)
	ps(state=state_dc, state_prv=state_prv_dc, timestep=timestep)

	assert 'fake_variable' in state_dc
	s = state['air_isentropic_density'].values
	f = state_dc['fake_variable'].values
	assert np.allclose(f, 2*s)

	assert 'air_isentropic_density' in state_prv_dc
	s1 = state_prv['air_isentropic_density'].values
	s2 = s + timestep.total_seconds() * s**2
	s3 = s + timestep.total_seconds() * 1e-5*f
	s_out = s1 + (s2 - s) + (s3 - s)
	assert np.allclose(state_prv_dc['air_isentropic_density'].values, s_out)

	assert 'x_momentum_isentropic' in state_prv_dc
	su = state['x_momentum_isentropic'].values
	su1 = state_prv['x_momentum_isentropic'].values
	su2 = su + timestep.total_seconds() * s**3
	su_out = su1 + (su2 - su)
	assert np.allclose(state_prv_dc['x_momentum_isentropic'].values, su_out)

	assert 'y_momentum_isentropic' in state_prv_dc
	v = 3.6 * state['y_velocity_at_v_locations'].values
	sv = state['y_momentum_isentropic'].values
	sv1 = state_prv['y_momentum_isentropic'].values
	sv3 = sv + timestep.total_seconds() * 0.5*(v[:, :-1, :] + v[:, 1:, :])
	sv_out = sv1 + (sv3 - sv)
	assert np.allclose(state_prv_dc['y_momentum_isentropic'].values, sv_out)


def test_numerics_rk2(
	isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	timestep = timedelta(seconds=10)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, moist=False, time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind', horizontal_boundary_type='periodic',
		intermediate_tendencies=None, intermediate_diagnostics=None,
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=False,
		smooth=True, smooth_type='second_order', smooth_damp_depth=0,
		smooth_coeff=.03, smooth_at_every_stage=False,
		backend=gt.mode.NUMPY, dtype=state['air_isentropic_density'].dtype,
	)

	tendency1 = make_fake_tendency_1(grid)
	tendency2 = make_fake_tendency_2(grid)

	state_dc = deepcopy(state)
	state_prv = dycore(state_dc, {}, timestep)
	state_prv_dc = deepcopy(state_prv)

	ps = ParallelSplitting(
		{'component': tendency1, 'time_integrator': 'rk2'},
		{'component': tendency2, 'time_integrator': 'rk2'},
		execution_policy='serial'
	)
	ps(state=state_dc, state_prv=state_prv_dc, timestep=timestep)

	assert 'fake_variable' in state_dc
	s = state['air_isentropic_density'].values
	f = state_dc['fake_variable'].values
	assert np.allclose(f, 2*s)

	assert 'air_isentropic_density' in state_prv_dc
	s1 = state_prv['air_isentropic_density'].values
	s2b = s + 0.5 * timestep.total_seconds() * s**2
	s2 = s + timestep.total_seconds() * s2b**2
	s3b = s + 0.5 * timestep.total_seconds() * 1e-5*f
	s3 = s + timestep.total_seconds() * 1e-5*f
	s_out = s1 + (s2 - s) + (s3 - s)
	assert np.allclose(state_prv_dc['air_isentropic_density'].values, s_out)

	assert 'x_momentum_isentropic' in state_prv_dc
	su = state['x_momentum_isentropic'].values
	su1 = state_prv['x_momentum_isentropic'].values
	su2b = su + 0.5 * timestep.total_seconds() * s**3
	su2 = su + timestep.total_seconds() * s2b**3
	su_out = su1 + (su2 - su)
	assert np.allclose(state_prv_dc['x_momentum_isentropic'].values, su_out)

	assert 'y_momentum_isentropic' in state_prv_dc
	v = 3.6 * state['y_velocity_at_v_locations'].values
	sv = state['y_momentum_isentropic'].values
	sv1 = state_prv['y_momentum_isentropic'].values
	sv3 = sv + timestep.total_seconds() * 0.5*(v[:, :-1, :] + v[:, 1:, :])
	sv_out = sv1 + (sv3 - sv)
	assert np.allclose(state_prv_dc['y_momentum_isentropic'].values, sv_out)


def test_numerics_rk2_substepping(
	isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	timestep = timedelta(seconds=10)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, moist=False, time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind', horizontal_boundary_type='periodic',
		intermediate_tendencies=None, intermediate_diagnostics=None,
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=False,
		smooth=True, smooth_type='second_order', smooth_damp_depth=0,
		smooth_coeff=.03, smooth_at_every_stage=False,
		backend=gt.mode.NUMPY, dtype=state['air_isentropic_density'].dtype,
	)

	tendency1 = make_fake_tendency_1(grid)
	tendency2 = make_fake_tendency_2(grid)

	state_dc = deepcopy(state)
	state_prv = dycore(state_dc, {}, timestep)
	state_prv_dc = deepcopy(state_prv)

	ps = ParallelSplitting(
		{'component': tendency1, 'time_integrator': 'forward_euler', 'substeps': 3},
		{'component': tendency2, 'time_integrator': 'rk2'},
		execution_policy='serial'
	)
	ps(state=state_dc, state_prv=state_prv_dc, timestep=timestep)

	assert 'fake_variable' in state_dc
	s = state['air_isentropic_density'].values
	f = state_dc['fake_variable'].values
	assert np.allclose(f, 2*s)

	assert 'air_isentropic_density' in state_prv_dc
	s1 = state_prv['air_isentropic_density'].values
	s21 = s + timestep.total_seconds()/3 * s**2
	s22 = s21 + timestep.total_seconds()/3 * s21**2
	s2 = s22 + timestep.total_seconds()/3 * s22**2
	s3 = s + timestep.total_seconds() * 1e-5*f
	s_out = s1 + (s2 - s) + (s3 - s)
	assert np.allclose(state_prv_dc['air_isentropic_density'].values, s_out)

	assert 'x_momentum_isentropic' in state_prv_dc
	su = state['x_momentum_isentropic'].values
	su1 = state_prv['x_momentum_isentropic'].values
	su21 = su + timestep.total_seconds()/3 * s**3
	su22 = su21 + timestep.total_seconds()/3 * s21**3
	su2 = su22 + timestep.total_seconds()/3 * s22**3
	su_out = su1 + (su2 - su)
	assert np.allclose(state_prv_dc['x_momentum_isentropic'].values, su_out)

	assert 'y_momentum_isentropic' in state_prv_dc
	v = 3.6 * state['y_velocity_at_v_locations'].values
	sv = state['y_momentum_isentropic'].values
	sv1 = state_prv['y_momentum_isentropic'].values
	sv3 = sv + timestep.total_seconds() * 0.5*(v[:, :-1, :] + v[:, 1:, :])
	sv_out = sv1 + (sv3 - sv)
	assert np.allclose(state_prv_dc['y_momentum_isentropic'].values, sv_out)


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
	#test_numerics_rk2(isentropic_dry_data(), make_fake_tendency_1(), make_fake_tendency_2())
