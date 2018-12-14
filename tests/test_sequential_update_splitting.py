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
from tasmania.core.physics_composite import SequentialUpdateSplitting
from tasmania.dynamics.homogeneous_isentropic_dycore import HomogeneousIsentropicDynamicalCore


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
	sus1 = SequentialUpdateSplitting(tendency2, tendency1, grid=grid)
	try:
		_ = sus1(state=state_dc, timestep=timestep)
		assert False
	except InvalidStateError:
		assert True

	#
	# successful
	#
	state_dc = deepcopy(state)
	sus2 = SequentialUpdateSplitting(tendency1, tendency2, grid=grid)
	try:
		_ = sus2(state=state_dc, timestep=timestep)
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
		grid, moist_on=False, time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind', horizontal_boundary_type='periodic',
		intermediate_parameterizations=None, diagnostics=None,
		damp_on=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=False,
		smooth_on=True, smooth_type='second_order', smooth_damp_depth=0,
		smooth_coeff=.03, smooth_at_every_stage=False,
		backend=gt.mode.NUMPY, dtype=state['air_isentropic_density'].dtype,
	)

	tendency1 = make_fake_tendency_1(grid)
	tendency2 = make_fake_tendency_2(grid)

	state_prv = dycore(state, {}, timestep)
	state_prv_dc = deepcopy(state_prv)

	sus = SequentialUpdateSplitting(
		tendency1, tendency2, grid=grid, time_integration_scheme='forward_euler'
	)
	_ = sus(state=state_prv_dc, timestep=timestep)

	assert 'fake_variable' in state_prv_dc
	s1 = state_prv['air_isentropic_density'].values
	f = state_prv_dc['fake_variable'].values
	assert np.allclose(f, 2*s1)

	assert 'air_isentropic_density' in state_prv_dc
	s2 = s1 + timestep.total_seconds() * s1**2
	s3 = s2 + timestep.total_seconds() * 1e-5*f
	assert np.allclose(state_prv_dc['air_isentropic_density'].values, s3)

	assert 'x_momentum_isentropic' in state_prv_dc
	su1 = state_prv['x_momentum_isentropic'].values
	su2 = su1 + timestep.total_seconds() * s1**3
	assert np.allclose(state_prv_dc['x_momentum_isentropic'].values, su2)

	assert 'y_momentum_isentropic' in state_prv_dc
	v1 = 3.6 * state_prv['y_velocity_at_v_locations'].values
	sv1 = state_prv['y_momentum_isentropic'].values
	sv3 = sv1 + timestep.total_seconds() * 0.5*(v1[:, :-1, :] + v1[:, 1:, :])
	assert np.allclose(state_prv_dc['y_momentum_isentropic'].values, sv3)


def test_numerics_rk2(
	isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	timestep = timedelta(seconds=10)

	dycore = HomogeneousIsentropicDynamicalCore(
		grid, moist_on=False, time_integration_scheme='rk3cosmo',
		horizontal_flux_scheme='fifth_order_upwind', horizontal_boundary_type='periodic',
		intermediate_parameterizations=None, diagnostics=None,
		damp_on=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=False,
		smooth_on=True, smooth_type='second_order', smooth_damp_depth=0,
		smooth_coeff=.03, smooth_at_every_stage=False,
	 	backend=gt.mode.NUMPY, dtype=state['air_isentropic_density'].dtype,
	)

	tendency1 = make_fake_tendency_1(grid)
	tendency2 = make_fake_tendency_2(grid)

	state_prv = dycore(state, {}, timestep)
	state_prv_dc = deepcopy(state_prv)

	sus = SequentialUpdateSplitting(
		tendency1, tendency2, grid=grid, time_integration_scheme='rk2',
	)
	_ = sus(state=state_prv_dc, timestep=timestep)

	assert 'fake_variable' in state_prv_dc
	s1 = state_prv['air_isentropic_density'].values
	f = state_prv_dc['fake_variable'].values
	assert np.allclose(f, 2*s1)

	assert 'air_isentropic_density' in state_prv_dc
	s2b = s1 + 0.5 * timestep.total_seconds() * s1**2
	s2 = s1 + timestep.total_seconds() * s2b**2
	s3 = s2 + timestep.total_seconds() * 1e-5*f
	assert np.allclose(state_prv_dc['air_isentropic_density'].values, s3)

	assert 'x_momentum_isentropic' in state_prv_dc
	su1 = state_prv['x_momentum_isentropic'].values
	su2b = su1 + timestep.total_seconds() * s1**3
	su2 = su1 + timestep.total_seconds() * s2b**3
	assert np.allclose(state_prv_dc['x_momentum_isentropic'].values, su2)

	assert 'y_momentum_isentropic' in state_prv_dc
	v1 = 3.6 * state_prv['y_velocity_at_v_locations'].values
	sv1 = state_prv['y_momentum_isentropic'].values
	sv3 = sv1 + timestep.total_seconds() * 0.5*(v1[:, :-1, :] + v1[:, 1:, :])
	assert np.allclose(state_prv_dc['y_momentum_isentropic'].values, sv3)


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
	#test_numerics_forward_euler(isentropic_dry_data(), make_fake_tendency_1(), make_fake_tendency_2())
