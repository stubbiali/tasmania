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

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling


def test_compatibility(
	isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	tendency1 = make_fake_tendency_1(grid)
	tendency2 = make_fake_tendency_2(grid)

	dt = timedelta(minutes=1)

	#
	# failing
	#
	state_dc = deepcopy(state)
	cc1 = ConcurrentCoupling(tendency1, tendency2, execution_policy='as_parallel')
	try:
		cc1(state_dc, dt)
		assert False
	except InvalidStateError:
		assert True

	#
	# failing
	#
	state_dc = deepcopy(state)
	cc2 = ConcurrentCoupling(tendency2, tendency1, execution_policy='serial')
	try:
		cc2(state_dc, dt)
		assert False
	except InvalidStateError:
		assert True

	#
	# successful
	#
	state_dc = deepcopy(state)
	cc3 = ConcurrentCoupling(tendency1, tendency2, execution_policy='serial')
	try:
		cc3(state_dc, dt)
		assert True
	except InvalidStateError:
		assert False


def test_numerics(
	isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2
):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	tendency1 = make_fake_tendency_1(grid)
	tendency2 = make_fake_tendency_2(grid)

	dt = timedelta(seconds=100)

	cc = ConcurrentCoupling(tendency1, tendency2, execution_policy='serial')
	tendencies, diagnostics = cc(state, dt)

	assert 'fake_variable' in diagnostics
	s = state['air_isentropic_density'].values
	f = diagnostics['fake_variable'].values
	assert np.allclose(f, 2*s)

	assert 'air_isentropic_density' in tendencies
	assert np.allclose(tendencies['air_isentropic_density'].values, s**2 + 1e-5*f)

	assert 'x_momentum_isentropic' in tendencies
	assert np.allclose(tendencies['x_momentum_isentropic'].values, s**3)

	assert 'y_momentum_isentropic' in tendencies
	v = 3.6 * state['y_velocity_at_v_locations'].values
	assert np.allclose(
		tendencies['y_momentum_isentropic'].values,
		0.5 * (v[:, :-1, :] + v[:, 1:, :]),
	)


if __name__ == '__main__':
	pytest.main([__file__])

	#from conftest import FakeTendency1, FakeTendency2
	#from tasmania.python.utils.storage_utils import load_netcdf_dataset
	#
	#isentropic_dry_data = load_netcdf_dataset('baseline_datasets/isentropic_dry.nc')
	#make_fake_tendency_1 = lambda grid: FakeTendency1(grid)
	#make_fake_tendency_2 = lambda grid: FakeTendency2(grid)
	#
	#test_numerics(isentropic_dry_data, make_fake_tendency_1, make_fake_tendency_2)
