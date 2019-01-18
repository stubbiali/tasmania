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
from datetime import timedelta
import pytest

import gridtools as gt
from tasmania.python.physics.microphysics import \
	RaindropFallVelocity, Sedimentation


def test_first_order_upwind(isentropic_moist_sedimentation_data):
	grid, states = isentropic_moist_sedimentation_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY

	rfv = RaindropFallVelocity(grid, backend=backend)

	diagnostics = rfv(state)
	state.update(diagnostics)

	sd = Sedimentation(
		grid, sedimentation_flux_scheme='first_order_upwind', backend=backend
	)

	dt = timedelta(seconds=120)

	tendencies, diagnostics = sd(state, dt)

	assert 'mass_fraction_of_precipitation_water_in_air' in tendencies
	assert len(tendencies) == 1

	assert 'precipitation' in diagnostics
	assert len(diagnostics) == 1


def test_second_order_upwind(isentropic_moist_sedimentation_data):
	grid, states = isentropic_moist_sedimentation_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY

	rfv = RaindropFallVelocity(grid, backend=backend)

	diagnostics = rfv(state)
	state.update(diagnostics)

	sd = Sedimentation(
		grid, sedimentation_flux_scheme='second_order_upwind', backend=backend
	)

	dt = timedelta(seconds=120)

	tendencies, diagnostics = sd(state, dt)

	assert 'mass_fraction_of_precipitation_water_in_air' in tendencies
	assert len(tendencies) == 1

	assert 'precipitation' in diagnostics
	assert len(diagnostics) == 1


if __name__ == '__main__':
	pytest.main([__file__])
