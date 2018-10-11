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
import numpy as np
import pytest

import gridtools as gt
from tasmania.physics.isentropic import IsentropicDiagnostics, \
										IsentropicVelocityComponents


def test_dry(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diag = IsentropicDiagnostics(grid, False,
								 state['air_pressure_on_interface_levels'][0, 0, 0],
								 backend=backend, dtype=dtype)

	diagnostics = diag(state)

	assert 'air_pressure_on_interface_levels' in diagnostics.keys()
	assert np.allclose(state['air_pressure_on_interface_levels'],
					   diagnostics['air_pressure_on_interface_levels'])
	diagnostics.pop('air_pressure_on_interface_levels')

	assert 'exner_function_on_interface_levels' in diagnostics.keys()
	assert np.allclose(state['exner_function_on_interface_levels'],
					   diagnostics['exner_function_on_interface_levels'])
	diagnostics.pop('exner_function_on_interface_levels')

	assert 'montgomery_potential' in diagnostics.keys()
	assert np.allclose(state['montgomery_potential'],
					   diagnostics['montgomery_potential'])
	diagnostics.pop('montgomery_potential')

	assert 'height_on_interface_levels' in diagnostics.keys()
	assert np.allclose(state['height_on_interface_levels'],
					   diagnostics['height_on_interface_levels'])
	diagnostics.pop('height_on_interface_levels')

	assert diagnostics == {}


def test_moist(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diag = IsentropicDiagnostics(grid, True,
								 state['air_pressure_on_interface_levels'][0, 0, 0],
								 backend=backend, dtype=dtype)

	diagnostics = diag(state)

	assert 'air_pressure_on_interface_levels' in diagnostics.keys()
	assert np.allclose(state['air_pressure_on_interface_levels'],
					   diagnostics['air_pressure_on_interface_levels'])
	diagnostics.pop('air_pressure_on_interface_levels')

	assert 'exner_function_on_interface_levels' in diagnostics.keys()
	assert np.allclose(state['exner_function_on_interface_levels'],
					   diagnostics['exner_function_on_interface_levels'])
	diagnostics.pop('exner_function_on_interface_levels')

	assert 'montgomery_potential' in diagnostics.keys()
	assert np.allclose(state['montgomery_potential'],
					   diagnostics['montgomery_potential'])
	diagnostics.pop('montgomery_potential')

	assert 'height_on_interface_levels' in diagnostics.keys()
	assert np.allclose(state['height_on_interface_levels'],
					   diagnostics['height_on_interface_levels'])
	diagnostics.pop('height_on_interface_levels')

	assert 'air_density' in diagnostics.keys()
	assert np.allclose(state['air_density'],
					   diagnostics['air_density'])
	diagnostics.pop('air_density')

	assert 'air_temperature' in diagnostics.keys()
	assert np.allclose(state['air_temperature'],
					   diagnostics['air_temperature'])
	diagnostics.pop('air_temperature')

	assert diagnostics == {}


def test_velocity_components(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float32

	diag = IsentropicVelocityComponents(grid, 'relaxed', states[0],
					   				    backend=backend, dtype=dtype)

	diagnostics = diag(state)

	assert 'x_velocity_at_u_locations' in diagnostics.keys()
	assert np.allclose(state['x_velocity_at_u_locations'],
					   diagnostics['x_velocity_at_u_locations'])

	assert 'y_velocity_at_v_locations' in diagnostics.keys()
	assert np.allclose(state['y_velocity_at_v_locations'],
					   diagnostics['y_velocity_at_v_locations'])


if __name__ == '__main__':
	pytest.main([__file__])
