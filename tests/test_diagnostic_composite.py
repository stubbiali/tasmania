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
import copy
import numpy as np
import pytest

import gridtools as gt
from tasmania.python.core.physics_composite import TasmaniaDiagnosticComponentComposite
from tasmania.python.isentropic.physics.diagnostics import IsentropicDiagnostics
from tasmania.python.physics.microphysics import SaturationAdjustmentKessler


def test_serial(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	state_dc = copy.deepcopy(state)
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float64

	dv = IsentropicDiagnostics(
		grid, True, pt=state['air_pressure_on_interface_levels'][0, 0, 0],
		backend=backend, dtype=dtype
	)
	sa = SaturationAdjustmentKessler(grid, backend=backend)

	dcc = TasmaniaDiagnosticComponentComposite(
		dv, sa, execution_policy='serial'
	)

	assert 'air_isentropic_density' in dcc.input_properties
	assert 'mass_fraction_of_water_vapor_in_air' in dcc.input_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in dcc.input_properties
	assert len(dcc.input_properties) == 3

	assert 'air_density' in dcc.diagnostic_properties
	assert 'air_pressure_on_interface_levels' in dcc.diagnostic_properties
	assert 'air_temperature' in dcc.diagnostic_properties
	assert 'exner_function_on_interface_levels' in dcc.diagnostic_properties
	assert 'height_on_interface_levels' in dcc.diagnostic_properties
	assert 'mass_fraction_of_water_vapor_in_air' in dcc.diagnostic_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in dcc.diagnostic_properties
	assert 'montgomery_potential' in dcc.diagnostic_properties
	assert len(dcc.diagnostic_properties) == 8

	diagnostics = dcc(state)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	assert len(state) == len(state_dc)

	assert 'air_density' in diagnostics
	assert 'air_pressure_on_interface_levels' in diagnostics
	assert 'air_temperature' in diagnostics
	assert 'exner_function_on_interface_levels' in diagnostics
	assert 'height_on_interface_levels' in diagnostics
	assert 'mass_fraction_of_water_vapor_in_air' in diagnostics
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in diagnostics
	assert 'montgomery_potential' in diagnostics
	assert len(diagnostics) == 8

	assert np.allclose(
		diagnostics['air_density'],
		state['air_density']
	)
	assert np.allclose(
		diagnostics['air_pressure_on_interface_levels'],
		state['air_pressure_on_interface_levels']
	)
	assert np.allclose(
		diagnostics['air_temperature'],
		state['air_temperature']
	)
	assert np.allclose(
		diagnostics['exner_function_on_interface_levels'],
		state['exner_function_on_interface_levels']
	)
	assert np.allclose(
		diagnostics['height_on_interface_levels'],
		state['height_on_interface_levels']
	)
	assert np.allclose(
		diagnostics['montgomery_potential'],
		state['montgomery_potential']
	)


def test_asparallel(isentropic_moist_data):
	grid, states = isentropic_moist_data
	state = states[-1]
	state_dc = copy.deepcopy(state)
	grid.update_topography(state['time'] - states[0]['time'])

	backend = gt.mode.NUMPY
	dtype = np.float64

	dv = IsentropicDiagnostics(
		grid, True, pt=state['air_pressure_on_interface_levels'][0, 0, 0],
		backend=backend, dtype=dtype
	)
	sa = SaturationAdjustmentKessler(grid, backend=backend)

	dcc = TasmaniaDiagnosticComponentComposite(
		dv, sa, execution_policy='as_parallel'
	)

	assert 'air_isentropic_density' in dcc.input_properties
	assert 'air_pressure_on_interface_levels' in dcc.input_properties
	assert 'air_temperature' in dcc.input_properties
	assert 'mass_fraction_of_water_vapor_in_air' in dcc.input_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in dcc.input_properties
	assert len(dcc.input_properties) == 5

	assert 'air_density' in dcc.diagnostic_properties
	assert 'air_pressure_on_interface_levels' in dcc.diagnostic_properties
	assert 'air_temperature' in dcc.diagnostic_properties
	assert 'exner_function_on_interface_levels' in dcc.diagnostic_properties
	assert 'height_on_interface_levels' in dcc.diagnostic_properties
	assert 'mass_fraction_of_water_vapor_in_air' in dcc.diagnostic_properties
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in dcc.diagnostic_properties
	assert 'montgomery_potential' in dcc.diagnostic_properties
	assert len(dcc.diagnostic_properties) == 8

	diagnostics = dcc(state)

	for key in state:
		if key == 'time':
			assert state['time'] == state_dc['time']
		else:
			assert np.allclose(state[key], state_dc[key])

	assert 'air_density' in diagnostics
	assert 'air_pressure_on_interface_levels' in diagnostics
	assert 'air_temperature' in diagnostics
	assert 'exner_function_on_interface_levels' in diagnostics
	assert 'height_on_interface_levels' in diagnostics
	assert 'mass_fraction_of_water_vapor_in_air' in diagnostics
	assert 'mass_fraction_of_cloud_liquid_water_in_air' in diagnostics
	assert 'montgomery_potential' in diagnostics
	assert len(diagnostics) == 8

	assert np.allclose(
		diagnostics['air_density'],
		state['air_density']
	)
	assert np.allclose(
		diagnostics['air_pressure_on_interface_levels'],
		state['air_pressure_on_interface_levels']
	)
	assert np.allclose(
		diagnostics['air_temperature'],
		state['air_temperature']
	)
	assert np.allclose(
		diagnostics['exner_function_on_interface_levels'],
		state['exner_function_on_interface_levels']
	)
	assert np.allclose(
		diagnostics['height_on_interface_levels'],
		state['height_on_interface_levels']
	)
	assert np.allclose(
		diagnostics['montgomery_potential'],
		state['montgomery_potential']
	)


if __name__ == '__main__':
	pytest.main([__file__])
	#from conftest import isentropic_dry_data, isentropic_moist_data
	#test_ps_asparallel_moist(isentropic_moist_data())
