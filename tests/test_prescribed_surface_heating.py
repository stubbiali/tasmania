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
import pytest
from sympl import DataArray

from tasmania.python.isentropic.physics.tendencies import PrescribedSurfaceHeating
from tasmania.python.utils.data_utils import make_dataarray_3d
from tasmania.python.utils.utils import equal_to


def test(isentropic_dry_data):
	grid, states = isentropic_dry_data
	state = deepcopy(states[-1])
	grid.update_topography(state['time'] - states[0]['time'])

	amplitude_at_day = DataArray(0.8, attrs={'units': 'kW m^-2'})
	amplitude_at_night = DataArray(-75000.0, attrs={'units': 'mW m^-2'})
	attenuation_coefficient_at_day = DataArray(1.0/6.0, attrs={'units': 'hm^-1'})
	attenuation_coefficient_at_night = DataArray(1.0/75.0, attrs={'units': 'm^-1'})
	characteristic_length = DataArray(25.0, attrs={'units': 'km'})

	#
	# tendency_of_air_potential_temperature_in_diagnostics=False
	# tendency_of_air_potential_temperature_on_interface_levels=True
	#
	psh = PrescribedSurfaceHeating(
		grid, tendency_of_air_potential_temperature_in_diagnostics=False,
		tendency_of_air_potential_temperature_on_interface_levels=True,
		air_pressure_on_interface_levels=True
	)

	assert 'air_pressure_on_interface_levels' in psh.input_properties
	assert 'height_on_interface_levels' in psh.input_properties
	assert len(psh.input_properties) == 2

	assert 'air_potential_temperature_on_interface_levels' in psh.tendency_properties
	assert len(psh.tendency_properties) == 1

	assert psh.diagnostic_properties == {}

	#
	# tendency_of_air_potential_temperature_in_diagnostics=True
	# tendency_of_air_potential_temperature_on_interface_levels=True
	#
	psh = PrescribedSurfaceHeating(
		grid, tendency_of_air_potential_temperature_in_diagnostics=True,
		tendency_of_air_potential_temperature_on_interface_levels=True,
		air_pressure_on_interface_levels=False
	)

	assert 'air_pressure' in psh.input_properties
	assert 'height_on_interface_levels' in psh.input_properties
	assert len(psh.input_properties) == 2

	assert psh.tendency_properties == {}

	assert 'tendency_of_air_potential_temperature' in psh.diagnostic_properties
	assert len(psh.diagnostic_properties) == 1

	#
	# air_pressure_on_interface_levels=True
	# tendency_of_air_potential_temperature_on_interface_levels=False
	#
	state['time'] = datetime(year=1992, month=2, day=20, hour=15)
	starting_time = state['time'] - timedelta(hours=2)

	psh = PrescribedSurfaceHeating(
		grid, tendency_of_air_potential_temperature_in_diagnostics=True,
		air_pressure_on_interface_levels=True,
		amplitude_at_day_sw=amplitude_at_day,
		amplitude_at_night_sw=amplitude_at_night,
		attenuation_coefficient_at_day=attenuation_coefficient_at_day,
		attenuation_coefficient_at_night=attenuation_coefficient_at_night,
		characteristic_length=characteristic_length,
		starting_time=starting_time,
	)

	assert equal_to(psh._f0d_sw, 800.0)
	assert equal_to(psh._f0d_fw, 400.0)
	assert equal_to(psh._f0n_sw, -75.0)
	assert equal_to(psh._f0n_fw, -37.5)
	assert equal_to(psh._ad, 1.0/600.0)
	assert equal_to(psh._an, 1.0/75.0)
	assert equal_to(psh._cl, 25000.0)

	tendencies, diagnostics = psh(state)

	assert tendencies == {}

	assert 'tendency_of_air_potential_temperature' in diagnostics
	assert len(diagnostics) == 1

	#
	# air_pressure_on_interface_levels=False
	# tendency_of_air_potential_temperature_on_interface_levels=False
	#
	state['time'] = datetime(year=1992, month=2, day=20, hour=3)
	starting_time = state['time'] - timedelta(hours=2)
	p = state['air_pressure_on_interface_levels'].values
	state['air_pressure'] = make_dataarray_3d(0.5 * (p[:, :, :-1] + p[:, :, 1:]), grid, 'Pa')
	state.pop('air_pressure_on_interface_levels')

	psh = PrescribedSurfaceHeating(
		grid, tendency_of_air_potential_temperature_in_diagnostics=False,
		air_pressure_on_interface_levels=False,
		amplitude_at_day_fw=amplitude_at_day,
		amplitude_at_night_fw=amplitude_at_night,
		attenuation_coefficient_at_day=attenuation_coefficient_at_day,
		attenuation_coefficient_at_night=attenuation_coefficient_at_night,
		characteristic_length=characteristic_length,
		starting_time=starting_time,
	)

	assert equal_to(psh._f0d_sw, 800.0)
	assert equal_to(psh._f0d_fw, 800.0)
	assert equal_to(psh._f0n_sw, -75.0)
	assert equal_to(psh._f0n_fw, -75.0)
	assert equal_to(psh._ad, 1.0/600.0)
	assert equal_to(psh._an, 1.0/75.0)
	assert equal_to(psh._cl, 25000.0)

	tendencies, diagnostics = psh(state)

	assert 'air_potential_temperature' in tendencies
	assert len(tendencies) == 1

	assert len(diagnostics) == 0


if __name__ == '__main__':
	pytest.main([__file__])
