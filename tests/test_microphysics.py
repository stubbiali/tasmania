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
import os
import pickle
import pytest
from sympl import DataArray

from tasmania.utils.data_utils import make_dataarray_3d


def make_grid(filename):
	with open(filename, 'rb') as data:
		grid_ = pickle.load(data)

	domain_x = DataArray([grid_.x.values[0], grid_.x.values[-1]], dims=grid_.x.dims,
						 attrs={'units': grid_.x.attrs['units']})
	nx = grid_.nx
	domain_y = DataArray([grid_.y.values[0], grid_.y.values[-1]], dims=grid_.y.dims,
						 attrs={'units': grid_.y.attrs['units']})
	ny = grid_.ny
	domain_z = DataArray([grid_.z.values[0], grid_.z.values[-1]], dims=grid_.z.dims,
						 attrs={'units': grid_.z.attrs['units']})
	nz = grid_.nz

	from tasmania.grids.grid_xyz import GridXYZ as Grid
	from datetime import timedelta
	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
			 	topo_type='gaussian', topo_time=timedelta(seconds=0),
			 	topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							 'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


@pytest.fixture
def grid_and_state():
	filename = os.path.join(os.environ['TASMANIA_ROOT'],
							'tests/baseline_datasets/verification_moist.pickle')

	grid = make_grid(filename)

	with open(filename, 'rb') as data:
		_ = pickle.load(data)
		states = pickle.load(data)

	state_ = states[10]
	state = {
		'time': state_['time'],
		'air_density':
			make_dataarray_3d(state_['air_density'].values,
							  grid, 'kg m^-3'),
		'air_temperature':
			make_dataarray_3d(state_['air_temperature'].values,
							  grid, 'K'),
		'air_pressure_on_interface_levels':
			make_dataarray_3d(state_['air_pressure_on_interface_levels'].values,
							  grid, 'Pa'),
		'exner_function_on_interface_levels':
			make_dataarray_3d(state_['exner_function_on_interface_levels'].values,
							  grid, 'J K^-1 kg^-1'),
		'mass_fraction_of_water_vapor_in_air':
			make_dataarray_3d(state_['mass_fraction_of_water_vapor_in_air'].values,
							  grid, 'g g^-1'),
		'mass_fraction_of_cloud_liquid_water_in_air':
			make_dataarray_3d(1e3 * state_['mass_fraction_of_cloud_liquid_water_in_air'].values,
							  grid, 'g kg^-1'),
		'mass_fraction_of_precipitation_water_in_air':
			make_dataarray_3d(state_['mass_fraction_of_precipitation_water_in_air'].values,
							  grid, 'kg kg^-1'),
	}

	return grid, state


def test_kessler(grid_and_state):
	from tasmania.physics.microphysics import Kessler
	grid, state = grid_and_state

	kp = Kessler(grid)

	assert kp._a == 0.5e-3
	assert kp._k1 == 0.001
	assert kp._k2 == 2.2

	assert hasattr(kp, 'input_properties')
	assert hasattr(kp, 'tendency_properties')
	assert hasattr(kp, 'diagnostic_properties')

	tendencies, diagnostics = kp(state)

	tnd_name_1 = 'tendency_of_air_potential_temperature'
	tnd_name_2 = 'tendency_of_mass_fraction_of_water_vapor_in_air'
	tnd_name_3 = 'tendency_of_mass_fraction_of_cloud_liquid_water_in_air'
	tnd_name_4 = 'tendency_of_mass_fraction_of_precipitation_water_in_air'

	assert tnd_name_1 in tendencies.keys()
	assert tnd_name_2 in tendencies.keys()
	assert tnd_name_3 in tendencies.keys()
	assert tnd_name_4 in tendencies.keys()

	assert tendencies[tnd_name_1].attrs['units'] == 'K s^-1'
	assert tendencies[tnd_name_2].attrs['units'] == 'g g^-1 s^-1'
	assert tendencies[tnd_name_3].attrs['units'] == 'g g^-1 s^-1'
	assert tendencies[tnd_name_4].attrs['units'] == 'g g^-1 s^-1'

	dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	assert tendencies[tnd_name_1].dims == dims
	assert tendencies[tnd_name_2].dims == dims
	assert tendencies[tnd_name_3].dims == dims
	assert tendencies[tnd_name_4].dims == dims

	assert diagnostics == {}


def test_saturation_adjustment_kessler(grid_and_state):
	from tasmania.physics.microphysics import SaturationAdjustmentKessler
	grid, state = grid_and_state

	sakp = SaturationAdjustmentKessler(grid)

	assert hasattr(sakp, 'input_properties')
	assert hasattr(sakp, 'diagnostic_properties')

	diagnostics = sakp(state)

	diag_name_1 = 'mass_fraction_of_water_vapor_in_air'
	diag_name_2 = 'mass_fraction_of_cloud_liquid_water_in_air'

	assert diag_name_1 in diagnostics.keys()
	assert diag_name_2 in diagnostics.keys()

	assert diagnostics[diag_name_1].attrs['units'] == 'g g^-1'
	assert diagnostics[diag_name_2].attrs['units'] == 'g g^-1'

	dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	assert diagnostics[diag_name_1].dims == dims
	assert diagnostics[diag_name_2].dims == dims


def test_raindrop_fall_velocity(grid_and_state):
	from tasmania.physics.microphysics import RaindropFallVelocity
	grid, state = grid_and_state

	rfvp = RaindropFallVelocity(grid)

	assert hasattr(rfvp, 'input_properties')
	assert hasattr(rfvp, 'diagnostic_properties')

	diagnostics = rfvp(state)

	diag_name = 'raindrop_fall_velocity'

	assert diag_name in diagnostics.keys()

	assert diagnostics[diag_name].attrs['units'] == 'm s^-1'

	dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
	assert diagnostics[diag_name].dims == dims


if __name__ == '__main__':
	pytest.main([__file__])
