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
import os
import pickle
import pytest
from sympl import DataArray

from tasmania.grids.grid_xyz import GridXYZ as Grid
from tasmania.grids.grid_xz import GridXZ
from tasmania.utils.data_utils import make_data_array_3d


@pytest.fixture(scope='module')
def grid():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 101
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 51
	domain_z, nz = DataArray([400., 300.], dims='air_potential_temperature',
							 attrs={'units': 'K'}), 50

	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				topo_type='gaussian', topo_time=timedelta(seconds=0),
				topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							 'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


@pytest.fixture(scope='module')
def grid_xz():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 101
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 1
	domain_z, nz = DataArray([400., 300.], dims='air_potential_temperature',
							 attrs={'units': 'K'}), 50

	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				topo_type='gaussian', topo_time=timedelta(seconds=0),
				topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							 'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


@pytest.fixture(scope='module')
def grid_xz_2d():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 101
	domain_z, nz = DataArray([400., 300.], dims='air_potential_temperature',
							 attrs={'units': 'K'}), 50

	return GridXZ(domain_x, nx, domain_z, nz,
				  topo_type='gaussian', topo_time=timedelta(seconds=0),
				  topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							   'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


@pytest.fixture(scope='module')
def grid_yz():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 1
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 51
	domain_z, nz = DataArray([400., 300.], dims='air_potential_temperature',
							 attrs={'units': 'K'}), 50

	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				topo_type='gaussian', topo_time=timedelta(seconds=0),
				topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							 'topo_width_x': DataArray(25.0, attrs={'units': 'km'})})


def make_grid(filename):
	with open(filename, 'rb') as data:
		grid_ = pickle.load(data)

	domain_x = DataArray([grid_.x.values[0], grid_.x.values[-1]], dims=grid_.x.dims,
						 attrs={'units': grid_.x.attrs['units']})
	nx = grid_.nx
	domain_y = DataArray([grid_.y.values[0], grid_.y.values[-1]], dims=grid_.y.dims,
						 attrs={'units': grid_.y.attrs['units']})
	ny = grid_.ny
	domain_z = DataArray([grid_.z_half_levels.values[0],
						  grid_.z_half_levels.values[-1]], dims=grid_.z.dims,
						 attrs={'units': grid_.z.attrs['units']})
	nz = grid_.nz

	return Grid(domain_x, nx, domain_y, ny, domain_z, nz,
				topo_type='gaussian', topo_time=timedelta(seconds=0),
				topo_kwargs={'topo_max_height': DataArray(1000.0, attrs={'units': 'm'}),
							 'topo_width_x': DataArray(25.0, attrs={'units': 'km'})},
				dtype=domain_x.values.dtype)


@pytest.fixture
def grid_and_state():
	#filename = os.path.join(os.environ['TASMANIA_ROOT'],
	#						'tests/baseline_datasets/verification_moist.pickle')
	filename = 'baseline_datasets/verification_moist.pickle'

	g = make_grid(filename)

	with open(filename, 'rb') as data:
		_ = pickle.load(data)
		states = pickle.load(data)

	state_ = states[-1]
	state = {
		'time': state_['time'],
		'air_density':
			make_data_array_3d(state_['air_density'].values,
							  g, 'kg m^-3'),
		'air_isentropic_density':
			make_data_array_3d(state_['air_isentropic_density'].values,
							  g, 'kg m^-2 K^-1'),
		'air_pressure_on_interface_levels':
			make_data_array_3d(state_['air_pressure_on_interface_levels'].values,
							  g, 'Pa'),
		'air_temperature':
			make_data_array_3d(state_['air_temperature'].values,
							  g, 'K'),
		'exner_function_on_interface_levels':
			make_data_array_3d(state_['exner_function_on_interface_levels'].values,
							  g, 'J K^-1 kg^-1'),
		'height_on_interface_levels':
			make_data_array_3d(state_['height_on_interface_levels'].values,
							  g, 'm'),
		'mass_fraction_of_water_vapor_in_air':
			make_data_array_3d(state_['mass_fraction_of_water_vapor_in_air'].values,
							  g, 'g g^-1'),
		'mass_fraction_of_cloud_liquid_water_in_air':
			make_data_array_3d(1e3 * state_['mass_fraction_of_cloud_liquid_water_in_air'].values,
							  g, 'g kg^-1'),
		'mass_fraction_of_precipitation_water_in_air':
			make_data_array_3d(state_['mass_fraction_of_precipitation_water_in_air'].values,
							  g, 'kg kg^-1'),
		'montgomery_potential':
			make_data_array_3d(state_['montgomery_potential'].values,
							  g, 'J kg^-1'),
		'x_momentum_isentropic':
			make_data_array_3d(state_['x_momentum_isentropic'].values,
							  g, 'kg m^-1 K^-1 s^-1'),
		'x_velocity_at_u_locations':
			make_data_array_3d(state_['x_velocity_at_u_locations'].values,
							  g, 'm s^-1'),
		'y_momentum_isentropic':
			make_data_array_3d(state_['y_momentum_isentropic'].values,
							  g, 'kg m^-1 K^-1 s^-1'),
		'y_velocity_at_v_locations':
			make_data_array_3d(state_['y_velocity_at_v_locations'].values,
							  g, 'm s^-1'),
	}

	return g, state


@pytest.fixture(scope='module')
def physical_constants():
	return {
		'air_pressure_at_sea_level':
			DataArray(1e5, attrs={'units': 'Pa'}),
		'gas_constant_of_dry_air':
			DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gravitational_acceleration':
			DataArray(9.81, attrs={'units': 'm s^-2'}),
		'specific_heat_of_dry_air_at_constant_pressure':
			DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
	}
