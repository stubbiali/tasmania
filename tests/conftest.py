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
from sympl import DataArray, TendencyComponent

from tasmania.python.grids.grid import PhysicalGrid
from tasmania.python.plot.contour import Contour
from tasmania.python.plot.profile import LineProfile
from tasmania.python.utils.storage_utils import load_netcdf_dataset


@pytest.fixture(scope='module')
def grid():
	domain_x = DataArray([0., 500.], dims='x', attrs={'units': 'km'})
	nx       = 101
	domain_y = DataArray([-50., 50.], dims='y', attrs={'units': 'km'})
	ny       = 51
	domain_z = DataArray(
		[400., 300.], dims='air_potential_temperature', attrs={'units': 'K'}
	)
	nz       = 50

	return PhysicalGrid(
		domain_x, nx, domain_y, ny, domain_z, nz,
		topography_type='gaussian',
		topography_kwargs={
			'time': timedelta(seconds=0),
			'max_height': DataArray(1000.0, attrs={'units': 'm'}),
			'width_x': DataArray(25.0, attrs={'units': 'km'})
		}
	)


@pytest.fixture(scope='module')
def grid_xz():
	domain_x = DataArray([0., 500.], dims='x', attrs={'units': 'km'})
	nx       = 101
	domain_y = DataArray([-50., 50.], dims='y', attrs={'units': 'km'})
	ny       = 1
	domain_z = DataArray(
		[400., 300.], dims='air_potential_temperature', attrs={'units': 'K'}
	)
	nz       = 50

	return PhysicalGrid(
		domain_x, nx, domain_y, ny, domain_z, nz,
		topography_type='gaussian',
		topography_kwargs={
			'time': timedelta(seconds=0),
			'max_height': DataArray(1000.0, attrs={'units': 'm'}),
			'width_x': DataArray(25.0, attrs={'units': 'km'})
		}
	)


@pytest.fixture(scope='module')
def grid_yz():
	domain_x = DataArray([0., 500.], dims='x', attrs={'units': 'km'})
	nx       = 1
	domain_y = DataArray([-50., 50.], dims='y', attrs={'units': 'km'})
	ny       = 51
	domain_z = DataArray(
		[400., 300.], dims='air_potential_temperature', attrs={'units': 'K'}
	)
	nz       = 50

	return PhysicalGrid(
		domain_x, nx, domain_y, ny, domain_z, nz,
		topography_type='gaussian',
		topography_kwargs={
			'time': timedelta(seconds=0),
			'max_height': DataArray(1000.0, attrs={'units': 'm'}),
			'width_x': DataArray(25.0, attrs={'units': 'km'})
		}
	)


@pytest.fixture(scope='module')
def isentropic_dry_data():
	return load_netcdf_dataset('baseline_datasets/isentropic_dry.nc')


@pytest.fixture(scope='module')
def isentropic_moist_data():
	return load_netcdf_dataset('baseline_datasets/isentropic_moist.nc')


@pytest.fixture(scope='module')
def isentropic_moist_sedimentation_data():
	return load_netcdf_dataset('baseline_datasets/isentropic_moist_sedimentation.nc')


@pytest.fixture(scope='module')
def isentropic_moist_sedimentation_evaporation_data():
	return load_netcdf_dataset('baseline_datasets/isentropic_moist_sedimentation_evaporation.nc')


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


@pytest.fixture(scope='module')
def drawer_topography1d():
	def _drawer_topography1d(grid, topography_units='m', x=None, y=None,
							 axis_name=None, axis_units=None):
		properties = {'linecolor': 'black', 'linewidth': 1.3}
		return LineProfile(grid, 'topography', topography_units, x=x, y=y, z=-1,
						   axis_name=axis_name, axis_units=axis_units, properties=properties)

	return _drawer_topography1d


@pytest.fixture(scope='module')
def drawer_topography2d():
	def _drawer_topography2d(grid, topography_units='m', xaxis_name=None, xaxis_units=None,
							 yaxis_name=None, yaxis_units=None):
		properties = {'colors': 'darkgray', 'draw_vertical_levels': False}
		return Contour(grid, 'topography', topography_units, z=-1,
					   xaxis_name=xaxis_name, xaxis_units=xaxis_units,
					   yaxis_name=yaxis_name, yaxis_units=yaxis_units,
					   properties=properties)

	return _drawer_topography2d


class FakeTendency1(TendencyComponent):
	def __init__(self, grid):
		self._grid = grid
		super().__init__()

	@property
	def input_properties(self):
		g = self._grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'}
		}

		return return_dict

	@property
	def tendency_properties(self):
		g = self._grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		g = self._grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'fake_variable': {'dims': dims, 'units': 'm'},
		}

		return return_dict

	def array_call(self, state):
		s = state['air_isentropic_density']

		tendencies = {
			'air_isentropic_density': s**2,
			'x_momentum_isentropic': s**3,
		}

		diagnostics = {
			'fake_variable': 2*s,
		}

		return tendencies, diagnostics


@pytest.fixture(scope='module')
def make_fake_tendency_1():
	def _make_fake_tendency_1(grid):
		return FakeTendency1(grid)

	return _make_fake_tendency_1


class FakeTendency2(TendencyComponent):
	def __init__(self, grid):
		self._grid = grid
		super().__init__()

	@property
	def input_properties(self):
		g = self._grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

		return_dict = {
			'fake_variable': {'dims': dims, 'units': 'km'},
			'y_velocity_at_v_locations': {'dims': dims_y, 'units': 'km hr^-1'},
		}

		return return_dict

	@property
	def tendency_properties(self):
		g = self._grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		return return_dict

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		f = state['fake_variable']
		v = state['y_velocity_at_v_locations']

		tendencies = {
			'air_isentropic_density': f/100,
			'y_momentum_isentropic': 0.5*(v[:, :-1, :] + v[:, 1:, :]),
		}

		diagnostics = {}

		return tendencies, diagnostics


@pytest.fixture(scope='module')
def make_fake_tendency_2():
	def _make_fake_tendency_2(grid):
		return FakeTendency2(grid)

	return _make_fake_tendency_2
