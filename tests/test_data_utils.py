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
from hypothesis import \
	assume, given, HealthCheck, reproduce_failure, settings, strategies as hyp_st
import numpy as np
import pytest
from sympl import DataArray

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

from tasmania.python.utils import data_utils as du
from tasmania.python.utils.exceptions import ConstantNotFoundError


def test_get_constant():
	u = du.get_constant('gravitational_acceleration', 'm s^-2')
	assert u == 9.80665

	v = du.get_constant('foo', 'm', default_value=DataArray(10., attrs={'units': 'km'}))
	assert v == 10000.0

	w = du.get_constant('pippo', '1', default_value=DataArray(10., attrs={'units': '1'}))
	assert w == 10.

	try:
		_ = du.get_constant('foo', 'K')
	except ValueError:
		assert True

	try:
		_ = du.get_constant('bar', 'K')
	except ConstantNotFoundError:
		assert True


def test_get_physical_constants():
	d_physical_constants = {
		'gravitational_acceleration': DataArray(9.80665e-3, attrs={'units': 'km s^-2'}),
		'gas_constant_of_dry_air': DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
		'gas_constant_of_water_vapor': DataArray(461.52, attrs={'units': 'hJ K^-1 g^-1'}),
		'latent_heat_of_vaporization_of_water':	DataArray(2.5e6, attrs={'units': 'J kg^-1'}),
		'foo_constant': DataArray(1, attrs={'units': '1'}),
	}

	physical_constants = {
		'latent_heat_of_vaporization_of_water': DataArray(1.5e3, attrs={'units': 'kJ kg^-1'}),
	}

	raw_constants = du.get_physical_constants(d_physical_constants, physical_constants)

	assert raw_constants['gravitational_acceleration'] == 9.80665e-3
	assert raw_constants['gas_constant_of_dry_air'] == 287.0
	assert raw_constants['gas_constant_of_water_vapor'] == 461.5e-5
	assert raw_constants['latent_heat_of_vaporization_of_water'] == 1.5e6
	assert raw_constants['foo_constant'] == 1.0


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_make_dataarray_2d(data):
	grid = data.draw(utils.st_physical_horizontal_grid())
	dtype = grid.x.dtype

	#
	# nx, ny
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (grid.nx, grid.ny), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_2d(raw_array, grid, units, name)

	assert array.shape == (grid.nx, grid.ny)
	assert array.dims == (grid.x.dims[0], grid.y.dims[0])
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx+1, ny
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (grid.nx+1, grid.ny), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_2d(raw_array, grid, units, name)

	assert array.shape == (grid.nx+1, grid.ny)
	assert array.dims == (grid.x_at_u_locations.dims[0], grid.y.dims[0])
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx, ny+1
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (grid.nx, grid.ny+1), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_2d(raw_array, grid, units, name)

	assert array.shape == (grid.nx, grid.ny+1)
	assert array.dims == (grid.x.dims[0], grid.y_at_v_locations.dims[0])
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx+1, ny+1
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (grid.nx+1, grid.ny+1), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_2d(raw_array, grid, units, name)

	assert array.shape == (grid.nx+1, grid.ny+1)
	assert array.dims == (grid.x_at_u_locations.dims[0], grid.y_at_v_locations.dims[0])
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_make_dataarray_3d(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(utils.st_physical_grid())

	nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
	dtype = grid.z.dtype

	#
	# nx, ny, nz
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx, ny, nz), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx, ny, nz)
	assert array.dims == (
		grid.grid_xy.x.dims[0], grid.grid_xy.y.dims[0], grid.z.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx+1, ny, nz
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx+1, ny, nz), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx+1, ny, nz)
	assert array.dims == (
		grid.grid_xy.x_at_u_locations.dims[0], grid.grid_xy.y.dims[0], grid.z.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx, ny+1, nz
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx, ny+1, nz), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx, ny+1, nz)
	assert array.dims == (
		grid.grid_xy.x.dims[0], grid.grid_xy.y_at_v_locations.dims[0], grid.z.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx+1, ny+1, nz
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx+1, ny+1, nz), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx+1, ny+1, nz)
	assert array.dims == (
		grid.grid_xy.x_at_u_locations.dims[0],
		grid.grid_xy.y_at_v_locations.dims[0],
		grid.z.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx, ny, nz+1
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx, ny, nz+1), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx, ny, nz+1)
	assert array.dims == (
		grid.grid_xy.x.dims[0], grid.grid_xy.y.dims[0],
		grid.z_on_interface_levels.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx+1, ny, nz+1
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx+1, ny, nz+1), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx+1, ny, nz+1)
	assert array.dims == (
		grid.grid_xy.x_at_u_locations.dims[0],
		grid.grid_xy.y.dims[0],
		grid.z_on_interface_levels.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx, ny+1, nz+1
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx, ny+1, nz+1), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx, ny+1, nz+1)
	assert array.dims == (
		grid.grid_xy.x.dims[0],
		grid.grid_xy.y_at_v_locations.dims[0],
		grid.z_on_interface_levels.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx+1, ny+1, nz+1
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx+1, ny+1, nz+1), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx+1, ny+1, nz+1)
	assert array.dims == (
		grid.grid_xy.x_at_u_locations.dims[0],
		grid.grid_xy.y_at_v_locations.dims[0],
		grid.z_on_interface_levels.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)

	#
	# nx+1, ny+1, 1
	#
	raw_array = data.draw(
		utils.st_raw_field(dtype, (nx+1, ny+1, 1), min_value=-1e5, max_value=1e5)
	)
	units = data.draw(hyp_st.text(max_size=10))
	name = data.draw(hyp_st.text(max_size=10))

	array = du.make_dataarray_3d(raw_array, grid, units, name)

	assert array.shape == (nx+1, ny+1, 1)
	assert array.dims == (
		grid.grid_xy.x_at_u_locations.dims[0],
		grid.grid_xy.y_at_v_locations.dims[0],
		grid.z.dims[0] + '_at_surface_level' if nz > 1 else grid.z.dims[0]
	)
	assert array.attrs['units'] == units
	assert array.name == name
	assert np.allclose(raw_array, array.values)
	assert id(raw_array) == id(array.values)


if __name__ == '__main__':
	pytest.main([__file__])
