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
import pytest
import xarray as xr

from tasmania.utils.storage_utils import load_netcdf_dataset


def test_load_netcdf_dataset():
	filename = 'baseline_datasets/isentropic_dry.nc'
	grid, states = load_netcdf_dataset(filename)

	assert grid.nx == 51
	assert grid.dx.values.item() == 10e3
	assert grid.x.values[0] == 0
	assert grid.x.values[-1] == 500e3

	assert grid.ny == 51
	assert grid.dy.values.item() == 10e3
	assert grid.y.values[0] == -250e3
	assert grid.y.values[-1] == 250e3

	assert grid.nz == 50
	assert grid.dz.values.item() == 2.0
	assert grid.z_on_interface_levels.values[0] == 400.0
	assert grid.z_on_interface_levels.values[-1] == 300.0

	topo = grid.topography
	assert topo.topo_type == 'gaussian'
	assert topo.topo_time.total_seconds() == 1800.0
	assert topo.topo_kwargs['topo_max_height'] == 1000.0
	assert topo.topo_kwargs['topo_width_x'] == 50e3
	assert topo.topo_kwargs['topo_width_y'] == 50e3
	assert topo.topo_kwargs['topo_center_x'] == 250e3
	assert topo.topo_kwargs['topo_center_y'] == 0.0

	assert len(states) == 61
	assert all(['time' in state.keys() for state in states])
	assert all(['air_isentropic_density' in state.keys() for state in states])
	assert all(['air_pressure_on_interface_levels' in state.keys() for state in states])
	assert all(['exner_function_on_interface_levels' in state.keys() for state in states])
	assert all(['height_on_interface_levels' in state.keys() for state in states])
	assert all(['montgomery_potential' in state.keys() for state in states])
	assert all(['x_momentum_isentropic' in state.keys() for state in states])
	assert all(['x_velocity_at_u_locations' in state.keys() for state in states])
	assert all(['y_momentum_isentropic' in state.keys() for state in states])
	assert all(['y_velocity_at_v_locations' in state.keys() for state in states])


if __name__ == '__main__':
	pytest.main([__file__])
	#test_load_netcdf_dataset()

