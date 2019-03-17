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
import pytest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

from tasmania.python.grids.domain import Domain
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_domain(data):
	# ========================================
	# random data generation
	# ========================================
	nx = data.draw(utils.st_length(axis_name='x'), label='nx')
	ny = data.draw(utils.st_length(axis_name='y'), label='ny')
	nz = data.draw(utils.st_length(axis_name='z'), label='nz')

	assume(not(nx == 1 and ny == 1))

	domain_x = data.draw(utils.st_interval(axis_name='x'), label='x')
	domain_y = data.draw(utils.st_interval(axis_name='y'), label='y')
	domain_z = data.draw(utils.st_interval(axis_name='z'), label='z')

	zi = data.draw(utils.st_interface(domain_z), label='zi')

	hb_type = data.draw(utils.st_horizontal_boundary_type())
	nb = data.draw(utils.st_horizontal_boundary_layers(nx, ny))
	hb_kwargs = data.draw(utils.st_horizontal_boundary_kwargs(hb_type, nx, ny, nb))

	topo_kwargs = data.draw(utils.st_topography_kwargs(domain_x, domain_y), label='kwargs')
	topo_type = topo_kwargs['type']

	dtype = data.draw(utils.st_one_of(conf.datatype), label='dtype')

	# ========================================
	# test bed
	# ========================================
	x, xu, dx = utils.get_xaxis(domain_x, nx, dtype)
	y, yv, dy = utils.get_yaxis(domain_y, ny, dtype)
	z, zhl, dz = utils.get_zaxis(domain_z, nz, dtype)

	hb = HorizontalBoundary.factory(hb_type, nx, ny, nb, **hb_kwargs)

	domain = Domain(
		domain_x, nx, domain_y, ny, domain_z, nz, zi,
		horizontal_boundary_type=hb_type, nb=nb, horizontal_boundary_kwargs=hb_kwargs,
		topography_type=topo_type, topography_kwargs=topo_kwargs, dtype=dtype
	)

	grid = domain.physical_grid
	utils.compare_dataarrays(x, grid.grid_xy.x)
	utils.compare_dataarrays(xu, grid.grid_xy.x_at_u_locations)
	utils.compare_dataarrays(dx, grid.grid_xy.dx)
	utils.compare_dataarrays(y, grid.grid_xy.y)
	utils.compare_dataarrays(yv, grid.grid_xy.y_at_v_locations)
	utils.compare_dataarrays(dy, grid.grid_xy.dy)
	utils.compare_dataarrays(z, grid.z)
	utils.compare_dataarrays(zhl, grid.z_on_interface_levels)
	utils.compare_dataarrays(dz, grid.dz)

	grid = domain.computational_grid
	utils.compare_dataarrays(
		hb.get_computational_xaxis(x, dims='c_' + x.dims[0]), grid.grid_xy.x
	)
	utils.compare_dataarrays(
		hb.get_computational_xaxis(xu, dims='c_' + xu.dims[0]), grid.grid_xy.x_at_u_locations
	)
	utils.compare_dataarrays(dx, grid.grid_xy.dx)
	utils.compare_dataarrays(
		hb.get_computational_yaxis(y, dims='c_' + y.dims[0]), grid.grid_xy.y
	)
	utils.compare_dataarrays(
		hb.get_computational_yaxis(yv, dims='c_' + yv.dims[0]), grid.grid_xy.y_at_v_locations
	)
	utils.compare_dataarrays(dy, grid.grid_xy.dy)
	utils.compare_dataarrays(z, grid.z)
	utils.compare_dataarrays(zhl, grid.z_on_interface_levels)
	utils.compare_dataarrays(dz, grid.dz)


if __name__ == '__main__':
	pytest.main([__file__])
