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
from sympl import DataArray

import sympl
from tasmania.namelist import datatype as dtype


def test_topography_1d_flat():
	domain_x, nx, dims_x, units_x = [0, 100e3], 100, 'x', 'm'

	xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x', attrs={'units': units_x})

	from tasmania.grids.topography import Topography1d as Topography
	topo = Topography(x, topo_type='flat_terrain')

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_1d_gaussian():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'

	xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x', attrs={'units': units_x})

	from tasmania.grids.topography import Topography1d as Topography
	topo = Topography(x, topo_type='gaussian',
					  topo_max_height=DataArray(1., attrs={'units': 'km'}),
					  topo_width_x=DataArray(10., attrs={'units': 'km'}))

	topo_ref = 1.e3 * np.exp(-((xv-50e3) / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)

	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_1d_update():
	domain_x, nx, dims_x, units_x = [0, 100e3], 101, 'x', 'm'

	xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x', attrs={'units': units_x})

	from tasmania.grids.topography import Topography1d as Topography
	from datetime import timedelta
	topo = Topography(x, topo_type='gaussian', topo_time=timedelta(seconds=60),
					  topo_max_height=DataArray(1., attrs={'units': 'km'}),
					  topo_width_x=DataArray(10., attrs={'units': 'km'}))

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 60.
	assert topo.topo_fact == 0.

	topo.update(timedelta(seconds=30))

	topo_ref = 500. * np.exp(-((xv-50e3) / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 0.5

	topo.update(timedelta(seconds=60))

	topo_ref = 1.e3 * np.exp(-((xv-50e3) / 10.e3)**2)
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.

	topo.update(timedelta(seconds=90))
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.


def test_topography_2d_flat():
	domain_x, nx = DataArray([0, 500], dims='x', attrs={'units': 'km'}), 201
	domain_y, ny = DataArray([-50, 50], dims='y', attrs={'units': 'km'}), 101

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny)

	from tasmania.grids.topography import Topography2d as Topography
	topo = Topography(g, topo_type='flat_terrain')

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_2d_gaussian():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 201
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 101

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny)

	from tasmania.grids.topography import Topography2d as Topography
	topo = Topography(g, topo_type='gaussian',
					  topo_max_height=DataArray(1., attrs={'units': 'km'}),
					  topo_width_x=DataArray(10., attrs={'units': 'km'}),
					  topo_width_y=DataArray(10., attrs={'units': 'km'}))

	xv_ = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		  else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	yv_ = np.array([0.5 * (domain_y[0]+domain_y[1])], dtype=dtype) if ny == 1 \
		  else np.linspace(domain_y[0], domain_y[1], ny, dtype=dtype)
	xv = np.repeat(xv_[:, np.newaxis], ny, axis=1)
	yv = np.repeat(yv_[np.newaxis, :], nx, axis=0)
	topo_ref = 1.e3 * np.exp(- ((xv-250.) / 10.)**2 - (yv / 10.)**2)
	assert np.allclose(topo.topo, topo_ref)

	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_2d_schaer():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 201
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 101

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny)

	from tasmania.grids.topography import Topography2d as Topography
	topo = Topography(g, topo_type='gaussian',
					  topo_max_height=DataArray(1., attrs={'units': 'km'}),
					  topo_width_x=DataArray(10., attrs={'units': 'km'}),
					  topo_width_y=DataArray(10., attrs={'units': 'km'}))

	xv_ = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		  else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	yv_ = np.array([0.5 * (domain_y[0]+domain_y[1])], dtype=dtype) if ny == 1 \
		  else np.linspace(domain_y[0], domain_y[1], ny, dtype=dtype)
	xv = np.repeat(xv_[:, np.newaxis], ny, axis=1)
	yv = np.repeat(yv_[np.newaxis, :], nx, axis=0)
	topo_ref = 1.e3 * np.exp(- ((xv-250.) / 10.)**2 - (yv / 10.)**2)
	assert np.allclose(topo.topo, topo_ref)

	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


def test_topography_2d_update():
	domain_x, nx = DataArray([0., 500.], dims='x', attrs={'units': 'km'}), 201
	domain_y, ny = DataArray([-50., 50.], dims='y', attrs={'units': 'km'}), 101

	from tasmania.grids.grid_xy import GridXY as Grid
	g = Grid(domain_x, nx, domain_y, ny)

	from tasmania.grids.topography import Topography2d as Topography
	from datetime import timedelta
	topo = Topography(g, topo_type='gaussian', topo_time=timedelta(seconds=60),
					  topo_max_height=DataArray(1., attrs={'units': 'km'}),
					  topo_width_x=DataArray(10., attrs={'units': 'km'}),
					  topo_width_y=DataArray(10., attrs={'units': 'km'}))

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 60.
	assert topo.topo_fact == 0.

	topo.update(timedelta(seconds=30))

	xv_ = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
		  else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	yv_ = np.array([0.5 * (domain_y[0]+domain_y[1])], dtype=dtype) if ny == 1 \
		  else np.linspace(domain_y[0], domain_y[1], ny, dtype=dtype)
	xv = np.repeat(xv_[:, np.newaxis], ny, axis=1)
	yv = np.repeat(yv_[np.newaxis, :], nx, axis=0)
	topo_ref = 500. * np.exp(- ((xv-250.) / 10.)**2 - (yv / 10.)**2)
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 0.5

	topo.update(timedelta(seconds=60))
	topo_ref = 1.e3 * np.exp(- ((xv-250.) / 10.)**2 - (yv / 10.)**2)
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.

	topo.update(timedelta(seconds=90))
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.


if __name__ == '__main__':
	pytest.main([__file__])
