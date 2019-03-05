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
from hypothesis import given, strategies as hyp_st, reproduce_failure
import numpy as np
import pytest
from sympl import DataArray

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

from tasmania import Topography1d, Topography2d


@given(hyp_st.data())
def test_topography1d_flat(data):
	domain_x = data.draw(utils.st_interval(axis_name='x'))
	nx = data.draw(utils.st_length(axis_name='x'))
	topo_kwargs = data.draw(
		utils.st_topography1d_kwargs(
			domain_x, topo_type='flat_terrain', topo_time=timedelta(seconds=0)
		)
	)
	dtype = data.draw(utils.st_one_of(conf.datatype))

	xv = \
		np.array([0.5 * (domain_x.values[0]+domain_x.values[1])], dtype=dtype) \
		if nx == 1 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = DataArray(
		xv, coords=[xv], dims=domain_x.dims[0],
		name='x', attrs={'units': domain_x.attrs['units']}
	)

	topo = Topography1d(x, **topo_kwargs)

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


@given(hyp_st.data())
def test_topography1d_gaussian(data):
	domain_x = data.draw(utils.st_interval(axis_name='x'))
	nx = data.draw(utils.st_length(axis_name='x'))
	topo_kwargs = data.draw(
		utils.st_topography1d_kwargs(
			domain_x, topo_type='gaussian', topo_time=timedelta(seconds=0)
		)
	)
	dtype = data.draw(utils.st_one_of(conf.datatype))

	xv = \
		np.array([0.5 * (domain_x.values[0]+domain_x.values[1])], dtype=dtype) \
		if nx == 1 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = DataArray(
		xv, coords=[xv], dims=domain_x.dims[0],
		name='x', attrs={'units': domain_x.attrs['units']}
	)
	topo = Topography1d(x, **topo_kwargs)

	xu = domain_x.attrs['units']
	xc = topo_kwargs['topo_center_x'].to_units(xu).values.item()
	wx = topo_kwargs['topo_width_x'].to_units(xu).values.item()
	hmax = topo_kwargs['topo_max_height'].to_units('m').values.item()

	topo_ref = hmax * np.exp(-((xv-xc) / wx)**2)
	if topo_kwargs['topo_smooth']:
		topo_ref[1:-1] += 0.25 * (
			topo_ref[:-2] - 2*topo_ref[1:-1] + topo_ref[2:]
		)

	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


@given(hyp_st.data())
def test_topography1d_update(data):
	domain_x = data.draw(utils.st_interval(axis_name='x'))
	nx = data.draw(utils.st_length(axis_name='x'))
	topo_kwargs = data.draw(
		utils.st_topography1d_kwargs(
			domain_x, topo_type='gaussian', topo_time=timedelta(seconds=60)
		)
	)
	dtype = data.draw(utils.st_one_of(conf.datatype))

	xv = \
		np.array([0.5 * (domain_x.values[0]+domain_x.values[1])], dtype=dtype) \
		if nx == 1 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
	x = DataArray(
		xv, coords=[xv], dims=domain_x.dims[0],
		name='x', attrs={'units': domain_x.attrs['units']}
	)

	topo = Topography1d(x, **topo_kwargs)

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 60.
	assert topo.topo_fact == 0.

	xu = domain_x.attrs['units']
	xc = topo_kwargs['topo_center_x'].to_units(xu).values.item()
	wx = topo_kwargs['topo_width_x'].to_units(xu).values.item()
	hmax = topo_kwargs['topo_max_height'].to_units('m').values.item()

	topo_ref = hmax * np.exp(-((xv-xc) / wx)**2)
	if topo_kwargs['topo_smooth']:
		topo_ref[1:-1] += 0.25 * (
			topo_ref[:-2] - 2*topo_ref[1:-1] + topo_ref[2:]
		)

	topo.update(timedelta(seconds=30))
	assert np.allclose(topo.topo, 0.5*topo_ref)
	assert topo.topo_fact == 0.5

	topo.update(timedelta(seconds=60))
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.

	topo.update(timedelta(seconds=90))
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.


@given(utils.st_grid_xy())
def test_topography2d_flat(grid):
	topo = Topography2d(grid, topo_type='flat_terrain')

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.


@given(hyp_st.data())
def test_topography2d_gaussian(data):
	grid = data.draw(utils.st_grid_xy())
	topo_kwargs = data.draw(
		utils.st_topography2d_kwargs(
			grid.x, grid.y, topo_type='gaussian', topo_time=timedelta(seconds=0)
		)
	)

	topo = Topography2d(grid, **topo_kwargs)

	nx, ny = grid.nx, grid.ny
	x1d, y1d = grid.x.values, grid.y.values
	x = np.repeat(x1d[:, np.newaxis], ny, axis=1)
	y = np.repeat(y1d[np.newaxis, :], nx, axis=0)

	xc = topo_kwargs['topo_center_x'].to_units(grid.x.attrs['units']).values.item()
	yc = topo_kwargs['topo_center_y'].to_units(grid.y.attrs['units']).values.item()
	wx = topo_kwargs['topo_width_x'].to_units(grid.x.attrs['units']).values.item()
	wy = topo_kwargs['topo_width_y'].to_units(grid.y.attrs['units']).values.item()
	hmax = topo_kwargs['topo_max_height'].to_units('m').values.item()

	topo_ref = hmax * np.exp(- ((x-xc) / wx)**2 - ((y-yc) / wy)**2)
	if topo_kwargs['topo_smooth']:
		topo_ref[1:-1, 1:-1] += 0.125 * (
			topo_ref[:-2, 1:-1] - 2*topo_ref[1:-1, 1:-1] + topo_ref[2:, 1:-1] +
			topo_ref[1:-1, :-2] - 2*topo_ref[1:-1, 1:-1] + topo_ref[1:-1, 2:]
		)

	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.
	assert np.allclose(topo.topo, topo_ref)


@given(hyp_st.data())
def test_topography2d_schaer(data):
	grid = data.draw(utils.st_grid_xy())
	topo_kwargs = data.draw(
		utils.st_topography2d_kwargs(
			grid.x, grid.y, topo_type='schaer', topo_time=timedelta(seconds=0)
		)
	)

	topo = Topography2d(grid, **topo_kwargs)

	nx, ny = grid.nx, grid.ny
	x1d, y1d = grid.x.values, grid.y.values
	x = np.repeat(x1d[:, np.newaxis], ny, axis=1)
	y = np.repeat(y1d[np.newaxis, :], nx, axis=0)

	xc = topo_kwargs['topo_center_x'].to_units(grid.x.attrs['units']).values.item()
	yc = topo_kwargs['topo_center_y'].to_units(grid.y.attrs['units']).values.item()
	wx = topo_kwargs['topo_width_x'].to_units(grid.x.attrs['units']).values.item()
	wy = topo_kwargs['topo_width_y'].to_units(grid.y.attrs['units']).values.item()
	hmax = topo_kwargs['topo_max_height'].to_units('m').values.item()

	topo_ref = hmax / ((1 + ((x-xc) / wx)**2 + ((y-yc) / wy)**2) ** 1.5)
	if topo_kwargs['topo_smooth']:
		topo_ref[1:-1, 1:-1] += 0.125 * (
			topo_ref[:-2, 1:-1] - 2*topo_ref[1:-1, 1:-1] + topo_ref[2:, 1:-1] +
			topo_ref[1:-1, :-2] - 2*topo_ref[1:-1, 1:-1] + topo_ref[1:-1, 2:]
		)

	assert topo.topo_time.total_seconds() == 0.
	assert topo.topo_fact == 1.
	assert np.allclose(topo.topo, topo_ref)


@given(hyp_st.data())
def test_topography2d_update(data):
	grid = data.draw(utils.st_grid_xy())
	topo_kwargs = data.draw(
		utils.st_topography2d_kwargs(
			grid.x, grid.y, topo_type='gaussian', topo_time=timedelta(seconds=60)
		)
	)

	topo = Topography2d(grid, **topo_kwargs)

	nx, ny = grid.nx, grid.ny
	x1d, y1d = grid.x.values, grid.y.values
	x = np.repeat(x1d[:, np.newaxis], ny, axis=1)
	y = np.repeat(y1d[np.newaxis, :], nx, axis=0)

	xc = topo_kwargs['topo_center_x'].to_units(grid.x.attrs['units']).values.item()
	yc = topo_kwargs['topo_center_y'].to_units(grid.y.attrs['units']).values.item()
	wx = topo_kwargs['topo_width_x'].to_units(grid.x.attrs['units']).values.item()
	wy = topo_kwargs['topo_width_y'].to_units(grid.y.attrs['units']).values.item()
	hmax = topo_kwargs['topo_max_height'].to_units('m').values.item()

	topo_ref = hmax * np.exp(- ((x-xc) / wx)**2 - ((y-yc) / wy)**2)
	if topo_kwargs['topo_smooth']:
		topo_ref[1:-1, 1:-1] += 0.125 * (
			topo_ref[:-2, 1:-1] - 2*topo_ref[1:-1, 1:-1] + topo_ref[2:, 1:-1] +
			topo_ref[1:-1, :-2] - 2*topo_ref[1:-1, 1:-1] + topo_ref[1:-1, 2:]
		)

	assert np.count_nonzero(topo.topo.values) == 0
	assert topo.topo_time.total_seconds() == 60.
	assert topo.topo_fact == 0.

	topo.update(timedelta(seconds=30))
	assert np.allclose(topo.topo, 0.5*topo_ref)
	assert topo.topo_fact == 0.5

	topo.update(timedelta(seconds=60))
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.

	topo.update(timedelta(seconds=90))
	assert np.allclose(topo.topo, topo_ref)
	assert topo.topo_fact == 1.


if __name__ == '__main__':
	pytest.main([__file__])
	#test_topography2d_gaussian()
