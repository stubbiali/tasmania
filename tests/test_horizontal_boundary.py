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
from hypothesis import \
	assume, given, HealthCheck, settings, strategies as hyp_st, reproduce_failure
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

from tasmania.python.grids.horizontal_boundary import HorizontalBoundary
from tasmania.python.grids._horizontal_boundary import \
	Relaxed, Relaxed1DX, Relaxed1DY, Periodic, Periodic1DX, Periodic1DY
from tasmania.python.utils.utils import equal_to as eq


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_relaxed(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(utils.st_physical_grid(), label="grid")
	nx, ny = grid.grid_xy.nx, grid.grid_xy.ny

	nb = data.draw(utils.st_horizontal_boundary_layers(nx, ny), label="nb")
	hb_kwargs = data.draw(
		utils.st_horizontal_boundary_kwargs("relaxed", nx, ny, nb), label="hb_kwargs"
	)

	state = data.draw(
		utils.st_isentropic_state(grid, moist=True, precipitation=True), label="state"
	)

	# ========================================
	# test
	# ========================================
	hb = HorizontalBoundary.factory("relaxed", nx, ny, nb, **hb_kwargs)

	if ny == 1:
		assert isinstance(hb, Relaxed1DX)
	elif nx == 1:
		assert isinstance(hb, Relaxed1DY)
	else:
		assert isinstance(hb, Relaxed)

	assert hb.nx == nx
	assert hb.ny == ny
	assert hb.nb == nb

	if ny == 1:
		assert hb.ni == nx
		assert hb.nj == 2*nb + 1
	elif nx == 1:
		assert hb.ni == 2*nb + 1
		assert hb.nj == ny
	else:
		assert hb.ni == nx
		assert hb.nj == ny

	#
	# x-axis
	#
	x, xu = grid.grid_xy.x, grid.grid_xy.x_at_u_locations

	if nx == 1:
		cx = hb.get_computational_xaxis(x)
		assert len(cx.dims) == 1
		assert cx.dims[0] == x.dims[0]
		assert cx.attrs['units'] == x.attrs['units']
		assert cx.values.shape[0] == 2*nb+1
		assert all([cx.values[i] == x.values[0] for i in range(2*nb+1)])
		assert len(cx.coords) == 1
		assert all([
			cx.coords[cx.dims[0]].values[i] == x.values[0]
			for i in range(2*nb+1)
		])

		cx = hb.get_computational_xaxis(xu)
		assert len(cx.dims) == 1
		assert cx.dims[0] == xu.dims[0]
		assert cx.attrs['units'] == xu.attrs['units']
		assert cx.values.shape[0] == 2*nb+2
		assert all([cx.values[i] == xu.values[0] for i in range(nb+1)])
		assert all([cx.values[i] == xu.values[1] for i in range(nb+1, 2*nb+2)])
		assert len(cx.coords) == 1
		assert all([
			cx.coords[cx.dims[0]].values[i] == xu.values[0]
			for i in range(nb+1)
		])
		assert all([
			cx.coords[cx.dims[0]].values[i] == xu.values[1]
			for i in range(nb+1, 2*nb+2)
		])
	else:
		utils.compare_dataarrays(x, hb.get_computational_xaxis(x))
		utils.compare_dataarrays(xu, hb.get_computational_xaxis(xu))

	utils.compare_dataarrays(
		x, hb.get_physical_xaxis(hb.get_computational_xaxis(x))
	)
	utils.compare_dataarrays(
		xu, hb.get_physical_xaxis(hb.get_computational_xaxis(xu))
	)

	#
	# y-axis
	#
	y, yv = grid.grid_xy.y, grid.grid_xy.y_at_v_locations

	if ny == 1:
		cy = hb.get_computational_yaxis(y)
		assert len(cy.dims) == 1
		assert cy.dims[0] == y.dims[0]
		assert cy.attrs['units'] == y.attrs['units']
		assert cy.values.shape[0] == 2*nb+1
		assert all([cy.values[i] == y.values[0] for i in range(2*nb+1)])
		assert len(cy.coords) == 1
		assert all([
			cy.coords[cy.dims[0]].values[i] == y.values[0]
			for i in range(2*nb+1)
		])

		cy = hb.get_computational_yaxis(yv)
		assert len(cy.dims) == 1
		assert cy.dims[0] == yv.dims[0]
		assert cy.attrs['units'] == yv.attrs['units']
		assert cy.values.shape[0] == 2*nb+2
		assert all([cy.values[i] == yv.values[0] for i in range(nb+1)])
		assert all([cy.values[i] == yv.values[1] for i in range(nb+1, 2*nb+2)])
		assert len(cy.coords) == 1
		assert all([
			cy.coords[cy.dims[0]].values[i] == yv.values[0]
			for i in range(nb+1)
		])
		assert all([
			cy.coords[cy.dims[0]].values[i] == yv.values[1]
			for i in range(nb+1, 2*nb+2)
		])
	else:
		utils.compare_dataarrays(y, hb.get_computational_yaxis(y))
		utils.compare_dataarrays(yv, hb.get_computational_yaxis(yv))

	utils.compare_dataarrays(
		y, hb.get_physical_yaxis(hb.get_computational_yaxis(y))
	)
	utils.compare_dataarrays(
		yv, hb.get_physical_yaxis(hb.get_computational_yaxis(yv))
	)

	#
	# computational and physical field
	#
	field = state['air_isentropic_density'].values
	field_stgx = state['x_velocity_at_u_locations'].values
	field_stgy = state['y_velocity_at_v_locations'].values

	if ny == 1:
		assert np.allclose(
			hb.get_computational_field(field), np.repeat(field, 2*nb+1, axis=1)
		)
		assert np.allclose(
			hb.get_computational_field(field_stgx), np.repeat(field_stgx, 2*nb+1, axis=1)
		)
		assert np.allclose(
			hb.get_computational_field(field_stgy)[:, :nb+1, :],
			np.repeat(field_stgy[:, 0:1, :], nb+1, axis=1)
		)
		assert np.allclose(
			hb.get_computational_field(field_stgy)[:, nb+1:, :],
			np.repeat(field_stgy[:, -1:, :], nb+1, axis=1)
		)
	elif nx == 1:
		assert np.allclose(
			hb.get_computational_field(field), np.repeat(field, 2*nb+1, axis=0)
		)
		assert np.allclose(
			hb.get_computational_field(field_stgx)[:nb+1, :, :],
			np.repeat(field_stgx[0:1, :, :], nb+1, axis=0)
		)
		assert np.allclose(
			hb.get_computational_field(field_stgx)[nb+1:, :, :],
			np.repeat(field_stgx[-1:, :, :], nb+1, axis=0)
		)
		assert np.allclose(
			hb.get_computational_field(field_stgy), np.repeat(field_stgy, 2*nb+1, axis=0)
		)
	else:
		assert np.allclose(hb.get_computational_field(field), field)
		assert np.allclose(hb.get_computational_field(field_stgx), field_stgx)
		assert np.allclose(hb.get_computational_field(field_stgy), field_stgy)

	assert np.allclose(
		hb.get_physical_field(hb.get_computational_field(field)), field
	)

	#
	# enforce
	#
	hb.enforce(state)

	#hb.reference_state = state

	#hb.enforce(state)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_periodic(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(utils.st_physical_grid(), label="grid")
	nx, ny = grid.grid_xy.nx, grid.grid_xy.ny

	nb = data.draw(utils.st_horizontal_boundary_layers(nx, ny), label="nb")
	hb_kwargs = data.draw(
		utils.st_horizontal_boundary_kwargs("periodic", nx, ny, nb), label="hb_kwargs"
	)

	state = data.draw(
		utils.st_isentropic_state(grid, moist=True, precipitation=True), label="state"
	)

	# ========================================
	# test
	# ========================================
	hb = HorizontalBoundary.factory("periodic", nx, ny, nb, **hb_kwargs)

	if ny == 1:
		assert isinstance(hb, Periodic1DX)
	elif nx == 1:
		assert isinstance(hb, Periodic1DY)
	else:
		assert isinstance(hb, Periodic)

	assert hb.nx == nx
	assert hb.ny == ny
	assert hb.nb == nb
	assert hb.ni == nx + 2*nb
	assert hb.nj == ny + 2*nb

	#
	# x-axis
	#
	x, xu = grid.grid_xy.x, grid.grid_xy.x_at_u_locations

	if nx == 1:
		cx = hb.get_computational_xaxis(x)
		assert len(cx.dims) == 1
		assert cx.dims[0] == x.dims[0]
		assert cx.attrs['units'] == x.attrs['units']
		assert cx.values.shape[0] == 2*nb+1
		assert all([cx.values[i] == x.values[0] for i in range(2*nb+1)])
		assert len(cx.coords) == 1
		assert all([
			cx.coords[cx.dims[0]].values[i] == x.values[0]
			for i in range(2*nb+1)
		])

		cx = hb.get_computational_xaxis(xu)
		assert len(cx.dims) == 1
		assert cx.dims[0] == xu.dims[0]
		assert cx.attrs['units'] == xu.attrs['units']
		assert cx.values.shape[0] == 2*nb+2
		assert all([cx.values[i] == xu.values[0] for i in range(nb+1)])
		assert all([cx.values[i] == xu.values[1] for i in range(nb+1, 2*nb+2)])
		assert len(cx.coords) == 1
		assert all([
			cx.coords[cx.dims[0]].values[i] == xu.values[0]
			for i in range(nb+1)
		])
		assert all([
			cx.coords[cx.dims[0]].values[i] == xu.values[1]
			for i in range(nb+1, 2*nb+2)
		])
	else:
		dx = grid.grid_xy.dx.values.item()

		cx = hb.get_computational_xaxis(x)
		assert len(cx.dims) == 1
		assert cx.dims[0] == x.dims[0]
		assert cx.attrs['units'] == x.attrs['units']
		assert cx.values.shape[0] == nx+2*nb
		assert all(
			[eq(cx.values[i+1] - cx.values[i], dx) for i in range(nx+2*nb-1)])
		assert len(cx.coords) == 1
		assert all([
			eq(cx.coords[cx.dims[0]].values[i+1] - cx.coords[cx.dims[0]].values[i], dx)
			for i in range(nx+2*nb-1)
		])

		cx = hb.get_computational_xaxis(xu)
		assert len(cx.dims) == 1
		assert cx.dims[0] == xu.dims[0]
		assert cx.attrs['units'] == xu.attrs['units']
		assert cx.values.shape[0] == nx+2*nb+1
		assert all(
			[eq(cx.values[i+1] - cx.values[i], dx) for i in range(nx+2*nb)])
		assert len(cx.coords) == 1
		assert all([
			eq(cx.coords[cx.dims[0]].values[i+1] - cx.coords[cx.dims[0]].values[i], dx)
			for i in range(nx+2*nb)
		])

	#
	# y-axis
	#
	y, yv = grid.grid_xy.y, grid.grid_xy.y_at_v_locations

	if ny == 1:
		cy = hb.get_computational_yaxis(y)
		assert len(cy.dims) == 1
		assert cy.dims[0] == y.dims[0]
		assert cy.attrs['units'] == y.attrs['units']
		assert cy.values.shape[0] == 2*nb+1
		assert all([cy.values[i] == y.values[0] for i in range(2*nb+1)])
		assert len(cy.coords) == 1
		assert all([
			cy.coords[cy.dims[0]].values[i] == y.values[0]
			for i in range(2*nb+1)
		])

		cy = hb.get_computational_yaxis(yv)
		assert len(cy.dims) == 1
		assert cy.dims[0] == yv.dims[0]
		assert cy.attrs['units'] == yv.attrs['units']
		assert cy.values.shape[0] == 2*nb+2
		assert all([cy.values[i] == yv.values[0] for i in range(nb+1)])
		assert all([cy.values[i] == yv.values[1] for i in range(nb+1, 2*nb+2)])
		assert len(cy.coords) == 1
		assert all([
			cy.coords[cy.dims[0]].values[i] == yv.values[0]
			for i in range(nb+1)
		])
		assert all([
			cy.coords[cy.dims[0]].values[i] == yv.values[1]
			for i in range(nb+1, 2*nb+2)
		])
	else:
		dy = grid.grid_xy.dy.values.item()

		cy = hb.get_computational_yaxis(y)
		assert len(cy.dims) == 1
		assert cy.dims[0] == y.dims[0]
		assert cy.attrs['units'] == y.attrs['units']
		assert cy.values.shape[0] == ny+2*nb
		assert all(
			[eq(cy.values[i+1] - cy.values[i], dy) for i in range(ny+2*nb-1)])
		assert len(cy.coords) == 1
		assert all([
			eq(cy.coords[cy.dims[0]].values[i+1] - cy.coords[cy.dims[0]].values[i], dy)
			for i in range(ny+2*nb-1)
		])

		cy = hb.get_computational_yaxis(yv)
		assert len(cy.dims) == 1
		assert cy.dims[0] == yv.dims[0]
		assert cy.attrs['units'] == yv.attrs['units']
		assert cy.values.shape[0] == ny+2*nb+1
		assert all(
			[eq(cy.values[i+1] - cy.values[i], dy) for i in range(ny+2*nb)])
		assert len(cy.coords) == 1
		assert all([
			eq(cy.coords[cy.dims[0]].values[i+1] - cy.coords[cy.dims[0]].values[i], dy)
			for i in range(ny+2*nb)
		])

	utils.compare_dataarrays(
		x, hb.get_physical_xaxis(hb.get_computational_xaxis(x))
	)
	utils.compare_dataarrays(
		xu, hb.get_physical_xaxis(hb.get_computational_xaxis(xu))
	)

	#
	# computational and physical unstaggered field
	#
	field = state['air_isentropic_density'].values

	field[-1, :] = field[0, :]
	field[:, -1] = field[:, 0]
	cfield = hb.get_computational_field(field)

	assert cfield.shape[0:2] == (nx+2*nb, ny+2*nb)

	avgx = 0.5 * (cfield[nb-1:-nb-1, :] + cfield[nb+1:-nb+1, :]) if nb > 1 else \
		0.5 * (cfield[nb-1:-nb-1, :] + cfield[nb+1:, :])
	assert np.allclose(avgx[0, :], avgx[-1, :])
	if nx == 1:
		assert np.allclose(avgx, cfield[nb:nb, :])
	avgy = 0.5 * (cfield[:, nb-1:-nb-1] + cfield[:, nb+1:-nb+1]) if nb > 1 else \
		0.5 * (cfield[:, nb-1:-nb-1] + cfield[:, nb+1:])
	assert np.allclose(avgy[:, 0], avgy[:, -1])
	if ny == 1:
		assert np.allclose(avgy, cfield[:, nb:nb])

	if nb > 1:
		avgx = 0.5 * (cfield[nb-2:-nb-2, :] + cfield[nb+2:-nb+2, :]) if nb > 2 else \
			0.5 * (cfield[nb-2:-nb-2, :] + cfield[nb+2:, :])
		assert np.allclose(avgx[0, :], avgx[-1, :])
		if nx == 1:
			assert np.allclose(avgx, cfield[nb:nb, :])
		avgy = 0.5 * (cfield[:, nb-2:-nb-2] + cfield[:, nb+2:-nb+2]) if nb > 2 else \
			0.5 * (cfield[:, nb-2:-nb-2] + cfield[:, nb+2:])
		assert np.allclose(avgy[:, 0], avgy[:, -1])
		if ny == 1:
			assert np.allclose(avgy, cfield[:, nb:nb])

	if nb > 2:
		avgx = 0.5 * (cfield[nb-3:-nb-3, :] + cfield[nb+3:-nb+3, :]) if nb > 3 else \
			0.5 * (cfield[nb-3:-nb-3, :] + cfield[nb+3:, :])
		assert np.allclose(avgx[0, :], avgx[-1, :])
		if nx == 1:
			assert np.allclose(avgx, cfield[nb:nb, :])
		avgy = 0.5 * (cfield[:, nb-3:-nb-3] + cfield[:, nb+3:-nb+3]) if nb > 3 else \
			0.5 * (cfield[:, nb-3:-nb-3] + cfield[:, nb+3:])
		assert np.allclose(avgy[:, 0], avgy[:, -1])
		if ny == 1:
			assert np.allclose(avgy, cfield[:, nb:nb])

	#
	# computational and physical x-staggered field
	#
	field = state['x_velocity_at_u_locations'].values

	field[-2, :] = field[0, :]
	field[-1, :] = field[1, :]
	field[:, -1] = field[:, 0]
	cfield = hb.get_computational_field(field)

	assert cfield.shape[0:2] == (nx+2*nb+1, ny+2*nb)

	avgx = 0.5 * (cfield[nb-1:-nb-1, :] + cfield[nb+1:-nb+1, :]) if nb > 1 else \
		0.5 * (cfield[nb-1:-nb-1, :] + cfield[nb+1:, :])
	assert np.allclose(avgx[0, :], avgx[-2, :])
	assert np.allclose(avgx[1, :], avgx[-1, :])
	avgy = 0.5 * (cfield[:, nb-1:-nb-1] + cfield[:, nb+1:-nb+1]) if nb > 1 else \
		0.5 * (cfield[:, nb-1:-nb-1] + cfield[:, nb+1:])
	assert np.allclose(avgy[:, 0], avgy[:, -1])
	if ny == 1:
		assert np.allclose(avgy, cfield[:, nb:nb])

	if nb > 1:
		avgx = 0.5 * (cfield[nb-2:-nb-2, :] + cfield[nb+2:-nb+2, :]) if nb > 2 else \
			0.5 * (cfield[nb-2:-nb-2, :] + cfield[nb+2:, :])
		assert np.allclose(avgx[0, :], avgx[-2, :])
		assert np.allclose(avgx[1, :], avgx[-1, :])
		avgy = 0.5 * (cfield[:, nb-2:-nb-2] + cfield[:, nb+2:-nb+2]) if nb > 2 else \
			0.5 * (cfield[:, nb-2:-nb-2] + cfield[:, nb+2:])
		assert np.allclose(avgy[:, 0], avgy[:, -1])
		if ny == 1:
			assert np.allclose(avgy, cfield[:, nb:nb])

	if nb > 2:
		avgx = 0.5 * (cfield[nb-3:-nb-3, :] + cfield[nb+3:-nb+3, :]) if nb > 3 else \
			0.5 * (cfield[nb-3:-nb-3, :] + cfield[nb+3:, :])
		assert np.allclose(avgx[0, :], avgx[-2, :])
		assert np.allclose(avgx[1, :], avgx[-1, :])
		avgy = 0.5 * (cfield[:, nb-3:-nb-3] + cfield[:, nb+3:-nb+3]) if nb > 3 else \
			0.5 * (cfield[:, nb-3:-nb-3] + cfield[:, nb+3:])
		assert np.allclose(avgy[:, 0], avgy[:, -1])
		if ny == 1:
			assert np.allclose(avgy, cfield[:, nb:nb])

	#
	# computational and physical y-staggered field
	#
	field = state['y_velocity_at_v_locations'].values

	field[-1, :] = field[0, :]
	field[:, -2] = field[:, 0]
	field[:, -1] = field[:, 1]
	cfield = hb.get_computational_field(field)

	assert cfield.shape[0:2] == (nx+2*nb, ny+2*nb+1)

	avgx = 0.5 * (cfield[nb-1:-nb-1, :] + cfield[nb+1:-nb+1, :]) if nb > 1 else \
		0.5 * (cfield[nb-1:-nb-1, :] + cfield[nb+1:, :])
	assert np.allclose(avgx[0, :], avgx[-1, :])
	if nx == 1:
		assert np.allclose(avgx, cfield[nb:nb, :])
	avgy = 0.5 * (cfield[:, nb-1:-nb-1] + cfield[:, nb+1:-nb+1]) if nb > 1 else \
		0.5 * (cfield[:, nb-1:-nb-1] + cfield[:, nb+1:])
	assert np.allclose(avgy[:, 0], avgy[:, -2])
	assert np.allclose(avgy[:, 1], avgy[:, -1])

	if nb > 1:
		avgx = 0.5 * (cfield[nb-2:-nb-2, :] + cfield[nb+2:-nb+2, :]) if nb > 2 else \
			0.5 * (cfield[nb-2:-nb-2, :] + cfield[nb+2:, :])
		assert np.allclose(avgx[0, :], avgx[-1, :])
		if nx == 1:
			assert np.allclose(avgx, cfield[nb:nb, :])
		avgy = 0.5 * (cfield[:, nb-2:-nb-2] + cfield[:, nb+2:-nb+2]) if nb > 2 else \
			0.5 * (cfield[:, nb-2:-nb-2] + cfield[:, nb+2:])
		assert np.allclose(avgy[:, 0], avgy[:, -2])
		assert np.allclose(avgy[:, 1], avgy[:, -1])

	if nb > 2:
		avgx = 0.5 * (cfield[nb-3:-nb-3, :] + cfield[nb+3:-nb+3, :]) if nb > 3 else \
			0.5 * (cfield[nb-3:-nb-3, :] + cfield[nb+3:, :])
		assert np.allclose(avgx[0, :], avgx[-1, :])
		if nx == 1:
			assert np.allclose(avgx, cfield[nb:nb, :])
		avgy = 0.5 * (cfield[:, nb-3:-nb-3] + cfield[:, nb+3:-nb+3]) if nb > 3 else \
			0.5 * (cfield[:, nb-3:-nb-3] + cfield[:, nb+3:])
		assert np.allclose(avgy[:, 0], avgy[:, -2])
		assert np.allclose(avgy[:, 1], avgy[:, -1])

	#
	# enforce
	#
	field = state['air_isentropic_density'].values
	cfield = hb.get_computational_field(field)
	vfield = deepcopy(cfield)
	hb.enforce_field(cfield)
	assert np.allclose(cfield, vfield)

	field = state['x_velocity_at_u_locations'].values
	cfield = hb.get_computational_field(field)
	vfield = deepcopy(cfield)
	hb.enforce_field(cfield)
	assert np.allclose(cfield, vfield)

	field = state['y_velocity_at_v_locations'].values
	cfield = hb.get_computational_field(field)
	vfield = deepcopy(cfield)
	hb.enforce_field(cfield)
	assert np.allclose(cfield, vfield)


if __name__ == "__main__":
	pytest.main([__file__])
	#test_periodic()
