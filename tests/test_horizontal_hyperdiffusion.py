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
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
import pytest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

from tasmania.python.dwarfs.horizontal_hyperdiffusion import \
	HorizontalHyperDiffusion as HHD


def laplacian_x(dx, phi):
	out = np.zeros_like(phi, phi.dtype)

	out[1:-1, :, :] = \
		(phi[2:, :, :] - 2.0*phi[1:-1, :, :] + phi[:-2, :, :]) / (dx*dx)

	return out


def laplacian_y(dy, phi):
	out = np.zeros_like(phi, phi.dtype)

	out[:, 1:-1, :] = \
		(phi[:, 2:, :] - 2.0*phi[:, 1:-1, :] + phi[:, :-2, :]) / (dy*dy)

	return out


def laplacian2d(dx, dy, phi):
	return laplacian_x(dx, phi) + laplacian_y(dy, phi)


def first_order_diffusion_xyz(dx, dy, phi):
	lap = laplacian2d(dx, dy, phi)
	return lap


def first_order_diffusion_xz(dx, phi):
	lap = laplacian_x(dx, phi)
	return lap


def first_order_diffusion_yz(dy, phi):
	lap = laplacian_y(dy, phi)
	return lap


def first_order_assert_xyz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[1:-1, 1:-1, :], phi_tnd[1:-1, 1:-1, :])


def first_order_assert_xz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[1:-1, :, :], phi_tnd[1:-1, :, :])


def first_order_assert_yz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[:, 1:-1, :], phi_tnd[:, 1:-1, :])


def first_order_validation(phi_rnd, ni, nj, nk, grid, diffusion_depth):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_tnd = np.zeros_like(phi)

	hhd = HHD.factory(
		'first_order', (ni, nj, nk), grid, diffusion_depth, 0.5, 1.0,
		xaxis_units='m', yaxis_units='m', dtype=phi.dtype
	)
	hhd(phi, phi_tnd)

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	if ni < 3:
		phi_tnd_assert = hhd._gamma * first_order_diffusion_yz(dy, phi)
		first_order_assert_yz(phi_tnd, phi_tnd_assert)
	elif nj < 3:
		phi_tnd_assert = hhd._gamma * first_order_diffusion_xz(dx, phi)
		first_order_assert_xz(phi_tnd, phi_tnd_assert)
	else:
		phi_tnd_assert = hhd._gamma * first_order_diffusion_xyz(dx, dy, phi)
		first_order_assert_xyz(phi_tnd, phi_tnd_assert)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_first_order(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(1, 15), yaxis_length=(1, 15), zaxis_length=(1, 15)
		), label='grid'
	)
	assume(not(grid.nx < 3 and grid.ny < 3))
	phi_rnd = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=utils.st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		)
	)
	depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))

	# ========================================
	# test
	# ========================================
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	first_order_validation(phi_rnd, nx  , ny  , nz  , grid, depth)
	first_order_validation(phi_rnd, nx+1, ny  , nz  , grid, depth)
	first_order_validation(phi_rnd, nx  , ny+1, nz  , grid, depth)
	first_order_validation(phi_rnd, nx  , ny  , nz+1, grid, depth)
	first_order_validation(phi_rnd, nx+1, ny+1, nz  , grid, depth)
	first_order_validation(phi_rnd, nx+1, ny  , nz+1, grid, depth)
	first_order_validation(phi_rnd, nx  , ny+1, nz+1, grid, depth)
	first_order_validation(phi_rnd, nx+1, ny+1, nz+1, grid, depth)


def second_order_diffusion_xyz(dx, dy, phi):
	lap0 = laplacian2d(dx, dy, phi)
	lap1 = laplacian2d(dx, dy, lap0)
	return lap1


def second_order_diffusion_xz(dx, phi):
	lap0 = laplacian_x(dx, phi)
	lap1 = laplacian_x(dx, lap0)
	return lap1


def second_order_diffusion_yz(dy, phi):
	lap0 = laplacian_y(dy, phi)
	lap1 = laplacian_y(dy, lap0)
	return lap1


def second_order_assert_xyz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[2:-2, 2:-2, :], phi_tnd[2:-2, 2:-2, :])


def second_order_assert_xz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[2:-2, :, :], phi_tnd[2:-2, :, :])


def second_order_assert_yz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[:, 2:-2, :], phi_tnd[:, 2:-2, :])


def second_order_validation(phi_rnd, ni, nj, nk, grid, diffusion_depth):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_tnd = np.zeros_like(phi)

	hhd = HHD.factory(
		'second_order', (ni, nj, nk), grid, diffusion_depth, 0.5, 1.0,
		xaxis_units='m', yaxis_units='m', dtype=phi.dtype
	)
	hhd(phi, phi_tnd)

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	if ni < 5:
		phi_tnd_assert = hhd._gamma * second_order_diffusion_yz(dy, phi)
		second_order_assert_yz(phi_tnd, phi_tnd_assert)
	elif nj < 5:
		phi_tnd_assert = hhd._gamma * second_order_diffusion_xz(dx, phi)
		second_order_assert_xz(phi_tnd, phi_tnd_assert)
	else:
		phi_tnd_assert = hhd._gamma * second_order_diffusion_xyz(dx, dy, phi)
		second_order_assert_xyz(phi_tnd, phi_tnd_assert)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_second_order(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(1, 15), yaxis_length=(1, 15), zaxis_length=(1, 15)
		)
	)
	assume(not(grid.nx < 5 and grid.ny < 5))
	phi_rnd = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=utils.st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		)
	)
	depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))

	# ========================================
	# test
	# ========================================
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	second_order_validation(phi_rnd, nx  , ny  , nz  , grid, depth)
	second_order_validation(phi_rnd, nx+1, ny  , nz  , grid, depth)
	second_order_validation(phi_rnd, nx  , ny+1, nz  , grid, depth)
	second_order_validation(phi_rnd, nx  , ny  , nz+1, grid, depth)
	second_order_validation(phi_rnd, nx+1, ny+1, nz  , grid, depth)
	second_order_validation(phi_rnd, nx+1, ny  , nz+1, grid, depth)
	second_order_validation(phi_rnd, nx  , ny+1, nz+1, grid, depth)
	second_order_validation(phi_rnd, nx+1, ny+1, nz+1, grid, depth)


def third_order_diffusion_xyz(dx, dy, phi):
	lap0 = laplacian2d(dx, dy, phi)
	lap1 = laplacian2d(dx, dy, lap0)
	lap2 = laplacian2d(dx, dy, lap1)
	return lap2


def third_order_diffusion_xz(dx, phi):
	lap0 = laplacian_x(dx, phi)
	lap1 = laplacian_x(dx, lap0)
	lap2 = laplacian_x(dx, lap1)
	return lap2


def third_order_diffusion_yz(dy, phi):
	lap0 = laplacian_y(dy, phi)
	lap1 = laplacian_y(dy, lap0)
	lap2 = laplacian_y(dy, lap1)
	return lap2


def third_order_assert_xyz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[3:-3, 3:-3, :], phi_tnd[3:-3, 3:-3, :])


def third_order_assert_xz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[3:-3, :, :], phi_tnd[3:-3, :, :])


def third_order_assert_yz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[:, 3:-3, :], phi_tnd[:, 3:-3, :])


def third_order_validation(phi_rnd, ni, nj, nk, grid, diffusion_depth):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_tnd = np.zeros_like(phi)

	hhd = HHD.factory(
		'third_order', (ni, nj, nk), grid, diffusion_depth, 0.5, 1.0,
		xaxis_units='m', yaxis_units='m', dtype=phi.dtype
	)
	hhd(phi, phi_tnd)

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	if ni < 7:
		phi_tnd_assert = hhd._gamma * third_order_diffusion_yz(dy, phi)
		third_order_assert_yz(phi_tnd, phi_tnd_assert)
	elif nj < 7:
		phi_tnd_assert = hhd._gamma * third_order_diffusion_xz(dx, phi)
		third_order_assert_xz(phi_tnd, phi_tnd_assert)
	else:
		phi_tnd_assert = hhd._gamma * third_order_diffusion_xyz(dx, dy, phi)
		third_order_assert_xyz(phi_tnd, phi_tnd_assert)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_third_order(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(
		utils.st_grid_xyz(
			xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 10)
		)
	)
	assume(not(grid.nx < 7 and grid.ny < 7))
	phi_rnd = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx+1, grid.ny+1, grid.nz+1),
			elements=utils.st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		)
	)
	depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))

	# ========================================
	# test
	# ========================================
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	third_order_validation(phi_rnd, nx  , ny  , nz  , grid, depth)
	third_order_validation(phi_rnd, nx+1, ny  , nz  , grid, depth)
	third_order_validation(phi_rnd, nx  , ny+1, nz  , grid, depth)
	third_order_validation(phi_rnd, nx  , ny  , nz+1, grid, depth)
	third_order_validation(phi_rnd, nx+1, ny+1, nz  , grid, depth)
	third_order_validation(phi_rnd, nx+1, ny  , nz+1, grid, depth)
	third_order_validation(phi_rnd, nx  , ny+1, nz+1, grid, depth)
	third_order_validation(phi_rnd, nx+1, ny+1, nz+1, grid, depth)


if __name__ == '__main__':
	pytest.main([__file__])
	#test_third_order()
