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

from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing as HS


def first_order_smoothing_xyz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(1, ni-1), slice(1, nj-1), slice(0, nk)
	im1, ip1 = slice(0, ni-2), slice(2, ni)
	jm1, jp1 = slice(0, nj-2), slice(2, nj)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - g[i, j, k]) * phi[i, j, k] + \
		0.25 * g[i, j, k] * (
			phi[ip1, j, k] + phi[im1, j, k] + phi[i, jm1, k] + phi[i, jp1, k]
		)

	return phi_smooth


def first_order_smoothing_xz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(1, ni-1), slice(0, nj), slice(0, nk)
	im1, ip1 = slice(0, ni-2), slice(2, ni)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - 0.5 * g[i, j, k]) * phi[i, j, k] + \
		0.25 * g[i, j, k] * (phi[ip1, j, k] + phi[im1, j, k])

	return phi_smooth


def first_order_smoothing_yz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(0, ni), slice(1, nj-1), slice(0, nk)
	jm1, jp1 = slice(0, nj-2), slice(2, nj)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - 0.5 * g[i, j, k]) * phi[i, j, k] + \
		0.25 * g[i, j, k] * (phi[i, jm1, k] + phi[i, jp1, k])

	return phi_smooth


def first_order_assert_xyz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[1:-1, 1:-1, :], phi_new[1:-1, 1:-1, :])
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def first_order_assert_xz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[1:-1, :, :], phi_new[1:-1, :, :])
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])


def first_order_assert_yz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[:, 1:-1, :], phi_new[:, 1:-1, :])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def first_order_validation(phi_rnd, ni, nj, nk, grid, smooth_depth):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_new = np.zeros_like(phi)

	hs = HS.factory(
		'first_order', (ni, nj, nk), grid, smooth_depth, 0.5, 1.0, dtype=phi.dtype
	)
	hs(phi, phi_new)

	if ni < 3:
		phi_new_assert = first_order_smoothing_yz(phi, hs._gamma)
		first_order_assert_yz(phi, phi_new, phi_new_assert)
	elif nj < 3:
		phi_new_assert = first_order_smoothing_xz(phi, hs._gamma)
		first_order_assert_xz(phi, phi_new, phi_new_assert)
	else:
		phi_new_assert = first_order_smoothing_xyz(phi, hs._gamma)
		first_order_assert_xyz(phi, phi_new, phi_new_assert)


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


def second_order_smoothing_xyz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(2, ni-2), slice(2, nj-2), slice(0, nk)
	im1, ip1 = slice(1, ni-3), slice(3, ni-1)
	im2, ip2 = slice(0, ni-4), slice(4, ni)
	jm1, jp1 = slice(1, nj-3), slice(3, nj-1)
	jm2, jp2 = slice(0, nj-4), slice(4, nj)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - 0.75 * g[i, j, k]) * phi[i, j, k] + \
		0.0625 * g[i, j, k] * (
			- phi[im2, j, k] + 4.0 * phi[im1, j, k]
			- phi[ip2, j, k] + 4.0 * phi[ip1, j, k]
			- phi[i, jm2, k] + 4.0 * phi[i, jm1, k]
			- phi[i, jp2, k] + 4.0 * phi[i, jp1, k]
		)

	return phi_smooth


def second_order_smoothing_xz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(2, ni-2), slice(0, nj), slice(0, nk)
	im1, ip1 = slice(1, ni-3), slice(3, ni-1)
	im2, ip2 = slice(0, ni-4), slice(4, ni)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - 0.375 * g[i, j, k]) * phi[i, j, k] + \
		0.0625 * g[i, j, k] * (
			- phi[im2, j, k] + 4.0 * phi[im1, j, k]
			- phi[ip2, j, k] + 4.0 * phi[ip1, j, k]
		)

	return phi_smooth


def second_order_smoothing_yz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(0, ni), slice(2, nj-2), slice(0, nk)
	jm1, jp1 = slice(1, nj-3), slice(3, nj-1)
	jm2, jp2 = slice(0, nj-4), slice(4, nj)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - 0.375 * g[i, j, k]) * phi[i, j, k] + \
		0.0625 * g[i, j, k] * (
			- phi[i, jm2, k] + 4.0 * phi[i, jm1, k]
			- phi[i, jp2, k] + 4.0 * phi[i, jp1, k]
		)

	return phi_smooth


def second_order_assert_xyz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[2:-2, 2:-2, :], phi_new[2:-2, 2:-2, :])
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[1, :, :], phi_new[1, :, :])
	assert np.allclose(phi[-2, :, :], phi_new[-2, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, 1, :], phi_new[:, 1, :])
	assert np.allclose(phi[:, -2, :], phi_new[:, -2, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def second_order_assert_xz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[2:-2, :, :], phi_new[2:-2, :, :])
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[1, :, :], phi_new[1, :, :])
	assert np.allclose(phi[-2, :, :], phi_new[-2, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])


def second_order_assert_yz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[:, 2:-2, :], phi_new[:, 2:-2, :])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, 1, :], phi_new[:, 1, :])
	assert np.allclose(phi[:, -2, :], phi_new[:, -2, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def second_order_validation(phi_rnd, ni, nj, nk, grid, smooth_depth):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_new = np.zeros_like(phi)

	hs = HS.factory(
		'second_order', (ni, nj, nk), grid, smooth_depth, 0.5, 1.0, dtype=phi.dtype
	)
	hs(phi, phi_new)

	if ni < 5:
		phi_new_assert = second_order_smoothing_yz(phi, hs._gamma)
		second_order_assert_yz(phi, phi_new, phi_new_assert)
	elif nj < 5:
		phi_new_assert = second_order_smoothing_xz(phi, hs._gamma)
		second_order_assert_xz(phi, phi_new, phi_new_assert)
	else:
		phi_new_assert = second_order_smoothing_xyz(phi, hs._gamma)
		second_order_assert_xyz(phi, phi_new, phi_new_assert)


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


def third_order_smoothing_xyz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(3, ni-3), slice(3, nj-3), slice(0, nk)
	im1, ip1 = slice(2, ni-4), slice(4, ni-2)
	im2, ip2 = slice(1, ni-5), slice(5, ni-1)
	im3, ip3 = slice(0, ni-6), slice(6, ni)
	jm1, jp1 = slice(2, nj-4), slice(4, nj-2)
	jm2, jp2 = slice(1, nj-5), slice(5, nj-1)
	jm3, jp3 = slice(0, nj-6), slice(6, nj)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - 0.625 * g[i, j, k]) * phi[i, j, k] + \
		0.015625 * g[i, j, k] * (
			phi[im3, j, k] - 6.0 * phi[im2, j, k] + 15.0 * phi[im1, j, k] +
			phi[ip3, j, k] - 6.0 * phi[ip2, j, k] + 15.0 * phi[ip1, j, k] +
			phi[i, jm3, k] - 6.0 * phi[i, jm2, k] + 15.0 * phi[i, jm1, k] +
			phi[i, jp3, k] - 6.0 * phi[i, jp2, k] + 15.0 * phi[i, jp1, k]
	)

	return phi_smooth


def third_order_smoothing_xz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(3, ni-3), slice(0, nj), slice(0, nk)
	im1, ip1 = slice(2, ni-4), slice(4, ni-2)
	im2, ip2 = slice(1, ni-5), slice(5, ni-1)
	im3, ip3 = slice(0, ni-6), slice(6, ni)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - 0.3125 * g[i, j, k]) * phi[i, j, k] + \
		0.015625 * g[i, j, k] * (
			phi[im3, j, k] - 6.0 * phi[im2, j, k] + 15.0 * phi[im1, j, k] +
			phi[ip3, j, k] - 6.0 * phi[ip2, j, k] + 15.0 * phi[ip1, j, k]
		)

	return phi_smooth


def third_order_smoothing_yz(phi, g):
	ni, nj, nk = phi.shape

	i, j, k = slice(0, ni), slice(3, nj-3), slice(0, nk)
	jm1, jp1 = slice(2, nj-4), slice(4, nj-2)
	jm2, jp2 = slice(1, nj-5), slice(5, nj-1)
	jm3, jp3 = slice(0, nj-6), slice(6, nj)

	phi_smooth = np.zeros((ni, nj, nk), dtype=phi.dtype)
	phi_smooth[i, j, k] = (1 - 0.3125 * g[i, j, k]) * phi[i, j, k] + \
		0.015625 * g[i, j, k] * (
			phi[i, jm3, k] - 6.0 * phi[i, jm2, k] + 15.0 * phi[i, jm1, k] +
			phi[i, jp3, k] - 6.0 * phi[i, jp2, k] + 15.0 * phi[i, jp1, k]
		)

	return phi_smooth


def third_order_assert_xyz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[3:-3, 3:-3, :], phi_new[3:-3, 3:-3, :])
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[1, :, :], phi_new[1, :, :])
	assert np.allclose(phi[2, :, :], phi_new[2, :, :])
	assert np.allclose(phi[-3, :, :], phi_new[-3, :, :])
	assert np.allclose(phi[-2, :, :], phi_new[-2, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, 1, :], phi_new[:, 1, :])
	assert np.allclose(phi[:, 2, :], phi_new[:, 2, :])
	assert np.allclose(phi[:, -3, :], phi_new[:, -3, :])
	assert np.allclose(phi[:, -2, :], phi_new[:, -2, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def third_order_assert_xz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[3:-3, :, :], phi_new[3:-3, :, :])
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[1, :, :], phi_new[1, :, :])
	assert np.allclose(phi[2, :, :], phi_new[2, :, :])
	assert np.allclose(phi[-3, :, :], phi_new[-3, :, :])
	assert np.allclose(phi[-2, :, :], phi_new[-2, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])


def third_order_assert_yz(phi, phi_new, phi_new_assert):
	assert np.allclose(phi_new_assert[:, 3:-3, :], phi_new[:, 3:-3, :])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, 1, :], phi_new[:, 1, :])
	assert np.allclose(phi[:, 2, :], phi_new[:, 2, :])
	assert np.allclose(phi[:, -3, :], phi_new[:, -3, :])
	assert np.allclose(phi[:, -2, :], phi_new[:, -2, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def third_order_validation(phi_rnd, ni, nj, nk, grid, smooth_depth):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_new = np.zeros_like(phi)

	hs = HS.factory(
		'third_order', (ni, nj, nk), grid, smooth_depth, 0.5, 1.0, dtype=phi.dtype
	)
	hs(phi, phi_new)

	if ni < 7:
		phi_new_assert = third_order_smoothing_yz(phi, hs._gamma)
		third_order_assert_yz(phi, phi_new, phi_new_assert)
	elif nj < 7:
		phi_new_assert = third_order_smoothing_xz(phi, hs._gamma)
		third_order_assert_xz(phi, phi_new, phi_new_assert)
	else:
		phi_new_assert = third_order_smoothing_xyz(phi, hs._gamma)
		third_order_assert_xyz(phi, phi_new, phi_new_assert)


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
	#test_first_order()

