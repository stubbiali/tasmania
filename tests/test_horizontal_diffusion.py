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

from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion as HD


def second_order_laplacian_x(dx, phi):
	out = np.zeros_like(phi, phi.dtype)

	out[1:-1, :, :] = \
		(phi[2:, :, :] - 2.0*phi[1:-1, :, :] + phi[:-2, :, :]) / (dx*dx)

	return out


def second_order_laplacian_y(dy, phi):
	out = np.zeros_like(phi, phi.dtype)

	out[:, 1:-1, :] = \
		(phi[:, 2:, :] - 2.0*phi[:, 1:-1, :] + phi[:, :-2, :]) / (dy*dy)

	return out


def second_order_diffusion_xyz(dx, dy, phi):
	return second_order_laplacian_x(dx, phi) + second_order_laplacian_y(dy, phi)


def second_order_diffusion_xz(dx, phi):
	return second_order_laplacian_x(dx, phi)


def second_order_diffusion_yz(dy, phi):
	return second_order_laplacian_y(dy, phi)


def second_order_assert_xyz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[1:-1, 1:-1, :], phi_tnd[1:-1, 1:-1, :])


def second_order_assert_xz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[1:-1, :, :], phi_tnd[1:-1, :, :])


def second_order_assert_yz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[:, 1:-1, :], phi_tnd[:, 1:-1, :])


def second_order_validation(phi_rnd, ni, nj, nk, grid, diffusion_depth):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_tnd = np.zeros_like(phi)

	hd = HD.factory(
		'second_order', (ni, nj, nk), grid, diffusion_depth, 0.5, 1.0,
		xaxis_units='m', yaxis_units='m', dtype=phi.dtype
	)
	hd(phi, phi_tnd)

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	if ni < 3:
		phi_tnd_assert = hd._gamma * second_order_diffusion_yz(dy, phi)
		second_order_assert_yz(phi_tnd, phi_tnd_assert)
	elif nj < 3:
		phi_tnd_assert = hd._gamma * second_order_diffusion_xz(dx, phi)
		second_order_assert_xz(phi_tnd, phi_tnd_assert)
	else:
		phi_tnd_assert = hd._gamma * second_order_diffusion_xyz(dx, dy, phi)
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
	second_order_validation(phi_rnd, nx  , ny  , nz  , grid, depth)
	second_order_validation(phi_rnd, nx+1, ny  , nz  , grid, depth)
	second_order_validation(phi_rnd, nx  , ny+1, nz  , grid, depth)
	second_order_validation(phi_rnd, nx  , ny  , nz+1, grid, depth)
	second_order_validation(phi_rnd, nx+1, ny+1, nz  , grid, depth)
	second_order_validation(phi_rnd, nx+1, ny  , nz+1, grid, depth)
	second_order_validation(phi_rnd, nx  , ny+1, nz+1, grid, depth)
	second_order_validation(phi_rnd, nx+1, ny+1, nz+1, grid, depth)


def fourth_order_laplacian_x(dx, phi):
	out = np.zeros_like(phi, phi.dtype)

	out[2:-2, :, :] = (
		- phi[:-4, :, :] + 16.0*phi[1:-3, :, :] - 30.0*phi[2:-2, :, :]
		+ 16.0*phi[3:-1, :, :] - phi[4:, :, :]
	) / (12.0*dx*dx)

	return out


def fourth_order_laplacian_y(dy, phi):
	out = np.zeros_like(phi, phi.dtype)

	out[:, 2:-2, :] = (
		- phi[:, :-4, :] + 16.0*phi[:, 1:-3, :] - 30.0*phi[:, 2:-2, :]
		+ 16.0*phi[:, 3:-1, :] - phi[:, 4:, :]
	) / (12.0*dy*dy)

	return out


def fourth_order_diffusion_xyz(dx, dy, phi):
	return fourth_order_laplacian_x(dx, phi) + fourth_order_laplacian_y(dy, phi)


def fourth_order_diffusion_xz(dx, phi):
	return fourth_order_laplacian_x(dx, phi)


def fourth_order_diffusion_yz(dy, phi):
	return fourth_order_laplacian_y(dy, phi)


def fourth_order_assert_xyz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[2:-2, 2:-2, :], phi_tnd[2:-2, 2:-2, :])


def fourth_order_assert_xz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[2:-2, :, :], phi_tnd[2:-2, :, :])


def fourth_order_assert_yz(phi_tnd, phi_tnd_assert):
	assert np.allclose(phi_tnd_assert[:, 2:-2, :], phi_tnd[:, 2:-2, :])


def fourth_order_validation(phi_rnd, ni, nj, nk, grid, diffusion_depth):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_tnd = np.zeros_like(phi)

	hd = HD.factory(
		'fourth_order', (ni, nj, nk), grid, diffusion_depth, 0.5, 1.0,
		xaxis_units='m', yaxis_units='m', dtype=phi.dtype
	)
	hd(phi, phi_tnd)

	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	if ni < 5:
		phi_tnd_assert = hd._gamma * fourth_order_diffusion_yz(dy, phi)
		fourth_order_assert_yz(phi_tnd, phi_tnd_assert)
	elif nj < 5:
		phi_tnd_assert = hd._gamma * fourth_order_diffusion_xz(dx, phi)
		fourth_order_assert_xz(phi_tnd, phi_tnd_assert)
	else:
		phi_tnd_assert = hd._gamma * fourth_order_diffusion_xyz(dx, dy, phi)
		fourth_order_assert_xyz(phi_tnd, phi_tnd_assert)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_fourth_order(data):
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
	fourth_order_validation(phi_rnd, nx  , ny  , nz  , grid, depth)
	fourth_order_validation(phi_rnd, nx+1, ny  , nz  , grid, depth)
	fourth_order_validation(phi_rnd, nx  , ny+1, nz  , grid, depth)
	fourth_order_validation(phi_rnd, nx  , ny  , nz+1, grid, depth)
	fourth_order_validation(phi_rnd, nx+1, ny+1, nz  , grid, depth)
	fourth_order_validation(phi_rnd, nx+1, ny  , nz+1, grid, depth)
	fourth_order_validation(phi_rnd, nx  , ny+1, nz+1, grid, depth)
	fourth_order_validation(phi_rnd, nx+1, ny+1, nz+1, grid, depth)


if __name__ == '__main__':
	pytest.main([__file__])
	#test_third_order()
