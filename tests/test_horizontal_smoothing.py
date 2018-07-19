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

from tasmania.dynamics.horizontal_smoothing import HorizontalSmoothing as HS


def test_first_order(grid):
	nx, ny, nz = grid.nx, grid.ny, grid.nz

	ni, nj, nk = nx+1, ny, nz+1
	phi = np.random.rand(ni, nj, nk)
	phi_new = np.zeros_like(phi)

	hs = HS.factory('first_order', (ni, nj, nk), grid,
					15, 0.03, 0.24, dtype=phi.dtype)
	hs(phi, phi_new)

	g = hs._gamma
	i, j, k = slice(1, ni-1), slice(1, nj-1), slice(0, nk)
	im1, ip1 = slice(0, ni-2), slice(2, ni)
	jm1, jp1 = slice(0, nj-2), slice(2, nj)
	phi_new_assert = np.zeros((ni, nj, nk), dtype=phi_new.dtype)
	phi_new_assert[i, j, k] = (1 - 4. * g[i, j, k]) * phi[i, j, k] + \
					   		  g[i, j, k] * (phi[ip1, j, k] + phi[im1, j, k] +
									 		phi[i, jp1, k] + phi[i, jm1, k])
	assert np.allclose(phi_new_assert[i, j, k], phi_new[i, j, k])
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def test_first_order_xz(grid_xz):
	nx, ny, nz = grid_xz.nx, grid_xz.ny, grid_xz.nz

	ni, nj, nk = nx, ny, nz+1
	phi = np.random.rand(ni, nj, nk)
	phi_new = np.zeros_like(phi)

	hs = HS.factory('first_order', (ni, nj, nk), grid_xz,
					15, 0.03, 0.48, dtype=phi.dtype)
	hs(phi, phi_new)

	g = hs._gamma
	i, k = slice(1, ni-1), slice(0, nk)
	im1, ip1 = slice(0, ni-2), slice(2, ni)
	phi_new_assert = np.zeros((ni, nj, nk), dtype=phi_new.dtype)
	phi_new_assert[i, 0, k] = (1 - 2. * g[i, 0, k]) * phi[i, 0, k] + \
							  g[i, 0, k] * (phi[ip1, 0, k] + phi[im1, 0, k])
	assert np.allclose(phi_new_assert[i, 0, k], phi_new[i, 0, k])
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])


def test_first_order_yz(grid_yz):
	nx, ny, nz = grid_yz.nx, grid_yz.ny, grid_yz.nz

	ni, nj, nk = nx, ny+1, nz
	phi = np.random.rand(ni, nj, nk)
	phi_new = np.zeros_like(phi)

	hs = HS.factory('first_order', (ni, nj, nk), grid_yz,
					10, 0.1, 0.48, dtype=phi.dtype)
	hs(phi, phi_new)

	g = hs._gamma
	j, k = slice(1, nj-1), slice(0, nk)
	jm1, jp1 = slice(0, nj-2), slice(2, nj)
	phi_new_assert = np.zeros((ni, nj, nk), dtype=phi_new.dtype)
	phi_new_assert[0, j, k] = (1 - 2. * g[0, j, k]) * phi[0, j, k] + \
							  g[0, j, k] * (phi[0, jp1, k] + phi[0, jm1, k])
	assert np.allclose(phi_new_assert[0, j, k], phi_new[0, j, k])
	assert np.allclose(phi[:,  0, :], phi_new[:,  0, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def test_second_order(grid):
	nx, ny, nz = grid.nx, grid.ny, grid.nz

	ni, nj, nk = nx+1, ny, nz+1
	phi = np.random.rand(ni, nj, nk)
	phi_new = np.zeros_like(phi)

	hs = HS.factory('second_order', (ni, nj, nk), grid,
					15, 0.03, 0.24, dtype=phi.dtype)
	hs(phi, phi_new)

	g = hs._gamma
	i, j, k = slice(2, ni-2), slice(2, nj-2), slice(0, nk)
	im1, ip1 = slice(1, ni-3), slice(3, ni-1)
	im2, ip2 = slice(0, ni-4), slice(4, ni)
	jm1, jp1 = slice(1, nj-3), slice(3, nj-1)
	jm2, jp2 = slice(0, nj-4), slice(4, nj)
	phi_new_assert = np.zeros((ni, nj, nk), dtype=phi_new.dtype)
	phi_new_assert[i, j, k] = (1 - 12. * g[i, j, k]) * phi[i, j, k] + \
							  g[i, j, k] * (- (phi[im2, j, k] + phi[ip2, j, k])
											+ 4. * (phi[im1, j, k] + phi[ip1, j, k])
											+ 4. * (phi[i, jm1, k] + phi[i, jp1, k])
											- (phi[i, jm2, k] + phi[i, jp2, k]))
	assert np.allclose(phi_new_assert[i, j, k], phi_new[i, j, k], atol=1e-7)
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[1, :, :], phi_new[1, :, :])
	assert np.allclose(phi[-2, :, :], phi_new[-2, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, 1, :], phi_new[:, 1, :])
	assert np.allclose(phi[:, -2, :], phi_new[:, -2, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


def test_second_order_xz(grid_xz):
	nx, ny, nz = grid_xz.nx, grid_xz.ny, grid_xz.nz

	ni, nj, nk = nx, ny, nz+1
	phi = np.random.rand(ni, nj, nk)
	phi_new = np.zeros_like(phi)

	hs = HS.factory('second_order', (ni, nj, nk), grid_xz,
					15, 0.03, 0.48, dtype=phi.dtype)
	hs(phi, phi_new)

	g = hs._gamma
	i, k = slice(2, ni-2), slice(0, nk)
	im1, ip1 = slice(1, ni-3), slice(3, ni-1)
	im2, ip2 = slice(0, ni-4), slice(4, ni)
	phi_new_assert = np.zeros((ni, nj, nk), dtype=phi_new.dtype)
	phi_new_assert[i, 0, k] = (1 - 6. * g[i, 0, k]) * phi[i, 0, k] + \
							  g[i, 0, k] * (- phi[im2, 0, k]
											+ 4. * phi[im1, 0, k]
											+ 4. * phi[ip1, 0, k]
											- phi[ip2, 0, k])
	assert np.allclose(phi_new_assert[i, 0, k], phi_new[i, 0, k], atol=1e-7)
	assert np.allclose(phi[0, :, :], phi_new[0, :, :])
	assert np.allclose(phi[1, :, :], phi_new[1, :, :])
	assert np.allclose(phi[-2, :, :], phi_new[-2, :, :])
	assert np.allclose(phi[-1, :, :], phi_new[-1, :, :])


def test_second_order_yz(grid_yz):
	nx, ny, nz = grid_yz.nx, grid_yz.ny, grid_yz.nz

	ni, nj, nk = nx, ny+1, nz
	phi = np.random.rand(ni, nj, nk)
	phi_new = np.zeros_like(phi)

	hs = HS.factory('second_order', (ni, nj, nk), grid_yz,
					10, 0.1, 0.48, dtype=phi.dtype)
	hs(phi, phi_new)

	g = hs._gamma
	j, k = slice(2, nj-2), slice(0, nk)
	jm1, jp1 = slice(1, nj-3), slice(3, nj-1)
	jm2, jp2 = slice(0, nj-4), slice(4, nj)
	phi_new_assert = np.zeros((ni, nj, nk), dtype=phi_new.dtype)
	phi_new_assert[0, j, k] = (1. - 6. * g[0, j, k]) * phi[0, j, k] + \
							  g[0, j, k] * (- phi[0, jm2, k]
											+ 4. * phi[0, jm1, k]
											+ 4. * phi[0, jp1, k]
											- phi[0, jp2, k])
	assert np.allclose(phi_new_assert[0, j, k], phi_new[0, j, k])
	assert np.allclose(phi[:, 0, :], phi_new[:, 0, :])
	assert np.allclose(phi[:, 1, :], phi_new[:, 1, :])
	assert np.allclose(phi[:, -2, :], phi_new[:, -2, :])
	assert np.allclose(phi[:, -1, :], phi_new[:, -1, :])


if __name__ == '__main__':
	pytest.main([__file__])

