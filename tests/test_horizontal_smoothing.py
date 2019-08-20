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

from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing as HS

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .utils import compare_arrays, st_domain, st_floats, st_one_of
except (ImportError, ModuleNotFoundError):
	from conf import backend as conf_backend  # nb as conf_nb
	from utils import compare_arrays, st_domain, st_floats, st_one_of


def assert_xyz(phi, phi_new, phi_new_assert, nb):
	compare_arrays(phi_new_assert[nb:-nb, nb:-nb, :], phi_new[nb:-nb, nb:-nb, :])
	compare_arrays(phi[:nb, :, :], phi_new[:nb, :, :])
	compare_arrays(phi[-nb:, :, :], phi_new[-nb:, :, :])
	compare_arrays(phi[nb:-nb, :nb, :], phi_new[nb:-nb, :nb, :])
	compare_arrays(phi[nb:-nb, -nb:, :], phi_new[nb:-nb, -nb:, :])


def assert_xz(phi, phi_new, phi_new_assert, nb):
	compare_arrays(phi_new_assert[nb:-nb, :, :], phi_new[nb:-nb, :, :])
	compare_arrays(phi[:nb, :, :], phi_new[:nb, :, :])
	compare_arrays(phi[-nb:, :, :], phi_new[-nb:, :, :])


def assert_yz(phi, phi_new, phi_new_assert, nb):
	compare_arrays(phi_new_assert[:, nb:-nb, :], phi_new[:, nb:-nb, :])
	compare_arrays(phi[:, :nb, :], phi_new[:, :nb, :])
	compare_arrays(phi[:, -nb:, :], phi_new[:, -nb:, :])


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


def first_order_validation(phi_rnd, ni, nj, nk, smooth_depth, nb, backend):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_new = np.zeros_like(phi)

	hs = HS.factory(
		'first_order', (ni, nj, nk), 0.5, 1.0, smooth_depth,
		nb, backend, dtype=phi.dtype
	)
	hs(phi, phi_new)

	gamma = hs._gamma if isinstance(hs._gamma, np.ndarray) else \
		hs._gamma * np.ones_like(phi)

	if ni < 3:
		phi_new_assert = first_order_smoothing_yz(phi, gamma)
		assert_yz(phi, phi_new, phi_new_assert, nb)
	elif nj < 3:
		phi_new_assert = first_order_smoothing_xz(phi, gamma)
		assert_xz(phi, phi_new, phi_new_assert, nb)
	else:
		phi_new_assert = first_order_smoothing_xyz(phi, gamma)
		assert_xyz(phi, phi_new, phi_new_assert, nb)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_first_order(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 1  # TODO: nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, taz_conf.nb)))

	domain = data.draw(
		st_domain(
			xaxis_length=(1, 30),
			yaxis_length=(1, 30),
			zaxis_length=(1, 30),
			nb=nb
		),
		label='grid'
	)

	pgrid = domain.physical_grid
	cgrid = domain.numerical_grid
	assume(
		(pgrid.nx == 1 and pgrid.ny >= 3) or
		(pgrid.nx >= 3 and pgrid.ny == 1) or
		(pgrid.nx >= 3 and pgrid.ny >= 3)
	)

	pphi_rnd = data.draw(
		st_arrays(
			pgrid.x.dtype, (pgrid.nx+1, pgrid.ny+1, pgrid.nz+1),
			elements=st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='pphi_rnd'
	)
	cphi_rnd = data.draw(
		st_arrays(
			cgrid.x.dtype, (cgrid.nx+1, cgrid.ny+1, cgrid.nz+1),
			elements=st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='cphi_rnd'
	)

	depth = data.draw(hyp_st.integers(min_value=0, max_value=pgrid.nz), label='depth')

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test
	# ========================================
	nx, ny, nz = pgrid.nx, pgrid.ny, pgrid.nz
	first_order_validation(pphi_rnd, nx  , ny  , nz  , depth, nb, backend)
	first_order_validation(pphi_rnd, nx+1, ny  , nz  , depth, nb, backend)
	first_order_validation(pphi_rnd, nx  , ny+1, nz  , depth, nb, backend)
	first_order_validation(pphi_rnd, nx  , ny  , nz+1, depth, nb, backend)
	first_order_validation(pphi_rnd, nx+1, ny+1, nz  , depth, nb, backend)
	first_order_validation(pphi_rnd, nx+1, ny  , nz+1, depth, nb, backend)
	first_order_validation(pphi_rnd, nx  , ny+1, nz+1, depth, nb, backend)
	first_order_validation(pphi_rnd, nx+1, ny+1, nz+1, depth, nb, backend)

	nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
	first_order_validation(cphi_rnd, nx  , ny  , nz  , depth, nb, backend)
	first_order_validation(cphi_rnd, nx+1, ny  , nz  , depth, nb, backend)
	first_order_validation(cphi_rnd, nx  , ny+1, nz  , depth, nb, backend)
	first_order_validation(cphi_rnd, nx  , ny  , nz+1, depth, nb, backend)
	first_order_validation(cphi_rnd, nx+1, ny+1, nz  , depth, nb, backend)
	first_order_validation(cphi_rnd, nx+1, ny  , nz+1, depth, nb, backend)
	first_order_validation(cphi_rnd, nx  , ny+1, nz+1, depth, nb, backend)
	first_order_validation(cphi_rnd, nx+1, ny+1, nz+1, depth, nb, backend)


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


def second_order_validation(phi_rnd, ni, nj, nk, smooth_depth, nb, backend):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_new = np.zeros_like(phi)

	hs = HS.factory(
		'second_order', (ni, nj, nk), 0.5, 1.0, smooth_depth,
		nb, backend, dtype=phi.dtype
	)
	hs(phi, phi_new)

	gamma = hs._gamma if isinstance(hs._gamma, np.ndarray) else \
		hs._gamma * np.ones_like(phi)

	if ni < 5:
		phi_new_assert = second_order_smoothing_yz(phi, gamma)
		assert_yz(phi, phi_new, phi_new_assert, nb)
	elif nj < 5:
		phi_new_assert = second_order_smoothing_xz(phi, gamma)
		assert_xz(phi, phi_new, phi_new_assert, nb)
	else:
		phi_new_assert = second_order_smoothing_xyz(phi, gamma)
		assert_xyz(phi, phi_new, phi_new_assert, nb)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_second_order(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 2  # TODO: nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, taz_conf.nb)))

	domain = data.draw(
		st_domain(
			xaxis_length=(1, 30),
			yaxis_length=(1, 30),
			zaxis_length=(1, 30),
			nb=nb
		),
		label='grid'
	)

	pgrid = domain.physical_grid
	cgrid = domain.numerical_grid
	assume(
		(pgrid.nx == 1 and pgrid.ny >= 5) or
		(pgrid.nx >= 5 and pgrid.ny == 1) or
		(pgrid.nx >= 5 and pgrid.ny >= 5)
	)

	pphi_rnd = data.draw(
		st_arrays(
			pgrid.x.dtype, (pgrid.nx+1, pgrid.ny+1, pgrid.nz+1),
			elements=st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='pphi_rnd'
	)
	cphi_rnd = data.draw(
		st_arrays(
			cgrid.x.dtype, (cgrid.nx+1, cgrid.ny+1, cgrid.nz+1),
			elements=st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='cphi_rnd'
	)

	depth = data.draw(hyp_st.integers(min_value=0, max_value=pgrid.nz), label='depth')

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test
	# ========================================
	nx, ny, nz = pgrid.nx, pgrid.ny, pgrid.nz
	second_order_validation(pphi_rnd, nx  , ny  , nz  , depth, nb, backend)
	second_order_validation(pphi_rnd, nx+1, ny  , nz  , depth, nb, backend)
	second_order_validation(pphi_rnd, nx  , ny+1, nz  , depth, nb, backend)
	second_order_validation(pphi_rnd, nx  , ny  , nz+1, depth, nb, backend)
	second_order_validation(pphi_rnd, nx+1, ny+1, nz  , depth, nb, backend)
	second_order_validation(pphi_rnd, nx+1, ny  , nz+1, depth, nb, backend)
	second_order_validation(pphi_rnd, nx  , ny+1, nz+1, depth, nb, backend)
	second_order_validation(pphi_rnd, nx+1, ny+1, nz+1, depth, nb, backend)

	nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
	second_order_validation(cphi_rnd, nx  , ny  , nz  , depth, nb, backend)
	second_order_validation(cphi_rnd, nx+1, ny  , nz  , depth, nb, backend)
	second_order_validation(cphi_rnd, nx  , ny+1, nz  , depth, nb, backend)
	second_order_validation(cphi_rnd, nx  , ny  , nz+1, depth, nb, backend)
	second_order_validation(cphi_rnd, nx+1, ny+1, nz  , depth, nb, backend)
	second_order_validation(cphi_rnd, nx+1, ny  , nz+1, depth, nb, backend)
	second_order_validation(cphi_rnd, nx  , ny+1, nz+1, depth, nb, backend)
	second_order_validation(cphi_rnd, nx+1, ny+1, nz+1, depth, nb, backend)


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


def third_order_validation(phi_rnd, ni, nj, nk, smooth_depth, nb, backend):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_new = np.zeros_like(phi)

	hs = HS.factory(
		'third_order', (ni, nj, nk), 0.5, 1.0, smooth_depth,
		nb, backend, dtype=phi.dtype
	)
	hs(phi, phi_new)

	gamma = hs._gamma if isinstance(hs._gamma, np.ndarray) else \
		hs._gamma * np.ones_like(phi)

	if ni < 7:
		phi_new_assert = third_order_smoothing_yz(phi, gamma)
		assert_yz(phi, phi_new, phi_new_assert, nb)
	elif nj < 7:
		phi_new_assert = third_order_smoothing_xz(phi, gamma)
		assert_xz(phi, phi_new, phi_new_assert, nb)
	else:
		phi_new_assert = third_order_smoothing_xyz(phi, gamma)
		assert_xyz(phi, phi_new, phi_new_assert, nb)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_third_order(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 3  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, taz_conf.nb)))

	domain = data.draw(
		st_domain(
			xaxis_length=(1, 30),
			yaxis_length=(1, 30),
			zaxis_length=(1, 30),
			nb=nb
		),
		label='grid'
	)

	pgrid = domain.physical_grid
	cgrid = domain.numerical_grid
	assume(
		(pgrid.nx == 1 and pgrid.ny >= 7) or
		(pgrid.nx >= 7 and pgrid.ny == 1) or
		(pgrid.nx >= 7 and pgrid.ny >= 7)
	)

	pphi_rnd = data.draw(
		st_arrays(
			pgrid.x.dtype, (pgrid.nx+1, pgrid.ny+1, pgrid.nz+1),
			elements=st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='pphi_rnd'
	)
	cphi_rnd = data.draw(
		st_arrays(
			cgrid.x.dtype, (cgrid.nx+1, cgrid.ny+1, cgrid.nz+1),
			elements=st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='cphi_rnd'
	)

	depth = data.draw(hyp_st.integers(min_value=0, max_value=pgrid.nz), label='depth')

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test
	# ========================================
	nx, ny, nz = pgrid.nx, pgrid.ny, pgrid.nz
	third_order_validation(pphi_rnd, nx  , ny  , nz  , depth, nb, backend)
	third_order_validation(pphi_rnd, nx+1, ny  , nz  , depth, nb, backend)
	third_order_validation(pphi_rnd, nx  , ny+1, nz  , depth, nb, backend)
	third_order_validation(pphi_rnd, nx  , ny  , nz+1, depth, nb, backend)
	third_order_validation(pphi_rnd, nx+1, ny+1, nz  , depth, nb, backend)
	third_order_validation(pphi_rnd, nx+1, ny  , nz+1, depth, nb, backend)
	third_order_validation(pphi_rnd, nx  , ny+1, nz+1, depth, nb, backend)
	third_order_validation(pphi_rnd, nx+1, ny+1, nz+1, depth, nb, backend)

	nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
	third_order_validation(cphi_rnd, nx  , ny  , nz  , depth, nb, backend)
	third_order_validation(cphi_rnd, nx+1, ny  , nz  , depth, nb, backend)
	third_order_validation(cphi_rnd, nx  , ny+1, nz  , depth, nb, backend)
	third_order_validation(cphi_rnd, nx  , ny  , nz+1, depth, nb, backend)
	third_order_validation(cphi_rnd, nx+1, ny+1, nz  , depth, nb, backend)
	third_order_validation(cphi_rnd, nx+1, ny  , nz+1, depth, nb, backend)
	third_order_validation(cphi_rnd, nx  , ny+1, nz+1, depth, nb, backend)
	third_order_validation(cphi_rnd, nx+1, ny+1, nz+1, depth, nb, backend)


if __name__ == '__main__':
	pytest.main([__file__])

