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

from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion as HD

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .utils import compare_arrays, st_domain, st_floats, st_one_of
except (ImportError, ModuleNotFoundError):
	from conf import backend as conf_backend  # nb as conf_nb
	from utils import compare_arrays, st_domain, st_floats, st_one_of


def assert_xyz(phi_tnd, phi_tnd_assert, nb):
	compare_arrays(phi_tnd_assert[nb:-nb, nb:-nb, :], phi_tnd[nb:-nb, nb:-nb, :])


def assert_xz(phi_tnd, phi_tnd_assert, nb):
	compare_arrays(phi_tnd_assert[nb:-nb, :, :], phi_tnd[nb:-nb, :, :])


def assert_yz(phi_tnd, phi_tnd_assert, nb):
	compare_arrays(phi_tnd_assert[:, nb:-nb, :], phi_tnd[:, nb:-nb, :])


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


def second_order_validation(phi_rnd, ni, nj, nk, grid, diffusion_depth, nb, backend):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_tnd = np.zeros_like(phi)

	dx = grid.dx.values.item()
	dy = grid.dy.values.item()

	hd = HD.factory(
		'second_order', (ni, nj, nk), dx, dy, 0.5, 1.0, diffusion_depth,
		nb=nb, backend=backend, dtype=phi.dtype
	)
	hd(phi, phi_tnd)

	if ni < 3:
		phi_tnd_assert = hd._gamma * second_order_diffusion_yz(dy, phi)
		assert_yz(phi_tnd, phi_tnd_assert, nb)
	elif nj < 3:
		phi_tnd_assert = hd._gamma * second_order_diffusion_xz(dx, phi)
		assert_xz(phi_tnd, phi_tnd_assert, nb)
	else:
		phi_tnd_assert = hd._gamma * second_order_diffusion_xyz(dx, dy, phi)
		assert_xyz(phi_tnd, phi_tnd_assert, nb)


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
	nb = 1  # TODO: nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, taz_conf.nb)))

	domain = data.draw(
		st_domain(
			xaxis_length=(1, 15),
			yaxis_length=(1, 15),
			zaxis_length=(1, 15),
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
	second_order_validation(pphi_rnd, nx  , ny  , nz  , pgrid, depth, nb, backend)
	second_order_validation(pphi_rnd, nx+1, ny  , nz  , pgrid, depth, nb, backend)
	second_order_validation(pphi_rnd, nx  , ny+1, nz  , pgrid, depth, nb, backend)
	second_order_validation(pphi_rnd, nx  , ny  , nz+1, pgrid, depth, nb, backend)
	second_order_validation(pphi_rnd, nx+1, ny+1, nz  , pgrid, depth, nb, backend)
	second_order_validation(pphi_rnd, nx+1, ny  , nz+1, pgrid, depth, nb, backend)
	second_order_validation(pphi_rnd, nx  , ny+1, nz+1, pgrid, depth, nb, backend)
	second_order_validation(pphi_rnd, nx+1, ny+1, nz+1, pgrid, depth, nb, backend)

	nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
	second_order_validation(cphi_rnd, nx  , ny  , nz  , cgrid, depth, nb, backend)
	second_order_validation(cphi_rnd, nx+1, ny  , nz  , cgrid, depth, nb, backend)
	second_order_validation(cphi_rnd, nx  , ny+1, nz  , cgrid, depth, nb, backend)
	second_order_validation(cphi_rnd, nx  , ny  , nz+1, cgrid, depth, nb, backend)
	second_order_validation(cphi_rnd, nx+1, ny+1, nz  , cgrid, depth, nb, backend)
	second_order_validation(cphi_rnd, nx+1, ny  , nz+1, cgrid, depth, nb, backend)
	second_order_validation(cphi_rnd, nx  , ny+1, nz+1, cgrid, depth, nb, backend)
	second_order_validation(cphi_rnd, nx+1, ny+1, nz+1, cgrid, depth, nb, backend)


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


def fourth_order_validation(phi_rnd, ni, nj, nk, grid, diffusion_depth, nb, backend):
	phi = phi_rnd[:ni, :nj, :nk]
	phi_tnd = np.zeros_like(phi)

	dx = grid.dx.values.item()
	dy = grid.dy.values.item()

	hd = HD.factory(
		'fourth_order', (ni, nj, nk), dx, dy, 0.5, 1.0, diffusion_depth,
		nb=nb, backend=backend, dtype=phi.dtype
	)
	hd(phi, phi_tnd)

	if ni < 5:
		phi_tnd_assert = hd._gamma * fourth_order_diffusion_yz(dy, phi)
		assert_yz(phi_tnd, phi_tnd_assert, nb)
	elif nj < 5:
		phi_tnd_assert = hd._gamma * fourth_order_diffusion_xz(dx, phi)
		assert_xz(phi_tnd, phi_tnd_assert, nb)
	else:
		phi_tnd_assert = hd._gamma * fourth_order_diffusion_xyz(dx, dy, phi)
		assert_xyz(phi_tnd, phi_tnd_assert, nb)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_fourth_order(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 2  # TODO: nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, taz_conf.nb)))

	domain = data.draw(
		st_domain(
			xaxis_length=(1, 15),
			yaxis_length=(1, 15),
			zaxis_length=(1, 15),
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
	fourth_order_validation(pphi_rnd, nx  , ny  , nz  , pgrid, depth, nb, backend)
	fourth_order_validation(pphi_rnd, nx+1, ny  , nz  , pgrid, depth, nb, backend)
	fourth_order_validation(pphi_rnd, nx  , ny+1, nz  , pgrid, depth, nb, backend)
	fourth_order_validation(pphi_rnd, nx  , ny  , nz+1, pgrid, depth, nb, backend)
	fourth_order_validation(pphi_rnd, nx+1, ny+1, nz  , pgrid, depth, nb, backend)
	fourth_order_validation(pphi_rnd, nx+1, ny  , nz+1, pgrid, depth, nb, backend)
	fourth_order_validation(pphi_rnd, nx  , ny+1, nz+1, pgrid, depth, nb, backend)
	fourth_order_validation(pphi_rnd, nx+1, ny+1, nz+1, pgrid, depth, nb, backend)

	nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
	fourth_order_validation(cphi_rnd, nx  , ny  , nz  , cgrid, depth, nb, backend)
	fourth_order_validation(cphi_rnd, nx+1, ny  , nz  , cgrid, depth, nb, backend)
	fourth_order_validation(cphi_rnd, nx  , ny+1, nz  , cgrid, depth, nb, backend)
	fourth_order_validation(cphi_rnd, nx  , ny  , nz+1, cgrid, depth, nb, backend)
	fourth_order_validation(cphi_rnd, nx+1, ny+1, nz  , cgrid, depth, nb, backend)
	fourth_order_validation(cphi_rnd, nx+1, ny  , nz+1, cgrid, depth, nb, backend)
	fourth_order_validation(cphi_rnd, nx  , ny+1, nz+1, cgrid, depth, nb, backend)
	fourth_order_validation(cphi_rnd, nx+1, ny+1, nz+1, cgrid, depth, nb, backend)


if __name__ == '__main__':
	pytest.main([__file__])
