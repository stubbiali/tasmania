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
import numpy as np
import pytest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

from tasmania.python.isentropic.physics.pressure_gradient import \
	IsentropicNonconservativePressureGradient, \
	IsentropicConservativePressureGradient
from tasmania.python.utils.data_utils import make_dataarray_3d


def pressure_thickness_weighted_validation(dx, dy, eps, p, mtg):
	u_tnd = np.zeros_like(mtg, dtype=mtg.dtype)
	v_tnd = np.zeros_like(mtg, dtype=mtg.dtype)

	dp = p[:, :, 1:] - p[:, :, :-1]
	dpx = 0.5 * (dp[:-1, :] + dp[1:, :])
	dpy = 0.5 * (dp[:, :-1] + dp[:, 1:])

	wgtx_c = np.minimum(eps, np.minimum(dpx[1:-2, 2:-2], dpx[2:-1, 2:-2])) / eps
	wgtx_w = np.minimum(eps, np.minimum(dpx[:-3, 2:-2], dpx[1:-2, 2:-2]))
	wgtx_e = np.minimum(eps, np.minimum(dpx[2:-1, 2:-2], dpx[3:, 2:-2]))
	wgtx_s = np.minimum(eps, np.minimum(dpx[1:-2, 1:-3], dpx[2:-1, 1:-3]))
	wgtx_n = np.minimum(eps, np.minimum(dpx[1:-2, 3:-1], dpx[2:-1, 3:-1]))

	pgx_c = (mtg[3:-1, 2:-2] - mtg[1:-3, 2:-2]) / (2.0 * dx)
	pgx_w = (mtg[2:-2, 2:-2] - mtg[:-4, 2:-2]) / (2.0 * dx)
	pgx_e = (mtg[4:, 2:-2] - mtg[2:-2, 2:-2]) / (2.0 * dx)
	pgx_s = (mtg[3:-1, 1:-3] - mtg[1:-3, 1:-3]) / (2.0 * dx)
	pgx_n = (mtg[3:-1, 3:-1] - mtg[1:-3, 3:-1]) / (2.0 * dx)
	pgx_avg = \
		(wgtx_w * pgx_w + wgtx_e * pgx_e + wgtx_s * pgx_s + wgtx_n * pgx_n) / \
		(wgtx_w + wgtx_e + wgtx_s + wgtx_n)

	wgty_c = np.minimum(eps, np.minimum(dpy[2:-2, 1:-2], dpy[2:-2, 2:-1])) / eps
	wgty_w = np.minimum(eps, np.minimum(dpy[1:-3, 1:-2], dpy[1:-3, 2:-1]))
	wgty_e = np.minimum(eps, np.minimum(dpy[3:-1, 1:-2], dpy[3:-1, 2:-1]))
	wgty_s = np.minimum(eps, np.minimum(dpy[2:-2, :-3], dpy[2:-2, 1:-2]))
	wgty_n = np.minimum(eps, np.minimum(dpy[2:-2, 2:-1], dpy[2:-2, 3:]))

	pgy_c = (mtg[2:-2, 3:-1] - mtg[2:-2, 1:-3]) / (2.0 * dy)
	pgy_w = (mtg[1:-3, 3:-1] - mtg[1:-3, 1:-3]) / (2.0 * dy)
	pgy_e = (mtg[3:-1, 3:-1] - mtg[3:-1, 1:-3]) / (2.0 * dy)
	pgy_s = (mtg[2:-2, 2:-2] - mtg[2:-2, :-4]) / (2.0 * dy)
	pgy_n = (mtg[2:-2, 4:] - mtg[2:-2, 2:-2]) / (2.0 * dy)
	pgy_avg = \
		(wgty_w * pgy_w + wgty_e * pgy_e + wgty_s * pgy_s + wgty_n * pgy_n) / \
		(wgty_w + wgty_e + wgty_s + wgty_n)

	u_tnd[2:-2, 2:-2] = - (pgx_avg + wgtx_c * (pgx_c - pgx_avg))
	v_tnd[2:-2, 2:-2] = - (pgy_avg + wgty_c * (pgy_c - pgy_avg))

	return u_tnd, v_tnd


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_nonconservative(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(
		utils.st_domain(xaxis_length=(5, 40), yaxis_length=(5, 40)), label='domain'
	)

	grid = domain.numerical_grid
	dtype = grid.x.dtype

	state = data.draw(utils.st_isentropic_state(grid, moist=False), label='state')

	backend = data.draw(utils.st_one_of(conf.backend))

	# ========================================
	# test bed
	# ========================================
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	p = state['air_pressure_on_interface_levels'].to_units('Pa').values
	mtg = state['montgomery_potential'].to_units('m^2 s^-2').values
	u_tnd = np.zeros((nx, ny, nz), dtype=dtype)
	v_tnd = np.zeros((nx, ny, nz), dtype=dtype)

	#
	# second order
	#
	pg = IsentropicNonconservativePressureGradient(
		domain, 'second_order', backend=backend, dtype=dtype
	)

	tendencies, diagnostics = pg(state)

	nb = 1  # TODO: nb = 1 if domain.horizontal_boundary.nb < 1 else domain.horizontal_boundary.nb
	u_tnd[1:-1, 1:-1, :] = - (mtg[2:, 1:-1, :] - mtg[:-2, 1:-1, :]) / (2. * dx)
	u_tnd_val = make_dataarray_3d(u_tnd, grid, 'm s^-2', name='x_velocity')
	assert 'x_velocity' in tendencies.keys()
	assert np.allclose(
		tendencies['x_velocity'][nb:-nb, nb:-nb, :],
		u_tnd_val[nb:-nb, nb:-nb, :]
	)

	nb = 1  # TODO: nb = 1 if domain.horizontal_boundary.nb < 1 else domain.horizontal_boundary.nb
	v_tnd[1:-1, 1:-1, :] = - (mtg[1:-1, 2:, :] - mtg[1:-1, :-2, :]) / (2. * dy)
	v_tnd_val = make_dataarray_3d(v_tnd, grid, 'm s^-2', name='y_velocity')
	assert 'y_velocity' in tendencies.keys()
	assert np.allclose(
		tendencies['y_velocity'][nb:-nb, nb:-nb, :],
		v_tnd_val[nb:-nb, nb:-nb, :]
	)

	assert len(tendencies) == 2

	assert len(diagnostics) == 0

	#
	# fourth order
	#
	pg = IsentropicNonconservativePressureGradient(
		domain, 'fourth_order', backend=backend, dtype=dtype
	)

	tendencies, diagnostics = pg(state)

	nb = 2  # TODO: nb = 2 if domain.horizontal_boundary.nb < 2 else domain.horizontal_boundary.nb
	u_tnd[2:-2, 2:-2, :] = - (
		mtg[:-4, 2:-2, :] - 8. * mtg[1:-3, 2:-2, :] +
		8. * mtg[3:-1, 2:-2, :] - mtg[4:, 2:-2, :]
	) / (12. * dx)
	u_tnd_val = make_dataarray_3d(u_tnd, grid, 'm s^-2', name='x_velocity')
	assert 'x_velocity' in tendencies.keys()
	assert np.allclose(
		tendencies['x_velocity'][nb:-nb, nb:-nb, :],
		u_tnd_val[nb:-nb, nb:-nb, :]
	)

	nb = 2  # TODO: nb = 2 if domain.horizontal_boundary.nb < 2 else domain.horizontal_boundary.nb
	v_tnd[2:-2, 2:-2, :] = - (
		mtg[2:-2, :-4, :] - 8. * mtg[2:-2, 1:-3, :] +
		8. * mtg[2:-2, 3:-1, :] - mtg[2:-2, 4:, :]
	) / (12. * dy)
	v_tnd_val = make_dataarray_3d(v_tnd, grid, 'm s^-2', name='y_velocity')
	assert 'y_velocity' in tendencies.keys()
	assert np.allclose(
		tendencies['y_velocity'][nb:-nb, nb:-nb, :],
		v_tnd_val[nb:-nb, nb:-nb, :]
	)

	assert len(tendencies) == 2

	assert len(diagnostics) == 0

	#
	# weighted
	#
	pg = IsentropicNonconservativePressureGradient(
		domain, 'pressure_thickness_weighted', backend=backend, dtype=dtype
	)

	tendencies, diagnostics = pg(state)

	nb = 2  # TODO: nb = 2 if domain.horizontal_boundary.nb < 2 else domain.horizontal_boundary.nb
	u_tnd_val, v_tnd_val = pressure_thickness_weighted_validation(dx, dy, pg._eps, p, mtg)
	u_tnd_val = make_dataarray_3d(u_tnd_val, grid, 'm s^-2')
	v_tnd_val = make_dataarray_3d(v_tnd_val, grid, 'm s^-2')

	assert 'x_velocity' in tendencies
	utils.compare_dataarrays(
		tendencies['x_velocity'][nb:-nb, nb:-nb], u_tnd_val[nb:-nb, nb:-nb],
		compare_coordinate_values=False
	)

	assert 'y_velocity' in tendencies
	utils.compare_dataarrays(
		tendencies['y_velocity'][nb:-nb, nb:-nb], v_tnd_val[nb:-nb, nb:-nb],
		compare_coordinate_values=False
	)

	assert len(tendencies) == 2

	assert len(diagnostics) == 0


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_conservative(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(
		utils.st_domain(xaxis_length=(5, 40), yaxis_length=(5, 40)), label='domain'
	)

	grid = domain.numerical_grid
	dtype = grid.x.dtype

	state = data.draw(utils.st_isentropic_state(grid, moist=False), label='state')

	backend = data.draw(utils.st_one_of(conf.backend))

	# ========================================
	# test bed
	# ========================================
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	s = state['air_isentropic_density'].to_units('kg m^-2 K^-1').values
	p = state['air_pressure_on_interface_levels'].to_units('Pa').values
	mtg = state['montgomery_potential'].to_units('m^2 s^-2').values
	su_tnd = np.zeros((nx, ny, nz), dtype=dtype)
	sv_tnd = np.zeros((nx, ny, nz), dtype=dtype)

	#
	# second order
	#
	pg = IsentropicConservativePressureGradient(
		domain, 'second_order', backend=backend, dtype=dtype
	)

	tendencies, diagnostics = pg(state)

	nb = 1  # TODO: nb = 1 if domain.horizontal_boundary.nb < 1 else domain.horizontal_boundary.nb
	su_tnd[1:-1, 1:-1, :] = - s[1:-1, 1:-1, :] * \
		(mtg[2:, 1:-1, :] - mtg[:-2, 1:-1, :]) / (2. * dx)
	su_tnd_val = make_dataarray_3d(
		su_tnd, grid, 'kg m^-1 K^-1 s^-2', name='x_momentum_isentropic'
	)
	assert 'x_momentum_isentropic' in tendencies.keys()
	assert np.allclose(
		tendencies['x_momentum_isentropic'][nb:-nb, nb:-nb, :],
		su_tnd_val[nb:-nb, nb:-nb, :]
	)

	nb = 1  # TODO: nb = 1 if domain.horizontal_boundary.nb < 1 else domain.horizontal_boundary.nb
	sv_tnd[1:-1, 1:-1, :] = - s[1:-1, 1:-1, :] * \
		(mtg[1:-1, 2:, :] - mtg[1:-1, :-2, :]) / (2. * dy)
	sv_tnd_val = make_dataarray_3d(
		sv_tnd, grid, 'kg m^-1 K^-1 s^-2', name='y_momentum_isentropic'
	)
	assert 'y_momentum_isentropic' in tendencies.keys()
	assert np.allclose(
		tendencies['y_momentum_isentropic'][nb:-nb, nb:-nb, :],
		sv_tnd_val[nb:-nb, nb:-nb, :]
	)

	assert len(tendencies) == 2

	assert len(diagnostics) == 0

	#
	# fourth order
	#
	pg = IsentropicConservativePressureGradient(
		domain, 'fourth_order', backend=backend, dtype=dtype
	)

	tendencies, diagnostics = pg(state)

	nb = 2  # TODO: nb = 2 if domain.horizontal_boundary.nb < 2 else domain.horizontal_boundary.nb
	su_tnd[2:-2, 2:-2, :] = - s[2:-2, 2:-2, :] * (
		mtg[:-4, 2:-2, :] - 8. * mtg[1:-3, 2:-2, :] +
		8. * mtg[3:-1, 2:-2, :] - mtg[4:, 2:-2, :]
	) / (12. * dx)
	su_tnd_val = make_dataarray_3d(
		su_tnd, grid, 'kg m^-1 K^-1 s^-2', name='x_momentum_isentropic'
	)
	assert 'x_momentum_isentropic' in tendencies.keys()
	assert np.allclose(
		tendencies['x_momentum_isentropic'][nb:-nb, nb:-nb, :],
		su_tnd_val[nb:-nb, nb:-nb, :]
	)

	nb = 2  # TODO: nb = 2 if domain.horizontal_boundary.nb < 2 else domain.horizontal_boundary.nb
	sv_tnd[2:-2, 2:-2, :] = - s[2:-2, 2:-2, :] * (
		mtg[2:-2, :-4, :] - 8. * mtg[2:-2, 1:-3, :] +
		8. * mtg[2:-2, 3:-1, :] - mtg[2:-2, 4:, :]
	) / (12. * dy)
	sv_tnd_val = make_dataarray_3d(
		sv_tnd, grid, 'kg m^-1 K^-1 s^-2', name='y_momentum_isentropic'
	)
	assert 'y_momentum_isentropic' in tendencies.keys()
	assert np.allclose(
		tendencies['y_momentum_isentropic'][nb:-nb, nb:-nb, :],
		sv_tnd_val[nb:-nb, nb:-nb, :]
	)

	assert len(tendencies) == 2

	assert len(diagnostics) == 0

	#
	# weighted
	#
	pg = IsentropicConservativePressureGradient(
		domain, 'pressure_thickness_weighted', backend=backend, dtype=dtype
	)

	tendencies, diagnostics = pg(state)

	nb = 2  # TODO: nb = 2 if domain.horizontal_boundary.nb < 2 else domain.horizontal_boundary.nb
	u_tnd_val, v_tnd_val = pressure_thickness_weighted_validation(dx, dy, pg._eps, p, mtg)
	su_tnd_val = make_dataarray_3d(s * u_tnd_val, grid, 'kg m^-1 K^-1 s^-2')
	sv_tnd_val = make_dataarray_3d(s * v_tnd_val, grid, 'kg m^-1 K^-1 s^-2')

	assert 'x_momentum_isentropic' in tendencies
	utils.compare_dataarrays(
		tendencies['x_momentum_isentropic'][nb:-nb, nb:-nb], su_tnd_val[nb:-nb, nb:-nb],
		compare_coordinate_values=False
	)

	assert 'y_momentum_isentropic' in tendencies
	utils.compare_dataarrays(
		tendencies['y_momentum_isentropic'][nb:-nb, nb:-nb], sv_tnd_val[nb:-nb, nb:-nb],
		compare_coordinate_values=False
	)

	assert len(tendencies) == 2

	assert len(diagnostics) == 0


if __name__ == '__main__':
	pytest.main([__file__])
