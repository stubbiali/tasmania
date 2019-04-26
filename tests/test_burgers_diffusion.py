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
	assume, given, HealthCheck, reproduce_failure, seed, settings, strategies as hyp_st
import pytest
from sympl import DataArray

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils
import test_horizontal_diffusion as thd

from tasmania.python.burgers.physics.diffusion import BurgersHorizontalDiffusion


def second_order_validation(grid, smooth_coeff, phi, phi_tnd, nb):
	nx, ny = grid.nx, grid.ny
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	if nx < 3:
		phi_tnd_assert = smooth_coeff * thd.second_order_diffusion_yz(dy, phi)
		thd.assert_yz(phi_tnd, phi_tnd_assert, nb)
	elif ny < 3:
		phi_tnd_assert = smooth_coeff * thd.second_order_diffusion_xz(dx, phi)
		thd.assert_xz(phi_tnd, phi_tnd_assert, nb)
	else:
		phi_tnd_assert = smooth_coeff * thd.second_order_diffusion_xyz(dx, dy, phi)
		thd.assert_xyz(phi_tnd, phi_tnd_assert, nb)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_second_order(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 1  # TODO: nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, taz_conf.nb)))

	domain = data.draw(
		utils.st_domain(
			xaxis_length=(1, 15),
			yaxis_length=(1, 15),
			zaxis_length=(1, 1),
			nb=nb
		),
		label='domain'
	)

	pgrid = domain.physical_grid
	cgrid = domain.numerical_grid
	assume(
		(pgrid.nx == 1 and pgrid.ny >= 3) or
		(pgrid.nx >= 3 and pgrid.ny == 1) or
		(pgrid.nx >= 3 and pgrid.ny >= 3)
	)

	pstate = data.draw(utils.st_burgers_state(pgrid), label='pgrid')
	cstate = data.draw(utils.st_burgers_state(cgrid), label='cgrid')

	smooth_coeff = data.draw(utils.st_floats(min_value=0, max_value=1), label='smooth_coeff')

	backend = data.draw(utils.st_one_of(conf.backend), label='backend')

	# ========================================
	# test
	# ========================================
	#
	# physical grid
	#
	pbhd = BurgersHorizontalDiffusion(
		domain, 'physical', 'second_order',
		DataArray(smooth_coeff, attrs={'units': 'm^2 s^-1'}),
		backend=backend, dtype=pgrid.x.dtype
	)

	tendencies, diagnostics = pbhd(pstate)

	assert len(diagnostics) == 0

	assert 'x_velocity' in tendencies
	assert 'y_velocity' in tendencies
	assert len(tendencies) == 2

	assert tendencies['x_velocity'].attrs['units'] == 'm s^-2'
	second_order_validation(
		pgrid, smooth_coeff, pstate['x_velocity'].to_units('m s^-1').values,
		tendencies['x_velocity'].values, nb
	)

	assert tendencies['y_velocity'].attrs['units'] == 'm s^-2'
	second_order_validation(
		pgrid, smooth_coeff, pstate['y_velocity'].to_units('m s^-1').values,
		tendencies['y_velocity'].values, nb
	)

	#
	# numerical grid
	#
	cbhd = BurgersHorizontalDiffusion(
		domain, 'numerical', 'second_order',
		DataArray(smooth_coeff, attrs={'units': 'm^2 s^-1'}),
		backend=backend, dtype=cgrid.x.dtype
	)

	tendencies, diagnostics = cbhd(cstate)

	assert len(diagnostics) == 0

	assert 'x_velocity' in tendencies
	assert 'y_velocity' in tendencies
	assert len(tendencies) == 2

	assert tendencies['x_velocity'].attrs['units'] == 'm s^-2'
	second_order_validation(
		cgrid, smooth_coeff, cstate['x_velocity'].to_units('m s^-1').values,
		tendencies['x_velocity'].values, nb
	)

	assert tendencies['y_velocity'].attrs['units'] == 'm s^-2'
	second_order_validation(
		cgrid, smooth_coeff, cstate['y_velocity'].to_units('m s^-1').values,
		tendencies['y_velocity'].values, nb
	)


def fourth_order_validation(grid, smooth_coeff, phi, phi_tnd, nb):
	nx, ny = grid.nx, grid.ny
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	if nx < 5:
		phi_tnd_assert = smooth_coeff * thd.fourth_order_diffusion_yz(dy, phi)
		thd.assert_yz(phi_tnd, phi_tnd_assert, nb)
	elif ny < 5:
		phi_tnd_assert = smooth_coeff * thd.fourth_order_diffusion_xz(dx, phi)
		thd.assert_xz(phi_tnd, phi_tnd_assert, nb)
	else:
		phi_tnd_assert = smooth_coeff * thd.fourth_order_diffusion_xyz(dx, dy, phi)
		thd.assert_xyz(phi_tnd, phi_tnd_assert, nb)


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_fourth_order(data):
	# ========================================
	# random data generation
	# ========================================
	nb = 2  # TODO: nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, taz_conf.nb)))

	domain = data.draw(
		utils.st_domain(
			xaxis_length=(1, 15),
			yaxis_length=(1, 15),
			zaxis_length=(1, 1),
			nb=nb
		),
		label='domain'
	)

	pgrid = domain.physical_grid
	cgrid = domain.numerical_grid
	assume(
		(pgrid.nx == 1 and pgrid.ny >= 5) or
		(pgrid.nx >= 5 and pgrid.ny == 1) or
		(pgrid.nx >= 5 and pgrid.ny >= 5)
	)

	pstate = data.draw(utils.st_burgers_state(pgrid), label='pgrid')
	cstate = data.draw(utils.st_burgers_state(cgrid), label='cgrid')

	smooth_coeff = data.draw(utils.st_floats(min_value=0, max_value=1), label='smooth_coeff')

	backend = data.draw(utils.st_one_of(conf.backend), label='backend')

	# ========================================
	# test
	# ========================================
	#
	# physical grid
	#
	pbhd = BurgersHorizontalDiffusion(
		domain, 'physical', 'fourth_order',
		DataArray(smooth_coeff, attrs={'units': 'm^2 s^-1'}),
		backend=backend, dtype=pgrid.x.dtype
	)

	tendencies, diagnostics = pbhd(pstate)

	assert len(diagnostics) == 0

	assert 'x_velocity' in tendencies
	assert 'y_velocity' in tendencies
	assert len(tendencies) == 2

	assert tendencies['x_velocity'].attrs['units'] == 'm s^-2'
	fourth_order_validation(
		pgrid, smooth_coeff, pstate['x_velocity'].to_units('m s^-1').values,
		tendencies['x_velocity'].values, nb
	)

	assert tendencies['y_velocity'].attrs['units'] == 'm s^-2'
	fourth_order_validation(
		pgrid, smooth_coeff, pstate['y_velocity'].to_units('m s^-1').values,
		tendencies['y_velocity'].values, nb
	)

	#
	# numerical grid
	#
	cbhd = BurgersHorizontalDiffusion(
		domain, 'numerical', 'fourth_order',
		DataArray(smooth_coeff, attrs={'units': 'm^2 s^-1'}),
		backend=backend, dtype=cgrid.x.dtype
	)

	tendencies, diagnostics = cbhd(cstate)

	assert len(diagnostics) == 0

	assert 'x_velocity' in tendencies
	assert 'y_velocity' in tendencies
	assert len(tendencies) == 2

	assert tendencies['x_velocity'].attrs['units'] == 'm s^-2'
	fourth_order_validation(
		cgrid, smooth_coeff, cstate['x_velocity'].to_units('m s^-1').values,
		tendencies['x_velocity'].values, nb
	)

	assert tendencies['y_velocity'].attrs['units'] == 'm s^-2'
	fourth_order_validation(
		cgrid, smooth_coeff, cstate['y_velocity'].to_units('m s^-1').values,
		tendencies['y_velocity'].values, nb
	)


if __name__ == '__main__':
	pytest.main([__file__])
