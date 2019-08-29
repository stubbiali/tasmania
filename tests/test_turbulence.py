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

from tasmania.python.physics.turbulence import Smagorinsky2d
from tasmania.python.utils.data_utils import make_dataarray_3d

try:
	from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
	from .utils import compare_dataarrays, st_domain, st_floats, st_one_of
except (ImportError, ModuleNotFoundError):
	from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
	from utils import compare_dataarrays, st_domain, st_floats, st_one_of


def smagorinsky2d_validation(dx, dy, cs, u, v):
	u_tnd = np.zeros_like(u, dtype=u.dtype)
	v_tnd = np.zeros_like(v, dtype=v.dtype)

	s00 = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
	s01 = 0.5 * (
		(u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy) +
		(v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
	)
	s11 = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
	nu = (cs**2) * (dx * dy) * (2.0 * s00**2 + 4.0 * s01**2 + 2.0 * s11**2)**0.5
	u_tnd[2:-2, 2:-2] = 2.0 * (
		(nu[2:, 1:-1] * s00[2:, 1:-1] - nu[:-2, 1:-1] * s00[:-2, 1:-1]) / (2.0 * dx) +
		(nu[1:-1, 2:] * s01[1:-1, 2:] - nu[1:-1, :-2] * s01[1:-1, :-2]) / (2.0 * dy)
	)
	v_tnd[2:-2, 2:-2] = 2.0 * (
		(nu[2:, 1:-1] * s01[2:, 1:-1] - nu[:-2, 1:-1] * s01[:-2, 1:-1]) / (2.0 * dx) +
		(nu[1:-1, 2:] * s11[1:-1, 2:] - nu[1:-1, :-2] * s11[1:-1, :-2]) / (2.0 * dy)
	)

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
def test_smagorinsky2d(data):
	# ========================================
	# random data generation
	# ========================================
	nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label='nb')

	domain = data.draw(st_domain(nb=nb), label='domain')
	grid = domain.numerical_grid

	cs = data.draw(hyp_st.floats(min_value=0, max_value=10), label='cs')

	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dtype = grid.x.dtype
	field = data.draw(
		st_arrays(
			dtype, (nx+1, ny+1, nz),
			elements=st_floats(min_value=-1e3, max_value=1e3),
			fill=hyp_st.nothing(),
		),
		label='field'
	)

	time = data.draw(hyp_st.datetimes(), label='time')

	backend = data.draw(st_one_of(conf_backend), label='backend')
	halo = data.draw(st_one_of(conf_halo), label='halo')

	# ========================================
	# test bed
	# ========================================
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()

	u = field[:-1, :-1, :]
	v = field[1:, 1:, :]
	state = {
		'time': time,
		'x_velocity': make_dataarray_3d(u, grid, 'm s^-1'),
		'y_velocity': make_dataarray_3d(v, grid, 'm s^-1')
	}

	u_tnd, v_tnd = smagorinsky2d_validation(dx, dy, cs, u, v)

	smag = Smagorinsky2d(
		domain, smagorinsky_constant=cs, backend=backend, dtype=dtype, halo=halo
	)

	tendencies, diagnostics = smag(state)

	assert 'x_velocity' in tendencies
	compare_dataarrays(
		tendencies['x_velocity'][nb:-nb, nb:-nb, :],
		make_dataarray_3d(u_tnd, grid, 'm s^-2')[nb:-nb, nb:-nb, :],
		compare_coordinate_values=False
	)
	assert 'y_velocity' in tendencies
	compare_dataarrays(
		tendencies['y_velocity'][nb:-nb, nb:-nb, :],
		make_dataarray_3d(v_tnd, grid, 'm s^-2')[nb:-nb, nb:-nb, :],
		compare_coordinate_values=False
	)
	assert len(tendencies) == 2

	assert len(diagnostics) == 0


if __name__ == '__main__':
	pytest.main([__file__])
