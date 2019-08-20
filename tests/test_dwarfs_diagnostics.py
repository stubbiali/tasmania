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

from tasmania.python.dwarfs.diagnostics import HorizontalVelocity, WaterConstituent

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .utils import st_floats, st_one_of, st_physical_grid
except (ImportError, ModuleNotFoundError):
	from conf import backend as conf_backend  # nb as conf_nb
	from utils import st_floats, st_one_of, st_physical_grid


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_horizontal_velocity_staggered(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(st_physical_grid(), label='grid')
	dtype = grid.x.dtype

	r = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label='r'
	)
	u = data.draw(
		st_arrays(
			dtype, (grid.nx+1, grid.ny, grid.nz),
			elements=st_floats(min_value=-1e3, max_value=1e3),
			fill=hyp_st.nothing(),
		),
		label='u'
	)
	v = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny+1, grid.nz),
			elements=st_floats(min_value=-1e3, max_value=1e3),
			fill=hyp_st.nothing(),
		),
		label='v'
	)

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	hv = HorizontalVelocity(grid, True, backend, dtype)

	ru = np.zeros_like(r, dtype=dtype)
	rv = np.zeros_like(r, dtype=dtype)

	hv.get_momenta(r, u, v, ru, rv)

	ru_val = r * 0.5 * (u[:-1, :] + u[1:, :])
	assert np.allclose(ru, ru_val)
	rv_val = r * 0.5 * (v[:, :-1] + v[:, 1:])
	assert np.allclose(rv, rv_val)

	u_new = np.zeros_like(u, dtype=dtype)
	v_new = np.zeros_like(v, dtype=dtype)

	hv.get_velocity_components(r, ru, rv, u_new, v_new)

	u_new_val = np.zeros_like(u, dtype=dtype)
	u_new_val[1:-1, :] = (ru[:-1, :] + ru[1:, :]) / (r[:-1, :] + r[1:, :])
	assert np.allclose(u_new[1:-1, :], u_new_val[1:-1, :])
	v_new_val = np.zeros_like(v, dtype=dtype)
	v_new_val[:, 1:-1] = (rv[:, :-1] + rv[:, 1:]) / (r[:, :-1] + r[:, 1:])

	assert np.allclose(v_new[:, 1:-1], v_new_val[:, 1:-1])


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_horizontal_velocity(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(st_physical_grid(), label='grid')
	dtype = grid.x.dtype

	r = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label='r'
	)
	u = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=-1e3, max_value=1e3),
			fill=hyp_st.nothing(),
		),
		label='u'
	)
	v = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=-1e3, max_value=1e3),
			fill=hyp_st.nothing(),
		),
		label='v'
	)

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	hv = HorizontalVelocity(grid, False, backend, dtype)

	ru = np.zeros_like(r, dtype=dtype)
	rv = np.zeros_like(r, dtype=dtype)

	hv.get_momenta(r, u, v, ru, rv)

	ru_val = r * u
	assert np.allclose(ru, ru_val)
	rv_val = r * v
	assert np.allclose(rv, rv_val)

	u_new = np.zeros_like(u, dtype=dtype)
	v_new = np.zeros_like(v, dtype=dtype)

	hv.get_velocity_components(r, ru, rv, u_new, v_new)

	assert np.allclose(u_new, u)
	assert np.allclose(v_new, v)


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_water_constituent(data):
	# ========================================
	# random data generation
	# ========================================
	grid = data.draw(st_physical_grid(), label='grid')
	dtype = grid.x.dtype

	r = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=1, max_value=1e4),
			fill=hyp_st.nothing(),
		),
		label='r'
	)
	q = data.draw(
		st_arrays(
			dtype, (grid.nx, grid.ny, grid.nz),
			elements=st_floats(min_value=-1e3, max_value=1e3),
			fill=hyp_st.nothing(),
		),
		label='u'
	)

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	rq = np.zeros_like(q, dtype=dtype)
	q_new = np.zeros_like(q, dtype=dtype)

	wc = WaterConstituent(grid, backend, dtype)

	wc.get_density_of_water_constituent(r, q, rq, clipping=False)
	rq_val = r * q
	assert np.allclose(rq, rq_val)

	wc.get_mass_fraction_of_water_constituent_in_air(r, rq, q_new, clipping=False)
	q_new_val = rq / r
	assert np.allclose(q_new, q_new_val)

	wc.get_mass_fraction_of_water_constituent_in_air(r, rq, q_new, clipping=True)
	rq[rq < 0.] = 0.
	q_new_val = rq / r
	assert np.allclose(q_new, q_new_val)

	wc.get_density_of_water_constituent(r, q, rq, clipping=True)
	rq_val = r * q
	rq_val[rq_val < 0.] = 0.
	assert np.allclose(rq, rq_val)


if __name__ == '__main__':
	pytest.main([__file__])
