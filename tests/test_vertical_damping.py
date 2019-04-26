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
from pandas import Timedelta
import pytest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

from tasmania.python.dwarfs.vertical_damping import VerticalDamping as VD


def assert_rayleigh(
	ni, nj, nk, grid, depth, backend, dt, phi_now, phi_new, phi_ref, phi_out
):
	__phi_now = phi_now[:ni, :nj, :nk]
	__phi_new = phi_new[:ni, :nj, :nk]
	__phi_ref = phi_ref[:ni, :nj, :nk]
	__phi_out = phi_out[:ni, :nj, :nk]
	dtype = phi_now.dtype

	vd = VD.factory(
		'rayleigh', (ni, nj, nk), grid, depth, 0.01,
		time_units='s', backend=backend, dtype=dtype
	)

	rmat = vd._rmat

	vd(dt, __phi_now, __phi_new, __phi_ref, __phi_out)

	phi_val = __phi_new - dt.total_seconds() * rmat * (__phi_now - __phi_ref)
	assert np.allclose(__phi_out[:, :, :depth], phi_val[:, :, :depth])
	assert np.allclose(__phi_out[:, :, depth:], __phi_new[:, :, depth:])


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test_rayleigh(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(
		utils.st_domain(
			xaxis_length=(1, 30),
			yaxis_length=(1, 30),
			zaxis_length=(1, 30),
		),
		label='grid'
	)

	cgrid = domain.numerical_grid

	phi_now = data.draw(
		st_arrays(
			cgrid.x.dtype, (cgrid.nx+1, cgrid.ny+1, cgrid.nz+1),
			elements=utils.st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='phi_now'
	)
	phi_new = data.draw(
		st_arrays(
			cgrid.x.dtype, (cgrid.nx+1, cgrid.ny+1, cgrid.nz+1),
			elements=utils.st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='phi_new'
	)
	phi_ref = data.draw(
		st_arrays(
			cgrid.x.dtype, (cgrid.nx+1, cgrid.ny+1, cgrid.nz+1),
			elements=utils.st_floats(min_value=-1e10, max_value=1e10),
			fill=hyp_st.nothing(),
		),
		label='phi_ref'
	)

	dt = data.draw(
		utils.st_timedeltas(min_value=Timedelta(seconds=0), max_value=Timedelta(hours=1)),
		label='dt'
	)

	depth = data.draw(hyp_st.integers(min_value=0, max_value=cgrid.nz), label='depth')

	backend = data.draw(utils.st_one_of(conf.backend))

	# ========================================
	# test
	# ========================================
	phi_out = np.zeros_like(phi_now, dtype=phi_now.dtype)
	nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz

	assert_rayleigh(
		nx  , ny  , nz  , cgrid, depth, backend,
		dt, phi_now, phi_new, phi_ref, phi_out
	)
	assert_rayleigh(
		nx+1, ny  , nz  , cgrid, depth, backend,
		dt, phi_now, phi_new, phi_ref, phi_out
	)
	assert_rayleigh(
		nx  , ny+1, nz  , cgrid, depth, backend,
		dt, phi_now, phi_new, phi_ref, phi_out
	)
	assert_rayleigh(
		nx  , ny  , nz+1, cgrid, depth, backend,
		dt, phi_now, phi_new, phi_ref, phi_out
	)
	assert_rayleigh(
		nx+1, ny+1, nz  , cgrid, depth, backend,
		dt, phi_now, phi_new, phi_ref, phi_out
	)
	assert_rayleigh(
		nx+1, ny  , nz+1, cgrid, depth, backend,
		dt, phi_now, phi_new, phi_ref, phi_out
	)
	assert_rayleigh(
		nx  , ny+1, nz+1, cgrid, depth, backend,
		dt, phi_now, phi_new, phi_ref, phi_out
	)
	assert_rayleigh(
		nx+1, ny+1, nz+1, cgrid, depth, backend,
		dt, phi_now, phi_new, phi_ref, phi_out
	)


if __name__ == '__main__':
	pytest.main([__file__])
