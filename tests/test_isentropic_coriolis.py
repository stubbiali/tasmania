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
from sympl import DataArray

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

from tasmania.python.isentropic.physics.coriolis import IsentropicConservativeCoriolis
from tasmania.python.utils.data_utils import make_dataarray_3d


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much
	),
	deadline=None
)
@given(hyp_st.data())
def test_conservative(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(utils.st_domain(), label="domain")
	grid_type = data.draw(utils.st_one_of(('physical', 'numerical')), label="grid_type")
	grid = domain.physical_grid if grid_type == 'physical' else domain.numerical_grid
	f = data.draw(utils.st_floats(min_value=0, max_value=1), label="f")
	time = data.draw(hyp_st.datetimes(), label="time")
	field = data.draw(
		st_arrays(
			grid.x.dtype, (grid.nx, grid.ny, grid.nz+1),
			elements=utils.st_floats(),
			fill=hyp_st.nothing(),
		)
	)
	su_units = data.draw(
		utils.st_one_of(('kg m^-1 K^-1 s^-1', 'g km^-1 K^-1 hr^-1')), label="su_units"
	)
	sv_units = data.draw(
		utils.st_one_of(('kg m^-1 K^-1 s^-1', 'g km^-1 K^-1 hr^-1')), label="sv_units"
	)
	backend = data.draw((utils.st_one_of(conf.backend)), label="backend")

	# ========================================
	# test bed
	# ========================================
	nb = domain.horizontal_boundary.nb if grid_type == 'numerical' else 0
	x, y = slice(nb, grid.nx-nb), slice(nb, grid.ny-nb)
	coriolis_parameter = DataArray(f, attrs={'units': 'rad s^-1'})
	state = {
		'time': time,
		'x_momentum_isentropic':
			make_dataarray_3d(field[:, :, :-1], grid, su_units),
		'y_momentum_isentropic':
			make_dataarray_3d(field[:, :, 1:], grid, sv_units),
	}

	icc = IsentropicConservativeCoriolis(
		domain, grid_type, coriolis_parameter, backend, grid.x.dtype
	)

	assert 'x_momentum_isentropic' in icc.input_properties
	assert 'y_momentum_isentropic' in icc.input_properties

	assert 'x_momentum_isentropic' in icc.tendency_properties
	assert 'y_momentum_isentropic' in icc.tendency_properties

	assert icc.diagnostic_properties == {}

	tendencies, diagnostics = icc(state)

	su_val = make_dataarray_3d(
		f * state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values,
		grid, 'kg m^-1 K^-1 s^-2'
	)
	assert 'x_momentum_isentropic' in tendencies
	utils.compare_dataarrays(tendencies['x_momentum_isentropic'][x, y], su_val[x, y])

	sv_val = make_dataarray_3d(
		- f * state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values,
		grid, 'kg m^-1 K^-1 s^-2'
	)
	assert 'y_momentum_isentropic' in tendencies
	utils.compare_dataarrays(tendencies['y_momentum_isentropic'][x, y], sv_val[x, y])

	assert len(tendencies) == 2

	assert len(diagnostics) == 0


if __name__ == '__main__':
	pytest.main([__file__])
