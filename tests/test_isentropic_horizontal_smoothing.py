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

from tasmania.python.isentropic.physics.horizontal_smoothing import \
	IsentropicHorizontalSmoothing
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.utils.data_utils import make_dataarray_3d

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .utils import compare_dataarrays, st_domain, st_floats, st_one_of, \
		st_isentropic_state_f
except (ImportError, ModuleNotFoundError):
	from conf import backend as conf_backend  # nb as conf_nb
	from utils import compare_dataarrays, st_domain, st_floats, st_one_of, \
		st_isentropic_state_f


mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


@settings(
	suppress_health_check=(
		HealthCheck.too_slow,
		HealthCheck.data_too_large,
		HealthCheck.filter_too_much,
	),
	deadline=None
)
@given(hyp_st.data())
def test(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(
		st_domain(xaxis_length=(7, 40), yaxis_length=(7, 40)), label='domain'
	)

	grid = domain.numerical_grid
	dtype = grid.x.dtype

	state = data.draw(st_isentropic_state_f(grid, moist=True), label='state')

	smooth_coeff = data.draw(st_floats(min_value=0, max_value=1))
	smooth_coeff_max = data.draw(st_floats(min_value=smooth_coeff, max_value=1))
	smooth_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))
	smooth_moist_coeff = data.draw(st_floats(min_value=0, max_value=1))
	smooth_moist_coeff_max = data.draw(st_floats(min_value=smooth_moist_coeff, max_value=1))
	smooth_moist_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	nb = domain.horizontal_boundary.nb

	smooth_types = ('first_order', 'second_order', 'third_order')

	for smooth_type in smooth_types:
		#
		# validation data
		#
		hs = HorizontalSmoothing.factory(
			smooth_type, (nx, ny, nz), smooth_coeff, smooth_coeff_max,
			smooth_damp_depth, nb, backend, dtype
		)
		hs_moist = HorizontalSmoothing.factory(
			smooth_type, (nx, ny, nz), smooth_moist_coeff, smooth_moist_coeff_max,
			smooth_moist_damp_depth, nb, backend, dtype
		)

		val = {}

		names = (
			'air_isentropic_density',
			'x_momentum_isentropic',
			'y_momentum_isentropic',
		)
		units = (
			'kg m^-2 K^-1',
			'kg m^-1 K^-1 s^-1',
			'kg m^-1 K^-1 s^-1',
		)
		for i in range(len(names)):
			field = state[names[i]].to_units(units[i]).values
			field_val = np.zeros_like(field, dtype=dtype)
			hs(field, field_val)
			val[names[i]] = field_val

		names = (mfwv, mfcw, mfpw)
		units = ('g g^-1',) * 3
		for i in range(len(names)):
			field = state[names[i]].to_units(units[i]).values
			field_val = np.zeros_like(field, dtype=dtype)
			hs_moist(field, field_val)
			val[names[i]] = field_val

		#
		# dry
		#
		ihs = IsentropicHorizontalSmoothing(
			domain, smooth_type,
			smooth_coeff=smooth_coeff, smooth_coeff_max=smooth_coeff_max,
			smooth_damp_depth=smooth_damp_depth,
			backend=backend, dtype=dtype
		)

		diagnostics = ihs(state)

		names = (
			'air_isentropic_density',
			'x_momentum_isentropic',
			'y_momentum_isentropic',
		)
		units = (
			'kg m^-2 K^-1',
			'kg m^-1 K^-1 s^-1',
			'kg m^-1 K^-1 s^-1',
		)
		for i in range(len(names)):
			assert names[i] in diagnostics
			field_val = make_dataarray_3d(val[names[i]], grid, units[i], name=names[i])
			compare_dataarrays(diagnostics[names[i]], field_val)

		assert len(diagnostics) == len(names)

		#
		# moist
		#
		ihs = IsentropicHorizontalSmoothing(
			domain, smooth_type,
			smooth_coeff=smooth_coeff, smooth_coeff_max=smooth_coeff_max,
			smooth_damp_depth=smooth_damp_depth,
			moist=True,
			smooth_moist_coeff=smooth_moist_coeff, smooth_moist_coeff_max=smooth_moist_coeff_max,
			smooth_moist_damp_depth=smooth_moist_damp_depth,
			backend=backend, dtype=dtype
		)

		diagnostics = ihs(state)

		names = (
			'air_isentropic_density',
			'x_momentum_isentropic',
			'y_momentum_isentropic',
			mfwv,
			mfcw,
			mfpw,
		)
		units = (
			'kg m^-2 K^-1',
			'kg m^-1 K^-1 s^-1',
			'kg m^-1 K^-1 s^-1',
			'g g^-1',
			'g g^-1',
			'g g^-1',
		)
		for i in range(len(names)):
			assert names[i] in diagnostics
			field_val = make_dataarray_3d(val[names[i]], grid, units[i], name=names[i])
			compare_dataarrays(diagnostics[names[i]], field_val)

		assert len(diagnostics) == len(names)


if __name__ == '__main__':
	pytest.main([__file__])
