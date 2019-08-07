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
from sympl import DataArray

from tasmania.python.isentropic.physics.horizontal_diffusion import \
	IsentropicHorizontalDiffusion
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion
from tasmania.python.utils.data_utils import make_dataarray_3d

try:
	from .conf import backend as conf_backend  # nb as conf_nb
	from .utils import st_domain, st_floats, st_isentropic_state_f, st_one_of, \
		compare_dataarrays
except ModuleNotFoundError:
	from conf import backend as conf_backend  # nb as conf_nb
	from utils import st_domain, st_floats, st_isentropic_state_f, st_one_of, \
		compare_dataarrays


__tracers = {
	'tracer0': {'units': 'g g^-1'},
	'tracer1': {'units': 'g g^-1'},
	'tracer2': {'units': 'g kg^-1'},
	'tracer3': {'units': 'kg^-1'}
}


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
		st_domain(xaxis_length=(5, 40), yaxis_length=(5, 40)), label='domain'
	)

	grid = domain.numerical_grid
	dtype = grid.x.dtype

	q_on = {
		name: data.draw(hyp_st.booleans(), label=name+'_on')
		for name in __tracers
	}
	tracers = {name: __tracers[name] for name in __tracers if q_on[name]}
	state = data.draw(st_isentropic_state_f(grid, tracers=tracers), label='state')

	diff_coeff = data.draw(st_floats(min_value=0, max_value=1))
	diff_coeff_max = data.draw(st_floats(min_value=diff_coeff, max_value=1))
	diff_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))
	diff_tracer_coeff = data.draw(st_floats(min_value=0, max_value=1))
	diff_tracer_coeff_max = data.draw(st_floats(min_value=diff_tracer_coeff, max_value=1))
	diff_tracer_damp_depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz))

	backend = data.draw(st_one_of(conf_backend))

	# ========================================
	# test bed
	# ========================================
	nx, ny, nz = grid.nx, grid.ny, grid.nz
	dx = grid.dx.to_units('m').values.item()
	dy = grid.dy.to_units('m').values.item()
	nb = domain.horizontal_boundary.nb

	diff_types = ('second_order', 'fourth_order')

	for diff_type in diff_types:
		#
		# validation data
		#
		hd = HorizontalDiffusion.factory(
			diff_type, (nx, ny, nz), dx, dy, diff_coeff, diff_coeff_max,
			diff_damp_depth, nb, backend, dtype
		)
		hd_tracer = HorizontalDiffusion.factory(
			diff_type, (nx, ny, nz), dx, dy, diff_tracer_coeff, diff_tracer_coeff_max,
			diff_tracer_damp_depth, nb, backend, dtype
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
			hd(field, field_val)
			val[names[i]] = field_val

		names = tuple(name for name in tracers)
		units = tuple(tracers[name]['units'] for name in tracers)
		for i in range(len(names)):
			field = state[names[i]].to_units(units[i]).values
			field_val = np.zeros_like(field, dtype=dtype)
			hd_tracer(field, field_val)
			val[names[i]] = field_val

		#
		# dry
		#
		ihd = IsentropicHorizontalDiffusion(
			domain, diff_type,
			diffusion_coeff=DataArray(diff_coeff, attrs={'units': 's^-1'}),
			diffusion_coeff_max=DataArray(diff_coeff_max, attrs={'units': 's^-1'}),
			diffusion_damp_depth=diff_damp_depth,
			tracers=tracers, backend=backend, dtype=dtype
		)

		tendencies, diagnostics = ihd(state)

		names = [
			'air_isentropic_density',
			'x_momentum_isentropic',
			'y_momentum_isentropic',
		]
		units = [
			'kg m^-2 K^-1 s^-1',
			'kg m^-1 K^-1 s^-2',
			'kg m^-1 K^-1 s^-2',
		]
		for i in range(len(names)):
			assert names[i] in tendencies
			field_val = make_dataarray_3d(val[names[i]], grid, units[i], name=names[i])
			compare_dataarrays(tendencies[names[i]], field_val)

		assert len(tendencies) == len(names)

		assert len(diagnostics) == 0

		#
		# moist
		#
		ihd = IsentropicHorizontalDiffusion(
			domain, diff_type,
			diffusion_coeff=DataArray(diff_coeff, attrs={'units': 's^-1'}),
			diffusion_coeff_max=DataArray(diff_coeff_max, attrs={'units': 's^-1'}),
			diffusion_damp_depth=diff_damp_depth,
			tracers=tracers,
			diffusion_tracer_coeff=DataArray(diff_tracer_coeff, attrs={'units': 's^-1'}),
			diffusion_tracer_coeff_max=DataArray(diff_tracer_coeff_max, attrs={'units': 's^-1'}),
			diffusion_tracer_damp_depth=diff_tracer_damp_depth,
			backend=backend, dtype=dtype
		)

		tendencies, diagnostics = ihd(state)

		names += [name for name in tracers]
		units += [tracers[name]['units'] + ' s^-1' for name in tracers]
		for i in range(len(names)):
			assert names[i] in tendencies
			field_val = make_dataarray_3d(val[names[i]], grid, units[i], name=names[i])
			compare_dataarrays(tendencies[names[i]], field_val)

		assert len(tendencies) == len(names)

		assert len(diagnostics) == 0


if __name__ == '__main__':
	pytest.main([__file__])
