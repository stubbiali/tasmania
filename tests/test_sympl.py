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
import numpy as np
import pytest


def test_import():
	try:
		import sympl
		assert True
	except ImportError:
		print('Hint: did you install sympl?')
		assert False

	import sys
	assert 'sympl' in sys.modules


def test_to_units():
	import sympl
	from tasmania.namelist import datatype

	domain_x, nx, dims_x, units_x = [-50, 50], 101, 'x', 'km'

	xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=datatype) if nx == 1 \
		 else np.linspace(domain_x[0], domain_x[1], nx, dtype=datatype)
	x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x', attrs={'units': units_x})

	x_to_units = x.to_units('m')

	assert x_to_units[0] == -50.e3
	assert x_to_units[-1] == 50.e3


def test_constants():
	from sympl import get_constant
	assert get_constant('gravitational_acceleration', 'm s^-2') == 9.80665

	from sympl import set_constant
	set_constant('beta', 42., 'K m^-1')
	assert get_constant('beta', 'K m^-1') == 42.0

	from sympl import reset_constants
	reset_constants()


def test_diagnostics(isentropic_dry_data):
	from sympl import DiagnosticComponent

	class TestDiagnostic(DiagnosticComponent):
		def __init__(self, grid):
			self._grid = grid
			super().__init__()

		@property
		def input_properties(self):
			dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

			return {'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'}}

		@property
		def diagnostic_properties(self):
			dims = (self._grid.x.dims[0], self._grid.z.dims[0])

			return {'maxy_air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'}}

		def array_call(self, state):
			maxy = np.max(state['air_isentropic_density'], axis=1)
			return {'maxy_air_isentropic_density': maxy}

	grid, states = isentropic_dry_data
	state = states[-1]
	grid.update_topography(state['time'] - states[0]['time'])

	td = TestDiagnostic(grid)
	diagnostics = td(state)
	maxy = diagnostics['maxy_air_isentropic_density']

	assert len(maxy.shape) == 2
	assert maxy.shape[0] == grid.nx
	assert maxy.shape[1] == grid.nz


if __name__ == '__main__':
	pytest.main([__file__])
