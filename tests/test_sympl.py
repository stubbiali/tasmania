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
	except ImportError:
		print('Hint: did you install sympl?')

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


if __name__ == '__main__':
	pytest.main([__file__])
