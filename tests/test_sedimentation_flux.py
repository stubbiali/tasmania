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
import pytest

import gridtools as gt
from tasmania.python.physics.microphysics import SedimentationFlux


def test_first_order_upwind():
	k = gt.Index(axis=2)

	rho = gt.Equation()
	h   = gt.Equation()
	qr	= gt.Equation()
	vt  = gt.Equation()

	fluxer = SedimentationFlux.factory('first_order_upwind')

	flux = fluxer(k, rho, h, qr, vt)

	assert flux.get_name() == 'tmp_dfdz'


def test_second_order_upwind():
	k = gt.Index(axis=2)

	rho = gt.Equation()
	h   = gt.Equation()
	qr	= gt.Equation()
	vt  = gt.Equation()

	fluxer = SedimentationFlux.factory('second_order_upwind')

	flux = fluxer(k, rho, h, qr, vt)

	assert flux.get_name() == 'tmp_dfdz'


if __name__ == '__main__':
	pytest.main([__file__])