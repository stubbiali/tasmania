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


def test_upwind_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	mtg 	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc   	= gt.Equation()
	sqr    	= gt.Equation()
	s_tnd  	= gt.Equation()
	su_tnd 	= gt.Equation()
	sv_tnd 	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.python.isentropic.dynamics.fluxes import HorizontalIsentropicFlux

	fluxer = HorizontalIsentropicFlux.factory('upwind', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'flux_s_x'
	assert fluxes[1].get_name() == 'flux_s_y'
	assert fluxes[2].get_name() == 'flux_su_x'
	assert fluxes[3].get_name() == 'flux_su_y'
	assert fluxes[4].get_name() == 'flux_sv_x'
	assert fluxes[5].get_name() == 'flux_sv_y'

	fluxer = HorizontalIsentropicFlux.factory('upwind', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv, sqv, sqc, sqr,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd,
					qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'flux_s_x'
	assert fluxes[1].get_name()  == 'flux_s_y'
	assert fluxes[2].get_name()  == 'flux_su_x'
	assert fluxes[3].get_name()  == 'flux_su_y'
	assert fluxes[4].get_name()  == 'flux_sv_x'
	assert fluxes[5].get_name()  == 'flux_sv_y'
	assert fluxes[6].get_name()  == 'flux_sqv_x'
	assert fluxes[7].get_name()  == 'flux_sqv_y'
	assert fluxes[8].get_name()  == 'flux_sqc_x'
	assert fluxes[9].get_name()  == 'flux_sqc_y'
	assert fluxes[10].get_name() == 'flux_sqr_x'
	assert fluxes[11].get_name() == 'flux_sqr_y'


def test_centered_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	mtg 	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc 	= gt.Equation()
	sqr 	= gt.Equation()
	s_tnd  	= gt.Equation()
	su_tnd 	= gt.Equation()
	sv_tnd 	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.python.isentropic.dynamics.fluxes import HorizontalIsentropicFlux

	fluxer = HorizontalIsentropicFlux.factory('centered', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'flux_s_x'
	assert fluxes[1].get_name() == 'flux_s_y'
	assert fluxes[2].get_name() == 'flux_su_x'
	assert fluxes[3].get_name() == 'flux_su_y'
	assert fluxes[4].get_name() == 'flux_sv_x'
	assert fluxes[5].get_name() == 'flux_sv_y'

	fluxer = HorizontalIsentropicFlux.factory('centered', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv, sqv, sqc, sqr,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd,
					qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'flux_s_x'
	assert fluxes[1].get_name()  == 'flux_s_y'
	assert fluxes[2].get_name()  == 'flux_su_x'
	assert fluxes[3].get_name()  == 'flux_su_y'
	assert fluxes[4].get_name()  == 'flux_sv_x'
	assert fluxes[5].get_name()  == 'flux_sv_y'
	assert fluxes[6].get_name()  == 'flux_sqv_x'
	assert fluxes[7].get_name()  == 'flux_sqv_y'
	assert fluxes[8].get_name()  == 'flux_sqc_x'
	assert fluxes[9].get_name()  == 'flux_sqc_y'
	assert fluxes[10].get_name() == 'flux_sqr_x'
	assert fluxes[11].get_name() == 'flux_sqr_y'


def test_maccormack_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	mtg 	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc 	= gt.Equation()
	sqr 	= gt.Equation()
	s_tnd  	= gt.Equation()
	su_tnd 	= gt.Equation()
	sv_tnd 	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.python.isentropic.dynamics.fluxes import HorizontalIsentropicFlux

	fluxer = HorizontalIsentropicFlux.factory('maccormack', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'flux_s_x'
	assert fluxes[1].get_name() == 'flux_s_y'
	assert fluxes[2].get_name() == 'flux_su_x'
	assert fluxes[3].get_name() == 'flux_su_y'
	assert fluxes[4].get_name() == 'flux_sv_x'
	assert fluxes[5].get_name() == 'flux_sv_y'

	fluxer = HorizontalIsentropicFlux.factory('maccormack', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv, sqv, sqc, sqr,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd,
					qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'flux_s_x'
	assert fluxes[1].get_name()  == 'flux_s_y'
	assert fluxes[2].get_name()  == 'flux_su_x'
	assert fluxes[3].get_name()  == 'flux_su_y'
	assert fluxes[4].get_name()  == 'flux_sv_x'
	assert fluxes[5].get_name()  == 'flux_sv_y'
	assert fluxes[6].get_name()  == 'flux_sqv_x'
	assert fluxes[7].get_name()  == 'flux_sqv_y'
	assert fluxes[8].get_name()  == 'flux_sqc_x'
	assert fluxes[9].get_name()  == 'flux_sqc_y'
	assert fluxes[10].get_name() == 'flux_sqr_x'
	assert fluxes[11].get_name() == 'flux_sqr_y'


def test_third_order_upwind_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	mtg 	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc 	= gt.Equation()
	sqr 	= gt.Equation()
	s_tnd  	= gt.Equation()
	su_tnd 	= gt.Equation()
	sv_tnd 	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.python.isentropic.dynamics.fluxes import HorizontalIsentropicFlux

	fluxer = HorizontalIsentropicFlux.factory('third_order_upwind', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'third_order_flux_s_x'
	assert fluxes[1].get_name() == 'third_order_flux_s_y'
	assert fluxes[2].get_name() == 'third_order_flux_su_x'
	assert fluxes[3].get_name() == 'third_order_flux_su_y'
	assert fluxes[4].get_name() == 'third_order_flux_sv_x'
	assert fluxes[5].get_name() == 'third_order_flux_sv_y'

	fluxer = HorizontalIsentropicFlux.factory('third_order_upwind', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv, sqv, sqc, sqr,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd,
					qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'third_order_flux_s_x'
	assert fluxes[1].get_name()  == 'third_order_flux_s_y'
	assert fluxes[2].get_name()  == 'third_order_flux_su_x'
	assert fluxes[3].get_name()  == 'third_order_flux_su_y'
	assert fluxes[4].get_name()  == 'third_order_flux_sv_x'
	assert fluxes[5].get_name()  == 'third_order_flux_sv_y'
	assert fluxes[6].get_name()  == 'third_order_flux_sqv_x'
	assert fluxes[7].get_name()  == 'third_order_flux_sqv_y'
	assert fluxes[8].get_name()  == 'third_order_flux_sqc_x'
	assert fluxes[9].get_name()  == 'third_order_flux_sqc_y'
	assert fluxes[10].get_name() == 'third_order_flux_sqr_x'
	assert fluxes[11].get_name() == 'third_order_flux_sqr_y'


def test_fifth_order_upwind_horizontal_flux(grid):
	i = gt.Index()
	j = gt.Index()
	k = gt.Index()

	dt = gt.Global()

	s   	= gt.Equation()
	u   	= gt.Equation()
	v   	= gt.Equation()
	mtg 	= gt.Equation()
	su  	= gt.Equation()
	sv  	= gt.Equation()
	sqv 	= gt.Equation()
	sqc 	= gt.Equation()
	sqr 	= gt.Equation()
	s_tnd  	= gt.Equation()
	su_tnd 	= gt.Equation()
	sv_tnd 	= gt.Equation()
	qv_tnd 	= gt.Equation()
	qc_tnd 	= gt.Equation()
	qr_tnd 	= gt.Equation()

	from tasmania.python.isentropic.dynamics.fluxes import HorizontalIsentropicFlux

	fluxer = HorizontalIsentropicFlux.factory('fifth_order_upwind', grid, False)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd)

	assert len(fluxes) == 6
	assert fluxes[0].get_name() == 'fifth_order_flux_s_x'
	assert fluxes[1].get_name() == 'fifth_order_flux_s_y'
	assert fluxes[2].get_name() == 'fifth_order_flux_su_x'
	assert fluxes[3].get_name() == 'fifth_order_flux_su_y'
	assert fluxes[4].get_name() == 'fifth_order_flux_sv_x'
	assert fluxes[5].get_name() == 'fifth_order_flux_sv_y'

	fluxer = HorizontalIsentropicFlux.factory('fifth_order_upwind', grid, True)
	fluxes = fluxer(i, j, k, dt, s, u, v, mtg, su, sv, sqv, sqc, sqr,
					s_tnd=s_tnd, su_tnd=su_tnd, sv_tnd=sv_tnd,
					qv_tnd=qv_tnd, qc_tnd=qc_tnd, qr_tnd=qr_tnd)

	assert len(fluxes) == 12
	assert fluxes[0].get_name()  == 'fifth_order_flux_s_x'
	assert fluxes[1].get_name()  == 'fifth_order_flux_s_y'
	assert fluxes[2].get_name()  == 'fifth_order_flux_su_x'
	assert fluxes[3].get_name()  == 'fifth_order_flux_su_y'
	assert fluxes[4].get_name()  == 'fifth_order_flux_sv_x'
	assert fluxes[5].get_name()  == 'fifth_order_flux_sv_y'
	assert fluxes[6].get_name()  == 'fifth_order_flux_sqv_x'
	assert fluxes[7].get_name()  == 'fifth_order_flux_sqv_y'
	assert fluxes[8].get_name()  == 'fifth_order_flux_sqc_x'
	assert fluxes[9].get_name()  == 'fifth_order_flux_sqc_y'
	assert fluxes[10].get_name() == 'fifth_order_flux_sqr_x'
	assert fluxes[11].get_name() == 'fifth_order_flux_sqr_y'


if __name__ == '__main__':
	pytest.main([__file__])
