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
"""
This module contains:
	Upwind(IsentropicMinimalVerticalFlux)
	Centered(IsentropicMinimalVerticalFlux)
	ThirdOrderUpwind(IsentropicMinimalVerticalFlux)
	FifthOrderUpwind(IsentropicMinimalVerticalFlux)
"""
from tasmania.python.isentropic.dynamics.vertical_fluxes import \
	IsentropicMinimalVerticalFlux
from tasmania.python.isentropic.dynamics.implementations.ng_minimal_vertical_fluxes import \
	get_upwind_flux, get_centered_flux, \
	get_third_order_upwind_flux, get_fifth_order_upwind_flux


class Upwind(IsentropicMinimalVerticalFlux):
	"""
	Upwind scheme.
	"""
	extent = 1
	order = 1

	def __init__(self, grid, moist):
		super().__init__(grid, moist)

	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		out_s  = get_upwind_flux(k, w, s)
		out_su = get_upwind_flux(k, w, su)
		out_sv = get_upwind_flux(k, w, sv)
		return_list = [out_s, out_su, out_sv]

		if self._moist:
			out_sqv = get_upwind_flux(k, w, sqv)
			out_sqc = get_upwind_flux(k, w, sqc)
			out_sqr = get_upwind_flux(k, w, sqr)
			return_list += [out_sqv, out_sqc, out_sqr]

		return return_list


class Centered(IsentropicMinimalVerticalFlux):
	"""
	Centered scheme.
	"""
	extent = 1
	order = 2

	def __init__(self, grid, moist):
		super().__init__(grid, moist)

	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		out_s  = get_centered_flux(k, w, s)
		out_su = get_centered_flux(k, w, su)
		out_sv = get_centered_flux(k, w, sv)
		return_list = [out_s, out_su, out_sv]

		if self._moist:
			out_sqv = get_centered_flux(k, w, sqv)
			out_sqc = get_centered_flux(k, w, sqc)
			out_sqr = get_centered_flux(k, w, sqr)
			return_list += [out_sqv, out_sqc, out_sqr]

		return return_list


class ThirdOrderUpwind(IsentropicMinimalVerticalFlux):
	"""
	Third-order upwind scheme.
	"""
	extent = 2
	order = 3

	def __init__(self, grid, moist):
		super().__init__(grid, moist)

	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		out_s  = get_third_order_upwind_flux(k, w, s)
		out_su = get_third_order_upwind_flux(k, w, su)
		out_sv = get_third_order_upwind_flux(k, w, sv)
		return_list = [out_s, out_su, out_sv]

		if self._moist:
			out_sqv = get_third_order_upwind_flux(k, w, sqv)
			out_sqc = get_third_order_upwind_flux(k, w, sqc)
			out_sqr = get_third_order_upwind_flux(k, w, sqr)
			return_list += [out_sqv, out_sqc, out_sqr]

		return return_list


class FifthOrderUpwind(IsentropicMinimalVerticalFlux):
	"""
	Fifth-order upwind scheme.
	"""
	extent = 3
	order = 5

	def __init__(self, grid, moist):
		super().__init__(grid, moist)

	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		out_s  = get_fifth_order_upwind_flux(k, w, s)
		out_su = get_fifth_order_upwind_flux(k, w, su)
		out_sv = get_fifth_order_upwind_flux(k, w, sv)
		return_list = [out_s, out_su, out_sv]

		if self._moist:
			out_sqv = get_fifth_order_upwind_flux(k, w, sqv)
			out_sqc = get_fifth_order_upwind_flux(k, w, sqc)
			out_sqr = get_fifth_order_upwind_flux(k, w, sqr)
			return_list += [out_sqv, out_sqc, out_sqr]

		return return_list
