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
	Upwind(IsentropicBoussinesqMinimalVerticalFlux)
	Centered(IsentropicBoussinesqMinimalVerticalFlux)
	ThirdOrderUpwind(IsentropicBoussinesqMinimalVerticalFlux)
	FifthOrderUpwind(IsentropicBoussinesqMinimalVerticalFlux)
"""
from tasmania.python.isentropic.dynamics.vertical_fluxes import \
	IsentropicBoussinesqMinimalVerticalFlux
from tasmania.python.isentropic.dynamics.implementations.minimal_vertical_fluxes import \
	Upwind as CoreUpwind, get_upwind_flux, \
	Centered as CoreCentered, get_centered_flux, \
	ThirdOrderUpwind as CoreThirdOrderUpwind, get_third_order_upwind_flux, \
	FifthOrderUpwind as CoreFifthOrderUpwind, get_fifth_order_upwind_flux


class Upwind(IsentropicBoussinesqMinimalVerticalFlux):
	"""
	Upwind scheme.
	"""
	extent = 1
	order = 1

	def __init__(self, grid, moist):
		super().__init__(grid, moist)
		self._core = CoreUpwind(grid, moist)

	def __call__(self, k, w, s, su, sv, ddmtg, sqv=None, sqc=None, sqr=None):
		return_list = self._core(k, w, s, su, sv, sqv, sqc, sqr)
		return_list.insert(3, get_upwind_flux(k, w, ddmtg))
		return return_list


class Centered(IsentropicBoussinesqMinimalVerticalFlux):
	"""
	Centered scheme.
	"""
	extent = 1
	order = 2

	def __init__(self, grid, moist):
		super().__init__(grid, moist)
		self._core = CoreCentered(grid, moist)

	def __call__(self, k, w, s, su, sv, ddmtg, sqv=None, sqc=None, sqr=None):
		return_list = self._core(k, w, s, su, sv, sqv, sqc, sqr)
		return_list.insert(3, get_centered_flux(k, w, ddmtg))
		return return_list


class ThirdOrderUpwind(IsentropicBoussinesqMinimalVerticalFlux):
	"""
	Third-order upwind scheme.
	"""
	extent = 2
	order = 3

	def __init__(self, grid, moist):
		super().__init__(grid, moist)
		self._core = CoreThirdOrderUpwind(grid, moist)

	def __call__(self, k, w, s, su, sv, ddmtg, sqv=None, sqc=None, sqr=None):
		return_list = self._core(k, w, s, su, sv, sqv, sqc, sqr)
		return_list.insert(3, get_third_order_upwind_flux(k, w, ddmtg))
		return return_list


class FifthOrderUpwind(IsentropicBoussinesqMinimalVerticalFlux):
	"""
	Fifth-order upwind scheme.
	"""
	extent = 3
	order = 5

	def __init__(self, grid, moist):
		super().__init__(grid, moist)
		self._core = CoreFifthOrderUpwind(grid, moist)

	def __call__(self, k, w, s, su, sv, ddmtg, sqv=None, sqc=None, sqr=None):
		return_list = self._core(k, w, s, su, sv, sqv, sqc, sqr)
		return_list.insert(3, get_fifth_order_upwind_flux(k, w, ddmtg))
		return return_list
