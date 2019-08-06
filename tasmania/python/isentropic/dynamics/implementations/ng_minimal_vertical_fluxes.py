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
	Upwind(NGIsentropicMinimalVerticalFlux)
	Centered(NGIsentropicMinimalVerticalFlux)
	ThirdOrderUpwind(NGIsentropicMinimalVerticalFlux)
	FifthOrderUpwind(NGIsentropicMinimalVerticalFlux)
"""
from tasmania.python.isentropic.dynamics.fluxes import NGIsentropicMinimalVerticalFlux
from tasmania.python.isentropic.dynamics.implementations.minimal_vertical_fluxes import \
	get_upwind_flux, get_centered_flux, get_third_order_upwind_flux, get_fifth_order_upwind_flux


class Upwind(NGIsentropicMinimalVerticalFlux):
	"""
	Upwind scheme.
	"""
	extent = 1
	order = 1

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(self, k, w, s, su, sv, **tracer_kwargs):
		out_s  = get_upwind_flux(k, w, s)
		out_su = get_upwind_flux(k, w, su)
		out_sv = get_upwind_flux(k, w, sv)
		return_list = [out_s, out_su, out_sv]

		for tracer in self._tracers:
			# retrieve the tracer
			sq = tracer_kwargs['s' + tracer]

			# compute the flux
			out_sq = get_upwind_flux(k, w, sq)

			# update the return list
			return_list.append(out_sq)

		return return_list


class Centered(NGIsentropicMinimalVerticalFlux):
	"""
	Centered scheme.
	"""
	extent = 1
	order = 2

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(self, k, w, s, su, sv, **tracer_kwargs):
		out_s  = get_centered_flux(k, w, s)
		out_su = get_centered_flux(k, w, su)
		out_sv = get_centered_flux(k, w, sv)
		return_list = [out_s, out_su, out_sv]

		for tracer in self._tracers:
			# retrieve the tracer
			sq = tracer_kwargs['s' + tracer]

			# compute the flux
			out_sq = get_centered_flux(k, w, sq)

			# update the return list
			return_list.append(out_sq)

		return return_list


class ThirdOrderUpwind(NGIsentropicMinimalVerticalFlux):
	"""
	Third-order upwind scheme.
	"""
	extent = 2
	order = 3

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(self, k, w, s, su, sv, **tracer_kwargs):
		out_s  = get_third_order_upwind_flux(k, w, s)
		out_su = get_third_order_upwind_flux(k, w, su)
		out_sv = get_third_order_upwind_flux(k, w, sv)
		return_list = [out_s, out_su, out_sv]

		for tracer in self._tracers:
			# retrieve the tracer
			sq = tracer_kwargs['s' + tracer]

			# compute the flux
			out_sq = get_third_order_upwind_flux(k, w, sq)

			# update the return list
			return_list.append(out_sq)

		return return_list


class FifthOrderUpwind(NGIsentropicMinimalVerticalFlux):
	"""
	Fifth-order upwind scheme.
	"""
	extent = 3
	order = 5

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(self, k, w, s, su, sv, **tracer_kwargs):
		out_s  = get_fifth_order_upwind_flux(k, w, s)
		out_su = get_fifth_order_upwind_flux(k, w, su)
		out_sv = get_fifth_order_upwind_flux(k, w, sv)
		return_list = [out_s, out_su, out_sv]

		for tracer in self._tracers:
			# retrieve the tracer
			sq = tracer_kwargs['s' + tracer]

			# compute the flux
			out_sq = get_fifth_order_upwind_flux(k, w, sq)

			# update the return list
			return_list.append(out_sq)

		return return_list
