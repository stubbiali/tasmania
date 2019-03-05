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
	Upwind(VerticalHomogeneousIsentropicFlux)
	Centered(VerticalHomogeneousIsentropicFlux)
	ThirdOrderUpwind(VerticalHomogeneousIsentropicFlux)
	FifthOrderUpwind(VerticalHomogeneousIsentropicFlux)

	_get_upwind_flux
	_get_centered_flux
	_get_third_order_upwind_flux
	_get_fifth_order_upwind_flux
"""
import gridtools as gt
from tasmania.python.isentropic.dynamics.fluxes import VerticalHomogeneousIsentropicFlux


class Upwind(VerticalHomogeneousIsentropicFlux):
	"""
	This class inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.VerticalHomogeneousIsentropicFlux`
	to implement the upwind scheme.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 1

	@property
	def order(self):
		return 1

	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		out_s  = _get_upwind_flux(k, w, s)
		out_su = _get_upwind_flux(k, w, su)
		out_sv = _get_upwind_flux(k, w, sv)

		if self._moist_on:
			out_sqv = _get_upwind_flux(k, w, sqv)
			out_sqc = _get_upwind_flux(k, w, sqc)
			out_sqr = _get_upwind_flux(k, w, sqr)

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


def _get_upwind_flux(k, w, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_z'
	flux = gt.Equation(name=flux_name)

	flux[k] = w[k] * ((w[k] > 0.) * phi[k  ] + (w[k] < 0.) * phi[k-1])

	return flux


class Centered(VerticalHomogeneousIsentropicFlux):
	"""
	This class inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.VerticalHomogeneousIsentropicFlux`
	to implement the centered scheme.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 1

	@property
	def order(self):
		return 2

	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		out_s  = _get_centered_flux(k, w, s)
		out_su = _get_centered_flux(k, w, su)
		out_sv = _get_centered_flux(k, w, sv)

		if self._moist_on:
			out_sqv = _get_centered_flux(k, w, sqv)
			out_sqc = _get_centered_flux(k, w, sqc)
			out_sqr = _get_centered_flux(k, w, sqr)

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


def _get_centered_flux(k, w, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_z'
	flux = gt.Equation(name=flux_name)

	flux[k] = w[k] * 0.5 * (phi[k] + phi[k-1])

	return flux


class ThirdOrderUpwind(VerticalHomogeneousIsentropicFlux):
	"""
	This class inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.VerticalHomogeneousIsentropicFlux`
	to implement the third-order upwind scheme.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 2

	@property
	def order(self):
		return 3

	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		out_s  = _get_third_order_upwind_flux(k, w, s)
		out_su = _get_third_order_upwind_flux(k, w, su)
		out_sv = _get_third_order_upwind_flux(k, w, sv)

		if self._moist_on:
			out_sqv = _get_third_order_upwind_flux(k, w, sqv)
			out_sqc = _get_third_order_upwind_flux(k, w, sqc)
			out_sqr = _get_third_order_upwind_flux(k, w, sqr)

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


def _get_third_order_upwind_flux(k, w, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_z'
	flux = gt.Equation(name=flux_name)

	flux[k] = w[k] / 12.0 * (7.0 * (phi[k-1] + phi[k]) -
							 1.0 * (phi[k-2] + phi[k+1])) - \
			  (w[k] * (w[k] > 0.0) - w[k] * (w[k] < 0.0)) / 12.0 * \
			  (3.0 * (phi[k-1] - phi[k]) -
			   1.0 * (phi[k-2] - phi[k+1]))

	return flux


class FifthOrderUpwind(VerticalHomogeneousIsentropicFlux):
	"""
	This class inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.VerticalHomogeneousIsentropicFlux`
	to implement the fifth-order upwind scheme.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)

	@property
	def nb(self):
		return 3

	@property
	def order(self):
		return 5

	def __call__(self, k, w, s, su, sv, sqv=None, sqc=None, sqr=None):
		out_s  = _get_fifth_order_upwind_flux(k, w, s)
		out_su = _get_fifth_order_upwind_flux(k, w, su)
		out_sv = _get_fifth_order_upwind_flux(k, w, sv)

		if self._moist_on:
			out_sqv = _get_fifth_order_upwind_flux(k, w, sqv)
			out_sqc = _get_fifth_order_upwind_flux(k, w, sqc)
			out_sqr = _get_fifth_order_upwind_flux(k, w, sqr)

		if not self._moist_on:
			return out_s, out_su, out_sv
		else:
			return out_s, out_su, out_sv, out_sqv, out_sqc, out_sqr


def _get_fifth_order_upwind_flux(k, w, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_z'
	flux = gt.Equation(name=flux_name)

	flux[k] = w[k] / 60.0 * (37.0 * (phi[k-1] + phi[k]) -
							 8.0 * (phi[k-2] + phi[k+1]) +
							 1.0 * (phi[k-3] + phi[k+2])) - \
			  (w[k] * (w[k] > 0.0) - w[k] * (w[k] < 0.0)) / 60.0 * \
			  (10.0 * (phi[k-1] - phi[k]) -
			   5.0 * (phi[k-2] - phi[k+1]) +
			   1.0 * (phi[k-3] - phi[k+2]))

	return flux
