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
	Upwind(NGIsentropicHorizontalFlux)
	Centered(NGIsentropicHorizontalFlux)
	MacCormack(NGIsentropicHorizontalFlux)
	ThirdOrderUpwind(NGIsentropicHorizontalFlux)
	FifthOrderUpwind(NGIsentropicHorizontalFlux)

	get_upwind_flux_{x, y}
	get_centered_flux_{x, y}
	get_maccormack_flux_{x, y}
	get_maccormack_flux_{x, y}_s
	get_third_order_upwind_flux_{x, y}
	get_fourth_order_centered_flux_{x, y}
	get_fifth_order_upwind_flux_{x, y}
	get_sixth_order_centered_flux_{x, y}
"""
import gridtools as gt
from tasmania.python.isentropic.dynamics.fluxes import NGIsentropicHorizontalFlux
from .horizontal_fluxes import \
	get_upwind_flux_x, get_upwind_flux_y, \
	get_centered_flux_x, get_centered_flux_y, \
	get_maccormack_flux_x, get_maccormack_flux_y, \
	get_maccormack_flux_x_s, get_maccormack_flux_y_s, \
	get_third_order_upwind_flux_x, get_third_order_upwind_flux_y, \
	get_fifth_order_upwind_flux_x, get_fifth_order_upwind_flux_y


class Upwind(NGIsentropicHorizontalFlux):
	"""
	Upwind scheme.
	"""
	extent = 1
	order = 1

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(
		self, i, j, dt, s, u, v, mtg, su, sv,
		s_tnd=None, su_tnd=None, sv_tnd=None, **tracer_kwargs
	):
		"""
		Note
		----
		``s_tnd``, ``su_tnd`` and ``sv_tnd`` are not actually used, yet they
		are retained as default arguments for compliance with the class
		hierarchy interface.
		"""
		# compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_upwind_flux_x(i, j, u, s)
		flux_s_y  = get_upwind_flux_y(i, j, v, s)
		flux_su_x = get_upwind_flux_x(i, j, u, su)
		flux_su_y = get_upwind_flux_y(i, j, v, su)
		flux_sv_x = get_upwind_flux_x(i, j, u, sv)
		flux_sv_y = get_upwind_flux_y(i, j, v, sv)

		# initialize the return list
		return_list = [
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		]

		for tracer in self._tracers:
			# retrieve the tracer
			sq = tracer_kwargs['s' + tracer]

			# compute fluxes for the tracer
			flux_sq_x = get_upwind_flux_x(i, j, u, sq)
			flux_sq_y = get_upwind_flux_y(i, j, v, sq)

			# update the return list
			return_list += [flux_sq_x, flux_sq_y]

		return return_list


class Centered(NGIsentropicHorizontalFlux):
	"""
	Centered scheme.
	"""
	extent = 1
	order = 2

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(
		self, i, j, dt, s, u, v, mtg, su, sv,
		s_tnd=None, su_tnd=None, sv_tnd=None, **tracer_kwargs
	):
		"""
		Note
		----
		``s_tnd``, ``su_tnd`` and ``sv_tnd`` are not actually used, yet they
		are retained as default arguments for compliance with the class
		hierarchy interface.
		"""
		# compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_centered_flux_x(i, j, u, s)
		flux_s_y  = get_centered_flux_y(i, j, v, s)
		flux_su_x = get_centered_flux_x(i, j, u, su)
		flux_su_y = get_centered_flux_y(i, j, v, su)
		flux_sv_x = get_centered_flux_x(i, j, u, sv)
		flux_sv_y = get_centered_flux_y(i, j, v, sv)

		# initialize the return list
		return_list = [
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		]

		for tracer in self._tracers:
			# retrieve the tracer
			sq = tracer_kwargs['s' + tracer]

			# compute fluxes for the tracer
			flux_sq_x = get_centered_flux_x(i, j, u, sq)
			flux_sq_y = get_centered_flux_y(i, j, v, sq)

			# update the return list
			return_list += [flux_sq_x, flux_sq_y]

		return return_list


class MacCormack(NGIsentropicHorizontalFlux):
	"""
	MacCormack scheme.
	"""
	extent = 1
	order = 2

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(
		self, i, j, dt, s, u, v, mtg, su, sv,
		s_tnd=None, su_tnd=None, sv_tnd=None, **tracer_kwargs
	):
		# diagnose the velocity components at the mass points
		u_unstg = gt.Equation()
		u_unstg[i, j] = su[i, j] / s[i, j]
		v_unstg = gt.Equation()
		v_unstg[i, j] = sv[i, j] / s[i, j]

		# compute the predicted values for the isentropic density and the momenta
		s_prd = self._get_maccormack_horizontal_predicted_value_s(
			i, j, dt, s, su, sv
		)
		su_prd = self._get_maccormack_horizontal_predicted_value_su(
			i, j, dt, s, u_unstg, v_unstg, mtg, su, su_tnd
		)
		sv_prd = self._get_maccormack_horizontal_predicted_value_sv(
			i, j, dt, s, u_unstg, v_unstg, mtg, sv, sv_tnd
		)

		tracers_prd = {}
		for tracer in self._tracers:
			# retrieve the tracer and its tendencies
			sq = tracer_kwargs['s' + tracer]
			q_tnd = tracer_kwargs.get(tracer + '_tnd', None)

			# compute the predicted value for the tracer
			sq_prd = self._get_maccormack_horizontal_predicted_value_sq(
				i, j, dt, s, u_unstg, v_unstg, sq, q_tnd
			)

			# update the dictionary
			tracers_prd['s' + tracer] = sq_prd

		# diagnose the predicted values for the velocity components at the mass points
		u_prd_unstg = gt.Equation()
		u_prd_unstg[i, j] = su_prd[i, j] / s_prd[i, j]
		v_prd_unstg = gt.Equation()
		v_prd_unstg[i, j] = sv_prd[i, j] / s_prd[i, j]

		# compute the fluxes for the isentropic density and the momenta
		flux_s_x  = get_maccormack_flux_x_s(i, j, su, su_prd)
		flux_s_y  = get_maccormack_flux_y_s(i, j, sv, sv_prd)
		flux_su_x = get_maccormack_flux_x(i, j, u_unstg, su, u_prd_unstg, su_prd)
		flux_su_y = get_maccormack_flux_y(i, j, v_unstg, su, v_prd_unstg, su_prd)
		flux_sv_x = get_maccormack_flux_x(i, j, u_unstg, sv, u_prd_unstg, sv_prd)
		flux_sv_y = get_maccormack_flux_y(i, j, v_unstg, sv, v_prd_unstg, sv_prd)

		# initialize the return list
		return_list = [
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		]

		for tracer in self._tracers:
			# retrieve the tracer and its predicted value
			sq = tracer_kwargs['s' + tracer]
			sq_prd = tracers_prd['s' + tracer]

			# compute the fluxes for the tracer
			flux_sq_x = get_maccormack_flux_x(i, j, u_unstg, sq, u_prd_unstg, sq_prd)
			flux_sq_y = get_maccormack_flux_y(i, j, v_unstg, sq, v_prd_unstg, sq_prd)

			# update the return list
			return_list += [flux_sq_x, flux_sq_y]

		return return_list

	def _get_maccormack_horizontal_predicted_value_s(self, i, j, dt, s, su, sv):
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		s_prd = gt.Equation()
		s_prd[i, j] = s[i, j] - dt * (
			(su[i+1, j] - su[i, j]) / dx +
			(sv[i, j+1] - sv[i, j]) / dy
		)
		return s_prd

	def _get_maccormack_horizontal_predicted_value_su(
		self, i, j, dt, s, u_unstg, v_unstg, mtg, su, su_tnd
	):
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		su_prd = gt.Equation()

		if su_tnd is None:
			su_prd[i, j] = su[i, j] - dt * (
				(u_unstg[i+1, j] * su[i+1, j] -
				 u_unstg[  i, j] * su[  i, j]) / dx +
				(v_unstg[i, j+1] * su[i, j+1] -
				 v_unstg[i, j] * su[i, j]) / dy +
				s[i, j] * (mtg[i+1, j] - mtg[i, j]) / dx
			)
		else:
			su_prd[i, j] = su[i, j] - dt * (
				(u_unstg[i+1, j] * su[i+1, j] -
				 u_unstg[  i, j] * su[  i, j]) / dx +
				(v_unstg[i, j+1] * su[i, j+1] -
				 v_unstg[i, j] * su[i, j]) / dy +
				s[i, j] * (mtg[i+1, j] - mtg[i, j]) / dx -
				su_tnd[i, j]
			)

		return su_prd

	def _get_maccormack_horizontal_predicted_value_sv(
		self, i, j, dt, s, u_unstg, v_unstg, mtg, sv, sv_tnd
	):
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		sv_prd = gt.Equation()

		if sv_tnd is None:
			sv_prd[i, j] = sv[i, j] - dt * (
				(u_unstg[i+1, j] * sv[i+1, j] -
				 u_unstg[  i, j] * sv[  i, j]) / dx +
				(v_unstg[i, j+1] * sv[i, j+1] -
				 v_unstg[i, j] * sv[i, j]) / dy +
				s[i, j] * (mtg[i, j+1] - mtg[i, j]) / dy
			)
		else:
			sv_prd[i, j] = sv[i, j] - dt * (
				(u_unstg[i+1, j] * sv[i+1, j] -
				 u_unstg[  i, j] * sv[  i, j]) / dx +
				(v_unstg[i, j+1] * sv[i, j+1] -
				 v_unstg[i, j] * sv[i, j]) / dy +
				s[i, j] * (mtg[i, j+1] - mtg[i, j]) / dy -
				sv_tnd[i, j]
			)

		return sv_prd

	def _get_maccormack_horizontal_predicted_value_sq(
		self, i, j, dt, s, u_unstg, v_unstg, sq, q_tnd
	):
		dx, dy = self._grid.dx.values.item(), self._grid.dy.values.item()
		sq_name = sq.get_name()
		sq_prd_name = sq_name + '_prd'
		sq_prd = gt.Equation(name=sq_prd_name)

		if q_tnd is None:
			sq_prd[i, j] = sq[i, j] - dt * (
				(u_unstg[i+1, j] * sq[i+1, j] -
				 u_unstg[  i, j] * sq[  i, j]) / dx +
				(v_unstg[i, j+1] * sq[i, j+1] -
				 v_unstg[i, j] * sq[i, j]) / dy
			)
		else:
			sq_prd[i, j] = sq[i, j] - dt * (
				(u_unstg[i+1, j] * sq[i+1, j] -
				 u_unstg[  i, j] * sq[  i, j]) / dx +
				(v_unstg[i, j+1] * sq[i, j+1] -
				 v_unstg[i, j] * sq[i, j]) / dy -
				s[i, j] * q_tnd[i, j]
			)

		return sq_prd


class ThirdOrderUpwind(NGIsentropicHorizontalFlux):
	"""
	Third-order scheme.
	"""
	extent = 2
	order = 3

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(
		self, i, j, dt, s, u, v, mtg, su, sv,
		s_tnd=None, su_tnd=None, sv_tnd=None, **tracer_kwargs
	):
		"""
		Note
		----
		``s_tnd``, ``su_tnd`` and ``sv_tnd`` are not actually used, yet they
		are retained as default arguments for compliance with the class
		hierarchy interface.
		"""
		# compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_third_order_upwind_flux_x(i, j, u, s)
		flux_s_y  = get_third_order_upwind_flux_y(i, j, v, s)
		flux_su_x = get_third_order_upwind_flux_x(i, j, u, su)
		flux_su_y = get_third_order_upwind_flux_y(i, j, v, su)
		flux_sv_x = get_third_order_upwind_flux_x(i, j, u, sv)
		flux_sv_y = get_third_order_upwind_flux_y(i, j, v, sv)

		# initialize the return list
		return_list = [
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		]

		for tracer in self._tracers:
			# retrieve the tracer
			sq = tracer_kwargs['s' + tracer]

			# compute fluxes for the tracer
			flux_sq_x = get_third_order_upwind_flux_x(i, j, u, sq)
			flux_sq_y = get_third_order_upwind_flux_y(i, j, v, sq)

			# update the return list
			return_list += [flux_sq_x, flux_sq_y]

		return return_list


class FifthOrderUpwind(NGIsentropicHorizontalFlux):
	"""
	Fifth-order scheme.
	"""
	extent = 3
	order = 5

	def __init__(self, grid, tracers):
		super().__init__(grid, tracers)

	def __call__(
		self, i, j, dt, s, u, v, mtg, su, sv,
		s_tnd=None, su_tnd=None, sv_tnd=None, **tracer_kwargs
	):
		"""
		Note
		----
		``s_tnd``, ``su_tnd`` and ``sv_tnd`` are not actually used, yet they
		are retained as default arguments for compliance with the class
		hierarchy interface.
		"""
		# compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_fifth_order_upwind_flux_x(i, j, u, s)
		flux_s_y  = get_fifth_order_upwind_flux_y(i, j, v, s)
		flux_su_x = get_fifth_order_upwind_flux_x(i, j, u, su)
		flux_su_y = get_fifth_order_upwind_flux_y(i, j, v, su)
		flux_sv_x = get_fifth_order_upwind_flux_x(i, j, u, sv)
		flux_sv_y = get_fifth_order_upwind_flux_y(i, j, v, sv)

		# initialize the return list
		return_list = [
			flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		]

		for tracer in self._tracers:
			# retrieve the tracer
			sq = tracer_kwargs['s' + tracer]

			# compute fluxes for the tracer
			flux_sq_x = get_fifth_order_upwind_flux_x(i, j, u, sq)
			flux_sq_y = get_fifth_order_upwind_flux_y(i, j, v, sq)

			# update the return list
			return_list += [flux_sq_x, flux_sq_y]

		return return_list


