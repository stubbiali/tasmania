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
	Upwind(IsentropicMinimalHorizontalFlux)
	Centered(IsentropicMinimalHorizontalFlux)
	MacCormack(IsentropicMinimalHorizontalFlux)
	FifthOrderUpwind(IsentropicMinimalHorizontalFlux)
"""
from tasmania.python.isentropic.dynamics.horizontal_fluxes import IsentropicMinimalHorizontalFlux
from tasmania.python.isentropic.dynamics.implementations.horizontal_fluxes import \
	get_centered_flux_x, get_centered_flux_y, \
	get_fifth_order_upwind_flux_x, get_fifth_order_upwind_flux_y, \
	get_fourth_order_centered_flux_x, get_fourth_order_centered_flux_y, \
	get_maccormack_flux_x, get_maccormack_flux_x_s, \
	get_maccormack_flux_y, get_maccormack_flux_y_s, \
	get_maccormack_predicted_value_s, get_maccormack_predicted_value_sq, \
	get_sixth_order_centered_flux_x, get_sixth_order_centered_flux_y, \
	get_third_order_upwind_flux_x, get_third_order_upwind_flux_y, \
	get_upwind_flux_x, get_upwind_flux_y


class Upwind(IsentropicMinimalHorizontalFlux):
	"""	Upwind scheme. """
	extent = 1
	order = 1
	externals = {
		'get_upwind_flux_x': get_upwind_flux_x,
		'get_upwind_flux_y': get_upwind_flux_y
	}

	@staticmethod
	def __call__(
		dt, dx, dy, s, u, v, su, sv, sqv=None, sqc=None, sqr=None,
		s_tnd=None, su_tnd=None, sv_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		# compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_upwind_flux_x(u=u, phi=s)
		flux_s_y  = get_upwind_flux_y(v=v, phi=s)
		flux_su_x = get_upwind_flux_x(u=u, phi=su)
		flux_su_y = get_upwind_flux_y(v=v, phi=su)
		flux_sv_x = get_upwind_flux_x(u=u, phi=sv)
		flux_sv_y = get_upwind_flux_y(v=v, phi=sv)

		if not moist:
			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		else:
			# compute fluxes for the water constituents
			flux_sqv_x = get_upwind_flux_x(u=u, phi=sqv)
			flux_sqv_y = get_upwind_flux_y(v=v, phi=sqv)
			flux_sqc_x = get_upwind_flux_x(u=u, phi=sqc)
			flux_sqc_y = get_upwind_flux_y(v=v, phi=sqc)
			flux_sqr_x = get_upwind_flux_x(u=u, phi=sqr)
			flux_sqr_y = get_upwind_flux_y(v=v, phi=sqr)

			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y


class Centered(IsentropicMinimalHorizontalFlux):
	""" Centered scheme. """
	extent = 1
	order = 2
	externals = {
		'get_centered_flux_x': get_centered_flux_x,
		'get_centered_flux_y': get_centered_flux_y
	}

	@staticmethod
	def __call__(
		dt, dx, dy, s, u, v, su, sv, sqv=None, sqc=None, sqr=None,
		s_tnd=None, su_tnd=None, sv_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		# compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_centered_flux_x(u=u, phi=s)
		flux_s_y  = get_centered_flux_y(v=v, phi=s)
		flux_su_x = get_centered_flux_x(u=u, phi=su)
		flux_su_y = get_centered_flux_y(v=v, phi=su)
		flux_sv_x = get_centered_flux_x(u=u, phi=sv)
		flux_sv_y = get_centered_flux_y(v=v, phi=sv)

		if not moist:
			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		else:
			# compute fluxes for the water constituents
			flux_sqv_x = get_centered_flux_x(u=u, phi=sqv)
			flux_sqv_y = get_centered_flux_y(v=v, phi=sqv)
			flux_sqc_x = get_centered_flux_x(u=u, phi=sqc)
			flux_sqc_y = get_centered_flux_y(v=v, phi=sqc)
			flux_sqr_x = get_centered_flux_x(u=u, phi=sqr)
			flux_sqr_y = get_centered_flux_y(v=v, phi=sqr)

			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y


def get_maccormack_predicted_value_su(dt, dx, dy, s, u_unstg, v_unstg, su, su_tnd):
	if su_tnd_on:
		su_prd = su[0, 0, 0] - dt * (
			(u_unstg[1, 0, 0] * su[1, 0, 0] - u_unstg[0, 0, 0] * su[0, 0, 0]) / dx +
			(v_unstg[0, 1, 0] * su[0, 1, 0] - v_unstg[0, 0, 0] * su[0, 0, 0]) / dy
		)
	else:
		su_prd = su[0, 0, 0] - dt * (
			(u_unstg[1, 0, 0] * su[1, 0, 0] - u_unstg[0, 0, 0] * su[0, 0, 0]) / dx +
			(v_unstg[0, 1, 0] * su[0, 1, 0] - v_unstg[0, 0, 0] * su[0, 0, 0]) / dy +
			su_tnd[0, 0, 0]
		)
	return su_prd


def get_maccormack_predicted_value_sv(dt, dx, dy, s, u_unstg, v_unstg, sv, sv_tnd):
	if sv_tnd_on is None:
		sv_prd = sv[0, 0, 0] - dt * (
			(u_unstg[1, 0, 0] * sv[1, 0, 0] - u_unstg[0, 0, 0] * sv[0, 0, 0]) / dx +
			(v_unstg[0, 1, 0] * sv[0, 1, 0] - v_unstg[0, 0, 0] * sv[0, 0, 0]) / dy
		)
	else:
		sv_prd = sv[0, 0, 0] - dt * (
			(u_unstg[1, 0, 0] * sv[1, 0, 0] - u_unstg[0, 0, 0] * sv[0, 0, 0]) / dx +
			(v_unstg[0, 1, 0] * sv[0, 1, 0] - v_unstg[0, 0, 0] * sv[0, 0, 0]) / dy +
			sv_tnd[0, 0, 0]
		)
	return sv_prd


class MacCormack(IsentropicMinimalHorizontalFlux):
	""" MacCormack scheme. """
	extent = 1
	order = 2
	externals = {
		'get_maccormack_predicted_value_s': get_maccormack_predicted_value_s,
		'get_maccormack_predicted_value_su': get_maccormack_predicted_value_su,
		'get_maccormack_predicted_value_sv': get_maccormack_predicted_value_sv,
		'get_maccormack_predicted_value_sq': get_maccormack_predicted_value_sq,
		'get_maccormack_flux_x': get_maccormack_flux_x,
		'get_maccormack_flux_x_s': get_maccormack_flux_x_s,
		'get_maccormack_flux_y': get_maccormack_flux_y,
		'get_maccormack_flux_y_s': get_maccormack_flux_y_s,
	}

	def __init__(self, grid, moist):
		super().__init__(grid, moist)

	@staticmethod
	def __call__(
		dt, dx, dy, s, u, v, su, sv, sqv=None, sqc=None, sqr=None,
		s_tnd=None, su_tnd=None, sv_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		# diagnose the velocity components at the mass points
		u_unstg = su[0, 0, 0] / s[0, 0, 0]
		v_unstg = sv[0, 0, 0] / s[0, 0, 0]

		# compute the predicted values for the isentropic density and the momenta
		s_prd = get_maccormack_predicted_value_s(
			dt=dt, dx=dx, dy=dy, s=s, su=su, sv=sv
		)
		su_prd = get_maccormack_predicted_value_su(
			dt=dt, dx=dx, dy=dy, s=s, u_unstg=u_unstg, v_unstg=v_unstg,
			su=su, su_tnd=su_tnd
		)
		sv_prd = get_maccormack_predicted_value_sv(
			dt=dt, dx=dx, dy=dy, s=s, u_unstg=u_unstg, v_unstg=v_unstg,
			sv=sv, sv_tnd=sv_tnd
		)

		if moist:
			# compute the predicted values for the water constituents
			sqv_prd = get_maccormack_predicted_value_sq(
				dt=dt, dx=dx, dy=dy, s=s, u_unstg=u_unstg, v_unstg=v_unstg,
				sq=sqv, q_tnd_on=qv_tnd_on, q_tnd=qv_tnd
			)
			sqc_prd = get_maccormack_predicted_value_sq(
				dt=dt, dx=dx, dy=dy, s=s, u_unstg=u_unstg, v_unstg=v_unstg,
				sq=sqc, q_tnd_on=qc_tnd_on, q_tnd=qc_tnd
			)
			sqr_prd = get_maccormack_predicted_value_sq(
				dt=dt, dx=dx, dy=dy, s=s, u_unstg=u_unstg, v_unstg=v_unstg,
				sq=sqr, q_tnd_on=qr_tnd_on, q_tnd=qr_tnd
			)

		# diagnose the predicted values for the velocity components
		# at the mass points
		u_prd_unstg = su_prd[0, 0, 0] / s_prd[0, 0, 0]
		v_prd_unstg = sv_prd[0, 0, 0] / s_prd[0, 0, 0]

		# compute the fluxes for the isentropic density and the momenta
		flux_s_x  = get_maccormack_flux_x_s(su=su, su_prd=su_prd)
		flux_s_y  = get_maccormack_flux_y_s(sv=sv, sv_prd=sv_prd)
		flux_su_x = get_maccormack_flux_x(
			u_unstg=u_unstg, phi=su, u_prd_unstg=u_prd_unstg, phi_prd=su_prd
		)
		flux_su_y = get_maccormack_flux_y(
			v_unstg=v_unstg, phi=su, v_prd_unstg=v_prd_unstg, phi_prd=su_prd
		)
		flux_sv_x = get_maccormack_flux_x(
			u_unstg=u_unstg, phi=sv, u_prd_unstg=u_prd_unstg, phi_prd=sv_prd
		)
		flux_sv_y = get_maccormack_flux_y(
			v_unstg=v_unstg, phi=sv, v_prd_unstg=v_prd_unstg, phi_prd=sv_prd
		)

		if not moist:
			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		if moist:
			# compute the fluxes for the water constituents
			flux_sqv_x = get_maccormack_flux_x(
				u_unstg=u_unstg, phi=sqv, u_prd_unstg=u_prd_unstg, phi_prd=sqv_prd
			)
			flux_sqv_y = get_maccormack_flux_y(
				v_unstg=v_unstg, phi=sqv, v_prd_unstg=v_prd_unstg, phi_prd=sqv_prd
			)
			flux_sqc_x = get_maccormack_flux_x(
				u_unstg=u_unstg, phi=sqc, u_prd_unstg=u_prd_unstg, phi_prd=sqc_prd
			)
			flux_sqc_y = get_maccormack_flux_y(
				v_unstg=v_unstg, phi=sqc, v_prd_unstg=v_prd_unstg, phi_prd=sqc_prd
			)
			flux_sqr_x = get_maccormack_flux_x(
				u_unstg=u_unstg, phi=sqr, u_prd_unstg=u_prd_unstg, phi_prd=sqr_prd
			)
			flux_sqr_y = get_maccormack_flux_y(
				v_unstg=v_unstg, phi=sqr, v_prd_unstg=v_prd_unstg, phi_prd=sqr_prd
			)

			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y


class ThirdOrderUpwind(IsentropicMinimalHorizontalFlux):
	"""	Third-order upwind scheme. """
	extent = 2
	order = 3
	externals = {
		'get_fourth_order_centered_flux_x': get_fourth_order_centered_flux_x,
		'get_third_order_upwind_flux_x': get_third_order_upwind_flux_x,
		'get_fourth_order_centered_flux_y': get_fourth_order_centered_flux_y,
		'get_third_order_upwind_flux_y': get_third_order_upwind_flux_y,
	}

	@staticmethod
	def __call__(
		dt, dx, dy, s, u, v, su, sv, sqv=None, sqc=None, sqr=None,
		s_tnd=None, su_tnd=None, sv_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		# compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_third_order_upwind_flux_x(u=u, phi=s)
		flux_s_y  = get_third_order_upwind_flux_y(v=v, phi=s)
		flux_su_x = get_third_order_upwind_flux_x(u=u, phi=su)
		flux_su_y = get_third_order_upwind_flux_y(v=v, phi=su)
		flux_sv_x = get_third_order_upwind_flux_x(u=u, phi=sv)
		flux_sv_y = get_third_order_upwind_flux_y(v=v, phi=sv)

		if not moist:
			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		else:
			# compute fluxes for the water constituents
			flux_sqv_x = get_third_order_upwind_flux_x(u=u, phi=sqv)
			flux_sqv_y = get_third_order_upwind_flux_y(v=v, phi=sqv)
			flux_sqc_x = get_third_order_upwind_flux_x(u=u, phi=sqc)
			flux_sqc_y = get_third_order_upwind_flux_y(v=v, phi=sqc)
			flux_sqr_x = get_third_order_upwind_flux_x(u=u, phi=sqr)
			flux_sqr_y = get_third_order_upwind_flux_y(v=v, phi=sqr)

			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y


class FifthOrderUpwind(IsentropicMinimalHorizontalFlux):
	""" Fifth-order upwind scheme. """
	extent = 3
	order = 5
	externals = {
		'get_sixth_order_centered_flux_x': get_sixth_order_centered_flux_x,
		'get_fifth_order_upwind_flux_x': get_fifth_order_upwind_flux_x,
		'get_sixth_order_centered_flux_y': get_sixth_order_centered_flux_y,
		'get_fifth_order_upwind_flux_y': get_fifth_order_upwind_flux_y,
	}

	@staticmethod
	def __call__(
		dt, dx, dy, s, u, v, su, sv, sqv=None, sqc=None, sqr=None,
		s_tnd=None, su_tnd=None, sv_tnd=None, qv_tnd=None, qc_tnd=None, qr_tnd=None
	):
		# compute fluxes for the isentropic density and the momenta
		flux_s_x  = get_fifth_order_upwind_flux_x(u=u, phi=s)
		flux_s_y  = get_fifth_order_upwind_flux_y(v=v, phi=s)
		flux_su_x = get_fifth_order_upwind_flux_x(u=u, phi=su)
		flux_su_y = get_fifth_order_upwind_flux_y(v=v, phi=su)
		flux_sv_x = get_fifth_order_upwind_flux_x(u=u, phi=sv)
		flux_sv_y = get_fifth_order_upwind_flux_y(v=v, phi=sv)

		if not moist:
			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
		else:
			# compute fluxes for the water constituents
			flux_sqv_x = get_fifth_order_upwind_flux_x(u=u, phi=sqv)
			flux_sqv_y = get_fifth_order_upwind_flux_y(v=v, phi=sqv)
			flux_sqc_x = get_fifth_order_upwind_flux_x(u=u, phi=sqc)
			flux_sqc_y = get_fifth_order_upwind_flux_y(v=v, phi=sqc)
			flux_sqr_x = get_fifth_order_upwind_flux_x(u=u, phi=sqr)
			flux_sqr_y = get_fifth_order_upwind_flux_y(v=v, phi=sqr)

			return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y, \
				flux_sqv_x, flux_sqv_y, flux_sqc_x, flux_sqc_y, flux_sqr_x, flux_sqr_y

