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
	Upwind(IsentropicVerticalFlux)
	Centered(IsentropicVerticalFlux)
	MacCormack(IsentropicVerticalFlux)

	get_upwind_flux
	get_centered_flux
	get_maccormack_flux
"""
import gridtools as gt
from tasmania.python.isentropic.dynamics.vertical_fluxes import IsentropicVerticalFlux


class Upwind(IsentropicVerticalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicVerticalFlux`
	to implement the upwind scheme to compute the vertical
	numerical fluxes for the governing equations expressed
	in conservative form using isentropic coordinates.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	order : int
		Order of accuracy.
	"""
	def __init__(self, grid, moist):
		super().__init__(grid, moist)
		self.nb = 1
		self.order = 1

	def __call__(
		self, i, j, k, dt, w, s, s_prv, su, su_prv, sv, sv_prv,
		sqv=None, sqv_prv=None, sqc=None, sqc_prv=None, sqr=None, sqr_prv=None
	):
		# Interpolate the vertical velocity at the model half-levels
		w_mid = gt.Equation()
		w_mid[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k-1])

		# Compute flux for the isentropic density and the momenta
		flux_s_z  = get_upwind_flux(i, j, k, w_mid, s)
		flux_su_z = get_upwind_flux(i, j, k, w_mid, su)
		flux_sv_z = get_upwind_flux(i, j, k, w_mid, sv)

		# Initialize return list
		return_list = [flux_s_z, flux_su_z, flux_sv_z]

		if self._moist:
			# Compute flux for the water constituents
			flux_sqv_z = get_upwind_flux(i, j, k, w_mid, sqv)
			flux_sqc_z = get_upwind_flux(i, j, k, w_mid, sqc)
			flux_sqr_z = get_upwind_flux(i, j, k, w_mid, sqr)

			# Update the return list
			return_list += [flux_sqv_z, flux_sqc_z, flux_sqr_z]

		return return_list


def get_upwind_flux(i, j, k, w_mid, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_z'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = w_mid[i, j, k] * \
		((w_mid[i, j, k] > 0.) * phi[i, j, k-1] +
		 (w_mid[i, j, k] < 0.) * phi[i, j,	 k])

	return flux


class Centered(IsentropicVerticalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicVerticalFlux`
	to implement the centered scheme to compute the vertical
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	order : int
		Order of accuracy.
	"""
	def __init__(self, grid, moist):
		super().__init__(grid, moist)
		self.nb = 1
		self.order = 2

	def __call__(
		self, i, j, k, dt, w, s, s_prv, su, su_prv, sv, sv_prv,
		sqv=None, sqv_prv=None, sqc=None, sqc_prv=None, sqr=None, sqr_prv=None
	):
		# Interpolate the vertical velocity at the model half-levels
		w_mid = gt.Equation()
		w_mid[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k-1])

		# Compute flux for the isentropic density and the momenta
		flux_s_z  = get_centered_flux(i, j, k, w_mid, s)
		flux_su_z = get_centered_flux(i, j, k, w_mid, su)
		flux_sv_z = get_centered_flux(i, j, k, w_mid, sv)

		# Initialize return list
		return_list = [flux_s_z, flux_su_z, flux_sv_z]

		if self._moist:
			# Compute flux for the water constituents
			flux_sqv_z = get_centered_flux(i, j, k, w_mid, sqv)
			flux_sqc_z = get_centered_flux(i, j, k, w_mid, sqc)
			flux_sqr_z = get_centered_flux(i, j, k, w_mid, sqr)

			# Update the return list
			return_list += [flux_sqv_z, flux_sqc_z, flux_sqr_z]

		return return_list


def get_centered_flux(i, j, k, w_mid, phi):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_z'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = w_mid[i, j, k] * 0.5 * (phi[i, j, k-1] + phi[i, j, k])

	return flux


class MacCormack(IsentropicVerticalFlux):
	"""
	Class which inherits
	:class:`~tasmania.dynamics.isentropic_fluxes.IsentropicVerticalFlux`
	to implement the MacCormack scheme to compute the vertical
	numerical fluxes for the governing equations expressed in
	conservative form using isentropic coordinates.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	order : int
		Order of accuracy.
	"""
	def __init__(self, grid, moist):
		super().__init__(grid, moist)
		self.nb = 1
		self.order = 2

	def __call__(
		self, i, j, k, dt, w, s, s_prv, su, su_prv, sv, sv_prv,
		sqv=None, sqv_prv=None, sqc=None, sqc_prv=None, sqr=None, sqr_prv=None
	):
		# Compute the predicted values for the isentropic density
		# and the momenta
		s_prd  = self._get_maccormack_vertical_predicted_value(
			i, j, k, dt, w, s, s_prv)
		su_prd = self._get_maccormack_vertical_predicted_value(
			i, j, k, dt, w, su, su_prv)
		sv_prd = self._get_maccormack_vertical_predicted_value(
			i, j, k, dt, w, sv, sv_prv)

		if self._moist:
			# Compute the predicted values for the water constituents
			sqv_prd = self._get_maccormack_vertical_predicted_value(
				i, j, k, dt, w, sqv, sqv_prv)
			sqc_prd = self._get_maccormack_vertical_predicted_value(
				i, j, k, dt, w, sqc, sqc_prv)
			sqr_prd = self._get_maccormack_vertical_predicted_value(
				i, j, k, dt, w, sqr, sqr_prv)

		# Compute the flux for the isentropic density and the momenta
		flux_s_z  = get_maccormack_flux(i, j, k, w, s, s_prd)
		flux_su_z = get_maccormack_flux(i, j, k, w, su, su_prd)
		flux_sv_z = get_maccormack_flux(i, j, k, w, sv, sv_prd)

		# Initialize the return list
		return_list = [flux_s_z, flux_su_z, flux_sv_z]

		if self._moist:
			# Compute the flux for the water constituents
			flux_sqv_z = get_maccormack_flux(i, j, k, w, sqv, sqv_prd)
			flux_sqc_z = get_maccormack_flux(i, j, k, w, sqc, sqc_prd)
			flux_sqr_z = get_maccormack_flux(i, j, k, w, sqr, sqr_prd)

			# Update the return list
			return_list += [flux_sqv_z, flux_sqc_z, flux_sqr_z]

		return return_list

	def _get_maccormack_vertical_predicted_value(self, i, j, k, dt, w, phi, phi_prv):
		phi_name = phi.get_name()
		phi_prd_name = phi_name + '_prd'
		phi_prd = gt.Equation(name=phi_prd_name)

		dz = self._grid.dz.values.item()
		phi_prd[i, j, k] = phi_prv[i, j, k] - dt * \
						   (w[i, j, k-1] * phi[i, j, k-1] -
							w[i, j,   k] * phi[i, j,   k]) / dz

		return phi_prd


def get_maccormack_flux(i, j, k, w, phi, phi_prd):
	phi_name = phi.get_name()
	flux_name = 'flux_' + phi_name + '_z'
	flux = gt.Equation(name=flux_name)

	flux[i, j, k] = 0.5 * (w[i, j, k-1] * phi[i, j, k-1] +
						   w[i, j, k] * phi_prd[i, j, k])

	return flux
