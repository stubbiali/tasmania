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
import gridtools as gt
from tasmania.dycore.flux_isentropic import FluxIsentropic

class FluxIsentropicUpwind(FluxIsentropic):
	"""
	Class which inherits :class:`~dycore.flux_isentropic.FluxIsentropic` to implement the upwind scheme to compute 
	the numerical fluxes for the governing equations expressed in conservative form using isentropic coordinates.

	Attributes
	----------
	nb : int
		Number of boundary layers.
	order : int
		Order of accuracy.
	"""
	def __init__(self, grid, moist_on):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 1

	def _compute_horizontal_fluxes(self, i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, in_Qv, in_Qc, in_Qr):
		"""
		Method computing the :class:`gridtools.Equation`_s representing the upwind :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`_s are then set as instance attributes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		in_Qv : obj
			:class:`gridtools.Equation` representing the isentropic density of water vapour.
		in_Qc : obj
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		in_Qr : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		"""
		# Compute fluxes for the isentropic density and the momentums
		self._flux_s_x = self._get_upwind_flux_x(i, j, k, in_u, in_s)
		self._flux_s_y = self._get_upwind_flux_y(i, j, k, in_v, in_s)
		self._flux_U_x = self._get_upwind_flux_x(i, j, k, in_u, in_U)
		self._flux_U_y = self._get_upwind_flux_y(i, j, k, in_v, in_U)
		self._flux_V_x = self._get_upwind_flux_x(i, j, k, in_u, in_V)
		self._flux_V_y = self._get_upwind_flux_y(i, j, k, in_v, in_V)
		
		if self._moist_on:
			# Compute fluxes for the water constituents
			self._flux_Qv_x = self._get_upwind_flux_x(i, j, k, in_u, in_Qv)
			self._flux_Qv_y = self._get_upwind_flux_y(i, j, k, in_v, in_Qv)
			self._flux_Qc_x = self._get_upwind_flux_x(i, j, k, in_u, in_Qc)
			self._flux_Qc_y = self._get_upwind_flux_y(i, j, k, in_v, in_Qc)
			self._flux_Qr_x = self._get_upwind_flux_x(i, j, k, in_u, in_Qr)
			self._flux_Qr_y = self._get_upwind_flux_y(i, j, k, in_v, in_Qr)

	def _compute_vertical_fluxes(self, i, j, k, dt, in_w, in_s, in_s_prv, in_U, in_U_prv, in_V, in_V_prv, 
								 in_Qv, in_Qv_prv, in_Qc, in_Qc_prv, in_Qr, in_Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`_s representing the upwind :math:`z`-flux for all the conservative 
		model variables. The :class:`gridtools.Equation`_s are then set as instance attributes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time of potential temperature.
		in_s : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		in_s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		in_U : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		in_U_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum, i.e., the :math:`x`-momentum stepped
			disregarding the vertical advection.
		in_V : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		in_V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum, i.e., the :math:`y`-momentum stepped
			disregarding the vertical advection.
		in_Qv : obj
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		in_Qv_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		in_Qc : obj			
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		in_Qc_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		in_Qr : obj
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		in_Qr_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.
		"""
		# Interpolate the vertical velocity at the model half-levels
		tmp_w_mid = gt.Equation()
		tmp_w_mid[i, j, k] = 0.5 * (in_w[i, j, k] + in_w[i, j, k-1])

		# Compute flux for the isentropic density and the momentums
		self._flux_s_z = self._get_upwind_flux_z(i, j, k, tmp_w_mid, in_s)
		self._flux_U_z = self._get_upwind_flux_z(i, j, k, tmp_w_mid, in_U)
		self._flux_V_z = self._get_upwind_flux_z(i, j, k, tmp_w_mid, in_V)
		
		if self._moist_on:
			# Compute flux for the water constituents
			self._flux_Qv_z = self._get_upwind_flux_z(i, j, k, tmp_w_mid, in_Qv)
			self._flux_Qc_z = self._get_upwind_flux_z(i, j, k, tmp_w_mid, in_Qc)
			self._flux_Qr_z = self._get_upwind_flux_z(i, j, k, tmp_w_mid, in_Qr)

	def _get_upwind_flux_x(self, i, j, k, in_u, in_phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind flux in :math:`x`-direction for a generic 
		prognostic variable :math:`phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		in_phi : obj
			:class:`gridtools.Equation` representing the field :math:`phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the upwind flux in :math:`x`-direction for :math:`phi`.
		"""
		# Note: by default, a GT4Py's Equation instance is named with the name used by the user 
		# to reference the object itself. Here, this is likely to be dangerous as 
		# this method is called on multiple instances of the Equation class. Hence, we explicitly 
		# set the name for the flux based on the name of the prognostic variable.
		in_phi_name = in_phi.get_name()
		flux_name = 'flux_' + in_phi_name + '_x'
		flux = gt.Equation(name = flux_name)

		flux[i, j, k] = in_u[i+1, j, k] * ((in_u[i+1, j, k] > 0.) * in_phi[  i, j, k] + 
									  	   (in_u[i+1, j, k] < 0.) * in_phi[i+1, j, k])
									  

		return flux

	def _get_upwind_flux_y(self, i, j, k, in_v, in_phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind flux in :math:`y`-direction for a generic 
		prognostic variable :math:`phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		in_phi : obj
			:class:`gridtools.Equation` representing the field :math:`phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the upwind flux in :math:`y`-direction for :math:`phi`.
		"""
		# Note: by default, a GT4Py's Equation instance is named with the name used by the user 
		# to reference the object itself. Here, this is likely to be dangerous as 
		# this method is called on multiple instances of the Equation class. Hence, we explicitly 
		# set the name for the flux based on the name of the prognostic variable.
		in_phi_name = in_phi.get_name()
		flux_name = 'flux_' + in_phi_name + '_y'
		flux = gt.Equation(name = flux_name)

		flux[i, j, k] = in_v[i, j+1, k] * ((in_v[i, j+1, k] > 0.) * in_phi[i,   j, k] +
									  	   (in_v[i, j+1, k] < 0.) * in_phi[i, j+1, k])

		return flux

	def _get_upwind_flux_z(self, i, j, k, tmp_w_mid, in_phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind flux in :math:`z`-direction for a generic 
		prognostic variable :math:`phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		tmp_w_mid : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time in
			potential temperature, at the model half levels.
		in_phi : obj
			:class:`gridtools.Equation` representing the field :math:`phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the upwind flux in :math:`z`-direction for :math:`phi`.
		"""
		# Note: by default, a GT4Py's Equation instance is named with the name used by the user 
		# to reference the object itself. Here, this is likely to be dangerous as 
		# this method is called on multiple instances of the Equation class. Hence, we explicitly 
		# set the name for the flux based on the name of the prognostic variable.
		in_phi_name = in_phi.get_name()
		flux_name = 'flux_' + in_phi_name + '_z'
		flux = gt.Equation(name = flux_name)

		flux[i, j, k] = tmp_w_mid[i, j, k] * ((tmp_w_mid[i, j, k] > 0.) * in_phi[i, j, k] +
										  	  (tmp_w_mid[i, j, k] < 0.) * in_phi[i, j, k-1])

		return flux