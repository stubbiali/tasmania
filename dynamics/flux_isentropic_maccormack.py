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

class FluxIsentropicMacCormack(FluxIsentropic):
	"""
	Class which inherits :class:`~tasmania.dycore.flux_isentropic.FluxIsentropic` to implement the MacCormack scheme to compute 
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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		"""
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def _compute_horizontal_fluxes(self, i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, in_Qv, in_Qc, in_Qr, 
								   in_qv_tnd = None, in_qc_tnd = None, in_qr_tnd = None):
		"""
		Method computing the :class:`gridtools.Equation`\s representing the MacCormack :math:`x`- 
		and :math:`y`-fluxes for all the conservative prognostic variables. 
		The :class:`gridtools.Equation`\s are then set as instance attributes.

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
		in_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of water vapor.
		in_qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of cloud liquid water.
		in_qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of precipitation water.
		"""
		# Diagnose unstaggered velocities
		tmp_u_unstg = gt.Equation()
		tmp_u_unstg[i, j, k] = in_U[i, j, k] / in_s[i, j, k]
		tmp_v_unstg = gt.Equation()
		tmp_v_unstg[i, j, k] = in_V[i, j, k] / in_s[i, j, k]

		# Compute predicted values for the isentropic density and the momentums
		tmp_s_prd = self._get_maccormack_horizontal_predicted_value_s(i, j, k, dt, in_s, in_U, in_V)
		tmp_U_prd = self._get_maccormack_horizontal_predicted_value_U(i, j, k, dt, in_s, tmp_u_unstg, 
																	  tmp_v_unstg, in_mtg, in_U)
		tmp_V_prd = self._get_maccormack_horizontal_predicted_value_V(i, j, k, dt, in_s, tmp_u_unstg, 
																	  tmp_v_unstg, in_mtg, in_V)

		if self._moist_on:
			# Compute predicted values for the water constituents
			tmp_Qv_prd = self._get_maccormack_horizontal_predicted_value_Q(i, j, k, dt, in_s, tmp_u_unstg, 
																		   tmp_v_unstg, in_Qv, in_qr_tnd)
			tmp_Qc_prd = self._get_maccormack_horizontal_predicted_value_Q(i, j, k, dt, in_s, tmp_u_unstg, 
																		   tmp_v_unstg, in_Qc, in_qc_tnd)
			tmp_Qr_prd = self._get_maccormack_horizontal_predicted_value_Q(i, j, k, dt, in_s, tmp_u_unstg, 
																		   tmp_v_unstg, in_Qr, in_qv_tnd)
		
		# Diagnose predicted values for the velocities
		tmp_u_prd_unstg = self._get_velocity(i, j, k, tmp_s_prd, tmp_U_prd)
		tmp_v_prd_unstg = self._get_velocity(i, j, k, tmp_s_prd, tmp_V_prd)

		# Compute the fluxes for the isentropic density and the momentums
		self._flux_s_x = self._get_maccormack_flux_x_s(i, j, k, in_U, tmp_U_prd)
		self._flux_s_y = self._get_maccormack_flux_y_s(i, j, k, in_V, tmp_V_prd)
		self._flux_U_x = self._get_maccormack_flux_x(i, j, k, tmp_u_unstg, in_U, tmp_u_prd_unstg, tmp_U_prd)
		self._flux_U_y = self._get_maccormack_flux_y(i, j, k, tmp_v_unstg, in_U, tmp_v_prd_unstg, tmp_U_prd)
		self._flux_V_x = self._get_maccormack_flux_x(i, j, k, tmp_u_unstg, in_V, tmp_u_prd_unstg, tmp_V_prd)
		self._flux_V_y = self._get_maccormack_flux_y(i, j, k, tmp_v_unstg, in_V, tmp_v_prd_unstg, tmp_V_prd)

		if self._moist_on:
			# Compute the fluxes for the water constituents
			self._flux_Qv_x = self._get_maccormack_flux_x(i, j, k, tmp_u_unstg, in_Qv, tmp_u_prd_unstg, tmp_Qv_prd)
			self._flux_Qv_y = self._get_maccormack_flux_y(i, j, k, tmp_v_unstg, in_Qv, tmp_v_prd_unstg, tmp_Qv_prd)
			self._flux_Qc_x = self._get_maccormack_flux_x(i, j, k, tmp_u_unstg, in_Qc, tmp_u_prd_unstg, tmp_Qc_prd)
			self._flux_Qc_y = self._get_maccormack_flux_y(i, j, k, tmp_v_unstg, in_Qc, tmp_v_prd_unstg, tmp_Qc_prd)
			self._flux_Qr_x = self._get_maccormack_flux_x(i, j, k, tmp_u_unstg, in_Qr, tmp_u_prd_unstg, tmp_Qr_prd)
			self._flux_Qr_y = self._get_maccormack_flux_y(i, j, k, tmp_v_unstg, in_Qr, tmp_v_prd_unstg, tmp_Qr_prd)

	def _compute_vertical_fluxes(self, i, j, k, dt, in_w, in_s, in_s_prv, in_U, in_U_prv, in_V, in_V_prv, 
								 in_Qv, in_Qv_prv, in_Qc, in_Qc_prv, in_Qr, in_Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`\s representing the MacCormack :math:`\\theta`-flux 
		for all the conservative model variables. 
		The :class:`gridtools.Equation`\s are then set as instance attributes.

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
			:class:`gridtools.Equation` representing the vertical velocity, 
			i.e., the change over time of potential temperature.
		in_s : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		in_s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, 
			i.e., the isentropic density stepped disregarding the vertical advection.
		in_U : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		in_U_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum, 
			i.e., the :math:`x`-momentum stepped disregarding the vertical advection.
		in_V : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		in_V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum, 
			i.e., the :math:`y`-momentum stepped disregarding the vertical advection.
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
		# Compute predicted values for the isentropic density and the momentums
		tmp_s_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, in_w, in_s, in_s_prv)
		tmp_U_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, in_w, in_U, in_U_prv)
		tmp_V_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, in_w, in_V, in_V_prv)

		if self._moist_on:
			# Compute predicted values for the water constituents
			tmp_Qv_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, in_w, in_Qv, in_Qv_prv)
			tmp_Qc_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, in_w, in_Qc, in_Qc_prv)
			tmp_Qr_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, in_w, in_Qr, in_Qr_prv)

		# Compute the flux for the isentropic density and the momentums
		self._flux_s_z = self._get_maccormack_flux_z(i, j, k, in_w, in_s, in_s_prv, tmp_s_prd)
		self._flux_U_z = self._get_maccormack_flux_z(i, j, k, in_w, in_U, in_U_prv, tmp_U_prd)
		self._flux_V_z = self._get_maccormack_flux_z(i, j, k, in_w, in_V, in_V_prv, tmp_V_prd)

		if self._moist_on:
			# Compute the flux for the water constituents
			self._flux_Qv_z = self._get_maccormack_flux_z(i, j, k, in_w, in_Qv, in_Qv_prv, tmp_Qv_prd)
			self._flux_Qc_z = self._get_maccormack_flux_z(i, j, k, in_w, in_Qc, in_Qc_prv, tmp_Qc_prd)
			self._flux_Qr_z = self._get_maccormack_flux_z(i, j, k, in_w, in_Qr, in_Qr_prv, tmp_Qr_prd)

	def _get_maccormack_horizontal_predicted_value_s(self, i, j, k, dt, in_s, in_U, in_V):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for the isentropic density,
		computed without taking the vertical advection into account.

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
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the isentropic density.
		"""
		tmp_s_prd = gt.Equation()
		tmp_s_prd[i, j, k] = in_s[i, j, k] - dt * ((in_U[i+1, j, k] - in_U[i, j, k]) / self._grid.dx + 
											 	   (in_V[i, j+1, k] - in_V[i, j, k]) / self._grid.dy)
		return tmp_s_prd	

	def _get_maccormack_horizontal_predicted_value_U(self, i, j, k, dt, in_s, tmp_u_unstg, tmp_v_unstg, in_mtg, in_U):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for the :math:`x`-momentum,
		computed without taking the vertical advection into account.

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
		tmp_u_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		tmp_v_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the :math:`x`-momentum.
		"""
		tmp_U_prd = gt.Equation()
		tmp_U_prd[i, j, k] = in_U[i, j, k] - dt * ((tmp_u_unstg[i+1,   j, k] * in_U[i+1,   j, k] - 
											  	 	tmp_u_unstg[  i,   j, k] * in_U[  i,   j, k]) / self._grid.dx + 
										 	 	   (tmp_v_unstg[  i, j+1, k] * in_U[  i, j+1, k] - 
											  	 	tmp_v_unstg[  i,   j, k] * in_U[  i,   j, k]) / self._grid.dy +
											 	   in_s[i, j, k] * (in_mtg[i+1, j, k] - in_mtg[i, j, k]) / self._grid.dx)
		return tmp_U_prd	

	def _get_maccormack_horizontal_predicted_value_V(self, i, j, k, dt, in_s, tmp_u_unstg, tmp_v_unstg, in_mtg, in_V):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for the :math:`y`-momentum,
		computed without taking the vertical advection into account.

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
		tmp_u_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		tmp_v_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the :math:`y`-momentum.
		"""
		tmp_V_prd = gt.Equation()
		tmp_V_prd[i, j, k] = in_V[i, j, k] - dt * ((tmp_u_unstg[i+1,   j, k] * in_V[i+1,   j, k] - 
											  	 	tmp_u_unstg[  i,   j, k] * in_V[  i,   j, k]) / self._grid.dx + 
										 	 	   (tmp_v_unstg[  i, j+1, k] * in_V[  i, j+1, k] - 
											  	 	tmp_v_unstg[  i,   j, k] * in_V[  i,   j, k]) / self._grid.dy +
											 	   in_s[i, j, k] * (in_mtg[i, j+1, k] - in_mtg[i, j, k]) / self._grid.dy)
		return tmp_V_prd	

	def _get_maccormack_horizontal_predicted_value_Q(self, i, j, k, dt, in_s, tmp_u_unstg, tmp_v_unstg, 
													 in_Q, in_q_tnd):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for the isentropic 
		density of a generic water constituent, computed without taking the vertical advection into account.
		
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
			:class:`gridtools.Equation` representing the air isentropic density.
		tmp_u_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		tmp_v_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		in_Q : obj
			:class:`gridtools.Equation` representing the isentropic density of a generic water constituent.
		in_q_tnd : obj
			:class:`gridtools.Equation` representing the tendency of the mass fraction of the water constituent.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the water constituent.
		"""
		in_Q_name = in_Q.get_name()
		tmp_Q_prd_name = in_Q_name + '_prd'
		tmp_Q_prd = gt.Equation(name = tmp_Q_prd_name)
		if in_q_tnd is None:
			tmp_Q_prd[i, j, k] = in_Q[i, j, k] - dt * ((tmp_u_unstg[i+1,   j, k] * in_Q[i+1,   j, k] -
											  	 		tmp_u_unstg[  i,   j, k] * in_Q[  i,   j, k]) / self._grid.dx + 
										 	 	   	   (tmp_v_unstg[  i, j+1, k] * in_Q[  i, j+1, k] - 
											  	 		tmp_v_unstg[  i,   j, k] * in_Q[  i,   j, k]) / self._grid.dy)
		else:
			tmp_Q_prd[i, j, k] = in_Q[i, j, k] - dt * ((tmp_u_unstg[i+1,   j, k] * in_Q[i+1,   j, k] -
											  	 		tmp_u_unstg[  i,   j, k] * in_Q[  i,   j, k]) / self._grid.dx + 
										 	 	   	   (tmp_v_unstg[  i, j+1, k] * in_Q[  i, j+1, k] - 
											  	 		tmp_v_unstg[  i,   j, k] * in_Q[  i,   j, k]) / self._grid.dy +
													   in_s[i, j, k] * in_q_tnd[i, j, k])
		return tmp_Q_prd	

	def _get_maccormack_vertical_predicted_value(self, i, j, k, dt, in_w, in_phi, in_phi_prv):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for a generic conservative prognostic 
		variable :math:`\phi`, computed taking only the vertical advection into account.
		
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
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time in potential temperature.
		in_phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at current time.
		in_phi_prv : obj
			:class:`gridtools.Equation` representing the provisional value for :math:`\phi`, i.e., :math:`\phi` stepped 
			disregarding the vertical advection.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for :math:`\phi`.
		"""
		phi_name = in_phi.get_name()
		phi_prd_name = phi_name + '_prd'
		tmp_phi_prd = gt.Equation(name = phi_prd_name)
		tmp_phi_prd[i, j, k] = in_phi_prv[i, j, k] - dt * (in_w[i, j, k-1] * in_phi[i, j, k-1] -
											  	 	in_w[i, j,   k] * in_phi[i, j,   k]) / self._grid.dz
		return tmp_phi_prd	

	def _get_velocity(self, i, j, k, s, mnt):
		"""
		Get the :class:`gridtools.Equation` representing an unstaggered velocity component.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		s : obj
			:class:`gridtools.Equation` representing the isentropic density.
		mnt : obj
			:class:`gridtools.Equation` representing either the :math:`x`- or the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the diagnosed unstaggered velocity component.
		"""
		vel_name = mnt.get_name().lower()
		vel = gt.Equation(name = vel_name)
		vel[i, j, k] = mnt[i, j, k] / s[i, j, k]
		return vel

	def _get_maccormack_flux_x(self, i, j, k, tmp_u_unstg, in_phi, tmp_u_prd_unstg, tmp_phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction 
		for a generic prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		tmp_u_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity at the current time.
		in_phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at the current time.
		tmp_u_prd_unstg : obj
			:class:`gridtools.Equation` representing the predicted value for the unstaggered :math:`x`-velocity.
		tmp_phi_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for :math:`\phi`.
		"""
		phi_name = in_phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (tmp_u_unstg[i+1, j, k] * in_phi[i+1, j, k] + 
							   tmp_u_prd_unstg[i, j, k] * tmp_phi_prd[i, j, k])
		return flux

	def _get_maccormack_flux_x_s(self, i, j, k, in_U, tmp_U_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction 
		for the isentropic density.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum at the current time.
		tmp_U_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the :math:`x`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for the isentropic density.
		"""
		flux_in_s_x = gt.Equation()
		flux_in_s_x[i, j, k] = 0.5 * (in_U[i+1, j, k] + tmp_U_prd[i, j, k])
		return flux_in_s_x

	def _get_maccormack_flux_y(self, i, j, k, tmp_v_unstg, in_phi, tmp_v_prd_unstg, tmp_phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction 
		for a generic prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		tmp_v_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity at the current time.
		in_phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at the current time.
		tmp_v_prd_unstg : obj
			:class:`gridtools.Equation` representing the predicted value for the unstaggered :math:`y`-velocity.
		tmp_phi_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for :math:`\phi`.
		"""
		phi_name = in_phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (tmp_v_unstg[i, j+1, k] * in_phi[i, j+1, k] + 
							   tmp_v_prd_unstg[i, j, k] * tmp_phi_prd[i, j, k])
		return flux

	def _get_maccormack_flux_y_s(self, i, j, k, in_V, tmp_V_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction 
		for the isentropic density.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum at the current time.
		tmp_V_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for the isentropic density.
		"""
		flux_in_s_y = gt.Equation()
		flux_in_s_y[i, j, k] = 0.5 * (in_V[i, j+1, k] + tmp_V_prd[i, j, k])
		return flux_in_s_y

	def _get_maccormack_flux_z(self, i, j, k, in_w, in_phi, in_phi_prv, tmp_phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`\\theta`-direction for a 
		generic prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_w : obj
			:class:`gridtools.Equation` representing the vertical velocity, 
			i.e., the change over time in potential temperature.
		in_phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at current time.
		in_phi_prv : obj
			:class:`gridtools.Equation` representing the provisional value for :math:`\phi`, 
			i.e., :math:`\phi` stepped disregarding the vertical advection.
		tmp_phi_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`\\theta`-direction for :math:`\phi`.
		"""
		in_phi_name = in_phi.get_name()
		flux_name = 'flux_' + in_phi_name + '_z'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (in_w[i, j, k-1] * in_phi[i, j, k-1] + in_w[i, j, k] * tmp_phi_prd[i, j, k])
		return flux
