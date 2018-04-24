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
import abc

class FluxIsentropic:
	"""
	Abstract base class whose derived classes implement different schemes to compute the numerical fluxes for 
	the three-dimensional isentropic dynamical core. The conservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

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
		self._grid = grid
		self._moist_on = moist_on

	def get_horizontal_fluxes(self, i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
							  in_Qv = None, in_Qc = None, in_Qr = None,
							  in_qv_tnd = None, in_qc_tnd = None, in_qr_tnd = None):
		"""
		Method returning the :class:`gridtools.Equation`~s representing the :math:`x`- and :math:`y`-fluxes 
		for all the conservative model variables.

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
		in_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of water vapor.
		in_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		in_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		in_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of water vapor.
		in_qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of cloud liquid water.
		in_qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of precipitation water.

		Returns
		-------
		flux_s_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density.
		flux_s_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density.
		flux_U_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the :math:`x`-momentum.
		flux_U_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the :math:`x`-momentum.
		flux_V_x : obj
			:class:`gridtools.Equation` representing the :math:`x`-flux for the :math:`y`-momentum.
		flux_V_y : obj
			:class:`gridtools.Equation` representing the :math:`y`-flux for the :math:`y`-momentum.
		flux_Qv_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of water vapor.
		flux_Qv_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of water vapor.
		flux_Qc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of cloud liquid water.
		flux_Qc_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of cloud liquid water.
		flux_Qr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of precipitation water.
		flux_Qr_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of precipitation water.
		"""
		self._compute_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
										in_Qv, in_Qc, in_Qr,
										in_qv_tnd, in_qc_tnd, in_qr_tnd)
		if not self._moist_on:
			return self._flux_s_x, self._flux_s_y, \
				   self._flux_U_x, self._flux_U_y, \
				   self._flux_V_x, self._flux_V_y
		else:
			return self._flux_s_x,  self._flux_s_y,  \
				   self._flux_U_x,  self._flux_U_y,  \
				   self._flux_V_x,  self._flux_V_y,  \
				   self._flux_Qv_x, self._flux_Qv_y, \
				   self._flux_Qc_x, self._flux_Qc_y, \
				   self._flux_Qr_x, self._flux_Qr_y

	def get_vertical_fluxes(self, i, j, k, dt, in_w, in_s, in_s_prv, in_U, in_U_prv, in_V, in_V_prv, 
							in_Qv = None, in_Qv_prv = None, in_Qc = None, in_Qc_prv = None,	in_Qr = None, in_Qr_prv = None):
		"""
		Method returning the :class:`gridtools.Equation`~s representing the :math:`z`-flux for all the conservative 
		model variables.

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
			i.e., the change over time in potential temperature.
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
		in_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		in_Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		in_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		in_Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		in_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		in_Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.

		Returns
		-------
		flux_s_z : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density.
		flux_U_z : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the :math:`x`-momentum.
		flux_V_z : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the :math:`y`-momentum.
		flux_Qv_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of water vapor.
		flux_Qc_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of cloud water.
		flux_Qr_z : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of precipitation water.
		"""
		self._compute_vertical_fluxes(i, j, k, dt, in_w, in_s, in_s_prv, in_U, in_U_prv, in_V, in_V_prv, 
									  in_Qv, in_Qv_prv, in_Qc, in_Qc_prv, in_Qr, in_Qr_prv)
		if self._moist_on:
			return self._flux_s_z, self._flux_U_z, self._flux_V_z, self._flux_Qv_z, self._flux_Qc_z, self._flux_Qr_z
		else:
			return self._flux_s_z, self._flux_U_z, self._flux_V_z

	@staticmethod
	def factory(scheme, grid, moist_on):
		"""
		Static method which returns an instance of the derived class implementing the numerical scheme
		specified by :data:`scheme`.

		Parameters
		----------
		scheme : str
			String specifying the numerical scheme to implement. Either:

			* 'upwind', for the upwind scheme;
			* 'centered', for a second-order centered scheme;
			* 'maccormack', for the MacCormack scheme.

		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.

		Return
		------
		obj :
			Instance of the derived class implementing the scheme specified by :data:`scheme`.
		"""
		if scheme == 'upwind':
			from tasmania.dycore.flux_isentropic_upwind import FluxIsentropicUpwind
			return FluxIsentropicUpwind(grid, moist_on)
		elif scheme == 'centered':
			from tasmania.dycore.flux_isentropic_centered import FluxIsentropicCentered
			return FluxIsentropicCentered(grid, moist_on)
		else:
			from tasmania.dycore.flux_isentropic_maccormack import FluxIsentropicMacCormack
			return FluxIsentropicMacCormack(grid, moist_on)

	@abc.abstractmethod
	def _compute_horizontal_fluxes(self, i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, in_Qv, in_Qc, in_Qr, 
								   in_qv_tnd = None, in_qc_tnd = None, in_qr_tnd = None):
		"""
		Method computing the :class:`gridtools.Equation`~s representing the :math:`x`- and 
		:math:`y`-fluxes for all the conservative prognostic variables. 
		The :class:`gridtools.Equation`~s are then set as instance attributes.
		As this method is marked as abstract, the implementation is delegated to the derived classes.

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
			:class:`gridtools.Equation` representing the isentropic density of water vapor.
		in_Qc : obj
			:class:`gridtools.Equation` representing the isentropic density of cloud liquid water.
		in_Qr : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		in_qv_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of water vapor.
		in_qc_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of cloud liquid water.
		in_qr_tnd : `obj`, optional
			:class:`gridtools.Equation` representing the tendency of the mass fraction of precipitation water.
		"""

	@abc.abstractmethod
	def _compute_vertical_fluxes(self, i, j, k, dt, in_w, in_s, in_s_prv, in_U, in_U_prv, in_V, in_V_prv, 
								 in_Qv, in_Qv_prv, in_Qc, in_Qc_prv, in_Qr, in_Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`~s representing the :math:`z`-flux 
		for all the conservative model variables. The :class:`gridtools.Equation`~s are then 
		set as instance attributes.
		As this method is marked as abstract, the implementation is delegated to the derived classes.

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
			i.e., the change over time in potential temperature.
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
