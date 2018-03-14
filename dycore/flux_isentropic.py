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
Numerical fluxes for the three-dimensional moist isentropic dynamical core.
"""
import abc
import numpy

import gridtools as gt

class FluxIsentropic:
	"""
	Abstract base class whose derived classes implement different schemes for computing the numerical fluxes for 
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

	def get_horizontal_fluxes(self, i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, 
							  Qv_now = None, Qc_now = None, Qr_now = None):
		"""
		Method returning the :class:`gridtools.Equation`\s_now representing the :math:`x`- and :math:`y`-fluxes 
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
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg_now : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv_now : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of water vapour.
		Qc_now : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		Qr_now : `obj`, optional
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.

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
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of water vapour.
		flux_Qv_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of water vapour.
		flux_Qc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of cloud water.
		flux_Qc_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of cloud water.
		flux_Qr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`x`-flux for the isentropic density of precipitation water.
		flux_Qr_y : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`y`-flux for the isentropic density of precipitation water.
		"""
		self._compute_horizontal_fluxes(i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, Qv_now, Qc_now, Qr_now)
		if self._moist_on:
			return self._flux_s_x, self._flux_s_y, self._flux_U_x, self._flux_U_y, self._flux_V_x, self._flux_V_y, \
				   self._flux_Qv_x, self._flux_Qv_y, self._flux_Qc_x, self._flux_Qc_y, self._flux_Qr_x, self._flux_Qr_y
		else:
			return self._flux_s_x, self._flux_s_y, self._flux_U_x, self._flux_U_y, self._flux_V_x, self._flux_V_y

	def get_vertical_fluxes(self, i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
							Qv_now = None, Qv_prv = None, Qc_now = None, Qc_prv = None, Qr_now = None, Qr_prv = None):
		"""
		Method returning the :class:`gridtools.Equation`\s_now representing the :math:`z`-flux for all the conservative 
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
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time of potential temperature.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		U_now : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		U_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum, i.e., the :math:`x`-momentum stepped
			disregarding the vertical advection.
		V_now : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum, i.e., the :math:`y`-momentum stepped
			disregarding the vertical advection.
		Qv_now : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		Qc_now : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		Qr_now : `obj`, optional
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.

		Returns
		-------
		flux_s_z : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density.
		flux_U_x : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the :math:`x`-momentum.
		flux_V_x : obj
			:class:`gridtools.Equation` representing the :math:`z`-flux for the :math:`y`-momentum.
		flux_Qv_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of water vapour.
		flux_Qc_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of cloud water.
		flux_Qr_x : `obj`, optional
			:class:`gridtools.Equation` representing the :math:`z`-flux for the isentropic density of precipitation water.
		"""
		self._compute_vertical_fluxes(i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
									  Qv_now, Qv_prv, Qc_now, Qc_prv, Qr_now, Qr_prv)
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
			return FluxIsentropicUpwind(grid, moist_on)
		elif scheme == 'centered':
			return FluxIsentropicCentered(grid, moist_on)
		else:
			return FluxIsentropicMacCormack(grid, moist_on)

	@abc.abstractmethod
	def _compute_horizontal_fluxes(self, i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, Qv_now, Qc_now, Qr_now):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.
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
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg_now : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv_now : obj
			:class:`gridtools.Equation` representing the isentropic density of water vapour.
		Qc_now : obj
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		Qr_now : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		"""

	@abc.abstractmethod
	def _compute_vertical_fluxes(self, i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
								 Qv_now, Qv_prv, Qc_now, Qc_prv, Qr_now, Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the :math:`z`-flux for all the conservative 
		model variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.
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
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time of potential temperature.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		U_now : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		U_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum, i.e., the :math:`x`-momentum stepped
			disregarding the vertical advection.
		V_now : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum, i.e., the :math:`y`-momentum stepped
			disregarding the vertical advection.
		Qv_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qv_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		Qc_now : obj			
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		Qc_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		Qr_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qr_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.
		"""


class FluxIsentropicUpwind(FluxIsentropic):
	"""
	Class which inherits :class:`FluxIsentropic` to implement the upwind scheme applied to the governing equations
	in conservative form.

	Attributes
	----------
	nb : int
		Number of boundary layers.
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

	def _compute_horizontal_fluxes(self, i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, Qv_now, Qc_now, Qr_now):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the upwind :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.

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
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg_now : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv_now : obj
			:class:`gridtools.Equation` representing the isentropic density of water vapour.
		Qc_now : obj
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		Qr_now : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		"""
		# Compute fluxes for the isentropic density and the momentums
		self._flux_s_x = self._get_upwind_flux_x(i, j, k, u_now, s_now)
		self._flux_s_y = self._get_upwind_flux_y(i, j, k, v_now, s_now)
		self._flux_U_x = self._get_upwind_flux_x(i, j, k, u_now, U_now)
		self._flux_U_y = self._get_upwind_flux_y(i, j, k, v_now, U_now)
		self._flux_V_x = self._get_upwind_flux_x(i, j, k, u_now, V_now)
		self._flux_V_y = self._get_upwind_flux_y(i, j, k, v_now, V_now)
		
		if self._moist_on:
			# Compute fluxes for the water constituents
			self._flux_Qv_x = self._get_upwind_flux_x(i, j, k, u_now, Qv_now)
			self._flux_Qv_y = self._get_upwind_flux_y(i, j, k, v_now, Qv_now)
			self._flux_Qc_x = self._get_upwind_flux_x(i, j, k, u_now, Qc_now)
			self._flux_Qc_y = self._get_upwind_flux_y(i, j, k, v_now, Qc_now)
			self._flux_Qr_x = self._get_upwind_flux_x(i, j, k, u_now, Qr_now)
			self._flux_Qr_y = self._get_upwind_flux_y(i, j, k, v_now, Qr_now)

	def _compute_vertical_fluxes(self, i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
								 Qv_now, Qv_prv, Qc_now, Qc_prv, Qr_now, Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the upwind :math:`z`-flux for all the conservative 
		model variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.

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
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time of potential temperature.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		U_now : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		U_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum, i.e., the :math:`x`-momentum stepped
			disregarding the vertical advection.
		V_now : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum, i.e., the :math:`y`-momentum stepped
			disregarding the vertical advection.
		Qv_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qv_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		Qc_now : obj			
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		Qc_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		Qr_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qr_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.
		"""
		# Interpolate the vertical velocity at the model half-levels
		w_mid = gt.Equation()
		w_mid[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k+1])

		# Compute flux for the isentropic density and the momentums
		self._flux_s_z = self._get_upwind_flux_z(i, j, k, w_mid, s_now)
		self._flux_U_z = self._get_upwind_flux_z(i, j, k, w_mid, U_now)
		self._flux_V_z = self._get_upwind_flux_z(i, j, k, w_mid, V_now)
		
		if self._moist_on:
			# Compute flux for the water constituents
			self._flux_Qv_z = self._get_upwind_flux_z(i, j, k, w_mid, Qv_now)
			self._flux_Qc_z = self._get_upwind_flux_z(i, j, k, w_mid, Qc_now)
			self._flux_Qr_z = self._get_upwind_flux_z(i, j, k, w_mid, Qr_now)

	def _get_upwind_flux_x(self, i, j, k, u, phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind flux in :math:`x`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the upwind flux in :math:`x`-direction for :math:`\phi`.
		"""
		# Note: by default, a GT4Py's_now Equation instance is named with the name used by the user 
		# to reference the object itself. Here, this is likely to be dangerous as 
		# this method is called on multiple instances of the Equation class. Hence, we explicitly 
		# set the name for the flux based on the name of the prognostic variable.
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name = flux_name)

		flux[i, j, k] = u[i+1, j, k] * ((u[i+1, j, k] > 0.) * phi[  i, j, k] + 
									  	(u[i+1, j, k] < 0.) * phi[i+1, j, k])
									  

		return flux

	def _get_upwind_flux_y(self, i, j, k, v, phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind flux in :math:`y`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the upwind flux in :math:`y`-direction for :math:`\phi`.
		"""
		# Note: by default, a GT4Py's_now Equation instance is named with the name used by the user 
		# to reference the object itself. Here, this is likely to be dangerous as 
		# this method is called on multiple instances of the Equation class. Hence, we explicitly 
		# set the name for the flux based on the name of the prognostic variable.
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name = flux_name)

		flux[i, j, k] = v[i, j+1, k] * ((v[i, j+1, k] > 0.) * phi[i,   j, k] +
									  	(v[i, j+1, k] < 0.) * phi[i, j+1, k])

		return flux

	def _get_upwind_flux_z(self, i, j, k, w_mid, phi):
		"""
		Get the :class:`gridtools.Equation` representing the upwind flux in :math:`z`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		w_mid : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time in
			potential temperature, at the model half levels.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the upwind flux in :math:`z`-direction for :math:`\phi`.
		"""
		# Note: by default, a GT4Py's_now Equation instance is named with the name used by the user 
		# to reference the object itself. Here, this is likely to be dangerous as 
		# this method is called on multiple instances of the Equation class. Hence, we explicitly 
		# set the name for the flux based on the name of the prognostic variable.
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_z'
		flux = gt.Equation(name = flux_name)

		flux[i, j, k] = w_mid[i, j, k] * ((w_mid[i, j, k] > 0.) * phi[i, j,   k] +
										  (w_mid[i, j, k] < 0.) * phi[i, j, k+1])

		return flux


class FluxIsentropicCentered(FluxIsentropic):
	"""
	Class which inherits :class:`FluxIsentropic` to implement the centered scheme applied to the governing equations
	in conservative form.

	Attributes
	----------
	nb : int
		Number of boundary layers.
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

	def _compute_horizontal_fluxes(self, i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, Qv_now, Qc_now, Qr_now):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the centered :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.

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
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg_now : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv_now : obj
			:class:`gridtools.Equation` representing the isentropic density of water vapour.
		Qc_now : obj
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		Qr_now : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		"""
		# Compute fluxes for the isentropic density and the momentums
		self._flux_s_x = self._get_centered_flux_x(i, j, k, u_now, s_now)
		self._flux_s_y = self._get_centered_flux_y(i, j, k, v_now, s_now)
		self._flux_U_x = self._get_centered_flux_x(i, j, k, u_now, U_now)
		self._flux_U_y = self._get_centered_flux_y(i, j, k, v_now, U_now)
		self._flux_V_x = self._get_centered_flux_x(i, j, k, u_now, V_now)
		self._flux_V_y = self._get_centered_flux_y(i, j, k, v_now, V_now)
		
		if self._moist_on:
			# Compute fluxes for the water constituents
			self._flux_Qv_x = self._get_centered_flux_x(i, j, k, u_now, Qv_now)
			self._flux_Qv_y = self._get_centered_flux_y(i, j, k, v_now, Qv_now)
			self._flux_Qc_x = self._get_centered_flux_x(i, j, k, u_now, Qc_now)
			self._flux_Qc_y = self._get_centered_flux_y(i, j, k, v_now, Qc_now)
			self._flux_Qr_x = self._get_centered_flux_x(i, j, k, u_now, Qr_now)
			self._flux_Qr_y = self._get_centered_flux_y(i, j, k, v_now, Qr_now)

	def _compute_vertical_fluxes(self, i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
								 Qv_now, Qv_prv, Qc_now, Qc_prv, Qr_now, Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the centered :math:`z`-flux for all the conservative 
		model variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.

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
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time of potential temperature.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		U_now : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		U_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum, i.e., the :math:`x`-momentum stepped
			disregarding the vertical advection.
		V_now : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum, i.e., the :math:`y`-momentum stepped
			disregarding the vertical advection.
		Qv_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qv_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		Qc_now : obj			
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		Qc_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		Qr_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qr_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.
		"""
		# Interpolate the vertical velocity at the model half-levels
		w_mid = gt.Equation()
		w_mid[i, j, k] = 0.5 * (w[i, j, k] + w[i, j, k+1])

		# Compute flux for the isentropic density and the momentums
		self._flux_s_z = self._get_upwind_flux_z(i, j, k, w_mid, s_now)
		self._flux_U_z = self._get_upwind_flux_z(i, j, k, w_mid, U_now)
		self._flux_V_z = self._get_upwind_flux_z(i, j, k, w_mid, V_now)
		
		if self._moist_on:
			# Compute flux for the water constituents
			self._flux_Qv_z = self._get_upwind_flux_z(i, j, k, w_mid, Qv_now)
			self._flux_Qc_z = self._get_upwind_flux_z(i, j, k, w_mid, Qc_now)
			self._flux_Qr_z = self._get_upwind_flux_z(i, j, k, w_mid, Qr_now)

	def _get_centered_flux_x(self, i, j, k, u, phi):
		"""
		Get the :class:`gridtools.Equation` representing the centered flux in :math:`x`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the centered flux in :math:`x`-direction for :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = u[i+1, j, k] * 0.5 * (phi[i, j, k] + phi[i+1, j, k])
		return flux

	def _get_centered_flux_y(self, i, j, k, v, phi):
		"""
		Get the :class:`gridtools.Equation` representing the centered flux in :math:`y`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the centered flux in :math:`y`-direction for :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = v[i, j+1, k] * 0.5 * (phi[i, j, k] + phi[i, j+1, k])
		return flux
	
	def _get_centered_flux_z(self, i, j, k, w_mid, phi_now):
		"""
		Get the :class:`gridtools.Equation` representing the centered flux in :math:`z`-direction for a generic 
		prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		w_mid : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time in
			potential temperature, at the model half levels.
		phi : obj
			:class:`gridtools.Equation` representing the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the centered flux in :math:`z`-direction for :math:`\phi`.
		"""
		phi_name = phi.get_name()
		flux_name = 'flux_' + phi_name + '_z'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = w_mid[i, j, k] * 0.5 * (phi[i, j, k] + phi[i, j, k+1])
		return flux


class FluxIsentropicMacCormack(FluxIsentropic):
	"""
	Class which inherits :class:`FluxIsentropic` to implement the MacCormack scheme applied to the
	governing equations in conservative form.

	Attributes
	----------
	nb : int
		Number of boundary layers.
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

	def _compute_horizontal_fluxes(self, i, j, k, dt, s_now, u_now, v_now, mtg_now, U_now, V_now, Qv_now, Qc_now, Qr_now):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the MacCormack :math:`x`- and :math:`y`-fluxes for all 
		the conservative prognostic variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.

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
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity.
		v_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity.
		mtg_now : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.
		Qv_now : obj
			:class:`gridtools.Equation` representing the isentropic density of water vapour.
		Qc_now : obj
			:class:`gridtools.Equation` representing the isentropic density of cloud water.
		Qr_now : obj
			:class:`gridtools.Equation` representing the isentropic density of precipitation water.
		"""
		# Diagnose unstaggered velocities
		u_now_unstg = gt.Equation()
		u_now_unstg[i, j, k] = U_now[i, j, k] / s_now[i, j, k]
		v_now_unstg = gt.Equation()
		v_now_unstg[i, j, k] = V_now[i, j, k] / s_now[i, j, k]

		# Compute predicted values for the isentropic density and the momentums
		s_prd = self._get_maccormack_horizontal_predicted_value_density(i, j, k, dt, s_now, U_now, V_now)
		U_prd = self._get_maccormack_horizontal_predicted_value_momentum_x(i, j, k, dt, s_now, u_now_unstg, v_now_unstg, 
																		   mtg_now, U_now)
		V_prd = self._get_maccormack_horizontal_predicted_value_momentum_y(i, j, k, dt, s_now, u_now_unstg, v_now_unstg, 
																		   mtg_now, V_now)

		if self._moist_on:
			# Compute predicted values for the water constituents
			Qv_prd = self._get_maccormack_horizontal_predicted_value_constituent(i, j, k, dt, u_now_unstg, v_now_unstg, Qv_now)
			Qc_prd = self._get_maccormack_horizontal_predicted_value_constituent(i, j, k, dt, u_now_unstg, v_now_unstg, Qc_now)
			Qr_prd = self._get_maccormack_horizontal_predicted_value_constituent(i, j, k, dt, u_now_unstg, v_now_unstg, Qr_now)
		
		# Diagnose predicted values for the velocities
		u_prd_unstg = self._get_velocity(i, j, k, s_prd, U_prd)
		v_prd_unstg = self._get_velocity(i, j, k, s_prd, V_prd)

		# Compute the fluxes for the isentropic density and the momentums
		self._flux_s_x = self._get_maccormack_flux_x_density(i, j, k, U_now, U_prd)
		self._flux_s_y = self._get_maccormack_flux_y_density(i, j, k, V_now, V_prd)
		self._flux_U_x = self._get_maccormack_flux_x(i, j, k, u_now_unstg, U_now, u_prd_unstg, U_prd)
		self._flux_U_y = self._get_maccormack_flux_y(i, j, k, v_now_unstg, U_now, v_prd_unstg, U_prd)
		self._flux_V_x = self._get_maccormack_flux_x(i, j, k, u_now_unstg, V_now, u_prd_unstg, V_prd)
		self._flux_V_y = self._get_maccormack_flux_y(i, j, k, v_now_unstg, V_now, v_prd_unstg, V_prd)

		if self._moist_on:
			# Compute the fluxes for the water constituents
			self._flux_Qv_x = self._get_maccormack_flux_x(i, j, k, u_now_unstg, Qv_now, u_prd_unstg, Qv_prd)
			self._flux_Qv_y = self._get_maccormack_flux_y(i, j, k, v_now_unstg, Qv_now, v_prd_unstg, Qv_prd)
			self._flux_Qc_x = self._get_maccormack_flux_x(i, j, k, u_now_unstg, Qc_now, u_prd_unstg, Qc_prd)
			self._flux_Qc_y = self._get_maccormack_flux_y(i, j, k, v_now_unstg, Qc_now, v_prd_unstg, Qc_prd)
			self._flux_Qr_x = self._get_maccormack_flux_x(i, j, k, u_now_unstg, Qr_now, u_prd_unstg, Qr_prd)
			self._flux_Qr_y = self._get_maccormack_flux_y(i, j, k, v_now_unstg, Qr_now, v_prd_unstg, Qr_prd)

	def _compute_vertical_fluxes(self, i, j, k, dt, w, s_now, s_prv, U_now, U_prv, V_now, V_prv, 
								 Qv_now, Qv_prv, Qc_now, Qc_prv, Qr_now, Qr_prv):
		"""
		Method computing the :class:`gridtools.Equation`\s_now representing the MacCormack :math:`z`-flux for all the conservative 
		model variables. The :class:`gridtools.Equation`s_now are then set as instance attributes.

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
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time of potential temperature.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density.
		s_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density, i.e., the isentropic density stepped
			disregarding the vertical advection.
		U_now : obj
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		U_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum, i.e., the :math:`x`-momentum stepped
			disregarding the vertical advection.
		V_now : obj
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum, i.e., the :math:`y`-momentum stepped
			disregarding the vertical advection.
		Qv_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qv_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor, 
			i.e., the isentropic density of water vapor stepped disregarding the vertical advection.
		Qc_now : obj			
			:class:`gridtools.Equation` representing the current isentropic density of cloud water.
		Qc_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud water, 
			i.e., the isentropic density of cloud water stepped disregarding the vertical advection.
		Qr_now : obj
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qr_prv : obj
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water, 
			i.e., the isentropic density of precipitation water stepped disregarding the vertical advection.
		"""
		# Compute predicted values for the isentropic density and the momentums
		s_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, w, s_now, s_prv)
		U_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, w, U_now, U_prv)
		V_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, w, V_now, V_prv)

		if self._moist_on:
			# Compute predicted values for the water constituents
			Qv_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, w, Qv_now, Qv_prv)
			Qc_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, w, Qc_now, Qc_prv)
			Qr_prd = self._get_maccormack_vertical_predicted_value(i, j, k, dt, w, Qr_now, Qr_prv)

		# Compute the flux for the isentropic density and the momentums
		self._flux_s_z = self._get_maccormack_flux_z(i, j, k, w, s_now, s_prv, s_prd)
		self._flux_U_z = self._get_maccormack_flux_z(i, j, k, w, U_now, U_prv, U_prd)
		self._flux_V_z = self._get_maccormack_flux_z(i, j, k, w, V_now, V_prv, V_prd)

		if self._moist_on:
			# Compute the flux for the water constituents
			self._flux_Qv_z = self._get_maccormack_flux_z(i, j, k, w, Qv_now, Qv_prv, Qv_prd)
			self._flux_Qc_z = self._get_maccormack_flux_z(i, j, k, w, Qc_now, Qc_prv, Qc_prd)
			self._flux_Qr_z = self._get_maccormack_flux_z(i, j, k, w, Qr_now, Qr_prv, Qr_prd)

	def _get_maccormack_horizontal_predicted_value_density(self, i, j, k, dt, s_now, U_now, V_now):
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
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the isentropic density.
		"""
		s_prd = gt.Equation()
		s_prd[i, j, k] = s_now[i, j, k] - dt * ((U_now[i+1, j, k] - U_now[i, j, k]) / self._grid.dx + 
											 	(V_now[i, j+1, k] - V_now[i, j, k]) / self._grid.dy)
		return s_prd	

	def _get_maccormack_horizontal_predicted_value_momentum_x(self, i, j, k, dt, s_now, u_now_unstg, v_now_unstg, mtg_now, U_now):
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
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_now_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		v_now_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		mtg_now : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the :math:`x`-momentum.
		"""
		U_prd = gt.Equation()
		U_prd[i, j, k] = U_now[i, j, k] - dt * ((u_now_unstg[i+1,   j, k] * U_now[i+1,   j, k] - 
											  	 u_now_unstg[  i,   j, k] * U_now[  i,   j, k]) / self._grid.dx + 
										 	 	(v_now_unstg[  i, j+1, k] * U_now[  i, j+1, k] - 
											  	 v_now_unstg[  i,   j, k] * U_now[  i,   j, k]) / self._grid.dy +
											 	s_now[i, j, k] * (mtg_now[i+1, j, k] - mtg_now[i, j, k]) / self._grid.dx)
		return U_prd	

	def _get_maccormack_horizontal_predicted_value_momentum_y(self, i, j, k, dt, s_now, u_now_unstg, v_now_unstg, mtg_now, V_now):
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
		s_now : obj
			:class:`gridtools.Equation` representing the isentropic density.
		u_now_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		v_now_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		mtg_now : obj
			:class:`gridtools.Equation` representing the Montgomery potential.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for the :math:`y`-momentum.
		"""
		V_prd = gt.Equation()
		V_prd[i, j, k] = V_now[i, j, k] - dt * ((u_now_unstg[i+1,   j, k] * V_now[i+1,   j, k] - 
											  	 u_now_unstg[  i,   j, k] * V_now[  i,   j, k]) / self._grid.dx + 
										 	 	(v_now_unstg[  i, j+1, k] * V_now[  i, j+1, k] - 
											  	 v_now_unstg[  i,   j, k] * V_now[  i,   j, k]) / self._grid.dy +
											 	s_now[i, j, k] * (mtg_now[i, j+1, k] - mtg_now[i, j, k]) / self._grid.dy)
		return V_prd	

	def _get_maccormack_horizontal_predicted_value_constituent(self, i, j, k, dt, u_now_unstg, v_now_unstg, Q_now):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for the isentropic density of a generic 
		water constituent :math:`Q`, computed without taking the vertical advection into account.
		
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
		u_now_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity.
		v_now_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity.
		Q_now : obj
			:class:`gridtools.Equation` representing the isentropic density of a generic water constituent :math:`Q`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for :math:`Q`.
		"""
		Q_name = Q_now.get_name()
		Q_prd_name = Q_name + '_prd'
		Q_prd = gt.Equation(name = Q_prd_name)
		Q_prd[i, j, k] = Q_now[i, j, k] - dt * ((u_now_unstg[i+1,   j, k] * Q_now[i+1,   j, k] -
											  	 u_now_unstg[  i,   j, k] * Q_now[  i,   j, k]) / self._grid.dx + 
										 	 	(v_now_unstg[  i, j+1, k] * Q_now[  i, j+1, k] - 
											  	 v_now_unstg[  i,   j, k] * Q_now[  i,   j, k]) / self._grid.dy)
		return Q_prd	

	def _get_maccormack_vertical_predicted_value(self, i, j, k, dt, w, phi_now, phi_prv):
		"""
		Get the :class:`gridtools.Equation` representing the predicted value for a generic conservative prognostic variable :math:`\phi`,
		computed taking only the vertical advection into account.
		
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
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time in potential temperature.
		phi_now : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at current time.
		phi_prv : obj
			:class:`gridtools.Equation` representing the provisional value for :math:`\phi`, i.e., :math:`\phi` stepped disregarding 
			the vertical advection.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the predicted value for :math:`\phi`.
		"""
		phi_name = phi_now.get_name()
		phi_prd_name = phi_name + '_prd'
		phi_prd = gt.Equation(name = phi_prd_name)
		phi_prd[i, j, k] = phi_prv[i, j, k] - dt * (w[i, j, k+1] * phi_now[i, j, k+1] -
											  	 	w[i, j,   k] * phi_now[i, j,   k]) / self._grid.dz
		return phi_prd	

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

	def _get_maccormack_flux_x(self, i, j, k, u_now_unstg, phi_now, u_prd_unstg, phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for a 
		generic prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		u_now_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`x`-velocity at the current time.
		phi_now : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at the current time.
		u_prd_unstg : obj
			:class:`gridtools.Equation` representing the predicted value for the unstaggered :math:`x`-velocity.
		phi_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for :math:`\phi`.
		"""
		phi_name = phi_now.get_name()
		flux_name = 'flux_' + phi_name + '_x'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (u_now_unstg[i+1, j, k] * phi_now[i+1, j, k] + u_prd_unstg[i, j, k] * phi_prd[i, j, k])
		return flux

	def _get_maccormack_flux_x_density(self, i, j, k, U_now, U_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for the
		isentropic density.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		U_now : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum at the current time.
		U_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the :math:`x`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`x`-direction for the isentropic density.
		"""
		flux_s_x = gt.Equation()
		flux_s_x[i, j, k] = 0.5 * (U_now[i+1, j, k] + U_prd[i, j, k])
		return flux_s_x

	def _get_maccormack_flux_y(self, i, j, k, v_now_unstg, phi_now, v_prd_unstg, phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for a 
		generic prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		v_now_unstg : obj
			:class:`gridtools.Equation` representing the unstaggered :math:`y`-velocity at the current time.
		phi_now : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at the current time.
		v_prd_unstg : obj
			:class:`gridtools.Equation` representing the predicted value for the unstaggered :math:`y`-velocity.
		phi_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for :math:`\phi`.
		"""
		phi_name = phi_now.get_name()
		flux_name = 'flux_' + phi_name + '_y'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (v_now_unstg[i, j+1, k] * phi_now[i, j+1, k] + v_prd_unstg[i, j, k] * phi_prd[i, j, k])
		return flux

	def _get_maccormack_flux_y_density(self, i, j, k, V_now, V_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for the
		isentropic density.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		V_now : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum at the current time.
		V_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the :math:`y`-momentum.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`y`-direction for the isentropic density.
		"""
		flux_s_y = gt.Equation()
		flux_s_y[i, j, k] = 0.5 * (V_now[i, j+1, k] + V_prd[i, j, k])
		return flux_s_y

	def _get_maccormack_flux_z(self, i, j, k, w, phi_now, phi_prv, phi_prd):
		"""
		Get the :class:`gridtools.Equation` representing the MacCormack flux in :math:`z`-direction for a 
		generic prognostic variable :math:`\phi`.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		w : obj
			:class:`gridtools.Equation` representing the vertical velocity, i.e., the change over time in potential temperature.
		phi_now : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at current time.
		phi_prv : obj
			:class:`gridtools.Equation` representing the provisional value for :math:`\phi`, i.e., :math:`\phi` stepped disregarding 
			the vertical advection.
		phi_prd : obj
			:class:`gridtools.Equation` representing the predicted value for the field :math:`\phi`.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the MacCormack flux in :math:`z`-direction for :math:`\phi`.
		"""
		phi_name = phi_now.get_name()
		flux_name = 'flux_' + phi_name + '_z'
		flux = gt.Equation(name = flux_name)
		flux[i, j, k] = 0.5 * (w[i, j, k+1] * phi_now[i, j, k+1] + w[i, j, k] * phi_prd[i, j, k])
		return flux
