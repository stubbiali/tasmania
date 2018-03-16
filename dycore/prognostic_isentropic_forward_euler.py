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
import copy
import numpy as np

import gridtools as gt
from tasmania.dycore.flux_isentropic import FluxIsentropic
from tasmania.dycore.horizontal_boundary import RelaxedSymmetricXZ, RelaxedSymmetricYZ
from tasmania.dycore.prognostic_isentropic import PrognosticIsentropic
from tasmania.namelist import datatype
from tasmania.storages.grid_data import GridData
from tasmania.storages.state_isentropic import StateIsentropic

class PrognosticIsentropicForwardEuler(PrognosticIsentropic):
	"""
	This class inherits :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` to implement
	the forward Euler scheme carrying out the prognostic step of the three-dimensional moist isentropic dynamical core.

	Attributes
	----------
	time_levels : int
		Number of time levels the scheme relies on.
	steps : int
		Number of steps the scheme entails.
	"""
	def __init__(self, flux_scheme, grid, moist_on, backend):
		"""
		Constructor.
		
		Parameters
		----------
		flux_scheme : str 
			String specifying the flux scheme to use. Either:

			* 'upwind', for the upwind flux;
			* 'centered', for a second-order centered flux;
			* 'maccormack', for the MacCormack flux.

		grid : obj
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT$Py's stencils.

		Note
		----
		To instantiate an object of this class, one should prefer the static method
		:meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.factory` of 
		:class:`~dycore.prognostic_isentropic.PrognosticIsentropic`.
		"""
		super().__init__(flux_scheme, grid, moist_on, backend)

		# Number of time levels and steps entailed
		self.time_levels = 1
		self.steps = 1

		# The pointers to the stencils' compute function
		# They will be re-directed when the forward method is invoked for the first time
		self._stencil_stepping_neglecting_vertical_advection_first = None
		self._stencil_stepping_neglecting_vertical_advection_second = None

	def step_neglecting_vertical_advection(self, dt, state, state_old = None, diagnostics = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward via the forward Euler method.
		Only horizontal derivates are considered; possible vertical derivatives are disregarded.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure (:math:`z`-staggered);
			* montgomery_potential (isentropic);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		state_old : `obj`, optional
			:class:`~storages.state_isentropic.StateIsentropic` representing the old state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		diagnostics : `obj`, optional
			:class:`~storages.grid_data.GridData` possibly collecting useful diagnostics.

		Return
		------
		obj :
			:class:`~storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).
		"""
		# Initialize the output state
		state_new = StateIsentropic(state.time + dt, self._grid)

		# Extract the model variables which are needed
		s   = state['air_isentropic_density'].values[:,:,:,0]
		u   = state['x_velocity'].values[:,:,:,0]
		v   = state['y_velocity'].values[:,:,:,0]
		U   = state['x_momentum_isentropic'].values[:,:,:,0]
		V   = state['y_momentum_isentropic'].values[:,:,:,0]
		mtg = state['montgomery_potential'].values[:,:,:,0]
		Qv	= None if not self._moist_on else state['water_vapor_isentropic_density'].values[:,:,:,0]
		Qc	= None if not self._moist_on else state['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
		Qr	= None if not self._moist_on else state['precipitation_water_isentropic_density'].values[:,:,:,0]

		# Extend the arrays to accommodate the horizontal boundary conditions
		s_   = self.boundary.from_physical_to_computational_domain(s)
		u_   = self.boundary.from_physical_to_computational_domain(u)
		v_   = self.boundary.from_physical_to_computational_domain(v)
		mtg_ = self.boundary.from_physical_to_computational_domain(mtg)
		U_   = self.boundary.from_physical_to_computational_domain(U)
		V_   = self.boundary.from_physical_to_computational_domain(V)
		Qv_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qv)
		Qc_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qc)
		Qr_  = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qr)

		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_stepping_neglecting_vertical_advection_first is None:
			self._initialize_stencils_stepping_neglecting_vertical_advection(s_, u_, v_)

		# Update the attributes which serve as inputs to the first GT4Py's stencil
		self._set_inputs_of_stencils_stepping_neglecting_vertical_advection(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_)
		
		# Run the compute function of the stencil stepping the isentropic density and the water constituents,
		# and providing provisional values for the momentums
		self._stencil_stepping_neglecting_vertical_advection_first.compute()

		# Bring the updated density and water constituents back to the original dimensions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		s_new  = self.boundary.from_computational_to_physical_domain(self._out_s_, (nx, ny, nz))
		Qv_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qv_, (nx, ny, nz))
		Qc_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qc_, (nx, ny, nz))
		Qr_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qr_, (nx, ny, nz))

		# Apply the boundary conditions on the updated isentropic density and water constituents
		self.boundary.apply(s_new, s)
		if self._moist_on:
			self.boundary.apply(Qv_new, Qv)
			self.boundary.apply(Qc_new, Qc)
			self.boundary.apply(Qr_new, Qr)

		# Compute the provisional isentropic density; this may be scheme-dependent
		if self._flux_scheme in ['upwind', 'centered']:
			s_prov = s_new
		elif self._flux_scheme in ['maccormack']:
			s_prov = .5 * (s + s_new)

		# Diagnose the Montgomery potential from the provisional isentropic density
		state_prov = StateIsentropic(state.time + .5 * dt, self._grid, air_isentropic_density = s_prov) 
		gd = self.diagnostic.get_diagnostic_variables(state_prov, state['air_pressure'].values[0,0,0,0])

		# Extend the update isentropic density and Montgomery potential to accomodate the horizontal boundary conditions
		self._prv_s[:,:,:]   = self.boundary.from_physical_to_computational_domain(s_prov)
		self._prv_mtg[:,:,:] = self.boundary.from_physical_to_computational_domain(gd['montgomery_potential'].values[:,:,:,0])

		# Run the compute function of the stencil stepping the momentums
		self._stencil_stepping_neglecting_vertical_advection_second.compute()

		# Bring the momentums back to the original dimensions
		if type(self.boundary) == RelaxedSymmetricXZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz), change_sign = False)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz), change_sign = True) 
		elif type(self.boundary) == RelaxedSymmetricYZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz), change_sign = True)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz))
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz)) 

		# Apply the boundary conditions on the momentums
		self.boundary.apply(U_new, U)
		self.boundary.apply(V_new, V)

		# Update the output state
		state_new.add(air_isentropic_density                 = s_new, 
					  x_momentum_isentropic                  = U_new, 
					  y_momentum_isentropic                  = V_new,
					  water_vapor_isentropic_density         = Qv_new, 
					  cloud_liquid_water_isentropic_density  = Qc_new,
					  precipitation_water_isentropic_density = Qr_new)

		return state_new

	def _initialize_stencils_stepping_neglecting_vertical_advection(self, s_, u_, v_):
		"""
		Initialize the GT4Py's stencils implementing the forward Euler scheme.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
			at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
			at current time.
		"""
		# Allocate the Numpy arrays which will serve as inputs to the first stencil
		self._allocate_inputs_of_stencils_stepping_neglecting_vertical_advection(s_, u_, v_)

		# Allocate the Numpy arrays which will store temporary fields
		self._allocate_temporaries_of_stencils_stepping_neglecting_vertical_advection(s_)

		# Allocate the Numpy arrays which will store the output fields
		self._allocate_outputs_of_stencils_stepping_neglecting_vertical_advection(s_)

		# Set the computational domain and the backend
		ni, nj, nk = s_.shape[0] - 2 * self.nb, s_.shape[1] - 2 * self.nb, s_.shape[2]
		_domain = gt.domain.Rectangle((self.nb, self.nb, 0), 
									  (self.nb + ni - 1, self.nb + nj - 1, nk - 1))
		_mode = self._backend

		# Instantiate the first stencil
		if not self._moist_on:
			self._stencil_stepping_neglecting_vertical_advection_first = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_neglecting_vertical_advection_first,
				inputs = {'in_s': self._in_s_, 'in_u': self._in_u_, 'in_v': self._in_v_, 
						  'in_mtg': self._in_mtg_, 'in_U': self._in_U_, 'in_V': self._in_V_},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s_, 'out_U': self._prv_U, 'out_V': self._prv_V},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil_stepping_neglecting_vertical_advection_first = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_neglecting_vertical_advection_first,
				inputs = {'in_s': self._in_s_, 'in_u': self._in_u_, 'in_v': self._in_v_, 
						  'in_mtg': self._in_mtg_, 'in_U': self._in_U_, 'in_V': self._in_V_,  
						  'in_Qv': self._in_Qv_, 'in_Qc': self._in_Qc_, 'in_Qr': self._in_Qr_},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s_, 'out_U': self._prv_U, 'out_V': self._prv_V,
						   'out_Qv': self._out_Qv_, 'out_Qc': self._out_Qc_, 'out_Qr': self._out_Qr_},
				domain = _domain, 
				mode = _mode)

		# Instantiate the second stencil
		self._stencil_stepping_neglecting_vertical_advection_second = gt.NGStencil( 
			definitions_func = self._defs_stencil_stepping_neglecting_vertical_advection_second,
			inputs = {'in_s': self._prv_s, 'in_mtg': self._prv_mtg, 'in_U': self._prv_U, 'in_V': self._prv_V},
			global_inputs = {'dt': self._dt},
			outputs = {'out_U': self._out_U_, 'out_V': self._out_V_},
			domain = _domain, 
			mode = _mode)

	def _allocate_temporaries_of_stencils_stepping_neglecting_vertical_advection(self, s_):
		"""
		Allocate the Numpy arrays which will store temporary fields.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			at current time.
		"""
		self._prv_U   = np.zeros_like(s_)
		self._prv_V   = np.zeros_like(s_)
		self._prv_s   = np.zeros_like(s_)
		self._prv_mtg = np.zeros_like(s_)

	def _defs_stencil_stepping_neglecting_vertical_advection_first(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
					  											   in_Qv = None, in_Qc = None, in_Qr = None):
		"""
		GT4Py's stencil stepping the isentropic density and the water constituents via the forward Euler scheme.
		Further, it computes provisional values for the momentums, i.e., it updates the momentums disregarding
		the forcing terms involving the Montgomery potential.
		
		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the isentropic density at the current time.
		in_u : obj
			:class:`gridtools.Equation` representing the :math:`x`-velocity at the current time.
		in_v : obj
			:class:`gridtools.Equation` representing the :math:`y`-velocity at the current time.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential at the current time.
		in_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum at the current time.
		in_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum at the current time.
		in_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass of water vapour at the current time.
		in_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass of cloud water at the current time.
		in_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass of precipitation water at the current time.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		out_U : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		out_V : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
		out_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of water vapour.
		out_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of cloud water.
		out_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the stepped mass of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_s = gt.Equation()
		out_U = gt.Equation()
		out_V = gt.Equation()
		if self._moist_on:
			out_Qv = gt.Equation()
			out_Qc = gt.Equation()
			out_Qr = gt.Equation()

		# Computations
		if not self._moist_on:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V)
		else:
			flux_s_x, flux_s_y, flux_U_x, flux_U_y, flux_V_x, flux_V_y, \
			flux_Qv_x, flux_Qv_y, flux_Qc_x, flux_Qc_y, flux_Qr_x, flux_Qr_y = \
				self._flux.get_horizontal_fluxes(i, j, k, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, in_Qv, in_Qc, in_Qr)

		out_s[i, j, k] = in_s[i, j, k] - dt * ((flux_s_x[i, j, k] - flux_s_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_s_y[i, j, k] - flux_s_y[i, j-1, k]) / self._grid.dy)
		out_U[i, j, k] = in_U[i, j, k] - dt * ((flux_U_x[i, j, k] - flux_U_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_U_y[i, j, k] - flux_U_y[i, j-1, k]) / self._grid.dy)
		out_V[i, j, k] = in_V[i, j, k] - dt * ((flux_V_x[i, j, k] - flux_V_x[i-1, j, k]) / self._grid.dx +
						 					   (flux_V_y[i, j, k] - flux_V_y[i, j-1, k]) / self._grid.dy)
		if self._moist_on:
			out_Qv[i, j, k] = in_Qv[i, j, k] - dt * ((flux_Qv_x[i, j, k] - flux_Qv_x[i-1, j, k]) / self._grid.dx +
						 						  	 (flux_Qv_y[i, j, k] - flux_Qv_y[i, j-1, k]) / self._grid.dy)
			out_Qc[i, j, k] = in_Qc[i, j, k] - dt * ((flux_Qc_x[i, j, k] - flux_Qc_x[i-1, j, k]) / self._grid.dx +
						 						  	 (flux_Qc_y[i, j, k] - flux_Qc_y[i, j-1, k]) / self._grid.dy)
			out_Qr[i, j, k] = in_Qr[i, j, k] - dt * ((flux_Qr_x[i, j, k] - flux_Qr_x[i-1, j, k]) / self._grid.dx +
						 						  	 (flux_Qr_y[i, j, k] - flux_Qr_y[i, j-1, k]) / self._grid.dy)

			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_s, out_U, out_V

	def _defs_stencil_stepping_neglecting_vertical_advection_second(self, dt, in_s, in_mtg, in_U, in_V):
		"""
		GT4Py's stencil stepping the momentums via a one-time-level scheme.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		in_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		in_mtg : obj
			:class:`gridtools.Equation` representing the Montgomery potential diagnosed from the stepped isentropic density.
		in_U : obj
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		in_V : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.

		Returns
		-------
		out_U : obj
			:class:`gridtools.Equation` representing the stepped :math:`x`-momentum.
		out_V : obj
			:class:`gridtools.Equation` representing the stepped :math:`y`-momentum.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		out_U = gt.Equation()
		out_V = gt.Equation()

		# Computations
		out_U[i, j, k] = in_U[i, j, k] - dt * 0.5 * in_s[i, j, k] * (in_mtg[i+1, j, k] - in_mtg[i-1, j, k]) / self._grid.dx
		out_V[i, j, k] = in_V[i, j, k] - dt * 0.5 * in_s[i, j, k] * (in_mtg[i, j+1, k] - in_mtg[i, j-1, k]) / self._grid.dy

		return out_U, out_V

	def _defs_stencil_stepping_coupling_physics_with_dynamics(dt, s_now, U_now, V_now, s_prv, U_prv, V_prv,
															  Qv_now = None, Qc_now = None, Qr_now = None,
															  Qv_prv = None, Qc_prv = None, Qr_prv = None):
		"""
		GT4Py's stencil stepping the solution by coupling physics with dynamics, i.e., by accounting for the
		change over time in potential temperature.

		Parameters
		----------
		dt : obj 
			:class:`gridtools.Global` representing the time step.
		s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density. 
		U_now : obj 
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		V_now : obj 
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		s_prv : obj 
			:class:`gridtools.Equation` representing the provisional isentropic density. 
		U_prv : obj 
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
		Qv_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		Qc_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of cloud liquid water.
		Qr_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor.
		Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud liquid water.
		Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water.

		Returns
		-------
		s_new : obj
			:class:`gridtools.Equation` representing the updated isentropic density. 
		U_new : obj 
			:class:`gridtools.Equation` representing the updated :math:`x`-momentum.
		V_new : obj 
			:class:`gridtools.Equation` representing the updated :math:`y`-momentum.
		Qv_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of water vapor.
		Qc_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of cloud liquid water.
		Qr_new : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of precipitation water.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output fields
		new_s = gt.Equation()
		new_U = gt.Equation()
		new_V = gt.Equation()
		if self._moist_on:
			new_Qv = gt.Equation()
			new_Qc = gt.Equation()
			new_Qr = gt.Equation()

		# Computations
		if not self._moist_on:
			flux_s_z, flux_U_z, flux_V_z = self._flux.get_vertical_fluxes(i, j, k, dt, w, now_s, prv_s, 
																		  now_U, prv_U, now_V, prv_V)
		else:	
			flux_s_z, flux_U_z, flux_V_z, flux_Qv_z, flux_Qc_z, flux_Qr_z = \
				self._flux.get_vertical_fluxes(i, j, k, dt, w, now_s, prv_s, now_U, prv_U, now_V, prv_V,
											   now_Qv, prv_Qv, now_Qc, prv_Qc, now_Qr, prv_Qr)

		new_s[i, j, k] = prv_s[i, j, k] - dt * (flux_s_z[i, j, k] - flux_s_z[i, j, k+1]) / self._grid.dz
		new_U[i, j, k] = prv_U[i, j, k] - dt * (flux_U_z[i, j, k] - flux_U_z[i, j, k+1]) / self._grid.dz
		new_V[i, j, k] = prv_V[i, j, k] - dt * (flux_V_z[i, j, k] - flux_V_z[i, j, k+1]) / self._grid.dz
		if self._moist_on:
			new_Qv[i, j, k] = prv_Qv[i, j, k] - dt * (flux_Qv_z[i, j, k] - flux_Qv_z[i, j, k+1]) / self._grid.dz
			new_Qc[i, j, k] = prv_Qc[i, j, k] - dt * (flux_Qc_z[i, j, k] - flux_Qc_z[i, j, k+1]) / self._grid.dz
			new_Qr[i, j, k] = prv_Qr[i, j, k] - dt * (flux_Qr_z[i, j, k] - flux_Qr_z[i, j, k+1]) / self._grid.dz

		if not self._moist_on:
			return new_s, new_U, new_V
		else:
			return new_s, new_U, new_V, new_Qv, new_Qc, new_Qr
