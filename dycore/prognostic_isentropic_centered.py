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

class PrognosticIsentropicCentered(PrognosticIsentropic):
	"""
	This class inherits :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` to implement 
	a centered time-integration scheme to carry out the prognostic step of the three-dimensional 
	moist isentropic dynamical core.

	Attributes
	----------
	ni : int
		Extent of the computational domain in the :math:`x`-direction.
	nj : int
		Extent of the computational domain in the :math:`y`-direction.
	nk : int
		Extent of the computational domain in the :math:`z`-direction.
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
			:class:`gridtools.mode` specifying the backend for the GT4Py's stencils.

		Note
		----
		To instantiate an object of this class, one should prefer the static method
		:meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.factory` of
		:class:`~dycore.prognostic_isentropic.PrognosticIsentropic`.
		"""
		super().__init__(flux_scheme, grid, moist_on, backend)

		# Number of time levels and steps entailed
		self.time_levels = 2
		self.steps = 1

		# The pointers to the stencil's compute function
		# This will be re-directed when the forward method is invoked for the first time
		self._stencil_stepping_by_neglecting_vertical_advection = None

	def step_neglecting_vertical_advection(self, dt, state, state_old = None, diagnostics = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward via a centered time-integration scheme.
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
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
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

		# Extract the needed model variables at the current time level
		s   = state['air_isentropic_density'].values[:,:,:,0]
		u   = state['x_velocity'].values[:,:,:,0]
		v   = state['y_velocity'].values[:,:,:,0]
		U   = state['x_momentum_isentropic'].values[:,:,:,0]
		V   = state['y_momentum_isentropic'].values[:,:,:,0]
		p   = state['air_pressure'].values[:,:,:,0]
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

		if state_old is not None:
			# Extract the needed model variables at the previous time level
			s_old  = state_old['air_isentropic_density'].values[:,:,:,0]
			U_old  = state_old['x_momentum_isentropic'].values[:,:,:,0]
			V_old  = state_old['y_momentum_isentropic'].values[:,:,:,0]
			Qv_old = None if not self._moist_on else state_old['water_vapor_isentropic_density'].values[:,:,:,0]
			Qc_old = None if not self._moist_on else state_old['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			Qr_old = None if not self._moist_on else state_old['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Extend the arrays to accommodate the horizontal boundary conditions
			self._s_old_  = self.boundary.from_physical_to_computational_domain(s_old)
			self._U_old_  = self.boundary.from_physical_to_computational_domain(U_old)
			self._V_old_  = self.boundary.from_physical_to_computational_domain(V_old)
			self._Qv_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qv_old)
			self._Qc_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qc_old)
			self._Qr_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qr_old)
		elif not hasattr(self, '_s_old_'):
			# Extract the needed model variables at the previous time level
			s_old  = state['air_isentropic_density'].values[:,:,:,0]
			U_old  = state['x_momentum_isentropic'].values[:,:,:,0]
			V_old  = state['y_momentum_isentropic'].values[:,:,:,0]
			Qv_old = None if not self._moist_on else state['water_vapor_isentropic_density'].values[:,:,:,0]
			Qc_old = None if not self._moist_on else state['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			Qr_old = None if not self._moist_on else state['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Extend the arrays to accommodate the horizontal boundary conditions
			self._s_old_  = self.boundary.from_physical_to_computational_domain(s_old)
			self._U_old_  = self.boundary.from_physical_to_computational_domain(U_old)
			self._V_old_  = self.boundary.from_physical_to_computational_domain(V_old)
			self._Qv_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qv_old)
			self._Qc_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qc_old)
			self._Qr_old_ = None if not self._moist_on else self.boundary.from_physical_to_computational_domain(Qr_old)

		# The first time this method is invoked, initialize the GT4Py's stencils
		if self._stencil_stepping_by_neglecting_vertical_advection is None:
			self._stencil_stepping_by_neglecting_vertical_advection_initialize(s_, u_, v_)

		# Update the attributes which serve as inputs to the first GT4Py's stencil
		self._set_stencils_stepping_by_neglecting_vertical_advection_inputs(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_, 
			self._s_old_, self._U_old_, self._V_old_, self._Qv_old_, self._Qc_old_, self._Qr_old_)
		
		# Run the stencil's compute function
		self._stencil_stepping_by_neglecting_vertical_advection.compute()
		
		# Bring the updated prognostic variables back to the original dimensions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		s_new = self.boundary.from_computational_to_physical_domain(self._out_s_, (nx, ny, nz))

		if type(self.boundary) == RelaxedSymmetricXZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz), change_sign = False)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz), change_sign = True) 
		elif type(self.boundary) == RelaxedSymmetricYZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz), change_sign = True)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U_, (nx, ny, nz))
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V_, (nx, ny, nz)) 

		Qv_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qv_, (nx, ny, nz))
		Qc_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qc_, (nx, ny, nz))
		Qr_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qr_, (nx, ny, nz))

		# Apply the boundary conditions
		self.boundary.apply(s_new, s)
		self.boundary.apply(U_new, U)
		self.boundary.apply(V_new, V)
		if self._moist_on:
			self.boundary.apply(Qv_new, Qv)
			self.boundary.apply(Qc_new, Qc)
			self.boundary.apply(Qr_new, Qr)

		# Update the output state
		state_new.add(air_isentropic_density                 = s_new, 
					  x_momentum_isentropic                  = U_new, 
					  y_momentum_isentropic                  = V_new, 
					  water_vapor_isentropic_density         = Qv_new, 
					  cloud_liquid_water_isentropic_density  = Qc_new,
					  precipitation_water_isentropic_density = Qr_new)

		# Keep track of the current state for the next timestep
		self._s_old_[:,:,:] = s_[:,:,:]
		self._U_old_[:,:,:] = U_[:,:,:]
		self._V_old_[:,:,:] = V_[:,:,:]
		if self._moist_on:
			self._Qv_old_[:,:,:] = Qv_[:,:,:]
			self._Qc_old_[:,:,:] = Qc_[:,:,:]
			self._Qr_old_[:,:,:] = Qr_[:,:,:]

		return state_new

	def _stencil_stepping_by_neglecting_vertical_advection_initialize(self, s_, u_, v_):
		"""
		Initialize the GT4Py's stencil implementing the centered time-integration scheme.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Allocate the attributes which will serve as inputs to the stencil
		self._stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(s_, u_, v_)

		# Allocate the Numpy arrays which will store the output fields
		self._stencils_stepping_by_neglecting_vertical_advection_allocate_outputs(s_)

		# Set the computational domain and the backend
		ni, nj, nk = s_.shape[0] - 2 * self.nb, s_.shape[1] - 2 * self.nb, s_.shape[2]
		_domain = gt.domain.Rectangle((self.nb, self.nb, 0), 
									  (self.nb + ni - 1, self.nb + nj - 1, nk - 1))
		_mode = self._backend

		# Instantiate the stencil
		if not self._moist_on:
			self._stencil_stepping_by_neglecting_vertical_advection = gt.NGStencil( 
				definitions_func = self._stencil_stepping_by_neglecting_vertical_advection_defs,
				inputs = {'in_s': self._in_s_, 'in_u': self._in_u_, 'in_v': self._in_v_, 
						  'in_mtg': self._in_mtg_, 'in_U': self._in_U_, 'in_V': self._in_V_,
						  'old_s': self._old_s_, 'old_U': self._old_U_, 'old_V': self._old_V_},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s_, 'out_U': self._out_U_, 'out_V': self._out_V_},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil_stepping_by_neglecting_vertical_advection = gt.NGStencil( 
				definitions_func = self._stencil_stepping_by_neglecting_vertical_advection_defs,
				inputs = {'in_s': self._in_s_, 'in_u': self._in_u_, 'in_v': self._in_v_, 
						  'in_mtg': self._in_mtg_, 'in_U': self._in_U_, 'in_V': self._in_V_,
						  'in_Qv': self._in_Qv_, 'in_Qc': self._in_Qc_, 'in_Qr': self._in_Qr_,
						  'old_s': self._old_s_, 'old_U': self._old_U_, 'old_V': self._old_V_,
						  'old_Qv': self._old_Qv_, 'old_Qc': self._old_Qc_, 'old_Qr': self._old_Qr_},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s_, 'out_U': self._out_U_, 'out_V': self._out_V_,
						   'out_Qv': self._out_Qv_, 'out_Qc': self._out_Qc_, 'out_Qr': self._out_Qr_},
				domain = _domain, 
				mode = _mode)

	def _stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(self, s_, u_, v_):
		"""
		Allocate the attributes which will serve as inputs to the GT4Py's stencil.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Instantiate a GT4Py's Global representing the timestep and the Numpy arrays
		# which will carry the solution at the current time step
		super()._stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(s_, u_, v_)

		# Allocate the Numpy arrays which will carry the solution at the previous time step
		self._old_s_ = np.zeros_like(s_)
		self._old_U_ = np.zeros_like(s_)
		self._old_V_ = np.zeros_like(s_)
		if self._moist_on:
			self._old_Qv_ = np.zeros_like(s_)
			self._old_Qc_ = np.zeros_like(s_)
			self._old_Qr_ = np.zeros_like(s_)

	def _set_stencils_stepping_by_neglecting_vertical_advection_inputs(self, dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_, 
																	   s_old_, U_old_, V_old_, Qv_old_, Qc_old_, Qr_old_):
		"""
		Update the attributes which serve as inputs to the GT4Py's stencil.

		Parameters
		----------
		dt : obj 
			A :class:`datetime.timedelta` representing the time step.
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity 
			at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity 
			at current time.
		mtg_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the Montgomery potential 
			at current time.
		U_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum 
			at current time.
		V_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
			at current time.
		Qv_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
			at current time.
		Qc_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
			at current time.
		Qr_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water
			at current time.
		s_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density
			at the previous time level.
		U_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-momentum
			at the previous time level.
		V_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-momentum 
			at the previous time level.
		Qv_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of water vapour 
			at the previous time level.
		Qc_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of cloud water 
			at the previous time level.
		Qr_old_ : `array_like`, optional 
			:class:`numpy.ndarray` representing the stencils' computational domain for the mass of precipitation water 
			at the previous time level.
		"""
		# Update the time step and the Numpy arrays carrying the current solution
		super()._set_stencils_stepping_by_neglecting_vertical_advection_inputs(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_)
		
		# Update the Numpy arrays carrying the solution at the previous time step
		self._old_s_[:,:,:] = s_old_[:,:,:]
		self._old_U_[:,:,:] = U_old_[:,:,:]
		self._old_V_[:,:,:] = V_old_[:,:,:]
		if self._moist_on:
			self._old_Qv_[:,:,:] = Qv_old_[:,:,:]
			self._old_Qc_[:,:,:] = Qc_old_[:,:,:]
			self._old_Qr_[:,:,:] = Qr_old_[:,:,:]

	def _stencil_stepping_by_neglecting_vertical_advection_defs(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
															 	old_s, old_U, old_V,
					  										 	in_Qv = None, in_Qc = None, in_Qr = None, 
															 	old_Qv = None, old_Qc = None, old_Qr = None):
		"""
		GT4Py's stencil implementing the centered time-integration scheme.

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
		old_s : obj
			:class:`gridtools.Equation` representing the isentropic density at the previous time level.
		old_U : obj
			:class:`gridtools.Equation` representing the :math:`x`-momentum at the previous time level.
		old_V : obj
			:class:`gridtools.Equation` representing the :math:`y`-momentum at the previous time level.
		in_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass of water vapour at the current time.
		in_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass of cloud water at the current time.
		in_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass of precipitation water at the current time.
		old_Qv : `obj`, optional
			:class:`gridtools.Equation` representing the mass of water vapour at the previous time level.
		old_Qc : `obj`, optional
			:class:`gridtools.Equation` representing the mass of cloud water at the previous time level.
		old_Qr : `obj`, optional
			:class:`gridtools.Equation` representing the mass of precipitation water at the previous time level.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the stepped isentropic density.
		out_U : obj
			:class:`gridtools.Equation` representing the stepped :math:`x`-momentum.
		out_V : obj
			:class:`gridtools.Equation` representing the stepped :math:`y`-momentum.
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

		out_s[i, j, k] = old_s[i, j, k] - 2. * dt * ((flux_s_x[i, j, k] - flux_s_x[i-1, j, k]) / self._grid.dx +
						 					         (flux_s_y[i, j, k] - flux_s_y[i, j-1, k]) / self._grid.dy)
		out_U[i, j, k] = old_U[i, j, k] - 2. * dt * ((flux_U_x[i, j, k] - flux_U_x[i-1, j, k]) / self._grid.dx +
						 					         (flux_U_y[i, j, k] - flux_U_y[i, j-1, k]) / self._grid.dy) \
										- dt * in_s[i, j, k] * (in_mtg[i+1, j, k] - in_mtg[i-1, j, k]) / self._grid.dx
		out_V[i, j, k] = old_V[i, j, k] - 2. * dt * ((flux_V_x[i, j, k] - flux_V_x[i-1, j, k]) / self._grid.dx +
						 					         (flux_V_y[i, j, k] - flux_V_y[i, j-1, k]) / self._grid.dy) \
										- dt * in_s[i, j, k] * (in_mtg[i, j+1, k] - in_mtg[i, j-1, k]) / self._grid.dy
		if self._moist_on:
			out_Qv[i, j, k] = old_Qv[i, j, k] - 2. * dt * ((flux_Qv_x[i, j, k] - flux_Qv_x[i-1, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qv_y[i, j, k] - flux_Qv_y[i, j-1, k]) / self._grid.dy)
			out_Qc[i, j, k] = old_Qc[i, j, k] - 2. * dt * ((flux_Qc_x[i, j, k] - flux_Qc_x[i-1, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qc_y[i, j, k] - flux_Qc_y[i, j-1, k]) / self._grid.dy)
			out_Qr[i, j, k] = old_Qr[i, j, k] - 2. * dt * ((flux_Qr_x[i, j, k] - flux_Qr_x[i-1, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qr_y[i, j, k] - flux_Qr_y[i, j-1, k]) / self._grid.dy)

			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_s, out_U, out_V

	def _stencil_stepping_by_coupling_physics_with_dynamics_defs(dt, s_now, U_now, V_now, s_prv, U_prv, V_prv,
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

		new_s[i, j, k] = prv_s[i, j, k] - 2. * dt * (flux_s_z[i, j, k] - flux_s_z[i, j, k+1]) / self._grid.dz
		new_U[i, j, k] = prv_U[i, j, k] - 2. * dt * (flux_U_z[i, j, k] - flux_U_z[i, j, k+1]) / self._grid.dz
		new_V[i, j, k] = prv_V[i, j, k] - 2. * dt * (flux_V_z[i, j, k] - flux_V_z[i, j, k+1]) / self._grid.dz
		if self._moist_on:
			new_Qv[i, j, k] = prv_Qv[i, j, k] - 2. * dt * (flux_Qv_z[i, j, k] - flux_Qv_z[i, j, k+1]) / self._grid.dz
			new_Qc[i, j, k] = prv_Qc[i, j, k] - 2. * dt * (flux_Qc_z[i, j, k] - flux_Qc_z[i, j, k+1]) / self._grid.dz
			new_Qr[i, j, k] = prv_Qr[i, j, k] - 2. * dt * (flux_Qr_z[i, j, k] - flux_Qr_z[i, j, k+1]) / self._grid.dz

		if not self._moist_on:
			return new_s, new_U, new_V
		else:
			return new_s, new_U, new_V, new_Qv, new_Qc, new_Qr
