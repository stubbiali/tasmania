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
Classes implementing different schemes to carry out the prognostic steps of the three-dimensional 
moist isentropic dynamical core.
"""
import abc
import copy
import numpy as np

from dycore.flux_isentropic import FluxIsentropic
from dycore.horizontal_boundary import RelaxedSymmetricXZ, RelaxedSymmetricYZ
import gridtools as gt
from namelist import datatype
from storages.grid_data import GridData
from storages.state_isentropic import StateIsentropic

class PrognosticIsentropic:
	"""
	Abstract base class whose derived classes implement different schemes to carry out the prognostic steps of 
	the three-dimensional moist isentropic dynamical core. The conservative form of the governing equations is used.

	Attributes
	----------
	ni : int
		Extent of the computational domain in the :math:`x`-direction.
	nj : int
		Extent of the computational domain in the :math:`y`-direction.
	nk : int
		Extent of the computational domain in the :math:`z`-direction.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

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
		"""
		# Keep track of the input parameters
		self._flux_scheme, self._grid, self._moist_on, self._backend = flux_scheme, grid, moist_on, backend

		# Instantiate the class computing the numerical fluxes
		self._flux = FluxIsentropic.factory(flux_scheme, grid, moist_on)

		# Initialize the attributes representing the diagnostic step and the lateral boundary conditions
		# Remark: these should be suitably set before calling the stepping method for the first time
		self._diagnostic, self._boundary = None, None

	@property
	def diagnostic(self):
		"""
		Get the attribute implementing the diagnostic step of the three-dimensional moist isentropic dynamical core.
		If this is set to :obj:`None`, a :class:`ValueError` is thrown.
		
		Return
		------
		obj :
			:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the 
			three-dimensional moist isentropic dynamical core.
		"""
		if self._diagnostic is None:
			raise ValueError('''The attribute which is supposed to implement the diagnostic step of the moist isentroic ''' \
							 '''dynamical core is actually :obj:`None`. Please set it correctly.''')
		return self._diagnostic

	@diagnostic.setter
	def diagnostic(self, value):
		"""
		Set the attribute implementing the diagnostic step of the three-dimensional moist isentropic dynamical core.

		Parameter
		---------
		value : obj
			:class:`~dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the 
			three-dimensional moist isentropic dynamical core.
		"""
		self._diagnostic = value

	@property
	def boundary(self):
		"""
		Get the attribute implementing the horizontal boundary conditions.
		If this is set to :obj:`None`, a :class:`ValueError` is thrown.
		
		Return
		------
		obj :
			Instance of the derived class of :class:`~dycore.horizontal_boundary.HorizontalBoundary` implementing
			the horizontal boundary conditions.
		"""
		if self._boundary is None:
			raise ValueError('''The attribute which is supposed to implement the horizontal boundary conditions ''' \
							 '''is actually :obj:`None`. Please set it correctly.''')
		return self._boundary

	@boundary.setter
	def boundary(self, value):
		"""
		Set the attribute implementing the horizontal boundary conditions.

		Parameter
		---------
		value : obj
			Instance of the derived class of :class:`~dycore.horizontal_boundary.HorizontalBoundary` implementing the 
			horizontal boundary conditions.
		"""
		self._boundary = value

	@property
	def nb(self):
		"""
		Get the number of boundary layers.

		Return
		------
		int :
			The number of boundary layers.
		"""
		return self._flux.nb

	@abc.abstractmethod
	def step_without_vertical_advection(self, dt, state, state_old = None, diagnostics = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward.
		Only horizontal derivates are considered; possible vertical derivatives are disregarded.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

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

	@abc.abstractmethod
	def step_with_vertical_advection(self, dt, state_now, state_prv, state_old = None, diagnostics = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward by resolving the vertical advection.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state_now : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

		state_prv : obj
			:class:`~storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
			the state stepped taking only the horizontal derivatives into account. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

			This may be the output of :meth:`~dycore.prognostic_isentropic.PrognosticIsentropic.step_without_vertical_advection`.
		diagnostics : `obj`, optional
			:class:`~storages.grid_data.GridData` collecting the following variables:
			
			* change_over_time_in_air_potential_temperature (unstaggered).

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

	@staticmethod
	def factory(time_scheme, flux_scheme, grid, moist_on, backend):
		"""
		Static method returning an instace of the derived class implementing the time stepping scheme specified 
		by :data:`time_scheme`, using the flux scheme specified by :data:`flux_scheme`.

		Parameters
		----------
		time_scheme : str
			String specifying the time stepping method to implement. Either:

			* 'forward_euler', for the forward Euler scheme;
			* 'centered', for a centered scheme.

		flux_scheme : str 
			String specifying the scheme to use. Either:

			* 'upwind', for the upwind flux;
			* 'centered', for a second-order centered flux;
			* 'maccormack', for the MacCormack flux.

		grid : obj 
			:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.Mode` specifying the backend for the GT4Py's stencils.

		Return
		------
		obj :
			An instace of the derived class implementing the scheme specified by :data:`scheme`.
		"""
		if time_scheme == 'forward_euler':
			return PrognosticIsentropicForwardEuler(flux_scheme, grid, moist_on, backend)
		elif time_scheme == 'centered':
			return PrognosticIsentropicCentered(flux_scheme, grid, moist_on, backend)
		else:
			raise ValueError('Unknown time integration scheme.')

	def _allocate_inputs(self, s_, u_, v_):
		"""
		Allocate the attributes which will serve as inputs to the GT4Py's stencils.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density at current time.
		u_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`x`-velocity at current time.
		v_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the :math:`y`-velocity at current time.
		"""
		# Determine the extent of the computational domain based on the input Numpy arrays 
		# Note that, compared to the DataArrays carrying the state fields, these arrays may be larger, 
		# as they might have been decorated with some extra layers to accommodate the horizontal boundary conditions
		self.ni = s_.shape[0] - 2 * self.nb
		self.nj = s_.shape[1] - 2 * self.nb
		self.nk = s_.shape[2]

		# Instantiate a GT4Py's Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy arrays which will carry the input fields
		self._in_s   = np.zeros_like(s_)
		self._in_u   = np.zeros_like(u_)
		self._in_v   = np.zeros_like(v_)
		self._in_mtg = np.zeros_like(s_)
		self._in_U   = np.zeros_like(s_)
		self._in_V   = np.zeros_like(s_)
		if self._moist_on:
			self._in_Qv = np.zeros_like(s_)
			self._in_Qc = np.zeros_like(s_)
			self._in_Qr = np.zeros_like(s_)

	def _allocate_outputs(self, s_):
		"""
		Allocate the Numpy arrays which will store the updated solution.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density.
		"""
		# Allocate the Numpy arrays which will store the output fields
		# Note: allocation is performed here, i.e., the first time the entry-point method is invoked,
		# so to make this step independent of the boundary conditions type
		self._out_s = np.zeros_like(s_)
		self._out_U = np.zeros_like(s_)
		self._out_V = np.zeros_like(s_)
		if self._moist_on:
			self._out_Qv = np.zeros_like(s_)
			self._out_Qc = np.zeros_like(s_)
			self._out_Qr = np.zeros_like(s_)

	def _set_inputs(self, dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_):
		"""
		Update the attributes which serve as inputs to the GT4Py's stencils.

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
		p_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the pressure	at current time.
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
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			of water vapour at current time.
		Qc_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			of cloud water at current time.
		Qr_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			of precipitation water at current time.
		"""
		# Time step
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds
		
		# Current state
		self._in_s[:,:,:]   = s_[:,:,:]
		self._in_u[:,:,:]   = u_[:,:,:]
		self._in_v[:,:,:]   = v_[:,:,:]
		self._in_mtg[:,:,:] = mtg_[:,:,:]
		self._in_U[:,:,:]   = U_[:,:,:]
		self._in_V[:,:,:]   = V_[:,:,:]
		if self._moist_on:
			self._in_Qv[:,:,:] = Qv_[:,:,:]
			self._in_Qc[:,:,:] = Qc_[:,:,:]
			self._in_Qr[:,:,:] = Qr_[:,:,:]
		

class PrognosticIsentropicForwardEuler(PrognosticIsentropic):
	"""
	This class inherits :class:`~dycore.prognostic_isentropic.PrognosticIsentropic` to implement
	the forward Euler scheme carrying out the prognostic step of the three-dimensional moist isentropic dynamical core.

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
		self._stencil_stepping_isentropic_density_and_water_constituents = None
		self._stencil_stepping_momentums = None

	def step_without_vertical_advection(self, dt, state, state_old = None, diagnostics = None):
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
		if self._stencil_stepping_isentropic_density_and_water_constituents is None:
			self._initialize_stencils(s_, u_, v_)

		# Update the attributes which serve as inputs to the first GT4Py's stencil
		self._set_inputs(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_)
		
		# Run the compute function of the stencil stepping the isentropic density and the water constituents,
		# and providing provisional values for the momentums
		self._stencil_stepping_isentropic_density_and_water_constituents.compute()

		# Bring the updated density and water constituents back to the original dimensions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		s_new  = self.boundary.from_computational_to_physical_domain(self._out_s, (nx, ny, nz))
		Qv_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qv, (nx, ny, nz))
		Qc_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qc, (nx, ny, nz))
		Qr_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qr, (nx, ny, nz))

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
		self._prov_s[:,:,:]   = self.boundary.from_physical_to_computational_domain(s_prov)
		self._prov_mtg[:,:,:] = self.boundary.from_physical_to_computational_domain(gd['montgomery_potential'].values[:,:,:,0])

		# Run the compute function of the stencil stepping the momentums
		self._stencil_stepping_momentums.compute()

		# Bring the momentums back to the original dimensions
		if type(self.boundary) == RelaxedSymmetricXZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U, (nx, ny, nz), change_sign = False)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V, (nx, ny, nz), change_sign = True) 
		elif type(self.boundary) == RelaxedSymmetricYZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U, (nx, ny, nz), change_sign = True)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U, (nx, ny, nz))
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V, (nx, ny, nz)) 

		# Apply the boundary conditions on the momentums
		self.boundary.apply(U_new, U)
		self.boundary.apply(V_new, V)

		# Update the output state
		state_new.add(air_isentropic_density = s_new, 
					  x_momentum_isentropic = U_new, 
					  y_momentum_isentropic = V_new,
					  water_vapor_isentropic_density = Qv_new, 
					  cloud_liquid_water_isentropic_density = Qc_new,
					  precipitation_water_isentropic_density = Qr_new)

		return state_new

	def _initialize_stencils(self, s_, u_, v_):
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
		self._allocate_inputs(s_, u_, v_)

		# Allocate the Numpy arrays which will store temporary fields
		self._allocate_temporaries(s_)

		# Allocate the Numpy arrays which will store the output fields
		self._allocate_outputs(s_)

		# Set the computational domain and the backend
		_domain = gt.domain.Rectangle((self.nb, self.nb, 0), 
									  (self.nb + self.ni - 1, self.nb + self.nj - 1, self.nk - 1))
		_mode = self._backend

		# Instantiate the first stencil
		if not self._moist_on:
			self._stencil_stepping_isentropic_density_and_water_constituents = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_isentropic_density_and_water_constituents,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s, 'out_U': self._tmp_U, 'out_V': self._tmp_V},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil_stepping_isentropic_density_and_water_constituents = gt.NGStencil( 
				definitions_func = self._defs_stencil_stepping_isentropic_density_and_water_constituents,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V,  
						  'in_Qv': self._in_Qv, 'in_Qc': self._in_Qc, 'in_Qr': self._in_Qr},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s, 'out_U': self._tmp_U, 'out_V': self._tmp_V,
						   'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr},
				domain = _domain, 
				mode = _mode)

		# Instantiate the second stencil
		self._stencil_stepping_momentums = gt.NGStencil( 
			definitions_func = self._defs_stencil_stepping_momentums,
			inputs = {'in_s': self._prov_s, 'in_mtg': self._prov_mtg, 'in_U': self._tmp_U, 'in_V': self._tmp_V},
			global_inputs = {'dt': self._dt},
			outputs = {'out_U': self._out_U, 'out_V': self._out_V},
			domain = _domain, 
			mode = _mode)

	def _allocate_temporaries(self, s_):
		"""
		Allocate the Numpy arrays which will store temporary fields.

		Parameters
		----------
		s_ : array_like 
			:class:`numpy.ndarray` representing the stencils' computational domain for the isentropic density 
			at current time.
		"""
		self._tmp_U   = np.zeros_like(s_)
		self._tmp_V   = np.zeros_like(s_)
		self._prov_s   = np.zeros_like(s_)
		self._prov_mtg = np.zeros_like(s_)

	def _defs_stencil_stepping_isentropic_density_and_water_constituents(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, 
					  													 in_Qv = None, in_Qc = None, in_Qr = None):
		"""
		GT4Py's stencil stepping the isentropic density and the water constituents via the forward Euler scheme.
		Further, it computes the provisional values for the momentums.
		
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
			out_Qv[i, j, k] = in_Qv[i, j, k] - dt * ((flux_Qv_x[i+1, j, k] - flux_Qv_x[i, j, k]) / self._grid.dx +
						 						  	 (flux_Qv_y[i, j+1, k] - flux_Qv_y[i, j, k]) / self._grid.dy)
			out_Qc[i, j, k] = in_Qc[i, j, k] - dt * ((flux_Qc_x[i+1, j, k] - flux_Qc_x[i, j, k]) / self._grid.dx +
						 						  	 (flux_Qc_y[i, j+1, k] - flux_Qc_y[i, j, k]) / self._grid.dy)
			out_Qr[i, j, k] = in_Qr[i, j, k] - dt * ((flux_Qr_x[i+1, j, k] - flux_Qr_x[i, j, k]) / self._grid.dx +
						 						  	 (flux_Qr_y[i, j+1, k] - flux_Qr_y[i, j, k]) / self._grid.dy)

			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_s, out_U, out_V

	def _defs_stencil_stepping_momentums(self, dt, in_s, in_mtg, in_U, in_V):
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
		self._stencil = None

	def step_without_vertical_advection(self, dt, state, state_old = None, diagnostics = None):
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
		if self._stencil is None:
			self._initialize_stencil(s_, u_, v_)

		# Update the attributes which serve as inputs to the first GT4Py's stencil
		self._set_inputs(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_, 
						 self._s_old_, self._U_old_, self._V_old_, self._Qv_old_, self._Qc_old_, self._Qr_old_)
		
		# Run the stencil's compute function
		self._stencil.compute()
		
		# Bring the updated prognostic variables back to the original dimensions
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		s_new = self.boundary.from_computational_to_physical_domain(self._out_s, (nx, ny, nz))

		if type(self.boundary) == RelaxedSymmetricXZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U, (nx, ny, nz), change_sign = False)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V, (nx, ny, nz), change_sign = True) 
		elif type(self.boundary) == RelaxedSymmetricYZ:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U, (nx, ny, nz), change_sign = True)
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V, (nx, ny, nz), change_sign = False) 
		else:
			U_new = self.boundary.from_computational_to_physical_domain(self._out_U, (nx, ny, nz))
			V_new = self.boundary.from_computational_to_physical_domain(self._out_V, (nx, ny, nz)) 

		Qv_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qv, (nx, ny, nz))
		Qc_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qc, (nx, ny, nz))
		Qr_new = None if not self._moist_on else self.boundary.from_computational_to_physical_domain(self._out_Qr, (nx, ny, nz))

		# Apply the boundary conditions
		self.boundary.apply(s_new, s)
		self.boundary.apply(U_new, U)
		self.boundary.apply(V_new, V)
		if self._moist_on:
			self.boundary.apply(Qv_new, Qv)
			self.boundary.apply(Qc_new, Qc)
			self.boundary.apply(Qr_new, Qr)

		# Update the output state
		state_new.add(air_isentropic_density = s_new, 
					  x_momentum_isentropic = U_new, 
					  y_momentum_isentropic = V_new, 
					  water_vapor_isentropic_density = Qv_new, 
					  cloud_liquid_water_isentropic_density = Qc_new,
					  precipitation_water_isentropic_density = Qr_new)

		# Keep track of the current state for the next timestep
		self._s_old_[:,:,:]  = s_[:,:,:]
		self._U_old_[:,:,:]  = U_[:,:,:]
		self._V_old_[:,:,:]  = V_[:,:,:]
		if self._moist_on:
			self._Qv_old_[:,:,:] = Qv_[:,:,:]
			self._Qc_old_[:,:,:] = Qc_[:,:,:]
			self._Qr_old_[:,:,:] = Qr_[:,:,:]

		return state_new

	def _initialize_stencil(self, s_, u_, v_):
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
		self._allocate_inputs(s_, u_, v_)

		# Allocate the Numpy arrays which will store the output fields
		self._allocate_outputs(s_)

		# Set the computational domain and the backend
		_domain = gt.domain.Rectangle((self.nb, self.nb, 0), 
									  (self.nb + self.ni - 1, self.nb + self.nj - 1, self.nk - 1))
		_mode = self._backend

		# Instantiate the stencil
		if not self._moist_on:
			self._stencil = gt.NGStencil( 
				definitions_func = self._defs_stencil,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V,
						  'old_s': self._old_s, 'old_U': self._old_U, 'old_V': self._old_V},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s, 'out_U': self._out_U, 'out_V': self._out_V},
				domain = _domain, 
				mode = _mode)
		else:
			self._stencil = gt.NGStencil( 
				definitions_func = self._defs_stencil,
				inputs = {'in_s': self._in_s, 'in_u': self._in_u, 'in_v': self._in_v, 
						  'in_mtg': self._in_mtg, 'in_U': self._in_U, 'in_V': self._in_V,
						  'in_Qv': self._in_Qv, 'in_Qc': self._in_Qc, 'in_Qr': self._in_Qr,
						  'old_s': self._old_s, 'old_U': self._old_U, 'old_V': self._old_V,
						  'old_Qv': self._old_Qv, 'old_Qc': self._old_Qc, 'old_Qr': self._old_Qr},
				global_inputs = {'dt': self._dt},
				outputs = {'out_s': self._out_s, 'out_U': self._out_U, 'out_V': self._out_V,
						   'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr},
				domain = _domain, 
				mode = _mode)

	def _allocate_inputs(self, s_, u_, v_):
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
		super()._allocate_inputs(s_, u_, v_)

		# Allocate the Numpy arrays which will carry the solution at the previous time step
		self._old_s = np.zeros_like(s_)
		self._old_U = np.zeros_like(s_)
		self._old_V = np.zeros_like(s_)
		if self._moist_on:
			self._old_Qv = np.zeros_like(s_)
			self._old_Qc = np.zeros_like(s_)
			self._old_Qr = np.zeros_like(s_)

	def _set_inputs(self, dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_, 
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
		super()._set_inputs(dt, s_, u_, v_, mtg_, U_, V_, Qv_, Qc_, Qr_)
		
		# Update the Numpy arrays carrying the solution at the previous time step
		self._old_s[:,:,:] = s_old_[:,:,:]
		self._old_U[:,:,:] = U_old_[:,:,:]
		self._old_V[:,:,:] = V_old_[:,:,:]
		if self._moist_on:
			self._old_Qv[:,:,:] = Qv_old_[:,:,:]
			self._old_Qc[:,:,:] = Qc_old_[:,:,:]
			self._old_Qr[:,:,:] = Qr_old_[:,:,:]

	def _defs_stencil(self, dt, in_s, in_u, in_v, in_mtg, in_U, in_V, old_s, old_U, old_V,
					  in_Qv = None, in_Qc = None, in_Qr = None, old_Qv = None, old_Qc = None, old_Qr = None):
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

		out_s[i, j, k] = old_s[i, j, k] - 2. * dt * ((flux_s_x[i+1, j, k] - flux_s_x[i, j, k]) / self._grid.dx +
						 					         (flux_s_y[i, j+1, k] - flux_s_y[i, j, k]) / self._grid.dy)
		out_U[i, j, k] = old_U[i, j, k] - 2. * dt * ((flux_U_x[i+1, j, k] - flux_U_x[i, j, k]) / self._grid.dx +
						 					         (flux_U_y[i, j+1, k] - flux_U_y[i, j, k]) / self._grid.dy) \
										- dt * in_s[i, j, k] * (in_mtg[i+1, j, k] - in_mtg[i-1, j, k]) / self._grid.dx
		out_V[i, j, k] = old_V[i, j, k] - 2. * dt * ((flux_V_x[i+1, j, k] - flux_V_x[i, j, k]) / self._grid.dx +
						 					         (flux_V_y[i, j+1, k] - flux_V_y[i, j, k]) / self._grid.dy) \
										- dt * in_s[i, j, k] * (in_mtg[i, j+1, k] - in_mtg[i, j-1, k]) / self._grid.dy
		if self._moist_on:
			out_Qv[i, j, k] = old_Qv[i, j, k] - 2. * dt * ((flux_Qv_x[i+1, j, k] - flux_Qv_x[i, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qv_y[i, j+1, k] - flux_Qv_y[i, j, k]) / self._grid.dy)
			out_Qc[i, j, k] = old_Qc[i, j, k] - 2. * dt * ((flux_Qc_x[i+1, j, k] - flux_Qc_x[i, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qc_y[i, j+1, k] - flux_Qc_y[i, j, k]) / self._grid.dy)
			out_Qr[i, j, k] = old_Qr[i, j, k] - 2. * dt * ((flux_Qr_x[i+1, j, k] - flux_Qr_x[i, j, k]) / self._grid.dx +
						 						  	 	   (flux_Qr_y[i, j+1, k] - flux_Qr_y[i, j, k]) / self._grid.dy)

			return out_s, out_U, out_V, out_Qv, out_Qc, out_Qr
		else:
			return out_s, out_U, out_V

