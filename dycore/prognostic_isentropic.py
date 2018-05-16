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
import copy
import numpy as np

import gridtools as gt
from tasmania.dycore.flux_isentropic import FluxIsentropic
from tasmania.dycore.flux_sedimentation import FluxSedimentation
from tasmania.namelist import datatype
from tasmania.storages.grid_data import GridData
from tasmania.storages.state_isentropic import StateIsentropic
import tasmania.utils.utils as utils

class PrognosticIsentropic:
	"""
	Abstract base class whose derived classes implement different schemes to carry out the prognostic steps of 
	the three-dimensional moist isentropic dynamical core. The conservative form of the governing equations is used.

	Attributes
	----------
	fast_tendency_parameterizations : list
		List containing instances of derived classes of 
		:class:`~tasmania.parameterizations.fast_tendencies.FastTendency` which are in charge of
		calculating fast-varying tendencies.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, flux_scheme, grid, moist_on, backend, physics_dynamics_coupling_on, 
				 sedimentation_on, sedimentation_flux_type, sedimentation_substeps):
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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		physics_dynamics_coupling_on : bool
			:obj:`True` to couple physics with dynamics, i.e., to account for the change over time in potential temperature,
			:obj:`False` otherwise.
		sedimentation_on : bool
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
		sedimentation_flux_type : str
			String specifying the method used to compute the numerical sedimentation flux. Available options are:

			- 'first_order_upwind', for the first-order upwind scheme;
			- 'second_order_upwind', for the second-order upwind scheme.

		sedimentation_substeps : int
			Number of sub-timesteps to perform in order to integrate the sedimentation flux. 
		"""
		# Keep track of the input parameters
		self._flux_scheme                  = flux_scheme
		self._grid                         = grid
		self._moist_on                     = moist_on
		self._backend                      = backend
		self._physics_dynamics_coupling_on = physics_dynamics_coupling_on
		self._sedimentation_on             = sedimentation_on
		self._sedimentation_flux_type      = sedimentation_flux_type
		self._sedimentation_substeps       = sedimentation_substeps

		# Instantiate the class computing the numerical horizontal and vertical fluxes
		self._flux = FluxIsentropic.factory(flux_scheme, grid, moist_on)

		# Instantiate the class computing the vertical derivative of the sedimentation flux
		if sedimentation_on:
			self._flux_sedimentation = FluxSedimentation.factory(sedimentation_flux_type)

		# Initialize the attributes representing the diagnostic step and the lateral boundary conditions
		# Remark: these should be suitably set before calling the stepping method for the first time
		self._diagnostic, self._boundary = None, None

		# Initialize the attribute in charge of calculating the raindrop fall velocity
		self._microphysics = None

		# Initialize the list of parameterizations providing fast-varying cloud microphysical tendencies
		self.fast_tendency_parameterizations = []

		# Initialize the pointer to the compute function of the stencil in charge of coupling physics with dynamics
		# This will be properly re-directed the first time the corresponding forward method is invoked
		self._stencil_stepping_by_coupling_physics_with_dynamics = None

	@property
	def diagnostic(self):
		"""
		Get the attribute implementing the diagnostic step of the three-dimensional moist isentropic dynamical core.
		If this is set to :obj:`None`, a :class:`ValueError` is thrown.
		
		Return
		------
		obj :
			:class:`~tasmania.dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the 
			three-dimensional moist isentropic dynamical core.
		"""
		if self._diagnostic is None:
			raise ValueError("""The attribute which is supposed to implement the diagnostic step of the moist isentroic """ \
							 """dynamical core is actually :obj:`None`. Please set it correctly.""")
		return self._diagnostic

	@diagnostic.setter
	def diagnostic(self, value):
		"""
		Set the attribute implementing the diagnostic step of the three-dimensional moist isentropic dynamical core.

		Parameter
		---------
		value : obj
			:class:`~tasmania.dycore.diagnostic_isentropic.DiagnosticIsentropic` carrying out the diagnostic step of the 
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
			Instance of the derived class of :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` implementing
			the horizontal boundary conditions.
		"""
		if self._boundary is None:
			raise ValueError("""The attribute which is supposed to implement the horizontal boundary conditions """ \
							 """is actually None. Please set it correctly.""")
		return self._boundary

	@boundary.setter
	def boundary(self, value):
		"""
		Set the attribute implementing the horizontal boundary conditions.

		Parameter
		---------
		value : obj
			Instance of the derived class of :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` implementing the 
			horizontal boundary conditions.
		"""
		self._boundary = value

	@property
	def nb(self):
		"""
		Get the number of lateral boundary layers.

		Return
		------
		int :
			The number of lateral boundary layers.
		"""
		return self._flux.nb

	@property
	def microphysics(self):
		"""
		Get the attribute in charge of calculating the raindrop fall velocity.
		If this is set to :obj:`None`, a :class:`ValueError` is thrown.

		Return
		------
		obj :
			Instance of a derived class of either 
			:class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics` or 
			:class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` 
			which provides the raindrop fall velocity.
		"""
		if self._microphysics is None:
			return ValueError('The attribute taking care of microphysics has not been set.')
		return self._microphysics

	@microphysics.setter
	def microphysics(self, micro):
		"""
		Set the attribute in charge of calculating the raindrop fall velocity.

		Parameters
		----------
		micro : obj
			Instance of a derived class of either 
			:class:`~tasmania.parameterizations.tendencies.TendencyMicrophysics` or 
			:class:`~tasmania.parameterizations.adjustments.AdjustmentMicrophysics` 
			which provides the raindrop fall velocity.
		"""
		self._microphysics = micro

	@abc.abstractmethod
	def step_neglecting_vertical_advection(self, dt, state, state_old = None, tendencies = None):
		"""
		Method advancing the conservative, prognostic model variables one time step forward.
		Only horizontal derivates are considered; possible vertical derivatives are disregarded.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure or air_pressure_on_interface_levels (:math:`z`-staggered);
			* montgomery_potential (isentropic);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		state_old : `obj`, optional
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the old state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		tendencies : `obj`, optional
			:class:`~tasmania.storages.grid_data.GridData` storing the following tendencies:

			* tendency_of_mass_fraction_of_water_vapor_in_air (unstaggered);
			* tendency_of_mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* tendency_of_mass_fraction_of_precipitation_water_in_air (unstaggered).

			Default is :obj:`None`.

		Return
		------
		obj :
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).
		"""

	def step_coupling_physics_with_dynamics(self, dt, state_now, state_prv, tendencies):
		"""
		Method advancing the conservative, prognostic model variables one time step forward by coupling physics with
		dynamics, i.e., by accounting for the change over time in potential temperature.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state_now : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

		state_prv : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
			the state stepped taking only the horizontal derivatives into account. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

			This may be the output of 
			:meth:`~tasmania.dycore.prognostic_isentropic.PrognosticIsentropic.step_neglecting_vertical_advection`.
		tendencies : obj
			:class:`~tasmania.storages.grid_data.GridData` collecting the following tendencies:
			
			* tendency_of_air_potential_temperature (unstaggered).

		Return
		------
		obj :
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` containing the updated prognostic variables, i.e.,

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).
		"""
		# The first time this method is invoked, initialize the GT4Py stencil
		if self._stencil_stepping_by_coupling_physics_with_dynamics is None:
			self._stencil_stepping_by_coupling_physics_with_dynamics_initialize(state_now)

		# Set stencil's inputs
		self._stencil_stepping_by_coupling_physics_with_dynamics_set_inputs(dt, state_now, state_prv, tendencies)

		# Run the stencil
		self._stencil_stepping_by_coupling_physics_with_dynamics.compute()

		# Set the lower and upper layers
		nb = self._flux.nb
		self._out_s[:,:,:nb], self._out_s[:,:,-nb:] = self._in_s_prv[:,:,:nb], self._in_s_prv[:,:,-nb:]
		self._out_U[:,:,:nb], self._out_U[:,:,-nb:] = self._in_U_prv[:,:,:nb], self._in_U_prv[:,:,-nb:]
		self._out_V[:,:,:nb], self._out_V[:,:,-nb:] = self._in_V_prv[:,:,:nb], self._in_V_prv[:,:,-nb:]
		if self._moist_on:
			self._out_Qv[:,:,:nb], self._out_Qv[:,:,-nb:] = self._in_Qv_prv[:,:,:nb], self._in_Qv_prv[:,:,-nb:]
			self._out_Qc[:,:,:nb], self._out_Qc[:,:,-nb:] = self._in_Qc_prv[:,:,:nb], self._in_Qc_prv[:,:,-nb:]
			self._out_Qr[:,:,:nb], self._out_Qr[:,:,-nb:] = self._in_Qr_prv[:,:,:nb], self._in_Qr_prv[:,:,-nb:]

		# Instantiate the output state
		time_now = utils.convert_datetime64_to_datetime(state_now['air_isentropic_density'].coords['time'].values[0])
		state_new = StateIsentropic(time_now + dt, self._grid,
									air_isentropic_density = self._out_s, 
					  				x_momentum_isentropic  = self._out_U, 
					  				y_momentum_isentropic  = self._out_V)
		if self._moist_on:
			state_new.add_variables(time_now + dt,
									water_vapor_isentropic_density         = self._out_Qv, 
					  	  			cloud_liquid_water_isentropic_density  = self._out_Qc,
					  	  			precipitation_water_isentropic_density = self._out_Qr)

		return state_new

	@abc.abstractmethod
	def step_integrating_sedimentation_flux(self, dt, state_now, state_prv, diagnostics = None):
		"""
		Method advancing the mass fraction of precipitation water by taking the sedimentation into account.
		For the sake of numerical stability, a time-splitting strategy is pursued, i.e., sedimentation is resolved
		using a timestep which may be smaller than that specified by the user.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state_now : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* height or height_on_interface_levels (:math:`z`-staggered);
			* mass_fraction_of_precipitation_water_in air (unstaggered).

		state_prv : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
			the state stepped without taking the sedimentation flux into account. 
			It should contain the following variables:

			* mass_fraction_of_precipitation_water_in_air (unstaggered).

			This may be the output of either
			:meth:`~tasmania.dycore.prognostic_isentropic.PrognosticIsentropic.step_neglecting_vertical_advection` or
			:meth:`~tasmania.dycore.prognostic_isentropic.PrognosticIsentropic.step_coupling_physics_with_dynamics`.
		diagnostics : `obj`, optional
			:class:`~tasmania.storages.grid_data.GridData` collecting the following diagnostics:

			* accumulated_precipitation (unstaggered, two-dimensional);
			* precipitation (unstaggered, two-dimensional).

		Returns
		-------
		state_new : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` containing the following updated variables:
			
			* mass_fraction_of_precipitation_water_in air (unstaggered).

		diagnostics_out : obj
			:class:`~tasmania.storages.grid_data.GridData` collecting the output diagnostics, i.e.:

			* accumulated_precipitation (unstaggered, two-dimensional);
			* precipitation (unstaggered, two-dimensional).
		"""

	@staticmethod
	def factory(time_scheme, flux_scheme, grid, moist_on, backend, physics_dynamics_coupling_on, 
				sedimentation_on, sedimentation_flux_type, sedimentation_substeps):
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
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool 
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		backend : obj 
			:class:`gridtools.Mode` specifying the backend for the GT4Py stencils.
		physics_dynamics_coupling_on : bool
			:obj:`True` to couple physics with dynamics, i.e., to account for the change over time in potential temperature,
			:obj:`False` otherwise.
		sedimentation_on : bool
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
		sedimentation_flux_type : str
			String specifying the method used to compute the numerical sedimentation flux. Available options are:

			- 'first_order_upwind', for the first-order upwind scheme;
			- 'second_order_upwind', for the second-order upwind scheme.

		sedimentation_substeps : int
			Number of sub-timesteps to perform in order to integrate the sedimentation flux. 

		Return
		------
		obj :
			An instace of the derived class implementing the scheme specified by :data:`scheme`.
		"""
		if time_scheme == 'forward_euler':
			from tasmania.dycore.prognostic_isentropic_forward_euler import PrognosticIsentropicForwardEuler
			return PrognosticIsentropicForwardEuler(flux_scheme, grid, moist_on, backend, 
													physics_dynamics_coupling_on, sedimentation_on, 
													sedimentation_flux_type, sedimentation_substeps)
		elif time_scheme == 'centered':
			from tasmania.dycore.prognostic_isentropic_centered import PrognosticIsentropicCentered
			return PrognosticIsentropicCentered(flux_scheme, grid, moist_on, backend,
												physics_dynamics_coupling_on, sedimentation_on, 
												sedimentation_flux_type, sedimentation_substeps)
		else:
			raise ValueError('Unknown time integration scheme.')

	def _stencils_stepping_by_neglecting_vertical_advection_allocate_inputs(self, mi, mj, tendencies):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencils which step the solution
		disregarding the vertical advection.

		Parameters
		----------
		mi : int
			:math:`x`-extent of an input array representing an :math:`x`-unstaggered field.
		mj : int
			:math:`y`-extent of an input array representing a :math:`y`-unstaggered field.
		tendencies : obj
			:class:`~tasmania.storages.grid_data.GridData` storing the following tendencies:

			* tendency_of_mass_fraction_of_water_vapor_in_air (unstaggered);
			* tendency_of_mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* tendency_of_mass_fraction_of_precipitation_water_in_air (unstaggered).
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Keep track of the input arguments
		self._mi, self._mj = mi, mj

		# Instantiate a GT4Py Global representing the timestep
		self._dt = gt.Global()

		# Determine the size of the input arrays
		# These arrays may be shared with the stencil in charge of coupling physics with dynamics
		li = mi if not self._physics_dynamics_coupling_on else max(mi, nx)
		lj = mj if not self._physics_dynamics_coupling_on else max(mj, ny)

		# Allocate the input Numpy arrays which may be shared with the stencil 
		# in charge of coupling physics with dynamics
		self._in_s = np.zeros((li, lj, nz), dtype = datatype)
		self._in_U = np.zeros((li, lj, nz), dtype = datatype)
		self._in_V = np.zeros((li, lj, nz), dtype = datatype)
		if self._moist_on:
			self._in_Qv = np.zeros((li, lj, nz), dtype = datatype)
			self._in_Qc = np.zeros((li, lj, nz), dtype = datatype)

			# The array which will store the input mass fraction of precipitation water may be shared
			# either with stencil in charge of coupling physics with dynamics, or the stencil taking 
			# care of sedimentation
			li = mi if not (self._sedimentation_on and self._physics_dynamics_coupling_on) else max(mi, nx)
			lj = mj if not (self._sedimentation_on and self._physics_dynamics_coupling_on) else max(mj, ny)
			self._in_Qr = np.zeros((li, lj, nz), dtype = datatype)

		# Allocate the input Numpy arrays not shared with any other stencil
		self._in_u   = np.zeros((mi+1,   mj, nz), dtype = datatype)
		self._in_v   = np.zeros((  mi, mj+1, nz), dtype = datatype)
		self._in_mtg = np.zeros((  mi,   mj, nz), dtype = datatype)
		if tendencies is not None:
			if tendencies['tendency_of_mass_fraction_of_water_vapor_in_air'] is not None:
				self._in_qv_tnd = np.zeros((mi, mj, nz), dtype = datatype)
			if tendencies['tendency_of_mass_fraction_of_cloud_liquid_water_in_air'] is not None:
				self._in_qc_tnd = np.zeros((mi, mj, nz), dtype = datatype)
			if tendencies['tendency_of_mass_fraction_of_precipitation_water_in_air'] is not None:
				self._in_qr_tnd = np.zeros((mi, mj, nz), dtype = datatype)

	def _stencils_stepping_by_neglecting_vertical_advection_allocate_outputs(self, mi, mj):
		"""
		Allocate the Numpy arrays which will store the solution updated by neglecting the vertical advection.

		Parameters
		----------
		mi : int
			:math:`x`-extent of an output array representing an :math:`x`-unstaggered field.
		mj : int
			:math:`y`-extent of an output array representing a :math:`y`-unstaggered field.
		"""
		# Keep track of the input arguments
		self._mi, self._mj = mi, mj

		# Determine the size of the output arrays
		# These arrays may be shared with the stencil in charge of coupling physics with dynamics
		li = mi if not self._physics_dynamics_coupling_on else max(mi, nx)
		lj = mj if not self._physics_dynamics_coupling_on else max(mj, ny)
		nz = self._grid.nz

		# Allocate the output Numpy arrays which may be shared with the stencil in charge 
		# of coupling physics with dynamics
		self._out_s = np.zeros((li, lj, nz), dtype = datatype)
		self._out_U = np.zeros((li, lj, nz), dtype = datatype)
		self._out_V = np.zeros((li, lj, nz), dtype = datatype)
		if self._moist_on:
			self._out_Qv = np.zeros((li, lj, nz), dtype = datatype)
			self._out_Qc = np.zeros((li, lj, nz), dtype = datatype)

			# The array which will store the output mass fraction of precipitation water may be shared
			# either with stencil in charge of coupling physics with dynamics, or the stencil taking 
			# care of sedimentation
			li = mi if not (self._sedimentation_on and self._physics_dynamics_coupling_on) else max(mi, nx)
			lj = mj if not (self._sedimentation_on and self._physics_dynamics_coupling_on) else max(mj, ny)
			self._out_Qr = np.zeros((li, lj, nz), dtype = datatype)

	def _stencils_stepping_by_neglecting_vertical_advection_set_inputs(self, dt, state, tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils which step the solution
		disregarding the vertical advection.

		Parameters
		----------
		dt : obj 
			A :class:`datetime.timedelta` representing the time step.
		state : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* y_velocity (:math:`y`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* montgomery_potential (isentropic);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

		tendencies : obj
			:class:`~tasmania.storages.grid_data.GridData` storing the following tendencies:

			* tendency_of_mass_fraction_of_water_vapor_in_air (unstaggered);
			* tendency_of_mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* tendency_of_mass_fraction_of_precipitation_water_in_air (unstaggered).
		"""
		# Shortcuts
		mi, mj = self._mi, self._mj

		# Update the local time step
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds

		# Extract the Numpy arrays representing the current solution
		s   = state['air_isentropic_density'].values[:,:,:,0]
		u   = state['x_velocity'].values[:,:,:,0]
		v   = state['y_velocity'].values[:,:,:,0]
		mtg = state['montgomery_potential'].values[:,:,:,0]
		U   = state['x_momentum_isentropic'].values[:,:,:,0]
		V   = state['y_momentum_isentropic'].values[:,:,:,0]
		if self._moist_on:
			Qv = state['water_vapor_isentropic_density'].values[:,:,:,0]
			Qc = state['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			Qr = state['precipitation_water_isentropic_density'].values[:,:,:,0]
		if tendencies is not None:
			if tendencies['tendency_of_mass_fraction_of_water_vapor_in_air'] is not None:
				qv_tnd = tendencies['tendency_of_mass_fraction_of_water_vapor_in_air'].values[:,:,:,0]
			if tendencies['tendency_of_mass_fraction_of_cloud_liquid_water_in_air'] is not None:
				qc_tnd = tendencies['tendency_of_mass_fraction_of_cloud_liquid_water_in_air'].values[:,:,:,0]
			if tendencies['tendency_of_mass_fraction_of_precipitation_water_in_air'] is not None:
				qr_tnd = tendencies['tendency_of_mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]
		
		# Update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._in_s  [  :mi,   :mj, :] = self.boundary.from_physical_to_computational_domain(s)
		self._in_u  [:mi+1,   :mj, :] = self.boundary.from_physical_to_computational_domain(u)
		self._in_v  [  :mi, :mj+1, :] = self.boundary.from_physical_to_computational_domain(v)
		self._in_mtg[  :mi,   :mj, :] = self.boundary.from_physical_to_computational_domain(mtg)
		self._in_U  [  :mi,   :mj, :] = self.boundary.from_physical_to_computational_domain(U)
		self._in_V  [  :mi,   :mj, :] = self.boundary.from_physical_to_computational_domain(V)
		if self._moist_on:
			self._in_Qv[:mi, :mj, :] = self.boundary.from_physical_to_computational_domain(Qv)
			self._in_Qc[:mi, :mj, :] = self.boundary.from_physical_to_computational_domain(Qc)
			self._in_Qr[:mi, :mj, :] = self.boundary.from_physical_to_computational_domain(Qr)
		if tendencies is not None:
			if tendencies['tendency_of_mass_fraction_of_water_vapor_in_air'] is not None:
				self._in_qv_tnd[:mi, :mj, :] = self.boundary.from_physical_to_computational_domain(qv_tnd)
			if tendencies['tendency_of_mass_fraction_of_cloud_liquid_water_in_air'] is not None:
				self._in_qc_tnd[:mi, :mj, :] = self.boundary.from_physical_to_computational_domain(qc_tnd)
			if tendencies['tendency_of_mass_fraction_of_precipitation_water_in_air'] is not None:
				self._in_qr_tnd[:mi, :mj, :] = self.boundary.from_physical_to_computational_domain(qr_tnd)

	def _stencil_stepping_by_coupling_physics_with_dynamics_initialize(self, state_now):
		"""
		Initialize the GT4Py stencil in charge of stepping the solution by coupling physics with dynamics,
		i.e., by accounting for the change over time in potential temperature.

		Parameters
		----------
		state_now : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered).
		"""
		# Allocate stencil's inputs
		self._stencil_stepping_by_coupling_physics_with_dynamics_allocate_inputs()

		# Allocate stencil's outputs
		self._stencil_stepping_by_coupling_physics_with_dynamics_allocate_outputs()

		# Set stencil's inputs and outputs
		_inputs = {'in_w': self._in_w, 
				   'in_s_now': self._in_s, 'in_s_prv': self._in_s_prv, 
				   'in_U_now': self._in_U, 'in_U_prv': self._in_U_prv, 
				   'in_V_now': self._in_V, 'in_V_prv': self._in_V_prv}
		_outputs = {'out_s': self._out_s, 'out_U': self._out_U, 'out_V': self._out_V}
		if self._moist_on:
			_inputs.update({'in_Qv_now': self._in_Qv, 'in_Qv_prv': self._in_Qv_prv, 
						    'in_Qc_now': self._in_Qc, 'in_Qc_prv': self._in_Qc_prv, 
						    'in_Qr_now': self._in_Qr, 'in_Qr_prv': self._in_Qr_prv})
			_outputs.update({'out_Qv': self._out_Qv, 'out_Qc': self._out_Qc, 'out_Qr': self._out_Qr})

		# Set stencil's domain
		nb = self.nb
		ni, nj, nk = self._grid.nx, self._grid.ny, self._grid.nz - 2 * nb
		_domain = gt.domain.Rectangle((0, 0, nb), (ni - 1, nj - 1, nb + nk - 1))

		# Instantiate the stencil
		self._stencil_stepping_by_coupling_physics_with_dynamics = gt.NGStencil( 
			definitions_func = self._stencil_stepping_by_coupling_physics_with_dynamics_defs,
			inputs           = _inputs,
			global_inputs    = {'dt': self._dt},
			outputs          = _outputs,
			domain           = _domain, 
			mode             = self.backend)

	def _stencil_stepping_by_coupling_physics_with_dynamics_allocate_inputs(self):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencil which step the solution
		by coupling physics with dynamics, i.e., accounting for the change over time in potential temperature.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy array which will represent the vertical velocity
		self._in_w = np.zeros((ny, nx, nz), dtype = datatype)

		# Allocate the Numpy arrays which will represent the provisional model variables
		self._in_s_prv = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_U_prv = np.zeros((nx, ny, nz), dtype = datatype)
		self._in_V_prv = np.zeros((nx, ny, nz), dtype = datatype)
		if self._moist_on:
			self._in_Qv_prv = np.zeros((nx, ny, nz), dtype = datatype)
			self._in_Qc_prv = np.zeros((nx, ny, nz), dtype = datatype)
			self._in_Qr_prv = np.zeros((nx, ny, nz), dtype = datatype)

		# Allocate objects which may be shared with the stencil stepping the solution by neglecting vertical advection
		if self._stencil_stepping_by_neglecting_vertical_advection is None:
			# Instantiate a GT4Py Global representing the timestep
			self._dt = gt.Global()

			# Allocate the Numpy arrays which will represent the current time model variables
			self._in_s = np.zeros((nx, ny, nz), dtype = datatype)
			self._in_U = np.zeros((nx, ny, nz), dtype = datatype)
			self._in_V = np.zeros((nx, ny, nz), dtype = datatype)
			if self._moist_on:
				self._in_Qv = np.zeros((nx, ny, nz), dtype = datatype)
				self._in_Qc = np.zeros((nx, ny, nz), dtype = datatype)
				self._in_Qr = np.zeros((nx, ny, nz), dtype = datatype)

	def _stencil_stepping_by_coupling_physics_with_dynamics_allocate_outputs(self):
		"""
		Allocate the Numpy arrays which will store the solution updated by coupling physics with dynamics.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

		# Allocate the Numpy arrays which will store the output fields
		# These arrays may be shared with the stencil stepping the solution by neglecting vertical advection
		if self._stencil_stepping_by_neglecting_vertical_advection is None:
			self._out_s = np.zeros((nx, ny, nz), dtype = datatype)
			self._out_U = np.zeros((nx, ny, nz), dtype = datatype)
			self._out_V = np.zeros((nx, ny, nz), dtype = datatype)
			if self._moist_on:
				self._out_Qv = np.zeros((nx, ny, nz), dtype = datatype)
				self._out_Qc = np.zeros((nx, ny, nz), dtype = datatype)
				self._out_Qr = np.zeros((nx, ny, nz), dtype = datatype)

	def _stencil_stepping_by_coupling_physics_with_dynamics_set_inputs(self, dt, state_now, state_prv, tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencil which steps the solution
		by integrating the vertical advection, i.e., by accounting for the change over time in potential temperature.

		Parameters
		----------
		dt : obj 
			A :class:`datetime.timedelta` representing the time step.
		state_now : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

		state_prv : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the provisional state, i.e.,
			the state stepped taking only the horizontal derivatives into account. 
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_momentum_isentropic (unstaggered);
			* y_momentum_isentropic (unstaggered);
			* water_vapor_isentropic_density (unstaggered, optional);
			* cloud_liquid_water_isentropic_density (unstaggered, optional);
			* precipitation_water_isentropic_density (unstaggered, optional).

			This may be the output of 
			:meth:`~tasmania.dycore.prognostic_isentropic.PrognosticIsentropic.step_neglecting_vertical_advection`.
		tendencies : obj
			:class:`~tasmania.storages.grid_data.GridData` collecting the following tendencies:
			
			* tendency_of_air_potential_temperature (unstaggered).
		"""
		# Shortcuts
		nx, ny = self._grid.nx, self._grid.ny

		# Update the local time step
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds

		# Update the Numpy array representing the vertical velocity
		self._in_w[:,:,:] = tendencies['tendency_of_air_potential_temperature'].values[:,:,:,0]

		# Update the Numpy arrays representing the current time model variables
		# Recall: these arrays may be shared with the stencil stepping the solution by neglecting vertical advection
		self._in_s[:nx, :ny, :] = state_now['air_isentropic_density'].values[:,:,:,0]
		self._in_U[:nx, :ny, :] = state_now['x_momentum_isentropic'].values[:,:,:,0]
		self._in_V[:nx, :ny, :] = state_now['y_momentum_isentropic'].values[:,:,:,0]
		if self._moist_on:
			self._in_Qv[:nx, :ny, :] = state_now['water_vapor_isentropic_density'].values[:,:,:,0]
			self._in_Qc[:nx, :ny, :] = state_now['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			self._in_Qr[:nx, :ny, :] = state_now['precipitation_water_isentropic_density'].values[:,:,:,0]

		# Update the Numpy arrays representing the provisional model variables
		self._in_s_prv[:,:,:] = state_prv['air_isentropic_density'].values[:,:,:,0]
		self._in_U_prv[:,:,:] = state_prv['x_momentum_isentropic'].values[:,:,:,0]
		self._in_V_prv[:,:,:] = state_prv['y_momentum_isentropic'].values[:,:,:,0]
		if self._moist_on:
			self._in_Qv_prv[:,:,:] = state_prv['water_vapor_isentropic_density'].values[:,:,:,0]
			self._in_Qc_prv[:,:,:] = state_prv['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			self._in_Qr_prv[:,:,:] = state_prv['precipitation_water_isentropic_density'].values[:,:,:,0]

	@abc.abstractmethod
	def _stencil_stepping_by_coupling_physics_with_dynamics_defs(dt, in_w, 
																 in_s_now, in_s_prv, 
																 in_U_now, in_U_prv, 
																 in_V_now, in_V_prv,
															  	 Qv_now = None, Qv_prv = None, 
															  	 Qc_now = None, Qc_prv = None,
															  	 Qr_now = None, Qr_prv = None):
		"""
		GT4Py stencil stepping the solution by coupling physics with dynamics, i.e., by accounting for the
		change over time in potential temperature.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj 
			:class:`gridtools.Global` representing the time step.
		in_w : array_like
			:class:`numpy.ndarray` representing the vertical velocity, i.e., the change over time in potential temperature.
		in_s_now : obj
			:class:`gridtools.Equation` representing the current isentropic density. 
		in_s_prv : obj 
			:class:`gridtools.Equation` representing the provisional isentropic density. 
		in_U_now : obj 
			:class:`gridtools.Equation` representing the current :math:`x`-momentum.
		in_U_prv : obj 
			:class:`gridtools.Equation` representing the provisional :math:`x`-momentum.
		in_V_now : obj 
			:class:`gridtools.Equation` representing the current :math:`y`-momentum.
		in_V_prv : obj
			:class:`gridtools.Equation` representing the provisional :math:`y`-momentum.
		in_Qv_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of water vapor.
		in_Qv_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of water vapor.
		in_Qc_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of cloud liquid water.
		in_Qc_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of cloud liquid water.
		in_Qr_now : `obj`, optional 
			:class:`gridtools.Equation` representing the current isentropic density of precipitation water.
		in_Qr_prv : `obj`, optional
			:class:`gridtools.Equation` representing the provisional isentropic density of precipitation water.

		Returns
		-------
		out_s : obj
			:class:`gridtools.Equation` representing the updated isentropic density. 
		out_U : obj 
			:class:`gridtools.Equation` representing the updated :math:`x`-momentum.
		out_V : obj 
			:class:`gridtools.Equation` representing the updated :math:`y`-momentum.
		out_Qv : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of water vapor.
		out_Qc : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of cloud liquid water.
		out_Qr : `obj`, optional 
			:class:`gridtools.Equation` representing the updated isentropic density of precipitation water.
		"""
