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
import math
import numpy as np

import gridtools as gt
from tasmania.dynamics.diagnostics import IsentropicDiagnostics
from tasmania.dynamics.dycore import DynamicalCore
from tasmania.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.dynamics.horizontal_smoothing import HorizontalSmoothing
from tasmania.dynamics.isentropic_prognostic import IsentropicPrognostic
from tasmania.dynamics.vertical_damping import VerticalDamping
import tasmania.utils.utils as utils

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


class IsentropicDynamicalCore(DynamicalCore):
	"""
	This class inherits :class:`~tasmania.dynamics.dycore.DynamicalCore`
	to implement the three-dimensional (moist) isentropic dynamical core.
	The class supports different numerical schemes to carry out the prognostic
	steps of the dynamical core, and different types of lateral boundary conditions.
	The conservative form of the governing equations is used.
	"""
	def __init__(self, grid, moist_on, time_integration_scheme,
				 horizontal_flux_scheme, horizontal_boundary_type,
				 fast_parameterizations=None,
				 damp_on=True, damp_type='rayleigh', damp_depth=15,
				 damp_max=0.0002, damp_at_every_stage=True,
				 smooth_on=True, smooth_type='first_order', smooth_damp_depth=10,
				 smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
				 smooth_moist_on=False, smooth_moist_type='first_order',
				 smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
				 smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
				 adiabatic_flow=True, vertical_flux_scheme=None,
				 sedimentation_on=False, sedimentation_flux_type='first_order_upwind',
				 sedimentation_substeps=2, raindrop_fall_velocity_diagnostic=None,
				 backend=gt.mode.NUMPY, dtype=datatype, physical_constants=None):
		"""
		Constructor.

		Parameters
		----------
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		time_integration_scheme : str
			String specifying the time stepping method to implement.
			See :class:`~tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic`
			for sll available options.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalFlux`
			for sll available options.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for sll available options.
		fast_parameterizations : `obj`, None
			:class:`~tasmania.physics.composite.PhysicsComponentComposite`
			object, wrapping the fast physical parameterizations.
			Here, *fast* refers to the fact that these parameterizations
			are evaluated *before* each stage of the dynamical core.
		damp_on : `bool`, optional
			:obj:`True` to enable vertical damping, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		damp_type : `str`, optional
			String specifying the type of vertical damping to apply. Defaults to 'rayleigh'.
			See :class:`~tasmania.dynamics.vertical_damping.VerticalDamping`
			for sll available options.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Defaults to 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Defaults to 0.0002.
		damp_at_every_stage : `bool`, optional
			:obj:`True` to carry out the damping at each stage performed by the
			dynamical core, :obj:`False` to carry out the damping only at the end
			of each timestep. Defaults to :obj:`True`.
		smooth_on : `bool`, optional
			:obj:`True` to enable numerical smoothing, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		smooth_type: `str`, optional
			String specifying the smoothing technique to implement.
			Defaults to 'first-order'. See
			:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalSmoothing`
			for all available options.
		smooth_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region. Defaults to 10.
		smooth_coeff : `float`, optional
			Smoothing coefficient. Defaults to 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. Defaults to 0.24.
			See :class:`~tasmania.dynamics.horizontal_smoothing.HorizontalSmoothing`
			for further details.
		smooth_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing at each stage performed by the
			dynamical core, :obj:`False` to apply numerical smoothing only at the end
			of each timestep. Defaults to :obj:`True`.
		smooth_moist_on : `bool`, optional
			:obj:`True` to enable numerical smoothing on the water constituents,
			:obj:`False` otherwise. Defaults to :obj:`True`.
		smooth_moist_type: `str`, optional
			String specifying the smoothing technique to apply to the water constituents.
			Defaults to 'first-order'. See
			:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalSmoothing`
			for all available options.
		smooth_moist_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region for the
			water constituents. Defaults to 10.
		smooth_moist_coeff : `float`, optional
			Smoothing coefficient for the water constituents. Defaults to 0.03.
		smooth_moist_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient for the water constituents.
			Defaults to 0.24. See
			:class:`~tasmania.dynamics.horizontal_smoothing.HorizontalSmoothing`
			for further details.
		smooth_moist_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing to the water constituents
			at each stage performed by the dynamical core, :obj:`False` to apply
			numerical smoothing only at the end of each timestep. Defaults to :obj:`True`.
		adiabatic_flow : `bool`, optional
			:obj:`True` for an adiabatic atmosphere, in which the potential temperature
			is conserved, :obj:`False` otherwise. Defaults to :obj:`True`.
		sedimentation_on : `bool`, optional
			:obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		sedimentation_flux_type : `str`, optional
			String specifying the method seused to compute the numerical sedimentation flux.
			See :class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`
			for all available options.
		sedimentation_substeps : `int`, optional
			Number of sub-timesteps to perform in order to integrate the sedimentation flux.
			Defaults to 2.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils
			implementing the dynamical core. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. See
			:class:`~tasmania.dynamics.isentropic_prognostic.IsentropicPrognostic`
			and	:class:`~tasmania.dynamics.diagnostics.IsentropicDiagnostics`
			for the physical constants used.
		"""
		# Call parent constructor
		super().__init__(grid, moist_on, fast_parameterizations)

		# Keep track of the input parameters
		self._damp_on                      = damp_on
		self._damp_at_every_stage		   = damp_at_every_stage
		self._smooth_on                    = smooth_on
		self._smooth_at_every_stage		   = smooth_at_every_stage
		self._smooth_moist_on              = smooth_moist_on
		self._smooth_moist_at_every_stage  = smooth_moist_at_every_stage
		self._adiabatic_flow 			   = adiabatic_flow
		self._sedimentation_on             = sedimentation_on
		self._dtype						   = dtype

		# Instantiate the class taking care of the boundary conditions
		self._boundary = HorizontalBoundary.factory(horizontal_boundary_type, grid)

		# Instantiate the class implementing the diagnostic part of the dycore
		self._diagnostic = IsentropicDiagnostics(grid, moist_on, backend)

		# Instantiate the class implementing the prognostic part of the dycore
		self._prognostic = IsentropicPrognostic.factory(
			time_integration_scheme, grid, moist_on, backend, self._diagnostic,
			horizontal_flux_scheme, adiabatic_flow, vertical_flux_scheme,
			sedimentation_on, sedimentation_flux_type, sedimentation_substeps,
			raindrop_fall_velocity_diagnostic, dtype, physical_constants)

		# Instantiate the class in charge of applying vertical damping
		nx, ny, nz = grid.nx, grid.ny, grid.nz
		if damp_on: 
			self._damper = VerticalDamping.factory(damp_type, (nx, ny, nz), grid,
												   damp_depth, damp_max, backend, dtype)

		# Instantiate the classes in charge of applying numerical smoothing
		if smooth_on:
			self._smoother = HorizontalSmoothing.factory(smooth_type, (nx, ny, nz), grid,
														 smooth_damp_depth, smooth_coeff,
														 smooth_coeff_max, backend, dtype)
			if moist_on and smooth_moist_on:
				self._smoother_moist = HorizontalSmoothing.factory(
					smooth_moist_type, (nx, ny, nz), grid, smooth_moist_damp_depth,
					smooth_moist_coeff, smooth_moist_coeff_max, backend, dtype)

		# Set the pointer to the private method implementing each stage
		self._array_call = self._array_call_dry if not moist_on else self._array_call_moist

	@property
	def _input_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		dims_stg_x = (self._grid.x_at_u_locations.dims[0], self._grid.y.dims[0],
					  self._grid.z.dims[0])
		dims_stg_y = (self._grid.x.dims[0], self._grid.y_at_v_locations.dims[0],
					  self._grid.z.dims[0])
		dims_stg_z = (self._grid.x.dims[0], self._grid.y.dims[0],
					  self._grid.z_on_interface_levels.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'air_pressure_on_interface_levels': {'dims': dims_stg_z, 'units': 'Pa'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'x_velocity_at_u_locations': {'dims': dims_stg_x, 'units': 'm s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_stg_y, 'units': 'm s^-1'},
		}

		if self._smooth_moist_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def _tendency_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return_dict = {}

		if self._moist_on:
			return_dict['tendency_of_mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['tendency_of_mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['tendency_of_mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}

		return return_dict

	@property
	def _output_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		dims_stg_x = (self._grid.x_at_u_locations.dims[0], self._grid.y.dims[0],
					  self._grid.z.dims[0])
		dims_stg_y = (self._grid.x.dims[0], self._grid.y_at_v_locations.dims[0],
					  self._grid.z.dims[0])
		dims_stg_z = (self._grid.x.dims[0], self._grid.y.dims[0],
					  self._grid.z_on_interface_levels.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'air_pressure_on_interface_levels': {'dims': dims_stg_z, 'units': 'Pa'},
			'exner_function_on_interface_levels': {'dims': dims_stg_z, 'units': 'm^2 s^-2 K^-1'},
			'height_on_interface_levels': {'dims': dims_stg_z, 'units': 'm'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'x_velocity_at_u_locations': {'dims': dims_stg_x, 'units': 'm s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_stg_y, 'units': 'm s^-1'},
		}

		if self._smooth_moist_on:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def stages(self):
		return self._prognostic.stages

	def array_call(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Perform a stage of the isentropic dynamical core, either dry or moist.
		"""
		return self._array_call(stage, raw_state, raw_tendencies, timestep)

	def _array_call_dry(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Perform a stage of the dry dynamical core, either dry or moist.
		"""
		# If either damping or smoothing is enabled: deep-copy the prognostic model variables
		if (self._damp_on or self._smooth_on) and \
		   (self._smooth_at_every_stage or stage == self.stages-1):
			s_now = np.copy(raw_state['air_isentropic_density'].values[:,:,:,0])
			su_now = np.copy(raw_state['x_momentum_isentropic'].values[:,:,:,0])
			sv_now = np.copy(raw_state['y_momentum_isentropic'].values[:,:,:,0])

		# Perform the prognostic step
		raw_state_new = self._prognostic.step_neglecting_vertical_advection(
			timestep, raw_state, raw_tendencies)

		if self._damp_on:
			# If this is the first call to the entry-point method,
			# set the reference state
			if not hasattr(self, '_s_ref'):
				self._s_ref  = np.copy(raw_state['air_isentropic_density'])
				self._su_ref = np.copy(raw_state['x_momentum_isentropic'])
				self._sv_ref = np.copy(raw_state['y_momentum_isentropic'])

			if self._damp_at_every_stage or stage == self.stages-1:
				# Extract the current prognostic model variables
				s_now  = raw_state['air_isentropic_density']
				su_now = raw_state['x_momentum_isentropic']
				sv_now = raw_state['y_momentum_isentropic']

				# Extract the stepped prognostic model variables
				s_new  = raw_state_new['air_isentropic_density']
				su_new = raw_state_new['x_momentum_isentropic']
				sv_new = raw_state_new['y_momentum_isentropic']

				# Apply vertical damping
				s_new[:, :, :]  = self._damper(timestep, s_now,  s_new, self._s_ref)
				su_new[:, :, :] = self._damper(timestep, su_now, su_new, self._su_ref)
				sv_new[:, :, :] = self._damper(timestep, sv_now, sv_new, self._sv_ref)

		if self._smooth_on and (self._smooth_at_every_stage or stage == self.stages-1):
			if not self._damp_on and not (self._damp_at_every_stage or stage == self.stages-1):
				# Extract the prognostic model variables
				s_new  = raw_state_new['air_isentropic_density']
				su_new = raw_state_new['x_momentum_isentropic']
				sv_new = raw_state_new['y_momentum_isentropic']

			# Apply horizontal smoothing
			s_new[:,:,:] = self._smoother.apply(s_new)
			su_new[:,:,:] = self._smoother.apply(su_new)
			sv_new[:,:,:] = self._smoother.apply(sv_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(s_new, s_now)
			self._boundary.apply(su_new, su_now)
			self._boundary.apply(sv_new, sv_now)

		# Diagnose the velocity components
		state_new.extend(self._diagnostic.get_velocity_components(state_new, state))

		# Diagnose the pressure, the Exner function, the Montgomery potential and the geometric height of the half levels
		p_ = state['air_pressure'] if state['air_pressure'] is not None else state['air_pressure_on_interface_levels']
		state_new.extend(self._diagnostic.get_diagnostic_variables(state_new, p_.values[0,0,0,0]))

		return state_new, diagnostics_out

	def _array_call_moist(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Method advancing the moist isentropic state by a single time step.

		Parameters
		----------
		dt : obj 
			:class:`datetime.timedelta` representing the time step.
		state :obj 
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the current state.
			It should contain the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure or air_pressure_on_interface_levels (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered, optional);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered, optional);
			* mass_fraction_of_precipitation_water_in_air (unstaggered, optional).

		tendencies : `obj`, optional
			:class:`~tasmania.storages.grid_data.GridData` storing tendencies, namely:
			
			* tendency_of_air_potential_temperature (unstaggered);
			* tendency_of_mass_fraction_of_water_vapor_in_air (unstaggered);
			* tendency_of_mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* tendency_of_mass_fraction_of_precipitation_water_in_air (unstaggered).
			
			Default is obj:`None`.
		diagnostics : `obj`, optional 
			:class:`~tasmania.storages.grid_data.GridData` storing diagnostics, namely:
			
			* accumulated_precipitation (unstaggered).

			Default is :obj:`None`.

		Return
		------
		state_new : obj
			:class:`~tasmania.storages.state_isentropic.StateIsentropic` representing the state at the next time level.
			It contains the following variables:

			* air_isentropic_density (unstaggered);
			* x_velocity (:math:`x`-staggered);
			* x_momentum_isentropic (unstaggered);
			* y_velocity (:math:`y`-staggered);
			* y_momentum_isentropic (unstaggered);
			* air_pressure_on_interface_levels (:math:`z`-staggered);
			* exner_function_on_interface_levels (:math:`z`-staggered);
			* montgomery_potential (unstaggered);
			* height_on_interface_levels (:math:`z`-staggered);
			* mass_fraction_of_water_vapor_in_air (unstaggered);
			* mass_fraction_of_cloud_liquid_water_in_air (unstaggered);
			* mass_fraction_of_precipitation_water_in_air (unstaggered);
			* air_density (unstaggered, only if cloud microphysics is switched on);
			* air_temperature (unstaggered, only if cloud microphysics is switched on).

		diagnostics_out : obj
			:class:`~tasmania.storages.grid_data.GridData` collecting output diagnostics, namely:

			* precipitation (unstaggered, only if rain sedimentation is switched on);
			* accumulated_precipitation (unstaggered, only if rain sedimentation is switched on).
		"""
		# Initialize the GridData to return
		time_now = utils.convert_datetime64_to_datetime(state['air_isentropic_density'].coords['time'].values[0])
		diagnostics_out = GridData(time_now + dt, self._grid,
								   precipitation = np.zeros((self._grid.nx, self._grid.ny), dtype = datatype),
								   accumulated_precipitation = np.zeros((self._grid.nx, self._grid.ny), dtype = datatype))

		# Diagnose the isentropic density for each water constituent to build the conservative state
		state.extend_and_update(self._diagnostic.get_water_constituents_isentropic_density(state))

		# If either damping or smoothing is enabled: deep-copy the prognostic model variables
		if self._damp_on or self._smooth_on:
			s_now  = np.copy(state['air_isentropic_density'].values[:,:,:,0])
			su_now  = np.copy(state['x_momentum_isentropic'].values[:,:,:,0])
			sv_now  = np.copy(state['y_momentum_isentropic'].values[:,:,:,0])
			sqv_now = np.copy(state['water_vapor_isentropic_density'].values[:,:,:,0])
			sqc_now = np.copy(state['cloud_liquid_water_isentropic_density'].values[:,:,:,0])
			sqr_now = np.copy(state['precipitation_water_isentropic_density'].values[:,:,:,0])

		# Perform the prognostic step, neglecting the vertical advection
		state_new = self._prognostic.step_neglecting_vertical_advection(dt, state, tendencies = tendencies)

		if self._physics_dynamics_coupling_on:
			# Couple physics with dynamics
			state_new_ = self._prognostic.step_coupling_physics_with_dynamics(dt, state, state_new, tendencies)

			# Update the output state
			state_new.update(state_new_)

		if self._damp_on:
			# If this is the first call to the entry-point method: set the reference state
			if not hasattr(self, '_s_ref'):
				self._s_ref  = s_now
				self._su_ref  = su_now
				self._sv_ref  = sv_now
				self._sqv_ref = sqv_now
				self._sqc_ref = sqc_now
				self._sqr_ref = sqr_now

			# Extract the prognostic model variables
			s_new  = state_new['air_isentropic_density'].values[:,:,:,0]
			su_new  = state_new['x_momentum_isentropic'].values[:,:,:,0]
			sv_new  = state_new['y_momentum_isentropic'].values[:,:,:,0]
			sqv_new = state_new['water_vapor_isentropic_density'].values[:,:,:,0]
			sqc_new = state_new['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
			sqr_new = state_new['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Apply vertical damping
			s_new[:,:,:]  = self._damper.apply(dt, s_now, s_new, self._s_ref)
			su_new[:,:,:]  = self._damper.apply(dt, su_now, su_new, self._su_ref)
			sv_new[:,:,:]  = self._damper.apply(dt, sv_now, sv_new, self._sv_ref)
			sqv_new[:,:,:] = self._damper.apply(dt, sqv_now, sqv_new, self._sqv_ref)
			sqc_new[:,:,:] = self._damper.apply(dt, sqc_now, sqc_new, self._sqc_ref)
			sqr_new[:,:,:] = self._damper.apply(dt, sqr_now, sqr_new, self._sqr_ref)

		if self._smooth_on:
			if not self._damp_on:
				# Extract the dry prognostic model variables
				s_new = state_new['air_isentropic_density'].values[:,:,:,0]
				su_new = state_new['x_momentum_isentropic'].values[:,:,:,0]
				sv_new = state_new['y_momentum_isentropic'].values[:,:,:,0]

			# Apply horizontal smoothing
			s_new[:,:,:] = self._smoother.apply(s_new)
			su_new[:,:,:] = self._smoother.apply(su_new)
			sv_new[:,:,:] = self._smoother.apply(sv_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(s_new, s_now)
			self._boundary.apply(su_new, su_now)
			self._boundary.apply(sv_new, sv_now)

		if self._smooth_moist_on:
			if not self._damp_on:
				# Extract the moist prognostic model variables
				sqv_new = state_new['water_vapor_isentropic_density'].values[:,:,:,0]
				sqc_new = state_new['cloud_liquid_water_isentropic_density'].values[:,:,:,0]
				sqr_new = state_new['precipitation_water_isentropic_density'].values[:,:,:,0]

			# Apply horizontal smoothing
			sqv_new[:,:,:] = self._smoother_moist.apply(sqv_new)
			sqc_new[:,:,:] = self._smoother_moist.apply(sqc_new)
			sqr_new[:,:,:] = self._smoother_moist.apply(sqr_new)

			# Apply horizontal boundary conditions
			self._boundary.apply(sqv_new, sqv_now)
			self._boundary.apply(sqc_new, sqc_now)
			self._boundary.apply(sqr_new, sqr_now)

		# Diagnose the mass fraction of each water constituent, possibly clipping negative values
		state_new.extend(self._diagnostic.get_mass_fraction_of_water_constituents_in_air(state_new)) 

		# Diagnose the velocity components
		state_new.extend(self._diagnostic.get_velocity_components(state_new, state))

		# Diagnose the pressure, the Exner function, the Montgomery potential and the geometric height of the half levels
		p_ = state['air_pressure'] if state['air_pressure'] is not None else state['air_pressure_on_interface_levels']
		state_new.extend(self._diagnostic.get_diagnostic_variables(state_new, p_.values[0,0,0,0]))

		if self.microphysics is not None:
			# Diagnose the density
			state_new.extend(self._diagnostic.get_air_density(state_new))

			# Diagnose the temperature
			state_new.extend(self._diagnostic.get_air_temperature(state_new))

		if self._sedimentation_on:
			qr     = state['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]
			qr_new = state_new['mass_fraction_of_precipitation_water_in_air'].values[:,:,:,0]

			if np.any(qr > 0.) or np.any(qr_new > 0.):
				# Integrate rain sedimentation flux
				state_new_, diagnostics_out_ = self._prognostic.step_integrating_sedimentation_flux(dt, state, 
																									state_new, diagnostics)

				# Update the output state and the output diagnostics
				state_new.update(state_new_)
				diagnostics_out.update(diagnostics_out_)

		return state_new, diagnostics_out
