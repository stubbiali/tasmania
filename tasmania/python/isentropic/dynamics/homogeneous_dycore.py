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
	HomogeneousIsentropicDynamicalCore(DynamicalCore)
"""
import numpy as np

import gridtools as gt
from tasmania.python.isentropic.dynamics.diagnostics import \
	HorizontalVelocity, WaterConstituent
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.isentropic.dynamics.homogeneous_prognostic import \
	HomogeneousIsentropicPrognostic
from tasmania.python.dwarfs.vertical_damping import VerticalDamping

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


# Convenient shortcuts
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


class HomogeneousIsentropicDynamicalCore(DynamicalCore):
	"""
	The three-dimensional (moist) isentropic homogeneous dynamical core.
	Here, *homogeneous* refers to the fact that only the horizontal advection
	is included in the so-called *dynamics*. Any other process (e.g.,
	vertical advection, pressure gradient, Coriolis acceleration) might be
	included in the model only via parameterizations.
	The conservative form of the governing equations is used.
	Diverse numerical schemes to integrate the model forward in time are offered,
	and different types of lateral boundary conditions are supported.
	"""
	def __init__(
		self, grid, time_units='s',
		intermediate_tendencies=None, intermediate_diagnostics=None,
		substeps=0, fast_tendencies=None, fast_diagnostics=None,
		moist=False, time_integration_scheme='forward_euler',
		horizontal_flux_scheme='upwind', horizontal_boundary_type='periodic',
		damp=True, damp_type='rayleigh', damp_depth=15,
		damp_max=0.0002, damp_at_every_stage=True,
		smooth=True, smooth_type='first_order', smooth_damp_depth=10,
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_at_every_stage=True,
		smooth_moist=False, smooth_moist_type='first_order',
		smooth_moist_damp_depth=10, smooth_moist_coeff=.03,
		smooth_moist_coeff_max=.24, smooth_moist_at_every_stage=True,
		backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of :class:`tasmania.GridXYZ`
			or one of its derived classes.
		time_units : `str`, optional
			The time units used within this object. Defaults to 's', i.e., seconds.
		intermediate_tendencies : `obj`, optional
			An instance of either

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			calculating the intermediate physical tendencies.
			Here, *intermediate* refers to the fact that these physical 
			packages are called *before* each stage of the dynamical core 
			to calculate the physical tendencies.
		intermediate_diagnostics : `obj`, optional
			An instance of either

				* :class:`sympl.DiagnosticComponent`,
				* :class:`sympl.DiagnosticComponentComposite`, or
				* :class:`tasmania.DiagnosticComponentComposite`

			retrieving diagnostics at the end of each stage, once the 
			sub-timestepping routine is over.
		substeps : `int`, optional
			Number of sub-steps to perform. Defaults to 0, meaning that no
			sub-stepping technique is implemented.
		fast_tendencies : `obj`, optional
			An instance of either

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			calculating the fast physical tendencies.
			Here, *fast* refers to the fact that these physical packages are
			called *before* each sub-step of any stage of the dynamical core
			to calculate the physical tendencies.
			This parameter is ignored if `substeps` argument is not positive.
		fast_diagnostics : `obj`, optional
			An instance of either

				* :class:`sympl.DiagnosticComponent`,
				* :class:`sympl.DiagnosticComponentComposite`, or
				* :class:`tasmania.DiagnosticComponentComposite`

			retrieving diagnostics at the end of each sub-step of any stage 
			of the dynamical core.
			This parameter is ignored if `substeps` argument is not positive.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			Defaults to :obj:`False`.
		time_integration_scheme : str
			String specifying the time stepping method to implement. 
			See :class:`tasmania.HomogeneousIsentropicPrognostic` 
			for all available options. Defaults to 'forward_euler'.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux to use. 
			See :class:`tasmania.HorizontalHomogeneousIsentropicFlux` 
			for all available options. Defaults to 'upwind'.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions. 
			See :class:`tasmania.HorizontalBoundary` for all available options.
			Defaults to 'periodic'.
		damp : `bool`, optional
			:obj:`True` to enable vertical damping, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		damp_type : `str`, optional
			String specifying the type of vertical damping to apply. 
			See :class:`tasmania.VerticalDamping` for all available options.
			Defaults to 'rayleigh'.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Defaults to 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Defaults to 0.0002.
		damp_at_every_stage : `bool`, optional
			:obj:`True` to carry out the damping at each stage performed by the
			dynamical core, :obj:`False` to carry out the damping only at the end
			of each timestep. Defaults to :obj:`True`.
		smooth : `bool`, optional
			:obj:`True` to enable numerical smoothing, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		smooth_type: `str`, optional
			String specifying the smoothing technique to implement.
			See :class:`tasmania.HorizontalSmoothing` for all available options.
			Defaults to 'first_order'.
		smooth_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region. Defaults to 10.
		smooth_coeff : `float`, optional
			Smoothing coefficient. Defaults to 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. 
			See :class:`tasmania.HorizontalSmoothing` for further details.
			Defaults to 0.24.
		smooth_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing at each stage performed by the
			dynamical core, :obj:`False` to apply numerical smoothing only at the end
			of each timestep. Defaults to :obj:`True`.
		smooth_moist : `bool`, optional
			:obj:`True` to enable numerical smoothing on the water constituents,
			:obj:`False` otherwise. Defaults to :obj:`True`.
		smooth_moist_type: `str`, optional
			String specifying the smoothing technique to apply to the water constituents.
			See :class:`tasmania.HorizontalSmoothing` for all available options.
			Defaults to 'first-order'. 
		smooth_moist_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region for the
			water constituents. Defaults to 10.
		smooth_moist_coeff : `float`, optional
			Smoothing coefficient for the water constituents. Defaults to 0.03.
		smooth_moist_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient for the water constituents.
			See :class:`tasmania.HorizontalSmoothing` for further details. 
			Defaults to 0.24. 
		smooth_moist_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing to the water constituents
			at each stage performed by the dynamical core, :obj:`False` to apply
			numerical smoothing only at the end of each timestep. Defaults to :obj:`True`.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils
			implementing the dynamical core. Defaults to :class:`gridtools.mode.NUMPY`.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		# Keep track of the input parameters
		self._moist							= moist
		self._damp					   	   	= damp
		self._damp_at_every_stage		   	= damp_at_every_stage
		self._smooth					   	= smooth
		self._smooth_at_every_stage		   	= smooth_at_every_stage
		self._smooth_moist			   		= smooth_moist
		self._smooth_moist_at_every_stage  	= smooth_moist_at_every_stage
		self._dtype						   	= dtype

		# Call parent constructor
		super().__init__(
			grid, time_units, intermediate_tendencies, intermediate_diagnostics,
			substeps, fast_tendencies, fast_diagnostics,
		)

		# Instantiate the class taking care of the boundary conditions
		self._boundary = HorizontalBoundary.factory(horizontal_boundary_type, grid)

		# Instantiate the class implementing the prognostic part of the dycore
		self._prognostic = HomogeneousIsentropicPrognostic.factory(
			time_integration_scheme, 'xy', grid, moist, self._boundary,
			horizontal_flux_scheme, substeps, backend, dtype
		)
		self._prognostic.substep_output_properties = self._substep_output_properties

		# Instantiate the class in charge of applying vertical damping
		nx, ny, nz = grid.nx, grid.ny, grid.nz
		if damp:
			self._damper = VerticalDamping.factory(
				damp_type, (nx, ny, nz), grid,
				damp_depth, damp_max, backend, dtype
			)

		# Instantiate the classes in charge of applying numerical smoothing
		if smooth:
			self._smoother = HorizontalSmoothing.factory(
				smooth_type, (nx, ny, nz), grid, smooth_damp_depth,
				smooth_coeff, smooth_coeff_max, backend, dtype
			)
			if moist and smooth_moist:
				self._smoother_moist = HorizontalSmoothing.factory(
					smooth_moist_type, (nx, ny, nz), grid, smooth_moist_damp_depth,
					smooth_moist_coeff, smooth_moist_coeff_max, backend, dtype
				)

		# Instantiate the class in charge of diagnosing the velocity components
		self._velocity_components = HorizontalVelocity(grid, backend, dtype)

		# Instantiate the class in charge of diagnosing the mass fraction and
		# isentropic density of the water constituents
		if moist:
			self._water_constituent = WaterConstituent(grid, backend, dtype)

		# Set the pointer to the private method implementing each stage
		self._array_call = self._array_call_dry if not moist else self._array_call_moist

	@property
	def _input_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		dims_stg_x = (
			self._grid.x_at_u_locations.dims[0], self._grid.y.dims[0], self._grid.z.dims[0]
		)
		dims_stg_y = (
			self._grid.x.dims[0], self._grid.y_at_v_locations.dims[0], self._grid.z.dims[0]
		)

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'x_velocity_at_u_locations': {'dims': dims_stg_x, 'units': 'm s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_stg_y, 'units': 'm s^-1'},
		}

		if self._moist:
			return_dict[mfwv] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfcw] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfpw] = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def _substep_input_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		ftends, fdiags = self._fast_tends, self._fast_diags

		return_dict = {}

		if (
			ftends is not None and 'air_isentropic_density' in ftends.input_properties or
			ftends is not None and 'air_isentropic_density' in ftends.tendency_properties or
			fdiags is not None and 'air_isentropic_density' in fdiags.input_properties
		):
			return_dict['air_isentropic_density'] = \
				{'dims': dims, 'units': 'kg m^-2 K^-1'}

		if (
			ftends is not None and 'x_momentum_isentropic' in ftends.input_properties or
			ftends is not None and 'x_momentum_isentropic' in ftends.tendency_properties or
			fdiags is not None and 'x_momentum_isentropic' in fdiags.input_properties
		):
			return_dict['x_momentum_isentropic'] = \
				{'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'}

		if (
			ftends is not None and 'y_momentum_isentropic' in ftends.input_properties or
			ftends is not None and 'y_momentum_isentropic' in ftends.tendency_properties or
			fdiags is not None and 'y_momentum_isentropic' in fdiags.input_properties
		):
			return_dict['y_momentum_isentropic'] = \
				{'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'}

		if self._moist:
			if (
				ftends is not None and mfwv in ftends.input_properties or
				ftends is not None and mfwv in ftends.tendency_properties or
				fdiags is not None and mfwv in fdiags.input_properties
			):
				return_dict[mfwv] = {'dims': dims, 'units': 'g g^-1'}

			if (
				ftends is not None and mfcw in ftends.input_properties or
				ftends is not None and mfcw in ftends.tendency_properties or
				fdiags is not None and mfcw in fdiags.input_properties
			):
				return_dict[mfcw] = {'dims': dims, 'units': 'g g^-1'}

			if (
				ftends is not None and mfpw in ftends.input_properties or
				ftends is not None and mfpw in ftends.tendency_properties or
				fdiags is not None and mfpw in fdiags.input_properties
			):
				return_dict[mfpw] = {'dims': dims, 'units': 'g g^-1'}

			if (
				ftends is not None and 'precipitation' in ftends.diagnostic_properties
			):
				dims2d = (self._grid.x.dims[0], self._grid.y.dims[0])
				return_dict.update({
					'precipitation': {'dims': dims2d, 'units': 'mm hr^-1'},
					'accumulated_precipitation': {'dims': dims2d, 'units': 'mm'},
				})

		return return_dict

	@property
	def _tendency_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		if self._moist:
			return_dict['mass_fraction_of_water_vapor_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['mass_fraction_of_cloud_liquid_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict['mass_fraction_of_precipitation_water_in_air'] = \
				{'dims': dims, 'units': 'g g^-1 s^-1'}

		return return_dict

	@property
	def _substep_tendency_properties(self):
		return self._tendency_properties

	@property
	def _output_properties(self):
		dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])
		dims_stg_x = (
			self._grid.x_at_u_locations.dims[0], self._grid.y.dims[0], self._grid.z.dims[0]
		)
		dims_stg_y = (
			self._grid.x.dims[0], self._grid.y_at_v_locations.dims[0], self._grid.z.dims[0]
		)

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'x_velocity_at_u_locations': {'dims': dims_stg_x, 'units': 'm s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_stg_y, 'units': 'm s^-1'},
		}

		if self._moist:
			return_dict[mfwv] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfcw] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfpw] = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def _substep_output_properties(self):
		if not hasattr(self, '__substep_output_properties'):
			dims = (self._grid.x.dims[0], self._grid.y.dims[0], self._grid.z.dims[0])

			self.__substep_output_properties = {}

			if 'air_isentropic_density' in self._substep_input_properties:
				self.__substep_output_properties['air_isentropic_density'] = \
					{'dims': dims, 'units': 'kg m^-2 K^-1'}

			if 'x_momentum_isentropic' in self._substep_input_properties:
				self.__substep_output_properties['x_momentum_isentropic'] = \
					{'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'}

			if 'y_momentum_isentropic' in self._substep_input_properties:
				self.__substep_output_properties['y_momentum_isentropic'] = \
					{'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'}

			if self._moist:
				if mfwv in self._substep_input_properties:
					self.__substep_output_properties[mfwv] = {'dims': dims, 'units': 'g g^-1'}

				if mfcw in self._substep_input_properties:
					self.__substep_output_properties[mfcw] = {'dims': dims, 'units': 'g g^-1'}

				if mfpw in self._substep_input_properties:
					self.__substep_output_properties[mfpw] = {'dims': dims, 'units': 'g g^-1'}

				if 'precipitation' in self._substep_input_properties:
					dims2d = (self._grid.x.dims[0], self._grid.y.dims[0])
					self.__substep_output_properties['accumulated_precipitation'] = \
						{'dims': dims2d, 'units': 'mm'}

		return self.__substep_output_properties

	@property
	def stages(self):
		return self._prognostic.stages

	@property
	def substep_fractions(self):
		return self._prognostic.substep_fractions

	def array_call(self, stage, raw_state, raw_tendencies, timestep):
		return self._array_call(stage, raw_state, raw_tendencies, timestep)

	def _array_call_dry(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Perform a stage of the dry dynamical core.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# Perform the prognostic step
		raw_state_new = self._prognostic.stage_call(
			stage, timestep, raw_state, raw_tendencies
		)

		damped = False
		if self._damp:
			# If this is the first call to the entry-point method,
			# set the reference state...
			if not hasattr(self, '_s_ref'):
				self._s_ref  = np.copy(raw_state['air_isentropic_density'])
				self._su_ref = np.copy(raw_state['x_momentum_isentropic'])
				self._sv_ref = np.copy(raw_state['y_momentum_isentropic'])

			# ...and allocate memory to store damped fields
			if not hasattr(self, '_s_damped'):
				self._s_damped	= np.zeros((nx, ny, nz), dtype=dtype)
				self._su_damped = np.zeros((nx, ny, nz), dtype=dtype)
				self._sv_damped = np.zeros((nx, ny, nz), dtype=dtype)

			if self._damp_at_every_stage or stage == self.stages-1:
				damped = True

				# Extract the current prognostic model variables
				s_now_	= raw_state['air_isentropic_density']
				su_now_ = raw_state['x_momentum_isentropic']
				sv_now_ = raw_state['y_momentum_isentropic']

				# Extract the stepped prognostic model variables
				s_new_	= raw_state_new['air_isentropic_density']
				su_new_ = raw_state_new['x_momentum_isentropic']
				sv_new_ = raw_state_new['y_momentum_isentropic']

				# Apply vertical damping
				self._damper(timestep, s_now_,	s_new_,  self._s_ref,  self._s_damped)
				self._damper(timestep, su_now_, su_new_, self._su_ref, self._su_damped)
				self._damper(timestep, sv_now_, sv_new_, self._sv_ref, self._sv_damped)

		# Properly set pointers to current solution
		s_new  = self._s_damped if damped else raw_state_new['air_isentropic_density']
		su_new = self._su_damped if damped else raw_state_new['x_momentum_isentropic']
		sv_new = self._sv_damped if damped else raw_state_new['y_momentum_isentropic']

		smoothed = False
		if self._smooth:
			# If this is the first call to the entry-point method,
			# allocate memory to store smoothed fields
			if not hasattr(self, '_s_smoothed'):
				self._s_smoothed  = np.zeros((nx, ny, nz), dtype=dtype)
				self._su_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
				self._sv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

			if self._smooth_at_every_stage or stage == self.stages-1:
				smoothed = True

				# Apply horizontal smoothing
				self._smoother(s_new,  self._s_smoothed)
				self._smoother(su_new, self._su_smoothed)
				self._smoother(sv_new, self._sv_smoothed)

				# Apply horizontal boundary conditions
				self._boundary.enforce(self._s_smoothed,  s_new)
				self._boundary.enforce(self._su_smoothed, su_new)
				self._boundary.enforce(self._sv_smoothed, sv_new)

		# Properly set pointers to output solution
		s_out  = self._s_smoothed if smoothed else s_new
		su_out = self._su_smoothed if smoothed else su_new
		sv_out = self._sv_smoothed if smoothed else sv_new

		# Diagnose the velocity components
		u_out, v_out = self._velocity_components.get_velocity_components(s_out, su_out, sv_out)
		self._boundary.set_outermost_layers_x(u_out, raw_state['x_velocity_at_u_locations'])
		self._boundary.set_outermost_layers_y(v_out, raw_state['y_velocity_at_v_locations'])

		# Instantiate the output state
		raw_state_out = {
			'time': raw_state_new['time'],
			'air_isentropic_density': s_out,
			'x_momentum_isentropic': su_out,
			'x_velocity_at_u_locations': u_out,
			'y_momentum_isentropic': sv_out,
			'y_velocity_at_v_locations': v_out,
		}

		return raw_state_out

	def _array_call_moist(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Perform a stage of the moist dynamical core.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype

		# Allocate memory to store the isentropic density of all water constituents
		if not hasattr(self, '_sqv_now'):
			self._sqv_now = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqc_now = np.zeros((nx, ny, nz), dtype=dtype)
			self._sqr_now = np.zeros((nx, ny, nz), dtype=dtype)

		# Diagnosed the isentropic density of all water constituents
		s_now  = raw_state['air_isentropic_density']
		qv_now = raw_state['mass_fraction_of_water_vapor_in_air']
		qc_now = raw_state['mass_fraction_of_cloud_liquid_water_in_air']
		qr_now = raw_state['mass_fraction_of_precipitation_water_in_air']
		self._water_constituent.get_density_of_water_constituent(s_now, qv_now, self._sqv_now)
		self._water_constituent.get_density_of_water_constituent(s_now, qc_now, self._sqc_now)
		self._water_constituent.get_density_of_water_constituent(s_now, qr_now, self._sqr_now)
		raw_state['isentropic_density_of_water_vapor']		   = self._sqv_now
		raw_state['isentropic_density_of_cloud_liquid_water']  = self._sqc_now
		raw_state['isentropic_density_of_precipitation_water'] = self._sqr_now

		# Perform the prognostic step
		raw_state_new = self._prognostic.stage_call(
			stage, timestep, raw_state, raw_tendencies
		)

		damped = False
		if self._damp:
			# If this is the first call to the entry-point method,
			# set the reference state...
			if not hasattr(self, '_s_ref'):
				self._s_ref   = np.copy(raw_state['air_isentropic_density'])
				self._su_ref  = np.copy(raw_state['x_momentum_isentropic'])
				self._sv_ref  = np.copy(raw_state['y_momentum_isentropic'])
				self._sqv_ref = np.copy(raw_state['isentropic_density_of_water_vapor'])
				self._sqc_ref = np.copy(raw_state['isentropic_density_of_cloud_liquid_water'])
				self._sqr_ref = np.copy(raw_state['isentropic_density_of_precipitation_water'])

			# ...and allocate memory to store damped fields
			if not hasattr(self, '_s_damped'):
				self._s_damped	 = np.zeros((nx, ny, nz), dtype=dtype)
				self._su_damped  = np.zeros((nx, ny, nz), dtype=dtype)
				self._sv_damped  = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqv_damped = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqc_damped = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqr_damped = np.zeros((nx, ny, nz), dtype=dtype)

			if self._damp_at_every_stage or stage == self.stages-1:
				damped = True

				# Extract the current prognostic model variables
				s_now_	 = raw_state['air_isentropic_density']
				su_now_  = raw_state['x_momentum_isentropic']
				sv_now_  = raw_state['y_momentum_isentropic']
				sqv_now_ = raw_state['isentropic_density_of_water_vapor']
				sqc_now_ = raw_state['isentropic_density_of_cloud_liquid_water']
				sqr_now_ = raw_state['isentropic_density_of_precipitation_water']

				# Extract the stepped prognostic model variables
				s_new_	 = raw_state_new['air_isentropic_density']
				su_new_  = raw_state_new['x_momentum_isentropic']
				sv_new_  = raw_state_new['y_momentum_isentropic']
				sqv_new_ = raw_state_new['isentropic_density_of_water_vapor']
				sqc_new_ = raw_state_new['isentropic_density_of_cloud_liquid_water']
				sqr_new_ = raw_state_new['isentropic_density_of_precipitation_water']

				# Apply vertical damping
				self._damper(timestep, s_now_,	 s_new_,   self._s_ref,   self._s_damped)
				self._damper(timestep, su_now_,  su_new_,  self._su_ref,  self._su_damped)
				self._damper(timestep, sv_now_,  sv_new_,  self._sv_ref,  self._sv_damped)
				self._damper(timestep, sqv_now_, sqv_new_, self._sqv_ref, self._sqv_damped)
				self._damper(timestep, sqc_now_, sqc_new_, self._sqc_ref, self._sqc_damped)
				self._damper(timestep, sqr_now_, sqr_new_, self._sqr_ref, self._sqr_damped)

		# Properly set pointers to current solution
		s_new	= self._s_damped if damped else raw_state_new['air_isentropic_density']
		su_new	= self._su_damped if damped else raw_state_new['x_momentum_isentropic']
		sv_new	= self._sv_damped if damped else raw_state_new['y_momentum_isentropic']
		sqv_new = self._sqv_damped if damped else \
				  raw_state_new['isentropic_density_of_water_vapor']
		sqc_new = self._sqc_damped if damped else \
				  raw_state_new['isentropic_density_of_cloud_liquid_water']
		sqr_new = self._sqr_damped if damped else \
				  raw_state_new['isentropic_density_of_precipitation_water']

		smoothed = False
		if self._smooth:
			# If this is the first call to the entry-point method,
			# allocate memory to store smoothed fields
			if not hasattr(self, '_s_smoothed'):
				self._s_smoothed  = np.zeros((nx, ny, nz), dtype=dtype)
				self._su_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
				self._sv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

			if self._smooth_at_every_stage or stage == self.stages-1:
				smoothed = True

				# Apply horizontal smoothing
				self._smoother(s_new,  self._s_smoothed)
				self._smoother(su_new, self._su_smoothed)
				self._smoother(sv_new, self._sv_smoothed)

				# Apply horizontal boundary conditions
				self._boundary.enforce(self._s_smoothed,  s_new)
				self._boundary.enforce(self._su_smoothed, su_new)
				self._boundary.enforce(self._sv_smoothed, sv_new)

		# Properly set pointers to output solution
		s_out  = self._s_smoothed if smoothed else s_new
		su_out = self._su_smoothed if smoothed else su_new
		sv_out = self._sv_smoothed if smoothed else sv_new

		smoothed_moist = False
		if self._smooth_moist:
			# If this is the first call to the entry-point method,
			# allocate memory to store smoothed fields
			if not hasattr(self, '_sqv_smoothed'):
				self._sqv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqc_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
				self._sqr_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

			if self._smooth_moist_at_every_stage or stage == self.stages-1:
				smoothed_moist = True

				# Apply horizontal smoothing
				self._smoother(sqv_new, self._sqv_smoothed)
				self._smoother(sqc_new, self._sqc_smoothed)
				self._smoother(sqr_new, self._sqr_smoothed)

				# Apply horizontal boundary conditions
				self._boundary.enforce(self._sqv_smoothed, sqv_new)
				self._boundary.enforce(self._sqc_smoothed, sqc_new)
				self._boundary.enforce(self._sqr_smoothed, sqr_new)

		# Properly set pointers to output solution
		sqv_out = self._sqv_smoothed if smoothed_moist else sqv_new
		sqc_out = self._sqc_smoothed if smoothed_moist else sqc_new
		sqr_out = self._sqr_smoothed if smoothed_moist else sqr_new

		# Diagnose the velocity components
		u_out, v_out = self._velocity_components.get_velocity_components(s_out, su_out, sv_out)
		self._boundary.set_outermost_layers_x(u_out, raw_state['x_velocity_at_u_locations'])
		self._boundary.set_outermost_layers_y(v_out, raw_state['y_velocity_at_v_locations'])

		# Allocate memory to store the isentropic density of all water constituents
		if not hasattr(self, '_qv_out'):
			self._qv_out = np.zeros((nx, ny, nz), dtype=dtype)
			self._qc_out = np.zeros((nx, ny, nz), dtype=dtype)
			self._qr_out = np.zeros((nx, ny, nz), dtype=dtype)

		# Diagnose the mass fraction of all water constituents
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			s_out, sqv_out, self._qv_out, clipping=True)
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			s_out, sqc_out, self._qc_out, clipping=True)
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			s_out, sqr_out, self._qr_out, clipping=True)

		# Instantiate the output state
		raw_state_out = {
			'time': raw_state_new['time'],
			'air_isentropic_density': s_out,
			'mass_fraction_of_water_vapor_in_air': self._qv_out,
			'mass_fraction_of_cloud_liquid_water_in_air': self._qc_out,
			'mass_fraction_of_precipitation_water_in_air': self._qr_out,
			'x_momentum_isentropic': su_out,
			'x_velocity_at_u_locations': u_out,
			'y_momentum_isentropic': sv_out,
			'y_velocity_at_v_locations': v_out,
		}

		return raw_state_out

	def substep_array_call(
		self, stage, substep, raw_state, raw_stage_state, raw_tmp_state,
		raw_tendencies, timestep
	):
		# Merely perform the sub-step
		return self._prognostic.substep_call(
			stage, substep, timestep, raw_state, raw_stage_state,
			raw_tmp_state, raw_tendencies
		)
