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
	NGIsentropicDynamicalCore(DynamicalCore)
"""
import numpy as np

import gridtools as gt
from tasmania.python.dwarfs.diagnostics import \
	HorizontalVelocity, WaterConstituent
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.isentropic.dynamics.ng_prognostic import NGIsentropicPrognostic
from tasmania.python.utils.data_utils import make_dataarray_3d

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


# convenient shortcuts
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


class NGIsentropicDynamicalCore(DynamicalCore):
	"""
	The three-dimensional isentropic dynamical core. Note that only
	the pressure gradient is included in the so-called *dynamics*. Any other
	large-scale process (e.g., vertical advection, Coriolis acceleration) might
	be included in the model only via physical parameterizations.
	The conservative form of the governing equations is used.
	"""
	def __init__(
		self, domain, intermediate_tendencies=None, intermediate_diagnostics=None,
		substeps=0, fast_tendencies=None, fast_diagnostics=None,
		time_integration_scheme='forward_euler_si',
		horizontal_flux_scheme='upwind', time_integration_properties=None,
		damp=True, damp_at_every_stage=True,
		damp_type='rayleigh', damp_depth=15, damp_max=0.0002,
		smooth=True, smooth_at_every_stage=True, smooth_type='first_order',
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_damp_depth=10,
		tracers=None, smooth_tracer=False, smooth_tracer_at_every_stage=True,
		smooth_tracer_type='first_order', smooth_tracer_coeff=.03,
		smooth_tracer_coeff_max=.24, smooth_tracer_damp_depth=10,
		backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
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
		time_integration_scheme : str
			String specifying the time stepping method to implement. 
			See :class:`tasmania.IsentropicMinimalPrognostic`
			for all available options. Defaults to 'forward_euler'.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux to use. 
			See :class:`tasmania.HorizontalIsentropicMinimalFlux`
			for all available options. Defaults to 'upwind'.
		time_integration_properties : dict
			Additional properties to be passed to the constructor of
			:class:`~tasmania.python.isentropic.dynamics.IsentropicPrognostic`
			as keyword arguments.
		damp : `bool`, optional
			:obj:`True` to enable vertical damping, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		damp_at_every_stage : `bool`, optional
			:obj:`True` to carry out the damping at each stage of the multi-stage
			time-integrator, :obj:`False` to carry out the damping only at the end
			of each timestep. Defaults to :obj:`True`.
		damp_type : `str`, optional
			String specifying the vertical damping scheme to implement.
			See :class:`tasmania.VerticalDamping` for all available options.
			Defaults to 'rayleigh'.
		damp_depth : `int`, optional
			Number of vertical layers in the damping region. Defaults to 15.
		damp_max : `float`, optional
			Maximum value for the damping coefficient. Defaults to 0.0002.
		smooth : `bool`, optional
			:obj:`True` to enable horizontal numerical smoothing, :obj:`False` otherwise.
			Defaults to :obj:`True`.
		smooth_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing at each stage of the time-
			integrator, :obj:`False` to apply numerical smoothing only at the end
			of each timestep. Defaults to :obj:`True`.
		smooth_type: `str`, optional
			String specifying the smoothing technique to implement.
			See :class:`tasmania.HorizontalSmoothing` for all available options.
			Defaults to 'first_order'.
		smooth_coeff : `float`, optional
			Smoothing coefficient. Defaults to 0.03.
		smooth_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient. 
			See :class:`tasmania.HorizontalSmoothing` for further details.
			Defaults to 0.24.
		smooth_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region. Defaults to 10.
		tracers : 'ordered dict', optional
			(Ordered) dictionary whose keys are strings denoting the tracers
			which should be included in the model, and whose values are
			dictionaries specifying fundamental properties
			('units', 'stencil_symbol') for those tracers.
		smooth_tracer : `bool`, optional
			:obj:`True` to enable horizontal numerical smoothing on the water constituents,
			:obj:`False` otherwise. Defaults to :obj:`True`.
		smooth_tracer_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing on the water constituents
			at each stage of the time-integrator, :obj:`False` to apply numerical
			smoothing only at the end of each timestep. Defaults to :obj:`True`.
		smooth_tracer_type: `str`, optional
			String specifying the smoothing technique to apply on the water constituents.
			See :class:`tasmania.HorizontalSmoothing` for all available options.
			Defaults to 'first-order'. 
		smooth_tracer_coeff : `float`, optional
			Smoothing coefficient for the water constituents. Defaults to 0.03.
		smooth_tracer_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient for the water constituents.
			See :class:`tasmania.HorizontalSmoothing` for further details. 
			Defaults to 0.24. 
		smooth_tracer_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region for the
			water constituents. Defaults to 10.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		"""
		#
		# input parameters
		#
		self._damp					   	   	= damp
		self._damp_at_every_stage		   	= damp_at_every_stage
		self._smooth					   	= smooth
		self._smooth_at_every_stage		   	= smooth_at_every_stage
		self._tracers						= {} if tracers is None else tracers
		self._moist 						= len(self._tracers) > 0
		self._smooth_tracer			   		= smooth_tracer
		self._smooth_tracer_at_every_stage  = smooth_tracer_at_every_stage
		self._dtype						   	= dtype

		#
		# parent constructor
		#
		super().__init__(
			domain, 'numerical', 's',
			intermediate_tendencies, intermediate_diagnostics,
			substeps, fast_tendencies, fast_diagnostics, dtype
		)
		hb = self.horizontal_boundary

		#
		# prognostic
		#
		kwargs = {} if time_integration_properties is None else time_integration_properties
		self._prognostic = NGIsentropicPrognostic.factory(
			time_integration_scheme, horizontal_flux_scheme, self.grid,
			self.horizontal_boundary, self._tracers, backend, dtype, **kwargs
		)

		#
		# vertical damping
		#
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		if damp:
			self._damper = VerticalDamping.factory(
				damp_type, (nx, ny, nz), self.grid,	damp_depth, damp_max,
				time_units='s', backend=backend, dtype=dtype
			)

		#
		# numerical smoothing
		#
		if smooth:
			self._smoother = HorizontalSmoothing.factory(
				smooth_type, (nx, ny, nz), smooth_coeff, smooth_coeff_max,
				smooth_damp_depth, hb.nb, backend, dtype
			)
			if len(self._tracers) > 0 and smooth_tracer:
				self._smoother_tracer = HorizontalSmoothing.factory(
					smooth_tracer_type, (nx, ny, nz), smooth_tracer_coeff,
					smooth_tracer_coeff_max, smooth_tracer_damp_depth, hb.nb, backend, dtype
				)

		#
		# diagnostics
		#
		self._velocity_components = HorizontalVelocity(
			self.grid, staggering=True, backend=backend, dtype=dtype
		)
		if self._moist:
			self._water_constituent = WaterConstituent(self.grid, backend, dtype)

		#
		# temporary and output arrays
		#
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

		if damp:
			self._s_damped  = np.zeros((nx, ny, nz), dtype=dtype)
			self._su_damped = np.zeros((nx, ny, nz), dtype=dtype)
			self._sv_damped = np.zeros((nx, ny, nz), dtype=dtype)
			# self._q_damped = {
			#   tracer: np.zeros((nx, ny, nz), dtype=dtype)
			#   for tracer in self._tracers
			# }

		if smooth:
			self._s_smoothed  = np.zeros((nx, ny, nz), dtype=dtype)
			self._su_smoothed = np.zeros((nx, ny, nz), dtype=dtype)
			self._sv_smoothed = np.zeros((nx, ny, nz), dtype=dtype)

		if smooth_tracer:
			self._q_smoothed = {
				tracer: np.zeros((nx, ny, nz), dtype=dtype)
				for tracer in self._tracers
			}

		self._u_out = np.zeros((nx+1, ny, nz), dtype=dtype)
		self._v_out = np.zeros((nx, ny+1, nz), dtype=dtype)

		self._sq_now = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._tracers
		}

		self._q_new = {
			tracer: np.zeros((nx, ny, nz), dtype=dtype)
			for tracer in self._tracers
		}

	@property
	def _input_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_stg_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])
		dims_stg_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'montgomery_potential': {'dims': dims, 'units': 'm^2 s^-2'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'x_velocity_at_u_locations': {'dims': dims_stg_x, 'units': 'm s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_stg_y, 'units': 'm s^-1'},
		}

		for tracer, props in self._tracers.items():
			return_dict[tracer] = {'dims': dims, 'units': props['units']}

		return return_dict

	@property
	def _substep_input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
		ftends, fdiags = self._fast_tends, self._fast_diags

		prognostics = {
			'air_isentropic_density': 'kg m^-2 K^-1',
			'x_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
			'y_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
		}
		prognostics.update({
			tracer: self._tracers[tracer]['units'] for tracer in self._tracers
		})

		return_dict = {}

		for name, units in prognostics.items():
			if (
				ftends is not None and name in ftends.input_properties or
				ftends is not None and name in ftends.tendency_properties or
				fdiags is not None and name in fdiags.input_properties
			):
				return_dict[name] = {'dims': dims, 'units': units}

		if (
			ftends is not None and 'precipitation' in ftends.diagnostic_properties
		):
			dims2d = (self.grid.x.dims[0], self.grid.y.dims[0])
			return_dict.update({
				'precipitation': {'dims': dims2d, 'units': 'mm hr^-1'},
				'accumulated_precipitation': {'dims': dims2d, 'units': 'mm'},
			})

		return return_dict

	@property
	def _tendency_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1 s^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-2'},
		}

		for tracer, props in self._tracers.items():
			return_dict[tracer] = {'dims': dims, 'units': props['units'] + ' s^-1'}

		return return_dict

	@property
	def _substep_tendency_properties(self):
		return self._tendency_properties

	@property
	def _output_properties(self):
		g = self.grid
		dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
		dims_stg_x = (g.x_at_u_locations.dims[0], g.y.dims[0], g.z.dims[0])
		dims_stg_y = (g.x.dims[0], g.y_at_v_locations.dims[0], g.z.dims[0])

		return_dict = {
			'air_isentropic_density': {'dims': dims, 'units': 'kg m^-2 K^-1'},
			'x_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'x_velocity_at_u_locations': {'dims': dims_stg_x, 'units': 'm s^-1'},
			'y_momentum_isentropic': {'dims': dims, 'units': 'kg m^-1 K^-1 s^-1'},
			'y_velocity_at_v_locations': {'dims': dims_stg_y, 'units': 'm s^-1'},
		}

		for tracer, props in self._tracers.items():
			return_dict[tracer] = {'dims': dims, 'units': props['units']}

		return return_dict

	@property
	def _substep_output_properties(self):
		if not hasattr(self, '__substep_output_properties'):
			dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

			prognostics = {
				'air_isentropic_density': 'kg m^-2 K^-1',
				'x_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
				'y_momentum_isentropic': 'kg m^-1 K^-1 s^-1',
			}
			for tracer, props in self._tracers.items():
				prognostics[tracer] = props['units']

			self.__substep_output_properties = {
				name: {'dims': dims, 'units': units}
				for name, units in prognostics.items()
				if name in self._substep_input_properties
			}

			if 'precipitation' in self._substep_input_properties:
				dims2d = (self.grid.x.dims[0], self.grid.y.dims[0])
				self.__substep_output_properties['accumulated_precipitation'] = \
					{'dims': dims2d, 'units': 'mm'}

		return self.__substep_output_properties

	@property
	def stages(self):
		return self._prognostic.stages

	@property
	def substep_fractions(self):
		return self._prognostic.substep_fractions

	def _allocate_output_state(self):
		"""
		Allocate memory only for the prognostic fields.
		"""
		g = self.grid
		nx, ny, nz = g.nx, g.ny, g.nz
		dtype = self._dtype

		out_state = {}

		names = [
			'air_isentropic_density',
			'x_velocity_at_u_locations',
			'x_momentum_isentropic',
			'y_velocity_at_v_locations',
			'y_momentum_isentropic',
		]
		for tracer in self._tracers:
			names.append(tracer)

		for name in names:
			dims = self.output_properties[name]['dims']
			units = self.output_properties[name]['units']

			shape = (
				nx+1 if 'at_u_locations' in dims[0] else nx,
				ny+1 if 'at_v_locations' in dims[1] else ny,
				nz+1 if 'on_interface_levels' in dims[2] else nz,
			)

			out_state[name] = make_dataarray_3d(
				np.zeros(shape, dtype=dtype), g, units, name=name
			)

		return out_state

	def array_call(self, stage, raw_state, raw_tendencies, timestep):
		# shortcuts
		hb = self.horizontal_boundary
		out_properties = self.output_properties

		if self._damp and stage == 0:
			# set the reference state
			try:
				ref_state = hb.reference_state
				self._s_ref  = \
					ref_state['air_isentropic_density'].to_units('kg m^-2 K^-1').values
				self._su_ref = \
					ref_state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values
				self._sv_ref = \
					ref_state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values
				# self._q_ref  = {
				#    tracer: ref_state[tracer].to_units(props['units']).values
				#    for tracer, props in self._tracers
				# }
			except KeyError:
				raise RuntimeError(
					"Reference state not set in the object handling the horizontal "
					"boundary conditions, but needed by the wave absorber."
				)

			# save the current solution
			self._s_now  = raw_state['air_isentropic_density']
			self._su_now = raw_state['x_momentum_isentropic']
			self._sv_now = raw_state['y_momentum_isentropic']
			# self._q_now  = {tracer: raw_state[tracer] for tracer in self._tracers}

		# diagnose the isentropic density of all water constituents
		s_now  = raw_state['air_isentropic_density']
		for tracer in self._tracers:
			q_now = raw_state[tracer]
			self._water_constituent.get_density_of_water_constituent(
				s_now, q_now, self._sq_now[tracer]
			)
			raw_state['s_' + tracer] = self._sq_now[tracer]

		# perform the prognostic step
		raw_state_new = self._prognostic.stage_call(
			stage, timestep, raw_state, raw_tendencies
		)

		# diagnose the mass fraction of all water constituents
		for tracer in self._tracers:
			self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
				raw_state_new['air_isentropic_density'], raw_state_new['s_' + tracer],
				self._q_new[tracer], clipping=True
			)
			raw_state_new[tracer] = self._q_new[tracer]

		# apply the lateral boundary conditions
		hb.dmn_enforce_raw(raw_state_new, out_properties)

		damped = False
		if self._damp and (self._damp_at_every_stage or stage == self.stages-1):
			damped = True

			# extract the stepped prognostic model variables
			s_new  = raw_state_new['air_isentropic_density']
			su_new = raw_state_new['x_momentum_isentropic']
			sv_new = raw_state_new['y_momentum_isentropic']

			# apply vertical damping
			self._damper(timestep, self._s_now , s_new , self._s_ref , self._s_damped )
			self._damper(timestep, self._su_now, su_new, self._su_ref, self._su_damped)
			self._damper(timestep, self._sv_now, sv_new, self._sv_ref, self._sv_damped)
			# for tracer in self._tracers:
			#     self._damper(
			#         timestep, self._q_now[tracer], self._q_new[tracer],
			#         self._q_ref[tracer], self._q_damped[tracer]
			#     )

		# properly set pointers to current solution
		s_new  = self._s_damped if damped else raw_state_new['air_isentropic_density']
		su_new = self._su_damped if damped else raw_state_new['x_momentum_isentropic']
		sv_new = self._sv_damped if damped else raw_state_new['y_momentum_isentropic']
		q_new  = {tracer: self._q_new[tracer] for tracer in self._tracers}
		# q_new = {
		#     tracer: self._q_damped[tracer] if damped else self._q_new[tracer]
		#     for tracer in self._tracers
		# }

		smoothed = False
		if self._smooth and (self._smooth_at_every_stage or stage == self.stages-1):
			smoothed = True

			# apply horizontal smoothing
			self._smoother(s_new , self._s_smoothed )
			self._smoother(su_new, self._su_smoothed)
			self._smoother(sv_new, self._sv_smoothed)

			# apply horizontal boundary conditions
			raw_state_smoothed = {
				'time': raw_state_new['time'],
				'air_isentropic_density': self._s_smoothed,
				'x_momentum_isentropic': self._su_smoothed,
				'y_momentum_isentropic': self._sv_smoothed,
			}
			hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

		# properly set pointers to output solution
		s_out  = self._s_smoothed if smoothed else s_new
		su_out = self._su_smoothed if smoothed else su_new
		sv_out = self._sv_smoothed if smoothed else sv_new

		smoothed_tracer = False
		if self._smooth_tracer and (self._smooth_tracer_at_every_stage or stage == self.stages-1):
			smoothed_tracer = True

			for tracer in self._tracers:
				# apply horizontal smoothing
				self._smoother_tracer(q_new[tracer], self._q_smoothed[tracer])

			# apply horizontal boundary conditions
			raw_state_smoothed = {'time': raw_state_new['time']}
			raw_state_smoothed.update({
				tracer: self._q_smoothed[tracer] for tracer in self._tracers
			})
			hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

		# properly set pointers to output solution
		q_out = {
			tracer: self._q_smoothed[tracer] if smoothed_tracer else q_new[tracer]
			for tracer in self._tracers
		}

		# diagnose the velocity components
		self._velocity_components.get_velocity_components(
			s_out, su_out, sv_out, self._u_out, self._v_out
		)
		hb.dmn_set_outermost_layers_x(
			self._u_out, field_name='x_velocity_at_u_locations',
			field_units=out_properties['x_velocity_at_u_locations']['units'],
			time=raw_state_new['time']
		)
		hb.dmn_set_outermost_layers_y(
			self._v_out, field_name='y_velocity_at_v_locations',
			field_units=out_properties['y_velocity_at_v_locations']['units'],
			time=raw_state_new['time']
		)

		# instantiate the output state
		raw_state_out = {
			'time': raw_state_new['time'],
			'air_isentropic_density': s_out,
			'x_momentum_isentropic': su_out,
			'x_velocity_at_u_locations': self._u_out,
			'y_momentum_isentropic': sv_out,
			'y_velocity_at_v_locations': self._v_out,
		}
		raw_state_out.update({tracer: q_out[tracer] for tracer in self._tracers})

		return raw_state_out

	def substep_array_call(
		self, stage, substep, raw_state, raw_stage_state, raw_tmp_state,
		raw_tendencies, timestep
	):
		raise NotImplementedError()
