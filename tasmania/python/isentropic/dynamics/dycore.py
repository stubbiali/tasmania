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
	IsentropicDynamicalCore(DynamicalCore)
"""
import numpy as np

import gridtools as gt
from tasmania.python.dwarfs.diagnostics import \
	HorizontalVelocity, WaterConstituent
from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic
from tasmania.python.utils.data_utils import make_dataarray_3d
from tasmania.python.utils.storage_utils import get_storage_descriptor

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


# convenient shortcuts
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


class IsentropicDynamicalCore(DynamicalCore):
	"""
	The three-dimensional (moist) isentropic dynamical core. Note that only
	the pressure gradient is included in the so-called *dynamics*. Any other
	large-scale process (e.g., vertical advection, Coriolis acceleration) might
	be included in the model only via physical parameterizations.
	The conservative form of the governing equations is used.
	"""
	def __init__(
		self, domain, intermediate_tendencies=None, intermediate_diagnostics=None,
		substeps=0, fast_tendencies=None, fast_diagnostics=None,
		moist=False, time_integration_scheme='forward_euler_si',
		horizontal_flux_scheme='upwind', time_integration_properties=None,
		damp=True, damp_at_every_stage=True,
		damp_type='rayleigh', damp_depth=15, damp_max=0.0002,
		smooth=True, smooth_at_every_stage=True, smooth_type='first_order',
		smooth_coeff=.03, smooth_coeff_max=.24, smooth_damp_depth=10,
		smooth_moist=False, smooth_moist_at_every_stage=True, smooth_moist_type='first_order',
		smooth_moist_coeff=.03, smooth_moist_coeff_max=.24, smooth_moist_damp_depth=10,
		*, backend='numpy', backend_opts=None, build_info=None, dtype=datatype,
		exec_info=None, halo=None, rebuild=False, **kwargs
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
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
			Defaults to :obj:`False`.
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
		smooth_moist : `bool`, optional
			:obj:`True` to enable horizontal numerical smoothing on the water constituents,
			:obj:`False` otherwise. Defaults to :obj:`True`.
		smooth_moist_at_every_stage : `bool`, optional
			:obj:`True` to apply numerical smoothing on the water constituents
			at each stage of the time-integrator, :obj:`False` to apply numerical
			smoothing only at the end of each timestep. Defaults to :obj:`True`.
		smooth_moist_type: `str`, optional
			String specifying the smoothing technique to apply on the water constituents.
			See :class:`tasmania.HorizontalSmoothing` for all available options.
			Defaults to 'first-order'. 
		smooth_moist_coeff : `float`, optional
			Smoothing coefficient for the water constituents. Defaults to 0.03.
		smooth_moist_coeff_max : `float`, optional
			Maximum value for the smoothing coefficient for the water constituents.
			See :class:`tasmania.HorizontalSmoothing` for further details. 
			Defaults to 0.24. 
		smooth_moist_damp_depth : `int`, optional
			Number of vertical layers in the smoothing damping region for the
			water constituents. Defaults to 10.
		backend : `str`, optional
			TODO
		backend_opts : `dict`, optional
			TODO
		build_info : `dict`, optional
			TODO
		dtype : `numpy.dtype`, optional
			TODO
		exec_info : `dict`, optional
			TODO
		halo : `tuple`, optional
			TODO
		rebuild : `bool`, optional
			TODO
		**kwargs :
			TODO
		"""
		#
		# input parameters
		#
		self._moist							= moist
		self._damp					   	   	= damp
		self._damp_at_every_stage		   	= damp_at_every_stage
		self._smooth					   	= smooth
		self._smooth_at_every_stage		   	= smooth_at_every_stage
		self._smooth_moist			   		= smooth_moist
		self._smooth_moist_at_every_stage  	= smooth_moist_at_every_stage

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
		kwargs = {} if time_integration_properties is None else \
			time_integration_properties
		self._prognostic = IsentropicPrognostic.factory(
			time_integration_scheme, horizontal_flux_scheme, self.grid,
			self.horizontal_boundary, moist, backend=backend,
			backend_opts=backend_opts, build_info=build_info, dtype=dtype,
			exec_info=exec_info, halo=halo, rebuild=rebuild, **kwargs
		)

		#
		# vertical damping
		#
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		if damp:
			self._damper = VerticalDamping.factory(
				damp_type, self.grid, (nx+1, ny+1, nz+1), damp_depth, damp_max,
				time_units='s', backend=backend, backend_opts=backend_opts,
				build_info=build_info, dtype=dtype, exec_info=exec_info,
				halo=halo, rebuild=rebuild
			)

		#
		# numerical smoothing
		#
		if smooth:
			self._smoother = HorizontalSmoothing.factory(
				smooth_type, (nx+1, ny+1, nz+1), smooth_coeff, smooth_coeff_max,
				smooth_damp_depth, hb.nb, backend=backend,
				backend_opts=backend_opts, build_info=build_info, dtype=dtype,
				exec_info=exec_info, halo=halo, rebuild=rebuild
			)
			if moist and smooth_moist:
				self._smoother_moist = HorizontalSmoothing.factory(
					smooth_moist_type, (nx+1, ny+1, nz+1), smooth_moist_coeff,
					smooth_moist_coeff_max, smooth_moist_damp_depth, hb.nb,
					backend=backend, backend_opts=backend_opts, build_info=build_info,
					dtype=dtype, exec_info=exec_info, halo=halo, rebuild=False
				)

		#
		# diagnostics
		#
		self._velocity_components = HorizontalVelocity(
			self.grid, staggering=True, backend=backend,
			backend_opts=backend_opts, build_info=build_info,
			exec_info=exec_info, rebuild=rebuild
		)
		if moist:
			self._water_constituent = WaterConstituent(
				self.grid, clipping=True, backend=backend,
				backend_opts=backend_opts, build_info=build_info,
				exec_info=exec_info, rebuild=rebuild
			)

		#
		# the method implementing each stage
		#
		self._array_call = self._array_call_dry if not moist else self._array_call_moist

		#
		# temporary and output arrays
		#
		descriptor = get_storage_descriptor((nx+1, ny+1, nz+1), dtype, halo=halo)

		self._s_new  = gt.storage.zeros(descriptor, backend=backend)
		self._su_new = gt.storage.zeros(descriptor, backend=backend)
		self._sv_new = gt.storage.zeros(descriptor, backend=backend)
		if moist:
			self._s_now = gt.storage.zeros(descriptor, backend=backend)
			self._s_now_1 = gt.storage.zeros(descriptor, backend=backend)
			self._qv_now = gt.storage.zeros(descriptor, backend=backend)
			self._qc_now = gt.storage.zeros(descriptor, backend=backend)
			self._qr_now = gt.storage.zeros(descriptor, backend=backend)
			self._sqv_now = gt.storage.zeros(descriptor, backend=backend)
			self._sqc_now = gt.storage.zeros(descriptor, backend=backend)
			self._sqr_now = gt.storage.zeros(descriptor, backend=backend)
			self._sqv_new = gt.storage.zeros(descriptor, backend=backend)
			self._sqc_new = gt.storage.zeros(descriptor, backend=backend)
			self._sqr_new = gt.storage.zeros(descriptor, backend=backend)
			self._qv_new = gt.storage.zeros(descriptor, backend=backend)
			self._qc_new = gt.storage.zeros(descriptor, backend=backend)
			self._qr_new = gt.storage.zeros(descriptor, backend=backend)

		if damp:
			self._s_now     = gt.storage.zeros(descriptor, backend=backend)
			self._s_ref     = gt.storage.zeros(descriptor, backend=backend)
			self._s_damped  = gt.storage.zeros(descriptor, backend=backend)
			self._su_now    = gt.storage.zeros(descriptor, backend=backend)
			self._su_ref    = gt.storage.zeros(descriptor, backend=backend)
			self._su_damped = gt.storage.zeros(descriptor, backend=backend)
			self._sv_now    = gt.storage.zeros(descriptor, backend=backend)
			self._sv_ref    = gt.storage.zeros(descriptor, backend=backend)
			self._sv_damped = gt.storage.zeros(descriptor, backend=backend)
			if moist:
				self._qv_ref    = gt.storage.zeros(descriptor, backend=backend)
				self._qv_damped = gt.storage.zeros(descriptor, backend=backend)
				self._qc_ref    = gt.storage.zeros(descriptor, backend=backend)
				self._qc_damped = gt.storage.zeros(descriptor, backend=backend)
				self._qr_ref    = gt.storage.zeros(descriptor, backend=backend)
				self._qr_damped = gt.storage.zeros(descriptor, backend=backend)

		if smooth:
			self._s_smoothed  = gt.storage.zeros(descriptor, backend=backend)
			self._su_smoothed = gt.storage.zeros(descriptor, backend=backend)
			self._sv_smoothed = gt.storage.zeros(descriptor, backend=backend)

		if smooth_moist:
			self._qv_smoothed = gt.storage.zeros(descriptor, backend=backend)
			self._qc_smoothed = gt.storage.zeros(descriptor, backend=backend)
			self._qr_smoothed = gt.storage.zeros(descriptor, backend=backend)

		self._u_out = gt.storage.zeros(descriptor, backend=backend)
		self._v_out = gt.storage.zeros(descriptor, backend=backend)

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

		if self._moist:
			return_dict[mfwv] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfcw] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfpw] = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def _substep_input_properties(self):
		dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])
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

		if self._moist:
			return_dict[mfwv] = {'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict[mfcw] = {'dims': dims, 'units': 'g g^-1 s^-1'}
			return_dict[mfpw] = {'dims': dims, 'units': 'g g^-1 s^-1'}

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

		if self._moist:
			return_dict[mfwv] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfcw] = {'dims': dims, 'units': 'g g^-1'}
			return_dict[mfpw] = {'dims': dims, 'units': 'g g^-1'}

		return return_dict

	@property
	def _substep_output_properties(self):
		if not hasattr(self, '__substep_output_properties'):
			dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

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
		if self._moist:
			names.append(mfwv)
			names.append(mfcw)
			names.append(mfpw)

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
		return self._array_call(stage, raw_state, raw_tendencies, timestep)

	def _array_call_dry(self, stage, raw_state, raw_tendencies, timestep):
		""" Perform a stage of the dry dynamical core. """
		# shortcuts
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		hb = self.horizontal_boundary
		out_properties = self.output_properties

		if self._damp and stage == 0:
			# set the reference state
			try:
				ref_state = hb.reference_state
				self._s_ref.data[:nx, :ny, :nz]  = \
					ref_state['air_isentropic_density'].to_units('kg m^-2 K^-1').values
				self._su_ref.data[:nx, :ny, :nz] = \
					ref_state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values
				self._sv_ref.data[:nx, :ny, :nz] = \
					ref_state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values
			except KeyError:
				raise RuntimeError(
					"Reference state not set in the object handling the horizontal "
					"boundary conditions, but needed by the wave absorber."
				)

			# save the current solution
			self._s_now.data[:nx, :ny, :nz]  = raw_state['air_isentropic_density']
			self._su_now.data[:nx, :ny, :nz] = raw_state['x_momentum_isentropic']
			self._sv_now.data[:nx, :ny, :nz] = raw_state['y_momentum_isentropic']

		# perform the prognostic step
		raw_state_new = self._prognostic.stage_call(
			stage, timestep, raw_state, raw_tendencies
		)

		# apply the lateral boundary conditions
		hb.dmn_enforce_raw(raw_state_new, out_properties)

		# extract the stepped prognostic model variables
		self._s_new.data[:nx, :ny, :nz]  = raw_state_new['air_isentropic_density']
		self._su_new.data[:nx, :ny, :nz] = raw_state_new['x_momentum_isentropic']
		self._sv_new.data[:nx, :ny, :nz] = raw_state_new['y_momentum_isentropic']

		damped = False
		if self._damp and (self._damp_at_every_stage or stage == self.stages-1):
			damped = True

			# apply vertical damping
			self._damper(timestep, self._s_now , self._s_new , self._s_ref , self._s_damped )
			self._damper(timestep, self._su_now, self._su_new, self._su_ref, self._su_damped)
			self._damper(timestep, self._sv_now, self._sv_new, self._sv_ref, self._sv_damped)

		# properly set pointers to current solution
		s_new  = self._s_damped if damped else self._s_new
		su_new = self._su_damped if damped else self._su_new
		sv_new = self._sv_damped if damped else self._sv_new

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
				'air_isentropic_density': self._s_smoothed.data[:nx, :ny, :nz],
				'x_momentum_isentropic': self._su_smoothed.data[:nx, :ny, :nz],
				'y_momentum_isentropic': self._sv_smoothed.data[:nx, :ny, :nz],
			}
			hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

		# properly set pointers to output solution
		s_out  = self._s_smoothed if smoothed else s_new
		su_out = self._su_smoothed if smoothed else su_new
		sv_out = self._sv_smoothed if smoothed else sv_new

		# diagnose the velocity components
		self._velocity_components.get_velocity_components(
			s_out, su_out, sv_out, self._u_out, self._v_out
		)
		hb.dmn_set_outermost_layers_x(
			self._u_out.data[:nx+1, :ny, :nz], field_name='x_velocity_at_u_locations',
			field_units=out_properties['x_velocity_at_u_locations']['units'],
			time=raw_state_new['time']
		)
		hb.dmn_set_outermost_layers_y(
			self._v_out.data[:nx, :ny+1, :nz], field_name='y_velocity_at_v_locations',
			field_units=out_properties['y_velocity_at_v_locations']['units'],
			time=raw_state_new['time']
		)

		# instantiate the output state
		raw_state_out = {
			'time': raw_state_new['time'],
			'air_isentropic_density': s_out.data[:nx, :ny, :nz],
			'x_momentum_isentropic': su_out.data[:nx, :ny, :nz],
			'x_velocity_at_u_locations': self._u_out.data[:nx+1, :ny, :nz],
			'y_momentum_isentropic': sv_out.data[:nx, :ny, :nz],
			'y_velocity_at_v_locations': self._v_out.data[:nx, :ny+1, :nz],
		}

		return raw_state_out

	def _array_call_moist(self, stage, raw_state, raw_tendencies, timestep):
		"""	Perform a stage of the moist dynamical core. """
		# shortcuts
		nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
		hb = self.horizontal_boundary
		out_properties = self.output_properties

		if self._damp and stage == 0:
			# set the reference state
			try:
				ref_state = hb.reference_state
				self._s_ref.data[:nx, :ny, :nz] = \
					ref_state['air_isentropic_density'].to_units('kg m^-2 K^-1').values
				self._su_ref.data[:nx, :ny, :nz] = \
					ref_state['x_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values
				self._sv_ref.data[:nx, :ny, :nz] = \
					ref_state['y_momentum_isentropic'].to_units('kg m^-1 K^-1 s^-1').values
				self._qv_ref.data[:nx, :ny, :nz] = \
					ref_state[mfwv].to_units('g g^-1').values
				self._qc_ref.data[:nx, :ny, :nz] = \
					ref_state[mfcw].to_units('g g^-1').values
				self._qr_ref.data[:nx, :ny, :nz] = \
					ref_state[mfpw].to_units('g g^-1').values
			except KeyError:
				raise RuntimeError(
					"Reference state not set in the object handling the horizontal "
					"boundary conditions, but needed by the wave absorber."
				)

			# save the current solution
			self._s_now.data[:nx, :ny, :nz] = raw_state['air_isentropic_density']
			self._su_now.data[:nx, :ny, :nz] = raw_state['x_momentum_isentropic']
			self._sv_now.data[:nx, :ny, :nz] = raw_state['y_momentum_isentropic']
			# self._qv_now.data[:nx, :ny, :nz] = raw_state[mfwv]
			# self._qc_now.data[:nx, :ny, :nz] = raw_state[mfcw]
			# self._qr_now.data[:nx, :ny, :nz] = raw_state[mfpw]

		self._s_now_1.data[:nx, :ny, :nz] = raw_state['air_isentropic_density']
		self._qv_now.data[:nx, :ny, :nz] = raw_state[mfwv]
		self._qc_now.data[:nx, :ny, :nz] = raw_state[mfcw]
		self._qr_now.data[:nx, :ny, :nz] = raw_state[mfpw]

		# diagnose the isentropic density of all water constituents
		self._water_constituent.get_density_of_water_constituent(
			self._s_now_1, self._qv_now, self._sqv_now
		)
		self._water_constituent.get_density_of_water_constituent(
			self._s_now_1, self._qc_now, self._sqc_now
		)
		self._water_constituent.get_density_of_water_constituent(
			self._s_now_1, self._qr_now, self._sqr_now
		)
		raw_state['isentropic_density_of_water_vapor'] = \
			self._sqv_now.data[:nx, :ny, :nz]
		raw_state['isentropic_density_of_cloud_liquid_water'] = \
			self._sqc_now.data[:nx, :ny, :nz]
		raw_state['isentropic_density_of_precipitation_water'] = \
			self._sqr_now.data[:nx, :ny, :nz]

		# perform the prognostic step
		raw_state_new = self._prognostic.stage_call(
			stage, timestep, raw_state, raw_tendencies
		)

		# extract the stepped prognostic model variables
		self._s_new.data[:nx, :ny, :nz]   = raw_state_new['air_isentropic_density']
		self._sqv_new.data[:nx, :ny, :nz] = raw_state_new['isentropic_density_of_water_vapor']
		self._sqc_new.data[:nx, :ny, :nz] = raw_state_new['isentropic_density_of_cloud_liquid_water']
		self._sqr_new.data[:nx, :ny, :nz] = raw_state_new['isentropic_density_of_precipitation_water']

		# diagnose the mass fraction of all water constituents
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			self._s_new, self._sqv_new,	self._qv_new
		)
		raw_state_new[mfwv] = self._qv_new.data[:nx, :ny, :nz]
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			self._s_new, self._sqc_new,	self._qc_new
		)
		raw_state_new[mfcw] = self._qc_new.data[:nx, :ny, :nz]
		self._water_constituent.get_mass_fraction_of_water_constituent_in_air(
			self._s_new, self._sqr_new,	self._qr_new
		)
		raw_state_new[mfpw] = self._qr_new.data[:nx, :ny, :nz]

		# apply the lateral boundary conditions
		hb.dmn_enforce_raw(raw_state_new, out_properties)

		# extract the stepped prognostic model variables
		self._s_new.data[:nx, :ny, :nz]  = raw_state_new['air_isentropic_density']
		self._su_new.data[:nx, :ny, :nz] = raw_state_new['x_momentum_isentropic']
		self._sv_new.data[:nx, :ny, :nz] = raw_state_new['y_momentum_isentropic']

		damped = False
		if self._damp and (self._damp_at_every_stage or stage == self.stages-1):
			damped = True

			# apply vertical damping
			self._damper(timestep, self._s_now , self._s_new , self._s_ref , self._s_damped )
			self._damper(timestep, self._su_now, self._su_new, self._su_ref, self._su_damped)
			self._damper(timestep, self._sv_now, self._sv_new, self._sv_ref, self._sv_damped)
			# self._damper(timestep, self._qv_now, self._qv_new, self._qv_ref, self._qv_damped)
			# self._damper(timestep, self._qc_now, self._qc_new, self._qc_ref, self._qc_damped)
			# self._damper(timestep, self._qr_now, self._qr_new, self._qr_ref, self._qr_damped)

		# properly set pointers to current solution
		s_new  = self._s_damped if damped else self._s_new
		su_new = self._su_damped if damped else self._su_new
		sv_new = self._sv_damped if damped else self._sv_new
		qv_new = self._qv_new  # self._qv_damped if damped else self._qv_new
		qc_new = self._qc_new  # self._qc_damped if damped else self._qc_new
		qr_new = self._qr_new  # self._qr_damped if damped else self._qr_new

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
				'air_isentropic_density': self._s_smoothed.data[:nx, :ny, :nz],
				'x_momentum_isentropic': self._su_smoothed.data[:nx, :ny, :nz],
				'y_momentum_isentropic': self._sv_smoothed.data[:nx, :ny, :nz],
			}
			hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

		# properly set pointers to output solution
		s_out  = self._s_smoothed if smoothed else s_new
		su_out = self._su_smoothed if smoothed else su_new
		sv_out = self._sv_smoothed if smoothed else sv_new

		smoothed_moist = False
		if self._smooth_moist and (self._smooth_moist_at_every_stage or stage == self.stages-1):
			smoothed_moist = True

			# apply horizontal smoothing
			self._smoother_moist(qv_new, self._qv_smoothed)
			self._smoother_moist(qc_new, self._qc_smoothed)
			self._smoother_moist(qr_new, self._qr_smoothed)

			# apply horizontal boundary conditions
			raw_state_smoothed = {
				'time': raw_state_new['time'],
				mfwv: self._qv_smoothed.data[:nx, :ny, :nz],
				mfcw: self._qc_smoothed.data[:nx, :ny, :nz],
				mfpw: self._qr_smoothed.data[:nx, :ny, :nz],
			}
			hb.dmn_enforce_raw(raw_state_smoothed, out_properties)

		# properly set pointers to output solution
		qv_out = self._qv_smoothed if smoothed_moist else qv_new
		qc_out = self._qc_smoothed if smoothed_moist else qc_new
		qr_out = self._qr_smoothed if smoothed_moist else qr_new

		# diagnose the velocity components
		self._velocity_components.get_velocity_components(
			s_out, su_out, sv_out, self._u_out, self._v_out
		)
		hb.dmn_set_outermost_layers_x(
			self._u_out.data[:nx+1, :ny, :nz], field_name='x_velocity_at_u_locations',
			field_units=out_properties['x_velocity_at_u_locations']['units'],
			time=raw_state_new['time']
		)
		hb.dmn_set_outermost_layers_y(
			self._v_out.data[:nx, :ny+1, :nz], field_name='y_velocity_at_v_locations',
			field_units=out_properties['y_velocity_at_v_locations']['units'],
			time=raw_state_new['time']
		)

		# instantiate the output state
		raw_state_out = {
			'time': raw_state_new['time'],
			'air_isentropic_density': s_out.data[:nx, :ny, :nz],
			mfwv: qv_out.data[:nx, :ny, :nz],
			mfcw: qc_out.data[:nx, :ny, :nz],
			mfpw: qr_out.data[:nx, :ny, :nz],
			'x_momentum_isentropic': su_out.data[:nx, :ny, :nz],
			'x_velocity_at_u_locations': self._u_out.data[:nx+1, :ny, :nz],
			'y_momentum_isentropic': sv_out.data[:nx, :ny, :nz],
			'y_velocity_at_v_locations': self._v_out.data[:nx, :ny+1, :nz],
		}

		return raw_state_out

	def substep_array_call(
		self, stage, substep, raw_state, raw_stage_state, raw_tmp_state,
		raw_tendencies, timestep
	):
		raise NotImplementedError()
