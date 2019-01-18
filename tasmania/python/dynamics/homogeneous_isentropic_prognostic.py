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
	HomogeneousIsentropicPrognostic
"""
import abc
import numpy as np

import gridtools as gt
from tasmania.python.dynamics.isentropic_fluxes import \
	HorizontalHomogeneousIsentropicFlux

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


# Convenient aliases
mfwv = 'mass_fraction_of_water_vapor_in_air'
mfcw = 'mass_fraction_of_cloud_liquid_water_in_air'
mfpw = 'mass_fraction_of_precipitation_water_in_air'


class HomogeneousIsentropicPrognostic:
	"""
	Abstract base class whose derived classes implement different
	schemes to carry out the prognostic steps of the three-dimensional
	homogeneous, moist, isentropic dynamical core. Here, *homogeneous* means
	that the pressure gradient terms, i.e., the terms involving the gradient
	of the Montgomery potential, are not included in the dynamics, but
	rather parameterized. This holds also for any sedimentation motion.
	The conservative form of the governing equations is used.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(
		self, mode, grid, moist, horizontal_boundary_conditions,
		horizontal_flux_scheme, substeps, backend, dtype=datatype
	):
		"""
		Parameters
		----------
		mode : str
			TODO
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		horizontal_boundary_conditions : obj
			Instance of a derived class of
			:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			in charge of handling the lateral boundary conditions.
			This is modified in-place by setting the number of boundary layers.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux scheme to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalHomogeneousIsentropicFlux`
			for the complete list of the available options.
		substeps : int
			The number of substeps to perform.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.
		"""
		# Keep track of the input parameters
		self._mode			= mode if mode in ['x', 'y', 'xy'] else 'xy'
		self._grid          = grid
		self._moist      	= moist
		self._hboundary		= horizontal_boundary_conditions
		self._hflux_scheme	= horizontal_flux_scheme
		self._substeps		= substeps
		self._backend		= backend
		self._dtype			= dtype

		# Instantiate the classes computing the numerical horizontal fluxes
		self._hflux = HorizontalHomogeneousIsentropicFlux.factory(
			self._hflux_scheme, grid, moist,
		)
		self._hboundary.nb = self._hflux.nb

		# Initialize properties dictionary
		self._substep_output_properties = None

		# Initialize the pointer to the underlying GT4Py stencil in charge
		# of carrying out the sub-steps
		if substeps > 0:
			self._substep_stencil = None

	@property
	@abc.abstractmethod
	def stages(self):
		"""
		Return
		------
		int :
			The number of stages performed by the time-integration scheme.
		"""

	@property
	@abc.abstractmethod
	def substep_fractions(self):
		"""
		Return
		------
		float or tuple :
			In a partial time splitting framework, for each stage, fraction of the
			total number of substeps to carry out.
		"""

	@property
	def nb(self):
		"""
		Return
		------
		int :
			The number of lateral boundary layers.
		"""
		return self._hflux.nb

	@property
	def horizontal_boundary(self):
		"""
		Return
		------
		obj :
			Object in charge of handling the lateral boundary conditions.
		"""
		return self._hboundary

	@property
	def substep_output_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which are
			included in the output state returned by any substep, and whose
			values are fundamental properties (dims, units) of those variables.
		"""
		if self._substep_output_properties is None:
			raise RuntimeError('substep_output_properties required but not set.')
		return self._substep_output_properties

	@substep_output_properties.setter
	def substep_output_properties(self, value):
		"""
		Parameters
		----------
		value : dict
			Dictionary whose keys are strings denoting variables which are
			included in the output state returned by any substep, and whose
			values are fundamental properties (dims, units) of those variables.
		"""
		self._substep_output_properties = value

	@abc.abstractmethod
	def stage_call(self, stage, dt, state, tendencies=None):
		"""
		Perform a stage.

		Parameters
		----------
		stage : int
			The stage to perform.
		dt : timedelta
			:class:`datetime.timedelta` representing the time step.
		state : dict
			Dictionary whose keys are strings indicating model variables,
			and values are :class:`numpy.ndarray`\s representing the values
			for those variables.
		tendencies : dict
			Dictionary whose keys are strings indicating model variables,
			and values are :class:`numpy.ndarray`\s representing (slow and
			intermediate) tendencies for those variables.

		Return
		------
		dict :
			Dictionary whose keys are strings indicating the conservative
			prognostic model variables, and values are :class:`numpy.ndarray`\s
			containing new values for those variables.
		"""

	def substep_call(
		self, stage, substep, dt, state, stage_state, tmp_state, tendencies=None
	):
		"""
		Perform a sub-step.

		Parameters
		----------
		stage : int
			The stage to perform.
		substep : int
			The substep to perform.
		dt : timedelta
			:class:`datetime.timedelta` representing the time step.
		state : dict
			The raw state at the current *main* time level.
		stage_state : dict
			The (raw) state dictionary returned by the latest stage.
		tmp_state : dict
			The raw state to sub-step.
		tendencies : dict
			Dictionary whose keys are strings indicating model variables,
			and values are :class:`numpy.ndarray`\s representing
			(fast) tendencies for those values.

		Return
		------
		dict :
			Dictionary whose keys are strings indicating the conservative
			prognostic model variables, and values are :class:`numpy.ndarray`\s
			containing new values for those variables.
		"""
		# Initialize the stencil object
		if self._substep_stencil is None:
			self._substep_stencil_initialize(tendencies)

		# Set the stencil's inputs
		self._substep_stencil_set_inputs(
			stage, substep, dt, state, stage_state, tmp_state, tendencies
		)

		# Evaluate the stencil
		self._substep_stencil.compute()

		# Diagnose the accumulated precipitation
		if 'accumulated_precipitation' in self.substep_output_properties:
			if stage == self.stages-1:
				self._substep_stencil_outputs['accumulated_precipitation'][:, :] = \
					tmp_state['accumulated_precipitation'][:, :] + \
					tmp_state['precipitation'][:, :] * dt.total_seconds() / self._substeps / 3.6e3
			elif stage == 0 and substep == 0:
				self._substep_stencil_outputs['accumulated_precipitation'][...] = \
					tmp_state['accumulated_precipitation'][...]

		# Compose the output state
		out_time = state['time'] + dt / self._substeps if substep == 0 else \
			tmp_state['time'] + dt / self._substeps
		out_state = {'time': out_time}
		for key in self._substep_output_properties:
			out_state[key] = self._substep_stencil_outputs[key]

		return out_state

	@staticmethod
	def factory(
		scheme, mode, grid, moist, horizontal_boundary_conditions,
		horizontal_flux_scheme, substeps=0, backend=gt.mode.NUMPY, dtype=datatype
	):
		"""
		Static method returning an instance of the derived class implementing
		the time stepping scheme specified by :data:`time_scheme`.
		Parameters
		----------
		scheme : str
			String specifying the time stepping method to implement. Either:

				* 'forward_euler', for the forward Euler scheme;
				* 'centered', for a centered scheme;
				* 'rk2', for the two-stages, second-order Runge-Kutta (RK) scheme;
				* 'rk3cosmo', for the three-stages RK scheme as used in the
					`COSMO model <http://www.cosmo-model.org>`_; this method is
					nominally second-order, and third-order for linear problems;
				* 'rk3', for the three-stages, third-order RK scheme.

		mode : str
			TODO
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		moist : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		horizontal_boundary_conditions : obj
			Instance of a derived class of
			:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			in charge of handling the lateral boundary conditions.
		horizontal_flux_scheme : str
			String specifying the numerical horizontal flux scheme to use.
			See :class:`~tasmania.dynamics.isentropic_fluxes.HorizontalHomogeneousHorizontalFlux`
			for the complete list of the available options.
		substeps : `int`, optional
			Number of sub-steps to perform.
		backend : `obj`, optional
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`, or :obj:`numpy.float32`
			if :obj:`~tasmania.namelist.datatype` is not defined.

		Return
		------
		obj :
			An instance of the derived class implementing the scheme specified
			by :data:`scheme`.
		"""
		import tasmania.python.dynamics._homogeneous_isentropic_prognostic as module
		arg_list = [
			mode, grid, moist, horizontal_boundary_conditions,
			horizontal_flux_scheme, substeps, backend, dtype
		]

		if scheme == 'forward_euler':
			return module.ForwardEuler(*arg_list)
		elif scheme == 'centered':
			return module.Centered(*arg_list)
		elif scheme == 'rk2':
			return module.RK2(*arg_list)
		elif scheme == 'rk3cosmo':
			return module.RK3COSMO(*arg_list)
		elif scheme == 'rk3':
			return module.RK3(*arg_list)
		else:
			raise ValueError(
				'Unknown time integration scheme {}. Available options: '
				'forward_euler, centered, rk2, rk3cosmo, rk3.'.format(scheme)
			)

	def _stage_stencil_allocate_inputs(self, tendencies):
		"""
		Allocate the attributes which serve as inputs to the GT4Py stencils
		which implement the stages.
		"""
		# Shortcuts
		nz = self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype
		tendency_names = () if tendencies is None else tendencies.keys()

		# Instantiate a GT4Py Global representing the timestep
		self._dt = gt.Global()

		# Allocate the Numpy arrays which will store the current solution
		# and serve as stencil's inputs
		self._in_s	= np.zeros((  mi,	mj, nz), dtype=dtype)
		self._in_u	= np.zeros((mi+1,	mj, nz), dtype=dtype)
		self._in_v	= np.zeros((  mi, mj+1, nz), dtype=dtype)
		self._in_su = np.zeros((  mi,	mj, nz), dtype=dtype)
		self._in_sv = np.zeros((  mi,	mj, nz), dtype=dtype)
		if self._moist:
			self._in_sqv = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqc = np.zeros((mi, mj, nz), dtype=dtype)
			self._in_sqr = np.zeros((mi, mj, nz), dtype=dtype)

		# Allocate the input Numpy arrays which will store the tendencies
		# and serve as stencil's inputs
		if tendency_names is not None:
			if 'air_isentropic_density' in tendency_names:
				self._in_s_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if 'x_momentum_isentropic' in tendency_names:
				self._in_su_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if 'y_momentum_isentropic' in tendency_names:
				self._in_sv_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if mfwv in tendency_names:
				self._in_qv_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if mfcw in tendency_names:
				self._in_qc_tnd = np.zeros((mi, mj, nz), dtype=dtype)
			if mfpw in tendency_names:
				self._in_qr_tnd = np.zeros((mi, mj, nz), dtype=dtype)

	def _stage_stencil_allocate_outputs(self):
		"""
		Allocate the Numpy arrays which serve as outputs for the GT4Py stencils
		which perform the stages.
		"""
		# Shortcuts
		nz = self._grid.nz
		mi, mj = self._hboundary.mi, self._hboundary.mj
		dtype = self._dtype

		# Allocate the Numpy arrays which will serve as stencil's outputs
		self._out_s  = np.zeros((mi, mj, nz), dtype=dtype)
		self._out_su = np.zeros((mi, mj, nz), dtype=dtype)
		self._out_sv = np.zeros((mi, mj, nz), dtype=dtype)
		if self._moist:
			self._out_sqv = np.zeros((mi, mj, nz), dtype=dtype)
			self._out_sqc = np.zeros((mi, mj, nz), dtype=dtype)
			self._out_sqr = np.zeros((mi, mj, nz), dtype=dtype)

	def _stage_stencil_set_inputs(self, stage, dt, state, tendencies):
		"""
		Update the attributes which serve as inputs to the GT4Py stencils
		which perform the stages.
		"""
		# Shortcuts
		mi, mj = self._hboundary.mi, self._hboundary.mj
		if tendencies is not None:
			s_tnd_on  = tendencies.get('air_isentropic_density', None) is not None
			qv_tnd_on = tendencies.get(mfwv, None) is not None
			qc_tnd_on = tendencies.get(mfcw, None) is not None
			qr_tnd_on = tendencies.get(mfpw, None) is not None
			su_tnd_on = tendencies.get('x_momentum_isentropic', None) is not None
			sv_tnd_on = tendencies.get('y_momentum_isentropic', None) is not None
		else:
			s_tnd_on = su_tnd_on = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False

		# Update the local time step
		self._dt.value = dt.total_seconds()

		# Extract the Numpy arrays representing the current solution
		s	= state['air_isentropic_density']
		u	= state['x_velocity_at_u_locations']
		v	= state['y_velocity_at_v_locations']
		su	= state['x_momentum_isentropic']
		sv	= state['y_momentum_isentropic']
		if self._moist:
			sqv = state['isentropic_density_of_water_vapor']
			sqc = state['isentropic_density_of_cloud_liquid_water']
			sqr = state['isentropic_density_of_precipitation_water']
		if s_tnd_on:
			s_tnd = tendencies['air_isentropic_density']
		if qv_tnd_on:
			qv_tnd = tendencies[mfwv]
		if qc_tnd_on:
			qc_tnd = tendencies[mfcw]
		if qr_tnd_on:
			qr_tnd = tendencies[mfpw]
		if su_tnd_on:
			su_tnd = tendencies['x_momentum_isentropic']
		if sv_tnd_on:
			sv_tnd = tendencies['y_momentum_isentropic']

		# Update the Numpy arrays which serve as inputs to the GT4Py stencils
		self._in_s	[  :mi,   :mj, :] = \
			self._hboundary.from_physical_to_computational_domain(s)
		self._in_u	[:mi+1,   :mj, :] = \
			self._hboundary.from_physical_to_computational_domain(u)
		self._in_v	[  :mi, :mj+1, :] = \
			self._hboundary.from_physical_to_computational_domain(v)
		self._in_su [  :mi,   :mj, :] = \
			self._hboundary.from_physical_to_computational_domain(su)
		self._in_sv [  :mi,   :mj, :] = \
			self._hboundary.from_physical_to_computational_domain(sv)
		if self._moist:
			self._in_sqv[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqv)
			self._in_sqc[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqc)
			self._in_sqr[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sqr)
		if s_tnd_on:
			self._in_s_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(s_tnd)
		if su_tnd_on:
			self._in_su_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(su_tnd)
		if sv_tnd_on:
			self._in_sv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(sv_tnd)
		if qv_tnd_on:
			self._in_qv_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qv_tnd)
		if qc_tnd_on:
			self._in_qc_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qc_tnd)
		if qr_tnd_on:
			self._in_qr_tnd[:mi, :mj, :] = \
				self._hboundary.from_physical_to_computational_domain(qr_tnd)

	def _substep_stencil_initialize(self, tendencies):
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		dtype = self._dtype
		tendency_names = () if tendencies is None else tendencies.keys()

		self._dts = gt.Global()
		self._stage_substeps = gt.Global()

		self._substep_stencil_state_inputs = {}
		self._substep_stencil_stage_state_inputs = {}
		self._substep_stencil_tmp_state_inputs = {}
		self._substep_stencil_tendencies_inputs = {}
		self._substep_stencil_outputs = {}

		inputs = {}
		outputs = {}

		shorthands = {
			'air_isentropic_density': 's',
			'x_momentum_isentropic': 'su',
			'y_momentum_isentropic': 'sv',
			'mass_fraction_of_water_vapor_in_air': 'qv',
			'mass_fraction_of_cloud_liquid_water_in_air': 'qc',
			'mass_fraction_of_precipitation_water_in_air': 'qr',
		}

		for var, shand in shorthands.items():
			if var in self._substep_output_properties:
				self._substep_stencil_state_inputs[var] = \
					np.zeros((nx, ny, nz), dtype=dtype)
				inputs[shand] = self._substep_stencil_state_inputs[var]

				self._substep_stencil_stage_state_inputs[var] = \
					np.zeros((nx, ny, nz), dtype=dtype)
				inputs['stage_' + shand] = self._substep_stencil_stage_state_inputs[var]

				self._substep_stencil_tmp_state_inputs[var] = \
					np.zeros((nx, ny, nz), dtype=dtype)
				inputs['tmp_' + shand] = self._substep_stencil_tmp_state_inputs[var]

				if var in tendency_names:
					self._substep_stencil_tendencies_inputs[var] = \
						np.zeros((nx, ny, nz), dtype=dtype)
					inputs['tnd_' + shand] = self._substep_stencil_tendencies_inputs[var]

				self._substep_stencil_outputs[var] = np.zeros((nx, ny, nz), dtype=dtype)
				outputs['out_' + shand] = self._substep_stencil_outputs[var]

		if 'accumulated_precipitation' in self._substep_output_properties:
			self._substep_stencil_outputs['accumulated_precipitation'] = \
				np.zeros((nx, ny), dtype=dtype)

		self._substep_stencil = gt.NGStencil(
			definitions_func = self.__class__._substep_stencil_defs,
			inputs			 = inputs,
			global_inputs	 = {'dts': self._dts, 'substeps': self._stage_substeps},
			outputs			 = outputs,
			domain			 = gt.domain.Rectangle((0, 0, 0), (nx-1, ny-1, nz-1)),
			mode			 = self._backend,
		)

	def _substep_stencil_set_inputs(
		self, stage, substep, dt, state, stage_state, tmp_state, tendencies
	):
		tendency_names = () if tendencies is None else tendencies.keys()

		self._dts.value = dt.total_seconds() / self._substeps
		self._stage_substeps.value = \
			self._substeps if self.stages == 1 else \
			self.substep_fractions[stage] * self._substeps

		for var in self._substep_output_properties:
			if var != 'accumulated_precipitation':
				if substep == 0:
					self._substep_stencil_state_inputs[var][...] = state[var][...]
				self._substep_stencil_stage_state_inputs[var][...] = stage_state[var][...]
				self._substep_stencil_tmp_state_inputs[var][...] = \
					state[var][...] if substep == 0 else tmp_state[var][...]
				if var in tendency_names:
					self._substep_stencil_tendencies_inputs[var][...] = tendencies[var][...]

	@staticmethod
	def _substep_stencil_defs(
		dts, substeps,
		s=None,  stage_s=None,  tmp_s=None,  tnd_s=None,
		su=None, stage_su=None, tmp_su=None, tnd_su=None,
		sv=None, stage_sv=None, tmp_sv=None, tnd_sv=None,
		qv=None, stage_qv=None, tmp_qv=None, tnd_qv=None,
		qc=None, stage_qc=None, tmp_qc=None, tnd_qc=None,
		qr=None, stage_qr=None, tmp_qr=None, tnd_qr=None
	):
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		outs = []

		if s is not None:
			out_s = gt.Equation()

			if tnd_s is None:
				out_s[i, j, k] = tmp_s[i, j, k] + \
								 (stage_s[i, j, k] - s[i, j, k]) / substeps
			else:
				out_s[i, j, k] = tmp_s[i, j, k] + \
								 (stage_s[i, j, k] - s[i, j, k]) / substeps + \
								 dts * tnd_s[i, j, k]

			outs.append(out_s)

		if su is not None:
			out_su = gt.Equation()

			if tnd_su is None:
				out_su[i, j, k] = tmp_su[i, j, k] + \
								  (stage_su[i, j, k] - su[i, j, k]) / substeps
			else:
				out_su[i, j, k] = tmp_su[i, j, k] + \
								  (stage_su[i, j, k] - su[i, j, k]) / substeps + \
								  dts * tnd_su[i, j, k]

			outs.append(out_su)

		if sv is not None:
			out_sv = gt.Equation()

			if tnd_sv is None:
				out_sv[i, j, k] = tmp_sv[i, j, k] + \
								  (stage_sv[i, j, k] - sv[i, j, k]) / substeps
			else:
				out_sv[i, j, k] = tmp_sv[i, j, k] + \
								  (stage_sv[i, j, k] - sv[i, j, k]) / substeps + \
								  dts * tnd_sv[i, j, k]

			outs.append(out_sv)

		if qv is not None:
			out_qv = gt.Equation()

			if tnd_qv is None:
				out_qv[i, j, k] = tmp_qv[i, j, k] + \
								  (stage_qv[i, j, k] - qv[i, j, k]) / substeps
			else:
				out_qv[i, j, k] = tmp_qv[i, j, k] + \
								  (stage_qv[i, j, k] - qv[i, j, k]) / substeps + \
								  dts * tnd_qv[i, j, k]

			outs.append(out_qv)

		if qc is not None:
			out_qc = gt.Equation()

			if tnd_qc is None:
				out_qc[i, j, k] = tmp_qc[i, j, k] + \
								  (stage_qc[i, j, k] - qc[i, j, k]) / substeps
			else:
				out_qc[i, j, k] = tmp_qc[i, j, k] + \
								  (stage_qc[i, j, k] - qc[i, j, k]) / substeps + \
								  dts * tnd_qc[i, j, k]

			outs.append(out_qc)

		if qr is not None:
			out_qr = gt.Equation()

			if tnd_qr is None:
				out_qr[i, j, k] = tmp_qr[i, j, k] + \
								  (stage_qr[i, j, k] - qr[i, j, k]) / substeps
			else:
				out_qr[i, j, k] = tmp_qr[i, j, k] + \
								  (stage_qr[i, j, k] - qr[i, j, k]) / substeps + \
								  dts * tnd_qr[i, j, k]

			outs.append(out_qr)

		return (*outs,)
