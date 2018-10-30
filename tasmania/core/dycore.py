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
	TendencyChecker
	DynamicalCore
"""
import abc
from sympl import ComponentMissingOutputError
from sympl._core.base_components import InputChecker, \
										TendencyChecker as _TendencyChecker, \
										OutputChecker

from tasmania.core.physics_composite import ConcurrentCoupling, \
											DiagnosticComponentComposite, \
											PhysicsComponentComposite
from tasmania.utils.data_utils import add, make_state, make_raw_state
from tasmania.utils.utils import check_property_compatibility


class TendencyChecker(_TendencyChecker):
	def __init__(self, component):
		super().__init__(component)

	def check_tendencies(self, tendency_dict):
		self._check_extra_tendencies(tendency_dict)


class DynamicalCore:
	"""
	Abstract base class whose derived classes implement different dynamical cores.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, grid, moist_on, 
				 intermediate_parameterizations=None, diagnostics=None):
		"""
		Constructor.

		Parameters
		----------
		grid : obj 
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`,
			or one of its derived classes.
		moist_on : bool
			:obj:`True` for a moist dynamical core, :obj:`False` otherwise.
		intermediate_parameterizations : `obj`, None
			Instance of
			:class:`~tasmania.core.physics_composite.ConcurrentCoupling`
			object, wrapping the intermediate physical parameterizations.
			Here, *intermediate* refers to the fact that these parameterizations
			are evaluated *before* each stage of the dynamical core.
			In essence, feeding the dynamical core with intermediate
			parameterizations allows to pursue the concurrent splitting strategy
			when an explicit time marching scheme is employed.
		diagnostics : `obj`, None
			:class:`~tasmania.core.physics_composite.DiagnosticComponentComposite` 
			object  wrapping a set of diagnostic parameterizations, evaluated
			at the end of each stage of the dynamical core.
		"""
		self._grid, self._moist_on = grid, moist_on

		self._params = intermediate_parameterizations
		if self._params is not None:
			assert isinstance(self._params, ConcurrentCoupling), \
				"""The input argument ''intermediate_parameterizations'' 
				   should be an instance of ConcurrentCoupling."""

		self._diags = diagnostics
		if self._diags is not None:
			assert isinstance(self._diags, DiagnosticComponentComposite), \
				"""The input argument ''diagnostics'' 
				   should be an instance of DiagnosticComponentComposite."""

		self._input_checker    = InputChecker(self)
		self._tendency_checker = TendencyChecker(self)
		self._output_checker   = OutputChecker(self)

	@property
	def input_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which
			should be included in any input state, and whose values
			are fundamental properties (dims, units) of those variables.
			This dictionary results from fusing the requirements
			specified by the user via
			:meth:`~tasmania.dynamics.dycore.DynamicalCore._input_properties`,
			with the :obj:`input_properties` and :obj:`output_properties`
			dictionaries of the internal
			:class:`~tasmania.core.physics_composite.ConcurrentCoupling`
			attribute (if set).
		"""
		return_dict = {}

		if self._params is None:
			return_dict.update(self._input_properties)
		else:
			return_dict.update(self._params.current_state_input_properties)
			params_output_properties = self._params.current_state_output_properties
			dycore_input_properties  = self._input_properties

			# Ensure that the units and dimensions of the variables output
			# by the intermediate parameterizations are compatible with the
			# units and dimensions being expected by the dycore
			s = set(dycore_input_properties.keys())
			shared_vars = s.intersection(params_output_properties.keys())
			for name in shared_vars:
				check_property_compatibility(params_output_properties[name],
											 dycore_input_properties[name],
											 name=name)

			# Add to the requirements the variables to feed the dycore with
			# and which are not output by the intermediate parameterizations
			s = set(dycore_input_properties.keys())
			unshared_vars = s.difference(params_output_properties.keys())
			for name in unshared_vars:
				return_dict[name] = {}
				return_dict[name].update(dycore_input_properties[name])

		return return_dict

	@property
	@abc.abstractmethod
	def _input_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which
			should be included in any state passed to the stages performing
			the timestepping, and whose values are fundamental properties
			(dims, units) of those variables.
		"""

	@property
	def tendency_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting tendencies which
			may (or may not) be passed to the call operator, and whose
			values are fundamental properties (dims, units) of those
			tendencies. This dictionary results from fusing the requirements
			specified by the user via
			:meth:`~tasmania.dynamics.dycore.DynamicalCore._tendency_properties`
			with the :obj:`tendency_properties` dictionary of the internal
			:class:`~tasmania.core.physics_composite.ConcurrentCoupling`
			object (if set).
		"""
		return_dict = {}

		if self._params is None:
			return_dict.update(self._tendency_properties)
		else:
			return_dict.update(self._params.tendency_properties)

			# Ensure that the units and dimensions of the tendencies output
			# by the intermediate parameterizations are compatible with the
			# units and dimensions being expected by the dycore
			s = set(self._tendency_properties.keys())
			shared_vars = s.intersection(return_dict.keys())
			for name in shared_vars:
				check_property_compatibility(self._tendency_properties[name],
											 return_dict[name],
											 name=name)

			# Add to the requirements on the input slow tendencies those
			# tendencies to feed the dycore with and which are not provided
			# by the intermediate parameterizations
			s = set(self._tendency_properties.keys())
			unshared_vars = s.difference(return_dict.keys())
			for name in unshared_vars:
				return_dict[name] = {}
				return_dict[name].update(self._tendency_properties[name])

		return return_dict

	@property
	@abc.abstractmethod
	def _tendency_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting tendencies which
			may (or may not) be passed to the stages performing the
			timestepping, and whose values are fundamental properties
			(dims, units) of those tendencies.
		"""

	@property
	def output_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which are
			included in the output state, and whose values are fundamental
			properties (dims, units) of those variables.

		Raises
		------
		ComponentMissingOutputError :
			For a multi-stage dycore, if the output state misses some
			variables required by the intermediate parameterizations.
		ComponentMissingOutputError :
			If the state feeding the internal 
			:class:`sympl.DiagnosticComponentComposite` misses some
			variables required to retrieve the diagnostics.
		"""
		# Initialize the return dictionary with the variables included in
		# the output state before retrieving the diagnostics
		return_dict = self._output_properties

		if self._diags is not None:
			diags_inputs = self._diags.input_properties

			# Ensure that the variables output by any stage
			# can feed the DiagnosticComponentComposite object
			shared_vars = set(diags_inputs).intersection(return_dict)
			for name in shared_vars:
				check_property_compatibility(diags_inputs[name], return_dict[name],
											 name=name)

			# Ensure that the state output by any stage contains all the 
			# variables required by the DiagnosticComponentComposite object
			missing_vars = set(diags_inputs).difference(return_dict)
			if missing_vars != set():
				raise ComponentMissingOutputError('The state passed to the internal '
												  'DiagnosticComponentComposite should contain '
												  'the variables: {}.'.format(missing_vars))

			# Add the retrieved diagnostics to the return dictionary
			for name in self._diags.diagnostic_properties.keys():
				return_dict[name] = {}
				return_dict[name].update(self._diags.diagnostic_properties[name])

		if self.stages > 1 and self._params is not None:
			params_inputs = self._params.current_state_input_properties

			# Ensure that the variables output by any stage
			# can feed the intermediate parameterizations
			shared_vars = set(params_inputs).intersection(return_dict)
			for name in shared_vars:
				check_property_compatibility(params_inputs[name], return_dict[name],
											 name=name)

			# Ensure that the state output by any stage contains
			# all the variables required by the intermediate parameterizations
			missing_vars = set(params_inputs).difference(return_dict)
			if missing_vars != set():
				raise ComponentMissingOutputError('The output state should contain '
												  'the variables: {}.'.format(missing_vars))

		return return_dict

	@property
	@abc.abstractmethod
	def _output_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which are
			included in the output state before retrieving the diagnostics,
			and whose values are fundamental properties (dims, units) of 
			those variables.
		"""

	@property
	@abc.abstractmethod
	def stages(self):
		"""
		Return
		------
		int :
			Number of stages carried out by the dynamical core.
		"""

	def __call__(self, state, tendencies, timestep):
		"""
		Call operator advancing the input state one timestep forward.

		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting input model variables,
			and whose values are :class:`sympl.DataArray`\s storing values
			for those variables.
		tendencies : dict
			Dictionary whose keys are strings denoting input tendencies,
			and whose values are :class:`sympl.DataArray`\s storing values
			for those tendencies.
		timestep : timedelta
			:class:`datetime.timedelta` representing the timestep size, i.e.,
			the amount of time to step forward.

		Return
		------
		dict :
			Dictionary whose keys are strings denoting output model variables,
			and whose values are :class:`sympl.DataArray`\s storing values
			for those variables.
		"""
		# Ensure the input state contains all the required variables
		# in proper units and dimensions
		self._input_checker.check_inputs(state)

		# Initialize the output state
		out_state = {}
		out_state.update(state)

		# Initialize the dictionary collecting the dictionaries
		in_tendencies = {}

		for stage in range(self.stages):
			# Possibly, call all intermediate parameterizations,
			# and summed up fast and slow tendencies
			if self._params is None:
				in_tendencies.update(tendencies)
			else:
				intermediate_tendencies = self._params(state=out_state)
				in_tendencies.update(add(intermediate_tendencies, tendencies,
									 	 unshared_variables_in_output=True))

			# Extract Numpy arrays from current state
			in_state_units = {name: self._input_properties[name]['units']
							  for name in self._input_properties.keys()}
			raw_in_state = make_raw_state(out_state, units=in_state_units)

			# Extract Numpy arrays from tendencies
			in_tendencies_units = {name: self._tendency_properties[name]['units']
							   	   for name in self._tendency_properties.keys()}
			raw_in_tendencies = make_raw_state(in_tendencies,
											   units=in_tendencies_units)

			# Stepped the model raw state
			raw_out_state = self.array_call(
				stage, raw_in_state, raw_in_tendencies, timestep
			)

			# Create DataArrays out of the Numpy arrays contained in the stepped state
			out_state_units = {name: self.output_properties[name]['units']
							   for name in self.output_properties.keys()}
			out_state = make_state(raw_out_state, self._grid, units=out_state_units)

			# Ensure the time specified in the output state is correct
			if stage == self.stages-1:
				out_state['time'] = state['time'] + timestep
			else:
				out_state['time'] = raw_out_state['time']

			if self._diags is not None:
				# Retrieve the diagnostics, and update the output state
				_ = self._diags(out_state)

			# Ensure the state contains all the required variables
			# in the right dimensions and units
			self._output_checker.check_outputs({name: out_state[name] 
												for name in out_state if name != 'time'})

		return out_state

	@abc.abstractmethod
	def array_call(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Parameters
		----------
		stage : int
			The stage to perform.
		raw_state : dict
			Dictionary whose keys are strings denoting input model
			variables, and whose values are :class:`numpy.ndarray`\s
			storing values for those variables.
		raw_tendencies : dict
			Dictionary whose keys are strings denoting input
			tendencies, and whose values are :class:`numpy.ndarray`\s
			storing values for those tendencies.
		timestep : timedelta
			The timestep size.

		Return
		------
		dict :
			Dictionary whose keys are strings denoting output model
			variables, and whose values are :class:`numpy.ndarray`\s
			storing values for those variables.
		"""

	def update_topography(self, time):
		"""
		Update the underlying (time-dependent) topography.

		Parameters
		----------
		time : obj
			:class:`datetime.timedelta` representing the elapsed simulation time.
		"""
		self._grid.update_topography(time)
