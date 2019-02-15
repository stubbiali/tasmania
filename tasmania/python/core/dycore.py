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
from sympl import \
	DiagnosticComponent, DiagnosticComponentComposite as SymplDiagnosticComponentComposite, \
	TendencyComponent, TendencyComponentComposite, \
	ImplicitTendencyComponent, ImplicitTendencyComponentComposite
from sympl._core.base_components import \
	InputChecker, TendencyChecker as _TendencyChecker, OutputChecker

from tasmania.python.core.composite import \
	DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite
from tasmania.python.core.concurrent_coupling import ConcurrentCoupling
from tasmania.python.utils.data_utils import make_state, make_raw_state
from tasmania.python.utils.dict_utils import add
from tasmania.python.utils.framework_utils import \
	check_properties_compatibility, check_missing_properties


class TendencyChecker(_TendencyChecker):
	def __init__(self, component):
		super().__init__(component)

	def check_tendencies(self, tendency_dict):
		__tendency_dict = {
			key: value for key, value in tendency_dict.items() if key != 'time'
		}
		self._check_extra_tendencies(__tendency_dict)


class DynamicalCore:
	"""     
	Abstract base class representing a generic dynamical core based on the
	partial time-splitting method. Specification of the input requirements, 
	implementation of the differential operators, and definition of the 
	output properties are delegated to the derived classes.
	"""
	allowed_tendency_type = (
		TendencyComponent,
		TendencyComponentComposite,
		ImplicitTendencyComponent,
		ImplicitTendencyComponentComposite,
		ConcurrentCoupling,
	)
	allowed_diagnostic_type = (
		DiagnosticComponent,
		SymplDiagnosticComponentComposite,
		TasmaniaDiagnosticComponentComposite,
	)

	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(
		self, grid, time_units='s',
		intermediate_tendencies=None, intermediate_diagnostics=None,
		substeps=0, fast_tendencies=None, fast_diagnostics=None
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
		"""
		self._grid, self._tunits = grid, time_units

		self._inter_tends = intermediate_tendencies
		if self._inter_tends is not None:
			ttype = self.__class__.allowed_tendency_type
			assert isinstance(self._inter_tends, ttype), \
				"The input argument ''intermediate_tendencies'' " \
				"should be an instance of either {}.".format(
					', '.join(str(item) for item in ttype)
				)

		self._inter_diags = intermediate_diagnostics
		if self._inter_diags is not None:
			dtype = self.__class__.allowed_diagnostic_type
			assert isinstance(self._inter_diags, dtype), \
				"The input argument ''intermediate_diagnostics'' " \
				"should be an instance of either {}.".format(
					', '.join(str(item) for item in dtype)
				)

		self._substeps = substeps if substeps >= 0 else 0

		if self._substeps >= 0:
			self._fast_tends = fast_tendencies
			if self._fast_tends is not None:
				ttype = self.__class__.allowed_tendency_type
				assert isinstance(self._fast_tends, ttype), \
					"The input argument ''fast_tendencies'' " \
					"should be an instance of either {}.".format(
						', '.join(str(item) for item in ttype)
					)

			self._fast_diags = fast_diagnostics
			if self._fast_diags is not None:
				dtype = self.__class__.allowed_diagnostic_type
				assert isinstance(self._fast_diags, dtype), \
					"The input argument ''fast_diagnostics'' " \
					"should be an instance of either {}.".format(
						', '.join(str(item) for item in dtype)
					)

		# Initialize properties
		self.input_properties = self._init_input_properties()
		self.tendency_properties = self._init_tendency_properties()
		self.output_properties = self._init_output_properties()

		# Instantiate checkers
		self._input_checker    = InputChecker(self)
		self._tendency_checker = TendencyChecker(self)
		self._output_checker   = OutputChecker(self)

	def ensure_internal_consistency(self):
		"""
		Perform some controls aiming to verify internal consistency.
		In more detail:

			* #1: variables contained in both `_input_properties` and
				`_output_properties` should have compatible properties
				across the two dictionaries;
			* #2: variables contained in both `_substep_input_properties` and
				`_substep_output_properties` should have compatible properties
				across the two dictionaries;
			* #3: variables contained in both `_input_properties` and the 
				`input_properties` dictionary of `intermediate_tendencies` 
				should have compatible properties across the two dictionaries;
			* #4: dimensions and units of the variables diagnosed by
				`intermediate_tendencies` should be compatible with
				the dimensions and units specified in `_input_properties`;
			* #5: any intermediate tendency calculated by `intermediate_tendencies` 
				should be present in the `_tendency_properties` dictionary, 
				with compatible dimensions and units;
			* #6: dimensions and units of the variables diagnosed by
				`intermediate_tendencies` should be compatible with
				the dimensions and units specified in the `input_properties`
				dictionary of `fast_tendencies`, or the `_substep_input_properties`
				dictionary if `fast_tendencies` is not given;
			* #7: variables diagnosed by `fast_tendencies` should have dimensions
				and units compatible with those specified in the
				`_substep_input_properties` dictionary;
			* #8: variables contained in `_output_properties` for which
				`fast_tendencies` calculates a (fast) tendency should
				 have dimensions and units compatible with those specified
				 in the `tendency_properties` dictionary of `fast_tendencies`;
			* #9: any fast tendency calculated by `fast_tendencies`
				should be present in the `_substep_tendency_properties`
				dictionary, with compatible dimensions and units;
			* #10: any variable for which the `fast_tendencies`
				calculates a (fast) tendency should be present both in
				the `_substep_input_property` and `_substep_output_property`
				dictionaries, with compatible dimensions and units;
			* #11: any variable being expected by `fast_diagnostics` should be
				present in `_substep_output_properties`, with compatible
				dimensions and units;
			* #12: any variable being expected by `intermediate_diagnostics`
				should be present either in `_output_properties` or
				`_substep_output_properties`, with compatible dimensions
				and units.
		"""
		#============================================================
		# Check #1
		#============================================================
		check_properties_compatibility(
			self._input_properties, self._output_properties,
			properties1_name='_input_properties',
			properties2_name='_output_properties',
		)

		#============================================================
		# Check #2
		#============================================================
		check_properties_compatibility(
			self._substep_input_properties, self._substep_output_properties,
			properties1_name='_substep_input_properties',
			properties2_name='_substep_output_properties',
		)

		#============================================================
		# Check #3
		#============================================================
		if self._inter_tends is not None:
			check_properties_compatibility(
				self._inter_tends.input_properties, self._input_properties,
				properties1_name='intermediate_tendencies.input_properties',
				properties2_name='_input_properties',
			)

		#============================================================
		# Check #4
		#============================================================
		if self._inter_tends is not None:
			check_properties_compatibility(
				self._inter_tends.diagnostic_properties, self._input_properties,
				properties1_name='intermediate_tendencies.diagnostic_properties',
				properties2_name='_input_properties',
			)

		#============================================================
		# Check #5
		#============================================================
		if self._inter_tends is not None: 
			check_properties_compatibility(
				self._inter_tends.tendency_properties, self._tendency_properties,
				properties1_name='intermediate_tendencies.tendency_properties',
				properties2_name='_tendency_properties',
			)

			check_missing_properties(
				self._inter_tends.tendency_properties, self._tendency_properties,
				properties1_name='intermediate_tendencies.tendency_properties',
				properties2_name='_tendency_properties',
			)

		#============================================================
		# Check #6
		#============================================================
		if self._inter_tends is not None:
			if self._fast_tends is not None:
				check_properties_compatibility(
					self._inter_tends.diagnostic_properties, 
					self._fast_tends.input_properties,
					properties1_name='intermediate_tendencies.diagnostic_properties',
					properties2_name='fast_tendencies.input_properties',
				)
			else:
				check_properties_compatibility(
					self._inter_tends.diagnostic_properties, 
					self._substep_input_properties,
					properties1_name='intermediate_tendencies.diagnostics_properties',
					properties2_name='_substep_input_properties',
				)

		#============================================================
		# Check #7
		#============================================================
		if self._fast_tends is not None:
			check_properties_compatibility(
				self._fast_tends.diagnostics_properties, self._substep_input_properties,
				properties1_name='fast_tendencies.diagnostic_properties',
				properties2_name='_substep_input_properties',
			)

		#============================================================
		# Check #8
		#============================================================
		if self._fast_tends is not None:
			check_properties_compatibility(
				self._fast_tends.tendency_properties, self._output_properties,
				to_append=self._tunits,
				properties1_name='fast_tendencies.tendency_properties',
				properties2_name='_output_properties',
			)

		#============================================================
		# Check #9
		#============================================================
		if self._fast_tends is not None:
			check_properties_compatibility(
				self._fast_tends.tendency_properties, self._substep_tendency_properties,
				to_append=self._tunits,
				properties1_name='fast_tendencies.tendency_properties',
				properties2_name='_substep_tendency_properties',
			)

			check_missing_properties(
				self._fast_tends.tendency_properties, self._substep_tendency_properties,
				properties1_name='fast_tendencies.tendency_properties',
				properties2_name='_substep_tendency_properties',
			)

		#============================================================
		# Check #10
		#============================================================
		if self._fast_tends is not None:
			check_properties_compatibility(
				self._fast_tends.tendency_properties, self._substep_input_properties,
				to_append=self._tunits,
				properties1_name='fast_tendencies.tendency_properties',
				properties2_name='_substep_input_properties',
			)

			check_missing_properties(
				self._fast_tends.tendency_properties, self._substep_input_properties,
				properties1_name='fast_tendencies.tendency_properties',
				properties2_name='_substep_input_properties',
			)

			check_properties_compatibility(
				self._fast_tends.tendency_properties, self._substep_output_properties,
				to_append=self._tunits,
				properties1_name='fast_tendencies.tendency_properties',
				properties2_name='_substep_input_properties',
			)

			check_missing_properties(
				self._fast_tends.tendency_properties, self._substep_output_properties,
				properties1_name='fast_tendencies.tendency_properties',
				properties2_name='_substep_output_properties',
			)

		#============================================================
		# Check #11
		#============================================================
		if self._fast_diags is not None:
			check_properties_compatibility(
				self._fast_diags.input_properties, self._substep_output_properties,
				properties1_name='fast_diagnostics.input_properties',
				properties2_name='_substep_output_properties',
			)

			check_missing_properties(
				self._fast_diags.input_properties, self._substep_output_properties,
				properties1_name='fast_diagnostics.input_properties',
				properties2_name='_substep_output_properties',
			)

		#============================================================
		# Check #12
		#============================================================
		if self._inter_diags is not None:
			fused_output_properties = {}
			fused_output_properties.update(self._output_properties)
			fused_output_properties.update(self._substep_output_properties)

			check_properties_compatibility(
				self._inter_diags.input_properties, fused_output_properties,
				properties1_name='intermediate_diagnostics.input_properties',
				properties2_name='fused_output_properties',
			)

			check_missing_properties(
				self._inter_diags.input_properties, fused_output_properties,
				properties1_name='intermediate_diagnostics.input_properties',
				properties2_name='fused_output_properties',
			)

	def ensure_input_output_consistency(self):
		"""
		Perform some controls aiming to verify input-output consistency.
		In more detail:

			* #1: variables contained in both `input_properties` and
				`output_properties` should have compatible properties
				across the two dictionaries;
			* #2: in case of a multi-stage dynamical core, any variable
				present in `output_properties` should be also contained
				in `input_properties`.
		"""
		#============================================================
		# Safety-guard preamble
		#============================================================
		assert hasattr(self, 'input_properties'), \
			'Hint: did you call _initialize_input_properties?'
		assert hasattr(self, 'output_properties'), \
			'Hint: did you call _initialize_output_properties?'

		#============================================================
		# Check #1
		#============================================================
		check_properties_compatibility(
			self.input_properties, self.output_properties,
			properties1_name='input_properties',
			properties2_name='output_properties',
		)

		#============================================================
		# Check #2
		#============================================================
		if self.stages > 1:
			check_missing_properties(
				self.output_properties, self.input_properties,
				properties1_name='output_properties',
				properties2_name='input_properties',
			)

	def _init_input_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which
			should be included in the input state, and whose values
			are fundamental properties (dims, units) of those variables.
			This dictionary results from fusing the requirements
			specified by the user via
			:meth:`~tasmania.dynamics.dycore.DynamicalCore._input_properties` and
			:meth:`~tasmania.dynamics.dycore.DynamicalCore._substep_input_properties`,
			with the :obj:`input_properties` and :obj:`output_properties`
			dictionaries of the internal attributes representing the
			intermediate and fast tendency components (if set).
		"""
		return_dict = {}

		if self._inter_tends is None:
			return_dict.update(self._input_properties)
		else:
			return_dict.update(self._inter_tends.input_properties)
			inter_params_diag_properties = self._inter_tends.diagnostic_properties
			stage_input_properties = self._input_properties

			# Add to the requirements the variables to feed the stage with
			# and which are not output by the intermediate parameterizations
			unshared_vars = tuple(
				name for name in stage_input_properties
				if not (name in inter_params_diag_properties or name in return_dict)
			)
			for name in unshared_vars:
				return_dict[name] = {}
				return_dict[name].update(stage_input_properties[name])

		if self._substeps >= 0:
			fast_params_input_properties = \
				{} if self._fast_tends is None else self._fast_tends.input_properties
			fast_params_diag_properties = \
				{} if self._fast_tends is None else self._fast_tends.diagnostic_properties

			# Add to the requirements the variables to feed the fast
			# parameterizations with
			unshared_vars = tuple(
				name for name in fast_params_input_properties
				if name not in return_dict
			)        
			for name in unshared_vars:
				return_dict[name] = {}
				return_dict[name].update(fast_params_input_properties[name])

			# Add to the requirements the variables to feed the sub-step with
			# and which are not output by the either the intermediate parameterizations
			# or the fast parameterizations
			unshared_vars = tuple(
				name for name in self._substep_input_properties
				if not (name in fast_params_diag_properties or name in return_dict)
			)
			for name in unshared_vars:
				return_dict[name] = {}
				return_dict[name].update(self._substep_input_properties[name])

		return return_dict

	@property
	@abc.abstractmethod
	def _input_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which
			should be included in any state passed to the stages, and
			whose values are fundamental properties (dims, units, alias)
			of those variables.
		"""

	@property
	@abc.abstractmethod
	def _substep_input_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which
			should be included in any state passed to the substeps
			carrying out the sub-stepping routine, and whose values are
			fundamental properties (dims, units, alias) of those variables.
		"""

	def _init_tendency_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting (slow) tendencies which
			may (or may not) be passed to the call operator, and whose
			values are fundamental properties (dims, units) of those
			tendencies. This dictionary results from fusing the requirements
			specified by the user via
			:meth:`~tasmania.DynamicalCore._tendency_properties`
			with the :obj:`tendency_properties` dictionary of the internal
			attribute representing the intermediate tendency component (if set).
		"""
		return_dict = {}

		if self._inter_tends is None:
			return_dict.update(self._tendency_properties)
		else:
			return_dict.update(self._inter_tends.tendency_properties)

			# Add to the requirements on the input slow tendencies those
			# tendencies to feed the dycore with and which are not provided
			# by the intermediate parameterizations
			unshared_vars = tuple(
				name for name in self._tendency_properties
				if name not in return_dict
			)
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
			Dictionary whose keys are strings denoting (intermediate
			and slow) tendencies which may (or may not) be passed to
			the stages, and whose values are fundamental properties
			(dims, units, alias) of those tendencies.
		"""

	@property
	@abc.abstractmethod
	def _substep_tendency_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting (intermediate
			and slow) tendencies which may (or may not) be passed to
			the substeps, and whose values are fundamental properties
			(dims, units, alias) of those tendencies.
		"""

	def _init_output_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which are
			included in the output state, and whose values are fundamental
			properties (dims, units) of those variables. This dictionary
			results from fusing the requirements specified by the user via
			:meth:`~tasmania.dynamics.dycore.DynamicalCore._output_properties` and
			:meth:`~tasmania.dynamics.dycore.DynamicalCore._substep_output_properties`
			with the `diagnostic_properties` dictionary of the internal
			attributes representing the intermediate and fast diagnostic
			components (if set).
		"""
		return_dict = {}

		if self._substeps == 0:
			# Add to the return dictionary the variables included in
			# the state output by a stage
			return_dict.update(self._output_properties)
		else:
			# Add to the return dictionary the variables included in
			# the state output by a sub-step
			return_dict.update(self._substep_output_properties)

			if self._fast_diags is not None:
				# Add the fast diagnostics to the return dictionary
				for name, properties in self._fast_diags.diagnostic_properties.items():
					return_dict[name] = {}
					return_dict[name].update(properties)

			# Add to the return dictionary the non-sub-stepped variables
			return_dict.update(self._output_properties)

		if self._inter_diags is not None:
			# Add the retrieved diagnostics to the return dictionary
			for name, properties in self._inter_diags.diagnostic_properties.items():
				return_dict[name] = {}
				return_dict[name].update(properties)

		return return_dict

	@property
	@abc.abstractmethod
	def _output_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which are
			included in the output state returned by any stage,	and whose
			values are fundamental properties (dims, units) of those variables.
		"""

	@property
	@abc.abstractmethod
	def _substep_output_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting variables which are
			included in the output state returned by any sub-step, and whose
			values are fundamental properties (dims, units) of those variables.
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

	@property
	@abc.abstractmethod
	def substep_fractions(self):
		"""
		Return
		------
		float or tuple :
			For each stage, fraction of the total number of sub-steps
			(specified at instantiation) to carry out.
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
			Dictionary whose keys are strings denoting input (slow) tendencies,
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

		Note
		----
		Currently, variable aliasing is not supported.
		"""
		#============================================================
		# Preamble
		#============================================================
		# Ensure the input state and tendency dictionaries contain all the
		# required variables in proper units and dimensions
		self._input_checker.check_inputs(state)
		self._tendency_checker.check_tendencies(tendencies)

		# Initialize the latest state
		out_state = {}
		out_state.update(state)

		# Initialize the dictionary collecting the dictionaries
		tends = {}

		for stage in range(self.stages):
			#============================================================
			# Calculating intermediate tendencies
			#============================================================
			if self._inter_tends is None and stage == 0:
				# Collect the slow tendencies
				tends.update(tendencies)
			elif self._inter_tends is not None:
				# Calculate the intermediate tendencies 
				try:
					inter_tends, diags = self._inter_tends(out_state)
				except TypeError:
					inter_tends, diags = self._inter_tends(out_state, timestep)

				# Sum up the slow and intermediate tendencies
				tends.update(
					add(
						inter_tends, tendencies,
						unshared_variables_in_output=True
					)
				)

				# Update the state with the just computed diagnostics
				out_state.update(diags)

			#============================================================
			# Stage pre-processing
			#============================================================
			# Extract numpy arrays from state
			out_state_units = {
				name: self._input_properties[name]['units']
				for name in self._input_properties
			}
			raw_out_state = make_raw_state(out_state, units=out_state_units)

			# Extract numpy arrays from tendencies
			tends_units = {
				name: self._tendency_properties[name]['units']
				for name in self._tendency_properties
			}
			raw_tends = make_raw_state(tends, units=tends_units)

			#============================================================
			# Staging
			#============================================================
			# Carry out the stage
			raw_stage_state = self.array_call(
				stage, raw_out_state, raw_tends, timestep
			)

			if self._substeps == 0 or len(self._substep_output_properties) == 0:
				#============================================================
				# Stage post-processing, sub-stepping disabled
				#============================================================
				# Create dataarrays out of the numpy arrays contained in the stepped state
				stage_state_units = {
					name: self._output_properties[name]['units']
					for name in self._output_properties
				}
				stage_state = make_state(
					raw_stage_state, self._grid, units=stage_state_units
				)

				# Update the latest state
				out_state = {}
				out_state.update(stage_state)
			else:
				#============================================================
				# Stage post-processing, sub-stepping enabled
				#============================================================
				# Create dataarrays out of the numpy arrays contained in the stepped state
				# which represent variables which will not be affected by the sub-stepping
				raw_nosubstep_stage_state = {
					name: raw_stage_state[name]
					for name in raw_stage_state
					if name not in self._substep_output_properties
				}
				nosubstep_stage_state_units = {
					name: self._output_properties[name]['units']
					for name in self._output_properties
					if name not in self._substep_output_properties
				}
				nosubstep_stage_state = make_state(
					raw_nosubstep_stage_state, self._grid, units=nosubstep_stage_state_units
				)

				substep_frac = 1.0 if self.stages == 1 else self.substep_fractions[stage]
				substeps = int(substep_frac * self._substeps)
				for substep in range(substeps):
					#============================================================
					# Calculating fast tendencies
					#============================================================
					if self._fast_tends is None:
						tends = {}
					else:
						try:
							tends, diags = self._fast_tends(out_state)
						except TypeError:
							tends, diags = \
								self._fast_tends(out_state, timestep/self._substeps)

						out_state.update(diags)

					#============================================================
					# Sub-step pre-processing
					#============================================================
					# Extract numpy arrays from the latest state
					out_state_units = {
						name: self._substep_input_properties[name]['units']
						for name in self._substep_input_properties.keys()
					}
					raw_out_state = make_raw_state(out_state, units=out_state_units)

					# Extract numpy arrays from fast tendencies
					tends_units = {
						name: self._substep_tendency_properties[name]['units']
						for name in self._substep_tendency_properties.keys()
					}
					raw_tends = make_raw_state(tends, units=tends_units)

					#============================================================
					# Sub-stepping
					#============================================================
					# Carry out the sub-step
					raw_substep_state = self.substep_array_call(
						stage, substep, state, raw_stage_state, raw_out_state,
						raw_tends, timestep
					)

					#============================================================
					# Sub-step post-processing
					#============================================================
					# Create dataarrays out of the numpy arrays contained in sub-stepped state
					substep_state_units = {
						name: self._substep_output_properties[name]['units']
						for name in self._substep_output_properties
					}
					substep_state = make_state(
						raw_substep_state, self._grid, units=substep_state_units
					)

					#============================================================
					# Retrieving fast diagnostics
					#============================================================
					if self._fast_diags is not None:
						fast_diags = self._fast_diags(substep_state)
						substep_state.update(fast_diags)

					# Update the output state
					if substep < substeps-1:
						out_state.update(substep_state)
					else:
						out_state = {}
						out_state.update(substep_state)

				#============================================================
				# Including non-sub-stepped variables
				#============================================================
				out_state.update(nosubstep_stage_state)

			#============================================================
			# Retrieving intermediate diagnostics
			#============================================================
			if self._inter_diags is not None:
				inter_diags = self._inter_diags(out_state)
				out_state.update(inter_diags)

			# Ensure the time specified in the output state is correct
			if stage == self.stages-1:
				out_state['time'] = state['time'] + timestep
			else:
				out_state['time'] = out_state['time']

			#============================================================
			# Final checks
			#============================================================
			self._output_checker.check_outputs({
				name: out_state[name] for name in out_state if name != 'time'
			})

		return out_state

	@abc.abstractmethod
	def array_call(self, stage, raw_state, raw_tendencies, timestep):
		"""
		Parameters
		----------
		stage : int
			The stage we are in.
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

	@abc.abstractmethod
	def substep_array_call(
		self, stage, substep, raw_state, raw_stage_state, raw_tmp_state,
		raw_tendencies, timestep
	):
		"""
		Parameters
		----------
		stage : int
			The stage we are in.
		substep : int
			The sub-step we are in.
		raw_state : dict
			The raw state at the current *main* time level, i.e.,
			the raw version of the state dictionary passed to the call operator.
		raw_stage_state : dict
			The (raw) state dictionary returned by the latest stage.
		raw_tmp_state : dict
			The raw state to sub-step.
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
