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
	get_input_properties
	get_output_properties
	TasmaniaDiagnosticComponentComposite
	ConcurrentCoupling
	ParallelSplitting
	SequentialUpdateSplitting
"""
from sympl import \
	DiagnosticComponent, DiagnosticComponentComposite, \
	TendencyComponent, TendencyComponentComposite, \
	ImplicitTendencyComponent, ImplicitTendencyComponentComposite, \
	combine_component_properties

from tasmania.python.core.tendency_steppers import tendencystepper_factory
from tasmania.python.utils.data_utils import add, subtract
from tasmania.python.utils.utils import \
	assert_sequence, check_properties_compatibility, check_property_compatibility


def get_input_properties(
	components_list, component_attribute_name='input_properties',
	consider_diagnostics=True, return_dict=None
):
	# Initialize the return dictionary, i.e., the list of requirements
	return_dict = {} if return_dict is None else return_dict

	# Initialize the properties of the variables which the state will be
	# including after passing it to the call operator
	output_properties = {}

	for component in components_list:
		# Extract the desired property dictionary from the component
		component_dict = getattr(component, component_attribute_name)

		# Ensure the requirements of the component are compatible
		# with the variables already at disposal
		check_properties_compatibility(
			output_properties, component_dict,
			properties1_name='{} of {}'.format(
				component_attribute_name,
				getattr(component, 'name', str(component.__class__))
			),
			properties2_name='output_properties'
		)

		# Check if there exists any variable which the component
		# requires but which is not yet at disposal
		not_at_disposal = \
			set(component_dict.keys()).difference(output_properties.keys())

		for name in not_at_disposal:
			# Add the missing variable to the requirements and
			# to the output state
			return_dict[name] = {}
			return_dict[name].update(component_dict[name])
			output_properties[name] = {}
			output_properties[name].update(component_dict[name])

		if consider_diagnostics:
			# Use the diagnostics calculated by the component to update
			# the properties of the output variables
			for name, properties in component.diagnostic_properties.items():
				if name not in output_properties.keys():
					output_properties[name] = {}
				else:
					check_property_compatibility(
						output_properties[name], properties,
						property_name=name,
						origin1_name='output_properties',
						origin2_name='diagnostic_properties of {}'.format(
							getattr(component, 'name', str(component.__class__))
						)
					)

				output_properties[name].update(properties)

	return return_dict


def get_output_properties(
	components_list, component_attribute_name='input_properties',
	consider_diagnostics=True, return_dict=None
):
	"""
	Ansatz: the output property dictionary of a :class:`sympl.TendencyStepper`
	component is a subset of its input property component.
	"""
	# Initialize the return dictionary
	return_dict = {} if return_dict is None else return_dict

	for component in components_list:
		component_dict = getattr(component, component_attribute_name, None)

		if component_dict is not None:
			# Ensure the requirements of the component are compatible
			# with the variables already at disposal
			check_properties_compatibility(
				return_dict, component_dict,
				properties1_name='return_dict',
				properties2_name='{} of {}'.format(
					component_attribute_name,
					getattr(component, 'name', str(component.__class__))
				)
			)

			# Check if there exists any variable which the component
			# requires but which is not yet at disposal
			not_at_disposal = \
				set(component_dict.keys()).difference(return_dict.keys())

			for name in not_at_disposal:
				# Add the missing variable to the return dictionary
				return_dict[name] = {}
				return_dict[name].update(component_dict[name])

		# Consider the diagnostics calculated by the component to update
		# the return dictionary
		if consider_diagnostics:
			for name, properties in component.diagnostic_properties.items():
				if name not in return_dict.keys():
					return_dict[name] = {}
				else:
					check_property_compatibility(
						return_dict[name], properties,
						property_name=name,
						origin1_name='return_dict',
						origin2_name='diagnostic_properties of {}'.format(
							getattr(component, 'name', str(component.__class__)),
						)
					)

				return_dict[name].update(properties)

	return return_dict


class TasmaniaDiagnosticComponentComposite:
	"""
	Callable class wrapping and chaining a set of :class:`sympl.DiagnosticComponent`.

	Attributes
	----------
	input_properties : dict
		Dictionary whose keys are strings denoting model variables
		which should be present in the input state dictionary, and
		whose values are dictionaries specifying fundamental properties
		(dims, units) of those variables.
    diagnostic_properties : dict
        Dictionary whose keys are strings denoting model variables
        retrieved from the input state dictionary, and whose values
        are dictionaries specifying fundamental properties (dims, units)
        of those variables.
	output_properties : dict
        Dictionary whose keys are strings denoting model variables which
        will be present in the input state dictionary when the call operator
        returns, and whose values are dictionaries specifying fundamental
        properties (dims, units) for those variables.
	"""
	def __init__(self, *args, execution_policy='serial'):
		"""
		Parameters
		----------
		*args :
			The :class:`sympl.Diagnostic`\s to wrap and chain.
		execution_policy : `str`, optional
			String specifying the runtime policy according to which parameterizations
			should be invoked. Either:

				* 'serial', to call the physical packages sequentially,
					and add diagnostics to the current state incrementally;
				* 'as_parallel', to call the physical packages *as* we
					were in a parallel region where each package may be
					assigned to a different thread/process; in other terms,
					we do not rely on the order in which parameterizations
					are passed to this object, and diagnostics are not added
					to the current state before returning.

		"""
		assert_sequence(args, reftype=DiagnosticComponent)
		self._components_list = args

		self.input_properties = get_input_properties(
			self._components_list, consider_diagnostics=execution_policy == 'serial'
		)
		self.diagnostic_properties = combine_component_properties(
			self._components_list, 'diagnostic_properties'
		)

		self._call = self._call_serial if execution_policy == 'serial' \
			else self._call_asparallel

	def __call__(self, state):
		"""
		Retrieve diagnostics from the input state by sequentially calling
		the wrapped :class:`sympl.DiagnosticComponent`\s, and incrementally
		update the input state with those diagnostics.

		Parameters
		----------
		state : dict
			The input model state as a dictionary whose keys are strings denoting
			model variables, and whose values are :class:`sympl.DataArray`\s storing
			data for those variables.

		Return
		------
		dict :
			Dictionary whose keys are strings denoting diagnostic variables,
			and whose values are :class:`sympl.DataArray`\s storing data for
			those variables.
		"""
		return self._call(state)

	def _call_serial(self, state):
		return_dict = {}

		tmp_state = {}
		tmp_state.update(state)

		for component in self._components_list:
			diagnostics = component(tmp_state)
			tmp_state.update(diagnostics)
			return_dict.update(diagnostics)

		return return_dict

	def _call_asparallel(self, state):
		return_dict = {}

		for component in self._components_list:
			diagnostics = component(state)
			return_dict.update(diagnostics)

		return return_dict


class ConcurrentCoupling:
	"""
	Callable class which automates the execution of a bundle of physical
	parameterizations pursuing the *explicit* concurrent coupling strategy.

	Attributes
	----------
	input_properties : dict
        Dictionary whose keys are strings denoting model variables
        which should be present in the input state dictionary, and
        whose values are dictionaries specifying fundamental properties
        (dims, units) of those variables.
    output_properties : dict
        Dictionary whose keys are strings denoting model variables which
        will be present in the input state dictionary when the call operator
        returns, and whose values are dictionaries specifying fundamental
        properties (dims, units) for those variables.
    tendency_properties : dict
        Dictionary whose keys are strings denoting the model variables for
        which tendencies have been computed, and whose values are
        :class:`sympl.DataArray`\s storing the tendencies for those variables.

	References
	----------
	Staniforth, A., N. Wood, and J. C\^ot\'e. (2002). \
		A simple comparison of four physics-dynamics coupling schemes. \
		*Mon. Weather Rev.*, *130*:3129-3135.
	"""
	allowed_diagnostic_type = (
		DiagnosticComponent,
		DiagnosticComponentComposite,
	)
	allowed_tendency_type = (
		TendencyComponent,
		TendencyComponentComposite,
		ImplicitTendencyComponent,
		ImplicitTendencyComponentComposite,
	)
	allowed_component_type = allowed_diagnostic_type + allowed_tendency_type

	def __init__(self, *args, execution_policy='serial'):
		"""
		Parameters
		----------
		*args : obj
			Instances of

				* :class:`sympl.DiagnosticComponent`,
				* :class:`sympl.DiagnosticComponentComposite`,
				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`, or
				* :class:`sympl.ImplicitTendencyComponentComposite`

			representing the components to wrap.
		execution_policy : `str`, optional
			String specifying the runtime mode in which parameterizations
			should be invoked. Either:

				* 'serial', to call the physical packages sequentially,
					and add diagnostics to the current state incrementally;
				* 'as_parallel', to call the physical packages *as* we
					were in a parallel region where each package may be
					assigned to a different thread/process; in other terms,
					we do not rely on the order in which parameterizations
					are passed to this object, and diagnostics are added
					to the current state in a single step just before returning.
		"""
		assert_sequence(args, reftype=self.__class__.allowed_component_type)
		self._component_list = args

		self._policy = execution_policy
		if execution_policy == 'serial':
			self._call = self._call_serial
		else:
			self._call = self._call_asparallel

		# Set properties
		self.input_properties = self._init_input_properties()
		self.output_properties = self._init_output_properties()
		self.tendency_properties = self._init_tendency_properties()

		# Ensure that dimensions and units of the variables present
		# in both input_properties and output_properties are compatible
		# across the two dictionaries
		check_properties_compatibility(
			self.input_properties, self.output_properties,
			properties1_name='input_properties',
			properties2_name='output_properties',
		)

	def _init_input_properties(self):
		flag = self._policy == 'serial'
		return get_input_properties(self._component_list, consider_diagnostics=flag)

	def _init_output_properties(self):
		return get_output_properties(self._component_list)

	def _init_tendency_properties(self):
		tendency_list = tuple(
			c for c in self._component_list
			if isinstance(c, self.__class__.allowed_tendency_type)
		)
		return combine_component_properties(tendency_list, 'tendency_properties')

	@property
	def component_list(self):
		"""
		Return
		------
		tuple :
			The wrapped components.
		"""
		return self._component_list

	def __call__(self, state, timestep=None):
		"""
		Execute the wrapped components to calculate tendencies and retrieve
		diagnostics with the help of the input state.

		Note
		----
		The input state is updated in-place with the computed diagnostics.

		Parameters
		----------
		state : dict
			The input model state as a dictionary whose keys are strings denoting
			model variables, and whose values are :class:`sympl.DataArray`\s storing
			data for those variables.
		timestep : `timedelta`, optional
			The timestep size. Required only if at least one component is an
			instance of :class:`sympl.ImplicitTendencyComponent`.

		Return
		------
		dict :
			Dictionary whose keys are strings denoting the model variables for which
			tendencies have been computed, and whose values are :class:`sympl.DataArray`\s
			storing the tendencies for those variables.
		"""
		tendencies = self._call(state, timestep)

		try:
			tendencies['time'] = state['time']
		except KeyError:
			pass

		return tendencies

	def _call_serial(self, state, timestep):
		"""
		Process the components in 'serial' runtime mode.
		"""
		out_tendencies = {}
		tendency_units = {
			tendency: properties['units']
			for tendency, properties in self.tendency_properties.items()
		}

		for component in self._component_list:
			if isinstance(component, self.__class__.allowed_diagnostic_type):
				diagnostics = component(state)
				state.update(diagnostics)
			else:
				try:
					tendencies, diagnostics = component(state)
				except TypeError:
					tendencies, diagnostics = component(state, timestep)

				out_tendencies.update(
					add(
						out_tendencies, tendencies,
						units=tendency_units, unshared_variables_in_output=True
					)
				)
				state.update(diagnostics)

		return out_tendencies

	def _call_asparallel(self, state, timestep):
		"""
		Process the components in 'as_parallel' runtime mode.
		"""
		out_tendencies = {}
		tendency_units = {
			tendency: properties['units']
			for tendency, properties in self.tendency_properties.items()
		}

		agg_diagnostics = {}

		for component in self._component_list:
			if isinstance(component, self.__class__.allowed_diagnostic_type):
				diagnostics = component(state)
				agg_diagnostics.update(diagnostics)
			else:
				try:
					tendencies, diagnostics = component(state)
				except TypeError:
					tendencies, diagnostics = component(state, timestep)

				out_tendencies.update(
					add(
						out_tendencies, tendencies,
						units=tendency_units, unshared_variables_in_output=True
					)
				)
				agg_diagnostics.update(diagnostics)

		# Update the state with the previously computed diagnostics
		state.update(agg_diagnostics)

		return out_tendencies


class ParallelSplitting:
	"""
	Callable class which integrates a bundle of physical processes pursuing
	the parallel splitting strategy.

	Attributes
	----------
    input_properties : dict
        Dictionary whose keys are strings denoting variables which
        should be present in the input model dictionary representing
        the current state, and whose values are dictionaries specifying
        fundamental properties (dims, units) of those variables.
    provisional_input_properties : dict
        Dictionary whose keys are strings denoting variables which
        should be present in the input model dictionary representing
        the provisional state, and whose values are dictionaries specifying
        fundamental properties (dims, units) of those variables.
	output_properties : dict
        Dictionary whose keys are strings denoting variables which
        will be present in the input model dictionary representing
        the current state when the call operator returns, and
        whose values are dictionaries specifying fundamental properties
        (dims, units) of those variables.
	provisional_output_properties : dict
        Dictionary whose keys are strings denoting variables which
        will be present in the input model dictionary representing
        the provisional state when the call operator returns, and
        whose values are dictionaries specifying fundamental properties
        (dims, units) of those variables.

	References
	----------
    Donahue, A. S., and P. M. Caldwell. (2018). \
        Impact of physics parameterization ordering in a global atmosphere model. \
        *Journal of Advances in Modeling earth Systems*, *10*:481-499.
	"""
	allowed_diagnostic_type = (
		DiagnosticComponent,
		DiagnosticComponentComposite,
	)
	allowed_tendency_type = (
		TendencyComponent,
		TendencyComponentComposite,
		ImplicitTendencyComponent,
		ImplicitTendencyComponentComposite,
	)
	allowed_component_type = allowed_diagnostic_type + allowed_tendency_type

	def __init__(
		self, *args, execution_policy='serial',
		retrieve_diagnostics_from_provisional_state=False
	):
		"""
		Parameters
		----------
		*args : dict
			Dictionaries containing the components to wrap and specifying
			fundamental properties (time_integrator, substeps) of those processes.
			Particularly:

				* 'component' is the

						- :class:`sympl.DiagnosticComponent`
						- :class:`sympl.DiagnosticComponentComposite`
						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`, or
						- :class:`sympl.ImplicitTendencyComponentComposite`

					representing the process;
				* if 'component' is a

						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`, or
						- :class:`sympl.ImplicitTendencyComponentComposite`,

					'time_integrator' is a string specifying the scheme to
					integrate the process forward in time. Either:

                        - 'forward_euler', for the forward Euler scheme;
                        - 'rk2', for the two-stage second-order Runge-Kutta (RK) scheme;
                        - 'rk3cosmo', for the three-stage RK scheme as used in the
                            `COSMO model <http://www.cosmo-model.org>`_; this method is
                            nominally second-order, and third-order for linear problems;
                        - 'rk3', for the three-stages, third-order RK scheme.

                * if 'component' is a

						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`, or
						- :class:`sympl.ImplicitTendencyComponentComposite`,

                	'substeps' represents the number of substeps to carry out
                	to integrate the process. Defaults to 1.

		execution_policy : `str`, optional
			String specifying the runtime mode in which parameterizations
			should be invoked. Either:

				* 'serial' (default), to run the physical packages sequentially,
					and add diagnostics to the current state incrementally;
				* 'as_parallel', to run the physical packages *as* we
					were in a parallel region where each package may be
					assigned to a different thread/process; in other terms,
					we do not rely on the order in which parameterizations
					are passed to this object, and diagnostics are added
					to the current state in a single step just before returning.

		retrieve_diagnostics_from_provisional_state : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) to feed the
			:class:`sympl.DiagnosticComponent` objects with the provisional
			(resp., current) state, and add the so-retrieved diagnostics
			to the provisional (resp., current) state dictionary.
			Defaults to :obj:`False`.
		"""
		self._component_list = []
		self._substeps = []
		for process in args:
			try:
				bare_component = process['component']
			except KeyError:
				msg = "Missing mandatory key ''component'' in one item of ''processes''."
				raise KeyError(msg)

			assert isinstance(bare_component, self.__class__.allowed_component_type), \
				"''component'' value should be either a {}.".format(
					', '.join(str(ctype) for ctype in self.__class__.allowed_component_type)
				)

			if isinstance(bare_component, self.__class__.allowed_diagnostic_type):
				self._component_list.append(bare_component)
				self._substeps.append(1)
			else:
				integrator = process.get('time_integrator', 'forward_euler')
				TendencyStepper = tendencystepper_factory(integrator)
				self._component_list.append(TendencyStepper(bare_component))

				substeps = process.get('substeps', 1)
				self._substeps.append(substeps)

		self._policy = execution_policy
		if execution_policy == 'serial':
			self._call = self._call_serial
		else:
			self._call = self._call_asparallel

		if execution_policy == 'asparallel' and retrieve_diagnostics_from_provisional_state:
			import warnings
			warnings.warn(
				'Argument retrieve_diagnostics_from_provisional_state '
				'only effective when execution policy set on ''serial''.'
			)
			self._diagnostics_from_provisional = False
		else:
			self._diagnostics_from_provisional = retrieve_diagnostics_from_provisional_state

		# Set properties
		self.input_properties = self._init_input_properties()
		self.provisional_input_properties = self._init_provisional_input_properties()
		self.output_properties = self._init_output_properties()
		self.provisional_output_properties = self._init_provisional_output_properties()

		# Ensure that dimensions and units of the variables present
		# in both input_properties and output_properties are compatible
		# across the two dictionaries
		check_properties_compatibility(
			self.input_properties, self.output_properties,
			properties1_name='input_properties',
			properties2_name='output_properties',
		)

		# Ensure that dimensions and units of the variables present
		# in both provisional_input_properties and provisional_output_properties
		# are compatible across the two dictionaries
		check_properties_compatibility(
			self.provisional_input_properties, self.provisional_output_properties,
			properties1_name='provisional_input_properties',
			properties2_name='provisional_output_properties',
		)

	def _init_input_properties(self):
		if not self._diagnostics_from_provisional:
			flag = self._policy == 'serial'
			return get_input_properties(self._component_list, consider_diagnostics=flag)
		else:
			tendencystepper_components = tuple(
				component for component in self._component_list
				if not isinstance(component, self.__class__.allowed_diagnostic_type)
			)
			return get_input_properties(tendencystepper_components, consider_diagnostics=True)

	def _init_provisional_input_properties(self):
		# We require that all prognostic variables affected by the
		# parameterizations are included in the provisional state
		tendencystepper_components = tuple(
			component for component in self._component_list
			if not isinstance(component, self.__class__.allowed_diagnostic_type)
		)
		return_dict = get_input_properties(
			tendencystepper_components, component_attribute_name='output_properties',
			consider_diagnostics=False
		)

		if self._diagnostics_from_provisional:
			diagnostic_components = (
				component for component in self._component_list
				if isinstance(component, self.__class__.allowed_diagnostic_type)
			)

			return_dict.update(
				get_input_properties(
					diagnostic_components, consider_diagnostics=True,
					return_dict=return_dict
				)
			)

		return return_dict

	def _init_output_properties(self):
		if not self._diagnostics_from_provisional:
			return get_output_properties(self._component_list)
		else:
			tendencystepper_components = tuple(
				component for component in self._component_list
				if not isinstance(component, self.__class__.allowed_diagnostic_type)
			)
			return get_output_properties(tendencystepper_components)

	def _init_provisional_output_properties(self):
		return_dict = self.provisional_input_properties

		if self._diagnostics_from_provisional:
			diagnostic_components = (
				component for component in self._component_list
				if isinstance(component, self.__class__.allowed_diagnostic_type)
			)

			return_dict.update(
				get_output_properties(
					diagnostic_components, component_attribute_name='',
					consider_diagnostics=True, return_dict=return_dict
				)
			)

		return return_dict

	@property
	def component_list(self):
		"""
		Return
		------
		tuple :
			The wrapped components.
		"""
		return tuple(self._component_list)

	def __call__(self, state, state_prv, timestep):
		"""
		Advance the model state one timestep forward in time by pursuing
		the parallel splitting method.

		Parameters
		----------
		state : dict
			Model state dictionary representing the model state at the
			current time level, i.e., at the beginning of the current
			timestep. Its keys are strings denoting the model variables,
			and its values are :class:`sympl.DataArray`\s storing data
			for those variables.
		state_prv :
			Model state dictionary representing a provisional model state.
			Ideally, this should be the state output by the dynamical core,
			i.e., the outcome of a one-timestep time integration which
			takes only the dynamical processes into consideration.
			Its keys are strings denoting the model variables, and its values
			are :class:`sympl.DataArray`\s storing data for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the timestep size.

		Note
		----
		:obj:`state` may be modified in-place with the diagnostics retrieved
		from :obj:`state` itself.
		:obj:`state_prv` is modified in-place with the temporary states provided
		by each process. In other words, when this method returns, :obj:`state_prv`
		will represent the state at the next time level.
		"""
		self._call(state, state_prv, timestep)

		# Ensure the provisional state is now defined at the next time level
		state_prv['time'] = state['time'] + timestep

	def _call_serial(self, state, state_prv, timestep):
		"""
		Process the components in 'serial' runtime mode.
		"""
		out_units = {
			name: properties['units'] for name, properties in
			self.provisional_output_properties.items()
		}

		for component, substeps in zip(self._component_list, self._substeps):
			if not isinstance(component, self.__class__.allowed_diagnostic_type):
				diagnostics, state_tmp = component(state, timestep/substeps)

				if substeps > 1:
					state_tmp.update(
						{
							key: value for key, value in state.items()
						 	if key not in state_tmp
						}
					)

					for _ in range(1, substeps):
						_, state_aux = component(state_tmp, timestep/substeps)
						state_tmp.update(state_aux)

				increment = subtract(
					state_tmp, state,
					unshared_variables_in_output=False
				)
				state_prv.update(
					add(
						state_prv, increment,
						units=out_units, unshared_variables_in_output=True
					)
				)

				state.update(diagnostics)
			else:
				if self._diagnostics_from_provisional:
					diagnostics = component(state_prv)
					state_prv.update(diagnostics)
				else:
					diagnostics = component(state)
					state.update(diagnostics)

	def _call_asparallel(self, state, state_prv, timestep):
		"""
		Process the components in 'as_parallel' runtime mode.
		"""
		agg_diagnostics = {}
		out_units = {
			name: properties['units'] for name, properties in
			self.provisional_output_properties.items()
		}

		for component, substeps in zip(self._component_list, self._substeps):
			if not isinstance(component, self.__class__.allowed_diagnostic_type):
				diagnostics, state_tmp = component(state, timestep/substeps)

				if substeps > 1:
					state_tmp.update(
						{
							key: value for key, value in state.items()
							if key not in state_tmp
						}
					)

					for _ in range(1, substeps):
						_, state_aux = component(state_tmp, timestep/substeps)
						state_tmp.update(state_aux)

				increment = subtract(
					state_tmp, state,
					unshared_variables_in_output=False
				)
				state_prv.update(
					add(
						state_prv, increment,
						units=out_units, unshared_variables_in_output=True
					)
				)

				agg_diagnostics.update(diagnostics)
			else:
				diagnostics = component(state)
				agg_diagnostics.update(diagnostics)

		state.update(agg_diagnostics)


class SequentialUpdateSplitting:
	"""
	Callable class which integrates a bundle of physical processes pursuing
	the sequential update splitting strategy.

	Attributes
	----------
	input_properties : dict
        Dictionary whose keys are strings denoting model variables
        which should be present in the input state, and whose values
        are dictionaries specifying fundamental properties (dims, units)
        of those variables.
    output_properties : dict
        Dictionary whose keys are strings denoting model variables
        which will be present in the input state when the call operator
        returns, and whose values are dictionaries specifying fundamental
        properties (dims, units) for those variables.

	References
	----------
    Donahue, A. S., and P. M. Caldwell. (2018). \
        Impact of physics parameterization ordering in a global atmosphere model. \
        *Journal of Advances in Modeling earth Systems*, *10*:481-499.
	"""
	allowed_diagnostic_type = (
		DiagnosticComponent,
		DiagnosticComponentComposite,
	)
	allowed_tendency_type = (
		TendencyComponent,
		TendencyComponentComposite,
		ImplicitTendencyComponent,
		ImplicitTendencyComponentComposite,
	)
	allowed_component_type = allowed_diagnostic_type + allowed_tendency_type

	def __init__(self, *args):
		"""
		Parameters
		----------
		*args : dict
			Dictionaries containing the processes to wrap and specifying
			fundamental properties (time_integrator, substeps) of those processes.
			Particularly:

				* 'component' is the

						- :class:`sympl.DiagnosticComponent`,
						- :class:`sympl.DiagnosticComponentComposite`,
						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
				 		- :class:`sympl.ImplicitTendencyComponent`, or
				 		- :class:`sympl.ImplicitTendencyComponentComposite`

					representing the process;
				* if 'component' is a

						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`, or
						- :class:`sympl.ImplicitTendencyComponentComposite`,

					'time_integrator' is a string specifying the scheme to integrate
					the process forward in time. Either:

                        - 'forward_euler', for the forward Euler scheme;
                        - 'rk2', for the two-stage second-order Runge-Kutta (RK) scheme;
                        - 'rk3cosmo', for the three-stage RK scheme as used in the
                            `COSMO model <http://www.cosmo-model.org>`_; this method is
                            nominally second-order, and third-order for linear problems;
                        - 'rk3', for the three-stages, third-order RK scheme.

				* if 'component' is a

						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`, or
						- :class:`sympl.ImplicitTendencyComponentComposite`,

                	'substeps' represents the number of substeps to carry out to
                	integrate the process. Defaults to 1.
		"""
		self._component_list = []
		self._substeps = []
		for process in args:
			try:
				bare_component = process['component']
			except KeyError:
				msg = "Missing mandatory key ''component'' in one item of ''processes''."
				raise KeyError(msg)

			assert isinstance(bare_component, self.__class__.allowed_component_type), \
				"''component'' value should be either a {}.".format(
					', '.join(str(ctype) for ctype in self.__class__.allowed_component_type)
				)

			if isinstance(bare_component, self.__class__.allowed_diagnostic_type):
				self._component_list.append(bare_component)
				self._substeps.append(1)
			else:
				integrator = process.get('time_integrator', 'forward_euler')
				TendencyStepper = tendencystepper_factory(integrator)
				self._component_list.append(TendencyStepper(bare_component))

				substeps = process.get('substeps', 1)
				self._substeps.append(substeps)

		# Set properties
		self.input_properties = self._init_input_properties()
		self.output_properties = self._init_output_properties()

		# Ensure that dimensions and units of the variables present
		# in both input_properties and output_properties are compatible
		# across the two dictionaries
		check_properties_compatibility(
			self.input_properties, self.output_properties,
			properties1_name='input_properties',
			properties2_name='output_properties',
		)

	def _init_input_properties(self):
		return get_input_properties(self._component_list, consider_diagnostics=True)

	def _init_output_properties(self):
		return get_output_properties(self._component_list, consider_diagnostics=True)

	def __call__(self, state, timestep):
		"""
		Advance the model state one timestep forward in time by pursuing
		the parallel splitting method.

		Parameters
		----------
		state : dict
			Model state dictionary representing the model state to integrate.
			Its keys are strings denoting the model variables, and its values
			are :class:`sympl.DataArray`\s storing data for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the timestep size.

		Note
		----
		:obj:`state` is modified in-place to represent the final model state.
		"""
		current_time = state['time']

		for component, substeps in zip(self._component_list, self._substeps):
			if not isinstance(component, self.__class__.allowed_diagnostic_type):
				diagnostics, state_tmp = component(state, timestep/substeps)

				if substeps > 1:
					state_tmp.update(
						{
							key: value for key, value in state.items()
							if key not in state_tmp
						}
					)

					for _ in range(1, substeps):
						_, state_aux = component(state_tmp, timestep/substeps)
						state_tmp.update(state_aux)

				state.update(state_tmp)
				state.update(diagnostics)
			else:
				diagnostics = component(state)
				state.update(diagnostics)

			# Ensure state is still defined at current time level
			state['time'] = current_time

		# Ensure the state is defined at the next time level
		state['time'] = current_time + timestep
