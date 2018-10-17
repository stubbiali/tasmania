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
	tendencystepper_factory
	DiagnosticComponentComposite
	PhysicsComponentComposite
	ConcurrentCoupling
	ParallelSplitting
	SequentialUpdateSplitting
"""
import abc
from sympl import DiagnosticComponent as Diagnostic, \
				  TendencyComponent as Tendency, \
				  combine_component_properties

from tasmania.core.time_steppers import ForwardEuler, \
									   RungeKutta2 as RK2, \
									   RungeKutta3COSMO as RK3COSMO, \
									   RungeKutta3 as RK3
from tasmania.utils.data_utils import add, subtract
from tasmania.utils.utils import assert_sequence, check_property_compatibility


def get_input_properties(components_list, component_attribute_name='input_properties',
						 consider_diagnostics=True, return_dict=None):
	# Initialize the return dictionary, i.e., the list of requirements
	return_dict = {} if return_dict is None else return_dict

	# Initialize the properties of the variables which the state will be
	# including after passing it to the call operator
	output_properties = {}

	for component in components_list:
		# Extract the desired property dictionary from the component
		component_dict = getattr(component, component_attribute_name)

		# Get the set of variables which should be passed to
		# the component, and which are already at disposal
		already_at_disposal = \
			set(component_dict.keys()).intersection(output_properties.keys())

		# Ensure the requirements of the component are compatible
		# with the variables already at disposal
		for name in already_at_disposal:
			check_property_compatibility(output_properties[name],
									 	 component_dict[name],
									 	 name=name)

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
					check_property_compatibility(output_properties[name],
                                                 properties, name=name)

				output_properties[name].update(properties)

	return return_dict


def get_output_properties(components_list, component_attribute_name='input_properties',
						  consider_diagnostics=True):
	"""
	Ansatz: the output property dictionary of a :class:`sympl.TendencyStepper`
	component is a subset of its input property component.
	"""
	# Initialize the return dictionary
	return_dict = {}

	for component in components_list:
		component_dict = getattr(component, component_attribute_name)

		# Get the set of variables which should be passed to
		# the component, and which are already at disposal
		already_at_disposal = \
			set(component_dict.keys()).intersection(return_dict.keys())

		# Ensure the requirements of the component are compatible
		# with the variables already at disposal
		for name in already_at_disposal:
			check_property_compatibility(return_dict[name],
										 component_dict[name],
                                     	 name=name)

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
					check_property_compatibility(return_dict[name],
									 	 	 	 properties, name=name)

				return_dict[name].update(properties)

	return return_dict


def tendencystepper_factory(scheme):
	if scheme == 'forward_euler':
		return ForwardEuler
	elif scheme == 'rk2':
		return RK2
	elif scheme == 'rk3cosmo':
		return RK3COSMO
	elif scheme == 'rk3':
		return RK3
	else:
		raise ValueError('Unsupported time integration scheme ''{}''. '
						 'Available integrators: forward_euler, rk2, rk3cosmo, rk3.'
						 .format(scheme))


class DiagnosticComponentComposite:
	"""
	TODO
	"""
	def __init__(self, *args):
		assert_sequence(args, reftype=Diagnostic)
		self._components_list = args

	@property
	def input_properties(self):
		"""
		TODO
		"""
		return get_input_properties(self._components_list)

	@property
	def diagnostic_properties(self):
		"""
		TODO
		"""
		return combine_component_properties(self._components_list, 
											'diagnostic_properties')

	@property
	def output_properties(self):
		"""
		TODO
		"""
		return get_output_properties(self._components_list)

	def __call__(self, state):
		"""
		TODO
		"""
		return_dict = {}

		for component in self._components_list:
			diagnostics = component(state)
			state.update(diagnostics)
			return_dict.update(diagnostics)

		return return_dict


class PhysicsComponentComposite:
	"""
	Abstract base class whose derived classes automate the
	execution of a set of physical parameterizations, pursuing
	a specific coupling strategy.
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, *args):
		"""
		Constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.DiagnosticComponent` or
			:class:`sympl.PrognosticComponent`, representing the
			parameterizations to execute.

		References
		----------
		Donahue, A. S., and P. M. Caldwell. (2018). \
		Impact of physics parameterization ordering in a global atmosphere model. \
		*Journal of Advances in Modeling earth Systems*, *10*:481-499.
		"""
		assert_sequence(args, reftype=(Diagnostic, Tendency))
		self._components_list = self._initialize_components_list(args)

	@property
	def components_list(self):
		"""
		Returns
		-------
		tuple :
			Tuple of instances of :class:`sympl.DiagnosticComponent` or
			:class:`sympl.PrognosticComponent`, representing the
			parameterizations to execute.
			As this method is marked as abstract, its implementation is
			delegated to the derived classes.
		"""
		return self._components_list

	@property
	@abc.abstractmethod
	def current_state_input_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting model variables
			which must be present in the input dictionary representing
			the state at the current time level, and whose values are
			dictionaries specifying fundamental properties (dims, units)
			for those variables.
		"""

	@property
	@abc.abstractmethod
	def provisional_state_input_properties(self):
		"""
		Return
		------
		dict :
			Dictionary whose keys are strings denoting model variables
			which must be present in the input dictionary representing
			any arbitrary provisional state (e.g., the state updated
			by the dynamical core),	and whose values are dictionaries
			specifying fundamental properties (dims, units) for those
			variables.
		"""

	@property
	@abc.abstractmethod
	def current_state_output_properties(self):
		"""
		Return
		------
		dict :
            Dictionary whose keys are strings denoting model variables
            which will be surely present in the input dictionary representing
            the state at the current time level when the call operator
            returns, and whose values are dictionaries specifying
            fundamental properties (dims, units) for those variables.
            Note that the state could include some other variables (not used
            by any physical package).
		"""

	@property
	@abc.abstractmethod
	def provisional_state_output_properties(self):
		"""
		Return
		------
		dict :
            Dictionary whose keys are strings denoting model variables
            which will be surely present in the input state representing
            the provisional state when the call operator returns, and whose
            values are dictionaries specifying fundamental properties
            (dims, units) for those variables.
            Note that the state could include some other variables (not used
            by any physical package).
		"""

	@property
	@abc.abstractmethod
	def tendency_properties(self):
		"""
		Return
		------
		dict :
            Dictionary whose keys are strings denoting tendencies
            which are computed by this object, and whose values are
            dictionaries specifying fundamental properties (dims, units)
            for those tendencies.
		"""

	def __call__(self, *, state, state_prv, timestep):
		"""
		Invoke the parameterizations according to a specific coupling mechanism.
		All tendencies calculated by the :class:`sympl.PrognosticComponent`
		objects are summed up, and the input state is modified *in-place*
		by updating it via the diagnostics calculated by the
		:class:`sympl.PrognosticComponent` and :class:`sympl.DiagnosticComponent`
		objects, and the new model variable values output by the
		:class:`sympl.AdamsBashforth` components.

		Parameters
		----------
		state : dict
			The state at the current time level. This is a dictionary whose
			keys are strings denoting the model variables, and whose values
			are :sympl:`DataArray`\s storing values for those variables.
		state_prv : dict
			A provisional state, e.g., the state updated by the dynamical core.
			This is a dictionary whose keys are strings denoting the model
			variables, and whose values are :sympl:`DataArray`\s storing values
			for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the timestep size, i.e.,
			the amount of time to step forward.

		Return
		------
		dict :
            Dictionary whose keys are strings denoting the output
            tendencies, and whose values are :class:`sympl.DataArray`\s
            storing values for those tendencies.
		"""
		tendencies = self._call(state, state_prv, timestep)

		if 'time' not in tendencies.keys():
			try:
				tendencies['time'] = state['time']
			except KeyError:
				pass

		return tendencies

	@abc.abstractmethod
	def _initialize_components_list(self, components):
		"""
		This method initializes the list of physical components.
		"""

	@abc.abstractmethod
	def _call(self, state, state_prv, timestep):
		"""
		Internal routine invoking the parameterizations according to
		a specific coupling mechanism.
		As this method is marked as abstract, its implementation is
		delegated to the derived classes.

		Parameters
		----------
		state : dict
			The state at the current time level. This is a dictionary whose
			keys are strings denoting the model variables, and whose values
			are :sympl:`DataArray`\s storing values for those variables.
		state_prv : dict
			A provisional state, e.g., the state updated by the dynamical core.
			This is a dictionary whose keys are strings denoting the model
			variables, and whose values are :sympl:`DataArray`\s storing values
			for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the timestep size, i.e.,
			the amount of time to step forward.

		Return
		------
		dict :
            Dictionary whose keys are strings denoting the output
            tendencies, and whose values are :class:`sympl.DataArray`\s
            storing values for those tendencies.
		"""


class ConcurrentCoupling(PhysicsComponentComposite):
	"""
	This class inherits
	:class:`~tasmania.physics.composite.PhysicsComponentComposite`
	to implement the *explicit* concurrent coupling strategy.

	References
	----------
	Staniforth, A., N. Wood, and J. C\^ot\'e. (2002). \
		A simple comparison of four physics-dynamics coupling schemes. \
		*Mon. Weather Rev.*, *130*:3129-3135.
	"""
	def __init__(self, *args, mode='serial'):
		"""
		The constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.DiagnosticComponent` or
			:class:`sympl.PrognosticComponent`, representing the
			parameterizations to execute.
		mode : `str`, optional
			String specifying the runtime fashion in which parameterizations
			should be invoked. Either:

				* 'serial', to run the physical packages sequentially,
					and add diagnostics to the current state incrementally;
				* 'as_parallel', to run the physical packages *as* we
					were in a parallel region where each package may be
					assigned to a different thread/process; in other terms,
					we do not rely on the order in which parameterizations
					are passed to this object, and diagnostics are added
					to the current state in a single step just before returning.
		"""
		super().__init__(*args)

		self._mode = mode
		if mode == 'serial':
			self._call = lambda state, state_prv, timestep: self._call_serial(state)
		else:
			self._call = lambda state, state_prv, timestep: self._call_asparallel(state)

	@property
	def current_state_input_properties(self):
		flag = self._mode == 'serial'
		return get_input_properties(self.components_list, consider_diagnostics=flag)

	@property
	def provisional_state_input_properties(self):
		return {}

	@property
	def current_state_output_properties(self):
		return get_output_properties(self.components_list)

	@property
	def provisional_state_output_properties(self):
		return {}

	@property
	def tendency_properties(self):
		tendency_list = [c for c in self.components_list if isinstance(c, Tendency)]
		return combine_component_properties(tendency_list, 'tendency_properties')

	def __call__(self, *, state, state_prv=None, timestep=None):
		"""
		Couple the parameterizations pursuing the concurrent
		coupling strategy. Only the current state is required;
		neither any provisional state, nor even the timestep, is needed.
		"""
		return super().__call__(state=state, state_prv=state_prv,
								timestep=timestep)

	def _initialize_components_list(self, components):
		return components

	def _call_serial(self, state):
		out_tendencies = {}
		tendency_units = {tendency: properties['units']
						  for tendency, properties in self.tendency_properties.items()}

		for component in self.components_list:
			if isinstance(component, Tendency):
				tendencies, diagnostics = component(state)
				out_tendencies.update(add(out_tendencies, tendencies,
										  units=tendency_units,
										  unshared_variables_in_output=True))
				state.update(diagnostics)
			else:
				diagnostics = component(state)
				state.update(diagnostics)

		return out_tendencies

	def _call_asparallel(self, state):
		out_tendencies = {}
		tendency_units = {tendency: properties['units']
						  for tendency, properties in self.tendency_properties.items()}

		agg_diagnostics = {}

		for component in self.components_list:
			if isinstance(component, Tendency):
				tendencies, diagnostics = component(state)
				out_tendencies.update(add(out_tendencies, tendencies,
										  units=tendency_units,
										  unshared_variables_in_output=True))
				agg_diagnostics.update(diagnostics)
			else:
				diagnostics = component(state)
				agg_diagnostics.update(diagnostics)

		# Update the state with the diagnostics
		state.update(agg_diagnostics)

		return out_tendencies


class ParallelSplitting(PhysicsComponentComposite):
	"""
	This class inherits
	:class:`~tasmania.physics.composite.PhysicsComponentComposite`
	to implement the parallel splitting strategy.

	References
	----------
	Staniforth, A., N. Wood, and J. C\^ot\'e. (2002). \
		A simple comparison of four physics-dynamics coupling schemes. \
		*Mon. Weather Rev.*, *130*:3129-3135.
	"""
	def __init__(self, *args, time_integration_scheme='forward_euler',
				 grid, horizontal_boundary_type=None, mode='serial',
				 retrieve_diagnostics_from_provisional_state=False):
		"""
		The constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.DiagnosticComponent` or
			:class:`sympl.PrognosticComponent`, representing the
			parameterizations to execute.
		time_integration_scheme : `str`, optional
			String specifying the marching scheme to integrate
			each parameterization. Either:

				* 'forward_euler', for the forward Euler scheme;
				* 'rk2', for the two-stages, second-order Runge-Kutta (RK) scheme;
				* 'rk3cosmo', for the three-stages RK scheme as used in the
					`COSMO model <http://www.cosmo-model.org>`_; this method is
					nominally second-order, and third-order for linear problems;
				* 'rk3', for the three-stages, third-order RK scheme.

		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the
			underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		mode : `str`, optional
			String specifying the runtime fashion in which parameterizations
			should be invoked. Either:

				* 'serial', to run the physical packages sequentially,
					and add diagnostics to the current state incrementally;
				* 'as_parallel', to run the physical packages *as* we
					were in a parallel region where each package may be
					assigned to a different thread/process; in other terms,
					we do not rely on the order in which parameterizations
					are passed to this object, and diagnostics are added
					to the current state in a single step just before returning.

			Defaults to 'serial'.
		retrieve_diagnostics_from_provisional_state : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) to feed the
			:class:`sympl.DiagnosticComponent` objects with the provisional
			(resp., current) state, and add the so-retrieved diagnostics
			to the provisional (resp., current) state dictionary.
			Defaults to :obj:`False`.
		"""
		self._time_integrator = tendencystepper_factory(time_integration_scheme)
		self._grid = grid
		self._horizontal_boundary_type = horizontal_boundary_type

		super().__init__(*args)

		self._mode = mode
		if mode == 'serial':
			self._call = self._call_serial
		else:
			self._call = self._call_asparallel

		if mode == 'asparallel' and retrieve_diagnostics_from_provisional_state:
			import warnings
			warnings.warn('Argument retrieve_diagnostics_from_provisional_state '
						  'only effective when runtime mode set on ''serial''.')
			self._diagnostics_from_provisional = False
		else:
			self._diagnostics_from_provisional = retrieve_diagnostics_from_provisional_state

	@property
	def current_state_input_properties(self):
		if not self._diagnostics_from_provisional:
			flag = self._mode == 'serial'
			return get_input_properties(self.components_list,
										consider_diagnostics=flag)
		else:
			tendencystepper_components = (component for component in self.components_list
										  if isinstance(component, self._time_integrator))
			return get_input_properties(tendencystepper_components,
										consider_diagnostics=True)

	@property
	def provisional_state_input_properties(self):
		# We require that all prognostic variables affected by the
		# parameterizations are included in the provisional state
		tendencystepper_components = (component for component in self.components_list
                                      if isinstance(component, self._time_integrator))
		return_dict = get_input_properties(tendencystepper_components,
                                    	   component_attribute_name='output_properties',
                                    	   consider_diagnostics=False)

		if self._diagnostics_from_provisional:
			diagnostic_components = (component for component in self.components_list
            	                     if isinstance(component, Diagnostic))

			return_dict.update(get_input_properties(diagnostic_components,
													consider_diagnostics=True,
													return_dict=return_dict))

		return return_dict

	@property
	def current_state_output_properties(self):
		if not self._diagnostics_from_provisional:
			return get_output_properties(self.components_list)
		else:
			tendencystepper_components = (component for component in self.components_list
										  if isinstance(component, self._time_integrator))
			return get_output_properties(tendencystepper_components)

	@property
	def provisional_state_output_properties(self):
		return_dict = self.provisional_state_input_properties

		if self._diagnostics_from_provisional:
			diagnostic_components = (component for component in self.components_list
									 if isinstance(component, Diagnostic))

			return_dict.update(
				get_input_properties(diagnostic_components,
									 component_attribute_name='diagnostic_properties',
									 consider_diagnostics=True,
									 return_dict=return_dict))

		return return_dict

	@property
	def tendency_properties(self):
		return {}

	def _initialize_components_list(self, components):
		g, bnd_type = self._grid, self._horizontal_boundary_type

		Stepper = self._time_integrator

		out_list = []

		for component in components:
			if isinstance(component, Tendency):
				out_list.append(Stepper(component, grid=g,
										horizontal_boundary_type=bnd_type))
			else:
				out_list.append(component)

		return out_list

	def _call_serial(self, state, state_prv, timestep):
		out_tendencies = {}
		out_units = {name: properties['units'] for name, properties in
					 self.provisional_state_output_properties.items()}

		for component in self.components_list:
			if isinstance(component, self._time_integrator):
				diagnostics, state_tmp = component(state, timestep)

				increment = subtract(state_tmp, state,
									 unshared_variables_in_output=False)
				state_prv.update(add(state_prv, increment, units=out_units,
									 unshared_variables_in_output=True))

				state.update(diagnostics)
			else:
				if self._diagnostics_from_provisional:
					diagnostics = component(state_prv)
					state_prv.update(diagnostics)
				else:
					diagnostics = component(state)
					state.update(diagnostics)

		return out_tendencies

	def _call_asparallel(self, state, state_prv, timestep):
		out_tendencies = {}
		agg_diagnostics = {}
		out_units = {name: properties['units'] for name, properties in
					 self.provisional_state_output_properties.items()}

		for component in self.components_list:
			if isinstance(component, self._time_integrator):
				diagnostics, state_tmp = component(state, timestep)

				increment = subtract(state_tmp, state,
									 unshared_variables_in_output=False)
				state_prv.update(add(state_prv, increment, units=out_units,
									 unshared_variables_in_output=True))

				agg_diagnostics.update(diagnostics)
			else:
				diagnostics = component(state)
				agg_diagnostics.update(diagnostics)

		state.update(agg_diagnostics)

		return out_tendencies


class SequentialUpdateSplitting(PhysicsComponentComposite):
	"""
	This class inherits
	:class:`~tasmania.physics.composite.PhysicsComponentComposite`
	to implement the sequential-update splitting strategy.

	References
	----------
	Staniforth, A., N. Wood, and J. C\^ot\'e. (2002). \
		A simple comparison of four physics-dynamics coupling schemes. \
		*Mon. Weather Rev.*, *130*:3129-3135.
	"""
	def __init__(self, *args, time_integration_scheme='forward_euler',
				 grid, horizontal_boundary_type=None):
		"""
		The constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.DiagnosticComponent` or
			:class:`sympl.PrognosticComponent`, representing the
			parameterizations to execute.
		time_integration_scheme : `str`, optional
			String specifying the marching scheme to integrate
			each parameterization. Either:

				* 'forward_euler', for the forward Euler scheme;
				* 'rk2', for the two-stages, second-order Runge-Kutta (RK) scheme;
				* 'rk3cosmo', for the three-stages RK scheme as used in the
					`COSMO model <http://www.cosmo-model.org>`_; this method is
					nominally second-order, and third-order for linear problems;
				* 'rk3', for the three-stages, third-order RK scheme.

			Defaults to 'forward_euler'.
		grid : grid
			:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the
			underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
			for all available options.
		"""
		self._time_integrator = tendencystepper_factory(time_integration_scheme)
		self._grid = grid
		self._horizontal_boundary_type = horizontal_boundary_type
		super().__init__(*args)

	@property
	def current_state_input_properties(self):
		return get_input_properties(self.components_list,
									consider_diagnostics=True)

	@property
	def provisional_state_input_properties(self):
		return self.current_state_input_properties

	@property
	def current_state_output_properties(self):
		return get_output_properties(self.components_list,
									 consider_diagnostics=True)

	@property
	def provisional_state_output_properties(self):
		return self.current_state_output_properties

	@property
	def tendency_properties(self):
		return {}

	def __call__(self, *, state=None, state_prv=None, timestep):
		"""
		Couple the parameterizations pursuing the sequential update
		splitting strategy. Either the current state or the provisional
		state is required; if both are given, an exception is thrown.
		"""
		if (not state and not state_prv) or (state and state_prv):
			raise ValueError('Either the ''state'' or ''state_prv'' can '
							 'be not None, not both.')

		if state is not None:
			return super().__call__(state=state, state_prv=None,
									timestep=timestep)
		else:
			return super().__call__(state=state_prv, state_prv=None,
									timestep=timestep)

	def _initialize_components_list(self, components):
		g, bnd_type = self._grid, self._horizontal_boundary_type

		Stepper = self._time_integrator

		out_list = []

		for component in components:
			if isinstance(component, Tendency):
				out_list.append(Stepper(component, grid=g,
										horizontal_boundary_type=bnd_type))
			else:
				out_list.append(component)

		return out_list

	def _call(self, state, state_prv, timestep):
		out_tendencies = {}

		for component in self.components_list:
			if isinstance(component, self._time_integrator):
				diagnostics, state_tmp = component(state, timestep)
				state.update(state_tmp)
				state.update(diagnostics)
			else:
				diagnostics = component(state)
				state.update(diagnostics)

		return out_tendencies
