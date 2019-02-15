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
This module contain:
	get_increment
	tendencystepper_factory
	ForwardEuler
	RungeKutta2
	RungeKutta3COSMO
	RungeKutta3
"""
import abc
import copy
from sympl import \
	DataArray, TendencyComponent, TendencyComponentComposite, \
	ImplicitTendencyComponent, ImplicitTendencyComponentComposite
from sympl._components.timesteppers import convert_tendencies_units_for_state
from sympl._core.base_components import InputChecker, DiagnosticChecker, OutputChecker
from sympl._core.units import clean_units

from tasmania.python.core.concurrent_coupling import ConcurrentCoupling
from tasmania.python.dynamics.horizontal_boundary import HorizontalBoundary
from tasmania.python.utils.dict_utils import add, multiply
from tasmania.python.utils.utils import assert_sequence


def get_increment(state, timestep, prognostic):
	# Calculate tendencies and retrieve diagnostics
	tendencies, diagnostics = prognostic(state, timestep)

	# Convert tendencies in units compatible with the state
	convert_tendencies_units_for_state(tendencies, state)

	# Calculate the increment
	increment = multiply(timestep.total_seconds(), tendencies)

	# Set the correct units for the increment of each variable
	for key, val in increment.items():
		if isinstance(val, DataArray) and 'units' in val.attrs.keys():
			val.attrs['units'] += ' s'
			val.attrs['units'] = clean_units(val.attrs['units'])

	return increment, diagnostics


def tendencystepper_factory(scheme):
	if scheme == 'forward_euler':
		return ForwardEuler
	elif scheme == 'rk2':
		return RungeKutta2
	elif scheme == 'rk3cosmo':
		return RungeKutta3COSMO
	elif scheme == 'rk3':
		return RungeKutta3
	else:
		raise ValueError(
			'Unsupported time integration scheme ''{}''. '
			'Available integrators: forward_euler, rk2, rk3cosmo, rk3.'.format(scheme)
		)


class TendencyStepper:
	"""
	Callable abstract base class which steps a model state based on the 
	tendencies calculated by a set of wrapped prognostic components.
	"""
	__metaclass__ = abc.ABCMeta

	allowed_component_type = (
		TendencyComponent,
		TendencyComponentComposite,
		ImplicitTendencyComponent,
		ImplicitTendencyComponentComposite,
		ConcurrentCoupling,
	)

	def __init__(self, *args, **kwargs): 
		"""
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : `str`, optional
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		grid : grid
			:class:`tasmania.GridXYZ` object representing
			the underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`tasmania.HorizontalBoundary`
			for all available options.
		time_units : `str`, optional
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
		assert_sequence(args, reftype=self.__class__.allowed_component_type)

		self._prognostic_list = args
		self._prognostic = \
			args[0] if (len(args) == 1 and isinstance(args[0], ConcurrentCoupling)) \
			else ConcurrentCoupling(
				*args, execution_policy=kwargs.get('execution_policy', 'serial'),
				time_units=kwargs.get('time_units', 's')
			)

		self._tunits = kwargs.get('time_units', 's')

		self._input_checker = InputChecker(self)
		self._diagnostic_checker = DiagnosticChecker(self)
		self._output_checker = OutputChecker(self)

		horizontal_boundary_type = kwargs.get('horizontal_boundary_type', None)
		if horizontal_boundary_type is not None:
			# Determine the number of boundary layers by inspecting
			# the attribute nb of each prognostic component
			nb = 1
			for component in self.prognostic_list:
				nb = max(nb, getattr(component, 'nb', 1))

			# Instantiate the object which will take care of boundary conditions
			grid = kwargs.get('grid')
			self._bnd = HorizontalBoundary.factory(horizontal_boundary_type, grid, nb)
		else:
			self._bnd = None

	@property
	def prognostic(self):
		"""
		obj :
			The :class:`tasmania.ConcurrentCoupling` calculating the tendencies.
		"""
		return self._prognostic

	@property
	def input_properties(self):
		"""
		dict :
			Dictionary whose keys are strings denoting model variables
			which should be present in the input state dictionary, and
			whose values are dictionaries specifying fundamental properties
			(dims, units) of those variables.
		"""
		return self._prognostic.input_properties

	@property
	def diagnostic_properties(self):
		"""
		dict :
			Dictionary whose keys are strings denoting diagnostics
			which are retrieved from the input state dictionary, and 
			whose values are dictionaries specifying fundamental 
			properties (dims, units) of those diagnostics.
		"""
		return self._prognostic.diagnostic_properties

	@property
	def output_properties(self):
		"""
		dict :
			Dictionary whose keys are strings denoting model variables
			present in the output state dictionary, and whose values are 
			dictionaries specifying fundamental properties (dims, units) 
			of those variables.
		"""
		return_dict = {}

		for key, val in self._prognostic.tendency_properties.items():
			return_dict[key] = copy.deepcopy(val)
			if 'units' in return_dict[key]:
				return_dict[key]['units'] = \
					clean_units(return_dict[key]['units'] + self._tunits)

		return return_dict

	def __call__(self, state, timestep):
		"""
		Step the model state.

		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting the input model 
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the time step.
			
		Return
		------
		dict :
			Dictionary whose keys are strings denoting the output model 
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		"""
		self._input_checker.check_inputs(state)

		diagnostics, out_state = self._call(state, timestep)

		self._diagnostic_checker.check_diagnostics(
			{key: val for key, val in diagnostics.items() if key != 'time'}
		)
		diagnostics['time'] = state['time']

		self._output_checker.check_outputs(
			{key: val for key, val in out_state.items() if key != 'time'}
		)
		out_state['time'] = state['time'] + timestep

		return diagnostics, out_state

	@abc.abstractmethod
	def _call(self, state, timestep):
		"""
		Step the model state. As this method is marked as abstract, 
		its implementation is delegated to the derived classes.

		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting the input model 
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the time step.
			
		Return
		------
		dict :
			Dictionary whose keys are strings denoting the output model 
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		"""
		pass


class ForwardEuler(TendencyStepper):
	"""
	This class inherits :class:`tasmania.TendencyStepper` to
	implement the forward Euler time integration scheme.
	"""
	def __init__(self, *args, **kwargs): 
		"""
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : `str`, optional
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		grid : grid
			:class:`tasmania.GridXYZ` object representing
			the underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`tasmania.HorizontalBoundary`
			for all available options.
		time_units : `str`, optional
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
		super().__init__(*args, **kwargs)

	def _call(self, state, timestep):
		# Shortcuts
		dt = timestep.total_seconds()
		out_units = {
			name: properties['units']
			for name, properties in self.output_properties.items()
		}

		# Calculate the increment and the diagnostics
		increment, diagnostics = get_increment(state, timestep, self.prognostic)

		# Step the solution
		out_state = add(
			state, increment,
			units=out_units, unshared_variables_in_output=False
		)

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(
					out_state[name].values,
					state[name].to_units(out_units[name]).values
				)

		return diagnostics, out_state


class RungeKutta2(TendencyStepper):
	"""
	This class inherits :class:`sympl.TendencyStepper` to
	implement the two-stages, second-order Runge-Kutta scheme
	as described in the reference.

	References
	----------
	Gear, C. W. (1971). *Numerical initial value problems in \
		ordinary differential equations.* Prentice Hall PTR.
	"""
	def __init__(self, *args, **kwargs): 
		"""
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : `str`, optional
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		grid : grid
			:class:`tasmania.GridXYZ` object representing
			the underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`tasmania.HorizontalBoundary`
			for all available options.
		time_units : `str`, optional
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
		super().__init__(*args, **kwargs)

	def _call(self, state, timestep):
		# Shortcuts
		out_units = {
			name: properties['units']
			for name, properties in self.output_properties.items()
		}

		# First stage
		k0, diagnostics = get_increment(state, timestep, self.prognostic)
		state_1 = add(
			state, multiply(0.5, k0),
			units=out_units, unshared_variables_in_output=True
		)
		state_1['time'] = state['time'] + 0.5*timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(
					state_1[name].values,
					state[name].to_units(out_units[name]).values
				)

		# Second stage
		k1, _ = get_increment(state_1, timestep, self.prognostic)
		out_state = add(
			state, k1,
			units=out_units, unshared_variables_in_output=False
		)
		out_state['time'] = state['time'] + timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(out_state[name].values, state_1[name].values)

		return diagnostics, out_state


class RungeKutta3COSMO(TendencyStepper):
	"""
	This class inherits :class:`sympl.TendencyStepper` to
	implement the three-stages Runge-Kutta scheme as used in the
	`COSMO model <http://www.cosmo-model.org>`_. This integrator is
	nominally second-order, and third-order for linear problems.

	References
	----------
	Doms, G., and M. Baldauf. (2015). *A description of the nonhydrostatic \
		regional COSMO-model. Part I: Dynamics and numerics.* \
		Deutscher Wetterdienst, Germany.
	"""
	def __init__(self, *args, **kwargs): 
		"""
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : `str`, optional
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		grid : grid
			:class:`tasmania.GridXYZ` object representing
			the underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`tasmania.HorizontalBoundary`
			for all available options.
		time_units : `str`, optional
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
		super().__init__(*args, **kwargs)

	def _call(self, state, timestep):
		# Shortcuts
		out_units = {name: properties['units']
					 for name, properties in self.output_properties.items()}

		# First stage
		k0, diagnostics = get_increment(state, timestep, self.prognostic)
		state_1 = add(
			state, multiply(1.0/3.0, k0),
			units=out_units, unshared_variables_in_output=True
		)
		state_1['time'] = state['time'] + 1.0/3.0 * timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(
					state_1[name].values,
					state[name].to_units(out_units[name]).values
				)

		# Second stage
		k1, _ = get_increment(state_1, timestep, self.prognostic)
		state_2 = add(
			state, multiply(0.5, k1),
			units=out_units, unshared_variables_in_output=True
		)
		state_2['time'] = state['time'] + 0.5 * timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(state_2[name].values, state_1[name].values)

		# Second stage
		k2, _ = get_increment(state_2, timestep, self.prognostic)
		out_state = add(
			state, k2,
			units=out_units, unshared_variables_in_output=False
		)
		out_state['time'] = state['time'] + timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(out_state[name].values, state_2[name].values)

		return diagnostics, out_state


class RungeKutta3(TendencyStepper):
	"""
	This class inherits :class:`sympl.TendencyStepper` to
	implement the three-stages, third-order Runge-Kutta scheme
	as described in the reference.

	References
	----------
	Gear, C. W. (1971). *Numerical initial value problems in \
		ordinary differential equations.* Prentice Hall PTR.
	"""
	def __init__(self, *args, **kwargs): 
		"""
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : `str`, optional
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		grid : grid
			:class:`tasmania.GridXYZ` object representing
			the underlying grid.
		horizontal_boundary_type : str
			String specifying the horizontal boundary conditions.
			See :class:`tasmania.HorizontalBoundary`
			for all available options.
		time_units : `str`, optional
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
		super().__init__(*args, **kwargs)

		# Free parameters for RK3
		self._alpha1 = 1./2.
		self._alpha2 = 3./4.

		# Set the other parameters yielding a third-order method
		self._gamma1 = (3.*self._alpha2 - 2.) / \
					   (6. * self._alpha1 * (self._alpha2 - self._alpha1))
		self._gamma2 = (3.*self._alpha1 - 2.) / \
					   (6. * self._alpha2 * (self._alpha1 - self._alpha2))
		self._gamma0 = 1. - self._gamma1 - self._gamma2
		self._beta21 = self._alpha2 - 1. / (6. * self._alpha1 * self._gamma2)

	def _call(self, state, timestep):
		# Shortcuts
		out_units  = {
			name: properties['units']
			for name, properties in self.output_properties.items()
		}
		a1, a2     = self._alpha1, self._alpha2
		b21        = self._beta21
		g0, g1, g2 = self._gamma0, self._gamma1, self._gamma2

		# First stage
		k0, diagnostics = get_increment(state, timestep, self.prognostic)
		state_1 		= add(
			state, multiply(a1, k0),
			units=out_units, unshared_variables_in_output=True
		)
		state_1['time'] = state['time'] + a1 * timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(
					state_1[name].values,
					state[name].to_units(out_units[name]).values
				)

		# Second stage
		k1, _ 	= get_increment(state_1, timestep, self.prognostic)
		state_2 = add(
			state, add(multiply(b21, k0), multiply((a2 - b21), k1)),
			units=out_units, unshared_variables_in_output=True
		)
		state_2['time'] = state['time'] + a2 * timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(state_2[name].values, state_1[name].values)

		# Second stage
		k2, _     = get_increment(state_2, timestep, self.prognostic)
		k1k2      = add(multiply(g1, k1), multiply(g2, k2))
		k0k1k2 	  = add(multiply(g0, k0), k1k2)
		out_state = add(
			state, k0k1k2,
			units=out_units, unshared_variables_in_output=False
		)
		out_state['time'] = state['time'] + timestep

		if self._bnd is not None:
			# Enforce the boundary conditions on each prognostic variable
			for name in self.output_properties.keys():
				self._bnd.enforce(out_state[name].values, state_2[name].values)

		return diagnostics, out_state
