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
	SequentialUpdateSplitting
"""
from sympl import \
	DiagnosticComponent, DiagnosticComponentComposite as SymplDiagnosticComponentComposite, \
	TendencyComponent, TendencyComponentComposite, \
	ImplicitTendencyComponent, ImplicitTendencyComponentComposite, \
	combine_component_properties

from tasmania.python.core.composite import \
	DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite
from tasmania.python.core.concurrent_coupling import ConcurrentCoupling
from tasmania.python.core.tendency_steppers import tendencystepper_factory
from tasmania.python.utils.dict_utils import add, subtract
from tasmania.python.utils.framework_utils import \
	check_properties_compatibility, get_input_properties, get_output_properties
from tasmania.python.utils.utils import assert_sequence


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
		SymplDiagnosticComponentComposite,
		TasmaniaDiagnosticComponentComposite,
	)
	allowed_tendency_type = (
		TendencyComponent,
		TendencyComponentComposite,
		ImplicitTendencyComponent,
		ImplicitTendencyComponentComposite,
	)
	allowed_component_type = \
		allowed_diagnostic_type + allowed_tendency_type + (ConcurrentCoupling, )

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
						- :class:`tasmania.DiagnosticComponentComposite`,
						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`,
						- :class:`sympl.ImplicitTendencyComponentComposite`, or
						- :class:`tasmania.ConcurrentCoupling`

					representing the process;
				* if 'component' is a

						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`,
						- :class:`sympl.ImplicitTendencyComponentComposite`, or
						- :class:`tasmania.ConcurrentCoupling`,

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
						- :class:`sympl.ImplicitTendencyComponent`,
						- :class:`sympl.ImplicitTendencyComponentComposite`, or
						- :class:`tasmania.ConcurrentCoupling`,

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