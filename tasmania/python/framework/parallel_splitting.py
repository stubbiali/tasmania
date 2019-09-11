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
	ParallelSplitting
"""
from sympl import (
    DiagnosticComponent,
    DiagnosticComponentComposite as SymplDiagnosticComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
)

from tasmania.python.framework.composite import (
    DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite,
)
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.tendency_steppers import tendencystepper_factory
from tasmania.python.utils.dict_utils import add, subtract
from tasmania.python.utils.framework_utils import (
    check_properties_compatibility,
    get_input_properties,
    get_output_properties,
)


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
        SymplDiagnosticComponentComposite,
        TasmaniaDiagnosticComponentComposite,
    )
    allowed_tendency_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
    )
    allowed_component_type = (
        allowed_diagnostic_type + allowed_tendency_type + (ConcurrentCoupling,)
    )

    def __init__(
        self,
        *args,
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=False
    ):
        """
		Parameters
		----------
		*args : dict
			Dictionaries containing the components to wrap and specifying
			fundamental properties (time_integrator, substeps) of those processes.
			Particularly:

				* 'component' is an instance of

						- :class:`sympl.DiagnosticComponent`
						- :class:`sympl.DiagnosticComponentComposite`
						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`,
						- :class:`sympl.ImplicitTendencyComponentComposite`, or
						- :class:`tasmania.ConcurrentCoupling`

					representing the process;
				* if 'component' is an instance of

						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`,
						- :class:`sympl.ImplicitTendencyComponentComposite`, or
						- :class:`tasmania.ConcurrentCoupling`,

					'time_integrator' is a string specifying the scheme to
					integrate the process forward in time. Available options:

						- 'forward_euler', for the forward Euler scheme;
						- 'rk2', for the two-stage second-order Runge-Kutta (RK) scheme;
						- 'rk3ws', for the three-stage RK scheme as used in the
							`COSMO model <http://www.cosmo-model.org>`_; this method is
							nominally second-order, and third-order for linear problems;
						- 'rk3', for the three-stages, third-order RK scheme.

				* if 'component' is either an instance of or wraps objects of class

						- :class:`tasmania.TendencyComponent`,
						- :class:`tasmania.ImplicitTendencyComponent`, or
						- :class:`tasmania.ConcurrentCoupling`,

					'enforce_horizontal_boundary' is either :obj:`True` if the
					boundary conditions should be enforced after each stage of
					the time integrator, or :obj:`False` not to apply the boundary
					constraints at all. Defaults to :obj:`False`;
				* if 'component' is a

						- :class:`sympl.TendencyComponent`,
						- :class:`sympl.TendencyComponentComposite`,
						- :class:`sympl.ImplicitTendencyComponent`,
						- :class:`sympl.ImplicitTendencyComponentComposite`, or
						- :class:`tasmania.ConcurrentCoupling`,

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
                bare_component = process["component"]
            except KeyError:
                msg = (
                    "Missing mandatory key ''component'' in one item of ''processes''."
                )
                raise KeyError(msg)

            assert isinstance(
                bare_component, self.__class__.allowed_component_type
            ), "''component'' value should be either a {}.".format(
                ", ".join(str(ctype) for ctype in self.__class__.allowed_component_type)
            )

            if isinstance(bare_component, self.__class__.allowed_diagnostic_type):
                self._component_list.append(bare_component)
                self._substeps.append(1)
            else:
                integrator = process.get("time_integrator", "forward_euler")
                enforce_hb = process.get("enforce_horizontal_boundary", False)

                TendencyStepper = tendencystepper_factory(integrator)
                self._component_list.append(
                    TendencyStepper(
                        bare_component, enforce_horizontal_boundary=enforce_hb
                    )
                )

                substeps_ = process.get("substeps", 1)
                substeps = substeps_ if substeps_ > 0 else 1
                self._substeps.append(substeps)

        self._policy = execution_policy
        if execution_policy == "serial":
            self._call = self._call_serial
        else:
            self._call = self._call_asparallel

        if (
            execution_policy == "as_parallel"
            and retrieve_diagnostics_from_provisional_state
        ):
            import warnings

            warnings.warn(
                "Argument retrieve_diagnostics_from_provisional_state "
                "only effective when execution policy set on "
                "serial"
                "."
            )
            self._diagnostics_from_provisional = False
        else:
            self._diagnostics_from_provisional = (
                retrieve_diagnostics_from_provisional_state
            )

        # Set properties
        self.input_properties = self._init_input_properties()
        self.provisional_input_properties = self._init_provisional_input_properties()
        self.output_properties = self._init_output_properties()
        self.provisional_output_properties = self._init_provisional_output_properties()

        # Ensure that dimensions and units of the variables present
        # in both input_properties and output_properties are compatible
        # across the two dictionaries
        check_properties_compatibility(
            self.input_properties,
            self.output_properties,
            properties1_name="input_properties",
            properties2_name="output_properties",
        )

        # Ensure that dimensions and units of the variables present
        # in both provisional_input_properties and provisional_output_properties
        # are compatible across the two dictionaries
        check_properties_compatibility(
            self.provisional_input_properties,
            self.provisional_output_properties,
            properties1_name="provisional_input_properties",
            properties2_name="provisional_output_properties",
        )

    def _init_input_properties(self):
        if not self._diagnostics_from_provisional:
            flag = self._policy == "serial"
            return get_input_properties(self._component_list, consider_diagnostics=flag)
        else:
            tendencystepper_components = tuple(
                component
                for component in self._component_list
                if not isinstance(component, self.__class__.allowed_diagnostic_type)
            )
            return get_input_properties(
                tendencystepper_components, consider_diagnostics=True
            )

    def _init_provisional_input_properties(self):
        # We require that all prognostic variables affected by the
        # parameterizations are included in the provisional state
        tendencystepper_components = tuple(
            component
            for component in self._component_list
            if not isinstance(component, self.__class__.allowed_diagnostic_type)
        )
        return_dict = get_input_properties(
            tendencystepper_components,
            component_attribute_name="output_properties",
            consider_diagnostics=False,
        )

        if self._diagnostics_from_provisional:
            diagnostic_components = (
                component
                for component in self._component_list
                if isinstance(component, self.__class__.allowed_diagnostic_type)
            )

            return_dict.update(
                get_input_properties(
                    diagnostic_components,
                    consider_diagnostics=True,
                    return_dict=return_dict,
                )
            )

        return return_dict

    def _init_output_properties(self):
        if not self._diagnostics_from_provisional:
            return get_output_properties(self._component_list)
        else:
            tendencystepper_components = tuple(
                component
                for component in self._component_list
                if not isinstance(component, self.__class__.allowed_diagnostic_type)
            )
            return get_output_properties(tendencystepper_components)

    def _init_provisional_output_properties(self):
        return_dict = self.provisional_input_properties

        if self._diagnostics_from_provisional:
            diagnostic_components = (
                component
                for component in self._component_list
                if isinstance(component, self.__class__.allowed_diagnostic_type)
            )

            return_dict.update(
                get_output_properties(
                    diagnostic_components,
                    component_attribute_name="",
                    consider_diagnostics=True,
                    return_dict=return_dict,
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
        state_prv["time"] = state["time"] + timestep

    def _call_serial(self, state, state_prv, timestep):
        """
		Process the components in 'serial' runtime mode.
		"""
        out_units = {
            name: properties["units"]
            for name, properties in self.provisional_output_properties.items()
        }

        for component, substeps in zip(self._component_list, self._substeps):
            if not isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics, state_tmp = component(state, timestep / substeps)

                if substeps > 1:
                    state_tmp.update(
                        {
                            key: value
                            for key, value in state.items()
                            if key not in state_tmp
                        }
                    )

                    for _ in range(1, substeps):
                        _, state_aux = component(state_tmp, timestep / substeps)
                        state_tmp.update(state_aux)

                increment = subtract(
                    state_tmp, state, unshared_variables_in_output=False
                )
                state_prv.update(
                    add(
                        state_prv,
                        increment,
                        units=out_units,
                        unshared_variables_in_output=True,
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
            name: properties["units"]
            for name, properties in self.provisional_output_properties.items()
        }

        for component, substeps in zip(self._component_list, self._substeps):
            if not isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics, state_tmp = component(state, timestep / substeps)

                if substeps > 1:
                    state_tmp.update(
                        {
                            key: value
                            for key, value in state.items()
                            if key not in state_tmp
                        }
                    )

                    for _ in range(1, substeps):
                        _, state_aux = component(state_tmp, timestep / substeps)
                        state_tmp.update(state_aux)

                increment = subtract(
                    state_tmp, state, unshared_variables_in_output=False
                )
                state_prv.update(
                    add(
                        state_prv,
                        increment,
                        units=out_units,
                        unshared_variables_in_output=True,
                    )
                )

                agg_diagnostics.update(diagnostics)
            else:
                diagnostics = component(state)
                agg_diagnostics.update(diagnostics)

        state.update(agg_diagnostics)
