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
import numpy as np
from sympl import (
    DiagnosticComponent,
    DiagnosticComponentComposite as SymplDiagnosticComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
)
from typing import TYPE_CHECKING

from tasmania.python.framework.composite import (
    DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite,
)
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper
from tasmania.python.utils import typingx
from tasmania.python.utils.framework import (
    check_property_compatibility,
    get_input_properties,
    get_output_properties,
)

if TYPE_CHECKING:
    from tasmania.python.framework.options import TimeIntegrationOptions


class SequentialTendencySplitting:
    """
    Callable class which integrates a bundle of physical processes pursuing
    the sequential tendency splitting strategy.

    Attributes
    ----------
    input_properties : dict[str, dict]
        Dictionary whose keys are strings denoting model variables
        which should be present in the input state, and whose values
        are dictionaries specifying fundamental properties (dims, units)
        of those variables.
    provisional_input_properties : dict[str, dict]
        Dictionary whose keys are strings denoting variables which
        should be present in the input model dictionary representing
        the provisional state, and whose values are dictionaries specifying
        fundamental properties (dims, units) of those variables.
    output_properties : dict[str, dict]
        Dictionary whose keys are strings denoting variables which
        will be present in the input model dictionary representing
        the current state when the call operator returns, and
        whose values are dictionaries specifying fundamental properties
        (dims, units) of those variables.
    provisional_output_properties : dict[str, dict]
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
        ConcurrentCoupling,
    )
    allowed_component_type = allowed_diagnostic_type + allowed_tendency_type

    def __init__(self, *args: "TimeIntegrationOptions") -> None:
        """
        Parameters
        ----------
        *args : TimeIntegrationOptions
            TODO
        """
        self._component_list = []
        self._substeps = []
        for options in args:
            component = options.component

            assert isinstance(component, self.allowed_component_type), (
                f"""The component should be an instance of either """
                f"""{', '.join(str(ctype) for ctype in self.allowed_component_type)}"""
            )

            if isinstance(component, self.allowed_diagnostic_type):
                self._component_list.append(component)
                self._substeps.append(1)
            else:
                scheme = options.scheme or "forward_euler"
                self._component_list.append(
                    STSTendencyStepper.factory(
                        scheme,
                        component,
                        execution_policy="serial",
                        enforce_horizontal_boundary=options.enforce_horizontal_boundary,
                        backend=options.backend,
                        backend_options=options.backend_options,
                        storage_options=options.storage_options,
                        **options.kwargs
                    )
                )
                self._substeps.append(max(options.substeps, 1))

        # set properties
        self.input_properties = self._get_input_properties()
        self.provisional_input_properties = (
            self._get_provisional_input_properties()
        )
        self.output_properties = self._get_output_properties()
        self.provisional_output_properties = (
            self._get_provisional_output_properties()
        )

    def _get_input_properties(self) -> typingx.PropertiesDict:
        return get_input_properties(
            tuple(
                {
                    "component": component,
                    "attribute_name": "input_properties",
                    "consider_diagnostics": True,
                }
                for component in self._component_list
                if isinstance(component, STSTendencyStepper)
            )
        )

    def _get_provisional_input_properties(self) -> typingx.PropertiesDict:
        at_disposal = {}
        return_dict = {}

        for component in self._component_list:
            if isinstance(component, self.__class__.allowed_diagnostic_type):
                required = component.input_properties
                given = component.diagnostic_properties
            else:
                required = component.provisional_input_properties
                given = component.output_properties

            for key in required:
                if key in at_disposal:
                    check_property_compatibility(
                        at_disposal[key], required[key]
                    )
                else:
                    at_disposal[key] = required[key]
                    return_dict[key] = required[key]

            for key in given:
                if key in at_disposal:
                    check_property_compatibility(at_disposal[key], given[key])
                else:
                    at_disposal[key] = given[key]

        return return_dict

    def _get_output_properties(self) -> typingx.PropertiesDict:
        return_dict = self._get_input_properties()
        get_output_properties(
            tuple(
                {
                    "component": component,
                    "attribute_name": "diagnostic_properties",
                    "consider_diagnostics": False,
                }
                for component in self._component_list
                if isinstance(component, STSTendencyStepper)
            ),
            return_dict=return_dict,
        )
        return return_dict

    def _get_provisional_output_properties(self) -> typingx.PropertiesDict:
        return_dict = self._get_provisional_input_properties()

        for component in self._component_list:
            if isinstance(component, self.__class__.allowed_diagnostic_type):
                given = component.diagnostic_properties
            else:
                given = component.output_properties

            for key in given:
                if key in return_dict:
                    check_property_compatibility(return_dict[key], given[key])
                else:
                    return_dict[key] = given[key]

        return return_dict

    def __call__(
        self,
        state: typingx.DataArrayDict,
        state_prv: typingx.mutable_dataarray_dict_t,
        timestep: typingx.TimeDelta,
    ) -> None:
        """
        Advance the model state one timestep forward in time by pursuing
        the sequential-tendency splitting method.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The current state.
        state_prv : dict[str, sympl.DataArray]
            The provisional state.
        timestep : datetime.timedelta
            The timestep size.

        Note
        ----
        `state_prv` is modified in-place to represent the final model state.
        """
        current_time = state["time"]

        for component, substeps in zip(self._component_list, self._substeps):
            if not isinstance(component, self.allowed_diagnostic_type):
                diagnostics, state_tmp = component(
                    state, state_prv, timestep / substeps
                )

                if substeps > 1:
                    state_tmp.update(
                        {
                            key: value
                            for key, value in state.items()
                            if key not in state_tmp
                        }
                    )

                    for _ in range(1, substeps):
                        _, state_aux = component(
                            state_tmp, state_prv, timestep / substeps
                        )
                        state_tmp.update(state_aux)

                state_prv.update(state_tmp)
                state.update(diagnostics)
            else:
                try:
                    diagnostics = component(state_prv)
                except TypeError:
                    diagnostics = component(state_prv, timestep)

                state_prv.update(diagnostics)

            # ensure state is still defined at current time level
            state["time"] = current_time

        # ensure the provisional state is defined at the next time level
        state_prv["time"] = current_time + timestep
