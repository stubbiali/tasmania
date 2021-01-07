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
from tasmania.python.framework.tendency_stepper import TendencyStepper
from tasmania.python.utils import taz_types
from tasmania.python.utils.framework_utils import (
    check_properties_compatibility,
    get_input_properties,
    get_output_properties,
)

if TYPE_CHECKING:
    from tasmania.python.framework.options import TimeIntegrationOptions


class SequentialUpdateSplitting:
    """
    Callable class which integrates a bundle of physical processes pursuing
    the sequential update splitting strategy.

    Attributes
    ----------
    input_properties : dict[str, dict]
        Dictionary whose keys are strings denoting model variables
        which should be present in the input state, and whose values
        are dictionaries specifying fundamental properties (dims, units)
        of those variables.
    output_properties : dict[str, dict]
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
                    TendencyStepper.factory(
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
        self.input_properties = self._init_input_properties()
        self.output_properties = self._init_output_properties()

        # ensure that dimensions and units of the variables present
        # in both input_properties and output_properties are compatible
        # across the two dictionaries
        check_properties_compatibility(
            self.input_properties,
            self.output_properties,
            properties1_name="input_properties",
            properties2_name="output_properties",
        )

    def _init_input_properties(self) -> taz_types.properties_dict_t:
        return get_input_properties(
            tuple(
                {
                    "component": component,
                    "attribute_name": "input_properties",
                    "consider_diagnostics": True,
                }
                for component in self._component_list
            )
        )

    def _init_output_properties(self) -> taz_types.properties_dict_t:
        return get_output_properties(
            tuple(
                {
                    "component": component,
                    "attribute_name": "input_properties",
                    "consider_diagnostics": True,
                }
                for component in self._component_list
            )
        )

    def __call__(
        self,
        state: taz_types.mutable_dataarray_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> None:
        """
        Advance the model state one timestep forward in time by pursuing
        the parallel splitting method.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            Model state dictionary representing the model state to integrate.
            Its keys are strings denoting the model variables, and its values
            are :class:`sympl.DataArray`\s storing data for those variables.
        timestep : datetime.timedelta
            :class:`datetime.timedelta` representing the timestep size.

        Note
        ----
        `state` is modified in-place to represent the final model state.
        """
        current_time = state["time"]

        for component, substeps in zip(self._component_list, self._substeps):
            if not isinstance(component, self.allowed_diagnostic_type):
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
                        _, state_aux = component(
                            state_tmp, timestep / substeps
                        )
                        state_tmp.update(state_aux)

                state.update(state_tmp)
                state.update(diagnostics)
            else:
                try:
                    diagnostics = component(state)
                except TypeError:
                    diagnostics = component(state, timestep)

                state.update(diagnostics)

            # ensure state is still defined at current time level
            state["time"] = current_time

        # ensure the state is defined at the next time level
        state["time"] = current_time + timestep
