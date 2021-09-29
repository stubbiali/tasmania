# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from typing import TYPE_CHECKING

from sympl._core.composite import (
    DiagnosticComponentComposite as SymplDiagnosticComponentComposite,
    ImplicitTendencyComponentComposite,
    TendencyComponentComposite,
)
from sympl._core.core_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    TendencyComponent,
)

from tasmania.python.framework.composite import (
    DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite,
)
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.sequential_update_splitting_utils import (
    StaticOperator,
)
from tasmania.python.framework.static_checkers import (
    check_properties_are_compatible,
)
from tasmania.python.framework.steppers import TendencyStepper
from tasmania.python.utils import typingx
from tasmania.python.utils.dict import DataArrayDictOperator

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict
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
                        enable_checks=options.enable_checks,
                        backend=options.backend,
                        backend_options=options.backend_options,
                        storage_options=options.storage_options,
                        **options.kwargs
                    )
                )
                self._substeps.append(max(options.substeps, 1))

        # set properties
        self.input_properties = StaticOperator.get_input_properties(self)
        self.output_properties = StaticOperator.get_output_properties(self)

        # static checks
        check_properties_are_compatible(
            self, "input_properties", self, "output_properties"
        )

        self._dict_op = DataArrayDictOperator()

        self._out_diagnostics = [None] * len(self._component_list)
        self._out_state = [None] * len(self._component_list)

    @property
    def components(self):
        return tuple(self._component_list)

    def __call__(
        self,
        state: "DataArrayDict",
        timestep: typingx.TimeDelta,
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

        for idx, component in enumerate(self._component_list):
            if not isinstance(component, self.allowed_diagnostic_type):
                substeps = self._substeps[idx]

                self._out_diagnostics[idx], self._out_state[idx] = component(
                    state,
                    timestep / substeps,
                    out_diagnostics=self._out_diagnostics[idx],
                    out_state=self._out_state[idx],
                )

                if substeps > 1:
                    raise NotImplementedError()

                self._dict_op.update_swap(state, self._out_diagnostics[idx])
                self._dict_op.update_swap(state, self._out_state[idx])
            else:
                try:
                    self._out_diagnostics[idx] = component(
                        state, out=self._out_diagnostics[idx]
                    )
                except TypeError:
                    self._out_diagnostics[idx] = component(
                        state, timestep, out=self._out_diagnostics[idx]
                    )

                self._dict_op.update_swap(state, self._out_diagnostics[idx])

            # ensure state is still defined at current time level
            state["time"] = current_time

        # ensure the state is defined at the next time level
        state["time"] = current_time + timestep
