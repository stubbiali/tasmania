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
from tasmania.python.framework.sequential_tendency_splitting_utils import (
    StaticOperator,
)
from tasmania.python.framework.steppers import SequentialTendencyStepper
from tasmania.python.utils import typingx
from tasmania.python.utils.dict import DataArrayDictOperator

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict

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
                    SequentialTendencyStepper.factory(
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
        self.provisional_input_properties = (
            StaticOperator.get_provisional_input_properties(self)
        )
        self.output_properties = StaticOperator.get_output_properties(self)
        self.provisional_output_properties = (
            StaticOperator.get_provisional_output_properties(self)
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
        state_prv: "DataArrayDict",
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

        for idx, component in enumerate(self._component_list):
            if not isinstance(component, self.allowed_diagnostic_type):
                substeps = self._substeps[idx]

                self._out_diagnostics[idx], self._out_state[idx] = component(
                    state,
                    state_prv,
                    timestep / substeps,
                    out_diagnostics=self._out_diagnostics[idx],
                    out_state=self._out_state[idx],
                )

                if substeps > 1:
                    raise NotImplementedError()

                self._dict_op.update_swap(state, self._out_diagnostics[idx])
                self._dict_op.update_swap(state_prv, self._out_state[idx])
            else:
                try:
                    self._out_diagnostics[idx] = component(
                        state_prv, out=self._out_diagnostics[idx]
                    )
                except TypeError:
                    self._out_diagnostics[idx] = component(
                        state_prv, timestep, out=self._out_diagnostics[idx]
                    )

                self._dict_op.update_swap(
                    state_prv, self._out_diagnostics[idx]
                )

            # ensure state is still defined at current time level
            state["time"] = current_time

        # ensure the provisional state is defined at the next time level
        state_prv["time"] = current_time + timestep
