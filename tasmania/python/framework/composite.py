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
from sympl import (
    DiagnosticComponent,
    TendencyComponent,
    ImplicitTendencyComponent,
    combine_component_properties,
)

from tasmania.python.framework._base import (
    BaseConcurrentCoupling,
    BaseDiagnosticComponentComposite,
)
from tasmania.python.utils import taz_types
from tasmania.python.utils.framework_utils import get_input_properties
from tasmania.python.utils.utils import assert_sequence


class DiagnosticComponentComposite(BaseDiagnosticComponentComposite):
    """
    Callable class wrapping and chaining a set of component computing *only* diagnostics.

    Attributes
    ----------
    input_properties : dict[str, dict]
        Dictionary whose keys are strings denoting model variables
        which should be present in the input state dictionary, and
        whose values are dictionaries specifying fundamental properties
        (dims, units) of those variables.
    diagnostic_properties : dict[str, dict]
        Dictionary whose keys are strings denoting model variables
        retrieved from the input state dictionary, and whose values
        are dictionaries specifying fundamental properties (dims, units)
        of those variables.
    output_properties : dict[str, dict]
        Dictionary whose keys are strings denoting model variables which
        will be present in the input state dictionary when the call operator
        returns, and whose values are dictionaries specifying fundamental
        properties (dims, units) for those variables.
    """

    allowed_diagnostic_type = (
        DiagnosticComponent,
        BaseDiagnosticComponentComposite,
    )
    allowed_tendency_type = (
        BaseConcurrentCoupling,
        ImplicitTendencyComponent,
        TendencyComponent,
    )
    allowed_component_type = allowed_diagnostic_type + allowed_tendency_type

    def __init__(
        self,
        *args: taz_types.diagnostic_component_t,
        execution_policy: str = "serial"
    ) -> None:
        """
        Parameters
        ----------
        *args : obj
            The components to wrap and chain.
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
        # assert_sequence(args, reftype=self.__class__.allowed_component_type)

        # ensure that all the components compute only diagnostics
        for component in args:
            tendency_properties = getattr(component, "tendency_properties", {})
            assert (
                len(tendency_properties) == 0
            ), "Component {} computes tendencies, which is not allowed.".format(
                type(component)
            )

        self._components_list = args

        self.input_properties = get_input_properties(
            tuple(
                {
                    "component": component,
                    "attribute_name": "input_properties",
                    "consider_diagnostics": execution_policy == "serial",
                }
                for component in self._components_list
            )
        )
        self.diagnostic_properties = combine_component_properties(
            self._components_list, "diagnostic_properties"
        )

        self._call = (
            self._call_serial
            if execution_policy == "serial"
            else self._call_asparallel
        )

    def __call__(
        self,
        state: taz_types.dataarray_dict_t,
        timestep: taz_types.timedelta_t,
    ):
        """
        Retrieve diagnostics from the input state by sequentially calling
        the wrapped :class:`sympl.DiagnosticComponent`\s, and incrementally
        update the input state with those diagnostics.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The input model state as a dictionary whose keys are strings denoting
            model variables, and whose values are :class:`sympl.DataArray`\s storing
            data for those variables.
        timestep : datetime.timedelta
            The model timestep.

        Return
        ------
        dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting diagnostic variables,
            and whose values are :class:`sympl.DataArray`\s storing data for
            those variables.
        """
        return self._call(state, timestep)

    def _call_serial(
        self,
        state: taz_types.dataarray_dict_t,
        timestep: taz_types.timedelta_t,
    ):
        return_dict = {}

        tmp_state = {}
        tmp_state.update(state)

        for component in self._components_list:
            if isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics = component(tmp_state)
            else:
                try:
                    _, diagnostics = component(tmp_state)
                except TypeError:
                    _, diagnostics = component(tmp_state, timestep)

            tmp_state.update(diagnostics)
            return_dict.update(diagnostics)

        return return_dict

    def _call_asparallel(self, state, timestep):
        return_dict = {}

        for component in self._components_list:
            if isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics = component(state)
            else:
                try:
                    _, diagnostics = component(state)
                except TypeError:
                    _, diagnostics = component(state, timestep)

            return_dict.update(diagnostics)

        return return_dict
