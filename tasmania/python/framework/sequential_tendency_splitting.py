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
    SequentialTendencySplitting
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
from tasmania.python.framework.sts_tendency_steppers import (
    STSTendencyStepper,
    tendencystepper_factory,
)
from tasmania.python.utils.framework_utils import (
    check_property_compatibility,
    get_input_properties,
    get_output_properties,
)


class SequentialTendencySplitting:
    """
    Callable class which integrates a bundle of physical processes pursuing
    the sequential tendency splitting strategy.

    Attributes
    ----------
    input_properties : dict
        Dictionary whose keys are strings denoting model variables
        which should be present in the input state, and whose values
        are dictionaries specifying fundamental properties (dims, units)
        of those variables.
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
                    the process forward in time. Available options:

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

                    'substeps' represents the number of substeps to carry out to
                    integrate the process. Defaults to 1.
                * 'time_integrator_kwargs' : TODO
        """
        self._component_list = []
        self._substeps = []
        for process in args:
            try:
                bare_component = process["component"]
            except KeyError:
                msg = "Missing mandatory key ''component'' in one item of ''processes''."
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
                kwargs = process.get(
                    "time_integrator_kwargs", {"backend": None, "halo": None}
                )

                TendencyStepper = tendencystepper_factory(integrator)
                self._component_list.append(
                    TendencyStepper(
                        bare_component,
                        execution_policy="serial",
                        enforce_horizontal_boundary=enforce_hb,
                        **kwargs
                    )
                )

                substeps = process.get("substeps", 1)
                self._substeps.append(substeps)

        # set properties
        self.input_properties = self._get_input_properties()
        self.provisional_input_properties = self._get_provisional_input_properties()
        self.output_properties = self._get_output_properties()
        self.provisional_output_properties = self._get_provisional_output_properties()

    def _get_input_properties(self):
        sts_list = tuple(
            component
            for component in self._component_list
            if isinstance(component, STSTendencyStepper)
        )
        return get_input_properties(sts_list, consider_diagnostics=True)

    def _get_provisional_input_properties(self):
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
                    check_property_compatibility(at_disposal[key], required[key])
                else:
                    at_disposal[key] = required[key]
                    return_dict[key] = required[key]

            for key in given:
                if key in at_disposal:
                    check_property_compatibility(at_disposal[key], given[key])
                else:
                    at_disposal[key] = given[key]

        return return_dict

    def _get_output_properties(self):
        return_dict = self._get_input_properties()
        sts_list = tuple(
            component
            for component in self._component_list
            if isinstance(component, STSTendencyStepper)
        )
        get_output_properties(
            sts_list,
            component_attribute_name="diagnostic_properties",
            return_dict=return_dict,
            consider_diagnostics=False,
        )
        return return_dict

    def _get_provisional_output_properties(self):
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

    def __call__(self, state, state_prv, timestep):
        """
        Advance the model state one timestep forward in time by pursuing
        the sequential-tendency splitting method.

        Parameters
        ----------
        state : dict
            The current state.
        state_prv : dict
            The provisional state.
        timestep : timedelta
            :class:`datetime.timedelta` representing the timestep size.

        Note
        ----
        :obj:`state_prv` is modified in-place to represent the final model state.
        """
        current_time = state["time"]

        for component, substeps in zip(self._component_list, self._substeps):
            if not isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics, state_tmp = component(state, state_prv, timestep / substeps)

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
                diagnostics = component(state_prv)
                state_prv.update(diagnostics)

            # ensure state is still defined at current time level
            state["time"] = current_time

        # ensure the provisional state is defined at the next time level
        state_prv["time"] = current_time + timestep
