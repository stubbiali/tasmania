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
import abc
import copy
import numpy as np
from sympl import (
    DiagnosticComponent,
    DiagnosticComponentComposite as SymplDiagnosticComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
    combine_component_properties,
)
from sympl._core.units import clean_units

from tasmania.python.framework._base import (
    BaseConcurrentCoupling,
    BaseDiagnosticComponentComposite,
)
from tasmania.python.framework.promoters import Diagnostic2Tendency, Tendency2Diagnostic
from tasmania.python.utils.dict_utils import DataArrayDictOperator
from tasmania.python.utils.framework_utils import (
    check_t2d,
    check_properties_compatibility,
    get_input_properties,
    get_tendency_properties,
)
from tasmania.python.utils.utils import assert_sequence


class ConcurrentCoupling(BaseConcurrentCoupling):
    """
    Callable class which automates the execution of a bundle of physical
    parameterizations pursuing the *explicit* concurrent coupling strategy.

    Attributes
    ----------
    input_properties : dict[str, dict]
        Dictionary whose keys are strings denoting model variables
        which should be present in the input state dictionary, and
        whose values are dictionaries specifying fundamental properties
        (dims, units) of those variables.
    tendency_properties : dict[str, dict]
        Dictionary whose keys are strings denoting the model variables
        for	which tendencies are computed, and whose values are
        dictionaries specifying fundamental properties (dims, units)
        of those variables.
    diagnostics_properties : dict[str, dict]
        Dictionary whose keys are strings denoting the diagnostics
        which are retrieved, and whose values are dictionaries
        specifying fundamental properties (dims, units) of those variables.

    References
    ----------
    Staniforth, A., N. Wood, and J. C\^ot\'e. (2002). \
        A simple comparison of four physics-dynamics coupling schemes. \
        *Mon. Weather Rev.*, *130*:3129-3135.
    """

    allowed_diagnostic_type = (
        DiagnosticComponent,
        SymplDiagnosticComponentComposite,
        BaseDiagnosticComponentComposite,
    )
    allowed_tendency_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        BaseConcurrentCoupling,
    )
    allowed_promoter_type = (Diagnostic2Tendency, Tendency2Diagnostic)
    allowed_component_type = (
        allowed_diagnostic_type + allowed_tendency_type + allowed_promoter_type
    )

    def __init__(
        self,
        *args,
        execution_policy="serial",
        gt_powered=False,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=np.float64,
        rebuild=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        *args : obj
            Instances of

                * :class:`sympl.DiagnosticComponent`,
                * :class:`sympl.DiagnosticComponentComposite`,
                * :class:`tasmania.DiagnosticComponentComposite`,
                * :class:`sympl.TendencyComponent`,
                * :class:`sympl.TendencyComponentComposite`,
                * :class:`sympl.ImplicitTendencyComponent`,
                * :class:`sympl.ImplicitTendencyComponentComposite`,
                * :class:`tasmania.ConcurrentCoupling`, or
                * :class:`tasmania.TendencyPromoter`

            representing the components to wrap.
        execution_policy : `str`, optional
            String specifying the runtime mode in which parameterizations
            should be invoked. Either:

                * 'serial', to call the physical packages sequentially, and
                    make diagnostics computed by a component be available to
                    subsequent components;
                * 'as_parallel', to call the physical packages *as* we
                    were in a parallel region where each package may be
                    assigned to a different thread/process; in other terms,
                    we do not rely on the order in which parameterizations
                    are passed to this object, and diagnostics computed by
                    a component are not usable by any other component.

        gt_powered : `bool`, optional
            `True` to add the tendencies using GT4Py (leveraging field versioning),
            `False` to perform the summation in plain Python.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages passed to the stencil.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        **kwargs:
            Catch-all for unused keyword arguments.
        """
        assert_sequence(args, reftype=self.__class__.allowed_component_type)
        self._component_list = args

        self._policy = execution_policy
        self._call = (
            self._call_serial if execution_policy == "serial" else self._call_asparallel
        )

        # ensure that a tendency is actually computed before it gets moved around
        # by a Tendency2Diagnostic
        check_t2d(args, Tendency2Diagnostic)

        # set properties
        self.input_properties = self._init_input_properties()
        self.tendency_properties = self._init_tendency_properties()
        self.diagnostic_properties = self._init_diagnostic_properties()

        # ensure that dimensions and units of the variables present
        # in both input_properties and tendency_properties are compatible
        # across the two dictionaries
        output_properties = copy.deepcopy(self.tendency_properties)
        for key in output_properties:
            output_properties[key]["units"] = clean_units(
                self.tendency_properties[key]["units"] + " s"
            )
        check_properties_compatibility(
            self.input_properties,
            output_properties,
            properties1_name="input_properties",
            properties2_name="output_properties",
        )

        self._dict_op = DataArrayDictOperator(
            gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            rebuild=rebuild,
        )

    def _init_input_properties(self):
        t2d_type = Tendency2Diagnostic
        components_list = []
        for component in self.component_list:
            components_list.append(
                {
                    "component": component,
                    "attribute_name": "input_properties"
                    if not isinstance(component, t2d_type)
                    else None,
                    "consider_diagnostics": self._policy == "serial",
                }
            )
        return get_input_properties(components_list)

    def _init_tendency_properties(self):
        t2d_type = Tendency2Diagnostic
        return get_tendency_properties(self.component_list, t2d_type)

    def _init_diagnostic_properties(self):
        return combine_component_properties(self.component_list, "diagnostic_properties")

    @property
    def component_list(self):
        """
        Return
        ------
        tuple :
            The wrapped components.
        """
        return self._component_list

    def __call__(self, state, timestep):
        """
        Execute the wrapped components to calculate tendencies and retrieve
        diagnostics with the help of the input state.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The input model state as a dictionary whose keys are strings denoting
            model variables, and whose values are :class:`sympl.DataArray`\s storing
            data for those variables.
        timestep : `timedelta`, optional
            The time step size.

        Return
        ------
        dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting the model variables for which
            tendencies have been computed, and whose values are :class:`sympl.DataArray`\s
            storing the tendencies for those variables.
        """
        tendencies, diagnostics = self._call(state, timestep)

        try:
            tendencies["time"] = state["time"]
            diagnostics["time"] = state["time"]
        except KeyError:
            pass

        return tendencies, diagnostics

    def _call_serial(self, state, timestep):
        """ Process the components in 'serial' runtime mode. """
        aux_state = {}
        aux_state.update(state)

        out_tendencies = {}
        out_diagnostics = {}

        for component in self._component_list:
            if isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics = component(aux_state)
                aux_state.update(diagnostics)
                out_diagnostics.update(diagnostics)
            elif isinstance(component, self.__class__.allowed_tendency_type):
                try:
                    tendencies, diagnostics = component(aux_state)
                except TypeError:
                    tendencies, diagnostics = component(aux_state, timestep)

                self._dict_op.iadd(
                    out_tendencies,
                    tendencies,
                    field_properties=self.tendency_properties,
                    unshared_variables_in_output=True,
                )
                aux_state.update(diagnostics)
                out_diagnostics.update(diagnostics)
            elif isinstance(component, Tendency2Diagnostic):
                diagnostics = component(out_tendencies)
                aux_state.update(diagnostics)
                out_diagnostics.update(diagnostics)
            else:  # diagnostic to tendency
                tendencies = component(aux_state)
                self._dict_op.iadd(
                    out_tendencies,
                    tendencies,
                    field_properties=self.tendency_properties,
                    unshared_variables_in_output=True,
                )

        return out_tendencies, out_diagnostics

    def _call_asparallel(self, state, timestep):
        """ Process the components in 'as_parallel' runtime mode. """
        out_tendencies = {}
        out_diagnostics = {}

        for component in self._component_list:
            if isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics = component(state)
                out_diagnostics.update(diagnostics)
            elif isinstance(component, self.__class__.allowed_tendency_type):
                try:
                    tendencies, diagnostics = component(state)
                except TypeError:
                    tendencies, diagnostics = component(state, timestep)

                self._dict_op.iadd(
                    out_tendencies,
                    tendencies,
                    field_properties=self.tendency_properties,
                    unshared_variables_in_output=True,
                )
                out_diagnostics.update(diagnostics)
            elif isinstance(component, Tendency2Diagnostic):
                # do nothing
                pass
            else:  # diagnostics to tendencies
                tendencies = component(state)
                self._dict_op.iadd(
                    out_tendencies,
                    tendencies,
                    field_properties=self.tendency_properties,
                    unshared_variables_in_output=True,
                )

        return out_tendencies, out_diagnostics
