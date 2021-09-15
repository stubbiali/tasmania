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
from typing import Dict, Optional, TYPE_CHECKING, Tuple, Union

from sympl._core.composite import (
    DiagnosticComponentComposite as SymplDiagnosticComponentComposite,
    ImplicitTendencyComponentComposite,
    TendencyComponentComposite,
)
from sympl._core.core_components import (
    DiagnosticComponent,
    TendencyComponent,
    ImplicitTendencyComponent,
)
from sympl._core.dynamic_checkers import (
    InflowComponentChecker,
    OutflowComponentChecker,
)
from sympl._core.static_checkers import StaticComponentChecker
from sympl._core.static_operators import StaticComponentOperator

from tasmania.python.framework._base import (
    BaseConcurrentCoupling,
    BaseDiagnosticComponentComposite,
)
from tasmania.python.framework.concurrent_coupling_utils import (
    DynamicOperator,
    StaticChecker,
    StaticOperator,
)
from tasmania.python.framework.promoter import (
    FromDiagnosticToTendency,
    FromTendencyToDiagnostic,
)
from tasmania.python.utils import typingx
from tasmania.python.utils.dict import DataArrayDictOperator

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict, NDArrayLike

    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.utils.typingx import PromoterComponent


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
    allowed_promoter_type = (
        FromDiagnosticToTendency,
        FromTendencyToDiagnostic,
    )
    allowed_component_type = (
        allowed_diagnostic_type + allowed_tendency_type + allowed_promoter_type
    )

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], BaseConcurrentCoupling):
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(
        self,
        *args: Union[
            DiagnosticComponent,
            TendencyComponent,
            "PromoterComponent",
        ],
        execution_policy: str = "serial",
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
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

        enable_checks : `bool`, optional
            ``True`` to run all sanity checks, ``False`` otherwise.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        if not getattr(self, "_initialized", False):
            super().__init__()

            self._components = args
            self._policy = (
                execution_policy
                if execution_policy in ("serial", "as_parallel")
                else "serial"
            )
            self._enable_checks = enable_checks
            self._call = (
                self._call_serial
                if execution_policy == "serial"
                else self._call_asparallel
            )

            # static checks
            if enable_checks:
                StaticChecker.check(self)

            # set properties
            self.input_properties = StaticOperator.get_input_properties(self)
            self.tendency_properties = StaticOperator.get_tendency_properties(
                self
            )
            self.diagnostic_properties = (
                StaticOperator.get_diagnostic_properties(self)
            )

            # double-check properties
            if enable_checks:
                StaticComponentChecker.factory("input_properties").check(self)
                StaticComponentChecker.factory("tendency_properties").check(
                    self
                )
                StaticComponentChecker.factory("diagnostic_properties").check(
                    self
                )

            # set overwrite_tendencies
            self.overwrite_tendencies = (
                StaticOperator.get_overwrite_tendencies(self)
            )

            # retrieve the object handling the horizontal boundary
            self.horizontal_boundary = StaticOperator.get_horizontal_boundary(
                self
            )

            # dynamic checkers
            if enable_checks:
                self._input_checker = InflowComponentChecker.factory(
                    "input_properties", self
                )
                self._tendency_inflow_checker = InflowComponentChecker.factory(
                    "tendency_properties", self
                )
                self._tendency_outflow_checker = (
                    OutflowComponentChecker.factory(
                        "tendency_properties", self
                    )
                )
                self._diagnostic_inflow_checker = (
                    InflowComponentChecker.factory(
                        "diagnostic_properties", self
                    )
                )
                self._diagnostic_outflow_checker = (
                    OutflowComponentChecker.factory(
                        "diagnostic_properties", self
                    )
                )

            # dynamic operators
            self._cc_operator = DynamicOperator(self)

            # dict operator
            self._dict_operator = DataArrayDictOperator(
                backend=backend,
                backend_options=backend_options,
                storage_options=storage_options,
            )

            # avoiding infinite loops
            self._initialized = True

    @property
    def components(
        self,
    ) -> Tuple[
        Union[
            DiagnosticComponent, TendencyComponent, typingx.PromoterComponent
        ]
    ]:
        """
        Return
        ------
        tuple :
            The wrapped components.
        """
        return self._components

    @property
    def execution_policy(self) -> str:
        return self._policy

    def allocate_tendency(self, name: str) -> "NDArrayLike":
        return self._cc_operator.allocate_tendency(name)

    def allocate_tendency_dict(
        self, state: "DataArrayDict"
    ) -> "DataArrayDict":
        return self._cc_operator.allocate_tendencies(state)

    def allocate_diagnostic(self, name: str) -> "NDArrayLike":
        return self._cc_operator.allocate_diagnostic(name)

    def allocate_diagnostic_dict(
        self, state: "DataArrayDict"
    ) -> "DataArrayDict":
        return self._cc_operator.allocate_diagnostics(state)

    def __call__(
        self,
        state: "DataArrayDict",
        timestep: typingx.TimeDelta,
        *,
        out_tendencies: Optional["DataArrayDict"] = None,
        out_diagnostics: Optional["DataArrayDict"] = None,
        overwrite_tendencies: Optional[Dict[str, bool]] = None
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        """
        Execute the wrapped components to calculate tendencies and retrieve
        diagnostics with the help of the input state.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The input model state as a dictionary whose keys are strings
            denoting model variables, and whose values are
            :class:`sympl.DataArray`\s storing data for those variables.
        timestep : `timedelta`, optional
            The time step size.

        Return
        ------
        dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting the model variables for
            which tendencies have been computed, and whose values are
            :class:`sympl.DataArray`\s storing the tendencies for those
            variables.
        """
        out_tendencies = out_tendencies if out_tendencies is not None else {}
        out_diagnostics = (
            out_diagnostics if out_diagnostics is not None else {}
        )
        overwrite_tendencies = (
            overwrite_tendencies if overwrite_tendencies is not None else {}
        )

        if self.execution_policy == "serial":
            self._call_serial(
                state,
                timestep,
                out_tendencies,
                out_diagnostics,
                overwrite_tendencies,
            )
        else:
            self._call_asparallel(
                state,
                timestep,
                out_tendencies,
                out_diagnostics,
                overwrite_tendencies,
            )

        if "time" in state:
            out_tendencies["time"] = state["time"]
            out_diagnostics["time"] = state["time"]

        return out_tendencies, out_diagnostics

    def _call_serial(
        self,
        state: "DataArrayDict",
        timestep: typingx.TimeDelta,
        out_tendencies: "DataArrayDict",
        out_diagnostics: "DataArrayDict",
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        """Process the components in 'serial' runtime mode."""
        sco = StaticComponentOperator.factory("diagnostic_properties")
        aux_state = {}
        aux_state.update(state)

        for component, self_overwrite_tendencies in zip(
            self.components, self.overwrite_tendencies
        ):
            if isinstance(component, self.allowed_diagnostic_type):
                component(aux_state, out=out_diagnostics)
            elif isinstance(component, self.allowed_tendency_type):
                ot = {
                    name: self_overwrite_tendencies[name]
                    and overwrite_tendencies.get(name, True)
                    for name in self_overwrite_tendencies
                }

                try:
                    component(
                        aux_state,
                        out_tendencies=out_tendencies,
                        out_diagnostics=out_diagnostics,
                        overwrite_tendencies=ot,
                    )
                except TypeError:
                    component(
                        aux_state,
                        timestep,
                        out_tendencies=out_tendencies,
                        out_diagnostics=out_diagnostics,
                        overwrite_tendencies=ot,
                    )

                # Timer.start(label="tendency_type, iadd")
                # self._dict_op.iadd(
                #     out_tendencies,
                #     tendencies,
                #     field_properties=self.tendency_properties,
                #     unshared_variables_in_output=True,
                # )
                # Timer.stop()

            elif isinstance(component, FromTendencyToDiagnostic):
                component(out_tendencies, out=out_diagnostics)
            else:  # diagnostic to tendency
                component(aux_state, out=out_tendencies)
                # Timer.start(label="diagnostic_to_tendency, iadd")
                # self._dict_operator.iadd(
                #     out_tendencies,
                #     tendencies,
                #     field_properties=self.tendency_properties,
                #     unshared_variables_in_output=True,
                # )
                # Timer.stop()

            sub_diagnostics = {
                name: out_diagnostics[name]
                for name in sco.get_properties(component)
            }
            aux_state.update(sub_diagnostics)

    def _call_asparallel(
        self,
        state: "DataArrayDict",
        timestep: typingx.TimeDelta,
        out_tendencies: "DataArrayDict",
        out_diagnostics: "DataArrayDict",
        overwrite_tendencies: Dict[str, bool],
    ) -> None:
        """Process the components in 'as_parallel' runtime mode."""
        for component, self_overwrite_tendencies in zip(
            self.components, self.overwrite_tendencies
        ):
            if isinstance(component, self.allowed_diagnostic_type):
                component(state, out=out_diagnostics)
            elif isinstance(component, self.allowed_tendency_type):
                ot = {
                    name: self_overwrite_tendencies[name]
                    and overwrite_tendencies.get(name, True)
                    for name in self_overwrite_tendencies
                }
                try:
                    component(
                        state,
                        out_tendencies=out_tendencies,
                        out_diagnostics=out_diagnostics,
                        overwrite_tendencies=ot,
                    )
                except TypeError:
                    component(
                        state,
                        timestep,
                        out_tendencies=out_tendencies,
                        out_diagnostics=out_diagnostics,
                        overwrite_tendencies=ot,
                    )
            elif isinstance(component, FromTendencyToDiagnostic):
                # do nothing
                pass
            else:  # diagnostics to tendencies
                component(state, out=out_tendencies)
                # self._dict_operator.iadd(
                #     out_tendencies,
                #     tendencies,
                #     field_properties=self.tendency_properties,
                #     unshared_variables_in_output=True,
                # )
