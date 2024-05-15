# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
from typing import Optional, TYPE_CHECKING, Tuple, Union

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

from tasmania.python.framework.composite import (
    DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite,
)
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.parallel_splitting_utils import StaticOperator
from tasmania.python.framework.static_checkers import (
    check_properties_are_compatible,
)
from tasmania.python.framework.steppers import TendencyStepper
from tasmania.python.utils import typingx
from tasmania.python.utils.dict import DataArrayDictOperator

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict

    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
        TimeIntegrationOptions,
    )


class ParallelSplitting:
    """
    Callable class which integrates a bundle of physical processes pursuing
    the parallel splitting strategy.

    Attributes
    ----------
    input_properties : dict[str, dict]
        Dictionary whose keys are strings denoting variables which
        should be present in the input model dictionary representing
        the current state, and whose values are dictionaries specifying
        fundamental properties (dims, units) of those variables.
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

    def __init__(
        self,
        *args: "TimeIntegrationOptions",
        execution_policy: str = "serial",
        retrieve_diagnostics_from_provisional_state: bool = False,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        *args : TimeIntegrationOptions
            TODO
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
            ``True`` (respectively, ``False``) to feed the
            :class:`sympl.DiagnosticComponent` objects with the provisional
            (resp., current) state, and add the so-retrieved diagnostics
            to the provisional (resp., current) state dictionary.
            Defaults to ``False``.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.
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
                        enforce_horizontal_boundary=options.enforce_horizontal_boundary,
                        enable_checks=options.enable_checks,
                        backend=options.backend,
                        backend_options=options.backend_options,
                        storage_options=options.storage_options,
                        **options.kwargs
                    )
                )
                self._substeps.append(max(options.substeps, 1))

        self._policy = execution_policy

        if (
            execution_policy == "as_parallel"
            and retrieve_diagnostics_from_provisional_state
        ):
            import warnings

            warnings.warn(
                "Argument retrieve_diagnostics_from_provisional_state "
                "only effective when execution policy set on 'serial'."
            )
            self._diagnostics_from_provisional = False
        else:
            self._diagnostics_from_provisional = (
                retrieve_diagnostics_from_provisional_state
            )

        # set properties
        self.input_properties = StaticOperator.get_input_properties(self)
        self.provisional_input_properties = (
            StaticOperator.get_provisional_input_properties(self)
        )
        self.output_properties = StaticOperator.get_output_properties(self)
        self.provisional_output_properties = (
            StaticOperator.get_provisional_output_properties(self)
        )

        # static checks
        check_properties_are_compatible(
            self, "input_properties", self, "output_properties"
        )
        check_properties_are_compatible(
            self,
            "provisional_input_properties",
            self,
            "provisional_output_properties",
        )

        self._dict_op = DataArrayDictOperator(
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )

        self._out_diagnostics = [None] * len(self.components)
        self._out_state = [None] * len(self.components)

    @property
    def components(
        self,
    ) -> Tuple[Union[typingx.DiagnosticComponent, TendencyStepper], ...]:
        """
        Return
        ------
        tuple :
            The wrapped components.
        """
        return tuple(self._component_list)

    def __call__(
        self,
        state: "DataArrayDict",
        state_prv: "DataArrayDict",
        timestep: typingx.TimeDelta,
    ) -> None:
        """
        Advance the model state one timestep forward in time by pursuing
        the parallel splitting method.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The model state at the current time level, i.e., at the beginning
            of the current timestep.
        state_prv : dict[str, sympl.DataArray]
            A provisional model state. Ideally, this should be the state output
            by the dynamical core, i.e., the outcome of a one-timestep time
            integration which takes only the dynamical processes into consideration.
        timestep : datetime.timedelta
            The time step.

        Note
        ----
        `state` may be modified in-place with the diagnostics retrieved
        from `state` itself.
        `state_prv` is modified in-place with the temporary states provided
        by each process. In other words, when this method returns, `state_prv`
        will represent the state at the next time level.
        """
        # step the solution
        if self._policy == "serial":
            self._call_serial(state, state_prv, timestep)
        else:
            self._call_asparallel(state, state_prv, timestep)

        # Ensure the provisional state is now defined at the next time level
        state_prv["time"] = state["time"] + timestep

    def _call_serial(
        self,
        state: "DataArrayDict",
        state_prv: "DataArrayDict",
        timestep: typingx.TimeDelta,
    ) -> None:
        """Process the components in 'serial' runtime mode."""
        for idx, component in enumerate(self.components):
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

                self._dict_op.iaddsub(
                    state_prv,
                    self._out_state[idx],
                    state,
                    field_properties=self.provisional_output_properties,
                )
                self._dict_op.update_swap(state, self._out_diagnostics[idx])
            else:
                arg = (
                    state_prv if self._diagnostics_from_provisional else state
                )

                try:
                    self._out_diagnostics[idx] = component(
                        arg, out=self._out_diagnostics[idx]
                    )
                except TypeError:
                    self._out_diagnostics[idx] = component(
                        arg, timestep, out=self._out_diagnostics[idx]
                    )

                self._dict_op.update_swap(arg, self._out_diagnostics[idx])

    def _call_asparallel(
        self,
        state: "DataArrayDict",
        state_prv: "DataArrayDict",
        timestep: typingx.TimeDelta,
    ) -> None:
        """Process the components in 'as_parallel' runtime mode."""
        agg_diagnostics = {}

        for component, substeps in zip(self.components, self._substeps):
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

                self._dict_op.iaddsub(
                    state_prv,
                    state_tmp,
                    state,
                    field_properties=self.provisional_output_properties,
                )
                agg_diagnostics.update(diagnostics)
            else:
                try:
                    diagnostics = component(state)
                except TypeError:
                    diagnostics = component(state, timestep)

                agg_diagnostics.update(diagnostics)

        state.update(agg_diagnostics)
