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

from __future__ import annotations
import abc
from typing import TYPE_CHECKING

from sympl import (
    DiagnosticComponent,
    DiagnosticComponentComposite as SymplDiagnosticComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
)
from sympl._core.dynamic_checkers import InflowComponentChecker, OutflowComponentChecker
from sympl._core.dynamic_operators import InflowComponentOperator, OutflowComponentOperator
from sympl._core.static_checkers import StaticComponentChecker
from sympl._core.time import FakeTimer as Timer

from tasmania.framework.base_components import DomainComponent, GridComponent
from tasmania.framework.composite import (
    DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite,
)
from tasmania.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.framework.dycore_utils import DynamicOperator, StaticChecker, StaticOperator
from tasmania.framework.stencil import StencilFactory
from tasmania.utils.xarrayx import DataArrayDictOperator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Optional, Union

    from tasmania.domain.domain import Domain
    from tasmania.framework.options import BackendOptions, StorageOptions
    from tasmania.utils.typingx import DataArrayDict, NDArray, NDArrayDict, PropertyDict, TimeDelta


class DynamicalCore(DomainComponent, StencilFactory, abc.ABC):
    """A dynamical core which implements a multi-stage time-marching scheme
    coupled with some form of partial time-splitting (i.e. substepping)."""

    allowed_tendency_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        ConcurrentCoupling,
    )
    allowed_diagnostic_type = (
        DiagnosticComponent,
        SymplDiagnosticComponentComposite,
        TasmaniaDiagnosticComponentComposite,
        ConcurrentCoupling,
    )

    def __init__(
        self,
        domain: Domain,
        fast_tendency_component: Optional[TendencyComponent] = None,
        fast_diagnostic_component: Optional[Union[DiagnosticComponent, TendencyComponent]] = None,
        substeps: int = 0,
        superfast_tendency_component: Optional[TendencyComponent] = None,
        superfast_diagnostic_component: Optional[DiagnosticComponent] = None,
        *,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional[BackendOptions] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional[StorageOptions] = None,
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        fast_tendency_component : `obj`, optional
            An instance of either

            * :class:`~sympl.TendencyComponent`,
            * :class:`~sympl.TendencyComponentComposite`,
            * :class:`~sympl.ImplicitTendencyComponent`,
            * :class:`~sympl.ImplicitTendencyComponentComposite`, or
            * :class:`~tasmania.ConcurrentCoupling`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the beginning of each stage on the latest
            provisional state.
        fast_diagnostic_component : `obj`, optional
            An instance of either

            * :class:`sympl.TendencyComponent`,
            * :class:`sympl.TendencyComponentComposite`,
            * :class:`sympl.ImplicitTendencyComponent`,
            * :class:`sympl.ImplicitTendencyComponentComposite`,
            * :class:`tasmania.ConcurrentCoupling`,
            * :class:`sympl.DiagnosticComponent`,
            * :class:`sympl.DiagnosticComponentComposite`, or
            * :class:`tasmania.DiagnosticComponentComposite`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the end of each stage on the latest
            provisional state, once the substepping routine is over.
        substeps : `int`, optional
            Number of substeps to perform. Defaults to 0, meaning that no
            form of substepping is carried out.
        superfast_tendency_component : `obj`, optional
            An instance of either

            * :class:`sympl.TendencyComponent`,
            * :class:`sympl.TendencyComponentComposite`,
            * :class:`sympl.ImplicitTendencyComponent`,
            * :class:`sympl.ImplicitTendencyComponentComposite`, or
            * :class:`tasmania.ConcurrentCoupling`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the beginning of each substep on the
            latest provisional state. This parameter is ignored if ``substeps``
            is not positive.
        superfast_diagnostic_component : `obj`, optional
            An instance of either

            * :class:`sympl.DiagnosticComponent`,
            * :class:`sympl.DiagnosticComponentComposite`, or
            * :class:`tasmania.DiagnosticComponentComposite`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the end of each substep on the latest
            provisional state.
        enable_checks : `bool`, optional
            TODO
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            TODO
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(domain, "numerical")
        super(GridComponent, self).__init__(backend, backend_options, storage_options)
        self._initialized = True

        # store input arguments
        self._fast_tc = fast_tendency_component
        self._fast_dc = fast_diagnostic_component
        self._substeps = substeps if substeps >= 0 else 0
        self._superfast_tc = superfast_tendency_component
        self._superfast_dc = superfast_diagnostic_component
        self._enable_checks = enable_checks

        # run static checks
        if enable_checks:
            StaticChecker.check(self)

        # set default storage shape
        self.storage_shape = self.get_storage_shape(storage_shape)

        # initialize properties
        self.input_properties = StaticOperator.get_input_properties(self)
        self.input_tendency_properties = StaticOperator.get_input_tendency_properties(self)
        self.output_properties = StaticOperator.get_output_properties(self)

        # wrap auxiliary components in ConcurrentCoupling objects
        self._fast_tc = StaticOperator.wrap_component(self, self._fast_tc)
        self._fast_dc = StaticOperator.wrap_component(self, self._fast_dc)
        self._superfast_tc = StaticOperator.wrap_component(self, self._superfast_tc)
        self._superfast_dc = StaticOperator.wrap_component(self, self._superfast_dc)

        if enable_checks:
            # run static checks
            StaticComponentChecker.factory("input_properties").check(self)
            StaticComponentChecker.factory("input_tendency_properties").check(self)
            StaticComponentChecker.factory("output_properties").check(self)

            # instantiate dynamic checkers
            self._input_checker = InflowComponentChecker.factory("input_properties", self)
            self._input_tendency_checker = InflowComponentChecker.factory(
                "input_tendency_properties", self
            )
            self._output_inflow_checker = InflowComponentChecker.factory("output_properties", self)
            self._output_outflow_checker = OutflowComponentChecker.factory(
                "output_properties", self
            )
            self._stage_input_checker = InflowComponentChecker.factory(
                "stage_input_properties", self
            )
            self._stage_tendency_checker = InflowComponentChecker.factory(
                "stage_tendency_properties", self
            )
            self._stage_output_checker = OutflowComponentChecker.factory(
                "stage_output_properties", self
            )

        # instantiate dynamic operators
        self._stage_input_operator = InflowComponentOperator.factory("stage_input_properties", self)
        self._stage_tendency_operator = InflowComponentOperator.factory(
            "stage_tendency_properties", self
        )
        self._output_inflow_operator = InflowComponentOperator.factory("output_properties", self)
        self._output_outflow_operator = OutflowComponentOperator.factory("output_properties", self)
        self._stage_output_operator = OutflowComponentOperator.factory(
            "stage_output_properties", self
        )
        self._dynamic_operator = DynamicOperator(self)

        # instantiate the dictionary operator
        self._dict_operator = DataArrayDictOperator(
            backend=self.backend,
            backend_options=self.backend_options,
            storage_options=self.storage_options,
        )

        # auxiliary variables
        self._fast_tendencies: Optional[DataArrayDict] = None
        self._fast_tendency_component_diagnostics: Optional[Sequence[DataArrayDict]] = None
        self._fast_diagnostic_component_diagnostics: Optional[Sequence[DataArrayDict]] = None
        self._raw_stage_states: Optional[Sequence[NDArrayDict]] = None

    @property
    def fast_tendency_component(self):
        return self._fast_tc

    @property
    def fast_diagnostic_component(self):
        return self._fast_dc

    @property
    def superfast_tendency_component(self):
        return self._superfast_tc

    @property
    def superfast_diagnostic_component(self):
        return self._superfast_dc

    @property
    def substeps(self):
        return max(0, int(self._substeps))

    @property
    def enable_checks(self) -> bool:
        return self._enable_checks

    @property
    @abc.abstractmethod
    def stage_input_properties(self) -> PropertyDict:
        """
        Dictionary whose keys are strings denoting variables which
        should be included in any state passed to the ``stage_array_call``, and
        whose values are fundamental properties (dims, units, alias)
        of those variables.
        """

    @property
    @abc.abstractmethod
    def substep_input_properties(self) -> PropertyDict:
        """
        Dictionary whose keys are strings denoting variables which
        should be included in any state passed to the ``substep_array_call``
        carrying out the substepping routine, and whose values are
        fundamental properties (dims, units, alias) of those variables.
        """

    @property
    @abc.abstractmethod
    def stage_tendency_properties(self) -> PropertyDict:
        """
        Dictionary whose keys are strings denoting (slow and intermediate)
        tendencies which may (or may not) be passed to ``stage_array_call``,
        and whose values are fundamental properties (dims, units, alias)
        of those tendencies.
        """

    @property
    @abc.abstractmethod
    def substep_tendency_properties(self) -> PropertyDict:
        """
        Dictionary whose keys are strings denoting (slow, intermediate and fast)
        tendencies which may (or may not) be passed to ``substep_array_call``,
        and whose values are fundamental properties (dims, units, alias)
        of those tendencies.
        """

    @property
    @abc.abstractmethod
    def stage_output_properties(self) -> PropertyDict:
        """
        Dictionary whose keys are strings denoting variables which are
        included in the output state returned by ``stage_array_call``,
        and whose values are fundamental properties (dims, units)
        of those variables.
        """

    @property
    @abc.abstractmethod
    def substep_output_properties(self) -> PropertyDict:
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting variables which are
            included in the output state returned by any substep, and whose
            values are fundamental properties (dims, units) of those variables.
        """

    @property
    @abc.abstractmethod
    def stages(self) -> int:
        """Number of stages carried out by the dynamical core."""

    @property
    @abc.abstractmethod
    def substep_fractions(self) -> Union[float, Sequence[float]]:
        """
        For each stage, fraction of the total number of substeps
        (specified at instantiation) to carry out.
        """

    def allocate_stage_output(self, name) -> NDArray:
        """Allocate memory for an output field."""
        return self.zeros(shape=self.get_field_storage_shape(name, self.storage_shape))

    def allocate_stage_outputs(self) -> NDArrayDict:
        """Allocate memory for the return state."""
        out = {name: self.allocate_stage_output(name) for name in self.stage_output_properties}
        return out

    def allocate_substep_output(self, name) -> NDArray:
        """Allocate memory for an output field."""
        return self.zeros(shape=self.get_field_storage_shape(name, self.storage_shape))

    def allocate_substep_outputs(self) -> NDArrayDict:
        """Allocate memory for the return state."""
        out = {name: self.allocate_substep_output(name) for name in self.substep_output_properties}
        return out

    def allocate_fast_tendencies_and_diagnostics(self, state: DataArrayDict) -> None:
        self._fast_tendencies = (
            self.fast_tendency_component.allocate_tendency_dict(state)
            if self.fast_tendency_component is not None
            else {}
        )
        if self.fast_diagnostic_component is not None:
            self._fast_tendencies.update(
                self.fast_diagnostic_component.allocate_tendency_dict(state)
            )

        self._fast_tendency_component_diagnostics = []
        self._fast_diagnostic_component_diagnostics = []
        for _ in range(self.stages):
            self._fast_tendency_component_diagnostics.append(
                self.fast_tendency_component.allocate_diagnostic_dict(state)
                if self.fast_tendency_component is not None
                else {}
            )
            self._fast_diagnostic_component_diagnostics.append(
                self.fast_diagnostic_component.allocate_diagnostic_dict(state)
                if self.fast_diagnostic_component is not None
                else {}
            )

    def __call__(
        self,
        state: DataArrayDict,
        tendencies: DataArrayDict,
        timestep: TimeDelta,
        *,
        out_state: Optional[DataArrayDict] = None,
    ) -> DataArrayDict:
        """Advance the input state one timestep forward.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`~sympl.DataArray`\s storing values
            for those variables.
        tendencies : dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`~sympl.DataArray`\s storing tendencies
            for those variables.
        timestep : datetime.timedelta
            The step size, i.e. the amount of time to step forward.

        Return
        ------
        dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting model variables,
            and whose values are :class:`~sympl.DataArray`\s storing values
            for those variables at the next time level.

        Warning
        -------
        Variable aliasing is not supported at the moment.
        """
        # run checks on input dictionaries
        if self.enable_checks:
            self._input_checker.check(state)
            self._input_tendency_checker.check(tendencies, state)

        # allocate memory for fast tendencies and fast diagnostics
        if self._fast_tendencies is None:
            self.allocate_fast_tendencies_and_diagnostics(state)

        # run checks on output dictionary
        out_state = out_state if out_state is not None else {}
        if self.enable_checks:
            self._output_inflow_checker.check(out_state, state)

        # extract or allocate output buffers
        raw_out_state = self._output_inflow_operator.get_ndarray_dict(out_state)
        raw_out_state.update(
            {
                name: self.allocate_stage_output(name)
                for name in self.stage_output_properties
                if name not in out_state
            }
        )
        raw_out_state.update(
            {
                name: self.allocate_substep_output(name)
                for name in self.substep_output_properties
                if name not in out_state
            }
        )

        # allocate states output by each stage
        if self._raw_stage_states is None:
            self._raw_stage_states = [self.allocate_stage_outputs() for _ in range(self.stages - 1)]
            self._raw_stage_states.append(raw_out_state)
        else:
            self._raw_stage_states[-1] = raw_out_state

        stage_state, tmp_state = state, None
        for stage in range(self.stages):
            stage_state, tmp_state = tmp_state, stage_state
            stage_state = self.call(stage, timestep, state, tendencies, tmp_state)

        out_state.update(stage_state)

        return out_state

    def call(
        self,
        stage: int,
        timestep: TimeDelta,
        state: DataArrayDict,
        slow_tendencies: DataArrayDict,
        tmp_state: DataArrayDict,
    ) -> DataArrayDict:
        """Perform a single stage of the time integration algorithm.

        Parameters
        ----------
        stage : int
            The stage identifier.
        timestep : datetime.timedelta
            The step size, i.e. the amount of time to step forward.
        state : dict[str, sympl.DataArray]
            The state at the current time level.
        tmp_state : dict[str, sympl.DataArray]
            The provisional state calculated by the previous stage.
            It coincides with ``state`` when ``stage = 0``.
        slow_tendencies : dict[str, sympl.DataArray]
            The *slow* physics tendencies for the prognostic model variables.
        """
        # shortcuts
        ftc = self.fast_tendency_component
        fdc = self.fast_diagnostic_component
        ftc_diagnostics = self._fast_tendency_component_diagnostics
        fdc_diagnostics = self._fast_diagnostic_component_diagnostics
        ftc_slow = self._dynamic_operator.get_fast_and_slow_tendencies(
            self.fast_tendency_component, slow_tendencies
        )
        fdc_slow = self._dynamic_operator.get_fast_and_slow_tendencies(
            self.fast_diagnostic_component, slow_tendencies
        )

        # ============================================================
        # Calculating fast tendencies and diagnostics
        # ============================================================
        Timer.start(label="add_slow_and_fast_tendencies")
        # add the slow and fast tendencies up
        if stage == 0:
            self._fast_tendencies.update(
                {
                    key: slow_tendencies[key]
                    for key in slow_tendencies
                    if key not in ftc_slow and key not in fdc_slow
                }
            )
        self._dict_operator.iadd(
            self._fast_tendencies,
            {key: slow_tendencies[key] for key in fdc_slow},
            field_properties=self.input_tendency_properties,
            unshared_variables_in_output=False,
        )
        self._dict_operator.copy(
            self._fast_tendencies,
            {key: slow_tendencies[key] for key in ftc_slow if key not in fdc_slow},
            unshared_variables_in_output=False,
        )
        Timer.stop()

        if ftc is not None:
            Timer.start(label="call_fast_tendency_component")
            # calculate fast tendencies and diagnostics
            overwrite_tendencies = self._dynamic_operator.get_ovewrite_tendencies(slow_tendencies)
            ftc(
                tmp_state,
                timestep,
                out_tendencies=self._fast_tendencies,
                out_diagnostics=ftc_diagnostics[stage],
                overwrite_tendencies=overwrite_tendencies,
            )

            # update the state with the just computed diagnostics
            self._dict_operator.update_swap(
                tmp_state,
                {name: ftc_diagnostics[stage][name] for name in ftc.diagnostic_properties},
            )
            Timer.stop()

        # ============================================================
        # Stage: pre-processing
        # ============================================================
        if self.enable_checks:
            self._stage_input_checker.check(tmp_state)
            self._stage_tendency_checker.check(self._fast_tendencies)

        Timer.start(label="get_raw_tmp_state")
        # Extract raw storages from state
        raw_tmp_state = self._stage_input_operator.get_ndarray_dict(tmp_state)
        Timer.stop()

        Timer.start(label="get_raw_tends")
        # Extract raw storages from tendencies
        raw_tends = self._stage_tendency_operator.get_ndarray_dict(self._fast_tendencies)
        Timer.stop()

        # ============================================================
        # Stage: computing
        # ============================================================
        # Carry out the stage
        Timer.start(label="stage")
        self.stage_array_call(
            stage,
            raw_tmp_state,
            raw_tends,
            timestep,
            self._raw_stage_states[stage],
        )
        Timer.stop()

        if self.substeps == 0 or len(self.substep_output_properties) == 0:
            # ============================================================
            # Stage: post-processing, substepping disabled
            # ============================================================
            if self.enable_checks:
                self._stage_output_checker.check(self._raw_stage_states[stage], state)

            Timer.start(label="get_stage_state")
            # Create dataarrays out of the ndarrays contained in the
            # stepped state
            stage_state = self._stage_output_operator.get_dataarray_dict(
                self._raw_stage_states[stage], state
            )
            Timer.stop()
        else:
            # TODO: deprecated!
            raise NotImplementedError()

            # # ============================================================
            # # Stage: post-processing, substepping enabled
            # # ============================================================
            # # Create dataarrays out of the numpy arrays contained in the stepped state
            # # which represent variables which will not be affected by the substepping
            # raw_nosubstep_stage_state = {
            #     name: raw_stage_state[name]
            #     for name in raw_stage_state
            #     if name not in self._substep_output_properties
            # }
            # nosubstep_stage_state_units = {
            #     name: self._output_properties[name]["units"]
            #     for name in self._output_properties
            #     if name not in self._substep_output_properties
            # }
            # nosubstep_stage_state = get_dataarray_dict(
            #     raw_nosubstep_stage_state, self._grid, units=nosubstep_stage_state_units
            # )
            #
            # substep_frac = 1.0 if self.stages == 1 else self.substep_fractions[stage]
            # substeps = int(substep_frac * self._substeps)
            # for substep in range(substeps):
            #     # ============================================================
            #     # Calculating the fast tendencies
            #     # ============================================================
            #     if self._fast_tc is None:
            #         tends = {}
            #     else:
            #         try:
            #             tends, diags = self._fast_tc(out_state)
            #         except TypeError:
            #             tends, diags = self._fast_tc(
            #                 out_state, timestep / self._substeps
            #             )
            #
            #         out_state.update(diags)
            #
            #     # ============================================================
            #     # Substep: pre-processing
            #     # ============================================================
            #     # Extract numpy arrays from the latest state
            #     out_state_units = {
            #         name: self._substep_input_properties[name]["units"]
            #         for name in self._substep_input_properties.keys()
            #     }
            #     raw_out_state = get_array_dict(out_state, units=out_state_units)
            #
            #     # Extract numpy arrays from fast tendencies
            #     tends_units = {
            #         name: self._substep_tendency_properties[name]["units"]
            #         for name in self._substep_tendency_properties.keys()
            #     }
            #     raw_tends = get_array_dict(tends, units=tends_units)
            #
            #     # ============================================================
            #     # Substep: computing
            #     # ============================================================
            #     # Carry out the substep
            #     raw_substep_state = self.substep_array_call(
            #         stage,
            #         substep,
            #         state,
            #         raw_stage_state,
            #         raw_out_state,
            #         raw_tends,
            #         timestep,
            #     )
            #
            #     # ============================================================
            #     # Substep: post-processing
            #     # ============================================================
            #     # Create dataarrays out of the numpy arrays contained in substepped state
            #     substep_state_units = {
            #         name: self._substep_output_properties[name]["units"]
            #         for name in self._substep_output_properties
            #     }
            #     substep_state = get_dataarray_dict(
            #         raw_substep_state, self._grid, units=substep_state_units
            #     )
            #
            #     # ============================================================
            #     # Retrieving the fast diagnostics
            #     # ============================================================
            #     if self._fast_dc is not None:
            #         fast_diags = self._fast_dc(substep_state)
            #         substep_state.update(fast_diags)
            #
            #     # Update the output state
            #     if substep < substeps - 1:
            #         out_state.update(substep_state)
            #     else:
            #         out_state = {}
            #         out_state.update(substep_state)
            #
            # # ============================================================
            # # Including the non-substepped variables
            # # ============================================================
            # out_state.update(nosubstep_stage_state)

        # ============================================================
        # Calculating fast tendencies and diagnostics
        # ============================================================
        if fdc is not None:
            Timer.start(label="call_fast_diagnostic_component")
            fdc(
                stage_state,
                timestep,
                out_tendencies=self._fast_tendencies,
                out_diagnostics=fdc_diagnostics[stage],
            )
            self._dict_operator.update_swap(
                stage_state,
                {name: fdc_diagnostics[stage][name] for name in fdc.diagnostic_properties},
            )
            Timer.stop()

        # Ensure the time specified in the output state is correct
        if stage == self.stages - 1:
            stage_state["time"] = state["time"] + timestep

        # ============================================================
        # Final checks
        # ============================================================
        if self.enable_checks:
            self._output_outflow_checker.check(stage_state, state)

        return stage_state

    @abc.abstractmethod
    def stage_array_call(
        self,
        stage: int,
        state: NDArrayDict,
        tendencies: NDArrayDict,
        timestep: TimeDelta,
        out_state: NDArrayDict,
    ) -> None:
        """Integrate the state over a stage.

        Parameters
        ----------
        stage : int
            The stage identifier.
        state : dict[str, array_like]
            The latest provisional state.
        tendencies : dict[str, array_like]
            The tendencies for the model prognostic variables.
        timestep : datetime.timedelta
            The step size.

        Return
        ------
        dict[str, array_like]
            The next provisional state.
        """

    @abc.abstractmethod
    def substep_array_call(
        self,
        stage: int,
        substep: int,
        state: NDArrayDict,
        stage_state: NDArrayDict,
        tmp_state: NDArrayDict,
        slow_tendencies: NDArrayDict,
        timestep: TimeDelta,
    ) -> NDArrayDict:
        """Integrate the state over a substep.

        Parameters
        ----------
        stage : int
            The stage.
        substep : int
            The substep.
        raw_state : dict[str, array_like]
            The raw state at the current *main* time level, i.e.,
            the raw version of the state dictionary passed to the call operator.
        raw_stage_state : dict[str, array_like]
            The (raw) state dictionary returned by the latest stage.
        raw_tmp_state : dict[str, array_like]
            The raw state to substep.
        raw_tendencies : dict[str, array_like]
            The (raw) tendencies for the model prognostic variables.
        timestep : datetime.timedelta
            The timestep size.

        Return
        ------
        dict[str, array_like]
            The substepped (raw) state.
        """

    def update_topography(self, time: TimeDelta) -> None:
        """Update the underlying (time-dependent) topography.

        Parameters
        ----------
        time : datetime.timedelta
            The elapsed simulation time.
        """
        self.grid.update_topography(time)
