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
import numpy as np
from sympl import (
    DiagnosticComponent,
    DiagnosticComponentComposite as SymplDiagnosticComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
)
from sympl._core.base_components import InputChecker, OutputChecker
from typing import Optional, Sequence, TYPE_CHECKING, Union

from tasmania.python.framework.composite import (
    DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite,
)
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.tendency_checkers import SubsetTendencyChecker
from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import (
    get_array_dict,
    get_dataarray_dict,
)
from tasmania.python.utils.dict_utils import DataArrayDictOperator
from tasmania.python.utils.framework_utils import (
    check_properties_compatibility,
    check_missing_properties,
)
from tasmania.python.utils.utils import Timer

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain
    from tasmania.python.domain.grid import Grid
    from tasmania.python.domain.horizontal_boundary import HorizontalBoundary


class DynamicalCore(abc.ABC):
    """ A dynamical core which implements a multi-stage time-marching scheme
    coupled with some form of partial time-splitting (i.e. substepping). """

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
    )

    def __init__(
        self,
        domain: "Domain",
        grid_type: str,
        intermediate_tendency_component: Optional[
            taz_types.tendency_component_t
        ] = None,
        intermediate_diagnostic_component: Optional[
            Union[
                taz_types.diagnostic_component_t,
                taz_types.tendency_component_t,
            ]
        ] = None,
        substeps: int = 0,
        fast_tendency_component: Optional[
            taz_types.tendency_component_t
        ] = None,
        fast_diagnostic_component: Optional[
            taz_types.diagnostic_component_t
        ] = None,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        build_info: Optional[taz_types.options_dict_t] = None,
        rebuild: bool = False
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : str
            The type of grid over which instantiating the class.
            Either "physical" or "numerical".
        intermediate_tendency_component : `obj`, optional
            An instance of either

            * :class:`~sympl.TendencyComponent`,
            * :class:`~sympl.TendencyComponentComposite`,
            * :class:`~sympl.ImplicitTendencyComponent`,
            * :class:`~sympl.ImplicitTendencyComponentComposite`, or
            * :class:`~tasmania.ConcurrentCoupling`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the beginning of each stage on the latest
            provisional state.
        intermediate_diagnostic_component : `obj`, optional
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
        fast_tendency_component : `obj`, optional
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
        fast_diagnostic_component : `obj`, optional
            An instance of either

            * :class:`sympl.DiagnosticComponent`,
            * :class:`sympl.DiagnosticComponentComposite`, or
            * :class:`tasmania.DiagnosticComponentComposite`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the end of each substep on the latest
            provisional state.
        backend : `str`, optional
            The backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        dtype : `data-type`, optional
            Data type of the storages.
        build_info : `dict`, optional
            Dictionary of building options.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        """
        self._grid = (
            domain.physical_grid
            if grid_type == "physical"
            else domain.numerical_grid
        )
        self._hb = domain.horizontal_boundary
        self._dtype = dtype

        self._inter_tc = intermediate_tendency_component
        if self._inter_tc is not None:
            tend_type = self.__class__.allowed_tendency_type
            assert isinstance(self._inter_tc, tend_type), (
                "The input argument ''intermediate_tendencies'' "
                "should be an instance of either {}.".format(
                    ", ".join(str(item) for item in tend_type)
                )
            )

        self._inter_dc = intermediate_diagnostic_component
        if self._inter_dc is not None:
            diag_type = (
                self.__class__.allowed_diagnostic_type
                + self.__class__.allowed_tendency_type
            )
            assert isinstance(self._inter_dc, diag_type), (
                "The input argument ''intermediate_diagnostics'' "
                "should be an instance of either {}.".format(
                    ", ".join(str(item) for item in diag_type)
                )
            )

        self._substeps = substeps if substeps >= 0 else 0

        if self._substeps >= 0:
            self._fast_tc = fast_tendency_component
            if self._fast_tc is not None:
                tend_type = self.__class__.allowed_tendency_type
                assert isinstance(self._fast_tc, tend_type), (
                    "The input argument ''fast_tendencies'' "
                    "should be an instance of either {}.".format(
                        ", ".join(str(item) for item in tend_type)
                    )
                )

            self._fast_dc = fast_diagnostic_component
            if self._fast_dc is not None:
                diag_type = self.__class__.allowed_diagnostic_type
                assert isinstance(self._fast_dc, diag_type), (
                    "The input argument ''fast_diagnostics'' "
                    "should be an instance of either {}.".format(
                        ", ".join(str(item) for item in diag_type)
                    )
                )

        # initialize properties
        self.input_properties = self._init_input_properties()
        self.tendency_properties = self._init_tendency_properties()
        self.output_properties = self._init_output_properties()

        # instantiate checkers
        self._input_checker = InputChecker(self)
        self._tendency_checker = SubsetTendencyChecker(self)
        self._output_checker = OutputChecker(self)

        # instantiate the dictionary operator
        self._dict_op = DataArrayDictOperator(
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=self._dtype,
            rebuild=rebuild,
        )

        # allocate the output state
        self._out_state = self.allocate_output_state()

        # initialize the dictionary of intermediate tendencies
        self._inter_tendencies = {}

    @property
    def grid(self) -> "Grid":
        """ The underlying :class:`~tasmania.Grid`. """
        return self._grid

    @property
    def horizontal_boundary(self) -> "HorizontalBoundary":
        """
        The :class:`~tasmania.HorizontalBoundary` object handling the lateral
        boundary conditions.
        """
        return self._hb

    def ensure_internal_consistency(self) -> None:
        """ Perform some controls aiming to ensure internal consistency.

        In more detail:

        1. Variables contained in both ``stage_input_properties`` and \
            ``stage_output_properties`` should have compatible properties \
            across the two dictionaries;
        2. Variables contained in both ``substep_input_properties`` and \
            ``substep_output_properties`` should have compatible properties \
            across the two dictionaries;
        3. Variables contained in both ``stage_input_properties`` and the \
            ``input_properties`` dictionary of ``intermediate_tendency_component`` \
            should have compatible properties across the two dictionaries;
        4. Dimensions and units of the variables diagnosed by \
            ``intermediate_tendency_component`` should be compatible with \
            the dimensions and units specified in ``stage_input_properties``;
        5. Any intermediate tendency calculated by ``intermediate_tendency_component`` \
            should be present in the ``stage_tendency_properties`` dictionary, \
            with compatible dimensions and units;
        6. Dimensions and units of the variables diagnosed by \
            ``intermediate_tendency_component`` should be compatible with \
            the dimensions and units specified in the ``input_properties`` \
            dictionary of ``fast_tendency_component``, or the \
            ``substep_input_properties`` dictionary if \
            ``fast_tendency_component`` is not given;
        7. Variables diagnosed by ``fast_tendency_component`` should have dimensions \
            and units compatible with those specified in the \
            ``substep_input_properties`` dictionary;
        8. Variables contained in ``stage_output_properties`` for which \
            ``fast_tendency_component`` prescribes a (fast) tendency should \
            have dimensions and units compatible with those specified \
            in the ``tendency_properties`` dictionary of ``fast_tendency_component``;
        9. Any fast tendency calculated by ``fast_tendency_component`` \
            should be present in the ``substep_tendency_properties`` \
            dictionary, with compatible dimensions and units;
        10. Any variable for which``fast_tendency_component`` \
            prescribes a (fast) tendency should be present both in \
            the ``substep_input_property`` and ``substep_output_property`` \
            dictionaries, with compatible dimensions and units;
        11. Any variable being expected by ``fast_diagnostic_component`` should be \
            present in ``substep_output_properties``, with compatible \
            dimensions and units;
        12. Any variable being expected by ``intermediate_diagnostic_component`` \
            should be present either in ``stage_output_properties`` or \
            ``substep_output_properties``, with compatible dimensions \
            and units.
        13. ``stage_array_call`` should be able to handle any tendency \
            prescribed by ``intermediate_tendency_component``.
        """
        # ============================================================
        # Check #1
        # ============================================================
        check_properties_compatibility(
            self.stage_input_properties,
            self.stage_output_properties,
            properties1_name="_input_properties",
            properties2_name="_output_properties",
        )

        # ============================================================
        # Check #2
        # ============================================================
        check_properties_compatibility(
            self.substep_input_properties,
            self.substep_output_properties,
            properties1_name="_substep_input_properties",
            properties2_name="_substep_output_properties",
        )

        # ============================================================
        # Check #3
        # ============================================================
        if self._inter_tc is not None:
            check_properties_compatibility(
                self._inter_tc.input_properties,
                self.stage_input_properties,
                properties1_name="intermediate_tendencies.input_properties",
                properties2_name="_input_properties",
            )

        # ============================================================
        # Check #4
        # ============================================================
        if self._inter_tc is not None:
            check_properties_compatibility(
                self._inter_tc.diagnostic_properties,
                self.stage_input_properties,
                properties1_name="intermediate_tendencies.diagnostic_properties",
                properties2_name="_input_properties",
            )

        # ============================================================
        # Check #5
        # ============================================================
        if self._inter_tc is not None:
            check_properties_compatibility(
                self._inter_tc.tendency_properties,
                self.stage_tendency_properties,
                properties1_name="intermediate_tendencies.tendency_properties",
                properties2_name="_tendency_properties",
            )

            check_missing_properties(
                self._inter_tc.tendency_properties,
                self.stage_tendency_properties,
                properties1_name="intermediate_tendencies.tendency_properties",
                properties2_name="_tendency_properties",
            )

        # ============================================================
        # Check #6
        # ============================================================
        if self._inter_tc is not None:
            if self._fast_tc is not None:
                check_properties_compatibility(
                    self._inter_tc.diagnostic_properties,
                    self._fast_tc.input_properties,
                    properties1_name="intermediate_tendencies.diagnostic_properties",
                    properties2_name="fast_tendencies.input_properties",
                )
            else:
                check_properties_compatibility(
                    self._inter_tc.diagnostic_properties,
                    self.substep_input_properties,
                    properties1_name="intermediate_tendencies.diagnostics_properties",
                    properties2_name="_substep_input_properties",
                )

        # ============================================================
        # Check #7
        # ============================================================
        if self._fast_tc is not None:
            check_properties_compatibility(
                self._fast_tc.diagnostics_properties,
                self.substep_input_properties,
                properties1_name="fast_tendencies.diagnostic_properties",
                properties2_name="_substep_input_properties",
            )

        # ============================================================
        # Check #8
        # ============================================================
        if self._fast_tc is not None:
            check_properties_compatibility(
                self._fast_tc.tendency_properties,
                self.stage_output_properties,
                to_append=" s",
                properties1_name="fast_tendencies.tendency_properties",
                properties2_name="_output_properties",
            )

        # ============================================================
        # Check #9
        # ============================================================
        if self._fast_tc is not None:
            check_properties_compatibility(
                self._fast_tc.tendency_properties,
                self.substep_tendency_properties,
                to_append=" s",
                properties1_name="fast_tendencies.tendency_properties",
                properties2_name="_substep_tendency_properties",
            )

            check_missing_properties(
                self._fast_tc.tendency_properties,
                self.substep_tendency_properties,
                properties1_name="fast_tendencies.tendency_properties",
                properties2_name="_substep_tendency_properties",
            )

        # ============================================================
        # Check #10
        # ============================================================
        if self._fast_tc is not None:
            check_properties_compatibility(
                self._fast_tc.tendency_properties,
                self.substep_input_properties,
                to_append=" s",
                properties1_name="fast_tendencies.tendency_properties",
                properties2_name="_substep_input_properties",
            )

            check_missing_properties(
                self._fast_tc.tendency_properties,
                self.substep_input_properties,
                properties1_name="fast_tendencies.tendency_properties",
                properties2_name="_substep_input_properties",
            )

            check_properties_compatibility(
                self._fast_tc.tendency_properties,
                self.substep_output_properties,
                to_append=" s",
                properties1_name="fast_tendencies.tendency_properties",
                properties2_name="_substep_input_properties",
            )

            check_missing_properties(
                self._fast_tc.tendency_properties,
                self.substep_output_properties,
                properties1_name="fast_tendencies.tendency_properties",
                properties2_name="_substep_output_properties",
            )

        # ============================================================
        # Check #11
        # ============================================================
        if self._fast_dc is not None:
            check_properties_compatibility(
                self._fast_dc.input_properties,
                self.substep_output_properties,
                properties1_name="fast_diagnostics.input_properties",
                properties2_name="_substep_output_properties",
            )

            check_missing_properties(
                self._fast_dc.input_properties,
                self.substep_output_properties,
                properties1_name="fast_diagnostics.input_properties",
                properties2_name="_substep_output_properties",
            )

        # ============================================================
        # Check #12
        # ============================================================
        if self._inter_dc is not None:
            fused_output_properties = {}
            fused_output_properties.update(self.stage_output_properties)
            fused_output_properties.update(self.substep_output_properties)

            check_properties_compatibility(
                self._inter_dc.input_properties,
                fused_output_properties,
                properties1_name="intermediate_diagnostics.input_properties",
                properties2_name="fused_output_properties",
            )

            check_missing_properties(
                self._inter_dc.input_properties,
                fused_output_properties,
                properties1_name="intermediate_diagnostics.input_properties",
                properties2_name="fused_output_properties",
            )

        # ============================================================
        # Check #13
        # ============================================================
        if self._inter_dc is not None:
            src = getattr(self._inter_dc, "tendency_properties", {})
            trg = self.stage_tendency_properties

            check_properties_compatibility(
                src,
                trg,
                properties1_name="intermediate_diagnostics.tendency_properties",
                properties2_name="_tendency_properties",
            )

            check_missing_properties(
                src,
                trg,
                properties1_name="intermediate_diagnostics.tendency_properties",
                properties2_name="_tendency_properties",
            )

    def ensure_input_output_consistency(self) -> None:
        """ Perform some controls aiming to ensure input-output consistency.

        In more detail:

        1. Variables contained in both ``input_properties`` and \
            ``output_properties`` should have compatible properties \
            across the two dictionaries;
        2. In case of a multi-stage dynamical core, any variable \
            present in ``output_properties`` should be also contained \
            in ``input_properties``.
        """
        # ============================================================
        # Safety-guard preamble
        # ============================================================
        assert hasattr(
            self, "input_properties"
        ), "Hint: did you call _init_input_properties?"
        assert hasattr(
            self, "output_properties"
        ), "Hint: did you call _init_output_properties?"

        # ============================================================
        # Check #1
        # ============================================================
        check_properties_compatibility(
            self.input_properties,
            self.output_properties,
            properties1_name="input_properties",
            properties2_name="output_properties",
        )

        # ============================================================
        # Check #2
        # ============================================================
        if self.stages > 1:
            check_missing_properties(
                self.output_properties,
                self.input_properties,
                properties1_name="output_properties",
                properties2_name="input_properties",
            )

    def _init_input_properties(self) -> taz_types.properties_dict_t:
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting variables which
            should be included in the input state, and whose values
            are fundamental properties (dims, units) of those variables.
            This dictionary results from fusing the requirements
            specified by the user via
            :meth:`~tasmania.DynamicalCore.stage_input_properties` and
            :meth:`~tasmania.DynamicalCore.substep_input_properties`
            with the ``input_properties`` dictionary of
            ``intermediate_tendency_component`` and
            ``fast_tendency_component``.
        """
        return_dict = {}

        if self._inter_tc is None:
            return_dict.update(self.stage_input_properties)
        else:
            return_dict.update(self._inter_tc.input_properties)
            inter_params_diag_properties = self._inter_tc.diagnostic_properties
            stage_input_properties = self.stage_input_properties

            # Add to the requirements the variables to feed the stage with
            # and which are not output by the intermediate parameterizations
            unshared_vars = tuple(
                name
                for name in stage_input_properties
                if not (
                    name in inter_params_diag_properties or name in return_dict
                )
            )
            for name in unshared_vars:
                return_dict[name] = {}
                return_dict[name].update(stage_input_properties[name])

        if self._substeps >= 0:
            fast_params_input_properties = (
                {} if self._fast_tc is None else self._fast_tc.input_properties
            )
            fast_params_diag_properties = (
                {}
                if self._fast_tc is None
                else self._fast_tc.diagnostic_properties
            )

            # Add to the requirements the variables to feed the fast
            # parameterizations with
            unshared_vars = tuple(
                name
                for name in fast_params_input_properties
                if name not in return_dict
            )
            for name in unshared_vars:
                return_dict[name] = {}
                return_dict[name].update(fast_params_input_properties[name])

            # Add to the requirements the variables to feed the substep with
            # and which are not output by the either the intermediate parameterizations
            # or the fast parameterizations
            unshared_vars = tuple(
                name
                for name in self.substep_input_properties
                if not (
                    name in fast_params_diag_properties or name in return_dict
                )
            )
            for name in unshared_vars:
                return_dict[name] = {}
                return_dict[name].update(self.substep_input_properties[name])

        return return_dict

    @property
    @abc.abstractmethod
    def stage_input_properties(self) -> taz_types.properties_dict_t:
        """
        Dictionary whose keys are strings denoting variables which
        should be included in any state passed to the ``stage_array_call``, and
        whose values are fundamental properties (dims, units, alias)
        of those variables.
        """
        pass

    @property
    @abc.abstractmethod
    def substep_input_properties(self) -> taz_types.properties_dict_t:
        """
        Dictionary whose keys are strings denoting variables which
        should be included in any state passed to the ``substep_array_call``
        carrying out the substepping routine, and whose values are
        fundamental properties (dims, units, alias) of those variables.
        """
        pass

    def _init_tendency_properties(self) -> taz_types.properties_dict_t:
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting (slow) tendencies which
            may (or may not) be passed to the call operator, and whose
            values are fundamental properties (dims, units) of those
            tendencies. This dictionary results from fusing the requirements
            specified by the user via
            :meth:`tasmania.DynamicalCore.stage_tendency_properties`
            with the ``tendency_properties`` dictionary of
            ``intermediate_tendency_component``.
        """
        return_dict = {}

        if self._inter_tc is None:
            return_dict.update(self.stage_tendency_properties)
        else:
            return_dict.update(self._inter_tc.tendency_properties)

            # Add to the requirements on the input slow tendencies those
            # tendencies to feed the dycore with and which are not provided
            # by the intermediate parameterizations
            unshared_vars = tuple(
                name
                for name in self.stage_tendency_properties
                if name not in return_dict
            )
            for name in unshared_vars:
                return_dict[name] = {}
                return_dict[name].update(self.stage_tendency_properties[name])

        return return_dict

    @property
    @abc.abstractmethod
    def stage_tendency_properties(self) -> taz_types.properties_dict_t:
        """
        Dictionary whose keys are strings denoting (slow and intermediate)
        tendencies which may (or may not) be passed to ``stage_array_call``,
        and whose values are fundamental properties (dims, units, alias)
        of those tendencies.
        """
        pass

    @property
    @abc.abstractmethod
    def substep_tendency_properties(self) -> taz_types.properties_dict_t:
        """
        Dictionary whose keys are strings denoting (slow, intermediate and fast)
        tendencies which may (or may not) be passed to ``substep_array_call``,
        and whose values are fundamental properties (dims, units, alias)
        of those tendencies.
        """
        pass

    def _init_output_properties(self) -> taz_types.properties_dict_t:
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting variables which are
            included in the output state, and whose values are fundamental
            properties (dims, units) of those variables. This dictionary
            results from fusing the requirements specified by the user via
            :meth:`~tasmania.DynamicalCore.stage_output_properties` and
            :meth:`~tasmania.DynamicalCore.substep_output_properties`
            with the ``diagnostic_properties`` dictionary of
            ``intermediate_diagnostic_component`` and
            ``fast_diagnostic_component``.
        """
        return_dict = {}

        if self._substeps == 0:
            # Add to the return dictionary the variables included in
            # the state output by a stage
            return_dict.update(self.stage_output_properties)
        else:
            # Add to the return dictionary the variables included in
            # the state output by a substep
            return_dict.update(self.substep_output_properties)

            if self._fast_dc is not None:
                # Add the fast diagnostics to the return dictionary
                for (
                    name,
                    properties,
                ) in self._fast_dc.diagnostic_properties.items():
                    return_dict[name] = {}
                    return_dict[name].update(properties)

            # Add to the return dictionary the non-substepped variables
            return_dict.update(self.stage_output_properties)

        if self._inter_dc is not None:
            # Add the retrieved diagnostics to the return dictionary
            for (
                name,
                properties,
            ) in self._inter_dc.diagnostic_properties.items():
                return_dict[name] = {}
                return_dict[name].update(properties)

        return return_dict

    @property
    @abc.abstractmethod
    def stage_output_properties(self) -> taz_types.properties_dict_t:
        """
        Dictionary whose keys are strings denoting variables which are
        included in the output state returned by ``stage_array_call``,
        and whose values are fundamental properties (dims, units)
        of those variables.
        """
        pass

    @property
    @abc.abstractmethod
    def substep_output_properties(self) -> taz_types.properties_dict_t:
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting variables which are
            included in the output state returned by any substep, and whose
            values are fundamental properties (dims, units) of those variables.
        """
        pass

    @property
    @abc.abstractmethod
    def stages(self) -> int:
        """ Number of stages carried out by the dynamical core. """
        pass

    @property
    @abc.abstractmethod
    def substep_fractions(self) -> Union[float, Sequence[float]]:
        """
        For each stage, fraction of the total number of substeps
        (specified at instantiation) to carry out.
        """
        pass

    @abc.abstractmethod
    def allocate_output_state(self) -> taz_types.dataarray_dict_t:
        """ Allocate memory for the return state. """
        pass

    def __call__(
        self,
        state: taz_types.dataarray_dict_t,
        tendencies: taz_types.dataarray_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.dataarray_dict_t:
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
        self._input_checker.check_inputs(state)
        self._tendency_checker.check_tendencies(tendencies)

        out_state = self._out_state

        inter_tends = self.call(
            0,
            timestep,
            state,
            state,
            tendencies,
            self._inter_tendencies,
            out_state,
        )
        for stage in range(1, self.stages):
            inter_tends = self.call(
                stage,
                timestep,
                state,
                out_state,
                tendencies,
                inter_tends,
                out_state,
            )

        return_state = {"time": out_state["time"]}
        for name in self.output_properties:
            return_state[name] = out_state[name]

        self._inter_tendencies = inter_tends

        return return_state

    def call(
        self,
        stage: int,
        timestep: taz_types.timedelta_t,
        state: taz_types.dataarray_dict_t,
        tmp_state: taz_types.dataarray_dict_t,
        slow_tendencies: taz_types.dataarray_dict_t,
        inter_tendencies: taz_types.dataarray_dict_t,
        out_state: taz_types.mutable_dataarray_dict_t,
    ) -> taz_types.dataarray_dict_t:
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
            The latest provisional state.
            It coincides with ``state`` when ``stage = 0``.
        slow_tendencies : dict[str, sympl.DataArray]
            The *slow* physics tendencies for the prognostic model variables.
        inter_tendencies : dict[str, sympl.DataArray]
            The *intermediate* physics tendencies coming from the previous stage.
        out_state : dict[str, sympl.DataArray]
            The :class:`sympl.DataArray`\s into which the next provisional
            state will be written.
        """
        # ============================================================
        # Calculating the intermediate tendencies
        # ============================================================
        # add the slow and intermediate tendencies up
        Timer.start(label="add_slow_inter_tends")
        self._dict_op.iadd(
            inter_tendencies,
            slow_tendencies,
            field_properties=self.tendency_properties,
            unshared_variables_in_output=True,
        )
        Timer.stop()

        if self._inter_tc is None and stage == 0:
            # collect the slow tendencies, and possibly the intermediate
            # tendencies from the previous stage
            tends = {}
            tends.update(inter_tendencies)
        elif self._inter_tc is not None:
            Timer.start(label="get_inter_tends")
            # calculate the intermediate tendencies
            try:
                tends, diags = self._inter_tc(tmp_state)
            except TypeError:
                tends, diags = self._inter_tc(tmp_state, timestep)

            # sum up all the slow and intermediate tendencies
            self._dict_op.iadd(
                tends,
                inter_tendencies,
                field_properties=self.tendency_properties,
                unshared_variables_in_output=True,
            )

            # update the state with the just computed diagnostics
            tmp_state.update(diags)
            Timer.stop()
        else:
            tends = {}

        # ============================================================
        # Stage: pre-processing
        # ============================================================
        # Extract raw storages from state
        tmp_state_properties = {
            name: self.stage_input_properties[name]
            for name in self.stage_input_properties
        }
        Timer.start(label="get_raw_tmp_state")
        raw_tmp_state = get_array_dict(tmp_state, tmp_state_properties)
        Timer.stop()

        # Extract raw storages from tendencies
        tendency_properties = {
            name: self.stage_tendency_properties[name]
            for name in self.stage_tendency_properties
        }
        Timer.start(label="get_raw_tends")
        raw_tends = get_array_dict(tends, tendency_properties)
        Timer.stop()

        # ============================================================
        # Stage: computing
        # ============================================================
        # Carry out the stage
        Timer.start(label="stage")
        raw_stage_state = self.stage_array_call(
            stage, raw_tmp_state, raw_tends, timestep
        )
        Timer.stop()

        if self._substeps == 0 or len(self.substep_output_properties) == 0:
            # ============================================================
            # Stage: post-processing, substepping disabled
            # ============================================================
            # Create dataarrays out of the numpy arrays contained in the stepped state
            stage_state_properties = {
                name: dict(
                    **self.stage_output_properties[name], set_coordinates=False
                )
                for name in self.stage_output_properties
            }
            Timer.start(label="get_stage_state")
            stage_state = get_dataarray_dict(
                raw_stage_state, self._grid, stage_state_properties
            )
            Timer.stop()

            # Update the latest state
            Timer.start(label="update_out_state")
            self._dict_op.copy(out_state, stage_state)
            # out_state.update(stage_state)
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
        # Retrieving the intermediate diagnostics
        # ============================================================
        if self._inter_dc is not None:
            Timer.start(label="compute_inter_diags")
            if isinstance(
                self._inter_dc, self.__class__.allowed_diagnostic_type
            ):
                inter_tends = {}
                try:
                    inter_diags = self._inter_dc(out_state)
                except TypeError:
                    inter_diags = self._inter_dc(out_state, timestep)
            else:  # tendency component
                try:
                    inter_tends, inter_diags = self._inter_dc(out_state)
                except TypeError:
                    inter_tends, inter_diags = self._inter_dc(
                        out_state, timestep
                    )
            Timer.stop()

            diagnostic_fields = {}
            for name in inter_diags:
                if name != "time" and name not in self.stage_output_properties:
                    diagnostic_fields[name] = inter_diags[name]

            Timer.start(label="fill_inter_diags")
            self._dict_op.copy(out_state, inter_diags)
            out_state.update(diagnostic_fields)
            Timer.stop()
        else:
            inter_tends = {}

        # Ensure the time specified in the output state is correct
        if stage == self.stages - 1:
            out_state["time"] = state["time"] + timestep

        # ============================================================
        # Final checks
        # ============================================================
        self._output_checker.check_outputs(
            {
                name: out_state[name]
                for name in out_state
                if (name != "time" and name in self.output_properties)
            }
        )

        return inter_tends

    @abc.abstractmethod
    def stage_array_call(
        self,
        stage: int,
        raw_state: taz_types.array_dict_t,
        raw_tendencies: taz_types.array_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.array_dict_t:
        """Integrate the state over a stage.

        Parameters
        ----------
        stage : int
            The stage identifier.
        raw_state : dict[str, array_like]
            The latest provisional state.
        raw_tendencies : dict[str, array_like]
            The tendencies for the model prognostic variables.
        timestep : datetime.timedelta
            The step size.

        Return
        ------
        dict[str, array_like]
            The next provisional state.
        """
        pass

    @abc.abstractmethod
    def substep_array_call(
        self,
        stage: int,
        substep: int,
        raw_state: taz_types.array_dict_t,
        raw_stage_state: taz_types.array_dict_t,
        raw_tmp_state: taz_types.array_dict_t,
        raw_tendencies: taz_types.array_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.array_dict_t:
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
        pass

    def update_topography(self, time: taz_types.datetime_t) -> None:
        """Update the underlying (time-dependent) topography.

        Parameters
        ----------
        time : datetime.timedelta
            The elapsed simulation time.
        """
        self._grid.update_topography(time)
