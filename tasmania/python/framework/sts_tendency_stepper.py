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
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
)
from sympl._core.base_components import (
    InputChecker,
    DiagnosticChecker,
    OutputChecker,
)
from sympl._core.units import clean_units
from typing import Optional, Tuple

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.fakes import FakeComponent
from tasmania.python.framework.register import factorize
from tasmania.python.utils import taz_types
from tasmania.python.utils.dict_utils import DataArrayDictOperator
from tasmania.python.utils.framework_utils import check_property_compatibility
from tasmania.python.utils.storage_utils import deepcopy_dataarray
from tasmania.python.utils.utils import assert_sequence


class STSTendencyStepper(abc.ABC):
    """
    Callable abstract base class which steps a model state based on the
    tendencies calculated by a set of wrapped prognostic components,
    pursuing the sequential-tendency splitting (STS) approach.
    """

    registry = {}

    allowed_component_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        ConcurrentCoupling,
    )

    def __init__(
        self,
        *args: taz_types.tendency_component_t,
        execution_policy: str = "serial",
        enforce_horizontal_boundary: bool = False,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        build_info: Optional[taz_types.options_dict_t] = None,
        rebuild: bool = False
    ) -> None:
        """
        Parameters
        ----------
        args :
            Instances of

                * :class:`sympl.TendencyComponent`,
                * :class:`sympl.TendencyComponentComposite`,
                * :class:`sympl.ImplicitTendencyComponent`,
                * :class:`sympl.ImplicitTendencyComponentComposite`, or
                * :class:`tasmania.ConcurrentCoupling`

            providing tendencies for the prognostic variables.
        execution_policy : `str`, optional
            String specifying the runtime mode in which parameterizations
            should be invoked. See :class:`tasmania.ConcurrentCoupling`.
        enforce_horizontal_boundary : `bool`, optional
            ``True`` if the class should enforce the lateral boundary
            conditions after each stage of the time integrator,
            ``False`` otherwise. Defaults to ``False``.
            This argument is considered only if at least one of the wrapped
            objects is an instance of

                * :class:`tasmania.TendencyComponent`, or
                * :class:`tasmania.ImplicitTendencyComponent`.

        backend : `str`, optional
            The backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        """
        assert_sequence(args, reftype=self.__class__.allowed_component_type)

        self._prognostic_list = args
        self._prognostic = (
            args[0]
            if (len(args) == 1 and isinstance(args[0], ConcurrentCoupling))
            else ConcurrentCoupling(*args, execution_policy=execution_policy)
        )

        self.input_properties = self._get_input_properties()
        self.provisional_input_properties = (
            self._get_provisional_input_properties()
        )
        self.diagnostic_properties = self._get_diagnostic_properties()
        self.output_properties = self._get_output_properties()

        self._input_checker = InputChecker(self)
        self._provisional_input_checker = InputChecker(
            FakeComponent(self, "provisional_input_properties")
        )
        self._diagnostic_checker = DiagnosticChecker(self)
        self._output_checker = OutputChecker(self)

        enforce_hb = enforce_horizontal_boundary
        if enforce_hb:
            found = False
            for prognostic in args:
                if not found:

                    try:  # composite component
                        components = prognostic.component_list
                    except AttributeError:  # base component
                        components = (prognostic,)

                    for component in components:
                        try:  # tasmania's component
                            self._hb = component.horizontal_boundary
                            self._grid = component.grid
                            self._enforce_hb = True
                            found = True
                            break
                        except AttributeError:  # sympl's component
                            pass

            if not found:
                self._enforce_hb = False
        else:
            self._enforce_hb = False

        self._dict_op = DataArrayDictOperator(
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            rebuild=rebuild,
        )

        self._out_state = None

    @property
    def prognostic(self) -> ConcurrentCoupling:
        """
        Return
        ------
        tasmania.ConcurrentCoupling :
            The object calculating the tendencies.
        """
        return self._prognostic

    def _get_input_properties(self) -> taz_types.properties_dict_t:
        """
        Return
        ------
        dict[str, str] :
            Dictionary whose keys are strings denoting model variables
            which should be present in the input state dictionary, and
            whose values are dictionaries specifying fundamental properties
            (dims, units) of those variables.
        """
        return_dict = {}
        return_dict.update(self._prognostic.input_properties)

        tendency_properties = self._prognostic.tendency_properties
        for name in tendency_properties:
            mod_tendency_property = tendency_properties[name].copy()
            mod_tendency_property["units"] = clean_units(
                mod_tendency_property["units"] + " s"
            )

            if name in return_dict:
                check_property_compatibility(
                    property_name=name,
                    property1=return_dict[name],
                    origin1_name="self._prognostic.input_properties",
                    property2=mod_tendency_property,
                    origin2_name="self._prognostic.tendency_properties",
                )
            else:
                return_dict[name] = {}
                return_dict[name].update(mod_tendency_property)

        return return_dict

    def _get_provisional_input_properties(self) -> taz_types.properties_dict_t:
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting model variables
            which should be present in the provisional input state dictionary,
            and whose values are dictionaries specifying fundamental properties
            (dims, units) of those variables.
        """
        return_dict = {}

        for key, val in self._prognostic.tendency_properties.items():
            return_dict[key] = val.copy()
            if "units" in return_dict[key]:
                return_dict[key]["units"] = clean_units(
                    return_dict[key]["units"] + " s"
                )

        return return_dict

    def _get_diagnostic_properties(self) -> taz_types.properties_dict_t:
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting diagnostics
            which are retrieved from the input state dictionary, and
            whose values are dictionaries specifying fundamental
            properties (dims, units) of those diagnostics.
        """
        return self._prognostic.diagnostic_properties

    def _get_output_properties(self) -> taz_types.properties_dict_t:
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting model variables
            present in the output state dictionary, and whose values are
            dictionaries specifying fundamental properties (dims, units)
            of those variables.
        """
        return self._get_provisional_input_properties()

    def __call__(
        self,
        state: taz_types.dataarray_dict_t,
        prv_state: taz_types.dataarray_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> Tuple[taz_types.dataarray_dict_t, taz_types.dataarray_dict_t]:
        """
        Step the model state.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The current state dictionary.
        prv_state : dict[str, sympl.DataArray]
            The provisional state dictionary.
        timestep : datetime.timedelta
            The time step.

        Return
        ------
        diagnostics : dict[str, sympl.DataArray]
            The diagnostics retrieved from the input state.
        out_state : dict[str, sympl.DataArray]
            The output (stepped) state.
        """
        self._input_checker.check_inputs(state)
        self._provisional_input_checker.check_inputs(prv_state)

        diagnostics, out_state = self._call(state, prv_state, timestep)

        self._diagnostic_checker.check_diagnostics(
            {key: val for key, val in diagnostics.items() if key != "time"}
        )
        diagnostics["time"] = state["time"]

        self._output_checker.check_outputs(
            {key: val for key, val in out_state.items() if key != "time"}
        )
        out_state["time"] = state["time"] + timestep

        return diagnostics, out_state

    @staticmethod
    def factory(
        scheme: str,
        *args: taz_types.tendency_component_t,
        execution_policy: str = "serial",
        enforce_horizontal_boundary: bool = False,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        build_info: Optional[taz_types.options_dict_t] = None,
        rebuild: bool = False,
        **kwargs
    ) -> "STSTendencyStepper":
        """Get an instance of the desired derived class.

        Parameters
        ----------
        scheme : str
            The time integration scheme to implement.
        args :
            Instances of

                * :class:`sympl.TendencyComponent`,
                * :class:`sympl.TendencyComponentComposite`,
                * :class:`sympl.ImplicitTendencyComponent`,
                * :class:`sympl.ImplicitTendencyComponentComposite`, or
                * :class:`tasmania.ConcurrentCoupling`

            providing tendencies for the prognostic variables.
        execution_policy : `str`, optional
            String specifying the runtime mode in which parameterizations
            should be invoked. See :class:`tasmania.ConcurrentCoupling`.
        enforce_horizontal_boundary : `bool`, optional
            ``True`` if the class should enforce the lateral boundary
            conditions after each stage of the time integrator,
            ``False`` otherwise. Defaults to ``False``.
            This argument is considered only if at least one of the wrapped
            objects is an instance of

                * :class:`tasmania.TendencyComponent`, or
                * :class:`tasmania.ImplicitTendencyComponent`.

        backend : `str`, optional
            The backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        **kwargs :
            Scheme-specific arguments.

        Return
        ------
        obj :
            Instance of the desired derived class.
        """
        child_kwargs = {
            "execution_policy": execution_policy,
            "enforce_horizontal_boundary": enforce_horizontal_boundary,
            "backend": backend,
            "backend_opts": backend_opts,
            "dtype": dtype,
            "build_info": build_info,
            "rebuild": rebuild,
        }
        child_kwargs.update(kwargs)
        return factorize(scheme, STSTendencyStepper, args, child_kwargs)

    @abc.abstractmethod
    def _call(
        self,
        state: taz_types.dataarray_dict_t,
        prv_state: taz_types.dataarray_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> Tuple[taz_types.dataarray_dict_t, taz_types.dataarray_dict_t]:
        """
        Step the model state. As this method is marked as abstract,
        its implementation is delegated to the derived classes.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The current state dictionary.
        prv_state : dict[str, sympl.DataArray]
            The provisional state dictionary.
        timestep : datetime.timedelta
            The time step.

        Return
        ------
        diagnostics : dict[str, sympl.DataArray]
            The diagnostics retrieved from the input state.
        out_state : dict[str, sympl.DataArray]
            The output (stepped) state.
        """
        pass

    def _allocate_output_state(
        self, state: taz_types.dataarray_dict_t
    ) -> taz_types.dataarray_dict_t:
        out_state = self._out_state or {}

        if not out_state:
            for name in self.output_properties:
                units = self.output_properties[name]["units"]
                out_state[name] = deepcopy_dataarray(
                    state[name].to_units(units)
                )
                out_state[name].data[...] = 0.0

        return out_state
