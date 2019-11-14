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
from copy import deepcopy
import numpy as np
from sympl import (
    DataArray,
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
)
from sympl._core.base_components import InputChecker, DiagnosticChecker, OutputChecker
from sympl._core.units import clean_units

from gt4py import gtscript

# from gt4py.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.utils.dict_utils import add, add_inplace, multiply
from tasmania.python.utils.framework_utils import check_property_compatibility
from tasmania.python.utils.gtscript_utils import set_annotations
from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.utils import assert_sequence

try:
    from tasmania.conf import datatype
except ImportError:
    from numpy import float32 as datatype


def get_increment(state, timestep, prognostic):
    # calculate tendencies and retrieve diagnostics
    tendencies, diagnostics = prognostic(state, timestep)

    # "multiply" the tendencies by the time step
    for name in tendencies:
        if name != "time":
            tendencies[name].attrs["units"] += " s"

    return tendencies, diagnostics


def restore_tendency_units(tendencies):
    for name in tendencies:
        if name != "time":
            tendencies[name].attrs["units"] = clean_units(
                tendencies[name].attrs["units"] + " s^-1"
            )


def forward_euler(
    in_field: gtscript.Field[np.float64],
    in_tnd: gtscript.Field[np.float64],
    out_field: gtscript.Field[np.float64],
    *,
    dt: float
):
    with computation(PARALLEL), interval(...):
        out_field = in_field + dt * in_tnd


def tendencystepper_factory(scheme):
    if scheme == "forward_euler":
        return ForwardEuler
    elif scheme == "gt_forward_euler":
        return GTForwardEuler
    elif scheme == "rk2":
        return RungeKutta2
    elif scheme == "gt_rk2":
        return GTRungeKutta2
    elif scheme == "rk3ws":
        return RungeKutta3WS
    elif scheme == "gt_rk3ws":
        return GTRungeKutta3WS
    elif scheme == "rk3":
        return RungeKutta3
    else:
        raise ValueError(
            "Unsupported time integration scheme "
            "{}"
            ". "
            "Available integrators: forward_euler, rk2, rk3ws, rk3.".format(scheme)
        )


class TendencyStepper(abc.ABC):
    """
    Callable abstract base class which steps a model state based on the
    tendencies calculated by a set of wrapped prognostic components.
    """

    allowed_component_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        ConcurrentCoupling,
    )

    def __init__(
        self, *args, execution_policy="serial", enforce_horizontal_boundary=False
    ):
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
            `True` if the class should enforce the lateral boundary
            conditions after each stage of the time integrator,
            `False` otherwise. Defaults to `False`.
            This argument is considered only if at least one of the wrapped
            objects is an instance of

                * :class:`tasmania.TendencyComponent`, or
                * :class:`tasmania.ImplicitTendencyComponent`.

        """
        assert_sequence(args, reftype=self.__class__.allowed_component_type)

        self._prognostic_list = args
        self._prognostic = (
            args[0]
            if (len(args) == 1 and isinstance(args[0], ConcurrentCoupling))
            else ConcurrentCoupling(*args, execution_policy=execution_policy)
        )

        self._input_checker = InputChecker(self)
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

        self._out_state = None

    @property
    def prognostic(self):
        """
        Return
        ------
        tasmania.ConcurrentCoupling :
            The object calculating the tendencies.
        """
        return self._prognostic

    @property
    def input_properties(self):
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting model variables
            which should be present in the input state dictionary, and
            whose values are dictionaries specifying fundamental properties
            (dims, units) of those variables.
        """
        return_dict = {}
        return_dict.update(self._prognostic.input_properties)

        tendency_properties = self._prognostic.tendency_properties
        for name in tendency_properties:
            mod_tendency_property = deepcopy(tendency_properties[name])
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

    @property
    def diagnostic_properties(self):
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

    @property
    def output_properties(self):
        """
        Return
        ------
        dict[str, dict] :
            Dictionary whose keys are strings denoting model variables
            present in the output state dictionary, and whose values are
            dictionaries specifying fundamental properties (dims, units)
            of those variables.
        """
        return_dict = {}

        for key, val in self._prognostic.tendency_properties.items():
            return_dict[key] = deepcopy(val)
            if "units" in return_dict[key]:
                return_dict[key]["units"] = clean_units(return_dict[key]["units"] + " s")

        return return_dict

    def __call__(self, state, timestep):
        """
        Step the model state.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting the input model
            variables, and whose values are :class:`sympl.DataArray`\s
            storing values for those variables.
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

        diagnostics, out_state = self._call(state, timestep)

        self._diagnostic_checker.check_diagnostics(
            {key: val for key, val in diagnostics.items() if key != "time"}
        )
        diagnostics["time"] = state["time"]

        self._output_checker.check_outputs(
            {key: val for key, val in out_state.items() if key != "time"}
        )
        out_state["time"] = state["time"] + timestep

        return diagnostics, out_state

    @abc.abstractmethod
    def _call(self, state, timestep):
        """
        Step the model state. As this method is marked as abstract,
        its implementation is delegated to the derived classes.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            Dictionary whose keys are strings denoting the input model
            variables, and whose values are :class:`sympl.DataArray`\s
            storing values for those variables.
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

    def _allocate_output_state(self, state):
        backend = getattr(self, "_backend", None)
        default_origin = getattr(self, "_default_origin", None)
        managed_memory = getattr(self, "_managed_memory", False)

        out_state = self._out_state or {}

        if not out_state:
            for name in self.output_properties:
                storage_shape = state[name].shape
                dtype = state[name].dtype
                raw_buffer = (
                    zeros(
                        storage_shape,
                        backend,
                        dtype,
                        default_origin=default_origin,
                        managed_memory=managed_memory,
                    )
                    if backend
                    else np.zeros(storage_shape, dtype=dtype)
                )

                dims = state[name].dims
                coords = state[name].coords
                attrs = state[name].attrs.copy()
                attrs["units"] = self.output_properties[name]["units"]
                out_state[name] = DataArray(
                    raw_buffer, dims=dims, coords=coords, attrs=attrs
                )

        return out_state


class ForwardEuler(TendencyStepper):
    """ The forward Euler scheme. """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        default_origin=None,
        managed_memory=False,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary
        )
        self._backend = backend
        self._default_origin = default_origin
        self._managed_memory = managed_memory

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # calculate the tendencies and the diagnostics
        tendencies, diagnostics = get_increment(state, timestep, self.prognostic)

        # step the solution
        multiply(timestep.total_seconds(), tendencies, out_state)
        add_inplace(out_state, state, units=out_units, unshared_variables_in_output=False)

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(tendencies)

        return diagnostics, out_state


class GTForwardEuler(TendencyStepper):
    """ Gridtools-powered implementation of the forward Euler scheme. """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=np.float32,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        managed_memory=False,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary
        )

        self._backend = backend
        self._exec_info = exec_info
        self._default_origin = default_origin
        self._managed_memory = managed_memory

        set_annotations(forward_euler, dtype)

        self._stencil = gtscript.stencil(
            definition=forward_euler,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            **(backend_opts or {})
        )

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # calculate the tendencies and the diagnostics
        tendencies, diagnostics = get_increment(state, timestep, self.prognostic)

        # extract the raw fields in the correct units
        names = tuple(name for name in out_units)
        raw_state = {name: state[name].to_units(out_units[name]).values for name in names}
        raw_tendencies = {name: tendencies[name].values for name in names}
        raw_out_state = {name: out_state[name].values for name in names}

        # step the solution
        storage_shape = raw_state[names[0]].shape
        origin = (self._hb.nb, self._hb.nb, 0) if self._enforce_hb else (0, 0, 0)
        iteration_domain = tuple(storage_shape[i] - 2 * origin[i] for i in range(3))
        for name in raw_out_state:
            self._stencil(
                in_field=raw_state[name],
                in_tnd=raw_tendencies[name],
                out_field=raw_out_state[name],
                dt=timestep.total_seconds(),
                origin={"_all_": origin},
                domain=iteration_domain,
                exec_info=self._exec_info,
            )

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(tendencies)

        return diagnostics, out_state


class RungeKutta2(TendencyStepper):
    """
    The two-stages, second-order Runge-Kutta scheme.

    References
    ----------
    Gear, C. W. (1971). *Numerical initial value problems in \
        ordinary differential equations.* Prentice Hall PTR.
    """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        default_origin=None,
        managed_memory=False,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary
        )
        self._backend = backend
        self._default_origin = default_origin
        self._managed_memory = managed_memory

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)
        multiply(0.5 * timestep.total_seconds(), k0, out=out_state)
        add_inplace(out_state, state, units=out_units, unshared_variables_in_output=True)
        out_state["time"] = state["time"] + 0.5 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k0)

        # second stage
        k1, _ = get_increment(out_state, timestep, self.prognostic)

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        # step the solution
        multiply(timestep.total_seconds(), k1, out=out_state)
        add_inplace(out_state, state, units=out_units, unshared_variables_in_output=False)
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k1)

        return diagnostics, out_state


class GTRungeKutta2(TendencyStepper):
    """
    GridTools-powered implementation of the two-stages, second-order Runge-Kutta scheme.

    References
    ----------
    Gear, C. W. (1971). *Numerical initial value problems in \
        ordinary differential equations.* Prentice Hall PTR.
    """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=np.float32,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        managed_memory=False,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary
        )

        self._backend = backend
        self._exec_info = exec_info
        self._default_origin = default_origin
        self._managed_memory = managed_memory

        set_annotations(forward_euler, dtype)

        self._stencil = gtscript.stencil(
            definition=forward_euler,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            **(backend_opts or {})
        )

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # calculate the first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)

        # extract the raw arrays in the correct units
        names = tuple(name for name in out_units)
        raw_state = {name: state[name].to_units(out_units[name]).values for name in names}
        raw_k0 = {name: k0[name].values for name in names}
        raw_out_state = {name: out_state[name].values for name in names}

        # update the solution
        storage_shape = raw_state[names[0]].shape
        origin = (self._hb.nb, self._hb.nb, 0) if self._enforce_hb else (0, 0, 0)
        iteration_domain = tuple(storage_shape[i] - 2 * origin[i] for i in range(3))
        for name in raw_out_state:
            self._stencil(
                in_field=raw_state[name],
                in_tnd=raw_k0[name],
                out_field=raw_out_state[name],
                dt=0.5 * timestep.total_seconds(),
                origin={"_all_": origin},
                domain=iteration_domain,
                exec_info=self._exec_info,
            )
        out_state["time"] = state["time"] + 0.5 * timestep

        # populate out_state with all other variables from state
        for name in state:
            if name != "time" and name not in out_state:
                out_state[name] = state[name]

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k0)

        # second stage
        k1, _ = get_increment(out_state, timestep, self.prognostic)

        # extract the raw arrays in the correct units
        raw_k1 = {name: k1[name].values for name in names}

        # update the solution
        for name in raw_out_state:
            self._stencil(
                in_field=raw_state[name],
                in_tnd=raw_k1[name],
                out_field=raw_out_state[name],
                dt=timestep.total_seconds(),
                origin={"_all_": origin},
                domain=iteration_domain,
                exec_info=self._exec_info,
            )
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k1)

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        return diagnostics, out_state


class RungeKutta3WS(TendencyStepper):
    """
    The Wicker-Skamarock Runge-Kutta scheme.

    References
    ----------
    Doms, G., and M. Baldauf. (2015). *A description of the nonhydrostatic \
        regional COSMO-model. Part I: Dynamics and numerics.* \
        Deutscher Wetterdienst, Germany.
    """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        default_origin=None,
        managed_memory=False,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary
        )
        self._backend = backend
        self._default_origin = default_origin
        self._managed_memory = managed_memory

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)
        multiply(1.0 / 3.0 * timestep.total_seconds(), k0, out=out_state)
        add_inplace(out_state, state, units=out_units, unshared_variables_in_output=True)
        out_state["time"] = state["time"] + 1.0 / 3.0 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k0)

        # second stage
        k1, _ = get_increment(out_state, timestep, self.prognostic)

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        # step the solution
        multiply(0.5 * timestep.total_seconds(), k1, out=out_state)
        add_inplace(out_state, state, units=out_units, unshared_variables_in_output=True)
        out_state["time"] = state["time"] + 0.5 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k1)

        # second stage
        k2, _ = get_increment(out_state, timestep, self.prognostic)

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        # step the solution
        multiply(timestep.total_seconds(), k2, out=out_state)
        add_inplace(out_state, state, units=out_units, unshared_variables_in_output=False)
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k2)

        return diagnostics, out_state


class GTRungeKutta3WS(TendencyStepper):
    """
    GridTools-powered implementation of the Wicker-Skamarock Runge-Kutta scheme.

    References
    ----------
    Doms, G., and M. Baldauf. (2015). *A description of the nonhydrostatic \
        regional COSMO-model. Part I: Dynamics and numerics.* \
        Deutscher Wetterdienst, Germany.
    """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=np.float32,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        managed_memory=False,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary
        )

        self._backend = backend
        self._exec_info = exec_info
        self._default_origin = default_origin
        self._managed_memory = managed_memory

        set_annotations(forward_euler, dtype)

        self._stencil = gtscript.stencil(
            definition=forward_euler,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            **(backend_opts or {})
        )

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)

        # extract the raw arrays in the correct units
        names = tuple(name for name in out_units)
        raw_state = {name: state[name].to_units(out_units[name]).values for name in names}
        raw_k0 = {name: k0[name].values for name in names}
        raw_out_state = {name: out_state[name].values for name in names}

        # update the solution
        storage_shape = raw_state[names[0]].shape
        origin = (self._hb.nb, self._hb.nb, 0) if self._enforce_hb else (0, 0, 0)
        iteration_domain = tuple(storage_shape[i] - 2 * origin[i] for i in range(3))
        for name in raw_out_state:
            self._stencil(
                in_field=raw_state[name],
                in_tnd=raw_k0[name],
                out_field=raw_out_state[name],
                dt=timestep.total_seconds() / 3.0,
                origin={"_all_": origin},
                domain=iteration_domain,
                exec_info=self._exec_info,
            )
        out_state["time"] = state["time"] + timestep / 3.0

        # populate out_state with all other variables from state
        for name in state:
            if name != "time" and name not in out_state:
                out_state[name] = state[name]

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k0)

        # second stage
        k1, _ = get_increment(out_state, timestep, self.prognostic)

        # extract the raw arrays in the correct units
        raw_k1 = {name: k1[name].values for name in names}

        # update the solution
        for name in raw_out_state:
            self._stencil(
                in_field=raw_state[name],
                in_tnd=raw_k1[name],
                out_field=raw_out_state[name],
                dt=0.5 * timestep.total_seconds(),
                origin={"_all_": origin},
                domain=iteration_domain,
                exec_info=self._exec_info,
            )
        out_state["time"] = state["time"] + 0.5 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k1)

        # third stage
        k2, _ = get_increment(out_state, timestep, self.prognostic)

        # extract the raw arrays in the correct units
        raw_k2 = {name: k2[name].values for name in names}

        # update the solution
        for name in raw_out_state:
            self._stencil(
                in_field=raw_state[name],
                in_tnd=raw_k2[name],
                out_field=raw_out_state[name],
                dt=timestep.total_seconds(),
                origin={"_all_": origin},
                domain=iteration_domain,
                exec_info=self._exec_info,
            )
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        # restore_tendency_units(k2)

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        return diagnostics, out_state


class RungeKutta3(TendencyStepper):
    """
    The three-stages, third-order Runge-Kutta scheme.

    References
    ----------
    Gear, C. W. (1971). *Numerical initial value problems in \
        ordinary differential equations.* Prentice Hall PTR.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # free parameters for RK3
        self._alpha1 = 1.0 / 2.0
        self._alpha2 = 3.0 / 4.0

        # set the other parameters yielding a third-order method
        self._gamma1 = (3.0 * self._alpha2 - 2.0) / (
            6.0 * self._alpha1 * (self._alpha2 - self._alpha1)
        )
        self._gamma2 = (3.0 * self._alpha1 - 2.0) / (
            6.0 * self._alpha2 * (self._alpha1 - self._alpha2)
        )
        self._gamma0 = 1.0 - self._gamma1 - self._gamma2
        self._beta21 = self._alpha2 - 1.0 / (6.0 * self._alpha1 * self._gamma2)

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }
        a1, a2 = self._alpha1, self._alpha2
        b21 = self._beta21
        g0, g1, g2 = self._gamma0, self._gamma1, self._gamma2
        dt = timestep.total_seconds()

        # initialize the output state
        if self._out_state is None:
            self._out_state = self._allocate_output_state(state)
        out_state = self._out_state

        # first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)
        multiply(a1 * dt, k0, out=out_state, units=out_units)
        add_inplace(out_state, state, units=out_units, unshared_variables_in_output=True)
        out_state["time"] = state["time"] + a1 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # second stage
        k1, _ = get_increment(out_state, timestep, self.prognostic)
        state_2 = add(
            state,
            add(multiply(b21 * dt, k0), multiply((a2 - b21) * dt, k1)),
            units=out_units,
            unshared_variables_in_output=True,
        )
        state_2["time"] = state["time"] + a2 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                state_2, field_names=self.output_properties.keys(), grid=self._grid
            )

        # third stage
        k2, _ = get_increment(state_2, timestep, self.prognostic)
        k1k2 = add(multiply(g1 * dt, k1), multiply(g2 * dt, k2))
        k0k1k2 = add(multiply(g0 * dt, k0), k1k2)
        out_state = add(
            state, k0k1k2, units=out_units, unshared_variables_in_output=False
        )
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        restore_tendency_units(k0)
        restore_tendency_units(k1)
        restore_tendency_units(k2)

        return diagnostics, out_state
