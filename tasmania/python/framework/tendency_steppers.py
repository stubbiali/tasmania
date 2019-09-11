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
This module contain:
	get_increment
	tendencystepper_factory
	TendencyStepper
	ForwardEuler
	RungeKutta2
	RungeKutta3WS
	RungeKutta3
"""
import abc
import copy
from sympl import (
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
)
from sympl._core.base_components import InputChecker, DiagnosticChecker, OutputChecker
from sympl._core.units import clean_units

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.utils.dict_utils import add, multiply
from tasmania.python.utils.framework_utils import check_property_compatibility
from tasmania.python.utils.utils import assert_sequence


def get_increment(state, timestep, prognostic):
    # calculate tendencies and retrieve diagnostics
    tendencies, diagnostics = prognostic(state, timestep)

    # multiply the tendencies by the time step
    for name in tendencies:
        if name != "time":
            tendencies[name].attrs["units"] += " s"

    return tendencies, diagnostics


def restore_tendencies_units(tendencies):
    for name in tendencies:
        if name != "time":
            tendencies[name].attrs["units"] = clean_units(
                tendencies[name].attrs["units"] + " s^-1"
            )


def tendencystepper_factory(scheme):
    if scheme == "forward_euler":
        return ForwardEuler
    elif scheme == "rk2":
        return RungeKutta2
    elif scheme == "rk3ws":
        return RungeKutta3WS
    elif scheme == "rk3":
        return RungeKutta3
    else:
        raise ValueError(
            "Unsupported time integration scheme "
            "{}"
            ". "
            "Available integrators: forward_euler, rk2, rk3ws, rk3.".format(scheme)
        )


class TendencyStepper:
    """
	Callable abstract base class which steps a model state based on the 
	tendencies calculated by a set of wrapped prognostic components.
	"""

    __metaclass__ = abc.ABCMeta

    allowed_component_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        ConcurrentCoupling,
    )

    def __init__(self, *args, **kwargs):
        """
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : str
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		enforce_horizontal_boundary : bool
			:obj:`True` if the class should enforce the lateral boundary
			conditions after each stage of the time integrator,
			:obj:`False` otherwise. Defaults to :obj:`False`.
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
            else ConcurrentCoupling(
                *args,
                execution_policy=kwargs.get("execution_policy", "serial"),
                time_units=kwargs.get("time_units", "s")
            )
        )

        self._tunits = kwargs.get("time_units", "s")

        self._input_checker = InputChecker(self)
        self._diagnostic_checker = DiagnosticChecker(self)
        self._output_checker = OutputChecker(self)

        enforce_hb = kwargs.get("enforce_horizontal_boundary", False)
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

    @property
    def prognostic(self):
        """
		Return
		------
		obj :
			The :class:`tasmania.ConcurrentCoupling` calculating the tendencies.
		"""
        return self._prognostic

    @property
    def input_properties(self):
        """
		Return
		------
		dict :
			Dictionary whose keys are strings denoting model variables
			which should be present in the input state dictionary, and
			whose values are dictionaries specifying fundamental properties
			(dims, units) of those variables.
		"""
        return_dict = {}
        return_dict.update(self._prognostic.input_properties)

        tendency_properties = self._prognostic.tendency_properties
        for name in tendency_properties:
            mod_tendency_property = copy.deepcopy(tendency_properties[name])
            mod_tendency_property["units"] = clean_units(
                mod_tendency_property["units"] + self._tunits
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
		dict :
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
		dict :
			Dictionary whose keys are strings denoting model variables
			present in the output state dictionary, and whose values are 
			dictionaries specifying fundamental properties (dims, units) 
			of those variables.
		"""
        return_dict = {}

        for key, val in self._prognostic.tendency_properties.items():
            return_dict[key] = copy.deepcopy(val)
            if "units" in return_dict[key]:
                return_dict[key]["units"] = clean_units(
                    return_dict[key]["units"] + self._tunits
                )

        return return_dict

    def __call__(self, state, timestep):
        """
		Step the model state.

		Parameters
		----------
		state : dict
			Dictionary whose keys are strings denoting the input model 
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the time step.
			
		Return
		------
		diagnostics : dict
			Dictionary whose keys are strings denoting diagnostic
			variables retrieved from the input state, and whose values
			are :class:`sympl.DataArray`\s storing values for those
			variables.
		out_state : dict
			Dictionary whose keys are strings denoting the output model
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
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
		state : dict
			Dictionary whose keys are strings denoting the input model 
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		timestep : timedelta
			:class:`datetime.timedelta` representing the time step.
			
		Return
		------
		diagnostics : dict
			Dictionary whose keys are strings denoting diagnostic
			variables retrieved from the input state, and whose values
			are :class:`sympl.DataArray`\s storing values for those
			variables.
		out_state : dict
			Dictionary whose keys are strings denoting the output model
			variables, and whose values are :class:`sympl.DataArray`\s
			storing values for those variables.
		"""
        pass


class ForwardEuler(TendencyStepper):
    """
	This class inherits :class:`tasmania.TendencyStepper` to
	implement the forward Euler time integration scheme.
	"""

    def __init__(self, *args, **kwargs):
        """
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : str
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		enforce_horizontal_boundary : bool
			:obj:`True` if the class should enforce the lateral boundary
			conditions after each stage of the time integrator,
			:obj:`False` otherwise. Defaults to :obj:`False`.
			This argument is considered only if at least one of the wrapped
			objects is an instance of

				* :class:`tasmania.TendencyComponent`, or
				* :class:`tasmania.ImplicitTendencyComponent`.

		time_units : str
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
        super().__init__(*args, **kwargs)

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # calculate the increment and the diagnostics
        increment, diagnostics = get_increment(state, timestep, self.prognostic)

        # step the solution
        out_state = add(
            state,
            multiply(timestep.total_seconds(), increment),
            units=out_units,
            unshared_variables_in_output=False,
        )

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        restore_tendencies_units(increment)

        return diagnostics, out_state


class RungeKutta2(TendencyStepper):
    """
	This class inherits :class:`tasmania.TendencyStepper` to
	implement the two-stages, second-order Runge-Kutta scheme.

	References
	----------
	Gear, C. W. (1971). *Numerical initial value problems in \
		ordinary differential equations.* Prentice Hall PTR.
	"""

    def __init__(self, *args, **kwargs):
        """
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : str
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		enforce_horizontal_boundary : bool
			:obj:`True` if the class should enforce the lateral boundary
			conditions after each stage of the time integrator,
			:obj:`False` otherwise. Defaults to :obj:`False`.
			This argument is considered only if at least one of the wrapped
			objects is an instance of

				* :class:`tasmania.TendencyComponent`, or
				* :class:`tasmania.ImplicitTendencyComponent`.

		time_units : str
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
        super().__init__(*args, **kwargs)

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)
        state_1 = add(
            state,
            multiply(0.5 * timestep.total_seconds(), k0),
            units=out_units,
            unshared_variables_in_output=True,
        )
        state_1["time"] = state["time"] + 0.5 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                state_1, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        restore_tendencies_units(k0)

        # second stage
        k1, _ = get_increment(state_1, timestep, self.prognostic)
        out_state = add(
            state,
            multiply(timestep.total_seconds(), k1),
            units=out_units,
            unshared_variables_in_output=False,
        )
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        restore_tendencies_units(k1)

        return diagnostics, out_state


class RungeKutta3WS(TendencyStepper):
    """
	This class inherits :class:`tasmania.TendencyStepper` to
	implement the three-stages Runge-Kutta scheme as used in the
	`COSMO model <http://www.cosmo-model.org>`_. This integrator is
	nominally second-order, and third-order for linear problems.

	References
	----------
	Doms, G., and M. Baldauf. (2015). *A description of the nonhydrostatic \
		regional COSMO-model. Part I: Dynamics and numerics.* \
		Deutscher Wetterdienst, Germany.
	"""

    def __init__(self, *args, **kwargs):
        """
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : str
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		enforce_horizontal_boundary : bool
			:obj:`True` if the class should enforce the lateral boundary
			conditions after each stage of the time integrator,
			:obj:`False` otherwise. Defaults to :obj:`False`.
			This argument is considered only if at least one of the wrapped
			objects is an instance of

				* :class:`tasmania.TendencyComponent`, or
				* :class:`tasmania.ImplicitTendencyComponent`.

		time_units : str
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
        super().__init__(*args, **kwargs)

    def _call(self, state, timestep):
        # shortcuts
        out_units = {
            name: properties["units"]
            for name, properties in self.output_properties.items()
        }

        # first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)
        state_1 = add(
            state,
            multiply(1.0 / 3.0 * timestep.total_seconds(), k0),
            units=out_units,
            unshared_variables_in_output=True,
        )
        state_1["time"] = state["time"] + 1.0 / 3.0 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                state_1, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        restore_tendencies_units(k0)

        # second stage
        k1, _ = get_increment(state_1, timestep, self.prognostic)
        state_2 = add(
            state,
            multiply(0.5 * timestep.total_seconds(), k1),
            units=out_units,
            unshared_variables_in_output=True,
        )
        state_2["time"] = state["time"] + 0.5 * timestep

        if self._enforce_hb:
            # Enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                state_2, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        restore_tendencies_units(k1)

        # second stage
        k2, _ = get_increment(state_2, timestep, self.prognostic)
        out_state = add(
            state,
            multiply(timestep.total_seconds(), k2),
            units=out_units,
            unshared_variables_in_output=False,
        )
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(), grid=self._grid
            )

        # restore original units for the tendencies
        restore_tendencies_units(k2)

        return diagnostics, out_state


class RungeKutta3(TendencyStepper):
    """
	This class inherits :class:`tasmania.TendencyStepper` to
	implement the three-stages, third-order Runge-Kutta scheme.

	References
	----------
	Gear, C. W. (1971). *Numerical initial value problems in \
		ordinary differential equations.* Prentice Hall PTR.
	"""

    def __init__(self, *args, **kwargs):
        """
		Parameters
		----------
		obj :
			Instances of

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			providing tendencies for the prognostic variables.

		Keyword arguments
		-----------------
		execution_policy : str
			String specifying the runtime mode in which parameterizations
			should be invoked. See :class:`tasmania.ConcurrentCoupling`.
		enforce_horizontal_boundary : bool
			:obj:`True` if the class should enforce the lateral boundary
			conditions after each stage of the time integrator,
			:obj:`False` otherwise. Defaults to :obj:`False`.
			This argument is considered only if at least one of the wrapped
			objects is an instance of

				* :class:`tasmania.TendencyComponent`, or
				* :class:`tasmania.ImplicitTendencyComponent`.

		time_units : str
			The time units used within this object. Defaults to 's', i.e., seconds.
		"""
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

        # first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)
        state_1 = add(
            state,
            multiply(a1 * dt, k0),
            units=out_units,
            unshared_variables_in_output=True,
        )
        state_1["time"] = state["time"] + a1 * timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                state_1, field_names=self.output_properties.keys(), grid=self._grid
            )

        # second stage
        k1, _ = get_increment(state_1, timestep, self.prognostic)
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
        restore_tendencies_units(k0)
        restore_tendencies_units(k1)
        restore_tendencies_units(k2)

        return diagnostics, out_state
