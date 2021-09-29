# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from tasmania.python.framework.steppers import SequentialTendencyStepper


class RK3WS(SequentialTendencyStepper):
    """The Wicker-Skamarock three-stages Runge-Kutta scheme.

    References
    ----------
    Doms, G., and M. Baldauf. (2015). *A description of the nonhydrostatic \
        regional COSMO-model. Part I: Dynamics and numerics.* \
        Deutscher Wetterdienst, Germany.
    """

    name = "rk3ws"

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        enable_checks=True,
        backend="numpy",
        backend_options=None,
        storage_options=None,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary,
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options
        )
        self._increment = None
        self._diagnostics = None

    def _call(self, state, prv_state, timestep, out_diagnostics, out_state):
        # first stage
        (
            self._increment,
            out_diagnostics,
        ) = self._stepper_operator.get_increment(
            state,
            timestep,
            out_increment=self._increment,
            out_diagnostics=out_diagnostics,
        )
        self._dict_op.sts_rk3ws_0(
            timestep.total_seconds(),
            state,
            prv_state,
            self._increment,
            out=out_state,
            field_properties=self.output_properties,
        )
        out_state["time"] = state["time"] + timestep / 3.0

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys()
            )

        # populate out_state with all other variables from state
        for name in state:
            if name != "time" and name not in out_state:
                out_state[name] = state[name]

        # second stage
        (
            self._increment,
            self._diagnostics,
        ) = self._stepper_operator.get_increment(
            out_state,
            timestep,
            out_increment=self._increment,
            out_diagnostics=self._diagnostics,
        )

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        # step the solution
        self._dict_op.sts_rk2_0(
            timestep.total_seconds(),
            state,
            prv_state,
            self._increment,
            out=out_state,
            field_properties=self.output_properties,
        )
        out_state["time"] = state["time"] + 0.5 * timestep

        if self._enforce_hb:
            # Enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys()
            )

        # populate out_state with all other variables from state
        for name in state:
            if name != "time" and name not in out_state:
                out_state[name] = state[name]

        # third stage
        (
            self._increment,
            self._diagnostics,
        ) = self._stepper_operator.get_increment(
            out_state,
            timestep,
            out_increment=self._increment,
            out_diagnostics=self._diagnostics,
        )

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        # step the solution
        self._dict_op.fma(
            prv_state,
            self._increment,
            timestep.total_seconds(),
            out=out_state,
            field_properties=self.output_properties,
        )
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys()
            )

        return out_diagnostics, out_state
