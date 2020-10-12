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
import numpy as np

from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper
from tasmania.python.utils.framework_utils import get_increment, register


@register(name="rk3ws")
class RK3WS(STSTendencyStepper):
    """ The Wicker-Skamarock three-stages Runge-Kutta scheme.

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
        dtype=np.float64,
        build_info=None,
        rebuild=False,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary,
            backend=backend,
            backend_opts=backend_opts,
            dtype=dtype,
            build_info=build_info,
            rebuild=rebuild
        )

    def _call(self, state, prv_state, timestep):
        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # first stage
        k0, diagnostics = get_increment(state, timestep, self.prognostic)
        self._dict_op.sts_rk3ws_0(
            timestep.total_seconds(),
            state,
            prv_state,
            k0,
            out=out_state,
            field_properties=self.output_properties,
        )
        out_state["time"] = state["time"] + timestep / 3.0

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state,
                field_names=self.output_properties.keys(),
                grid=self._grid,
            )

        # populate out_state with all other variables from state
        for name in state:
            if name != "time" and name not in out_state:
                out_state[name] = state[name]

        # restore original units of the tendencies
        # restore_tendency_units(k0)

        # second stage
        k1, _ = get_increment(out_state, timestep, self.prognostic)

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        # step the solution
        self._dict_op.sts_rk2_0(
            timestep.total_seconds(),
            state,
            prv_state,
            k1,
            out=out_state,
            field_properties=self.output_properties,
        )
        out_state["time"] = state["time"] + 0.5 * timestep

        if self._enforce_hb:
            # Enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state,
                field_names=self.output_properties.keys(),
                grid=self._grid,
            )

        # populate out_state with all other variables from state
        for name in state:
            if name != "time" and name not in out_state:
                out_state[name] = state[name]

        # restore original units of the tendencies
        # restore_tendency_units(k1)

        # third stage
        k2, _ = get_increment(out_state, timestep, self.prognostic)

        # remove undesired variables
        for name in state:
            if name != "time" and name not in self.output_properties:
                out_state.pop(name, None)

        # step the solution
        self._dict_op.fma(
            prv_state,
            k2,
            timestep.total_seconds(),
            out=out_state,
            field_properties=self.output_properties,
        )
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state,
                field_names=self.output_properties.keys(),
                grid=self._grid,
            )

        # restore original units of the tendencies
        # restore_tendency_units(k2)

        return diagnostics, out_state
