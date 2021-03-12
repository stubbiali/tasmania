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
from tasmania.python.framework.register import register
from tasmania.python.framework.tendency_stepper import TendencyStepper
from tasmania.python.utils.framework import get_increment


@register(name="forward_euler")
class ForwardEuler(TendencyStepper):
    """The forward Euler scheme."""

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        backend_options=None,
        storage_options=None,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options
        )

    def _call(self, state, timestep):
        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # calculate the tendencies and the diagnostics
        tendencies, diagnostics = get_increment(
            state, timestep, self.prognostic
        )

        # step the solution
        self._dict_op.fma(
            state,
            tendencies,
            timestep.total_seconds(),
            out=out_state,
            field_properties=self.output_properties,
        )
        out_state["time"] = state["time"] + timestep

        if self._enforce_hb:
            # enforce the boundary conditions on each prognostic variable
            self._hb.enforce(
                out_state, field_names=self.output_properties.keys(),
            )

        # restore original units for the tendencies
        # restore_tendency_units(tendencies)

        return diagnostics, out_state
