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
from tasmania.python.framework.steppers import SequentialTendencyStepper


class ForwardEuler(SequentialTendencyStepper):
    """The forward Euler scheme."""

    name = "forward_euler"

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

    def _call(self, state, prv_state, timestep, out_diagnostics, out_state):
        # calculate the tendencies and the diagnostics
        (
            self._increment,
            out_diagnostics,
        ) = self._stepper_operator.get_increment(
            state,
            timestep,
            out_increment=self._increment,
            out_diagnostics=out_diagnostics,
        )

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
                out_state,
                field_names=self.output_properties.keys(),
            )

        return out_diagnostics, out_state
