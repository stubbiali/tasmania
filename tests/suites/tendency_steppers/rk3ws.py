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
from datetime import timedelta

from tests.suites.steppers import TendencyStepperTestSuite


class RK3WSTestSuite(TendencyStepperTestSuite):
    name = "rk3ws"

    def __init__(
        self,
        concurrent_coupling_suite,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
    ):
        super().__init__(
            concurrent_coupling_suite,
            "rk3ws",
            execution_policy,
            enforce_horizontal_boundary,
        )

    def get_validation_out_state(self, raw_state_np, dt):
        tends, _ = self.cc_suite.get_validation_tendencies_and_diagnostics(
            raw_state_np, dt
        )
        k0 = {"time": raw_state_np["time"] + timedelta(seconds=dt) / 3.0}
        k0.update(
            {
                name: raw_state_np[name] + dt * tends[name] / 3.0
                for name in tends
                if name != "time"
            }
        )
        k0.update(
            {
                name: raw_state_np[name]
                for name in raw_state_np
                if name not in k0
            }
        )
        if self.enforce_hb:
            self.hb_np.enforce_raw(k0, self.output_properties)

        tends, _ = self.cc_suite.get_validation_tendencies_and_diagnostics(
            k0, dt
        )
        k1 = {"time": raw_state_np["time"] + 0.5 * timedelta(seconds=dt)}
        k1.update(
            {
                name: raw_state_np[name] + 0.5 * dt * tends[name]
                for name in tends
                if name != "time"
            }
        )
        k1.update(
            {
                name: raw_state_np[name]
                for name in raw_state_np
                if name not in k1
            }
        )
        if self.enforce_hb:
            self.hb_np.enforce_raw(k1, self.output_properties)

        tends, _ = self.cc_suite.get_validation_tendencies_and_diagnostics(
            k1, dt
        )
        out = {"time": raw_state_np["time"] + timedelta(seconds=dt)}
        out.update(
            {name: raw_state_np[name] + dt * tends[name] for name in tends}
        )
        if self.enforce_hb:
            self.hb_np.enforce_raw(out, self.output_properties)
        return out
