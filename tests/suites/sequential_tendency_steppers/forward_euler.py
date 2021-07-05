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
from datetime import timedelta

from tests.suites.steppers import SequentialTendencyStepperTestSuite


class ForwardEulerTestSuite(SequentialTendencyStepperTestSuite):
    name = "forward_euler"

    def __init__(
        self,
        concurrent_coupling_suite,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
    ):
        super().__init__(
            concurrent_coupling_suite,
            "forward_euler",
            execution_policy,
            enforce_horizontal_boundary,
        )

    def get_validation_out_state(self, raw_state_np, raw_prv_state_np, dt):
        tends, _ = self.cc_suite.get_validation_tendencies_and_diagnostics(
            raw_state_np, dt
        )
        out = {"time": raw_state_np["time"] + timedelta(seconds=dt)}
        out.update(
            {name: raw_prv_state_np[name] + dt * tends[name] for name in tends}
        )
        if self.enforce_hb:
            self.hb_np.enforce_raw(out, self.output_properties)
        return out
