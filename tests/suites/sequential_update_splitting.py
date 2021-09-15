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
from typing import Optional, TYPE_CHECKING, Union

from sympl._core.units import units_are_same

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import TimeIntegrationOptions
from tasmania.python.framework.sequential_update_splitting import (
    SequentialUpdateSplitting,
)

from tests.suites.core_components import DiagnosticComponentTestSuite
from tests.suites.steppers import TendencyStepperTestSuite

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict, NDArrayLikeDict

    from tasmania.python.utils.typingx import TimeDelta


class SequentialUpdateSplittingTestSuite:
    def __init__(
        self,
        *args: Union[DiagnosticComponentTestSuite, TendencyStepperTestSuite]
    ):
        self.args = args

        sus_args = []
        for arg in args:
            if isinstance(arg, TendencyStepperTestSuite):
                sus_args.append(
                    TimeIntegrationOptions(
                        arg.cc_suite.component,
                        scheme=arg.name,
                        enforce_horizontal_boundary=arg.enforce_hb,
                        backend=arg.ds.backend,
                        backend_options=arg.ds.bo,
                        storage_options=arg.ds.so,
                    )
                )
            else:
                sus_args.append(TimeIntegrationOptions(arg.component))

        self.splitter = SequentialUpdateSplitting(*sus_args)

        self.input_properties = self.splitter.input_properties
        self.output_properties = self.splitter.output_properties

    def get_state(self) -> "DataArrayDict":
        return self.args[0].get_state()

    def get_timestep(self) -> "TimeDelta":
        return self.args[0].get_timestep()

    def get_validation_state(
        self, raw_state_np: "NDArrayLikeDict", dt: float
    ) -> "NDArrayLikeDict":
        out = raw_state_np.copy()

        for arg in self.args:
            if isinstance(arg, TendencyStepperTestSuite):
                (
                    _,
                    diags,
                ) = arg.cc_suite.get_validation_tendencies_and_diagnostics(
                    out, dt
                )
                new_state = arg.get_validation_out_state(out, dt)
                out.update(diags)
                out.update(new_state)
            else:
                diags = arg.get_validation_diagnostics(out)
                out.update(diags)

        return out

    def run(
        self,
        state: Optional["DataArrayDict"] = None,
        timestep: Optional["TimeDelta"] = None,
    ) -> None:
        state = state or self.get_state()
        timestep = timestep or self.get_timestep()

        raw_state_np = {
            name: to_numpy(
                state[name]
                .to_units(self.splitter.input_properties[name]["units"])
                .data
            )
            for name in self.splitter.input_properties
        }
        raw_state_np["time"] = state["time"]
        raw_state_val = self.get_validation_state(
            raw_state_np, timestep.total_seconds()
        )

        self.splitter(state, timestep)

        for name in self.splitter.output_properties:
            assert name in state
            assert units_are_same(
                state[name].attrs["units"],
                self.splitter.output_properties[name]["units"],
            )
            self.args[0].assert_allclose(
                name, state[name].data, raw_state_val[name]
            )
