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
from tasmania.python.framework.parallel_splitting import ParallelSplitting

from tests.suites.core_components import DiagnosticComponentTestSuite
from tests.suites.tendency_steppers import TendencyStepperTestSuite

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict, NDArrayLikeDict

    from tasmania.python.utils.typingx import TimeDelta

    from tests.suites.domain import DomainSuite


class ParallelSplittingTestSuite:
    def __init__(
        self,
        domain_suite: "DomainSuite",
        *args: Union[DiagnosticComponentTestSuite, TendencyStepperTestSuite],
        execution_policy: str = "serial",
        retrieve_diagnostics_from_provisional_state: bool = False
    ):
        self.ds = domain_suite
        self.args = args
        self.policy = execution_policy
        self.from_provisional = retrieve_diagnostics_from_provisional_state

        ps_args = []
        for arg in args:
            if isinstance(arg, TendencyStepperTestSuite):
                ps_args.append(
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
                ps_args.append(TimeIntegrationOptions(arg.component))

        self.splitter = ParallelSplitting(
            *ps_args,
            execution_policy=execution_policy,
            retrieve_diagnostics_from_provisional_state=retrieve_diagnostics_from_provisional_state,
            backend=domain_suite.backend,
            backend_options=domain_suite.bo,
            storage_options=domain_suite.so
        )

    def get_state(self) -> "DataArrayDict":
        if self.policy == "serial":
            return self.args[0].get_state()
        else:
            out = {}
            for arg in self.args:
                out.update(arg.get_state())
            return out

    def get_timestep(self) -> "TimeDelta":
        return self.args[0].get_timestep()

    def get_validation_state(
        self, raw_state_np: "NDArrayLikeDict", dt: float
    ) -> "NDArrayLikeDict":
        out = raw_state_np.copy()
        agg_diags = {}

        for arg in self.args:
            if isinstance(arg, TendencyStepperTestSuite):
                (
                    _,
                    diags,
                ) = arg.cc_suite.get_validation_tendencies_and_diagnostics(
                    out, dt
                )
                out.update(diags)
            elif self.policy == "serial":
                diags = arg.get_validation_diagnostics(out)
                out.update(diags)
            else:
                diags = arg.get_validation_diagnostics(raw_state_np)
                agg_diags.update(diags)

        if self.policy == "as_parallel":
            out.update(agg_diags)

        return out

    def get_validation_provisional_state(
        self,
        raw_state_np: "NDArrayLikeDict",
        raw_state_prv_np: "NDArrayLikeDict",
        dt: float,
    ) -> "NDArrayLikeDict":
        aux = raw_state_np.copy()
        out = {"time": raw_state_prv_np["time"]}
        out.update(
            {
                name: raw_state_prv_np[name].copy()
                for name in raw_state_prv_np
                if name != "time"
            }
        )

        for arg in self.args:
            if isinstance(arg, TendencyStepperTestSuite):
                (
                    _,
                    diags,
                ) = arg.cc_suite.get_validation_tendencies_and_diagnostics(
                    aux, dt
                )
                new_state = arg.get_validation_out_state(aux, dt)
                aux.update(diags)
                for name in new_state:
                    if name != "time":
                        out[name] += new_state[name] - aux[name]
            elif self.policy == "serial":
                if self.from_provisional:
                    diags = arg.get_validation_diagnostics(out)
                    out.update(diags)
                else:
                    diags = arg.get_validation_diagnostics(aux)
                    aux.update(diags)

        return out

    def run(
        self,
        state: Optional["DataArrayDict"] = None,
        state_prv: Optional["DataArrayDict"] = None,
        timestep: Optional["TimeDelta"] = None,
    ) -> None:
        state = state or self.get_state()
        state_prv = state_prv or self.get_state()
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
        raw_state_prv_np = {
            name: to_numpy(
                state_prv[name]
                .to_units(
                    self.splitter.provisional_input_properties[name]["units"]
                )
                .data
            )
            for name in self.splitter.provisional_input_properties
        }
        raw_state_prv_np["time"] = state_prv["time"]
        raw_state_val = self.get_validation_state(
            raw_state_np, timestep.total_seconds()
        )
        raw_state_prv_val = self.get_validation_provisional_state(
            raw_state_np, raw_state_prv_np, timestep.total_seconds()
        )

        self.splitter(state, state_prv, timestep)

        for name in self.splitter.output_properties:
            assert name in state
            assert units_are_same(
                state[name].attrs["units"],
                self.splitter.output_properties[name]["units"],
            )
            self.args[0].assert_allclose(
                name, state[name].data, raw_state_val[name]
            )

        for name in self.splitter.provisional_output_properties:
            assert name in state_prv
            assert units_are_same(
                state_prv[name].attrs["units"],
                self.splitter.provisional_output_properties[name]["units"],
            )
            self.args[0].assert_allclose(
                name, state_prv[name].data, raw_state_prv_val[name]
            )
