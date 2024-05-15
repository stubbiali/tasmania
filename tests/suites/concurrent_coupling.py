# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
from property_cached import cached_property
from typing import Optional, Dict, TYPE_CHECKING, Tuple

from sympl._core.units import units_are_same

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.utils.storage import deepcopy_dataarray_dict

from tests.suites.component import ComponentTestSuite
from tests.suites.core_components import (
    DiagnosticComponentTestSuite,
    TendencyComponentTestSuite,
)
from tests.suites.domain import DomainSuite

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict, NDArrayLike, NDArrayLikeDict

    from tasmania.python.utils.typingx import TendencyComponent, TimeDelta


class ConcurrentCouplingTestSuite(ComponentTestSuite):
    def __init__(
        self,
        domain_suite: DomainSuite,
        *args: [DiagnosticComponentTestSuite, TendencyComponentTestSuite],
        execution_policy: str = "serial"
    ):
        super().__init__(domain_suite)

        self.args = args
        components = [arg.component for arg in args]

        self.coupler = ConcurrentCoupling(
            *components,
            execution_policy=execution_policy,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_options=self.ds.so
        )

        self.input_properties = self.coupler.input_properties
        self.tendency_properties = self.coupler.tendency_properties
        self.diagnostic_properties = self.coupler.diagnostic_properties

        self.out_tendencies = self.get_out_tendencies()
        self.out_diagnostics = self.get_out_diagnostics()
        self.overwrite_tendencies = self.get_overwrite_tendencies()

        self.out_tendencies_dc = (
            deepcopy_dataarray_dict(self.out_tendencies)
            if self.out_tendencies is not None
            else {}
        )
        self.out_diagnostics_dc = (
            deepcopy_dataarray_dict(self.out_diagnostics)
            if self.out_diagnostics is not None
            else {}
        )
        overwrite_tendencies = self.overwrite_tendencies or {}
        self.overwrite_tendencies_dc = {
            name: overwrite_tendencies.get(name, True)
            or name not in self.out_tendencies_dc
            for name in self.tendency_properties
        }

    @cached_property
    def component(self) -> "TendencyComponent":
        return self.coupler

    def assert_allclose(
        self, name: str, field_a: "NDArrayLike", field_b: "NDArrayLike"
    ) -> None:
        self.args[0].assert_allclose(name, field_a, field_b)

    def get_out_tendencies(self) -> Optional["DataArrayDict"]:
        out = None
        for arg in self.args:
            if hasattr(arg, "get_out_tendencies"):
                tendencies = arg.get_out_tendencies()
                if out is None:
                    out = tendencies
                elif tendencies is not None:
                    out.update(tendencies)
        return out

    def get_out_diagnostics(self) -> Optional["DataArrayDict"]:
        out = None
        for arg in self.args:
            diagnostics = arg.get_out_diagnostics()
            if out is None:
                out = diagnostics
            elif diagnostics is not None:
                out.update(diagnostics)
        return out

    def get_overwrite_tendencies(self) -> Optional[Dict[str, bool]]:
        out = None
        for arg in self.args:
            if hasattr(arg, "get_overwrite_tendencies"):
                overwrite_tendencies = arg.get_overwrite_tendencies()
                if out is None:
                    out = overwrite_tendencies
                elif overwrite_tendencies is not None:
                    out.update(overwrite_tendencies)
        return out

    def get_state(self) -> "DataArrayDict":
        out = {}
        for arg in self.args:
            out.update(arg.get_state())
        return out

    def get_validation_tendencies_and_diagnostics(
        self, raw_state_np: "NDArrayLikeDict", dt: float
    ) -> Tuple["NDArrayLikeDict", "NDArrayLikeDict"]:
        raw_tendencies_np = {}
        raw_diagnostics_np = {}

        for arg in self.args:
            try:
                (
                    arg_tends_np,
                    arg_diags_np,
                ) = arg.get_validation_tendencies_and_diagnostics(
                    raw_state_np, dt
                )
            except AttributeError:
                arg_tends_np = {}
                arg_diags_np = arg.get_validation_diagnostics(raw_state_np)

            for name in arg_tends_np:
                if name in raw_tendencies_np:
                    raw_tendencies_np[name] += arg_tends_np[name]
                else:
                    raw_tendencies_np[name] = arg_tends_np[name]

            raw_diagnostics_np.update(arg_diags_np)

        return raw_tendencies_np, raw_diagnostics_np

    def run(
        self,
        state: Optional["DataArrayDict"] = None,
        timestep: Optional["TimeDelta"] = None,
    ):
        state = state or self.get_state()
        timestep = timestep or self.get_timestep()

        tendencies, diagnostics = self.component(
            state,
            timestep,
            out_tendencies=self.out_tendencies,
            out_diagnostics=self.out_diagnostics,
            overwrite_tendencies=self.overwrite_tendencies,
        )

        raw_state_np = {
            name: to_numpy(
                state[name].to_units(self.input_properties[name]["units"]).data
            )
            for name in self.input_properties
        }
        raw_state_np["time"] = state["time"]

        (
            raw_tendencies_np,
            raw_diagnostics_np,
        ) = self.get_validation_tendencies_and_diagnostics(
            raw_state_np, timestep.total_seconds()
        )

        for name in self.tendency_properties:
            assert name in tendencies
            assert units_are_same(
                tendencies[name].attrs["units"],
                self.tendency_properties[name]["units"],
            )
            if self.overwrite_tendencies_dc[name]:
                val = raw_tendencies_np[name]
            else:
                val = (
                    to_numpy(self.out_tendencies_dc[name].data)
                    + raw_tendencies_np[name]
                )
            self.args[0].assert_allclose(name, tendencies[name].data, val)

        for name in self.diagnostic_properties:
            assert name in diagnostics
            assert units_are_same(
                diagnostics[name].attrs["units"],
                self.diagnostic_properties[name]["units"],
            )
            self.args[0].assert_allclose(
                name, diagnostics[name].data, raw_diagnostics_np[name]
            )
