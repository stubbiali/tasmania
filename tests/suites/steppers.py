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
import abc
from property_cached import cached_property
from typing import Any, Optional, TYPE_CHECKING, Union

from sympl._core.factory import Factory
from sympl._core.units import units_are_same

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.steppers import (
    SequentialTendencyStepper,
    TendencyStepper,
)
from tasmania.python.utils.storage import deepcopy_dataarray_dict

from tests.suites.concurrent_coupling import ConcurrentCouplingTestSuite
from tests.suites.core_components import TendencyComponentTestSuite

if TYPE_CHECKING:
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLike,
        NDArrayLikeDict,
        Stepper,
    )

    from tasmania.python.utils.typingx import TimeDelta


class StepperTestSuite:

    stepper_cls = None

    def __init__(
        self,
        concurrent_coupling_suite: Union[
            ConcurrentCouplingTestSuite, TendencyComponentTestSuite
        ],
        scheme: str,
        execution_policy: str = "serial",
        enforce_horizontal_boundary: bool = False,
        **kwargs: Any
    ) -> None:
        self.cc_suite = concurrent_coupling_suite
        self.ds = concurrent_coupling_suite.ds
        self.enforce_hb = enforce_horizontal_boundary

        self.stepper = self.stepper_cls.factory(
            scheme,
            concurrent_coupling_suite.component,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_options=self.ds.so,
            **kwargs,
        )

        self.input_properties = self.stepper.input_properties
        self.diagnostic_properties = self.stepper.diagnostic_properties
        self.output_properties = self.stepper.output_properties

        self.out_diagnostics = concurrent_coupling_suite.get_out_diagnostics()
        self.out_state = self.get_out_state()

        self.out_diagnostics_dc = (
            deepcopy_dataarray_dict(self.out_diagnostics)
            if self.out_diagnostics is not None
            else {}
        )
        self.out_state_dc = (
            deepcopy_dataarray_dict(self.out_state)
            if self.out_state is not None
            else {}
        )

        self.hb_np = None

    @cached_property
    def component(self) -> "Stepper":
        return self.stepper

    def get_state(self) -> "DataArrayDict":
        return self.cc_suite.get_state()

    def get_timestep(self) -> "TimeDelta":
        return self.cc_suite.get_timestep()

    def get_out_state(self) -> Optional["DataArrayDict"]:
        out = self.cc_suite.get_out_tendencies()
        if out is not None:
            for name in out:
                out[name].attrs["units"] += " s"
        return out

    def assert_allclose(
        self, name: str, field_a: "NDArrayLike", field_b: "NDArrayLike"
    ) -> None:
        self.cc_suite.assert_allclose(name, field_a, field_b)


class TendencyStepperTestSuite(StepperTestSuite, Factory):
    stepper_cls = TendencyStepper

    def run(
        self,
        state: Optional["DataArrayDict"] = None,
        timestep: Optional["TimeDelta"] = None,
    ):
        state = state or self.get_state()
        timestep = timestep or self.get_timestep()

        if self.enforce_hb:
            self.ds.domain.horizontal_boundary.reference_state = state
            self.hb_np = self.ds.domain.copy(
                backend="numpy"
            ).horizontal_boundary

        diagnostics, out_state = self.stepper(
            state,
            timestep,
            out_diagnostics=self.out_diagnostics,
            out_state=self.out_state,
        )

        raw_state_np = {
            name: to_numpy(
                state[name].to_units(self.input_properties[name]["units"]).data
            )
            for name in self.input_properties
        }
        raw_state_np["time"] = state["time"]

        self.cc_suite.overwrite_tendencies = {}
        self.cc_suite.overwrite_tendencies_dc = {}
        (
            _,
            raw_diagnostics_np,
        ) = self.cc_suite.get_validation_tendencies_and_diagnostics(
            raw_state_np, timestep.total_seconds()
        )
        for name in self.diagnostic_properties:
            assert name in diagnostics
            assert units_are_same(
                diagnostics[name].attrs["units"],
                self.diagnostic_properties[name]["units"],
            )
            self.assert_allclose(
                name, diagnostics[name].data, raw_diagnostics_np[name]
            )

        raw_out_state_np = self.get_validation_out_state(
            raw_state_np, timestep.total_seconds()
        )
        for name in self.output_properties:
            assert name in out_state
            assert units_are_same(
                out_state[name].attrs["units"],
                self.output_properties[name]["units"],
            )

            self.assert_allclose(
                name, out_state[name].data, raw_out_state_np[name]
            )

    @abc.abstractmethod
    def get_validation_out_state(
        self, raw_state_np: "NDArrayLikeDict", dt: float
    ) -> "NDArrayLikeDict":
        pass


class SequentialTendencyStepperTestSuite(StepperTestSuite, Factory):
    stepper_cls = SequentialTendencyStepper

    def __init__(
        self,
        concurrent_coupling_suite: Union[
            ConcurrentCouplingTestSuite, TendencyComponentTestSuite
        ],
        scheme: str,
        execution_policy: str = "serial",
        enforce_horizontal_boundary: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            concurrent_coupling_suite,
            scheme,
            execution_policy,
            enforce_horizontal_boundary,
            **kwargs
        )
        self.provisional_input_properties = (
            self.stepper.provisional_input_properties
        )

    def run(
        self,
        state: Optional["DataArrayDict"] = None,
        prv_state: Optional["DataArrayDict"] = None,
        timestep: Optional["TimeDelta"] = None,
    ):
        state = state or self.get_state()
        prv_state = prv_state or self.get_state()
        timestep = timestep or self.get_timestep()

        if self.enforce_hb:
            self.ds.domain.horizontal_boundary.reference_state = state
            self.hb_np = self.ds.domain.copy(
                backend="numpy"
            ).horizontal_boundary

        diagnostics, out_state = self.stepper(
            state,
            prv_state,
            timestep,
            out_diagnostics=self.out_diagnostics,
            out_state=self.out_state,
        )

        raw_state_np = {
            name: to_numpy(
                state[name].to_units(self.input_properties[name]["units"]).data
            )
            for name in self.input_properties
        }
        raw_state_np["time"] = state["time"]
        raw_prv_state_np = {
            name: to_numpy(
                prv_state[name]
                .to_units(self.provisional_input_properties[name]["units"])
                .data
            )
            for name in self.provisional_input_properties
        }

        self.cc_suite.overwrite_tendencies = {}
        self.cc_suite.overwrite_tendencies_dc = {}
        (
            _,
            raw_diagnostics_np,
        ) = self.cc_suite.get_validation_tendencies_and_diagnostics(
            raw_state_np, timestep.total_seconds()
        )
        for name in self.diagnostic_properties:
            assert name in diagnostics
            assert units_are_same(
                diagnostics[name].attrs["units"],
                self.diagnostic_properties[name]["units"],
            )
            self.assert_allclose(
                name, diagnostics[name].data, raw_diagnostics_np[name]
            )

        raw_out_state_np = self.get_validation_out_state(
            raw_state_np, raw_prv_state_np, timestep.total_seconds()
        )
        for name in self.output_properties:
            assert name in out_state
            assert units_are_same(
                out_state[name].attrs["units"],
                self.output_properties[name]["units"],
            )

            self.assert_allclose(
                name, out_state[name].data, raw_out_state_np[name]
            )

    @abc.abstractmethod
    def get_validation_out_state(
        self,
        raw_state_np: "NDArrayLikeDict",
        raw_prv_state_np: "NDArrayLikeDict",
        dt: float,
    ) -> "NDArrayLikeDict":
        pass
