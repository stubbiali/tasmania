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
from hypothesis import (
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tests import conf
from tests.suites.concurrent_coupling import ConcurrentCouplingTestSuite
from tests.suites.core_components import (
    FakeTendencyComponent1TestSuite,
    FakeTendencyComponent2TestSuite,
)
from tests.suites.domain import DomainSuite
from tests.suites.parallel_splitting import ParallelSplittingTestSuite
from tests.suites.steppers import TendencyStepperTestSuite
from tests.utilities import hyp_settings


@hyp_settings
@given(data=hyp_st.data())
def test_properties(data):
    ds = DomainSuite(data, "numpy", float)
    tcts1 = FakeTendencyComponent1TestSuite(ds)
    tsts1 = TendencyStepperTestSuite.factory(
        "forward_euler", tcts1, enforce_horizontal_boundary=False
    )
    tcts2 = FakeTendencyComponent2TestSuite(ds)
    tsts2 = TendencyStepperTestSuite.factory(
        "forward_euler", tcts2, enforce_horizontal_boundary=False
    )

    # >>> test 1
    ps = ParallelSplittingTestSuite(
        ds, tsts1, tsts2, execution_policy="as_parallel"
    )

    assert "air_isentropic_density" in ps.splitter.input_properties
    assert "fake_variable" in ps.splitter.input_properties
    assert "x_momentum_isentropic" in ps.splitter.input_properties
    assert "x_velocity" in ps.splitter.input_properties
    assert "x_velocity_at_u_locations" in ps.splitter.input_properties
    assert "y_momentum_isentropic" in ps.splitter.input_properties
    assert "y_velocity_at_v_locations" in ps.splitter.input_properties
    assert len(ps.splitter.input_properties) == 7

    assert "air_isentropic_density" in ps.splitter.provisional_input_properties
    assert "x_momentum_isentropic" in ps.splitter.provisional_input_properties
    assert "x_velocity" in ps.splitter.provisional_input_properties
    assert "y_momentum_isentropic" in ps.splitter.provisional_input_properties
    assert len(ps.splitter.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.splitter.output_properties
    assert "fake_variable" in ps.splitter.output_properties
    assert "x_momentum_isentropic" in ps.splitter.output_properties
    assert "x_velocity" in ps.splitter.output_properties
    assert "x_velocity_at_u_locations" in ps.splitter.output_properties
    assert "y_momentum_isentropic" in ps.splitter.output_properties
    assert "y_velocity_at_v_locations" in ps.splitter.output_properties
    assert len(ps.splitter.output_properties) == 7

    assert (
        "air_isentropic_density" in ps.splitter.provisional_output_properties
    )
    assert "x_momentum_isentropic" in ps.splitter.provisional_output_properties
    assert "x_velocity" in ps.splitter.provisional_output_properties
    assert "y_momentum_isentropic" in ps.splitter.provisional_output_properties
    assert len(ps.splitter.provisional_output_properties) == 4

    # >>> test 2
    ps = ParallelSplittingTestSuite(
        ds,
        tsts1,
        tsts2,
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=False,
    )

    assert "air_isentropic_density" in ps.splitter.input_properties
    assert "x_momentum_isentropic" in ps.splitter.input_properties
    assert "x_velocity" in ps.splitter.input_properties
    assert "x_velocity_at_u_locations" in ps.splitter.input_properties
    assert "y_momentum_isentropic" in ps.splitter.input_properties
    assert "y_velocity_at_v_locations" in ps.splitter.input_properties
    assert len(ps.splitter.input_properties) == 6

    assert "air_isentropic_density" in ps.splitter.provisional_input_properties
    assert "x_momentum_isentropic" in ps.splitter.provisional_input_properties
    assert "x_velocity" in ps.splitter.provisional_input_properties
    assert "y_momentum_isentropic" in ps.splitter.provisional_input_properties
    assert len(ps.splitter.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.splitter.output_properties
    assert "fake_variable" in ps.splitter.output_properties
    assert "x_momentum_isentropic" in ps.splitter.output_properties
    assert "x_velocity" in ps.splitter.output_properties
    assert "x_velocity_at_u_locations" in ps.splitter.output_properties
    assert "y_momentum_isentropic" in ps.splitter.output_properties
    assert "y_velocity_at_v_locations" in ps.splitter.output_properties
    assert len(ps.splitter.output_properties) == 7

    assert (
        "air_isentropic_density" in ps.splitter.provisional_output_properties
    )
    assert "x_momentum_isentropic" in ps.splitter.provisional_output_properties
    assert "x_velocity" in ps.splitter.provisional_output_properties
    assert "y_momentum_isentropic" in ps.splitter.provisional_output_properties
    assert len(ps.splitter.provisional_output_properties) == 4

    # >>> test 3
    ps = ParallelSplittingTestSuite(
        ds,
        tsts1,
        tsts2,
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=True,
    )

    assert "air_isentropic_density" in ps.splitter.input_properties
    assert "x_momentum_isentropic" in ps.splitter.input_properties
    assert "x_velocity" in ps.splitter.input_properties
    assert "x_velocity_at_u_locations" in ps.splitter.input_properties
    assert "y_momentum_isentropic" in ps.splitter.input_properties
    assert "y_velocity_at_v_locations" in ps.splitter.input_properties
    assert len(ps.splitter.input_properties) == 6

    assert "air_isentropic_density" in ps.splitter.provisional_input_properties
    assert "x_momentum_isentropic" in ps.splitter.provisional_input_properties
    assert "x_velocity" in ps.splitter.provisional_input_properties
    assert "y_momentum_isentropic" in ps.splitter.provisional_input_properties
    assert len(ps.splitter.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.splitter.output_properties
    assert "fake_variable" in ps.splitter.output_properties
    assert "x_momentum_isentropic" in ps.splitter.output_properties
    assert "x_velocity" in ps.splitter.output_properties
    assert "x_velocity_at_u_locations" in ps.splitter.output_properties
    assert "y_momentum_isentropic" in ps.splitter.output_properties
    assert "y_velocity_at_v_locations" in ps.splitter.output_properties
    assert len(ps.splitter.output_properties) == 7

    assert (
        "air_isentropic_density" in ps.splitter.provisional_output_properties
    )
    assert "x_momentum_isentropic" in ps.splitter.provisional_output_properties
    assert "x_velocity" in ps.splitter.provisional_output_properties
    assert "y_momentum_isentropic" in ps.splitter.provisional_output_properties
    assert len(ps.splitter.provisional_output_properties) == 4


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("scheme", ("forward_euler", "rk2", "rk3ws"))
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_numerics(data, scheme, backend, dtype):
    ds = DomainSuite(data, backend, dtype, grid_type="numerical")
    tcts1 = FakeTendencyComponent1TestSuite(ds)
    tsts1 = TendencyStepperTestSuite.factory(
        scheme, tcts1, enforce_horizontal_boundary=False
    )
    tcts2 = FakeTendencyComponent2TestSuite(ds)
    tsts2 = TendencyStepperTestSuite.factory(
        scheme, tcts2, enforce_horizontal_boundary=False
    )
    ps = ParallelSplittingTestSuite(
        ds, tsts1, tsts2, execution_policy="serial"
    )

    state = ps.get_state()
    state_prv = ps.get_state()

    ps.run(state, state_prv)
    ps.run(state_prv, state)


if __name__ == "__main__":
    pytest.main([__file__])
