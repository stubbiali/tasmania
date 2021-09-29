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
from tests.suites.core_components import (
    FakeTendencyComponent1TestSuite,
    FakeTendencyComponent2TestSuite,
)
from tests.suites.domain import DomainSuite
from tests.suites.sequential_update_splitting import (
    SequentialUpdateSplittingTestSuite,
)
from tests.suites.steppers import TendencyStepperTestSuite
from tests.utilities import hyp_settings


@hyp_settings
@given(data=hyp_st.data())
def test_properties(data):
    ds = DomainSuite(data, "numpy", float, grid_type="numerical")
    tcts1 = FakeTendencyComponent1TestSuite(ds)
    tsts1 = TendencyStepperTestSuite.factory("forward_euler", tcts1)
    tcts2 = FakeTendencyComponent2TestSuite(ds)
    tsts2 = TendencyStepperTestSuite.factory("forward_euler", tcts2)

    # >>> test 1
    sus = SequentialUpdateSplittingTestSuite(tsts2, tsts1)

    assert "air_isentropic_density" in sus.input_properties
    assert "fake_variable" in sus.input_properties
    assert "x_momentum_isentropic" in sus.input_properties
    assert "x_velocity" in sus.input_properties
    assert "x_velocity_at_u_locations" in sus.input_properties
    assert "y_momentum_isentropic" in sus.input_properties
    assert "y_velocity_at_v_locations" in sus.input_properties
    assert len(sus.input_properties) == 7

    assert "air_isentropic_density" in sus.output_properties
    assert "fake_variable" in sus.output_properties
    assert "x_momentum_isentropic" in sus.output_properties
    assert "x_velocity" in sus.output_properties
    assert "x_velocity_at_u_locations" in sus.output_properties
    assert "y_momentum_isentropic" in sus.output_properties
    assert "y_velocity_at_v_locations" in sus.output_properties
    assert len(sus.output_properties) == 7

    # >>> test 2
    sus = SequentialUpdateSplittingTestSuite(tsts1, tsts2)

    assert "air_isentropic_density" in sus.input_properties
    assert "x_momentum_isentropic" in sus.input_properties
    assert "x_velocity" in sus.input_properties
    assert "x_velocity_at_u_locations" in sus.input_properties
    assert "y_momentum_isentropic" in sus.input_properties
    assert "y_velocity_at_v_locations" in sus.input_properties
    assert len(sus.input_properties) == 6

    assert "air_isentropic_density" in sus.output_properties
    assert "fake_variable" in sus.output_properties
    assert "x_momentum_isentropic" in sus.output_properties
    assert "x_velocity" in sus.output_properties
    assert "x_velocity_at_u_locations" in sus.output_properties
    assert "y_momentum_isentropic" in sus.output_properties
    assert "y_velocity_at_v_locations" in sus.output_properties
    assert len(sus.output_properties) == 7


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("scheme", ("forward_euler", "rk2", "rk3ws"))
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_numerics(data, scheme, backend, dtype):
    ds = DomainSuite(data, backend, dtype, grid_type="numerical")
    tcts1 = FakeTendencyComponent1TestSuite(ds)
    tsts1 = TendencyStepperTestSuite.factory(scheme, tcts1)
    tcts2 = FakeTendencyComponent2TestSuite(ds)
    tsts2 = TendencyStepperTestSuite.factory(scheme, tcts2)
    sus = SequentialUpdateSplittingTestSuite(tsts1, tsts2)

    state1 = sus.get_state()
    state2 = sus.get_state()

    sus.run(state1)
    sus.run(state2)


if __name__ == "__main__":
    pytest.main([__file__])
