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
from hypothesis import (
    given,
    strategies as hyp_st,
)
import pytest

from tasmania.python.framework.fakes import FakeTendencyComponent
from tasmania.python.framework.tendency_stepper import TendencyStepper
from tasmania.python.framework.subclasses.tendency_steppers import (
    ForwardEuler,
)
from tasmania.python.framework.subclasses.tendency_steppers import RK2
from tasmania.python.framework.subclasses.tendency_steppers import RK3WS

from tests import conf
from tests.strategies import st_domain, st_one_of
from tests.suites.concurrent_coupling import ConcurrentCouplingTestSuite
from tests.suites.core_components import FakeTendencyComponent1TestSuite
from tests.suites.domain import DomainSuite
from tests.suites.steppers import TendencyStepperTestSuite
from tests.utilities import hyp_settings


def test_registry():
    registry = TendencyStepper.registry[
        "tasmania.python.framework.tendency_stepper.TendencyStepper"
    ]

    # forward euler
    assert "forward_euler" in registry
    assert registry["forward_euler"] == ForwardEuler

    # rk2
    assert "rk2" in registry
    assert registry["rk2"] == RK2

    # rk3ws
    assert "rk3ws" in registry
    assert registry["rk3ws"] == RK3WS


@hyp_settings
@given(data=hyp_st.data())
def test_factory(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )

    # ========================================
    # test bed
    # ========================================
    ftc = FakeTendencyComponent(domain, grid_type)

    # forward euler
    obj = TendencyStepper.factory("forward_euler", ftc)
    assert isinstance(obj, ForwardEuler)

    # rk2
    obj = TendencyStepper.factory("rk2", ftc)
    assert isinstance(obj, RK2)

    # rk3ws
    obj = TendencyStepper.factory("rk3ws", ftc)
    assert isinstance(obj, RK3WS)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("scheme", ("forward_euler", "rk2", "rk3ws"))
@pytest.mark.parametrize("enforce_hb", (False, True))
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_numerics(data, scheme, enforce_hb, backend, dtype):
    ds = DomainSuite(data, backend, dtype, grid_type="numerical")
    tcts = FakeTendencyComponent1TestSuite(ds)
    tsts = TendencyStepperTestSuite.factory(
        scheme, tcts, enforce_horizontal_boundary=enforce_hb
    )
    tsts.run()


if __name__ == "__main__":
    pytest.main([__file__])
