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
    assume,
    given,
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

import gt4py as gt

from tasmania.python.framework.fakes import FakeTendencyComponent
from tasmania.python.framework.sts_tendency_steppers import STSTendencyStepper, register
from tasmania.python.framework.sts_tendency_steppers_rk import ForwardEuler, RK2, RK3WS
from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
)
from tasmania.python.isentropic.physics.sts_tendency_steppers import (
    IsentropicVerticalAdvection,
)

from tests.utilities import st_domain, st_one_of


def test_register():
    # forward euler
    assert "forward_euler" in register
    assert register["forward_euler"] == ForwardEuler

    # rk2
    assert "rk2" in register
    assert register["rk2"] == RK2

    # rk3ws
    assert "rk3ws" in register
    assert register["rk3ws"] == RK3WS

    # isentropic_prognostic vertical advection
    assert "isentropic_vertical_advection" in register
    assert register["isentropic_vertical_advection"] == IsentropicVerticalAdvection


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_factory(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")

    # ========================================
    # test bed
    # ========================================
    ftc = FakeTendencyComponent(domain, grid_type)

    # forward euler
    obj = STSTendencyStepper.factory("forward_euler", ftc)
    assert isinstance(obj, ForwardEuler)

    # rk2
    obj = STSTendencyStepper.factory("rk2", ftc)
    assert isinstance(obj, RK2)

    # rk3ws
    obj = STSTendencyStepper.factory("rk3ws", ftc)
    assert isinstance(obj, RK3WS)

    # isentropic_prognostic vertical advection
    arg = IsentropicImplicitVerticalAdvectionDiagnostic(domain)
    obj = STSTendencyStepper.factory("isentropic_vertical_advection", arg)
    assert isinstance(obj, IsentropicVerticalAdvection)


if __name__ == "__main__":
    pytest.main([__file__])
