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
from hypothesis import (
    assume,
    given,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

import gt4py as gt

from tasmania.python.framework.fakes import FakeTendencyComponent
from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper
from tasmania.python.framework.tendency_stepper import TendencyStepper

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import (
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_timedeltas,
)
from tests.utilities import hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_fake_tendency_component(
    data, backend, dtype, make_fake_tendency_component_1
):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(backend=backend, dtype=dtype),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
        ),
        label="state",
    )

    # ========================================
    # test bed
    # ========================================
    ftc = FakeTendencyComponent(domain, grid_type)

    tendencies, diagnostics = ftc(state)
    assert len(tendencies) == 0
    assert len(diagnostics) == 0

    tendencies, diagnostics = ftc({"time": state["time"]})
    assert len(tendencies) == 0
    assert len(diagnostics) == 0


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_fake_tendency_component_tendency_stepper(
    data, backend, dtype, make_fake_tendency_component_1
):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(backend=backend, dtype=dtype),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=1), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    ftc = FakeTendencyComponent(domain, grid_type)

    # forward euler
    ts = TendencyStepper.factory("forward_euler", ftc)
    assert len(ts.input_properties) == 0
    assert len(ts.diagnostic_properties) == 0
    assert len(ts.output_properties) == 0
    diagnostics, out_state = ts(state, timestep)
    assert "time" in diagnostics
    assert len(diagnostics) == 1
    assert "time" in out_state
    assert len(out_state) == 1

    # rk2
    ts = TendencyStepper.factory("rk2", ftc)
    assert len(ts.input_properties) == 0
    assert len(ts.diagnostic_properties) == 0
    assert len(ts.output_properties) == 0
    diagnostics, out_state = ts(state, timestep)
    assert "time" in diagnostics
    assert len(diagnostics) == 1
    assert "time" in out_state
    assert len(out_state) == 1

    # rk3ws
    ts = TendencyStepper.factory("rk3ws", ftc)
    assert len(ts.input_properties) == 0
    assert len(ts.diagnostic_properties) == 0
    assert len(ts.output_properties) == 0
    diagnostics, out_state = ts(state, timestep)
    assert "time" in diagnostics
    assert len(diagnostics) == 1
    assert "time" in out_state
    assert len(out_state) == 1

    # implicit
    ts = TendencyStepper.factory("implicit", ftc)
    assert len(ts.input_properties) == 0
    assert len(ts.diagnostic_properties) == 0
    assert len(ts.output_properties) == 0
    diagnostics, out_state = ts(state, timestep)
    assert "time" in diagnostics
    assert len(diagnostics) == 1
    assert "time" in out_state
    assert len(out_state) == 1


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_fake_tendency_component_sts_tendency_stepper(
    data, backend, dtype, make_fake_tendency_component_1
):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(backend=backend, dtype=dtype),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=1), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    ftc = FakeTendencyComponent(domain, grid_type)

    # forward euler
    ts = STSTendencyStepper.factory("forward_euler", ftc)
    assert len(ts.input_properties) == 0
    assert len(ts.diagnostic_properties) == 0
    assert len(ts.output_properties) == 0
    diagnostics, out_state = ts(state, state, timestep)
    assert "time" in diagnostics
    assert len(diagnostics) == 1
    assert "time" in out_state
    assert len(out_state) == 1

    # rk2
    ts = STSTendencyStepper.factory("rk2", ftc)
    assert len(ts.input_properties) == 0
    assert len(ts.diagnostic_properties) == 0
    assert len(ts.output_properties) == 0
    diagnostics, out_state = ts(state, state, timestep)
    assert "time" in diagnostics
    assert len(diagnostics) == 1
    assert "time" in out_state
    assert len(out_state) == 1

    # rk3ws
    ts = STSTendencyStepper.factory("rk3ws", ftc)
    assert len(ts.input_properties) == 0
    assert len(ts.diagnostic_properties) == 0
    assert len(ts.output_properties) == 0
    diagnostics, out_state = ts(state, state, timestep)
    assert "time" in diagnostics
    assert len(diagnostics) == 1
    assert "time" in out_state
    assert len(out_state) == 1


if __name__ == "__main__":
    pytest.main([__file__])
