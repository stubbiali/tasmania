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
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

import gt4py as gt

from tasmania.python.framework.fakes import FakeTendencyComponent
from tasmania.python.framework.sts_tendency_steppers import STSTendencyStepper
from tasmania.python.framework.tendency_steppers import TendencyStepper

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.utilities import (
    compare_arrays,
    compare_dataarrays,
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_raw_field,
    st_timedeltas,
)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_fake_tendency_component(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            gt_powered=gt_powered,
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


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_fake_tendency_component_tendency_stepper(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
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


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_fake_tendency_component_sts_tendency_stepper(
    data, make_fake_tendency_component_1
):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
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
