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
from sympl import units_are_same

import gt4py as gt

from tasmania.python.framework.tendency_stepper import TendencyStepper

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import st_domain, st_isentropic_state_f, st_one_of, st_timedeltas
from tests.utilities import compare_arrays


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_forward_euler(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    gt_powered_ts = gt_powered and data.draw(hyp_st.booleans(), label="gt_powered_ts")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    if gt_powered:
        gt.storage.prepare_numpy()

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    cgrid = domain.numerical_grid
    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="state",
    )

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    fe = TendencyStepper.factory(
        "forward_euler",
        tc1,
        execution_policy="serial",
        gt_powered=gt_powered_ts,
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in fe.output_properties
    assert units_are_same(
        fe.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in fe.output_properties
    assert units_are_same(
        fe.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in fe.output_properties
    assert units_are_same(
        fe.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(fe.output_properties) == 3

    out_diagnostics, out_state = fe(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s_new = s + dt.total_seconds() * s_tnd
    compare_arrays(s_new, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su_new = su + dt.total_seconds() * su_tnd
    compare_arrays(su_new, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u_new = u + dt.total_seconds() * u_tnd
    compare_arrays(u_new, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values, out_diagnostics["fake_variable"].values
    )

    _, _ = fe(out_state, dt)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_forward_euler_hb(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    gt_powered_ts = gt_powered and data.draw(hyp_st.booleans(), label="gt_powered_ts")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    if gt_powered:
        gt.storage.prepare_numpy()

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    cgrid = domain.numerical_grid
    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="state",
    )
    hb.reference_state = state

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    fe = TendencyStepper.factory(
        "forward_euler",
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        gt_powered=gt_powered_ts,
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in fe.output_properties
    assert units_are_same(
        fe.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in fe.output_properties
    assert units_are_same(
        fe.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in fe.output_properties
    assert units_are_same(
        fe.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(fe.output_properties) == 3

    out_diagnostics, out_state = fe(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s_new = s + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s_new,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(s_new, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su_new = su + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su_new,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(su_new, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u_new = u + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u_new,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(u_new, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values, out_diagnostics["fake_variable"].values
    )


if __name__ == "__main__":
    pytest.main([__file__])
