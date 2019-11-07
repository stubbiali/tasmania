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

from tasmania.python.framework.tendency_steppers import (
    ForwardEuler,
    GTForwardEuler,
    RungeKutta2,
    GTRungeKutta2,
    RungeKutta3WS,
    GTRungeKutta3WS,
    RungeKutta3,
)
from tasmania import get_dataarray_dict

try:
    from .conf import backend as conf_backend, default_origin as conf_dorigin
    from .utils import (
        compare_arrays,
        st_domain,
        st_isentropic_state_f,
        st_one_of,
        st_timedeltas,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, default_origin as conf_dorigin
    from utils import (
        compare_arrays,
        st_domain,
        st_isentropic_state_f,
        st_one_of,
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
def test_forward_euler(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
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

    fe = ForwardEuler(
        tc1, execution_policy="serial", backend=backend, default_origin=default_origin
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
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
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

    fe = ForwardEuler(
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        default_origin=default_origin,
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


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_gt_forward_euler(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(cgrid.nx + 1, cgrid.ny + 1, cgrid.nz + 1),
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

    fe = GTForwardEuler(
        tc1, execution_policy="serial", backend=backend, default_origin=default_origin
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
def test_gt_forward_euler_hb(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(cgrid.nx + 1, cgrid.ny + 1, cgrid.nz + 1),
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

    fe = GTForwardEuler(
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        default_origin=default_origin,
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
    compare_arrays(
        s_new[:-1, :-1, :-1], out_state["air_isentropic_density"].values[:-1, :-1, :-1]
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su_new = su + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su_new,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(
        su_new[:-1, :-1, :-1], out_state["x_momentum_isentropic"].values[:-1, :-1, :-1]
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u_new = u + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u_new,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(
        u_new[:, :-1, :-1], out_state["x_velocity_at_u_locations"].values[:, :-1, :-1]
    )

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values[:-1, :-1, :-1],
        out_diagnostics["fake_variable"].values[:-1, :-1, :-1],
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
def test_rk2(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
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

    rk2 = RungeKutta2(
        tc1, execution_policy="serial", backend=backend, default_origin=default_origin
    )

    assert "air_isentropic_density" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk2.output_properties) == 3

    out_diagnostics, out_state = rk2(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = s + 0.5 * dt.total_seconds() * s_tnd

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = su + 0.5 * dt.total_seconds() * su_tnd

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = u + 0.5 * dt.total_seconds() * u_tnd

    raw_state_1 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "x_velocity_at_u_locations": {"units": "m s^-1"},
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties)

    tendencies, _ = tc1(state_1)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s2 = s + dt.total_seconds() * s_tnd
    compare_arrays(s2, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su + dt.total_seconds() * su_tnd
    compare_arrays(su2, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u + dt.total_seconds() * u_tnd
    compare_arrays(u2, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values, out_diagnostics["fake_variable"].values
    )

    _, _ = rk2(out_state, dt)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_rk2_hb(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
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

    rk2 = RungeKutta2(
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        default_origin=default_origin,
    )

    assert "air_isentropic_density" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk2.output_properties) == 3

    out_diagnostics, out_state = rk2(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = s + 0.5 * dt.total_seconds() * s_tnd
    hb.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = su + 0.5 * dt.total_seconds() * su_tnd
    hb.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = u + 0.5 * dt.total_seconds() * u_tnd
    hb.enforce_field(
        u1,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    raw_state_1 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "x_velocity_at_u_locations": {"units": "m s^-1"},
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties)

    tendencies, _ = tc1(state_1)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s2 = s + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )

    compare_arrays(s2, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(su2, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u2,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(u2, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values, out_diagnostics["fake_variable"].values
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
def test_gt_rk2(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(nx + 1, ny + 1, nz + 1),
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

    rk2 = GTRungeKutta2(
        tc1, execution_policy="serial", backend=backend, default_origin=default_origin
    )

    assert "air_isentropic_density" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk2.output_properties) == 3

    out_diagnostics, out_state = rk2(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = s + 0.5 * dt.total_seconds() * s_tnd

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = su + 0.5 * dt.total_seconds() * su_tnd

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = u + 0.5 * dt.total_seconds() * u_tnd

    raw_state_1 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1", "grid_shape": (nx, ny, nz)},
        "x_momentum_isentropic": {
            "units": "kg m^-1 K^-1 s^-1",
            "grid_shape": (nx, ny, nz),
        },
        "x_velocity_at_u_locations": {"units": "m s^-1", "grid_shape": (nx + 1, ny, nz)},
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties, set_coordinates=False)

    tendencies, _ = tc1(state_1)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s2 = s + dt.total_seconds() * s_tnd
    compare_arrays(s2, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su + dt.total_seconds() * su_tnd
    compare_arrays(su2, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u + dt.total_seconds() * u_tnd
    compare_arrays(u2, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values, out_diagnostics["fake_variable"].values
    )

    _, _ = rk2(out_state, dt)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_gt_rk2_hb(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(nx + 1, ny + 1, nz + 1),
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

    rk2 = GTRungeKutta2(
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        default_origin=default_origin,
    )

    assert "air_isentropic_density" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk2.output_properties) == 3

    out_diagnostics, out_state = rk2(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = s + 0.5 * dt.total_seconds() * s_tnd
    hb.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = su + 0.5 * dt.total_seconds() * su_tnd
    hb.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = u + 0.5 * dt.total_seconds() * u_tnd
    hb.enforce_field(
        u1,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    raw_state_1 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1", "grid_shape": (nx, ny, nz)},
        "x_momentum_isentropic": {
            "units": "kg m^-1 K^-1 s^-1",
            "grid_shape": (nx, ny, nz),
        },
        "x_velocity_at_u_locations": {"units": "m s^-1", "grid_shape": (nx + 1, ny, nz)},
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties, set_coordinates=False)

    tendencies, _ = tc1(state_1)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s2 = s + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )

    compare_arrays(
        s2[:-1, :-1, :-1], out_state["air_isentropic_density"].values[:-1, :-1, :-1]
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(
        su2[:-1, :-1, :-1], out_state["x_momentum_isentropic"].values[:-1, :-1, :-1]
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u2,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(
        u2[:, :-1, :-1], out_state["x_velocity_at_u_locations"].values[:, :-1, :-1]
    )

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values[:-1, :-1, :-1],
        out_diagnostics["fake_variable"].values[:-1, :-1, :-1],
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
def test_rk3ws(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
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

    rk3 = RungeKutta3WS(
        tc1, execution_policy="serial", backend=backend, default_origin=default_origin
    )

    assert "air_isentropic_density" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk3.output_properties) == 3

    out_diagnostics, out_state = rk3(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = s + 1.0 / 3.0 * dt.total_seconds() * s_tnd

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = su + 1.0 / 3.0 * dt.total_seconds() * su_tnd

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = u + 1.0 / 3.0 * dt.total_seconds() * u_tnd

    raw_state_1 = {
        "time": state["time"] + dt / 3.0,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "x_velocity_at_u_locations": {"units": "m s^-1"},
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties)

    tendencies, _ = tc1(state_1)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s2 = s + 0.5 * dt.total_seconds() * s_tnd

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su + 0.5 * dt.total_seconds() * su_tnd

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u + 0.5 * dt.total_seconds() * u_tnd

    raw_state_2 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
        "x_velocity_at_u_locations": u2,
    }
    state_2 = get_dataarray_dict(raw_state_2, cgrid, properties)

    tendencies, _ = tc1(state_2)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s3 = s + dt.total_seconds() * s_tnd
    compare_arrays(s3, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su3 = su + dt.total_seconds() * su_tnd
    compare_arrays(su3, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u3 = u + dt.total_seconds() * u_tnd
    compare_arrays(u3, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values, out_diagnostics["fake_variable"].values
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
def test_rk3ws_hb(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
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

    rk3 = RungeKutta3WS(
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        default_origin=default_origin,
    )

    assert "air_isentropic_density" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk3.output_properties) == 3

    out_diagnostics, out_state = rk3(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = s + 1.0 / 3.0 * dt.total_seconds() * s_tnd
    hb.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = su + 1.0 / 3.0 * dt.total_seconds() * su_tnd
    hb.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = u + 1.0 / 3.0 * dt.total_seconds() * u_tnd
    hb.enforce_field(
        u1,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    raw_state_1 = {
        "time": state["time"] + dt / 3.0,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "x_velocity_at_u_locations": {"units": "m s^-1"},
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties)

    tendencies, _ = tc1(state_1)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s2 = s + 0.5 * dt.total_seconds() * s_tnd
    hb.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su + 0.5 * dt.total_seconds() * su_tnd
    hb.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u + 0.5 * dt.total_seconds() * u_tnd
    hb.enforce_field(
        u2,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    raw_state_2 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
        "x_velocity_at_u_locations": u2,
    }
    state_2 = get_dataarray_dict(raw_state_2, cgrid, properties)

    tendencies, _ = tc1(state_2)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s3 = s + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s3,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(s3, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su3 = su + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su3,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(su3, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u3 = u + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u3,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(u3, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values, out_diagnostics["fake_variable"].values
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
def test_gt_rk3ws(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(nx + 1, ny + 1, nz + 1),
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

    rk3 = GTRungeKutta3WS(
        tc1, execution_policy="serial", backend=backend, default_origin=default_origin
    )

    assert "air_isentropic_density" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk3.output_properties) == 3

    out_diagnostics, out_state = rk3(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = s + 1.0 / 3.0 * dt.total_seconds() * s_tnd

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = su + 1.0 / 3.0 * dt.total_seconds() * su_tnd

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = u + 1.0 / 3.0 * dt.total_seconds() * u_tnd

    raw_state_1 = {
        "time": state["time"] + dt / 3.0,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1", "grid_shape": (nx, ny, nz)},
        "x_momentum_isentropic": {
            "units": "kg m^-1 K^-1 s^-1",
            "grid_shape": (nx, ny, nz),
        },
        "x_velocity_at_u_locations": {"units": "m s^-1", "grid_shape": (nx + 1, ny, nz)},
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties, set_coordinates=False)

    tendencies, _ = tc1(state_1)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s2 = s + 0.5 * dt.total_seconds() * s_tnd

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su + 0.5 * dt.total_seconds() * su_tnd

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u + 0.5 * dt.total_seconds() * u_tnd

    raw_state_2 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
        "x_velocity_at_u_locations": u2,
    }
    state_2 = get_dataarray_dict(raw_state_2, cgrid, properties, set_coordinates=False)

    tendencies, _ = tc1(state_2)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s3 = s + dt.total_seconds() * s_tnd
    compare_arrays(s3, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su3 = su + dt.total_seconds() * su_tnd
    compare_arrays(su3, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u3 = u + dt.total_seconds() * u_tnd
    compare_arrays(u3, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values, out_diagnostics["fake_variable"].values
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
def test_gt_rk3ws_hb(data, make_fake_tendency_component_1):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(nx + 1, ny + 1, nz + 1),
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

    rk3 = GTRungeKutta3WS(
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        default_origin=default_origin,
    )

    assert "air_isentropic_density" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk3.output_properties) == 3

    out_diagnostics, out_state = rk3(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = s + 1.0 / 3.0 * dt.total_seconds() * s_tnd
    hb.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = su + 1.0 / 3.0 * dt.total_seconds() * su_tnd
    hb.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = u + 1.0 / 3.0 * dt.total_seconds() * u_tnd
    hb.enforce_field(
        u1,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    raw_state_1 = {
        "time": state["time"] + dt / 3.0,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1", "grid_shape": (nx, ny, nz)},
        "x_momentum_isentropic": {
            "units": "kg m^-1 K^-1 s^-1",
            "grid_shape": (nx, ny, nz),
        },
        "x_velocity_at_u_locations": {"units": "m s^-1", "grid_shape": (nx + 1, ny, nz)},
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties, set_coordinates=False)

    tendencies, _ = tc1(state_1)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s2 = s + 0.5 * dt.total_seconds() * s_tnd
    hb.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su + 0.5 * dt.total_seconds() * su_tnd
    hb.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u + 0.5 * dt.total_seconds() * u_tnd
    hb.enforce_field(
        u2,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    raw_state_2 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
        "x_velocity_at_u_locations": u2,
    }
    state_2 = get_dataarray_dict(raw_state_2, cgrid, properties, set_coordinates=False)

    tendencies, _ = tc1(state_2)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s3 = s + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s3,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(
        s3[:-1, :-1, :-1], out_state["air_isentropic_density"].values[:-1, :-1, :-1]
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su3 = su + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su3,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(
        su3[:-1, :-1, :-1], out_state["x_momentum_isentropic"].values[:-1, :-1, :-1]
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u3 = u + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u3,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(
        u3[:, :-1, :-1], out_state["x_velocity_at_u_locations"].values[:, :-1, :-1]
    )

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].values[:-1, :-1, :-1],
        out_diagnostics["fake_variable"].values[:-1, :-1, :-1],
    )


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(data=hyp_st.data())
def _test_rk3(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")
    cgrid = domain.numerical_grid

    state = data.draw(
        utils.st_isentropic_state(cgrid, moist=False, precipitation=False), label="state"
    )

    dt = data.draw(
        utils.st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    rk3 = RungeKutta3(tc1, execution_policy="serial")
    a1, a2 = rk3._alpha1, rk3._alpha2
    b21 = rk3._beta21
    g0, g1, g2 = rk3._gamma0, rk3._gamma1, rk3._gamma2

    assert "air_isentropic_density" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk3.output_properties) == 3

    out_diagnostics, out_state = rk3(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    k0_s = (
        dt.total_seconds()
        * tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    )
    s1 = s + a1 * k0_s

    k0_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su1 = su + a1 * k0_su

    k0_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u1 = u + a1 * k0_u

    raw_state_1 = {
        "time": state["time"] + a1 * dt,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    units = {
        "air_isentropic_density": "kg m^-2 K^-1",
        "x_momentum_isentropic": "kg m^-1 K^-1 s^-1",
        "x_velocity_at_u_locations": "m s^-1",
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, units)

    tendencies, _ = tc1(state_1)

    k1_s = (
        dt.total_seconds()
        * tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    )
    s2 = s + b21 * k0_s + (a2 - b21) * k1_s

    k1_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su2 = su + b21 * k0_su + (a2 - b21) * k1_su

    k1_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u2 = u + b21 * k0_u + (a2 - b21) * k1_u

    raw_state_2 = {
        "time": state["time"] + a2 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
        "x_velocity_at_u_locations": u2,
    }
    state_2 = get_dataarray_dict(raw_state_2, cgrid, units)

    tendencies, _ = tc1(state_2)

    k2_s = (
        dt.total_seconds()
        * tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    )
    s3 = s + g0 * k0_s + g1 * k1_s + g2 * k2_s
    compare_arrays(s3, out_state["air_isentropic_density"].values)

    k2_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su3 = su + g0 * k0_su + g1 * k1_su + g2 * k2_su
    compare_arrays(su3, out_state["x_momentum_isentropic"].values)

    k2_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u3 = u + g0 * k0_u + g1 * k1_u + g2 * k2_u
    compare_arrays(u3, out_state["x_velocity_at_u_locations"].values)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def _test_rk3_hb(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary

    assume(hb.type != "dirichlet")

    state = data.draw(
        utils.st_isentropic_state(cgrid, moist=False, precipitation=False), label="state"
    )
    hb.reference_state = state

    dt = data.draw(
        utils.st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    rk3 = RungeKutta3(tc1, execution_policy="serial", enforce_horizontal_boundary=True)
    a1, a2 = rk3._alpha1, rk3._alpha2
    b21 = rk3._beta21
    g0, g1, g2 = rk3._gamma0, rk3._gamma1, rk3._gamma2

    assert "air_isentropic_density" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_momentum_isentropic"]["units"], "kg m^-1 K^-1 s^-1"
    )
    assert "x_velocity_at_u_locations" in rk3.output_properties
    assert units_are_same(
        rk3.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(rk3.output_properties) == 3

    out_diagnostics, out_state = rk3(state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    k0_s = (
        dt.total_seconds()
        * tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    )
    s1 = s + a1 * k0_s
    hb.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + a1 * dt,
        grid=cgrid,
    )

    k0_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su1 = su + a1 * k0_su
    hb.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + a1 * dt,
        grid=cgrid,
    )

    k0_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u1 = u + a1 * k0_u
    hb.enforce_field(
        u1,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + a1 * dt,
        grid=cgrid,
    )

    raw_state_1 = {
        "time": state["time"] + a1 * dt,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    units = {
        "air_isentropic_density": "kg m^-2 K^-1",
        "x_momentum_isentropic": "kg m^-1 K^-1 s^-1",
        "x_velocity_at_u_locations": "m s^-1",
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, units)

    tendencies, _ = tc1(state_1)

    k1_s = (
        dt.total_seconds()
        * tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    )
    s2 = s + b21 * k0_s + (a2 - b21) * k1_s
    hb.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + a2 * dt,
        grid=cgrid,
    )

    k1_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su2 = su + b21 * k0_su + (a2 - b21) * k1_su
    hb.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + a2 * dt,
        grid=cgrid,
    )

    k1_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u2 = u + b21 * k0_u + (a2 - b21) * k1_u
    hb.enforce_field(
        u2,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + a2 * dt,
        grid=cgrid,
    )

    raw_state_2 = {
        "time": state["time"] + a2 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
        "x_velocity_at_u_locations": u2,
    }
    state_2 = get_dataarray_dict(raw_state_2, cgrid, units)

    tendencies, _ = tc1(state_2)

    k2_s = (
        dt.total_seconds()
        * tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    )
    s3 = s + g0 * k0_s + g1 * k1_s + g2 * k2_s
    hb.enforce_field(
        s3,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(s3, out_state["air_isentropic_density"].values)

    k2_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su3 = su + g0 * k0_su + g1 * k1_su + g2 * k2_su
    hb.enforce_field(
        su3,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(su3, out_state["x_momentum_isentropic"].values)

    k2_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u3 = u + g0 * k0_u + g1 * k1_u + g2 * k2_u
    hb.enforce_field(
        u3,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(u3, out_state["x_velocity_at_u_locations"].values)


if __name__ == "__main__":
    pytest.main([__file__])
