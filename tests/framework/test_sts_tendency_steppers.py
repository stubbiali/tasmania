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
import numpy as np
import pytest
from sympl import units_are_same

from tasmania.python.framework.sts_tendency_steppers import (
    ForwardEuler,
    RungeKutta2,
    RungeKutta3WS,
    RungeKutta3,
)
from tasmania import get_dataarray_dict

try:
    from .conf import backend as conf_backend, halo as conf_halo
    from .utils import st_domain, st_isentropic_state_f, st_one_of, st_timedeltas
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo
    from utils import st_domain, st_isentropic_state_f, st_one_of, st_timedeltas


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def _test_forward_euler(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="prv_state",
    )

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    fe = ForwardEuler(tc1, execution_policy="serial", backend=backend, halo=halo)

    assert "air_isentropic_density" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["x_momentum_isentropic"]["units"],
        "kg m^-1 K^-1 s^-1",
    )
    assert "x_velocity_at_u_locations" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(fe.provisional_input_properties) == 3

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

    out_diagnostics, out_state = fe(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su_prv = prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s_new = s_prv + dt.total_seconds() * s_tnd
    assert np.allclose(s_new, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su_new = su_prv + dt.total_seconds() * su_tnd
    assert np.allclose(su_new, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u_new = u_prv + dt.total_seconds() * u_tnd
    assert np.allclose(u_new, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    assert np.allclose(diagnostics["fake_variable"], out_diagnostics["fake_variable"])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def _test_forward_euler_hb(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="prv_state",
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
        halo=halo,
    )

    out_diagnostics, out_state = fe(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su_prv = prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s_new = s_prv + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s_new,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(s_new, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su_new = su_prv + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su_new,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(su_new, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u_new = u_prv + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u_new,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(u_new, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    assert np.allclose(diagnostics["fake_variable"], out_diagnostics["fake_variable"])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def _test_rk2(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="prv_state",
    )

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    rk2 = RungeKutta2(tc1, execution_policy="serial", backend=backend, halo=halo)

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

    out_diagnostics, out_state = rk2(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su_prv = prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = 0.5 * (s + s_prv + dt.total_seconds() * s_tnd)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = 0.5 * (su + su_prv + dt.total_seconds() * su_tnd)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = 0.5 * (u + u_prv + dt.total_seconds() * u_tnd)

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
    s2 = s_prv + dt.total_seconds() * s_tnd
    assert np.allclose(s2, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su_prv + dt.total_seconds() * su_tnd
    assert np.allclose(su2, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u_prv + dt.total_seconds() * u_tnd
    assert np.allclose(u2, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    assert np.allclose(diagnostics["fake_variable"], out_diagnostics["fake_variable"])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def _test_rk2_hb(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="prv_state",
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
        halo=halo,
    )

    out_diagnostics, out_state = rk2(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su_prv = prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = 0.5 * (s + s_prv + dt.total_seconds() * s_tnd)
    hb.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = 0.5 * (su + su_prv + dt.total_seconds() * su_tnd)
    hb.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = 0.5 * (u + u_prv + dt.total_seconds() * u_tnd)
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
    s2 = s_prv + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )

    assert np.allclose(s2, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = su_prv + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(su2, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = u_prv + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u2,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(u2, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    assert np.allclose(diagnostics["fake_variable"], out_diagnostics["fake_variable"])


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much),
    deadline=None,
)
@given(data=hyp_st.data())
def test_rk3ws(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="prv_state",
    )

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    rk3 = RungeKutta3WS(tc1, execution_policy="serial", backend=backend, halo=halo)

    out_diagnostics, out_state = rk3(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su_prv = prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = 1.0 / 3.0 * (2.0 * s + s_prv + dt.total_seconds() * s_tnd)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = 1.0 / 3.0 * (2.0 * su + su_prv + dt.total_seconds() * su_tnd)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = 1.0 / 3.0 * (2.0 * u + u_prv + dt.total_seconds() * u_tnd)

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
    s2 = 0.5 * (s + s_prv + dt.total_seconds() * s_tnd)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = 0.5 * (su + su_prv + dt.total_seconds() * su_tnd)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = 0.5 * (u + u_prv + dt.total_seconds() * u_tnd)

    raw_state_2 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
        "x_velocity_at_u_locations": u2,
    }
    state_2 = get_dataarray_dict(raw_state_2, cgrid, properties)

    tendencies, _ = tc1(state_2)

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s3 = s_prv + dt.total_seconds() * s_tnd
    assert np.allclose(s3, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su3 = su_prv + dt.total_seconds() * su_tnd
    assert np.allclose(su3, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u3 = u_prv + dt.total_seconds() * u_tnd
    assert np.allclose(u3, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    assert np.allclose(diagnostics["fake_variable"], out_diagnostics["fake_variable"])


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
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            cgrid, moist=False, precipitation=False, backend=backend, halo=halo
        ),
        label="prv_state",
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
        halo=halo,
    )

    out_diagnostics, out_state = rk3(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su_prv = prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_tnd = tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    s1 = 1.0 / 3.0 * (2.0 * s + s_prv + dt.total_seconds() * s_tnd)
    hb.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su1 = 1.0 / 3.0 * (2.0 * su + su_prv + dt.total_seconds() * su_tnd)
    hb.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u1 = 1.0 / 3.0 * (2.0 * u + u_prv + dt.total_seconds() * u_tnd)
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
    s2 = 0.5 * (s + s_prv + dt.total_seconds() * s_tnd)
    hb.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su2 = 0.5 * (su + su_prv + dt.total_seconds() * su_tnd)
    hb.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u2 = 0.5 * (u + u_prv + dt.total_seconds() * u_tnd)
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
    s3 = s_prv + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s3,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(s3, out_state["air_isentropic_density"].values)

    su_tnd = tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    su3 = su_prv + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su3,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(su3, out_state["x_momentum_isentropic"].values)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    u3 = u_prv + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u3,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(u3, out_state["x_velocity_at_u_locations"].values)

    assert "fake_variable" in out_diagnostics
    assert np.allclose(diagnostics["fake_variable"], out_diagnostics["fake_variable"])


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(data=hyp_st.data())
def _test_rk3(data, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid

    state = data.draw(
        st_isentropic_state_f(cgrid, moist=False, precipitation=False), label="state"
    )
    prv_state = data.draw(
        st_isentropic_state_f(cgrid, moist=False, precipitation=False), label="prv_state"
    )

    dt = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
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

    out_diagnostics, out_state = rk3(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su_prv = prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").values

    k0_s = (
        dt.total_seconds()
        * tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    )
    s1 = (1 - a1) * s + a1 * (s_prv + k0_s)

    k0_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su1 = (1 - a1) * su + a1 * (su_prv + k0_su)

    k0_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u1 = (1 - a1) * u + a1 * (u_prv + k0_u)

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
    s2 = (1 - a2) * s + a2 * (s_prv + k1_s) + b21 * (k0_s - k1_s)

    k1_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su2 = (1 - a2) * su + a2 * (su_prv + k1_su) + b21 * (k0_su - k1_su)

    k1_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u2 = (1 - a2) * u + a2 * (u_prv + k1_u) + b21 * (k0_u - k1_u)

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
    s3 = s_prv + g0 * k0_s + g1 * k1_s + g2 * k2_s
    assert np.allclose(s3, out_state["air_isentropic_density"].values)

    k2_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su3 = su_prv + g0 * k0_su + g1 * k1_su + g2 * k2_su
    assert np.allclose(su3, out_state["x_momentum_isentropic"].values)

    k2_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u3 = u_prv + g0 * k0_u + g1 * k1_u + g2 * k2_u
    assert np.allclose(u3, out_state["x_velocity_at_u_locations"].values)


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
    domain = data.draw(st_domain(), label="domain")
    cgrid = domain.numerical_grid
    hb = domain.horizontal_boundary

    assume(hb.type != "dirichlet")

    state = data.draw(
        st_isentropic_state_f(cgrid, moist=False, precipitation=False), label="state"
    )
    prv_state = data.draw(
        st_isentropic_state_f(cgrid, moist=False, precipitation=False), label="prv_state"
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

    rk3 = RungeKutta3(tc1, execution_policy="serial", enforce_horizontal_boundary=True)
    a1, a2 = rk3._alpha1, rk3._alpha2
    b21 = rk3._beta21
    g0, g1, g2 = rk3._gamma0, rk3._gamma1, rk3._gamma2

    out_diagnostics, out_state = rk3(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").values

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su_prv = prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").values

    k0_s = (
        dt.total_seconds()
        * tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").values
    )
    s1 = (1 - a1) * s + a1 * (s_prv + k0_s)
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
    su1 = (1 - a1) * su + a1 * (su_prv + k0_su)
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
    u1 = (1 - a1) * u + a1 * (u_prv + k0_u)
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
    s2 = (1 - a2) * s + a2 * (s_prv + k1_s) + b21 * (k0_s - k1_s)
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
    su2 = (1 - a2) * su + a2 * (su_prv + k1_su) + b21 * (k0_su - k1_su)
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
    u2 = (1 - a2) * u + a2 * (u_prv + k1_u) + b21 * (k0_u - k1_u)
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
    s3 = s_prv + g0 * k0_s + g1 * k1_s + g2 * k2_s
    hb.enforce_field(
        s3,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(s3, out_state["air_isentropic_density"].values)

    k2_su = (
        dt.total_seconds()
        * tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").values
    )
    su3 = su_prv + g0 * k0_su + g1 * k1_su + g2 * k2_su
    hb.enforce_field(
        su3,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(su3, out_state["x_momentum_isentropic"].values)

    k2_u = (
        dt.total_seconds()
        * tendencies["x_velocity_at_u_locations"].to_units("m s^-2").values
    )
    u3 = u_prv + g0 * k0_u + g1 * k1_u + g2 * k2_u
    hb.enforce_field(
        u3,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    assert np.allclose(u3, out_state["x_velocity_at_u_locations"].values)


if __name__ == "__main__":
    pytest.main([__file__])
