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
    given,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest
from sympl import units_are_same

from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper

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
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, backend, dtype, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    backend_ts = (
        backend
        if data.draw(hyp_st.booleans(), label="same_backend")
        else "numpy"
    )
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    domain = data.draw(st_domain(backend=backend, dtype=dtype), label="domain")
    cgrid = domain.numerical_grid
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnz")
    storage_shape = (cgrid.nx + dnx, cgrid.ny + dny, cgrid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="prv_state",
    )

    dt = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    fe = STSTendencyStepper.factory(
        "forward_euler",
        tc1,
        execution_policy="serial",
        backend=backend_ts,
        dtype=dtype,
    )

    assert "air_isentropic_density" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["air_isentropic_density"]["units"],
        "kg m^-2 K^-1",
    )
    assert "x_momentum_isentropic" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["x_momentum_isentropic"]["units"],
        "kg m^-1 K^-1 s^-1",
    )
    assert "x_velocity_at_u_locations" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["x_velocity_at_u_locations"]["units"],
        "m s^-1",
    )
    assert len(fe.provisional_input_properties) == 3

    assert "air_isentropic_density" in fe.output_properties
    assert units_are_same(
        fe.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in fe.output_properties
    assert units_are_same(
        fe.output_properties["x_momentum_isentropic"]["units"],
        "kg m^-1 K^-1 s^-1",
    )
    assert "x_velocity_at_u_locations" in fe.output_properties
    assert units_are_same(
        fe.output_properties["x_velocity_at_u_locations"]["units"], "m s^-1"
    )
    assert len(fe.output_properties) == 3

    out_diagnostics, out_state = fe(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    su_prv = (
        prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").data

    s_tnd = (
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s_new = s_prv + dt.total_seconds() * s_tnd
    compare_arrays(s_new, out_state["air_isentropic_density"].data)

    su_tnd = (
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su_new = su_prv + dt.total_seconds() * su_tnd
    compare_arrays(su_new, out_state["x_momentum_isentropic"].data)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data
    u_new = u_prv + dt.total_seconds() * u_tnd
    compare_arrays(u_new, out_state["x_velocity_at_u_locations"].data)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].data,
        out_diagnostics["fake_variable"].data,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_hb(data, backend, dtype, make_fake_tendency_component_1):
    # ========================================
    # random data generation
    # ========================================
    backend_ts = (
        backend
        if data.draw(hyp_st.booleans(), label="same_backend")
        else "numpy"
    )
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    domain = data.draw(
        st_domain(backend=backend, dtype=dtype),
        label="domain",
    )
    cgrid = domain.numerical_grid
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnz")
    storage_shape = (cgrid.nx + dnx, cgrid.ny + dny, cgrid.nz + dnz)
    hb = domain.horizontal_boundary

    state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            cgrid,
            moist=False,
            precipitation=False,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="prv_state",
    )
    hb.reference_state = state

    dt = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    tc1 = make_fake_tendency_component_1(domain, "numerical")

    fe = STSTendencyStepper.factory(
        "forward_euler",
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend_ts,
        dtype=dtype,
    )

    out_diagnostics, out_state = fe(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    su_prv = (
        prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").data

    s_tnd = (
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s_new = s_prv + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s_new,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(s_new, out_state["air_isentropic_density"].data)

    su_tnd = (
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su_new = su_prv + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su_new,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(su_new, out_state["x_momentum_isentropic"].data)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data
    u_new = u_prv + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u_new,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(u_new, out_state["x_velocity_at_u_locations"].data)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].data,
        out_diagnostics["fake_variable"].data,
    )


if __name__ == "__main__":
    pytest.main([__file__])
