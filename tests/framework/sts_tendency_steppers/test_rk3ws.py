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

from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper
from tasmania.python.utils.storage_utils import get_dataarray_dict

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
    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
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
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype)

    tc1 = make_fake_tendency_component_1(domain, "numerical")

    rk3 = STSTendencyStepper.factory(
        "rk3ws",
        tc1,
        execution_policy="serial",
        backend=backend_ts,
        backend_options=bo,
        storage_options=so,
    )

    out_diagnostics, out_state = rk3(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").data

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    su_prv = (
        prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").data

    s_tnd = (
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s1 = 1.0 / 3.0 * (2.0 * s + s_prv + dt.total_seconds() * s_tnd)

    su_tnd = (
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su1 = 1.0 / 3.0 * (2.0 * su + su_prv + dt.total_seconds() * su_tnd)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data
    u1 = 1.0 / 3.0 * (2.0 * u + u_prv + dt.total_seconds() * u_tnd)

    raw_state_1 = {
        "time": state["time"] + dt / 3.0,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
        "x_velocity_at_u_locations": u1,
    }
    properties = {
        "air_isentropic_density": {
            "units": "kg m^-2 K^-1",
            "grid_shape": (nx, ny, nz),
            "set_coordinates": False,
        },
        "x_momentum_isentropic": {
            "units": "kg m^-1 K^-1 s^-1",
            "grid_shape": (nx, ny, nz),
            "set_coordinates": False,
        },
        "x_velocity_at_u_locations": {
            "units": "m s^-1",
            "grid_shape": (nx + 1, ny, nz),
            "set_coordinates": False,
        },
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties)

    tendencies, _ = tc1(state_1)

    s_tnd = (
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s2 = 0.5 * (s + s_prv + dt.total_seconds() * s_tnd)

    su_tnd = (
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su2 = 0.5 * (su + su_prv + dt.total_seconds() * su_tnd)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data
    u2 = 0.5 * (u + u_prv + dt.total_seconds() * u_tnd)

    raw_state_2 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
        "x_velocity_at_u_locations": u2,
    }
    state_2 = get_dataarray_dict(raw_state_2, cgrid, properties)

    tendencies, _ = tc1(state_2)

    s_tnd = (
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s3 = s_prv + dt.total_seconds() * s_tnd
    compare_arrays(s3, out_state["air_isentropic_density"].data)

    su_tnd = (
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su3 = su_prv + dt.total_seconds() * su_tnd
    compare_arrays(su3, out_state["x_momentum_isentropic"].data)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data
    u3 = u_prv + dt.total_seconds() * u_tnd
    compare_arrays(u3, out_state["x_velocity_at_u_locations"].data)

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

    domain = data.draw(st_domain(backend=backend, dtype=dtype), label="domain")
    cgrid = domain.numerical_grid
    nx, ny, nz = cgrid.nx, cgrid.ny, cgrid.nz
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
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype)

    tc1 = make_fake_tendency_component_1(domain, "numerical")

    rk3 = STSTendencyStepper.factory(
        "rk3ws",
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend_ts,
        backend_options=bo,
        storage_options=so,
    )

    out_diagnostics, out_state = rk3(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    u = state["x_velocity_at_u_locations"].to_units("m s^-1").data

    s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    su_prv = (
        prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    u_prv = prv_state["x_velocity_at_u_locations"].to_units("m s^-1").data

    s_tnd = (
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s1 = 1.0 / 3.0 * (2.0 * s + s_prv + dt.total_seconds() * s_tnd)
    hb.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    su_tnd = (
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su1 = 1.0 / 3.0 * (2.0 * su + su_prv + dt.total_seconds() * su_tnd)
    hb.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt / 3.0,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data
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
        "air_isentropic_density": {
            "units": "kg m^-2 K^-1",
            "grid_shape": (nx, ny, nz),
            "set_coordinates": False,
        },
        "x_momentum_isentropic": {
            "units": "kg m^-1 K^-1 s^-1",
            "grid_shape": (nx, ny, nz),
            "set_coordinates": False,
        },
        "x_velocity_at_u_locations": {
            "units": "m s^-1",
            "grid_shape": (nx + 1, ny, nz),
            "set_coordinates": False,
        },
    }
    state_1 = get_dataarray_dict(raw_state_1, cgrid, properties)

    tendencies, _ = tc1(state_1)

    s_tnd = (
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s2 = 0.5 * (s + s_prv + dt.total_seconds() * s_tnd)
    hb.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    su_tnd = (
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su2 = 0.5 * (su + su_prv + dt.total_seconds() * su_tnd)
    hb.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
        grid=cgrid,
    )

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data
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

    s_tnd = (
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s3 = s_prv + dt.total_seconds() * s_tnd
    hb.enforce_field(
        s3,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(s3, out_state["air_isentropic_density"].data)

    su_tnd = (
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su3 = su_prv + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su3,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(su3, out_state["x_momentum_isentropic"].data)

    u_tnd = tendencies["x_velocity_at_u_locations"].to_units("m s^-2").data
    u3 = u_prv + dt.total_seconds() * u_tnd
    hb.enforce_field(
        u3,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + dt,
        grid=cgrid,
    )
    compare_arrays(u3, out_state["x_velocity_at_u_locations"].data)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].data,
        out_diagnostics["fake_variable"].data,
    )


if __name__ == "__main__":
    pytest.main([__file__])
