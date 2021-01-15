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
from datetime import datetime, timedelta
from hypothesis import (
    assume,
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.burgers.dynamics.dycore import BurgersDynamicalCore
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.utils.backend import is_gt

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.burgers.test_burgers_advection import (
    first_order_advection,
    third_order_advection,
    fifth_order_advection,
)
from tests.strategies import (
    st_burgers_state,
    st_domain,
    st_one_of,
    st_timedeltas,
)
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_forward_euler(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 40),
            yaxis_length=(1, 40),
            zaxis_length=(1, 1),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    assume(domain.horizontal_boundary.type != "relaxed" or not is_gt(backend))
    grid = domain.numerical_grid

    state = data.draw(
        st_burgers_state(
            grid,
            time=datetime(year=1992, month=2, day=20),
            backend=backend,
            default_origin=default_origin,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(seconds=120)
        ),
        label="timestep",
    )

    # ========================================
    # test
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    dycore = BurgersDynamicalCore(
        domain,
        intermediate_tendency_component=None,
        time_integration_scheme="forward_euler",
        flux_scheme="first_order",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    domain.horizontal_boundary.reference_state = state

    new_state = dycore(state, {}, timestep)

    assert "time" in new_state
    assert "x_velocity" in new_state
    assert "y_velocity" in new_state
    assert len(new_state) == 3

    assert new_state["time"] == state["time"] + timestep

    dt = timestep.total_seconds()
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    u0 = state["x_velocity"].to_units("m s^-1").data
    v0 = state["y_velocity"].to_units("m s^-1").data

    adv_u_x, adv_u_y = first_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = first_order_advection(dx, dy, u0, v0, v0)

    u1 = u0 - dt * (adv_u_x + adv_u_y)
    v1 = v0 - dt * (adv_v_x + adv_v_y)

    hb = domain.horizontal_boundary
    hb.enforce_field(
        u1,
        field_name="x_velocity",
        field_units="m s^-1",
        time=new_state["time"],
        grid=grid,
    )
    hb.enforce_field(
        v1,
        field_name="y_velocity",
        field_units="m s^-1",
        time=new_state["time"],
        grid=grid,
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u1, new_state["x_velocity"])

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v1, new_state["y_velocity"])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_rk2(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 40),
            yaxis_length=(1, 40),
            zaxis_length=(1, 1),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    assume(domain.horizontal_boundary.type != "relaxed" or not is_gt(backend))
    grid = domain.numerical_grid

    state = data.draw(
        st_burgers_state(
            grid,
            time=datetime(year=1992, month=2, day=20),
            backend=backend,
            default_origin=default_origin,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(seconds=120)
        ),
        label="timestep",
    )

    # ========================================
    # test
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    dycore = BurgersDynamicalCore(
        domain,
        intermediate_tendency_component=None,
        time_integration_scheme="rk2",
        flux_scheme="third_order",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    domain.horizontal_boundary.reference_state = state

    new_state = dycore(state, {}, timestep)

    assert "time" in new_state
    assert "x_velocity" in new_state
    assert "y_velocity" in new_state
    assert len(new_state) == 3

    assert new_state["time"] == state["time"] + timestep

    dt = timestep.total_seconds()
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    u0 = state["x_velocity"].to_units("m s^-1").data
    v0 = state["y_velocity"].to_units("m s^-1").data

    adv_u_x, adv_u_y = third_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u0, v0, v0)

    u1 = u0 - 0.5 * dt * (adv_u_x + adv_u_y)
    v1 = v0 - 0.5 * dt * (adv_v_x + adv_v_y)

    hb = domain.horizontal_boundary
    hb.enforce_field(
        u1,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
        grid=grid,
    )
    hb.enforce_field(
        v1,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
        grid=grid,
    )

    adv_u_x, adv_u_y = third_order_advection(dx, dy, u1, v1, u1)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u1, v1, v1)

    u2 = u0 - dt * (adv_u_x + adv_u_y)
    v2 = v0 - dt * (adv_v_x + adv_v_y)

    hb.enforce_field(
        u2,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
        grid=grid,
    )
    hb.enforce_field(
        v2,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
        grid=grid,
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u2, new_state["x_velocity"])

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v2, new_state["y_velocity"])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_rk3ws(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 40),
            yaxis_length=(1, 40),
            zaxis_length=(1, 1),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    assume(domain.horizontal_boundary.type != "relaxed" or not is_gt(backend))
    grid = domain.numerical_grid

    state = data.draw(
        st_burgers_state(
            grid,
            time=datetime(year=1992, month=2, day=20),
            backend=backend,
            default_origin=default_origin,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(seconds=120)
        ),
        label="timestep",
    )

    # ========================================
    # test
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    dycore = BurgersDynamicalCore(
        domain,
        intermediate_tendency_component=None,
        time_integration_scheme="rk3ws",
        flux_scheme="fifth_order",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    domain.horizontal_boundary.reference_state = state

    new_state = dycore(state, {}, timestep)

    assert "time" in new_state
    assert "x_velocity" in new_state
    assert "y_velocity" in new_state
    assert len(new_state) == 3

    assert new_state["time"] == state["time"] + timestep

    dt = timestep.total_seconds()
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    u0 = state["x_velocity"].to_units("m s^-1").data
    v0 = state["y_velocity"].to_units("m s^-1").data

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u0, v0, v0)

    u1 = u0 - 1.0 / 3.0 * dt * (adv_u_x + adv_u_y)
    v1 = v0 - 1.0 / 3.0 * dt * (adv_v_x + adv_v_y)

    hb = domain.horizontal_boundary
    hb.enforce_field(
        u1,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 1.0 / 3.0 * timestep,
        grid=grid,
    )
    hb.enforce_field(
        v1,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 1.0 / 3.0 * timestep,
        grid=grid,
    )

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u1, v1, u1)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u1, v1, v1)

    u2 = u0 - 0.5 * dt * (adv_u_x + adv_u_y)
    v2 = v0 - 0.5 * dt * (adv_v_x + adv_v_y)

    hb.enforce_field(
        u2,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
        grid=grid,
    )
    hb.enforce_field(
        v2,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
        grid=grid,
    )

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u2, v2, u2)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u2, v2, v2)

    u3 = u0 - dt * (adv_u_x + adv_u_y)
    v3 = v0 - dt * (adv_v_x + adv_v_y)

    hb.enforce_field(
        u3,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
        grid=grid,
    )
    hb.enforce_field(
        v3,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
        grid=grid,
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u3, new_state["x_velocity"])

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v3, new_state["y_velocity"])


if __name__ == "__main__":
    pytest.main([__file__])
