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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.burgers.dynamics.stepper import (
    BurgersStepper,
    ForwardEuler,
    RK2,
    RK3WS,
)
from tasmania.python.utils.storage_utils import deepcopy_array_dict

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
    st_burgers_tendency,
    st_domain,
    st_one_of,
    st_timedeltas,
)
from tests.utilities import compare_arrays, compare_datetimes, hyp_settings


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

    if_tendency = data.draw(hyp_st.booleans(), label="if_tendency")
    tendency = (
        {}
        if not if_tendency
        else data.draw(
            st_burgers_tendency(
                grid,
                time=state["time"],
                backend=backend,
                default_origin=default_origin,
            ),
            label="tendency",
        )
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
    bs = BurgersStepper.factory(
        "forward_euler",
        grid.grid_xy,
        nb,
        "first_order",
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
    )

    assert isinstance(bs, ForwardEuler)

    raw_state = {
        "time": state["time"],
        "x_velocity": state["x_velocity"].to_units("m s^-1").data,
        "y_velocity": state["y_velocity"].to_units("m s^-1").data,
    }
    if if_tendency:
        raw_tendency = {
            "time": state["time"],
            "x_velocity": tendency["x_velocity"].to_units("m s^-2").data,
            "y_velocity": tendency["y_velocity"].to_units("m s^-2").data,
        }
    else:
        raw_tendency = {}

    out_state = bs(0, raw_state, raw_tendency, timestep)

    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u, v = raw_state["x_velocity"], raw_state["y_velocity"]
    if if_tendency:
        tnd_u, tnd_v = raw_tendency["x_velocity"], raw_tendency["y_velocity"]

    adv_u_x, adv_u_y = first_order_advection(dx, dy, u, v, u)
    adv_v_x, adv_v_y = first_order_advection(dx, dy, u, v, v)
    out_u = u[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
        adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
    )
    out_v = v[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
        adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
    )
    if if_tendency:
        out_u += timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
        out_v += timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

    compare_datetimes(out_state["time"], state["time"] + timestep)
    compare_arrays(out_u, out_state["x_velocity"][nb:-nb, nb:-nb, :])
    compare_arrays(out_v, out_state["y_velocity"][nb:-nb, nb:-nb, :])


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

    if_tendency = data.draw(hyp_st.booleans(), label="if_tendency")
    tendency = (
        {}
        if not if_tendency
        else data.draw(
            st_burgers_tendency(
                grid,
                time=state["time"],
                backend=backend,
                default_origin=default_origin,
            ),
            label="tendency",
        )
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
    bs = BurgersStepper.factory(
        "rk2",
        grid.grid_xy,
        nb,
        "third_order",
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
    )

    assert isinstance(bs, RK2)

    raw_state_0 = {
        "time": state["time"],
        "x_velocity": state["x_velocity"].to_units("m s^-1").data,
        "y_velocity": state["y_velocity"].to_units("m s^-1").data,
    }
    if if_tendency:
        raw_tendency = {
            "time": state["time"],
            "x_velocity": tendency["x_velocity"].to_units("m s^-2").data,
            "y_velocity": tendency["y_velocity"].to_units("m s^-2").data,
        }
    else:
        raw_tendency = {}

    # ========================================
    # stage 0
    # ========================================
    raw_state_1 = bs(0, raw_state_0, raw_tendency, timestep)

    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u0, v0 = raw_state_0["x_velocity"], raw_state_0["y_velocity"]
    if if_tendency:
        tnd_u, tnd_v = raw_tendency["x_velocity"], raw_tendency["y_velocity"]

    adv_u_x, adv_u_y = third_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u0, v0, v0)
    u1 = u0[nb:-nb, nb:-nb, :] - 0.5 * timestep.total_seconds() * (
        adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
    )
    v1 = v0[nb:-nb, nb:-nb, :] - 0.5 * timestep.total_seconds() * (
        adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
    )
    if if_tendency:
        u1 += 0.5 * timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
        v1 += 0.5 * timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

    compare_datetimes(raw_state_1["time"], state["time"] + 0.5 * timestep)
    compare_arrays(u1, raw_state_1["x_velocity"][nb:-nb, nb:-nb, :])
    compare_arrays(v1, raw_state_1["y_velocity"][nb:-nb, nb:-nb, :])

    # ========================================
    # stage 1
    # ========================================
    raw_state_1 = deepcopy_array_dict(raw_state_1)
    raw_state_2 = bs(1, raw_state_1, raw_tendency, timestep)

    u1, v1 = raw_state_1["x_velocity"], raw_state_1["y_velocity"]

    adv_u_x, adv_u_y = third_order_advection(dx, dy, u1, v1, u1)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u1, v1, v1)
    u2 = u0[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
        adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
    )
    v2 = v0[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
        adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
    )
    if if_tendency:
        u2 += timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
        v2 += timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

    compare_datetimes(raw_state_2["time"], state["time"] + timestep)
    compare_arrays(u2, raw_state_2["x_velocity"][nb:-nb, nb:-nb, :])
    compare_arrays(v2, raw_state_2["y_velocity"][nb:-nb, nb:-nb, :])


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

    if_tendency = data.draw(hyp_st.booleans(), label="if_tendency")
    tendency = (
        {}
        if not if_tendency
        else data.draw(
            st_burgers_tendency(
                grid,
                time=state["time"],
                backend=backend,
                default_origin=default_origin,
            ),
            label="tendency",
        )
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
    bs = BurgersStepper.factory(
        "rk3ws",
        grid.grid_xy,
        nb,
        "fifth_order",
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
    )

    assert isinstance(bs, RK2)
    assert isinstance(bs, RK3WS)

    raw_state_0 = {
        "time": state["time"],
        "x_velocity": state["x_velocity"].to_units("m s^-1").data,
        "y_velocity": state["y_velocity"].to_units("m s^-1").data,
    }
    if if_tendency:
        raw_tendency = {
            "time": state["time"],
            "x_velocity": tendency["x_velocity"].to_units("m s^-2").data,
            "y_velocity": tendency["y_velocity"].to_units("m s^-2").data,
        }
    else:
        raw_tendency = {}

    # ========================================
    # stage 0
    # ========================================
    raw_state_1 = bs(0, raw_state_0, raw_tendency, timestep)

    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u0, v0 = raw_state_0["x_velocity"], raw_state_0["y_velocity"]
    if if_tendency:
        tnd_u, tnd_v = raw_tendency["x_velocity"], raw_tendency["y_velocity"]

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u0, v0, v0)
    u1 = u0[nb:-nb, nb:-nb, :] - 1.0 / 3.0 * timestep.total_seconds() * (
        adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
    )
    v1 = v0[nb:-nb, nb:-nb, :] - 1.0 / 3.0 * timestep.total_seconds() * (
        adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
    )
    if if_tendency:
        u1 += 1.0 / 3.0 * timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
        v1 += 1.0 / 3.0 * timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

    compare_datetimes(
        raw_state_1["time"], state["time"] + 1.0 / 3.0 * timestep
    )
    compare_arrays(u1, raw_state_1["x_velocity"][nb:-nb, nb:-nb, :])
    compare_arrays(v1, raw_state_1["y_velocity"][nb:-nb, nb:-nb, :])

    # ========================================
    # stage 1
    # ========================================
    raw_state_1 = deepcopy_array_dict(raw_state_1)
    raw_state_2 = bs(1, raw_state_1, raw_tendency, timestep)

    u1, v1 = raw_state_1["x_velocity"], raw_state_1["y_velocity"]

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u1, v1, u1)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u1, v1, v1)
    u2 = u0[nb:-nb, nb:-nb, :] - 0.5 * timestep.total_seconds() * (
        adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
    )
    v2 = v0[nb:-nb, nb:-nb, :] - 0.5 * timestep.total_seconds() * (
        adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
    )
    if if_tendency:
        u2 += 0.5 * timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
        v2 += 0.5 * timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

    compare_datetimes(raw_state_2["time"], state["time"] + 0.5 * timestep)
    compare_arrays(u2, raw_state_2["x_velocity"][nb:-nb, nb:-nb, :])
    compare_arrays(v2, raw_state_2["y_velocity"][nb:-nb, nb:-nb, :])

    # ========================================
    # stage 2
    # ========================================
    raw_state_2 = deepcopy_array_dict(raw_state_2)
    raw_state_3 = bs(2, raw_state_2, raw_tendency, timestep)

    u2, v2 = raw_state_2["x_velocity"], raw_state_2["y_velocity"]

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u2, v2, u2)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u2, v2, v2)
    u3 = u0[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
        adv_u_x[nb:-nb, nb:-nb, :] + adv_u_y[nb:-nb, nb:-nb, :]
    )
    v3 = v0[nb:-nb, nb:-nb, :] - timestep.total_seconds() * (
        adv_v_x[nb:-nb, nb:-nb, :] + adv_v_y[nb:-nb, nb:-nb, :]
    )
    if if_tendency:
        u3 += timestep.total_seconds() * tnd_u[nb:-nb, nb:-nb, :]
        v3 += timestep.total_seconds() * tnd_v[nb:-nb, nb:-nb, :]

    compare_datetimes(raw_state_3["time"], state["time"] + timestep)
    compare_arrays(u3, raw_state_3["x_velocity"][nb:-nb, nb:-nb, :])
    compare_arrays(v3, raw_state_3["y_velocity"][nb:-nb, nb:-nb, :])


if __name__ == "__main__":
    pytest.main([__file__])
