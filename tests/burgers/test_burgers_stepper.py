# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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

from tasmania.python.burgers.dynamics.stepper import BurgersStepper
from tasmania.python.burgers.dynamics.subclasses.stepper.forward_euler import (
    ForwardEuler,
)
from tasmania.python.burgers.dynamics.subclasses.stepper.rk2 import RK2
from tasmania.python.burgers.dynamics.subclasses.stepper.rk3ws import RK3WS
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.utils.storage import deepcopy_array_dict

from tests import conf
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
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_forward_euler(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False, check_rebuild=True)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 40),
            yaxis_length=(1, 40),
            zaxis_length=(1, 1),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid = domain.numerical_grid

    state = data.draw(
        st_burgers_state(
            grid,
            time=datetime(year=1992, month=2, day=20),
            backend=backend,
            storage_options=so,
        ),
        label="state",
    )

    if_tendency = data.draw(hyp_st.booleans(), label="if_tendency")
    tendency = (
        {}
        if not if_tendency
        else data.draw(
            st_burgers_tendency(
                grid, time=state["time"], backend=backend, storage_options=so
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
        backend_options=bo,
        storage_options=so,
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
    out_state = {
        "x_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
        "y_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
    }

    bs(0, raw_state, raw_tendency, timestep, out_state)

    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u = to_numpy(raw_state["x_velocity"])
    v = to_numpy(raw_state["y_velocity"])
    if if_tendency:
        tnd_u = to_numpy(raw_tendency["x_velocity"])
        tnd_v = to_numpy(raw_tendency["y_velocity"])
    adv_u_x, adv_u_y = first_order_advection(dx, dy, u, v, u)
    adv_v_x, adv_v_y = first_order_advection(dx, dy, u, v, v)
    out_u = u - timestep.total_seconds() * (adv_u_x + adv_u_y)
    out_v = v - timestep.total_seconds() * (adv_v_x + adv_v_y)
    if if_tendency:
        out_u += timestep.total_seconds() * tnd_u
        out_v += timestep.total_seconds() * tnd_v

    compare_datetimes(out_state["time"], state["time"] + timestep)
    compare_arrays(
        out_u[nb:-nb, nb:-nb], out_state["x_velocity"][nb:-nb, nb:-nb]
    )
    compare_arrays(
        out_v[nb:-nb, nb:-nb], out_state["y_velocity"][nb:-nb, nb:-nb]
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_rk2(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False, check_rebuild=True)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=2, max_value=max(2, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 40),
            yaxis_length=(1, 40),
            zaxis_length=(1, 1),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid = domain.numerical_grid

    state = data.draw(
        st_burgers_state(
            grid,
            time=datetime(year=1992, month=2, day=20),
            backend=backend,
            storage_options=so,
        ),
        label="state",
    )

    if_tendency = data.draw(hyp_st.booleans(), label="if_tendency")
    tendency = (
        {}
        if not if_tendency
        else data.draw(
            st_burgers_tendency(
                grid, time=state["time"], backend=backend, storage_options=so
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
        backend_options=bo,
        storage_options=so,
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
    raw_state_1 = {
        "x_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
        "y_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
    }
    bs(0, raw_state_0, raw_tendency, timestep, raw_state_1)

    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u0 = to_numpy(raw_state_0["x_velocity"])
    v0 = to_numpy(raw_state_0["y_velocity"])
    if if_tendency:
        tnd_u = to_numpy(raw_tendency["x_velocity"])
        tnd_v = to_numpy(raw_tendency["y_velocity"])
    adv_u_x, adv_u_y = third_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u0, v0, v0)
    u1 = u0 - 0.5 * timestep.total_seconds() * (adv_u_x + adv_u_y)
    v1 = v0 - 0.5 * timestep.total_seconds() * (adv_v_x + adv_v_y)
    if if_tendency:
        u1 += 0.5 * timestep.total_seconds() * tnd_u
        v1 += 0.5 * timestep.total_seconds() * tnd_v

    compare_datetimes(raw_state_1["time"], state["time"] + 0.5 * timestep)
    compare_arrays(
        u1[nb:-nb, nb:-nb], raw_state_1["x_velocity"][nb:-nb, nb:-nb]
    )
    compare_arrays(
        v1[nb:-nb, nb:-nb], raw_state_1["y_velocity"][nb:-nb, nb:-nb]
    )

    # ========================================
    # stage 1
    # ========================================
    raw_state_1_dc = deepcopy_array_dict(raw_state_1)
    raw_state_2 = {
        "x_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
        "y_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
    }
    bs(1, raw_state_1, raw_tendency, timestep, raw_state_2)

    u1 = to_numpy(raw_state_1_dc["x_velocity"])
    v1 = to_numpy(raw_state_1_dc["y_velocity"])
    adv_u_x, adv_u_y = third_order_advection(dx, dy, u1, v1, u1)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u1, v1, v1)
    u2 = u0 - timestep.total_seconds() * (adv_u_x + adv_u_y)
    v2 = v0 - timestep.total_seconds() * (adv_v_x + adv_v_y)
    if if_tendency:
        u2 += timestep.total_seconds() * tnd_u
        v2 += timestep.total_seconds() * tnd_v

    compare_datetimes(raw_state_2["time"], state["time"] + timestep)
    compare_arrays(
        u2[nb:-nb, nb:-nb], raw_state_2["x_velocity"][nb:-nb, nb:-nb]
    )
    compare_arrays(
        v2[nb:-nb, nb:-nb], raw_state_2["y_velocity"][nb:-nb, nb:-nb]
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_rk3ws(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False, check_rebuild=True)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 40),
            yaxis_length=(1, 40),
            zaxis_length=(1, 1),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid = domain.numerical_grid

    state = data.draw(
        st_burgers_state(
            grid,
            time=datetime(year=1992, month=2, day=20),
            backend=backend,
            storage_options=so,
        ),
        label="state",
    )

    if_tendency = data.draw(hyp_st.booleans(), label="if_tendency")
    tendency = (
        {}
        if not if_tendency
        else data.draw(
            st_burgers_tendency(
                grid, time=state["time"], backend=backend, storage_options=so
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
        backend_options=bo,
        storage_options=so,
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
    raw_state_1 = {
        "x_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
        "y_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
    }
    bs(0, raw_state_0, raw_tendency, timestep, raw_state_1)

    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u0 = to_numpy(raw_state_0["x_velocity"])
    v0 = to_numpy(raw_state_0["y_velocity"])
    if if_tendency:
        tnd_u = to_numpy(raw_tendency["x_velocity"])
        tnd_v = to_numpy(raw_tendency["y_velocity"])
    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u0, v0, v0)
    u1 = u0 - 1.0 / 3.0 * timestep.total_seconds() * (adv_u_x + adv_u_y)
    v1 = v0 - 1.0 / 3.0 * timestep.total_seconds() * (adv_v_x + adv_v_y)
    if if_tendency:
        u1 += 1.0 / 3.0 * timestep.total_seconds() * tnd_u
        v1 += 1.0 / 3.0 * timestep.total_seconds() * tnd_v

    compare_datetimes(
        raw_state_1["time"], state["time"] + 1.0 / 3.0 * timestep
    )
    compare_arrays(
        u1[nb:-nb, nb:-nb], raw_state_1["x_velocity"][nb:-nb, nb:-nb]
    )
    compare_arrays(
        v1[nb:-nb, nb:-nb], raw_state_1["y_velocity"][nb:-nb, nb:-nb]
    )

    # ========================================
    # stage 1
    # ========================================
    raw_state_1_dc = deepcopy_array_dict(raw_state_1)
    raw_state_2 = {
        "x_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
        "y_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
    }
    bs(1, raw_state_1, raw_tendency, timestep, raw_state_2)

    u1 = to_numpy(raw_state_1["x_velocity"])
    v1 = to_numpy(raw_state_1["y_velocity"])
    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u1, v1, u1)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u1, v1, v1)
    u2 = u0 - 0.5 * timestep.total_seconds() * (adv_u_x + adv_u_y)
    v2 = v0 - 0.5 * timestep.total_seconds() * (adv_v_x + adv_v_y)
    if if_tendency:
        u2 += 0.5 * timestep.total_seconds() * tnd_u
        v2 += 0.5 * timestep.total_seconds() * tnd_v

    compare_datetimes(raw_state_2["time"], state["time"] + 0.5 * timestep)
    compare_arrays(
        u2[nb:-nb, nb:-nb], raw_state_2["x_velocity"][nb:-nb, nb:-nb]
    )
    compare_arrays(
        v2[nb:-nb, nb:-nb], raw_state_2["y_velocity"][nb:-nb, nb:-nb]
    )

    # ========================================
    # stage 2
    # ========================================
    raw_state_2_dc = deepcopy_array_dict(raw_state_2)
    raw_state_3 = {
        "x_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
        "y_velocity": bs.zeros(shape=(grid.nx, grid.ny, 1)),
    }
    bs(2, raw_state_2, raw_tendency, timestep, raw_state_3)

    u2 = to_numpy(raw_state_2_dc["x_velocity"])
    v2 = to_numpy(raw_state_2_dc["y_velocity"])
    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u2, v2, u2)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u2, v2, v2)
    u3 = u0 - timestep.total_seconds() * (adv_u_x + adv_u_y)
    v3 = v0 - timestep.total_seconds() * (adv_v_x + adv_v_y)
    if if_tendency:
        u3 += timestep.total_seconds() * tnd_u
        v3 += timestep.total_seconds() * tnd_v

    compare_datetimes(raw_state_3["time"], state["time"] + timestep)
    compare_arrays(
        u3[nb:-nb, nb:-nb], raw_state_3["x_velocity"][nb:-nb, nb:-nb]
    )
    compare_arrays(
        v3[nb:-nb, nb:-nb], raw_state_3["y_velocity"][nb:-nb, nb:-nb]
    )


if __name__ == "__main__":
    pytest.main([__file__])
