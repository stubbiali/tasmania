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

from tasmania.python.burgers.dynamics.dycore import BurgersDynamicalCore
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions

from tests import conf
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
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_forward_euler(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
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

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(seconds=120)
        ),
        label="timestep",
    )

    # ========================================
    # test
    # ========================================
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
    u0 = to_numpy(state["x_velocity"].to_units("m s^-1").data)
    v0 = to_numpy(state["y_velocity"].to_units("m s^-1").data)

    adv_u_x, adv_u_y = first_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = first_order_advection(dx, dy, u0, v0, v0)

    u1 = u0 - dt * (adv_u_x + adv_u_y)
    v1 = v0 - dt * (adv_v_x + adv_v_y)

    hb = domain.copy(backend="numpy").horizontal_boundary
    hb.enforce_field(
        u1,
        field_name="x_velocity",
        field_units="m s^-1",
        time=new_state["time"],
    )
    hb.enforce_field(
        v1,
        field_name="y_velocity",
        field_units="m s^-1",
        time=new_state["time"],
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u1, new_state["x_velocity"].data)

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v1, new_state["y_velocity"].data)


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
    bo = BackendOptions(rebuild=False)
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

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(seconds=120)
        ),
        label="timestep",
    )

    # ========================================
    # test
    # ========================================
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
    u0 = to_numpy(state["x_velocity"].to_units("m s^-1").data)
    v0 = to_numpy(state["y_velocity"].to_units("m s^-1").data)

    adv_u_x, adv_u_y = third_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u0, v0, v0)

    u1 = u0 - 0.5 * dt * (adv_u_x + adv_u_y)
    v1 = v0 - 0.5 * dt * (adv_v_x + adv_v_y)

    hb = domain.copy(backend="numpy").horizontal_boundary
    hb.enforce_field(
        u1,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
    )
    hb.enforce_field(
        v1,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
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
    )
    hb.enforce_field(
        v2,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u2, new_state["x_velocity"].data)

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v2, new_state["y_velocity"].data)


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
    bo = BackendOptions(rebuild=False)
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

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(seconds=120)
        ),
        label="timestep",
    )

    # ========================================
    # test
    # ========================================
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
    u0 = to_numpy(state["x_velocity"].to_units("m s^-1").data)
    v0 = to_numpy(state["y_velocity"].to_units("m s^-1").data)

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u0, v0, u0)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u0, v0, v0)

    u1 = u0 - 1.0 / 3.0 * dt * (adv_u_x + adv_u_y)
    v1 = v0 - 1.0 / 3.0 * dt * (adv_v_x + adv_v_y)

    hb = domain.copy(backend="numpy").horizontal_boundary
    hb.enforce_field(
        u1,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 1.0 / 3.0 * timestep,
    )
    hb.enforce_field(
        v1,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 1.0 / 3.0 * timestep,
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
    )
    hb.enforce_field(
        v2,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
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
    )
    hb.enforce_field(
        v3,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u3, new_state["x_velocity"].data)

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v3, new_state["y_velocity"].data)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_forward_euler("gt4py:gtx86", float)
