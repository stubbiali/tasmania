# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
import numpy as np
import pytest

from tasmania.python.burgers.dynamics.dycore import BurgersDynamicalCore
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.utils.storage import deepcopy_dataarray_dict

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
        fast_tendency_component=None,
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

    i, j = slice(nb, grid.nx - nb), slice(nb, grid.ny - nb)
    u1 = np.zeros_like(u0)
    u1[i, j] = u0[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v1 = np.zeros_like(v0)
    v1[i, j] = v0[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

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
def test_forward_euler_oop(data, backend, dtype):
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
    assume(domain.horizontal_boundary.type != "identity")
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
        fast_tendency_component=None,
        time_integration_scheme="forward_euler",
        flux_scheme="first_order",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    domain.horizontal_boundary.reference_state = state

    new_state = deepcopy_dataarray_dict(state)
    dycore(state, {}, timestep, out_state=new_state)

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

    i, j = slice(nb, grid.nx - nb), slice(nb, grid.ny - nb)
    u1 = np.zeros_like(u0)
    u1[i, j] = u0[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v1 = np.zeros_like(v0)
    v1[i, j] = v0[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    state, new_state = new_state, state
    dycore(state, {}, timestep, out_state=new_state)

    assert "time" in new_state
    assert "x_velocity" in new_state
    assert "y_velocity" in new_state
    assert len(new_state) == 3

    assert new_state["time"] == state["time"] + timestep

    adv_u_x, adv_u_y = first_order_advection(dx, dy, u1, v1, u1)
    adv_v_x, adv_v_y = first_order_advection(dx, dy, u1, v1, v1)

    u2 = np.zeros_like(u0)
    u2[i, j] = u1[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v2 = np.zeros_like(v0)
    v2[i, j] = v1[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

    hb.enforce_field(
        u2,
        field_name="x_velocity",
        field_units="m s^-1",
        time=new_state["time"],
    )
    hb.enforce_field(
        v2,
        field_name="y_velocity",
        field_units="m s^-1",
        time=new_state["time"],
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u2, new_state["x_velocity"].data)

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v2, new_state["y_velocity"].data)


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
        fast_tendency_component=None,
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

    i, j = slice(nb, grid.nx - nb), slice(nb, grid.ny - nb)
    u1 = np.zeros_like(u0)
    u1[i, j] = u0[i, j] - 0.5 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v1 = np.zeros_like(v0)
    v1[i, j] = v0[i, j] - 0.5 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    u2 = np.zeros_like(u0)
    u2[i, j] = u0[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v2 = np.zeros_like(v0)
    v2[i, j] = v0[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

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
def test_rk2_oop(data, backend, dtype):
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
    assume(domain.horizontal_boundary.type != "identity")
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
        fast_tendency_component=None,
        time_integration_scheme="rk2",
        flux_scheme="third_order",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    domain.horizontal_boundary.reference_state = state

    new_state = deepcopy_dataarray_dict(state)
    dycore(state, {}, timestep, out_state=new_state)

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

    i, j = slice(nb, grid.nx - nb), slice(nb, grid.ny - nb)
    u1 = np.zeros_like(u0)
    u1[i, j] = u0[i, j] - 0.5 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v1 = np.zeros_like(v0)
    v1[i, j] = v0[i, j] - 0.5 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    u2 = np.zeros_like(u0)
    u2[i, j] = u0[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v2 = np.zeros_like(v0)
    v2[i, j] = v0[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    new_state, state = state, new_state
    dycore(state, {}, timestep, out_state=new_state)

    assert "time" in new_state
    assert "x_velocity" in new_state
    assert "y_velocity" in new_state
    assert len(new_state) == 3

    assert new_state["time"] == state["time"] + timestep

    adv_u_x, adv_u_y = third_order_advection(dx, dy, u2, v2, u2)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u2, v2, v2)

    u3 = np.zeros_like(u0)
    u3[i, j] = u2[i, j] - 0.5 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v3 = np.zeros_like(v0)
    v3[i, j] = v2[i, j] - 0.5 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

    hb.enforce_field(
        u3,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
    )
    hb.enforce_field(
        v3,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
    )

    adv_u_x, adv_u_y = third_order_advection(dx, dy, u3, v3, u3)
    adv_v_x, adv_v_y = third_order_advection(dx, dy, u3, v3, v3)

    u4 = np.zeros_like(u0)
    u4[i, j] = u2[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v4 = np.zeros_like(v0)
    v4[i, j] = v2[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

    hb.enforce_field(
        u4,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
    )
    hb.enforce_field(
        v4,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u4, new_state["x_velocity"].data)

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v4, new_state["y_velocity"].data)


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
        fast_tendency_component=None,
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

    i, j = slice(nb, grid.nx - nb), slice(nb, grid.ny - nb)
    u1 = np.zeros_like(u0)
    u1[i, j] = u0[i, j] - 1.0 / 3.0 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v1 = np.zeros_like(v0)
    v1[i, j] = v0[i, j] - 1.0 / 3.0 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    u2 = np.zeros_like(u0)
    u2[i, j] = u0[i, j] - 0.5 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v2 = np.zeros_like(v0)
    v2[i, j] = v0[i, j] - 0.5 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    u3 = np.zeros_like(u0)
    u3[i, j] = u0[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v3 = np.zeros_like(v0)
    v3[i, j] = v0[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

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


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_rk3ws_oop(data, backend, dtype):
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
    assume(domain.horizontal_boundary.type != "identity")
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
        fast_tendency_component=None,
        time_integration_scheme="rk3ws",
        flux_scheme="fifth_order",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    domain.horizontal_boundary.reference_state = state

    new_state = deepcopy_dataarray_dict(state)
    dycore(state, {}, timestep, out_state=new_state)

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

    i, j = slice(nb, grid.nx - nb), slice(nb, grid.ny - nb)
    u1 = np.zeros_like(u0)
    u1[i, j] = u0[i, j] - 1.0 / 3.0 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v1 = np.zeros_like(v0)
    v1[i, j] = v0[i, j] - 1.0 / 3.0 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    u2 = np.zeros_like(u0)
    u2[i, j] = u0[i, j] - 0.5 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v2 = np.zeros_like(v0)
    v2[i, j] = v0[i, j] - 0.5 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    u3 = np.zeros_like(u0)
    u3[i, j] = u0[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v3 = np.zeros_like(v0)
    v3[i, j] = v0[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

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

    new_state, state = state, new_state
    dycore(state, {}, timestep, out_state=new_state)

    assert "time" in new_state
    assert "x_velocity" in new_state
    assert "y_velocity" in new_state
    assert len(new_state) == 3

    assert new_state["time"] == state["time"] + timestep

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u3, v3, u3)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u3, v3, v3)

    u4 = np.zeros_like(u0)
    u4[i, j] = u3[i, j] - 1.0 / 3.0 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v4 = np.zeros_like(v0)
    v4[i, j] = v3[i, j] - 1.0 / 3.0 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

    hb.enforce_field(
        u4,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 1.0 / 3.0 * timestep,
    )
    hb.enforce_field(
        v4,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 1.0 / 3.0 * timestep,
    )

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u4, v4, u4)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u4, v4, v4)

    u5 = np.zeros_like(u0)
    u5[i, j] = u3[i, j] - 0.5 * dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v5 = np.zeros_like(v0)
    v5[i, j] = v3[i, j] - 0.5 * dt * (adv_v_x[i, j] + adv_v_y[i, j])

    hb.enforce_field(
        u5,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
    )
    hb.enforce_field(
        v5,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * timestep,
    )

    adv_u_x, adv_u_y = fifth_order_advection(dx, dy, u5, v5, u5)
    adv_v_x, adv_v_y = fifth_order_advection(dx, dy, u5, v5, v5)

    u6 = np.zeros_like(u0)
    u6[i, j] = u3[i, j] - dt * (adv_u_x[i, j] + adv_u_y[i, j])
    v6 = np.zeros_like(v0)
    v6[i, j] = v3[i, j] - dt * (adv_v_x[i, j] + adv_v_y[i, j])

    hb.enforce_field(
        u6,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
    )
    hb.enforce_field(
        v6,
        field_name="y_velocity",
        field_units="m s^-1",
        time=state["time"] + timestep,
    )

    assert new_state["x_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(u6, new_state["x_velocity"].data)

    assert new_state["y_velocity"].attrs["units"] == "m s^-1"
    compare_arrays(v6, new_state["y_velocity"].data)


if __name__ == "__main__":
    pytest.main([__file__])
