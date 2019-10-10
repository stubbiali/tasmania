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
from copy import deepcopy
from datetime import timedelta
from hypothesis import given, HealthCheck, settings, strategies as hyp_st
import numpy as np
import pytest

from tasmania.python.isentropic.dynamics.minimal_prognostic import (
    IsentropicMinimalPrognostic,
)
from tasmania.python.isentropic.dynamics.implementations.minimal_prognostic import (
    Centered,
    ForwardEuler,
    RK2,
    RK3WS,
    RK3,
)
from tasmania import get_array_dict

try:
    from .conf import backend as conf_backend  # nb as conf_nb
    from .test_isentropic_horizontal_fluxes import (
        get_upwind_fluxes,
        get_centered_fluxes,
        get_third_order_upwind_fluxes,
        get_fifth_order_upwind_fluxes,
    )
    from .test_isentropic_prognostic import forward_euler_step
    from .utils import (
        compare_arrays,
        compare_datetimes,
        st_domain,
        st_floats,
        st_one_of,
        st_isentropic_state_f,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend  # nb as conf_nb
    from test_isentropic_horizontal_fluxes import (
        get_upwind_fluxes,
        get_centered_fluxes,
        get_third_order_upwind_fluxes,
        get_fifth_order_upwind_fluxes,
    )
    from test_isentropic_prognostic import forward_euler_step
    from utils import (
        compare_arrays,
        compare_datetimes,
        st_domain,
        st_floats,
        st_one_of,
        st_isentropic_state_f,
    )


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_factory(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=3), label="domain"
    )
    mode = data.draw(st_one_of(("x", "y", "xy")), label="mode")
    moist = data.draw(hyp_st.booleans(), label="moist")
    substeps = data.draw(hyp_st.integers(min_value=0, max_value=12), label="substeps")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    imp_centered = IsentropicMinimalPrognostic.factory(
        "centered",
        "centered",
        mode,
        grid,
        hb,
        moist,
        substeps,
        backend=backend,
        dtype=dtype,
    )
    imp_euler = IsentropicMinimalPrognostic.factory(
        "forward_euler",
        "upwind",
        mode,
        grid,
        hb,
        moist,
        substeps,
        backend=backend,
        dtype=dtype,
    )
    imp_rk2 = IsentropicMinimalPrognostic.factory(
        "rk2",
        "third_order_upwind",
        mode,
        grid,
        hb,
        moist,
        substeps,
        backend=backend,
        dtype=dtype,
    )
    imp_rk3ws = IsentropicMinimalPrognostic.factory(
        "rk3ws",
        "fifth_order_upwind",
        mode,
        grid,
        hb,
        moist,
        substeps,
        backend=backend,
        dtype=dtype,
    )
    imp_rk3 = IsentropicMinimalPrognostic.factory(
        "rk3",
        "fifth_order_upwind",
        mode,
        grid,
        hb,
        moist,
        substeps,
        backend=backend,
        dtype=dtype,
    )

    assert isinstance(imp_centered, Centered)
    assert isinstance(imp_euler, ForwardEuler)
    assert isinstance(imp_rk2, RK2)
    assert isinstance(imp_rk3ws, RK3WS)
    assert isinstance(imp_rk3, RK3)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_leapfrog(data):
    # ========================================
    # random data generation
    # ========================================
    nb = 1  # TODO: nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf.nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="tnd_s"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="tnd_su"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="tnd_sv"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="tnd_qv"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="tnd_qc"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="tnd_qr"):
            tendencies[mfpw] = state[mfpw]

    mode = data.draw(st_one_of(("x", "y", "xy")), label="mode")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    imp = IsentropicMinimalPrognostic.factory(
        "centered", "centered", mode, grid, hb, moist, backend=backend, dtype=dtype
    )

    raw_state = get_array_dict(state)
    if moist:
        raw_state["isentropic_density_of_water_vapor"] = (
            raw_state["air_isentropic_density"] * raw_state[mfwv]
        )
        raw_state["isentropic_density_of_cloud_liquid_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfcw]
        )
        raw_state["isentropic_density_of_precipitation_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfpw]
        )

    raw_tendencies = get_array_dict(tendencies)
    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    raw_state_new = imp.stage_call(0, timestep, raw_state, raw_tendencies)

    assert "time" in raw_state_new.keys()
    assert raw_state_new["time"] == raw_state["time"] + timestep

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = ["air_isentropic_density", "x_momentum_isentropic", "y_momentum_isentropic"]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    phi_out = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)

    for name in names:
        phi = raw_state[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_centered_fluxes, "xy", dx, dy, 2 * dt, u, v, phi, phi, phi_tnd, phi_out
        )
        assert name in raw_state_new
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_new[name][nb:-nb, nb:-nb], equal_nan=True
        )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_upwind(data):
    # ========================================
    # random data generation
    # ========================================
    nb = 1  # TODO: nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf.nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="tnd_s"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="tnd_su"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="tnd_sv"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="tnd_qv"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="tnd_qc"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="tnd_qr"):
            tendencies[mfpw] = state[mfpw]

    mode = data.draw(st_one_of(("x", "y", "xy")), label="mode")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    imp = IsentropicMinimalPrognostic.factory(
        "forward_euler", "upwind", mode, grid, hb, moist, backend=backend, dtype=dtype
    )

    raw_state = get_array_dict(state)
    if moist:
        raw_state["isentropic_density_of_water_vapor"] = (
            raw_state["air_isentropic_density"] * raw_state[mfwv]
        )
        raw_state["isentropic_density_of_cloud_liquid_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfcw]
        )
        raw_state["isentropic_density_of_precipitation_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfpw]
        )

    raw_tendencies = get_array_dict(tendencies)
    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    raw_state_new = imp.stage_call(0, timestep, raw_state, raw_tendencies)

    assert "time" in raw_state_new.keys()
    assert raw_state_new["time"] == raw_state["time"] + timestep

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = ["air_isentropic_density", "x_momentum_isentropic", "y_momentum_isentropic"]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    phi_out = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)

    for name in names:
        phi = raw_state[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_upwind_fluxes, mode, dx, dy, dt, u, v, phi, phi, phi_tnd, phi_out
        )
        assert name in raw_state_new
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_new[name][nb:-nb, nb:-nb], equal_nan=True
        )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_rk2(data):
    # ========================================
    # random data generation
    # ========================================
    nb = 2  # TODO: nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf.nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="tnd_s"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="tnd_su"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="tnd_sv"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="tnd_qv"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="tnd_qc"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="tnd_qr"):
            tendencies[mfpw] = state[mfpw]

    mode = data.draw(st_one_of(("x", "y", "xy")), label="mode")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-3), max_value=timedelta(minutes=60)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    imp = IsentropicMinimalPrognostic.factory(
        "rk2", "third_order_upwind", mode, grid, hb, moist, backend=backend, dtype=dtype
    )

    raw_state = get_array_dict(state)
    if moist:
        raw_state["isentropic_density_of_water_vapor"] = (
            raw_state["air_isentropic_density"] * raw_state[mfwv]
        )
        raw_state["isentropic_density_of_cloud_liquid_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfcw]
        )
        raw_state["isentropic_density_of_precipitation_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfpw]
        )

    raw_tendencies = get_array_dict(tendencies)
    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = ["air_isentropic_density", "x_momentum_isentropic", "y_momentum_isentropic"]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    phi_out = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)

    #
    # stage 0
    #
    raw_state_1 = imp.stage_call(0, timestep, raw_state, raw_tendencies)

    assert "time" in raw_state_1.keys()
    assert raw_state_1["time"] == raw_state["time"] + 0.5 * timestep

    for name in names:
        phi = raw_state[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_third_order_upwind_fluxes,
            mode,
            dx,
            dy,
            0.5 * dt,
            u,
            v,
            phi,
            phi,
            phi_tnd,
            phi_out,
        )
        assert name in raw_state_1
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_1[name][nb:-nb, nb:-nb], equal_nan=True
        )

    #
    # stage 1
    #
    raw_state_1["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_1["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    raw_state_1_dc = deepcopy(raw_state_1)

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    raw_state_2 = imp.stage_call(1, timestep, raw_state_1, raw_tendencies)

    assert "time" in raw_state_2.keys()
    assert raw_state_2["time"] == raw_state_1["time"] + 0.5 * timestep

    for name in names:
        phi = raw_state[name]
        phi_tmp = raw_state_1_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_third_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt,
            u,
            v,
            phi,
            phi_tmp,
            phi_tnd,
            phi_out,
        )
        assert name in raw_state_2
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_2[name][nb:-nb, nb:-nb], equal_nan=True
        )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_rk3ws(data):
    # ========================================
    # random data generation
    # ========================================
    nb = 3  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf.nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="tnd_s"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="tnd_su"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="tnd_sv"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="tnd_qv"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="tnd_qc"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="tnd_qr"):
            tendencies[mfpw] = state[mfpw]

    mode = data.draw(st_one_of(("x", "y", "xy")), label="mode")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    imp = IsentropicMinimalPrognostic.factory(
        "rk3ws", "fifth_order_upwind", mode, grid, hb, moist, backend=backend, dtype=dtype
    )

    raw_state = get_array_dict(state)
    if moist:
        raw_state["isentropic_density_of_water_vapor"] = (
            raw_state["air_isentropic_density"] * raw_state[mfwv]
        )
        raw_state["isentropic_density_of_cloud_liquid_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfcw]
        )
        raw_state["isentropic_density_of_precipitation_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfpw]
        )

    raw_tendencies = get_array_dict(tendencies)
    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = ["air_isentropic_density", "x_momentum_isentropic", "y_momentum_isentropic"]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    phi_out = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)

    #
    # stage 0
    #
    raw_state_1 = imp.stage_call(0, timestep, raw_state, raw_tendencies)

    assert "time" in raw_state_1.keys()
    assert raw_state_1["time"] == raw_state["time"] + 1.0 / 3.0 * timestep

    for name in names:
        phi = raw_state[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt / 3.0,
            u,
            v,
            phi,
            phi,
            phi_tnd,
            phi_out,
        )
        assert name in raw_state_1
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_1[name][nb:-nb, nb:-nb], equal_nan=True
        )

    #
    # stage 1
    #
    raw_state_1["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_1["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    raw_state_1_dc = deepcopy(raw_state_1)

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    raw_state_2 = imp.stage_call(1, timestep, raw_state_1, raw_tendencies)

    assert "time" in raw_state_2.keys()
    assert raw_state_2["time"] == raw_state_1["time"] + 1.0 / 6.0 * timestep

    for name in names:
        phi = raw_state[name]
        phi_tmp = raw_state_1_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            0.5 * dt,
            u,
            v,
            phi,
            phi_tmp,
            phi_tnd,
            phi_out,
        )
        assert name in raw_state_2
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_2[name][nb:-nb, nb:-nb], equal_nan=True
        )

    #
    # stage 2
    #
    raw_state_2["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_2["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    raw_state_2_dc = deepcopy(raw_state_1)

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state_2["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state_2["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state_2["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    raw_state_3 = imp.stage_call(2, timestep, raw_state_2, raw_tendencies)

    assert "time" in raw_state_3.keys()
    assert raw_state_3["time"] == raw_state_2["time"] + 0.5 * timestep

    for name in names:
        phi = raw_state[name]
        phi_tmp = raw_state_2_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt,
            u,
            v,
            phi,
            phi_tmp,
            phi_tnd,
            phi_out,
        )
        assert name in raw_state_3
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_3[name][nb:-nb, nb:-nb], equal_nan=True
        )


def rk3_first_step(get_fluxes, mode, dx, dy, dt, a1, u, v, phi, phi_tnd, phi_k0, phi_out):
    flux_x, flux_y = get_fluxes(u, v, phi)
    phi_k0[1:-1, 1:-1] = -dt * (
        ((flux_x[1:-1, 1:-1] - flux_x[:-2, 1:-1]) / dx if mode != "y" else 0.0)
        + ((flux_y[1:-1, 1:-1] - flux_y[1:-1, :-2]) / dy if mode != "x" else 0.0)
        - (phi_tnd[1:-1, 1:-1] if phi_tnd is not None else 0.0)
    )
    phi_out[1:-1, 1:-1] = phi[1:-1, 1:-1] + a1 * phi_k0[1:-1, 1:-1]


def rk3_second_step(
    get_fluxes,
    mode,
    dx,
    dy,
    dt,
    a2,
    b21,
    u_tmp,
    v_tmp,
    phi,
    phi_tmp,
    phi_k0,
    phi_tnd,
    phi_k1,
    phi_out,
):
    flux_x, flux_y = get_fluxes(u_tmp, v_tmp, phi_tmp)
    phi_k1[1:-1, 1:-1] = -dt * (
        ((flux_x[1:-1, 1:-1] - flux_x[:-2, 1:-1]) / dx if mode != "y" else 0.0)
        + ((flux_y[1:-1, 1:-1] - flux_y[1:-1, :-2]) / dy if mode != "x" else 0.0)
        - (phi_tnd[1:-1, 1:-1] if phi_tnd is not None else 0.0)
    )
    phi_out[1:-1, 1:-1] = (
        phi[1:-1, 1:-1] + b21 * phi_k0[1:-1, 1:-1] + (a2 - b21) * phi_k1[1:-1, 1:-1]
    )


def rk3_third_step(
    get_fluxes,
    mode,
    dx,
    dy,
    dt,
    g0,
    g1,
    g2,
    u_tmp,
    v_tmp,
    phi,
    phi_tmp,
    phi_k0,
    phi_k1,
    phi_tnd,
    phi_k2,
    phi_out,
):
    flux_x, flux_y = get_fluxes(u_tmp, v_tmp, phi_tmp)
    phi_k2[1:-1, 1:-1] = -dt * (
        ((flux_x[1:-1, 1:-1] - flux_x[:-2, 1:-1]) / dx if mode != "y" else 0.0)
        + ((flux_y[1:-1, 1:-1] - flux_y[1:-1, :-2]) / dy if mode != "x" else 0.0)
        - (phi_tnd[1:-1, 1:-1] if phi_tnd is not None else 0.0)
    )
    phi_out[1:-1, 1:-1] = (
        phi[1:-1, 1:-1]
        + g0 * phi_k0[1:-1, 1:-1]
        + g1 * phi_k1[1:-1, 1:-1]
        + g2 * phi_k2[1:-1, 1:-1]
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_rk3(data):
    # ========================================
    # random data generation
    # ========================================
    nb = 3  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf.nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="tnd_s"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="tnd_su"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="tnd_sv"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="tnd_qv"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="tnd_qc"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="tnd_qr"):
            tendencies[mfpw] = state[mfpw]

    mode = data.draw(st_one_of(("x", "y", "xy")), label="mode")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    imp = IsentropicMinimalPrognostic.factory(
        "rk3", "fifth_order_upwind", mode, grid, hb, moist, backend=backend, dtype=dtype
    )

    a1, a2 = imp._alpha1, imp._alpha2
    b21 = imp._beta21
    g0, g1, g2 = imp._gamma0, imp._gamma1, imp._gamma2

    g1_val = (3 * a2 - 2) / (6 * a1 * (a2 - a1))
    assert np.isclose(g1, g1_val)
    g2_val = (3 * a1 - 2) / (6 * a2 * (a1 - a2))
    assert np.isclose(g2, g2_val)
    g0_val = 1 - g1 - g2
    assert np.isclose(g0, g0_val)
    b21_val = a2 - 1 / (6 * a1 * g2)
    assert np.isclose(b21, b21_val)

    raw_state = get_array_dict(state)
    if moist:
        raw_state["isentropic_density_of_water_vapor"] = (
            raw_state["air_isentropic_density"] * raw_state[mfwv]
        )
        raw_state["isentropic_density_of_cloud_liquid_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfcw]
        )
        raw_state["isentropic_density_of_precipitation_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfpw]
        )

    raw_tendencies = get_array_dict(tendencies)
    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = ["air_isentropic_density", "x_momentum_isentropic", "y_momentum_isentropic"]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    phi_out = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)
    phi_k0 = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)
    phi_k1 = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)
    phi_k2 = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)
    k0 = {}
    k1 = {}

    #
    # stage 0
    #
    raw_state_1 = imp.stage_call(0, timestep, raw_state, raw_tendencies)

    assert "time" in raw_state_1.keys()
    assert raw_state_1["time"] == raw_state["time"] + a1 * timestep

    for name in names:
        phi = raw_state[name]
        phi_tnd = raw_tendencies.get(name, None)
        rk3_first_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt,
            a1,
            u,
            v,
            phi,
            phi_tnd,
            phi_k0,
            phi_out,
        )
        assert name in raw_state_1
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_1[name][nb:-nb, nb:-nb], equal_nan=True
        )
        k0[name] = deepcopy(phi_k0)

    #
    # stage 1
    #
    raw_state_1["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_1["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    raw_state_1_dc = deepcopy(raw_state_1)

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state_1["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    raw_state_2 = imp.stage_call(1, timestep, raw_state_1, raw_tendencies)

    assert "time" in raw_state_2.keys()
    assert raw_state_2["time"] == raw_state_1["time"] + (a2 - a1) * timestep

    for name in names:
        phi = raw_state[name]
        phi_tmp = raw_state_1_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        rk3_second_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt,
            a2,
            b21,
            u,
            v,
            phi,
            phi_tmp,
            k0[name],
            phi_tnd,
            phi_k1,
            phi_out,
        )
        assert name in raw_state_2
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_2[name][nb:-nb, nb:-nb], equal_nan=True
        )
        k1[name] = deepcopy(phi_k1)

    #
    # stage 2
    #
    raw_state_2["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_2["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    raw_state_2_dc = deepcopy(raw_state_1)

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state_2["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state_2["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state_2["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    raw_state_3 = imp.stage_call(2, timestep, raw_state_2, raw_tendencies)

    assert "time" in raw_state_3.keys()
    assert raw_state_3["time"] == raw_state_2["time"] + (1 - a2 - a1) * timestep

    for name in names:
        phi = raw_state[name]
        phi_tmp = raw_state_2_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        rk3_third_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt,
            g0,
            g1,
            g2,
            u,
            v,
            phi,
            phi_tmp,
            k0[name],
            k1[name],
            phi_tnd,
            phi_k2,
            phi_out,
        )
        assert name in raw_state_3
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_3[name][nb:-nb, nb:-nb], equal_nan=True
        )


def substep(substeps, dts, phi, phi_stage, phi_tmp, phi_tnd, phi_out):
    phi_out[...] = (
        phi_tmp[...]
        + (phi_stage[...] - phi[...]) / substeps
        + (dts * phi_tnd[...] if phi_tnd is not None else 0.0)
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_rk3ws_substepping(data):
    # ========================================
    # random data generation
    # ========================================
    nb = 3  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf.nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="tnd_s"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="tnd_su"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="tnd_sv"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="tnd_qv"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="tnd_qc"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="tnd_qr"):
            tendencies[mfpw] = state[mfpw]

    mode = data.draw(st_one_of(("x", "y", "xy")), label="mode")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    imp = IsentropicMinimalPrognostic.factory(
        "rk3ws",
        "fifth_order_upwind",
        mode,
        grid,
        hb,
        moist,
        substeps=6,
        backend=backend,
        dtype=dtype,
    )

    substep_output_properties = {
        "air_isentropic_density": "kg m^-2 K^-1",
        "x_momentum_isentropic": "kg m^-1 K^-1 s^-1",
        "y_momentum_isentropic": "kg m^-1 K^-1 s^-1",
    }
    if moist:
        substep_output_properties.update({mfwv: "g g^-1", mfcw: "g g^-1", mfpw: "g g^-1"})
    imp.substep_output_properties = substep_output_properties

    raw_state = get_array_dict(state)
    if moist:
        raw_state["isentropic_density_of_water_vapor"] = (
            raw_state["air_isentropic_density"] * raw_state[mfwv]
        )
        raw_state["isentropic_density_of_cloud_liquid_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfcw]
        )
        raw_state["isentropic_density_of_precipitation_water"] = (
            raw_state["air_isentropic_density"] * raw_state[mfpw]
        )

    raw_tendencies = get_array_dict(tendencies)
    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = ["air_isentropic_density", "x_momentum_isentropic", "y_momentum_isentropic"]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    substep_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        substep_names.append(mfwv)
        substep_names.append(mfcw)
        substep_names.append(mfpw)

    phi_out = np.zeros((grid.nx, grid.ny, grid.nz), dtype=grid.x.dtype)

    #
    # stage 0
    #
    raw_state_0 = imp.stage_call(0, timestep, raw_state, raw_tendencies)

    assert "time" in raw_state_0.keys()
    compare_datetimes(raw_state_0["time"], raw_state["time"] + 1.0 / 3.0 * timestep)

    for name in names:
        phi = raw_state[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt / 3.0,
            u,
            v,
            phi,
            phi,
            phi_tnd,
            phi_out,
        )
        assert name in raw_state_0
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_0[name][nb:-nb, nb:-nb], equal_nan=True
        )

    if moist:
        raw_state_0[mfwv] = (
            raw_state_0["isentropic_density_of_water_vapor"]
            / raw_state_0["air_isentropic_density"]
        )
        raw_state_0[mfcw] = (
            raw_state_0["isentropic_density_of_cloud_liquid_water"]
            / raw_state_0["air_isentropic_density"]
        )
        raw_state_0[mfpw] = (
            raw_state_0["isentropic_density_of_precipitation_water"]
            / raw_state_0["air_isentropic_density"]
        )

    #
    # stage 0, substep 0
    #
    raw_state_00 = imp.substep_call(
        0, 0, timestep, raw_state, raw_state_0, raw_state_0, raw_tendencies
    )

    assert "time" in raw_state_00.keys()
    compare_datetimes(raw_state_00["time"], raw_state["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_0[name]
        phi_tmp = phi
        phi_tnd = raw_tendencies.get(name, None)
        substep(2, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_00
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_00[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_00_dc = deepcopy(raw_state_00)

    #
    # stage 0, substep 1
    #
    raw_state_01 = imp.substep_call(
        0, 1, timestep, raw_state, raw_state_0, raw_state_00, raw_tendencies
    )

    assert "time" in raw_state_01.keys()
    compare_datetimes(raw_state_01["time"], raw_state_00["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_0[name]
        phi_tmp = raw_state_00_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        substep(2, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_01
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_01[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_01["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_01["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    if moist:
        raw_state_01["isentropic_density_of_water_vapor"] = (
            raw_state_01["air_isentropic_density"] * raw_state_01[mfwv]
        )
        raw_state_01["isentropic_density_of_cloud_liquid_water"] = (
            raw_state_01["air_isentropic_density"] * raw_state_01[mfcw]
        )
        raw_state_01["isentropic_density_of_precipitation_water"] = (
            raw_state_01["air_isentropic_density"] * raw_state_01[mfpw]
        )

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state_01["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state_01["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state_01["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    #
    # stage 1
    #
    raw_state_1 = imp.stage_call(1, timestep, raw_state_01, raw_tendencies)

    assert "time" in raw_state_1.keys()
    compare_datetimes(raw_state_1["time"], raw_state_01["time"] + 1.0 / 6.0 * timestep)

    for name in names:
        phi = raw_state[name]
        phi_tmp = raw_state_01[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt / 2.0,
            u,
            v,
            phi,
            phi_tmp,
            phi_tnd,
            phi_out,
        )
        assert name in raw_state_1
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_1[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_1["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_1["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    if moist:
        raw_state_1[mfwv] = (
            raw_state_1["isentropic_density_of_water_vapor"]
            / raw_state_1["air_isentropic_density"]
        )
        raw_state_1[mfcw] = (
            raw_state_1["isentropic_density_of_cloud_liquid_water"]
            / raw_state_1["air_isentropic_density"]
        )
        raw_state_1[mfpw] = (
            raw_state_1["isentropic_density_of_precipitation_water"]
            / raw_state_1["air_isentropic_density"]
        )

    #
    # stage 1, substep 0
    #
    raw_state_10 = imp.substep_call(
        1, 0, timestep, raw_state, raw_state_1, raw_state_0, raw_tendencies
    )

    assert "time" in raw_state_10.keys()
    compare_datetimes(raw_state_10["time"], raw_state["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_1[name]
        phi_tmp = phi
        phi_tnd = raw_tendencies.get(name, None)
        substep(3, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_10
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_10[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_10_dc = deepcopy(raw_state_10)

    #
    # stage 1, substep 1
    #
    raw_state_11 = imp.substep_call(
        1, 1, timestep, raw_state, raw_state_1, raw_state_10, raw_tendencies
    )

    assert "time" in raw_state_11.keys()
    compare_datetimes(raw_state_11["time"], raw_state_10["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_1[name]
        phi_tmp = raw_state_10_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        substep(3, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_11
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_11[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_11_dc = deepcopy(raw_state_11)

    #
    # stage 1, substep 2
    #
    raw_state_12 = imp.substep_call(
        1, 2, timestep, raw_state, raw_state_1, raw_state_11, raw_tendencies
    )

    assert "time" in raw_state_12.keys()
    compare_datetimes(raw_state_12["time"], raw_state_11["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_1[name]
        phi_tmp = raw_state_11_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        substep(3, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_12
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_12[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_12["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_12["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    if moist:
        raw_state_12["isentropic_density_of_water_vapor"] = (
            raw_state_12["air_isentropic_density"] * raw_state_12[mfwv]
        )
        raw_state_12["isentropic_density_of_cloud_liquid_water"] = (
            raw_state_12["air_isentropic_density"] * raw_state_12[mfcw]
        )
        raw_state_12["isentropic_density_of_precipitation_water"] = (
            raw_state_12["air_isentropic_density"] * raw_state_12[mfpw]
        )

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = (
                raw_state_12["air_isentropic_density"] * raw_tendencies[mfwv]
            )
        if mfcw in raw_tendencies:
            raw_tendencies["isentropic_density_of_cloud_liquid_water"] = (
                raw_state_12["air_isentropic_density"] * raw_tendencies[mfcw]
            )
        if mfpw in raw_tendencies:
            raw_tendencies["isentropic_density_of_precipitation_water"] = (
                raw_state_12["air_isentropic_density"] * raw_tendencies[mfpw]
            )

    #
    # stage 2
    #
    raw_state_2 = imp.stage_call(2, timestep, raw_state_12, raw_tendencies)

    assert "time" in raw_state_2.keys()
    compare_datetimes(raw_state_2["time"], raw_state_12["time"] + 0.5 * timestep)

    for name in names:
        phi = raw_state[name]
        phi_tmp = raw_state_12[name]
        phi_tnd = raw_tendencies.get(name, None)
        forward_euler_step(
            get_fifth_order_upwind_fluxes,
            mode,
            dx,
            dy,
            dt,
            u,
            v,
            phi,
            phi_tmp,
            phi_tnd,
            phi_out,
        )
        assert name in raw_state_2
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_2[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_2["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_2["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    if moist:
        raw_state_2[mfwv] = (
            raw_state_2["isentropic_density_of_water_vapor"]
            / raw_state_2["air_isentropic_density"]
        )
        raw_state_2[mfcw] = (
            raw_state_2["isentropic_density_of_cloud_liquid_water"]
            / raw_state_2["air_isentropic_density"]
        )
        raw_state_2[mfpw] = (
            raw_state_2["isentropic_density_of_precipitation_water"]
            / raw_state_2["air_isentropic_density"]
        )

    #
    # stage 2, substep 0
    #
    raw_state_20 = imp.substep_call(
        2, 0, timestep, raw_state, raw_state_2, raw_state_0, raw_tendencies
    )

    assert "time" in raw_state_20.keys()
    compare_datetimes(raw_state_20["time"], raw_state["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_2[name]
        phi_tmp = phi
        phi_tnd = raw_tendencies.get(name, None)
        substep(6, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_20
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_20[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_20_dc = deepcopy(raw_state_20)

    #
    # stage 2, substep 1
    #
    raw_state_21 = imp.substep_call(
        2, 1, timestep, raw_state, raw_state_2, raw_state_20, raw_tendencies
    )

    assert "time" in raw_state_21.keys()
    compare_datetimes(raw_state_21["time"], raw_state_20["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_2[name]
        phi_tmp = raw_state_20_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        substep(6, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_21
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_21[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_21_dc = deepcopy(raw_state_21)

    #
    # stage 2, substep 2
    #
    raw_state_22 = imp.substep_call(
        2, 2, timestep, raw_state, raw_state_2, raw_state_21, raw_tendencies
    )

    assert "time" in raw_state_22.keys()
    compare_datetimes(raw_state_22["time"], raw_state_21["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_2[name]
        phi_tmp = raw_state_21_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        substep(6, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_22
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_22[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_22_dc = deepcopy(raw_state_22)

    #
    # stage 2, substep 3
    #
    raw_state_23 = imp.substep_call(
        2, 3, timestep, raw_state, raw_state_2, raw_state_22, raw_tendencies
    )

    assert "time" in raw_state_23.keys()
    compare_datetimes(raw_state_23["time"], raw_state_22["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_2[name]
        phi_tmp = raw_state_22_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        substep(6, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_23
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_23[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_23_dc = deepcopy(raw_state_23)

    #
    # stage 2, substep 4
    #
    raw_state_24 = imp.substep_call(
        2, 4, timestep, raw_state, raw_state_2, raw_state_23, raw_tendencies
    )

    assert "time" in raw_state_24.keys()
    compare_datetimes(raw_state_24["time"], raw_state_23["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_2[name]
        phi_tmp = raw_state_23_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        substep(6, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_24
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_24[name][nb:-nb, nb:-nb], equal_nan=True
        )

    raw_state_24_dc = deepcopy(raw_state_24)

    #
    # stage 2, substep 5
    #
    raw_state_25 = imp.substep_call(
        2, 5, timestep, raw_state, raw_state_2, raw_state_24, raw_tendencies
    )

    assert "time" in raw_state_25.keys()
    compare_datetimes(raw_state_25["time"], raw_state_24["time"] + 1.0 / 6.0 * timestep)

    for name in substep_names:
        phi = raw_state[name]
        phi_stage = raw_state_2[name]
        phi_tmp = raw_state_24_dc[name]
        phi_tnd = raw_tendencies.get(name, None)
        substep(6, dt / 6.0, phi, phi_stage, phi_tmp, phi_tnd, phi_out)
        assert name in raw_state_25
        assert np.allclose(
            phi_out[nb:-nb, nb:-nb], raw_state_25[name][nb:-nb, nb:-nb], equal_nan=True
        )


if __name__ == "__main__":
    pytest.main([__file__])
