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

from tasmania.python.isentropic.dynamics.boussinesq_minimal_prognostic import (
    IsentropicBoussinesqMinimalPrognostic,
)
from tasmania.python.isentropic.dynamics.implementations.boussinesq_minimal_prognostic import (
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
    from .test_isentropic_minimal_prognostic import (
        forward_euler_step,
        rk3_first_step,
        rk3_second_step,
        rk3_third_step,
        substep,
    )
    from .utils import (
        compare_arrays,
        compare_datetimes,
        st_domain,
        st_one_of,
        st_isentropic_boussinesq_state_f,
        st_isentropic_boussinesq_state_ff,
    )
except ModuleNotFoundError:
    from conf import backend as conf_backend  # nb as conf_nb
    from test_isentropic_minimal_horizontal_fluxes import (
        get_upwind_fluxes,
        get_centered_fluxes,
        get_third_order_upwind_fluxes,
        get_fifth_order_upwind_fluxes,
    )
    from test_isentropic_minimal_prognostic import (
        forward_euler_step,
        rk3_first_step,
        rk3_second_step,
        rk3_third_step,
        substep,
    )
    from utils import (
        compare_arrays,
        compare_datetimes,
        st_domain,
        st_one_of,
        st_isentropic_boussinesq_state_f,
        st_isentropic_boussinesq_state_ff,
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

    imp_centered = IsentropicBoussinesqMinimalPrognostic.factory(
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
    imp_euler = IsentropicBoussinesqMinimalPrognostic.factory(
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
    imp_rk2 = IsentropicBoussinesqMinimalPrognostic.factory(
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
    imp_rk3ws = IsentropicBoussinesqMinimalPrognostic.factory(
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
    imp_rk3 = IsentropicBoussinesqMinimalPrognostic.factory(
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
    nb = 1  # TODO: nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_ff(grid, moist=moist), label="state")

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
    imp = IsentropicBoussinesqMinimalPrognostic.factory(
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
    compare_datetimes(raw_state_new["time"], raw_state["time"] + timestep)

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
        "dd_montgomery_potential",
    ]
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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_new[name][nb:-nb, nb:-nb])


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
    nb = 1  # TODO: nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_ff(grid, moist=moist), label="state")

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
    imp = IsentropicBoussinesqMinimalPrognostic.factory(
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
    compare_datetimes(raw_state_new["time"], raw_state["time"] + timestep)

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
        "dd_montgomery_potential",
    ]
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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_new[name][nb:-nb, nb:-nb])


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
    nb = 2  # TODO: nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_ff(grid, moist=moist), label="state")

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
    imp = IsentropicBoussinesqMinimalPrognostic.factory(
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

    names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
        "dd_montgomery_potential",
    ]
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
    compare_datetimes(raw_state_1["time"], raw_state["time"] + 0.5 * timestep)

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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_1[name][nb:-nb, nb:-nb])

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
    compare_datetimes(raw_state_2["time"], raw_state_1["time"] + 0.5 * timestep)

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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_2[name][nb:-nb, nb:-nb])


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
    nb = 3  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_ff(grid, moist=moist), label="state")

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
    imp = IsentropicBoussinesqMinimalPrognostic.factory(
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

    names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
        "dd_montgomery_potential",
    ]
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
    compare_datetimes(raw_state_1["time"], raw_state["time"] + 1.0 / 3.0 * timestep)

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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_1[name][nb:-nb, nb:-nb])

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
    compare_datetimes(raw_state_2["time"], raw_state_1["time"] + 1.0 / 6.0 * timestep)

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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_2[name][nb:-nb, nb:-nb])

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
    compare_datetimes(raw_state_3["time"], raw_state_2["time"] + 0.5 * timestep)

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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_3[name][nb:-nb, nb:-nb])


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
    nb = 3  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb))
    domain = data.draw(
        st_domain(xaxis_length=(7, 30), yaxis_length=(7, 30), nb=nb), label="domain"
    )

    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_ff(grid, moist=moist), label="state")

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
    imp = IsentropicBoussinesqMinimalPrognostic.factory(
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

    names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
        "dd_montgomery_potential",
    ]
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
    compare_datetimes(raw_state_1["time"], raw_state["time"] + a1 * timestep)

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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_1[name][nb:-nb, nb:-nb])
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
    compare_datetimes(raw_state_2["time"], raw_state_1["time"] + (a2 - a1) * timestep)

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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_2[name][nb:-nb, nb:-nb])
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
    compare_datetimes(raw_state_3["time"], raw_state_2["time"] + (1 - a2 - a1) * timestep)

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
        compare_arrays(phi_out[nb:-nb, nb:-nb], raw_state_3[name][nb:-nb, nb:-nb])


if __name__ == "__main__":
    pytest.main([__file__])
