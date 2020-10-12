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
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.isentropic.dynamics.diagnostics import IsentropicDiagnostics
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicMinimalHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic
from tasmania.python.isentropic.dynamics.subclasses.prognostics import (
    ForwardEulerSI,
    CenteredSI,
    RK3WSSI,
)
from tasmania.python.utils.storage_utils import (
    deepcopy_array_dict,
    get_array_dict,
    zeros,
)

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.isentropic.test_isentropic_horizontal_fluxes import (
    get_upwind_fluxes,
    get_fifth_order_upwind_fluxes,
)
from tests.strategies import st_domain, st_floats, st_one_of, st_isentropic_state_f
from tests.utilities import compare_arrays, compare_datetimes, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def forward_euler_step(
    get_fluxes, mode, dx, dy, dt, u_tmp, v_tmp, phi, phi_tmp, phi_tnd, phi_out
):
    flux_x, flux_y = get_fluxes(u_tmp, v_tmp, phi_tmp)
    phi_out[1:-1, 1:-1] = phi[1:-1, 1:-1] - dt * (
        ((flux_x[2:, 1:-1] - flux_x[1:-1, 1:-1]) / dx if mode != "y" else 0.0)
        + ((flux_y[1:-1, 2:] - flux_y[1:-1, 1:-1]) / dy if mode != "x" else 0.0)
        - (phi_tnd[1:-1, 1:-1] if phi_tnd is not None else 0.0)
    )


def forward_euler_step_momentum_x(
    get_fluxes,
    mode,
    eps,
    dx,
    dy,
    dt,
    s,
    s_new,
    u_tmp,
    v_tmp,
    mtg,
    mtg_new,
    su,
    su_tmp,
    su_tnd,
    su_out,
):
    flux_x, flux_y = get_fluxes(u_tmp, v_tmp, su_tmp)
    su_out[1:-1, 1:-1] = su[1:-1, 1:-1] - dt * (
        ((flux_x[2:, 1:-1] - flux_x[1:-1, 1:-1]) / dx if mode != "y" else 0.0)
        + ((flux_y[1:-1, 2:] - flux_y[1:-1, 1:-1]) / dy if mode != "x" else 0.0)
        + (1 - eps) * s[1:-1, 1:-1] * (mtg[2:, 1:-1] - mtg[:-2, 1:-1]) / (2.0 * dx)
        + eps * s_new[1:-1, 1:-1] * (mtg_new[2:, 1:-1] - mtg_new[:-2, 1:-1]) / (2.0 * dx)
        - (su_tnd[1:-1, 1:-1] if su_tnd is not None else 0.0)
    )


def forward_euler_step_momentum_y(
    get_fluxes,
    mode,
    eps,
    dx,
    dy,
    dt,
    s,
    s_new,
    u_tmp,
    v_tmp,
    mtg,
    mtg_new,
    sv,
    sv_tmp,
    sv_tnd,
    sv_out,
):
    flux_x, flux_y = get_fluxes(u_tmp, v_tmp, sv_tmp)
    sv_out[1:-1, 1:-1] = sv[1:-1, 1:-1] - dt * (
        ((flux_x[2:, 1:-1] - flux_x[1:-1, 1:-1]) / dx if mode != "y" else 0.0)
        + ((flux_y[1:-1, 2:] - flux_y[1:-1, 1:-1]) / dy if mode != "x" else 0.0)
        + (1 - eps) * s[1:-1, 1:-1] * (mtg[1:-1, 2:] - mtg[1:-1, :-2]) / (2.0 * dy)
        + eps * s_new[1:-1, 1:-1] * (mtg_new[1:-1, 2:] - mtg_new[1:-1, :-2]) / (2.0 * dy)
        - (sv_tnd[1:-1, 1:-1] if sv_tnd is not None else 0.0)
    )


def test_registry():
    assert "forward_euler_si" in IsentropicPrognostic.registry
    assert IsentropicPrognostic.registry["forward_euler_si"] == ForwardEulerSI
    assert "centered_si" in IsentropicPrognostic.registry
    assert IsentropicPrognostic.registry["centered_si"] == CenteredSI
    assert "rk3ws_si" in IsentropicPrognostic.registry
    assert IsentropicPrognostic.registry["rk3ws_si"] == RK3WSSI


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_factory(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=3,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    moist = data.draw(hyp_st.booleans(), label="moist")
    pt = DataArray(
        data.draw(st_floats(min_value=0, max_value=100), label="pt"),
        attrs={"units": "Pa"},
    )
    eps = data.draw(st_floats(min_value=0, max_value=1), label="eps")

    # ========================================
    # test bed
    # ========================================
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    dtype = grid.x.dtype

    imp_euler_si = IsentropicPrognostic.factory(
        "forward_euler_si",
        "upwind",
        grid,
        hb,
        moist,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        pt=pt,
        eps=eps,
    )
    assert isinstance(imp_euler_si, ForwardEulerSI)
    assert isinstance(imp_euler_si._hflux, IsentropicMinimalHorizontalFlux)

    imp_rk3ws_si = IsentropicPrognostic.factory(
        "rk3ws_si",
        "fifth_order_upwind",
        grid,
        hb,
        moist,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        pt=pt,
        eps=eps,
    )
    assert isinstance(imp_rk3ws_si, RK3WSSI)
    assert isinstance(imp_rk3ws_si._hflux, IsentropicMinimalHorizontalFlux)

    # imp_sil3 = IsentropicPrognostic.factory(
    # 	'sil3', 'fifth_order_upwind', grid, hb, moist,
    # 	backend=backend, dtype=dtype, default_origin=default_origin, pt=pt, a=a, b=b, c=c
    # )
    # assert isinstance(imp_sil3, SIL3)
    # assert isinstance(imp_sil3._hflux, IsentropicMinimalHorizontalFlux)
    # assert np.isclose(imp_sil3._a.value, a)
    # assert np.isclose(imp_sil3._b.value, b)
    # assert np.isclose(imp_sil3._c.value, c)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_forward_euler_si(data, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(2, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    hb = domain.horizontal_boundary
    grid = domain.numerical_grid

    moist = data.draw(hyp_st.booleans(), label="moist")

    storage_shape = (grid.nx + 1, grid.ny + 1, grid.nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=moist,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd_on"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="su_tnd_on"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="sv_tn_on"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd_on"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="qc_tnd_on"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="qr_tnd_on"):
            tendencies[mfpw] = state[mfpw]

    pt = state["air_pressure_on_interface_levels"][0, 0, 0]
    eps = data.draw(st_floats(min_value=0, max_value=1), label="eps")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    hb.reference_state = state

    imp = IsentropicPrognostic.factory(
        "forward_euler_si",
        "upwind",
        grid,
        hb,
        moist,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
        storage_shape=(grid.nx + 1, grid.ny + 1, grid.nz + 1),
        pt=pt,
        eps=eps,
    )

    raw_state = get_array_dict(state, properties={})
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

    raw_tendencies = get_array_dict(tendencies, properties={})
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
    u_now = raw_state["x_velocity_at_u_locations"]
    v_now = raw_state["y_velocity_at_v_locations"]

    # isentropic density
    s_now = raw_state["air_isentropic_density"]
    s_tnd = raw_tendencies.get("air_isentropic_density", None)
    s_new = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_upwind_fluxes, "xy", dx, dy, dt, u_now, v_now, s_now, s_now, s_tnd, s_new
    )
    hb.dmn_enforce_field(
        s_new, "air_isentropic_density", "kg m^-2 K^-1", time=state["time"] + timestep
    )
    assert "air_isentropic_density" in raw_state_new
    compare_arrays(
        s_new[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_new["air_isentropic_density"][nb : -nb - 1, nb : -nb - 1, :-1],
    )

    if moist:
        # water species
        names = [
            "isentropic_density_of_water_vapor",
            "isentropic_density_of_cloud_liquid_water",
            "isentropic_density_of_precipitation_water",
        ]
        sq_new = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        for name in names:
            sq_now = raw_state[name]
            sq_tnd = raw_tendencies.get(name, None)
            forward_euler_step(
                get_upwind_fluxes,
                "xy",
                dx,
                dy,
                dt,
                u_now,
                v_now,
                sq_now,
                sq_now,
                sq_tnd,
                sq_new,
            )
            # with subtests.test(name=name):
            assert name in raw_state_new
            compare_arrays(
                sq_new[nb : -nb - 1, nb : -nb - 1, :-1],
                raw_state_new[name][nb : -nb - 1, nb : -nb - 1, :-1],
            )

    # montgomery potential
    ids = IsentropicDiagnostics(
        grid,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    mtg_new = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    ids.get_montgomery_potential(s_new, pt.values.item(), mtg_new)
    compare_arrays(mtg_new, imp._mtg_new)

    # x-momentum
    mtg_now = raw_state["montgomery_potential"]
    su_now = raw_state["x_momentum_isentropic"]
    su_tnd = raw_tendencies.get("x_momentum_isentropic", None)
    su_new = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_upwind_fluxes, "xy", dx, dy, dt, u_now, v_now, su_now, su_now, su_tnd, su_new
    )
    su_new[1:-1, 1:-1] -= dt * (
        (1 - eps)
        * s_now[1:-1, 1:-1]
        * (mtg_now[2:, 1:-1] - mtg_now[:-2, 1:-1])
        / (2.0 * dx)
        + eps * s_new[1:-1, 1:-1] * (mtg_new[2:, 1:-1] - mtg_new[:-2, 1:-1]) / (2.0 * dx)
    )
    assert "x_momentum_isentropic" in raw_state_new
    compare_arrays(
        su_new[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_new["x_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
    )

    # y-momentum
    sv_now = raw_state["y_momentum_isentropic"]
    sv_tnd = raw_tendencies.get("y_momentum_isentropic", None)
    sv_new = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_upwind_fluxes, "xy", dx, dy, dt, u_now, v_now, sv_now, sv_now, sv_tnd, sv_new
    )
    sv_new[1:-1, 1:-1] -= dt * (
        (1 - eps)
        * s_now[1:-1, 1:-1]
        * (mtg_now[1:-1, 2:] - mtg_now[1:-1, :-2])
        / (2.0 * dy)
        + eps * s_new[1:-1, 1:-1] * (mtg_new[1:-1, 2:] - mtg_new[1:-1, :-2]) / (2.0 * dy)
    )
    assert "y_momentum_isentropic" in raw_state_new
    compare_arrays(
        sv_new[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_new["y_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_rk3ws_si(data, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(2, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    hb = domain.horizontal_boundary
    grid = domain.numerical_grid

    moist = data.draw(hyp_st.booleans(), label="moist")

    storage_shape = (grid.nx + 1, grid.ny + 1, grid.nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=moist,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd_on"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="su_tnd_on"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="sv_tn_on"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd_on"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="qc_tnd_on"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="qr_tnd_on"):
            tendencies[mfpw] = state[mfpw]

    pt = state["air_pressure_on_interface_levels"][0, 0, 0]
    eps = data.draw(st_floats(min_value=0, max_value=1), label="eps")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    hb.reference_state = state

    imp = IsentropicPrognostic.factory(
        "rk3ws_si",
        "fifth_order_upwind",
        grid,
        hb,
        moist,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
        storage_shape=storage_shape,
        pt=pt,
        eps=eps,
    )

    raw_state = get_array_dict(state, properties={})
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

    raw_tendencies = get_array_dict(tendencies, properties={})
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

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    u = raw_state["x_velocity_at_u_locations"]
    v = raw_state["y_velocity_at_v_locations"]

    names = [
        "isentropic_density_of_water_vapor",
        "isentropic_density_of_cloud_liquid_water",
        "isentropic_density_of_precipitation_water",
    ]
    sq_new = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    #
    # stage 0
    #
    dts = (timestep / 3.0).total_seconds()

    raw_state_1 = imp.stage_call(0, timestep, raw_state, raw_tendencies)

    assert "time" in raw_state_1.keys()
    compare_datetimes(raw_state_1["time"], raw_state["time"] + 1.0 / 3.0 * timestep)

    # isentropic density
    s0 = raw_state["air_isentropic_density"]
    s_tnd = raw_tendencies.get("air_isentropic_density", None)
    s1 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, s0, s0, s_tnd, s1
    )
    hb.dmn_enforce_field(
        s1,
        "air_isentropic_density",
        "kg m^-2 K^-1",
        time=state["time"] + 1.0 / 3.0 * timestep,
    )
    assert "air_isentropic_density" in raw_state_1
    compare_arrays(
        s1[:nx, :ny, :nz], raw_state_1["air_isentropic_density"][:nx, :ny, :nz]
    )

    if moist:
        # water species
        for name in names:
            sq0 = raw_state[name]
            sq_tnd = raw_tendencies.get(name, None)
            forward_euler_step(
                get_fifth_order_upwind_fluxes,
                "xy",
                dx,
                dy,
                dts,
                u,
                v,
                sq0,
                sq0,
                sq_tnd,
                sq_new,
            )
            # with subtests.test(name=name):
            assert name in raw_state_1
            compare_arrays(
                sq_new[nb : -nb - 1, nb : -nb - 1, :-1],
                raw_state_1[name][nb : -nb - 1, nb : -nb - 1, :-1],
            )

    # montgomery potential
    ids = IsentropicDiagnostics(
        grid,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )
    mtg1 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    ids.get_montgomery_potential(s1, pt.to_units("Pa").values.item(), mtg1)
    compare_arrays(mtg1[:nx, :ny, :nz], imp._mtg_new[:nx, :ny, :nz])

    # x-momentum
    mtg0 = raw_state["montgomery_potential"]
    compare_arrays(mtg0, imp._mtg_now)
    su0 = raw_state["x_momentum_isentropic"]
    su_tnd = raw_tendencies.get("x_momentum_isentropic", None)
    su1 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, su0, su0, su_tnd, su1
    )
    su1[nb:-nb, nb:-nb] -= dts * (
        (1 - eps)
        * s0[nb:-nb, nb:-nb]
        * (mtg0[nb + 1 : -nb + 1, nb:-nb] - mtg0[nb - 1 : -nb - 1, nb:-nb])
        / (2.0 * dx)
        + eps
        * s1[nb:-nb, nb:-nb]
        * (mtg1[nb + 1 : -nb + 1, nb:-nb] - mtg1[nb - 1 : -nb - 1, nb:-nb])
        / (2.0 * dx)
    )
    assert "x_momentum_isentropic" in raw_state_1
    compare_arrays(
        su1[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_1["x_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
    )

    # y-momentum
    sv0 = raw_state["y_momentum_isentropic"]
    sv_tnd = raw_tendencies.get("y_momentum_isentropic", None)
    sv1 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, sv0, sv0, sv_tnd, sv1
    )
    sv1[nb:-nb, nb:-nb] -= dts * (
        (1 - eps)
        * s0[nb:-nb, nb:-nb]
        * (mtg0[nb:-nb, nb + 1 : -nb + 1] - mtg0[nb:-nb, nb - 1 : -nb - 1])
        / (2.0 * dy)
        + eps
        * s1[nb:-nb, nb:-nb]
        * (mtg1[nb:-nb, nb + 1 : -nb + 1] - mtg1[nb:-nb, nb - 1 : -nb - 1])
        / (2.0 * dy)
    )
    assert "y_momentum_isentropic" in raw_state_1
    compare_arrays(
        sv1[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_1["y_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
    )

    #
    # stage 1
    #
    raw_state_1["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_1["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    raw_state_1_dc = deepcopy_array_dict(raw_state_1)

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

    dts = (0.5 * timestep).total_seconds()

    raw_state_2 = imp.stage_call(1, timestep, raw_state_1_dc, raw_tendencies)

    assert "time" in raw_state_2.keys()
    compare_datetimes(raw_state_2["time"], raw_state["time"] + 0.5 * timestep)

    # isentropic density
    s2 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, s0, s1, s_tnd, s2
    )
    hb.dmn_enforce_field(
        s2, "air_isentropic_density", "kg m^-2 K^-1", time=state["time"] + 0.5 * timestep
    )
    assert "air_isentropic_density" in raw_state_2
    compare_arrays(
        s2[:nx, :ny, :nz], raw_state_2["air_isentropic_density"][:nx, :ny, :nz]
    )

    if moist:
        # water species
        for name in names:
            sq0 = raw_state[name]
            sq1 = raw_state_1_dc[name]
            sq_tnd = raw_tendencies.get(name, None)
            forward_euler_step(
                get_fifth_order_upwind_fluxes,
                "xy",
                dx,
                dy,
                dts,
                u,
                v,
                sq0,
                sq1,
                sq_tnd,
                sq_new,
            )
            # with subtests.test(name=name):
            assert name in raw_state_2
            compare_arrays(
                sq_new[nb : -nb - 1, nb : -nb - 1, :-1],
                raw_state_2[name][nb : -nb - 1, nb : -nb - 1, :-1],
            )

    # montgomery potential
    mtg2 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    ids.get_montgomery_potential(s2, pt.to_units("Pa").values.item(), mtg2)
    compare_arrays(mtg2[:nx, :ny, :nz], imp._mtg_new[:nx, :ny, :nz])

    # x-momentum
    su1 = raw_state_1_dc["x_momentum_isentropic"]
    su2 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, su0, su1, su_tnd, su2
    )
    su2[nb:-nb, nb:-nb] -= dts * (
        (1 - eps)
        * s0[nb:-nb, nb:-nb]
        * (mtg0[nb + 1 : -nb + 1, nb:-nb] - mtg0[nb - 1 : -nb - 1, nb:-nb])
        / (2.0 * dx)
        + eps
        * s2[nb:-nb, nb:-nb]
        * (mtg2[nb + 1 : -nb + 1, nb:-nb] - mtg2[nb - 1 : -nb - 1, nb:-nb])
        / (2.0 * dx)
    )
    assert "x_momentum_isentropic" in raw_state_2
    compare_arrays(
        su2[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_2["x_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
    )

    # y-momentum
    sv1 = raw_state_1_dc["y_momentum_isentropic"]
    sv2 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, sv0, sv1, sv_tnd, sv2
    )
    sv2[nb:-nb, nb:-nb] -= dts * (
        (1 - eps)
        * s0[nb:-nb, nb:-nb]
        * (mtg0[nb:-nb, nb + 1 : -nb + 1] - mtg0[nb:-nb, nb - 1 : -nb - 1])
        / (2.0 * dy)
        + eps
        * s2[nb:-nb, nb:-nb]
        * (mtg2[nb:-nb, nb + 1 : -nb + 1] - mtg2[nb:-nb, nb - 1 : -nb - 1])
        / (2.0 * dy)
    )
    assert "y_momentum_isentropic" in raw_state_2
    compare_arrays(
        sv2[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_2["y_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
    )

    #
    # stage 2
    #
    raw_state_2["x_velocity_at_u_locations"] = raw_state["x_velocity_at_u_locations"]
    raw_state_2["y_velocity_at_v_locations"] = raw_state["y_velocity_at_v_locations"]
    raw_state_2_dc = deepcopy_array_dict(raw_state_2)

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

    dts = timestep.total_seconds()

    raw_state_3 = imp.stage_call(2, timestep, raw_state_2_dc, raw_tendencies)

    assert "time" in raw_state_3.keys()
    compare_datetimes(raw_state_3["time"], raw_state["time"] + timestep)

    # isentropic density
    s3 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, s0, s2, s_tnd, s3
    )
    hb.dmn_enforce_field(
        s3, "air_isentropic_density", "kg m^-2 K^-1", time=state["time"] + timestep
    )
    assert "air_isentropic_density" in raw_state_3
    compare_arrays(
        s3[:nx, :ny, :nz], raw_state_3["air_isentropic_density"][:nx, :ny, :nz]
    )

    if moist:
        # water species
        for name in names:
            sq0 = raw_state[name]
            sq2 = raw_state_2_dc[name]
            sq_tnd = raw_tendencies.get(name, None)
            forward_euler_step(
                get_fifth_order_upwind_fluxes,
                "xy",
                dx,
                dy,
                dts,
                u,
                v,
                sq0,
                sq2,
                sq_tnd,
                sq_new,
            )
            # with subtests.test(name=name):
            assert name in raw_state_3
            compare_arrays(
                sq_new[nb : -nb - 1, nb : -nb - 1, :-1],
                raw_state_3[name][nb : -nb - 1, nb : -nb - 1, :-1],
            )

    # montgomery potential
    mtg3 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    ids.get_montgomery_potential(s3, pt.to_units("Pa").values.item(), mtg3)
    compare_arrays(mtg3[:nx, :ny, :nz], imp._mtg_new[:nx, :ny, :nz])

    # x-momentum
    su2 = raw_state_2_dc["x_momentum_isentropic"]
    su3 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, su0, su2, su_tnd, su3
    )
    su3[nb:-nb, nb:-nb] -= dts * (
        (1 - eps)
        * s0[nb:-nb, nb:-nb]
        * (mtg0[nb + 1 : -nb + 1, nb:-nb] - mtg0[nb - 1 : -nb - 1, nb:-nb])
        / (2.0 * dx)
        + eps
        * s3[nb:-nb, nb:-nb]
        * (mtg3[nb + 1 : -nb + 1, nb:-nb] - mtg3[nb - 1 : -nb - 1, nb:-nb])
        / (2.0 * dx)
    )
    assert "x_momentum_isentropic" in raw_state_3
    compare_arrays(
        su3[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_3["x_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
    )

    # y-momentum
    sv2 = raw_state_2_dc["y_momentum_isentropic"]
    sv3 = zeros(
        storage_shape,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    forward_euler_step(
        get_fifth_order_upwind_fluxes, "xy", dx, dy, dts, u, v, sv0, sv2, sv_tnd, sv3
    )
    sv3[nb:-nb, nb:-nb] -= dts * (
        (1 - eps)
        * s0[nb:-nb, nb:-nb]
        * (mtg0[nb:-nb, nb + 1 : -nb + 1] - mtg0[nb:-nb, nb - 1 : -nb - 1])
        / (2.0 * dy)
        + eps
        * s3[nb:-nb, nb:-nb]
        * (mtg3[nb:-nb, nb + 1 : -nb + 1] - mtg3[nb:-nb, nb - 1 : -nb - 1])
        / (2.0 * dy)
    )
    assert "y_momentum_isentropic" in raw_state_3
    compare_arrays(
        sv3[nb : -nb - 1, nb : -nb - 1, :-1],
        raw_state_3["y_momentum_isentropic"][nb : -nb - 1, nb : -nb - 1, :-1],
    )


# @settings(
#     suppress_health_check=(
#         HealthCheck.too_slow,
#         HealthCheck.data_too_large,
#         HealthCheck.filter_too_much,
#     ),
#     deadline=None,
# )
# @given(hyp_st.data())
# def _test_sil3(data):
#     # ========================================
#     # random data generation
#     # ========================================
#     # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb))
#     nb = 3
#     domain = data.draw(
#         st_domain(xaxis_length=(7, 50), yaxis_length=(7, 50), nb=nb), label="domain"
#     )
#     hb = domain.horizontal_boundary
#     grid = domain.numerical_grid
#     moist = data.draw(hyp_st.booleans(), label="moist")
#     state = data.draw(st_isentropic_state_f(grid, moist=moist), label="state")
#     tendencies = {}
#     if data.draw(hyp_st.booleans(), label="s_tnd"):
#         tendencies["air_isentropic_density"] = state["air_isentropic_density"]
#     if data.draw(hyp_st.booleans(), label="su_tnd"):
#         tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
#     if data.draw(hyp_st.booleans(), label="sv_tn:"):
#         tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
#     if moist:
#         if data.draw(hyp_st.booleans(), label="qv_tnd"):
#             tendencies[mfwv] = state[mfwv]
#         if data.draw(hyp_st.booleans(), label="qc_tnd"):
#             tendencies[mfcw] = state[mfcw]
#         if data.draw(hyp_st.booleans(), label="qr_tnd"):
#             tendencies[mfpw] = state[mfpw]
#
#     backend = data.draw(st_one_of(conf_backend), label="backend")
#     pt = state["air_pressure_on_interface_levels"][0, 0, 0]
#     a = data.draw(st_floats(min_value=0, max_value=1), label="a")
#     b = data.draw(st_floats(min_value=0, max_value=1), label="b")
#     c = data.draw(st_floats(min_value=0, max_value=1), label="c")
#
#     timestep = data.draw(
#         hyp_st.timedeltas(
#             min_value=timedelta(seconds=1), max_value=timedelta(minutes=60)
#         ),
#         label="timestep",
#     )
#
#     # ========================================
#     # test bed
#     # ========================================
#     hb.reference_state = state
#     dtype = grid.x.dtype
#
#     imp = IsentropicPrognostic.factory(
#         "sil3",
#         "fifth_order_upwind",
#         grid,
#         hb,
#         moist,
#         backend=backend,
#         dtype=dtype,
#         pt=pt,
#         a=a,
#         b=b,
#         c=c,
#     )
#
#     raw_state = get_array_dict(state)
#     if moist:
#         raw_state["isentropic_density_of_water_vapor"] = (
#             raw_state["air_isentropic_density"] * raw_state[mfwv]
#         )
#         raw_state["isentropic_density_of_cloud_liquid_water"] = (
#             raw_state["air_isentropic_density"] * raw_state[mfcw]
#         )
#         raw_state["isentropic_density_of_precipitation_water"] = (
#             raw_state["air_isentropic_density"] * raw_state[mfpw]
#         )
#
#     raw_tendencies = get_array_dict(tendencies)
#
#     nx, ny, nz = grid.nx, grid.ny, grid.nz
#     dx = grid.dx.to_units("m").values.item()
#     dy = grid.dy.to_units("m").values.item()
#     u = raw_state["x_velocity_at_u_locations"]
#     v = raw_state["y_velocity_at_v_locations"]
#
#     dts = timestep.total_seconds()
#
#     #
#     # stage 0
#     #
#     raw_state_1 = imp.stage_call(0, timestep, raw_state, raw_tendencies)
#
#     assert "time" in raw_state_1.keys()
#     compare_datetimes(raw_state_1["time"], raw_state["time"] + 1.0 / 3.0 * timestep)
#
#     # isentropic density
#     s0 = raw_state["air_isentropic_density"]
#     s_tnd = raw_tendencies.get("air_isentropic_density", None)
#     flux_s_x_0, flux_s_y_0 = get_fifth_order_upwind_fluxes(u, v, s0)
#     s1 = np.zeros((nx, ny, nz), dtype=dtype)
#     s1[nb:-nb, nb:-nb] = s0[nb:-nb, nb:-nb] - dts / 3.0 * (
#         (flux_s_x_0[nb:-nb, nb:-nb] - flux_s_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#         + (flux_s_y_0[nb:-nb, nb:-nb] - flux_s_y_0[nb:-nb, nb - 1 : -nb - 1]) / dy
#         - (s_tnd[nb:-nb, nb:-nb] if s_tnd is not None else 0.0)
#     )
#     hb.dmn_enforce_field(
#         s1,
#         "air_isentropic_density",
#         "kg m^-2 K^-1",
#         time=state["time"] + 1.0 / 3.0 * timestep,
#     )
#     assert "air_isentropic_density" in raw_state_1
#     compare_arrays(s1, raw_state_1["air_isentropic_density"])
#
#     if moist:
#         # isentropic density of water vapor
#         sqv0 = raw_state["isentropic_density_of_water_vapor"]
#         qv_tnd = raw_tendencies.get(mfwv, None)
#         flux_sqv_x_0, flux_sqv_y_0 = get_fifth_order_upwind_fluxes(u, v, sqv0)
#         sqv1 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqv1[nb:-nb, nb:-nb] = sqv0[nb:-nb, nb:-nb] - dts / 3.0 * (
#             (flux_sqv_x_0[nb:-nb, nb:-nb] - flux_sqv_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sqv_y_0[nb:-nb, nb:-nb] - flux_sqv_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - (
#                 s0[nb:-nb, nb:-nb] * qv_tnd[nb:-nb, nb:-nb]
#                 if qv_tnd is not None
#                 else 0.0
#             )
#         )
#         assert "isentropic_density_of_water_vapor" in raw_state_1
#         compare_arrays(sqv1, raw_state_1["isentropic_density_of_water_vapor"])
#
#         # isentropic density of cloud liquid water
#         sqc0 = raw_state["isentropic_density_of_cloud_liquid_water"]
#         qc_tnd = raw_tendencies.get(mfcw, None)
#         flux_sqc_x_0, flux_sqc_y_0 = get_fifth_order_upwind_fluxes(u, v, sqc0)
#         sqc1 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqc1[nb:-nb, nb:-nb] = sqc0[nb:-nb, nb:-nb] - dts / 3.0 * (
#             (flux_sqc_x_0[nb:-nb, nb:-nb] - flux_sqc_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sqc_y_0[nb:-nb, nb:-nb] - flux_sqc_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - (
#                 s0[nb:-nb, nb:-nb] * qc_tnd[nb:-nb, nb:-nb]
#                 if qc_tnd is not None
#                 else 0.0
#             )
#         )
#         assert "isentropic_density_of_cloud_liquid_water" in raw_state_1
#         compare_arrays(sqc1, raw_state_1["isentropic_density_of_cloud_liquid_water"])
#
#         # isentropic density of precipitation water
#         sqr0 = raw_state["isentropic_density_of_precipitation_water"]
#         qr_tnd = raw_tendencies.get(mfpw, None)
#         flux_sqr_x_0, flux_sqr_y_0 = get_fifth_order_upwind_fluxes(u, v, sqr0)
#         sqr1 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqr1[nb:-nb, nb:-nb] = sqr0[nb:-nb, nb:-nb] - dts / 3.0 * (
#             (flux_sqr_x_0[nb:-nb, nb:-nb] - flux_sqr_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sqr_y_0[nb:-nb, nb:-nb] - flux_sqr_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - (
#                 s0[nb:-nb, nb:-nb] * qr_tnd[nb:-nb, nb:-nb]
#                 if qr_tnd is not None
#                 else 0.0
#             )
#         )
#         assert "isentropic_density_of_precipitation_water" in raw_state_1
#         compare_arrays(sqr1, raw_state_1["isentropic_density_of_precipitation_water"])
#
#     # montgomery potential
#     ids = IsentropicDiagnostics(grid, backend, dtype)
#     mtg1 = np.zeros((nx, ny, nz), dtype=dtype)
#     ids.get_montgomery_potential(s1, pt.to_units("Pa").values.item(), mtg1)
#     compare_arrays(mtg1, imp._mtg1)
#
#     # x-momentum
#     mtg0 = raw_state["montgomery_potential"]
#     compare_arrays(mtg0, imp._mtg_now)
#     su0 = raw_state["x_momentum_isentropic"]
#     su_tnd = raw_tendencies.get("x_momentum_isentropic", None)
#     flux_su_x_0, flux_su_y_0 = get_fifth_order_upwind_fluxes(u, v, su0)
#     su1 = np.zeros((nx, ny, nz), dtype=dtype)
#     su1[nb:-nb, nb:-nb] = su0[nb:-nb, nb:-nb] - dts * (
#         1.0
#         / 3.0
#         * (
#             (flux_su_x_0[nb:-nb, nb:-nb] - flux_su_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_su_y_0[nb:-nb, nb:-nb] - flux_su_y_0[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (su_tnd[nb:-nb, nb:-nb] if su_tnd is not None else 0.0)
#         )
#         + 1.0
#         / 6.0
#         * s0[nb:-nb, nb:-nb]
#         * (mtg0[nb + 1 : -nb + 1, nb:-nb] - mtg0[nb - 1 : -nb - 1, nb:-nb])
#         / (2.0 * dx)
#         + 1.0
#         / 6.0
#         * s1[nb:-nb, nb:-nb]
#         * (mtg1[nb + 1 : -nb + 1, nb:-nb] - mtg1[nb - 1 : -nb - 1, nb:-nb])
#         / (2.0 * dx)
#     )
#     assert "x_momentum_isentropic" in raw_state_1
#     compare_arrays(su1, raw_state_1["x_momentum_isentropic"])
#
#     # y-momentum
#     sv0 = raw_state["y_momentum_isentropic"]
#     sv_tnd = raw_tendencies.get("y_momentum_isentropic", None)
#     flux_sv_x_0, flux_sv_y_0 = get_fifth_order_upwind_fluxes(u, v, sv0)
#     sv1 = np.zeros((nx, ny, nz), dtype=dtype)
#     sv1[nb:-nb, nb:-nb] = sv0[nb:-nb, nb:-nb] - dts * (
#         1.0
#         / 3.0
#         * (
#             (flux_sv_x_0[nb:-nb, nb:-nb] - flux_sv_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sv_y_0[nb:-nb, nb:-nb] - flux_sv_y_0[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (sv_tnd[nb:-nb, nb:-nb] if sv_tnd is not None else 0.0)
#         )
#         + 1.0
#         / 6.0
#         * s0[nb:-nb, nb:-nb]
#         * (mtg0[nb:-nb, nb + 1 : -nb + 1] - mtg0[nb:-nb, nb - 1 : -nb - 1])
#         / (2.0 * dy)
#         + 1.0
#         / 6.0
#         * s1[nb:-nb, nb:-nb]
#         * (mtg1[nb:-nb, nb + 1 : -nb + 1] - mtg1[nb:-nb, nb - 1 : -nb - 1])
#         / (2.0 * dy)
#     )
#     assert "y_momentum_isentropic" in raw_state_1
#     compare_arrays(sv1, raw_state_1["y_momentum_isentropic"])
#
#     #
#     # stage 1
#     #
#     raw_state_1["x_velocity_at_u_locations"] = u
#     raw_state_1["y_velocity_at_v_locations"] = v
#     raw_state_1["montgomery_potential"] = mtg1
#
#     raw_state_2 = imp.stage_call(1, timestep, raw_state_1, raw_tendencies)
#
#     assert "time" in raw_state_2.keys()
#     compare_datetimes(raw_state_2["time"], raw_state["time"] + 2.0 / 3.0 * timestep)
#
#     # isentropic density
#     flux_s_x_1, flux_s_y_1 = get_fifth_order_upwind_fluxes(u, v, s1)
#     s2 = np.zeros((nx, ny, nz), dtype=dtype)
#     s2[nb:-nb, nb:-nb] = s0[nb:-nb, nb:-nb] - dts * (
#         1.0
#         / 6.0
#         * (flux_s_x_0[nb:-nb, nb:-nb] - flux_s_x_0[nb - 1 : -nb - 1, nb:-nb])
#         / dx
#         + 1.0
#         / 6.0
#         * (flux_s_y_0[nb:-nb, nb:-nb] - flux_s_y_0[nb:-nb, nb - 1 : -nb - 1])
#         / dy
#         + 0.5 * (flux_s_x_1[nb:-nb, nb:-nb] - flux_s_x_1[nb - 1 : -nb - 1, nb:-nb]) / dx
#         + 0.5 * (flux_s_y_1[nb:-nb, nb:-nb] - flux_s_y_1[nb:-nb, nb - 1 : -nb - 1]) / dy
#         - 2.0 / 3.0 * (s_tnd[nb:-nb, nb:-nb] if s_tnd is not None else 0.0)
#     )
#     hb.dmn_enforce_field(
#         s2,
#         "air_isentropic_density",
#         "kg m^-2 K^-1",
#         time=state["time"] + 2.0 / 3.0 * timestep,
#     )
#     assert "air_isentropic_density" in raw_state_2
#     compare_arrays(s2, raw_state_2["air_isentropic_density"])
#
#     if moist:
#         # isentropic density of water vapor
#         flux_sqv_x_1, flux_sqv_y_1 = get_fifth_order_upwind_fluxes(u, v, sqv1)
#         sqv2 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqv2[nb:-nb, nb:-nb] = sqv0[nb:-nb, nb:-nb] - dts * (
#             1.0
#             / 6.0
#             * (flux_sqv_x_0[nb:-nb, nb:-nb] - flux_sqv_x_0[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 1.0
#             / 6.0
#             * (flux_sqv_y_0[nb:-nb, nb:-nb] - flux_sqv_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 1.0 / 6.0 * ((s0 * qv_tnd)[nb:-nb, nb:-nb] if qv_tnd is not None else 0.0)
#             + 0.5
#             * (flux_sqv_x_1[nb:-nb, nb:-nb] - flux_sqv_x_1[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 0.5
#             * (flux_sqv_y_1[nb:-nb, nb:-nb] - flux_sqv_y_1[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 0.5 * ((s1 * qv_tnd)[nb:-nb, nb:-nb] if qv_tnd is not None else 0.0)
#         )
#         assert "isentropic_density_of_water_vapor" in raw_state_2
#         compare_arrays(sqv2, raw_state_2["isentropic_density_of_water_vapor"])
#
#         # isentropic density of cloud liquid water
#         flux_sqc_x_1, flux_sqc_y_1 = get_fifth_order_upwind_fluxes(u, v, sqc1)
#         sqc2 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqc2[nb:-nb, nb:-nb] = sqc0[nb:-nb, nb:-nb] - dts * (
#             1.0
#             / 6.0
#             * (flux_sqc_x_0[nb:-nb, nb:-nb] - flux_sqc_x_0[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 1.0
#             / 6.0
#             * (flux_sqc_y_0[nb:-nb, nb:-nb] - flux_sqc_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 1.0 / 6.0 * ((s0 * qc_tnd)[nb:-nb, nb:-nb] if qc_tnd is not None else 0.0)
#             + 0.5
#             * (flux_sqc_x_1[nb:-nb, nb:-nb] - flux_sqc_x_1[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 0.5
#             * (flux_sqc_y_1[nb:-nb, nb:-nb] - flux_sqc_y_1[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 0.5 * ((s1 * qc_tnd)[nb:-nb, nb:-nb] if qc_tnd is not None else 0.0)
#         )
#         assert "isentropic_density_of_cloud_liquid_water" in raw_state_2
#         compare_arrays(sqc2, raw_state_2["isentropic_density_of_cloud_liquid_water"])
#
#         # isentropic density of precipitation water
#         flux_sqr_x_1, flux_sqr_y_1 = get_fifth_order_upwind_fluxes(u, v, sqr1)
#         sqr2 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqr2[nb:-nb, nb:-nb] = sqr0[nb:-nb, nb:-nb] - dts * (
#             1.0
#             / 6.0
#             * (flux_sqr_x_0[nb:-nb, nb:-nb] - flux_sqr_x_0[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 1.0
#             / 6.0
#             * (flux_sqr_y_0[nb:-nb, nb:-nb] - flux_sqr_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 1.0 / 6.0 * ((s0 * qr_tnd)[nb:-nb, nb:-nb] if qr_tnd is not None else 0.0)
#             + 0.5
#             * (flux_sqr_x_1[nb:-nb, nb:-nb] - flux_sqr_x_1[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 0.5
#             * (flux_sqr_y_1[nb:-nb, nb:-nb] - flux_sqr_y_1[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 0.5 * ((s1 * qr_tnd)[nb:-nb, nb:-nb] if qr_tnd is not None else 0.0)
#         )
#         assert "isentropic_density_of_precipitation_water" in raw_state_2
#         compare_arrays(sqr2, raw_state_2["isentropic_density_of_precipitation_water"])
#
#     # montgomery potential
#     mtg2 = mtg1
#     ids.get_montgomery_potential(s2, pt.to_units("Pa").values.item(), mtg2)
#     compare_arrays(mtg2, imp._mtg2)
#
#     # x-momentum
#     flux_su_x_1, flux_su_y_1 = get_fifth_order_upwind_fluxes(u, v, su1)
#     su2 = np.zeros((nx, ny, nz), dtype=dtype)
#     su2[nb:-nb, nb:-nb] = su0[nb:-nb, nb:-nb] - dts * (
#         1.0
#         / 6.0
#         * (
#             (flux_su_x_0[nb:-nb, nb:-nb] - flux_su_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_su_y_0[nb:-nb, nb:-nb] - flux_su_y_0[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (su_tnd[nb:-nb, nb:-nb] if su_tnd is not None else 0.0)
#         )
#         + 0.5
#         * (
#             (flux_su_x_1[nb:-nb, nb:-nb] - flux_su_x_1[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_su_y_1[nb:-nb, nb:-nb] - flux_su_y_1[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (su_tnd[nb:-nb, nb:-nb] if su_tnd is not None else 0.0)
#         )
#         + 1.0
#         / 3.0
#         * s0[nb:-nb, nb:-nb]
#         * (mtg0[nb + 1 : -nb + 1, nb:-nb] - mtg0[nb - 1 : -nb - 1, nb:-nb])
#         / (2.0 * dx)
#         + 1.0
#         / 3.0
#         * s2[nb:-nb, nb:-nb]
#         * (mtg2[nb + 1 : -nb + 1, nb:-nb] - mtg2[nb - 1 : -nb - 1, nb:-nb])
#         / (2.0 * dx)
#     )
#     assert "x_momentum_isentropic" in raw_state_2
#     compare_arrays(su2, raw_state_2["x_momentum_isentropic"])
#
#     # y-momentum
#     flux_sv_x_1, flux_sv_y_1 = get_fifth_order_upwind_fluxes(u, v, sv1)
#     sv2 = np.zeros((nx, ny, nz), dtype=dtype)
#     sv2[nb:-nb, nb:-nb] = sv0[nb:-nb, nb:-nb] - dts * (
#         1.0
#         / 6.0
#         * (
#             (flux_sv_x_0[nb:-nb, nb:-nb] - flux_sv_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sv_y_0[nb:-nb, nb:-nb] - flux_sv_y_0[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (sv_tnd[nb:-nb, nb:-nb] if sv_tnd is not None else 0.0)
#         )
#         + 0.5
#         * (
#             (flux_sv_x_1[nb:-nb, nb:-nb] - flux_sv_x_1[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sv_y_1[nb:-nb, nb:-nb] - flux_sv_y_1[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (sv_tnd[nb:-nb, nb:-nb] if sv_tnd is not None else 0.0)
#         )
#         + 1.0
#         / 3.0
#         * s0[nb:-nb, nb:-nb]
#         * (mtg0[nb:-nb, nb + 1 : -nb + 1] - mtg0[nb:-nb, nb - 1 : -nb - 1])
#         / (2.0 * dy)
#         + 1.0
#         / 3.0
#         * s2[nb:-nb, nb:-nb]
#         * (mtg2[nb:-nb, nb + 1 : -nb + 1] - mtg2[nb:-nb, nb - 1 : -nb - 1])
#         / (2.0 * dy)
#     )
#     assert "y_momentum_isentropic" in raw_state_2
#     compare_arrays(sv2, raw_state_2["y_momentum_isentropic"])
#
#     #
#     # stage 2
#     #
#     raw_state_2["x_velocity_at_u_locations"] = u
#     raw_state_2["y_velocity_at_v_locations"] = v
#     raw_state_2["montgomery_potential"] = mtg2
#
#     raw_state_3 = imp.stage_call(2, timestep, raw_state_2, raw_tendencies)
#
#     assert "time" in raw_state_3.keys()
#     compare_datetimes(raw_state_3["time"], raw_state["time"] + timestep)
#
#     # isentropic density
#     flux_s_x_2, flux_s_y_2 = get_fifth_order_upwind_fluxes(u, v, s2)
#     s3 = np.zeros((nx, ny, nz), dtype=dtype)
#     s3[nb:-nb, nb:-nb] = s0[nb:-nb, nb:-nb] - dts * (
#         0.5 * (flux_s_x_0[nb:-nb, nb:-nb] - flux_s_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#         + 0.5 * (flux_s_y_0[nb:-nb, nb:-nb] - flux_s_y_0[nb:-nb, nb - 1 : -nb - 1]) / dy
#         - 0.5 * (flux_s_x_1[nb:-nb, nb:-nb] - flux_s_x_1[nb - 1 : -nb - 1, nb:-nb]) / dx
#         - 0.5 * (flux_s_y_1[nb:-nb, nb:-nb] - flux_s_y_1[nb:-nb, nb - 1 : -nb - 1]) / dy
#         + (flux_s_x_2[nb:-nb, nb:-nb] - flux_s_x_2[nb - 1 : -nb - 1, nb:-nb]) / dx
#         + (flux_s_y_2[nb:-nb, nb:-nb] - flux_s_y_2[nb:-nb, nb - 1 : -nb - 1]) / dy
#         - (s_tnd[nb:-nb, nb:-nb] if s_tnd is not None else 0.0)
#     )
#     hb.dmn_enforce_field(
#         s3, "air_isentropic_density", "kg m^-2 K^-1", time=state["time"] + timestep
#     )
#     assert "air_isentropic_density" in raw_state_3
#     compare_arrays(s3, raw_state_3["air_isentropic_density"])
#
#     if moist:
#         # isentropic density of water vapor
#         flux_sqv_x_2, flux_sqv_y_2 = get_fifth_order_upwind_fluxes(u, v, sqv2)
#         sqv3 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqv3[nb:-nb, nb:-nb] = sqv0[nb:-nb, nb:-nb] - dts * (
#             0.5
#             * (flux_sqv_x_0[nb:-nb, nb:-nb] - flux_sqv_x_0[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 0.5
#             * (flux_sqv_y_0[nb:-nb, nb:-nb] - flux_sqv_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 0.5 * ((s0 * qv_tnd)[nb:-nb, nb:-nb] if qv_tnd is not None else 0.0)
#             - 0.5
#             * (flux_sqv_x_1[nb:-nb, nb:-nb] - flux_sqv_x_1[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             - 0.5
#             * (flux_sqv_y_1[nb:-nb, nb:-nb] - flux_sqv_y_1[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             + 0.5 * ((s1 * qv_tnd)[nb:-nb, nb:-nb] if qv_tnd is not None else 0.0)
#             + (flux_sqv_x_2[nb:-nb, nb:-nb] - flux_sqv_x_2[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + (flux_sqv_y_2[nb:-nb, nb:-nb] - flux_sqv_y_2[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - ((s2 * qv_tnd)[nb:-nb, nb:-nb] if qv_tnd is not None else 0.0)
#         )
#         assert "isentropic_density_of_water_vapor" in raw_state_3
#         compare_arrays(sqv3, raw_state_3["isentropic_density_of_water_vapor"])
#
#         # isentropic density of cloud liquid water
#         flux_sqc_x_2, flux_sqc_y_2 = get_fifth_order_upwind_fluxes(u, v, sqc2)
#         sqc3 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqc3[nb:-nb, nb:-nb] = sqc0[nb:-nb, nb:-nb] - dts * (
#             0.5
#             * (flux_sqc_x_0[nb:-nb, nb:-nb] - flux_sqc_x_0[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 0.5
#             * (flux_sqc_y_0[nb:-nb, nb:-nb] - flux_sqc_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 0.5 * ((s0 * qc_tnd)[nb:-nb, nb:-nb] if qc_tnd is not None else 0.0)
#             - 0.5
#             * (flux_sqc_x_1[nb:-nb, nb:-nb] - flux_sqc_x_1[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             - 0.5
#             * (flux_sqc_y_1[nb:-nb, nb:-nb] - flux_sqc_y_1[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             + 0.5 * ((s1 * qc_tnd)[nb:-nb, nb:-nb] if qc_tnd is not None else 0.0)
#             + (flux_sqc_x_2[nb:-nb, nb:-nb] - flux_sqc_x_2[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + (flux_sqc_y_2[nb:-nb, nb:-nb] - flux_sqc_y_2[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - ((s2 * qc_tnd)[nb:-nb, nb:-nb] if qc_tnd is not None else 0.0)
#         )
#         assert "isentropic_density_of_cloud_liquid_water" in raw_state_3
#         compare_arrays(sqc3, raw_state_3["isentropic_density_of_cloud_liquid_water"])
#
#         # isentropic density of precipitation water
#         flux_sqr_x_2, flux_sqr_y_2 = get_fifth_order_upwind_fluxes(u, v, sqr2)
#         sqr3 = np.zeros((nx, ny, nz), dtype=dtype)
#         sqr3[nb:-nb, nb:-nb] = sqr0[nb:-nb, nb:-nb] - dts * (
#             0.5
#             * (flux_sqr_x_0[nb:-nb, nb:-nb] - flux_sqr_x_0[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + 0.5
#             * (flux_sqr_y_0[nb:-nb, nb:-nb] - flux_sqr_y_0[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - 0.5 * ((s0 * qr_tnd)[nb:-nb, nb:-nb] if qr_tnd is not None else 0.0)
#             - 0.5
#             * (flux_sqr_x_1[nb:-nb, nb:-nb] - flux_sqr_x_1[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             - 0.5
#             * (flux_sqr_y_1[nb:-nb, nb:-nb] - flux_sqr_y_1[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             + 0.5 * ((s1 * qr_tnd)[nb:-nb, nb:-nb] if qr_tnd is not None else 0.0)
#             + (flux_sqr_x_2[nb:-nb, nb:-nb] - flux_sqr_x_2[nb - 1 : -nb - 1, nb:-nb])
#             / dx
#             + (flux_sqr_y_2[nb:-nb, nb:-nb] - flux_sqr_y_2[nb:-nb, nb - 1 : -nb - 1])
#             / dy
#             - ((s2 * qr_tnd)[nb:-nb, nb:-nb] if qr_tnd is not None else 0.0)
#         )
#         assert "isentropic_density_of_precipitation_water" in raw_state_3
#         compare_arrays(sqr3, raw_state_3["isentropic_density_of_precipitation_water"])
#
#     # montgomery potential
#     mtg3 = np.zeros((nx, ny, nz), dtype=dtype)
#     ids.get_montgomery_potential(s3, pt.to_units("Pa").values.item(), mtg3)
#     compare_arrays(mtg3, imp._mtg_new)
#
#     # x-momentum
#     flux_su_x_2, flux_su_y_2 = get_fifth_order_upwind_fluxes(u, v, su2)
#     su3 = np.zeros((nx, ny, nz), dtype=dtype)
#     su3[nb:-nb, nb:-nb] = su0[nb:-nb, nb:-nb] - dts * (
#         0.5
#         * (
#             (flux_su_x_0[nb:-nb, nb:-nb] - flux_su_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_su_y_0[nb:-nb, nb:-nb] - flux_su_y_0[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (su_tnd[nb:-nb, nb:-nb] if su_tnd is not None else 0.0)
#         )
#         - 0.5
#         * (
#             (flux_su_x_1[nb:-nb, nb:-nb] - flux_su_x_1[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_su_y_1[nb:-nb, nb:-nb] - flux_su_y_1[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (su_tnd[nb:-nb, nb:-nb] if su_tnd is not None else 0.0)
#         )
#         + (
#             (flux_su_x_2[nb:-nb, nb:-nb] - flux_su_x_2[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_su_y_2[nb:-nb, nb:-nb] - flux_su_y_2[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (su_tnd[nb:-nb, nb:-nb] if su_tnd is not None else 0.0)
#         )
#         + a
#         * s0[nb:-nb, nb:-nb]
#         * (mtg0[nb + 1 : -nb + 1, nb:-nb] - mtg0[nb - 1 : -nb - 1, nb:-nb])
#         / (2.0 * dx)
#         + b
#         * s2[nb:-nb, nb:-nb]
#         * (mtg2[nb + 1 : -nb + 1, nb:-nb] - mtg2[nb - 1 : -nb - 1, nb:-nb])
#         / (2.0 * dx)
#         + c
#         * s3[nb:-nb, nb:-nb]
#         * (mtg3[nb + 1 : -nb + 1, nb:-nb] - mtg3[nb - 1 : -nb - 1, nb:-nb])
#         / (2.0 * dx)
#     )
#     assert "x_momentum_isentropic" in raw_state_3
#     compare_arrays(su3, raw_state_3["x_momentum_isentropic"])
#
#     # y-momentum
#     flux_sv_x_2, flux_sv_y_2 = get_fifth_order_upwind_fluxes(u, v, sv2)
#     sv3 = np.zeros((nx, ny, nz), dtype=dtype)
#     sv3[nb:-nb, nb:-nb] = sv0[nb:-nb, nb:-nb] - dts * (
#         0.5
#         * (
#             (flux_sv_x_0[nb:-nb, nb:-nb] - flux_sv_x_0[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sv_y_0[nb:-nb, nb:-nb] - flux_sv_y_0[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (sv_tnd[nb:-nb, nb:-nb] if sv_tnd is not None else 0.0)
#         )
#         - 0.5
#         * (
#             (flux_sv_x_1[nb:-nb, nb:-nb] - flux_sv_x_1[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sv_y_1[nb:-nb, nb:-nb] - flux_sv_y_1[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (sv_tnd[nb:-nb, nb:-nb] if sv_tnd is not None else 0.0)
#         )
#         + (
#             (flux_sv_x_2[nb:-nb, nb:-nb] - flux_sv_x_2[nb - 1 : -nb - 1, nb:-nb]) / dx
#             + (flux_sv_y_2[nb:-nb, nb:-nb] - flux_sv_y_2[nb:-nb, nb - 1 : -nb - 1]) / dy
#             - (sv_tnd[nb:-nb, nb:-nb] if sv_tnd is not None else 0.0)
#         )
#         + a
#         * s0[nb:-nb, nb:-nb]
#         * (mtg0[nb:-nb, nb + 1 : -nb + 1] - mtg0[nb:-nb, nb - 1 : -nb - 1])
#         / (2.0 * dy)
#         + b
#         * s2[nb:-nb, nb:-nb]
#         * (mtg2[nb:-nb, nb + 1 : -nb + 1] - mtg2[nb:-nb, nb - 1 : -nb - 1])
#         / (2.0 * dy)
#         + c
#         * s3[nb:-nb, nb:-nb]
#         * (mtg3[nb:-nb, nb + 1 : -nb + 1] - mtg3[nb:-nb, nb - 1 : -nb - 1])
#         / (2.0 * dy)
#     )
#     assert "y_momentum_isentropic" in raw_state_3
#     compare_arrays(sv3, raw_state_3["y_momentum_isentropic"])


if __name__ == "__main__":
    pytest.main([__file__])
