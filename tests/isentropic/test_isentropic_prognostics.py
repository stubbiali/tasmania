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
from datetime import timedelta
from hypothesis import (
    assume,
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.framework.allocators import as_storage
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.dynamics.diagnostics import (
    IsentropicDiagnostics,
)
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicMinimalHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.prognostic import IsentropicPrognostic
from tasmania.python.isentropic.dynamics.subclasses.prognostics import (
    ForwardEulerSI,
    CenteredSI,
    RK3WSSI,
)
from tasmania.python.utils.storage import (
    deepcopy_array_dict,
    get_array_dict,
)

from tests import conf
from tests.isentropic.test_isentropic_horizontal_fluxes import (
    get_upwind_fluxes,
    get_fifth_order_upwind_fluxes,
)
from tests.strategies import (
    st_domain,
    st_floats,
    st_one_of,
    st_isentropic_state_f,
)
from tests.utilities import compare_arrays, compare_datetimes, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def forward_euler_step(
    get_fluxes,
    mode,
    nx,
    ny,
    nz,
    nb,
    dx,
    dy,
    dt,
    u_tmp,
    v_tmp,
    phi,
    phi_tmp,
    phi_tnd,
    phi_out,
):
    flux_x, flux_y = get_fluxes(u_tmp, v_tmp, phi_tmp)
    ib, jb, kb = nb, nb, 0
    ie, je, ke = nx - nb, ny - nb, nz
    phi_out[ib:ie, jb:je, kb:ke] = phi[ib:ie, jb:je, kb:ke] - dt * (
        (
            (
                flux_x[ib + 1 : ie + 1, jb:je, kb:ke]
                - flux_x[ib:ie, jb:je, kb:ke]
            )
            / dx
            if mode != "y"
            else 0.0
        )
        + (
            (
                flux_y[ib:ie, jb + 1 : je + 1, kb:ke]
                - flux_y[ib:ie, jb:je, kb:ke]
            )
            / dy
            if mode != "x"
            else 0.0
        )
        - (phi_tnd[ib:ie, jb:je, kb:ke] if phi_tnd is not None else 0.0)
    )


def forward_euler_step_momentum_x(
    get_fluxes,
    mode,
    eps,
    nx,
    ny,
    nz,
    nb,
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
    ib, jb, kb = nb, nb, 0
    ie, je, ke = nx - nb, ny - nb, nz

    flux_x, flux_y = get_fluxes(u_tmp, v_tmp, su_tmp)
    su_out[ib:ie, jb:je, kb:ke] = su[ib:ie, jb:je, kb:ke] - dt * (
        (
            (
                flux_x[ib + 1 : ie + 1, jb:je, kb:ke]
                - flux_x[ib:ie, jb:je, kb:ke]
            )
            / dx
            if mode != "y"
            else 0.0
        )
        + (
            (
                flux_y[ib:ie, jb + 1 : je + 1, kb:ke]
                - flux_y[ib:ie, jb:je, kb:ke]
            )
            / dy
            if mode != "x"
            else 0.0
        )
        + (1 - eps)
        * s[ib:ie, jb:je, kb:ke]
        * (
            mtg[ib + 1 : ie + 1, jb:je, kb:ke]
            - mtg[ib - 1 : ie - 1, jb:je, kb:ke]
        )
        / (2.0 * dx)
        + eps
        * s_new[ib:ie, jb:je, kb:ke]
        * (
            mtg_new[ib + 1 : ie + 1, jb:je, kb:ke]
            - mtg_new[ib - 1 : ie - 1, jb:je, kb:ke]
        )
        / (2.0 * dx)
        - (su_tnd[ib:ie, jb:je, kb:ke] if su_tnd is not None else 0.0)
    )


def forward_euler_step_momentum_y(
    get_fluxes,
    mode,
    eps,
    nx,
    ny,
    nz,
    nb,
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
    ib, jb, kb = nb, nb, 0
    ie, je, ke = nx - nb, ny - nb, nz

    flux_x, flux_y = get_fluxes(u_tmp, v_tmp, sv_tmp)
    sv_out[ib:ie, jb:je, kb:ke] = sv[ib:ie, jb:je, kb:ke] - dt * (
        (
            (
                flux_x[ib + 1 : ie + 1, jb:je, kb:ke]
                - flux_x[ib:ie, jb:je, kb:ke]
            )
            / dx
            if mode != "y"
            else 0.0
        )
        + (
            (
                flux_y[ib:ie, jb + 1 : je + 1, kb:ke]
                - flux_y[ib:ie, jb:je, kb:ke]
            )
            / dy
            if mode != "x"
            else 0.0
        )
        + (1 - eps)
        * s[ib:ie, jb:je, kb:ke]
        * (
            mtg[ib:ie, jb + 1 : je + 1, kb:ke]
            - mtg[ib:ie, jb - 1 : je - 1, kb:ke]
        )
        / (2.0 * dy)
        + eps
        * s_new[ib:ie, jb:je, kb:ke]
        * (
            mtg_new[ib:ie, jb + 1 : je + 1, kb:ke]
            - mtg_new[ib:ie, jb - 1 : je - 1, kb:ke]
        )
        / (2.0 * dy)
        - (sv_tnd[ib:ie, jb:je, kb:ke] if sv_tnd is not None else 0.0)
    )


def test_registry():
    registry = IsentropicPrognostic.registry[
        "tasmania.python.isentropic.dynamics.prognostic.IsentropicPrognostic"
    ]
    assert "forward_euler_si" in registry
    assert registry["forward_euler_si"] == ForwardEulerSI
    assert "centered_si" in registry
    assert registry["centered_si"] == CenteredSI
    assert "rk3ws_si" in registry
    assert registry["rk3ws_si"] == RK3WSSI


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_factory(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=3,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    moist = data.draw(hyp_st.booleans(), label="moist")

    # ========================================
    # test bed
    # ========================================
    imp_euler_si = IsentropicPrognostic.factory(
        "forward_euler_si",
        "upwind",
        domain,
        moist,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    assert isinstance(imp_euler_si, ForwardEulerSI)
    assert isinstance(imp_euler_si._hflux, IsentropicMinimalHorizontalFlux)

    imp_rk3ws_si = IsentropicPrognostic.factory(
        "rk3ws_si",
        "fifth_order_upwind",
        domain,
        moist,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    assert isinstance(imp_rk3ws_si, RK3WSSI)
    assert isinstance(imp_rk3ws_si._hflux, IsentropicMinimalHorizontalFlux)

    # imp_sil3 = IsentropicPrognostic.factory(
    # 	'sil3', 'fifth_order_upwind', grid, hb, moist,
    # 	backend=backend, dtype=dtype, aligned_index=aligned_index, pt=pt, a=a, b=b, c=c
    # )
    # assert isinstance(imp_sil3, SIL3)
    # assert isinstance(imp_sil3._hflux, IsentropicMinimalHorizontalFlux)
    # assert np.isclose(imp_sil3._a.value, a)
    # assert np.isclose(imp_sil3._b.value, b)
    # assert np.isclose(imp_sil3._c.value, c)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_forward_euler_si(data, backend, dtype, subtests):
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
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(2, 20),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    assume(domain.horizontal_boundary.type != "identity")
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    hb = domain.horizontal_boundary
    storage_shape = (nx + 1, ny + 1, nz + 1)

    moist = data.draw(hyp_st.booleans(), label="moist")

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=moist,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state",
    )

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd_on"):
        tendencies["air_isentropic_density"] = state["air_isentropic_density"]
    if data.draw(hyp_st.booleans(), label="su_tnd_on"):
        tendencies["x_momentum_isentropic"] = state["x_momentum_isentropic"]
    if data.draw(hyp_st.booleans(), label="sv_tnd_on"):
        tendencies["y_momentum_isentropic"] = state["y_momentum_isentropic"]
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd_on"):
            tendencies[mfwv] = state[mfwv]
        if data.draw(hyp_st.booleans(), label="qc_tnd_on"):
            tendencies[mfcw] = state[mfcw]
        if data.draw(hyp_st.booleans(), label="qr_tnd_on"):
            tendencies[mfpw] = state[mfpw]

    pt_raw = state["air_pressure_on_interface_levels"].data[0, 0, 0]
    pt = DataArray(
        pt_raw,
        attrs={
            "units": state["air_pressure_on_interface_levels"].attrs["units"]
        },
    )
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
    hb_np = domain.copy(backend="numpy").horizontal_boundary

    slc = (slice(nb, nx - nb), slice(nb, ny - nb), slice(0, nz))

    imp = IsentropicPrognostic.factory(
        "forward_euler_si",
        "upwind",
        domain,
        moist,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
        pt=pt,
        eps=eps,
    )

    raw_state = get_array_dict(state, properties={})
    if moist:
        s_np = to_numpy(raw_state["air_isentropic_density"])
        raw_state["isentropic_density_of_water_vapor"] = as_storage(
            backend, data=s_np * to_numpy(raw_state[mfwv]), storage_options=so
        )
        raw_state["isentropic_density_of_cloud_liquid_water"] = as_storage(
            backend, data=s_np * to_numpy(raw_state[mfcw]), storage_options=so
        )
        raw_state["isentropic_density_of_precipitation_water"] = as_storage(
            backend, data=s_np * to_numpy(raw_state[mfpw]), storage_options=so
        )
    raw_state_new = deepcopy_array_dict(raw_state)

    raw_tendencies = get_array_dict(tendencies, properties={})
    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = as_storage(
                backend,
                data=s_np * to_numpy(raw_tendencies[mfwv]),
                storage_options=so,
            )
        if mfcw in raw_tendencies:
            raw_tendencies[
                "isentropic_density_of_cloud_liquid_water"
            ] = as_storage(
                backend,
                data=s_np * to_numpy(raw_tendencies[mfcw]),
                storage_options=so,
            )
        if mfpw in raw_tendencies:
            raw_tendencies[
                "isentropic_density_of_precipitation_water"
            ] = as_storage(
                backend,
                data=s_np * to_numpy(raw_tendencies[mfpw]),
                storage_options=so,
            )

    imp.stage_call(
        0, timestep, raw_state, raw_tendencies, out_state=raw_state_new
    )

    assert "time" in raw_state_new.keys()
    compare_datetimes(raw_state_new["time"], raw_state["time"] + timestep)

    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    dt = timestep.total_seconds()
    u_now = to_numpy(raw_state["x_velocity_at_u_locations"])
    v_now = to_numpy(raw_state["y_velocity_at_v_locations"])

    # isentropic density
    s_now = to_numpy(raw_state["air_isentropic_density"])
    s_tnd = raw_tendencies.get("air_isentropic_density", None)
    s_tnd = to_numpy(s_tnd) if s_tnd is not None else None
    s_new = np.zeros_like(s_now)
    forward_euler_step(
        get_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dt,
        u_now,
        v_now,
        s_now,
        s_now,
        s_tnd,
        s_new,
    )
    hb_np.enforce_field(
        s_new,
        "air_isentropic_density",
        "kg m^-2 K^-1",
        time=state["time"] + timestep,
    )
    assert "air_isentropic_density" in raw_state_new
    compare_arrays(
        s_new,
        raw_state_new["air_isentropic_density"],
        slice=(slice(nx), slice(ny), slice(nz)),
    )

    if moist:
        # water species
        names = [
            "isentropic_density_of_water_vapor",
            "isentropic_density_of_cloud_liquid_water",
            "isentropic_density_of_precipitation_water",
        ]
        sq_new = np.zeros_like(s_now)
        for name in names:
            sq_now = to_numpy(raw_state[name])
            sq_tnd = raw_tendencies.get(name, None)
            sq_tnd = to_numpy(sq_tnd) if sq_tnd is not None else None
            forward_euler_step(
                get_upwind_fluxes,
                "xy",
                nx,
                ny,
                nz,
                nb,
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
            compare_arrays(sq_new, raw_state_new[name], slice=slc)

    # montgomery potential
    ids = IsentropicDiagnostics(
        grid, backend="numpy", backend_options=bo, storage_options=so
    )
    mtg_new = np.zeros_like(s_now)
    ids.get_montgomery_potential(s_new, pt.values.item(), mtg_new)
    compare_arrays(
        mtg_new,
        imp._mtg_new,
        slice=(slice(grid.nx), slice(grid.ny), slice(grid.nz)),
    )

    # x-momentum
    mtg_now = to_numpy(raw_state["montgomery_potential"])
    mtg_new = to_numpy(mtg_new)
    su_now = to_numpy(raw_state["x_momentum_isentropic"])
    su_tnd = raw_tendencies.get("x_momentum_isentropic", None)
    su_tnd = to_numpy(su_tnd) if su_tnd is not None else None
    su_new = np.zeros_like(su_now)
    forward_euler_step(
        get_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dt,
        u_now,
        v_now,
        su_now,
        su_now,
        su_tnd,
        su_new,
    )
    su_new[1:-1, 1:-1] -= dt * (
        (1 - eps)
        * s_now[1:-1, 1:-1]
        * (mtg_now[2:, 1:-1] - mtg_now[:-2, 1:-1])
        / (2.0 * dx)
        + eps
        * s_new[1:-1, 1:-1]
        * (mtg_new[2:, 1:-1] - mtg_new[:-2, 1:-1])
        / (2.0 * dx)
    )
    assert "x_momentum_isentropic" in raw_state_new
    compare_arrays(su_new, raw_state_new["x_momentum_isentropic"], slice=slc)

    # y-momentum
    sv_now = to_numpy(raw_state["y_momentum_isentropic"])
    sv_tnd = raw_tendencies.get("y_momentum_isentropic", None)
    sv_tnd = to_numpy(sv_tnd) if sv_tnd is not None else None
    sv_new = np.zeros_like(sv_now)
    forward_euler_step(
        get_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dt,
        u_now,
        v_now,
        sv_now,
        sv_now,
        sv_tnd,
        sv_new,
    )
    sv_new[1:-1, 1:-1] -= dt * (
        (1 - eps)
        * s_now[1:-1, 1:-1]
        * (mtg_now[1:-1, 2:] - mtg_now[1:-1, :-2])
        / (2.0 * dy)
        + eps
        * s_new[1:-1, 1:-1]
        * (mtg_new[1:-1, 2:] - mtg_new[1:-1, :-2])
        / (2.0 * dy)
    )
    assert "y_momentum_isentropic" in raw_state_new
    compare_arrays(sv_new, raw_state_new["y_momentum_isentropic"], slice=slc)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_rk3ws_si(data, backend, dtype, subtests):
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
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(2, 20),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    assume(domain.horizontal_boundary.type != "identity")
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)
    hb = domain.horizontal_boundary

    moist = data.draw(hyp_st.booleans(), label="moist")

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=moist,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
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

    pt_raw = state["air_pressure_on_interface_levels"].data[0, 0, 0]
    pt = DataArray(
        pt_raw,
        attrs={
            "units": state["air_pressure_on_interface_levels"].attrs["units"]
        },
    )
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
    hb_np = domain.copy(backend="numpy").horizontal_boundary

    slc = (slice(nb, nx - nb), slice(nb, ny - nb), slice(0, nz))
    slc_ext = (slice(0, nx), slice(0, ny), slice(0, nz))

    imp = IsentropicPrognostic.factory(
        "rk3ws_si",
        "fifth_order_upwind",
        domain,
        moist,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
        pt=pt,
        eps=eps,
    )

    raw_state = get_array_dict(state, properties={})
    s0 = to_numpy(raw_state["air_isentropic_density"])
    if moist:
        raw_state["isentropic_density_of_water_vapor"] = as_storage(
            backend, data=s0 * to_numpy(raw_state[mfwv]), storage_options=so
        )
        raw_state["isentropic_density_of_cloud_liquid_water"] = as_storage(
            backend, data=s0 * to_numpy(raw_state[mfcw]), storage_options=so
        )
        raw_state["isentropic_density_of_precipitation_water"] = as_storage(
            backend, data=s0 * to_numpy(raw_state[mfpw]), storage_options=so
        )

    raw_tendencies = get_array_dict(tendencies, properties={})
    if moist:
        if mfwv in raw_tendencies:
            qv_tnd = to_numpy(raw_tendencies[mfwv])
            raw_tendencies["isentropic_density_of_water_vapor"] = as_storage(
                backend, data=s0 * qv_tnd, storage_options=so
            )
        if mfcw in raw_tendencies:
            qc_tnd = to_numpy(raw_tendencies[mfcw])
            raw_tendencies[
                "isentropic_density_of_cloud_liquid_water"
            ] = as_storage(backend, data=s0 * qc_tnd, storage_options=so)
        if mfpw in raw_tendencies:
            qr_tnd = to_numpy(raw_tendencies[mfpw])
            raw_tendencies[
                "isentropic_density_of_precipitation_water"
            ] = as_storage(backend, data=s0 * qr_tnd, storage_options=so)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()
    u = to_numpy(raw_state["x_velocity_at_u_locations"])
    v = to_numpy(raw_state["y_velocity_at_v_locations"])

    names = [
        "isentropic_density_of_water_vapor",
        "isentropic_density_of_cloud_liquid_water",
        "isentropic_density_of_precipitation_water",
    ]
    sq_new = np.zeros_like(u)

    #
    # stage 0
    #
    dts = (timestep / 3.0).total_seconds()

    raw_state_1 = deepcopy_array_dict(raw_state)
    imp.stage_call(
        0, timestep, raw_state, raw_tendencies, out_state=raw_state_1
    )

    assert "time" in raw_state_1
    compare_datetimes(
        raw_state_1["time"], raw_state["time"] + 1.0 / 3.0 * timestep
    )

    # isentropic density
    s_tnd = raw_tendencies.get("air_isentropic_density", None)
    s_tnd = to_numpy(s_tnd) if s_tnd is not None else None
    s1 = np.zeros_like(s0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        s0,
        s0,
        s_tnd,
        s1,
    )
    hb_np.enforce_field(
        s1,
        "air_isentropic_density",
        "kg m^-2 K^-1",
        time=state["time"] + 1.0 / 3.0 * timestep,
    )
    assert "air_isentropic_density" in raw_state_1
    compare_arrays(s1, raw_state_1["air_isentropic_density"], slice=slc_ext)

    if moist:
        # water species
        for name in names:
            sq0 = to_numpy(raw_state[name])
            sq_tnd = raw_tendencies.get(name, None)
            sq_tnd = to_numpy(sq_tnd) if sq_tnd is not None else None
            forward_euler_step(
                get_fifth_order_upwind_fluxes,
                "xy",
                nx,
                ny,
                nz,
                nb,
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
            compare_arrays(sq_new, raw_state_1[name], slice=slc)

    # montgomery potential
    ids = IsentropicDiagnostics(
        grid,
        backend="numpy",
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    mtg1 = np.zeros_like(s0)
    ids.get_montgomery_potential(s1, pt.to_units("Pa").values.item(), mtg1)
    compare_arrays(mtg1, imp._mtg_new, slice=slc_ext)

    # x-momentum
    mtg0 = to_numpy(raw_state["montgomery_potential"])
    compare_arrays(mtg0, imp._mtg_now, slice=slc_ext)
    su0 = to_numpy(raw_state["x_momentum_isentropic"])
    su_tnd = raw_tendencies.get("x_momentum_isentropic", None)
    su_tnd = to_numpy(su_tnd) if su_tnd is not None else None
    su1 = np.zeros_like(su0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        su0,
        su0,
        su_tnd,
        su1,
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
    compare_arrays(su1, raw_state_1["x_momentum_isentropic"], slice=slc)

    # y-momentum
    sv0 = to_numpy(raw_state["y_momentum_isentropic"])
    sv_tnd = raw_tendencies.get("y_momentum_isentropic", None)
    sv_tnd = to_numpy(sv_tnd) if sv_tnd is not None else None
    sv1 = np.zeros_like(sv0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        sv0,
        sv0,
        sv_tnd,
        sv1,
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
    compare_arrays(sv1, raw_state_1["y_momentum_isentropic"], slice=slc)

    #
    # stage 1
    #
    raw_state_1["x_velocity_at_u_locations"] = raw_state[
        "x_velocity_at_u_locations"
    ]
    raw_state_1["y_velocity_at_v_locations"] = raw_state[
        "y_velocity_at_v_locations"
    ]
    raw_state_1_dc = deepcopy_array_dict(raw_state_1)

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = as_storage(
                backend, data=s1 * qv_tnd, storage_options=so
            )
        if mfcw in raw_tendencies:
            raw_tendencies[
                "isentropic_density_of_cloud_liquid_water"
            ] = as_storage(backend, data=s1 * qc_tnd, storage_options=so)
        if mfpw in raw_tendencies:
            raw_tendencies[
                "isentropic_density_of_precipitation_water"
            ] = as_storage(backend, data=s1 * qr_tnd, storage_options=so)

    dts = (0.5 * timestep).total_seconds()

    raw_state_2 = deepcopy_array_dict(raw_state_1)
    imp.stage_call(
        1, timestep, raw_state_1_dc, raw_tendencies, out_state=raw_state_2
    )

    assert "time" in raw_state_2
    compare_datetimes(raw_state_2["time"], raw_state["time"] + 0.5 * timestep)

    # isentropic density
    s2 = np.zeros_like(s0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        s0,
        s1,
        s_tnd,
        s2,
    )
    hb_np.enforce_field(
        s2,
        "air_isentropic_density",
        "kg m^-2 K^-1",
        time=state["time"] + 0.5 * timestep,
    )
    assert "air_isentropic_density" in raw_state_2
    compare_arrays(s2, raw_state_2["air_isentropic_density"], slice=slc_ext)

    if moist:
        # water species
        for name in names:
            sq0 = to_numpy(raw_state[name])
            sq1 = to_numpy(raw_state_1_dc[name])
            sq_tnd = raw_tendencies.get(name, None)
            sq_tnd = to_numpy(sq_tnd) if sq_tnd is not None else None
            forward_euler_step(
                get_fifth_order_upwind_fluxes,
                "xy",
                nx,
                ny,
                nz,
                nb,
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
            compare_arrays(sq_new, raw_state_2[name], slice=slc)

    # montgomery potential
    mtg2 = np.zeros_like(mtg0)
    ids.get_montgomery_potential(s2, pt.to_units("Pa").values.item(), mtg2)
    compare_arrays(mtg2, imp._mtg_new, slice=slc_ext)

    # x-momentum
    su1 = to_numpy(raw_state_1_dc["x_momentum_isentropic"])
    su2 = np.zeros_like(su0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        su0,
        su1,
        su_tnd,
        su2,
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
    compare_arrays(su2, raw_state_2["x_momentum_isentropic"], slice=slc)

    # y-momentum
    sv1 = to_numpy(raw_state_1_dc["y_momentum_isentropic"])
    sv2 = np.zeros_like(sv0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        sv0,
        sv1,
        sv_tnd,
        sv2,
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
    compare_arrays(sv2, raw_state_2["y_momentum_isentropic"], slice=slc)

    #
    # stage 2
    #
    raw_state_2["x_velocity_at_u_locations"] = raw_state[
        "x_velocity_at_u_locations"
    ]
    raw_state_2["y_velocity_at_v_locations"] = raw_state[
        "y_velocity_at_v_locations"
    ]
    raw_state_2_dc = deepcopy_array_dict(raw_state_2)

    if moist:
        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = as_storage(
                backend, data=s2 * qv_tnd, storage_options=so
            )
        if mfcw in raw_tendencies:
            raw_tendencies[
                "isentropic_density_of_cloud_liquid_water"
            ] = as_storage(backend, data=s2 * qc_tnd, storage_options=so)
        if mfpw in raw_tendencies:
            raw_tendencies[
                "isentropic_density_of_precipitation_water"
            ] = as_storage(backend, data=s2 * qr_tnd, storage_options=so)

    dts = timestep.total_seconds()

    raw_state_3 = deepcopy_array_dict(raw_state_2)
    imp.stage_call(
        2, timestep, raw_state_2_dc, raw_tendencies, out_state=raw_state_3
    )

    assert "time" in raw_state_3.keys()
    compare_datetimes(raw_state_3["time"], raw_state["time"] + timestep)

    # isentropic density
    s3 = np.zeros_like(s0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        s0,
        s2,
        s_tnd,
        s3,
    )
    hb_np.enforce_field(
        s3,
        "air_isentropic_density",
        "kg m^-2 K^-1",
        time=state["time"] + timestep,
    )
    assert "air_isentropic_density" in raw_state_3
    compare_arrays(s3, raw_state_3["air_isentropic_density"], slice=slc_ext)

    if moist:
        # water species
        for name in names:
            sq0 = to_numpy(raw_state[name])
            sq2 = to_numpy(raw_state_2_dc[name])
            sq_tnd = raw_tendencies.get(name, None)
            sq_tnd = to_numpy(sq_tnd) if sq_tnd is not None else None
            forward_euler_step(
                get_fifth_order_upwind_fluxes,
                "xy",
                nx,
                ny,
                nz,
                nb,
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
            compare_arrays(sq_new, raw_state_3[name], slice=slc)

    # montgomery potential
    mtg3 = np.zeros_like(mtg0)
    ids.get_montgomery_potential(s3, pt.to_units("Pa").values.item(), mtg3)
    compare_arrays(mtg3, imp._mtg_new, slice=slc_ext)

    # x-momentum
    su2 = to_numpy(raw_state_2_dc["x_momentum_isentropic"])
    su3 = np.zeros_like(su0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        su0,
        su2,
        su_tnd,
        su3,
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
    compare_arrays(su3, raw_state_3["x_momentum_isentropic"], slice=slc)

    # y-momentum
    sv2 = to_numpy(raw_state_2_dc["y_momentum_isentropic"])
    sv3 = np.zeros_like(sv0)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        nb,
        dx,
        dy,
        dts,
        u,
        v,
        sv0,
        sv2,
        sv_tnd,
        sv3,
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
    compare_arrays(sv3, raw_state_3["y_momentum_isentropic"], slice=slc)


if __name__ == "__main__":
    pytest.main([__file__])
