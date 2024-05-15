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
from datetime import timedelta
from hypothesis import (
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest
from sympl import DataArray
from typing import TYPE_CHECKING

from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.dwarfs.vertical_damping import VerticalDamping
from tasmania.python.domain.domain import Domain
from tasmania.python.framework.allocators import as_storage, zeros
from tasmania import TendencyComponent
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.dynamics.diagnostics import (
    IsentropicDiagnostics as RawIsentropicDiagnostics,
)
from tasmania.python.isentropic.dynamics.dycore import IsentropicDynamicalCore

from tasmania.python.isentropic.physics.coriolis import (
    IsentropicConservativeCoriolis,
)
from tasmania.python.isentropic.physics.diagnostics import (
    IsentropicDiagnostics,
)
from tasmania.python.utils import typingx as ty
from tasmania.python.utils.storage import (
    deepcopy_array_dict,
    deepcopy_dataarray,
    deepcopy_dataarray_dict,
)

from tests import conf
from tests.isentropic.test_isentropic_horizontal_fluxes import (
    get_fifth_order_upwind_fluxes,
)
from tests.isentropic.test_isentropic_prognostics import (
    forward_euler_step,
    forward_euler_step_momentum_x,
    forward_euler_step_momentum_y,
)
from tests.strategies import (
    st_floats,
    st_one_of,
    st_domain,
    st_isentropic_state_f,
)
from tests.utilities import compare_arrays, compare_datetimes, hyp_settings

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain
    from tasmania.python.domain.grid import Grid
    from tasmania.python.domain.horizontal_boundary import HorizontalBoundary


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


dummy_domain = Domain(
    DataArray([0, 1], dims="x", attrs={"units": "m"}),
    10,
    DataArray([0, 1], dims="y", attrs={"units": "m"}),
    10,
    DataArray([0, 1], dims="z", attrs={"units": "K"}),
    10,
)
dummy_grid = dummy_domain.numerical_grid
rid = RawIsentropicDiagnostics(dummy_grid)


def get_density_of_water_constituent(
    s: np.ndarray, q: np.ndarray, sq: np.ndarray, clipping: bool = True
) -> None:
    sq[...] = s * q
    if clipping:
        sq[sq < 0.0] = 0.0
        sq[np.isnan(sq)] = 0.0


def get_mass_fraction_of_water_constituent_in_air(
    s: np.ndarray, sq: np.ndarray, q: np.ndarray, clipping: bool = True
) -> None:
    q[...] = sq / s
    if clipping:
        q[q < 0.0] = 0.0
        q[np.isnan(q)] = 0.0


def get_montgomery_potential(
    grid: "Grid", s: np.ndarray, pt: float, mtg: np.ndarray
) -> None:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz = grid.dz.to_units("K").values.item()
    theta_s = grid.z_on_interface_levels.to_units("K").values[-1]
    topo = grid.topography.profile.to_units("m").data

    pref = rid.rpc["air_pressure_at_sea_level"]
    rd = rid.rpc["gas_constant_of_dry_air"]
    g = rid.rpc["gravitational_acceleration"]
    cp = rid.rpc["specific_heat_of_dry_air_at_constant_pressure"]

    p = np.zeros_like(s)
    p[:nx, :ny, 0] = pt
    for k in range(1, nz + 1):
        p[:nx, :ny, k] = p[:nx, :ny, k - 1] + g * dz * s[:nx, :ny, k - 1]

    exn = cp * (p / pref) ** (rd / cp)

    mtg_s = theta_s * exn[:nx, :ny, nz] + g * topo[:nx, :ny]
    mtg[:nx, :ny, nz - 1] = mtg_s + 0.5 * dz * exn[:nx, :ny, nz]
    for k in reversed(range(0, nz - 1)):
        mtg[:nx, :ny, k] = mtg[:nx, :ny, k + 1] + dz * exn[:nx, :ny, k + 1]


def get_velocity_components(
    nx: int,
    ny: int,
    s: np.ndarray,
    su: np.ndarray,
    sv: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> None:
    u[1:nx, :] = (su[: nx - 1, :] + su[1:nx, :]) / (
        s[: nx - 1, :] + s[1:nx, :]
    )
    v[:, 1:ny] = (sv[:, : ny - 1] + sv[:, 1:ny]) / (
        s[:, : ny - 1] + s[:, 1:ny]
    )


def apply_rayleigh_damping(
    vd: VerticalDamping,
    dt: timedelta,
    phi_now: np.ndarray,
    phi_new: np.ndarray,
    phi_ref: np.ndarray,
    phi_out: np.ndarray,
) -> None:
    ni, nj, nk = phi_now.shape
    rmat = to_numpy(vd._rmat)
    dnk = vd._damp_depth
    phi_out[:ni, :nj, :dnk] = phi_new[
        :ni, :nj, :dnk
    ] - dt.total_seconds() * rmat[:ni, :nj, :dnk] * (
        phi_now[:ni, :nj, :dnk] - phi_ref[:ni, :nj, :dnk]
    )


def apply_second_order_smoothing(
    hs: HorizontalSmoothing,
    phi: np.ndarray,
    phi_out: np.ndarray,
) -> None:
    nb = hs._nb
    g = to_numpy(hs._gamma)
    ni, nj, nk = phi.shape

    i, j, k = slice(nb, ni - nb), slice(nb, nj - nb), slice(0, nk)
    im1, ip1 = slice(nb - 1, ni - nb - 1), slice(nb + 1, ni - nb + 1)
    im2, ip2 = slice(nb - 2, ni - nb - 2), slice(nb + 2, ni - nb + 2)
    jm1, jp1 = slice(nb - 1, nj - nb - 1), slice(nb + 1, nj - nb + 1)
    jm2, jp2 = slice(nb - 2, nj - nb - 2), slice(nb + 2, nj - nb + 2)

    phi_out[i, j, k] = (1 - 0.75 * g[i, j, k]) * phi[i, j, k] + 0.0625 * g[
        i, j, k
    ] * (
        -phi[im2, j, k]
        + 4.0 * phi[im1, j, k]
        - phi[ip2, j, k]
        + 4.0 * phi[ip1, j, k]
        - phi[i, jm2, k]
        + 4.0 * phi[i, jm1, k]
        - phi[i, jp2, k]
        + 4.0 * phi[i, jp1, k]
    )
    phi_out[0:nb, j, k] = phi[0:nb, j, k]
    phi_out[ni - nb : ni, j, k] = phi[ni - nb, j, k]
    phi_out[0:ni, 0:nb, k] = phi[0:ni, 0:nb, k]
    phi_out[0:ni, nj - nb : nj, k] = phi[0:ni, nj - nb : nj, k]


def rk3wssi_stage(
    stage: int,
    timestep: timedelta,
    grid: "Grid",
    raw_state_now: ty.StorageDict,
    raw_state_int: ty.StorageDict,
    raw_state_ref: ty.StorageDict,
    raw_tendencies: ty.StorageDict,
    raw_state_new: ty.StorageDict,
    field_properties: ty.PropertiesDict,
    hb: "HorizontalBoundary",
    hb_np: "HorizontalBoundary",
    moist: bool,
    damp: bool,
    vd: VerticalDamping,
    smooth: bool,
    hs: HorizontalSmoothing,
    eps: float,
    *,
    backend: str,
    storage_options: StorageOptions
) -> None:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    u_int = to_numpy(raw_state_int["x_velocity_at_u_locations"])
    v_int = to_numpy(raw_state_int["y_velocity_at_v_locations"])

    s_int = to_numpy(raw_state_int["air_isentropic_density"])
    if moist:
        qv_int = to_numpy(raw_state_int[mfwv])
        sqv_int = np.zeros_like(s_int)
        get_density_of_water_constituent(s_int, qv_int, sqv_int, clipping=True)
        raw_state_int["isentropic_density_of_water_vapor"] = as_storage(
            backend, data=sqv_int, storage_options=storage_options
        )

        qc_int = to_numpy(raw_state_int[mfcw])
        sqc_int = np.zeros_like(s_int)
        get_density_of_water_constituent(s_int, qc_int, sqc_int, clipping=True)
        raw_state_int["isentropic_density_of_cloud_liquid_water"] = as_storage(
            backend, data=sqc_int, storage_options=storage_options
        )

        qr_int = to_numpy(raw_state_int[mfpw])
        sqr_int = np.zeros_like(s_int)
        get_density_of_water_constituent(s_int, qr_int, sqr_int, clipping=True)
        raw_state_int[
            "isentropic_density_of_precipitation_water"
        ] = as_storage(backend, data=sqr_int, storage_options=storage_options)

        if mfwv in raw_tendencies:
            raw_tendencies["isentropic_density_of_water_vapor"] = as_storage(
                backend,
                data=s_int * to_numpy(raw_tendencies[mfwv]),
                storage_options=storage_options,
            )
        if mfcw in raw_tendencies:
            raw_tendencies[
                "isentropic_density_of_cloud_liquid_water"
            ] = as_storage(
                backend,
                data=s_int * to_numpy(raw_tendencies[mfcw]),
                storage_options=storage_options,
            )
        if mfpw in raw_tendencies:
            raw_tendencies[
                "isentropic_density_of_precipitation_water"
            ] = as_storage(
                backend,
                data=s_int * to_numpy(raw_tendencies[mfpw]),
                storage_options=storage_options,
            )

    if stage == 0:
        fraction = 1.0 / 3.0
    elif stage == 1:
        fraction = 0.5
    else:
        fraction = 1.0

    raw_state_new["time"] = raw_state_now["time"] + fraction * timestep

    dt = (fraction * timestep).total_seconds()

    # isentropic_prognostic density
    s_now = to_numpy(raw_state_now["air_isentropic_density"])
    s_tnd = raw_tendencies.get("air_isentropic_density", None)
    s_tnd = to_numpy(s_tnd) if s_tnd is not None else None
    s_new = np.zeros_like(s_now)
    forward_euler_step(
        get_fifth_order_upwind_fluxes,
        "xy",
        nx,
        ny,
        nz,
        hb_np.nb,
        dx,
        dy,
        dt,
        u_int,
        v_int,
        s_now,
        s_int,
        s_tnd,
        s_new,
    )
    hb_np.enforce_field(
        s_new,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=raw_state_new["time"],
    )
    raw_state_new["air_isentropic_density"] = as_storage(
        backend, data=s_new, storage_options=storage_options
    )

    if moist:
        # water species
        names = [
            "isentropic_density_of_water_vapor",
            "isentropic_density_of_cloud_liquid_water",
            "isentropic_density_of_precipitation_water",
        ]
        for name in names:
            sq_now = to_numpy(raw_state_now[name])
            sq_int = to_numpy(raw_state_int[name])
            sq_tnd = raw_tendencies.get(name, None)
            sq_tnd = to_numpy(sq_tnd) if sq_tnd is not None else None
            sq_new = np.zeros_like(sq_now)
            forward_euler_step(
                get_fifth_order_upwind_fluxes,
                "xy",
                nx,
                ny,
                nz,
                hb_np.nb,
                dx,
                dy,
                dt,
                u_int,
                v_int,
                sq_now,
                sq_int,
                sq_tnd,
                sq_new,
            )
            raw_state_new[name] = as_storage(
                backend, data=sq_new, storage_options=storage_options
            )

    # montgomery potential
    pt = raw_state_now["air_pressure_on_interface_levels"].data[0, 0, 0]
    mtg_new = np.zeros_like(s_now)
    get_montgomery_potential(grid, s_new, pt, mtg_new)
    raw_state_new["montgomery_potential"] = as_storage(
        backend, data=mtg_new, storage_options=storage_options
    )

    # x-momentum
    mtg_now = to_numpy(raw_state_now["montgomery_potential"])
    su_now = to_numpy(raw_state_now["x_momentum_isentropic"])
    su_int = to_numpy(raw_state_int["x_momentum_isentropic"])
    su_tnd = raw_tendencies.get("x_momentum_isentropic", None)
    su_tnd = to_numpy(su_tnd) if su_tnd is not None else None
    su_new = np.zeros_like(su_now)
    forward_euler_step_momentum_x(
        get_fifth_order_upwind_fluxes,
        "xy",
        eps,
        nx,
        ny,
        nz,
        hb_np.nb,
        dx,
        dy,
        dt,
        s_now,
        s_new,
        u_int,
        v_int,
        mtg_now,
        mtg_new,
        su_now,
        su_int,
        su_tnd,
        su_new,
    )
    raw_state_new["x_momentum_isentropic"] = as_storage(
        backend, data=su_new, storage_options=storage_options
    )

    # y-momentum
    sv_now = to_numpy(raw_state_now["y_momentum_isentropic"])
    sv_int = to_numpy(raw_state_int["y_momentum_isentropic"])
    sv_tnd = raw_tendencies.get("y_momentum_isentropic", None)
    sv_tnd = to_numpy(sv_tnd) if sv_tnd is not None else None
    sv_new = np.zeros_like(sv_now)
    forward_euler_step_momentum_y(
        get_fifth_order_upwind_fluxes,
        "xy",
        eps,
        nx,
        ny,
        nz,
        hb_np.nb,
        dx,
        dy,
        dt,
        s_now,
        s_new,
        u_int,
        v_int,
        mtg_now,
        mtg_new,
        sv_now,
        sv_int,
        sv_tnd,
        sv_new,
    )
    raw_state_new["y_momentum_isentropic"] = as_storage(
        backend, data=sv_new, storage_options=storage_options
    )

    if moist:
        sqv_new = to_numpy(raw_state_new["isentropic_density_of_water_vapor"])
        qv_new = np.zeros_like(sv_now)
        get_mass_fraction_of_water_constituent_in_air(
            s_new,
            sqv_new,
            qv_new,
            clipping=True,
        )
        raw_state_new[mfwv] = as_storage(
            backend, data=qv_new, storage_options=storage_options
        )

        sqc_new = to_numpy(
            raw_state_new["isentropic_density_of_cloud_liquid_water"]
        )
        qc_new = np.zeros_like(sv_now)
        get_mass_fraction_of_water_constituent_in_air(
            s_new,
            sqc_new,
            qc_new,
            clipping=True,
        )
        raw_state_new[mfcw] = as_storage(
            backend, data=qc_new, storage_options=storage_options
        )

        sqr_new = to_numpy(
            raw_state_new["isentropic_density_of_precipitation_water"]
        )
        qr_new = np.zeros_like(sv_now)
        get_mass_fraction_of_water_constituent_in_air(
            s_new,
            sqr_new,
            qr_new,
            clipping=True,
        )
        raw_state_new[mfpw] = as_storage(
            backend, data=qr_new, storage_options=storage_options
        )

    hb.enforce_raw(raw_state_new, field_properties=field_properties)

    if damp:
        names = [
            "air_isentropic_density",
            "x_momentum_isentropic",
            "y_momentum_isentropic",
        ]
        for name in names:
            phi_now = to_numpy(raw_state_now[name])
            phi_new = to_numpy(raw_state_new[name])
            phi_ref = to_numpy(raw_state_ref[name])
            phi_out = to_numpy(raw_state_new[name])
            apply_rayleigh_damping(
                vd, timestep, phi_now, phi_new, phi_ref, phi_out
            )
            raw_state_new[name] = as_storage(
                backend, data=phi_out, storage_options=storage_options
            )

    if smooth:
        for name in field_properties:
            phi = to_numpy(raw_state_new[name])
            phi_out = np.zeros_like(phi)
            apply_second_order_smoothing(hs, phi, phi_out)
            hb.enforce_field(
                phi_out,
                field_name=name,
                field_units=field_properties[name]["units"],
                time=raw_state_new["time"],
            )
            raw_state_new[name] = as_storage(
                backend, data=phi_out, storage_options=storage_options
            )

    s_new = to_numpy(raw_state_new["air_isentropic_density"])
    su_new = to_numpy(raw_state_new["x_momentum_isentropic"])
    sv_new = to_numpy(raw_state_new["y_momentum_isentropic"])
    u_new = np.zeros_like(s_new)
    v_new = np.zeros_like(s_new)
    get_velocity_components(nx, ny, s_new, su_new, sv_new, u_new, v_new)
    hb_np.set_outermost_layers_x(
        u_new,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=raw_state_new["time"],
    )
    hb_np.set_outermost_layers_y(
        v_new,
        field_name="y_velocity_at_v_locations",
        field_units="m s^-1",
        time=raw_state_new["time"],
    )
    raw_state_new["x_velocity_at_u_locations"] = as_storage(
        backend, data=u_new, storage_options=storage_options
    )
    raw_state_new["y_velocity_at_v_locations"] = as_storage(
        backend, data=v_new, storage_options=storage_options
    )


def rk3ws_step(
    domain: "Domain",
    moist: bool,
    timestep: timedelta,
    raw_state_0: ty.StorageDict,
    raw_tendencies: ty.StorageDict,
    damp: bool,
    damp_at_every_stage: bool,
    vd: VerticalDamping,
    smooth: bool,
    smooth_at_every_stage: bool,
    hs: HorizontalSmoothing,
    eps: float,
    *,
    backend: str,
    storage_options: StorageOptions
) -> ty.StorageDict:
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    hb_np = domain.copy(backend="numpy").horizontal_boundary

    raw_state_1 = deepcopy_array_dict(raw_state_0)
    raw_state_2 = deepcopy_array_dict(raw_state_0)
    raw_state_3 = deepcopy_array_dict(raw_state_0)

    field_properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "y_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    }
    if moist:
        field_properties.update(
            {
                mfwv: {"units": "g g^-1"},
                mfcw: {"units": "g g^-1"},
                mfpw: {"units": "g g^-1"},
            }
        )

    state_ref = hb.reference_state
    raw_state_ref = {}
    for name in field_properties:
        raw_state_ref[name] = (
            state_ref[name].to_units(field_properties[name]["units"]).data
        )

    # stage 0
    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        0,
        timestep,
        grid,
        raw_state_0,
        raw_state_0,
        raw_state_ref,
        raw_tendencies,
        raw_state_1,
        field_properties,
        hb,
        hb_np,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
        backend=backend,
        storage_options=storage_options,
    )

    # stage 1
    rk3wssi_stage(
        1,
        timestep,
        grid,
        raw_state_0,
        raw_state_1,
        raw_state_ref,
        raw_tendencies,
        raw_state_2,
        field_properties,
        hb,
        hb_np,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
        backend=backend,
        storage_options=storage_options,
    )

    # stage 2
    rk3wssi_stage(
        2,
        timestep,
        grid,
        raw_state_0,
        raw_state_2,
        raw_state_ref,
        raw_tendencies,
        raw_state_3,
        field_properties,
        hb,
        hb_np,
        moist,
        damp,
        vd,
        smooth,
        hs,
        eps,
        backend=backend,
        storage_options=storage_options,
    )

    return raw_state_3


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize(
    "backend", conf.backend_debug.union(conf.gt_cpu_backend)
)
@pytest.mark.parametrize("dtype", conf.dtype)
def test1(data, backend, dtype, subtests):
    """
    - Slow tendencies: no
    - Intermediate tendencies: no
    - Intermediate diagnostics: no
    - Sub-stepping: no
    """
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
            xaxis_length=(1, 50),
            yaxis_length=(1, 50),
            zaxis_length=(2, 50),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
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
    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    eps = data.draw(st_floats(min_value=0, max_value=1), label="eps")

    damp = data.draw(hyp_st.booleans(), label="damp")
    damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="damp_depth"
    )
    damp_at_every_stage = data.draw(
        hyp_st.booleans(), label="damp_at_every_stage"
    )

    smooth = False  # data.draw(hyp_st.booleans(), label="smooth")
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz),
        label="smooth_damp_depth",
    )
    smooth_at_every_stage = data.draw(
        hyp_st.booleans(), label="smooth_at_every_stage"
    )

    # ========================================
    # test bed
    # ========================================
    domain.horizontal_boundary.reference_state = state

    vd = VerticalDamping.factory(
        "rayleigh",
        grid,
        damp_depth,
        0.0002,
        "s",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    hs = HorizontalSmoothing.factory(
        "second_order",
        storage_shape,
        0.03,
        0.24,
        smooth_damp_depth,
        hb.nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    pt_raw = state["air_pressure_on_interface_levels"].data[0, 0, 0]
    pt = DataArray(
        pt_raw,
        attrs={
            "units": state["air_pressure_on_interface_levels"].attrs["units"]
        },
    )

    dycore = IsentropicDynamicalCore(
        domain,
        fast_tendency_component=None,
        fast_diagnostic_component=None,
        superfast_tendency_component=None,
        superfast_diagnostic_component=None,
        moist=moist,
        time_integration_scheme="rk3ws_si",
        horizontal_flux_scheme="fifth_order_upwind",
        time_integration_properties={"pt": pt, "eps": eps},
        damp=damp,
        damp_type="rayleigh",
        damp_depth=damp_depth,
        damp_max=0.0002,
        damp_at_every_stage=damp_at_every_stage,
        smooth=smooth,
        smooth_type="second_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=smooth_damp_depth,
        smooth_at_every_stage=smooth_at_every_stage,
        smooth_moist=smooth,
        smooth_moist_type="second_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=smooth_damp_depth,
        smooth_moist_at_every_stage=smooth_at_every_stage,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    #
    # test properties
    #
    assert "air_isentropic_density" in dycore.input_properties
    assert "montgomery_potential" in dycore.input_properties
    assert "x_momentum_isentropic" in dycore.input_properties
    assert "x_velocity_at_u_locations" in dycore.input_properties
    assert "y_momentum_isentropic" in dycore.input_properties
    assert "y_velocity_at_v_locations" in dycore.input_properties
    if moist:
        assert mfwv in dycore.input_properties
        assert mfcw in dycore.input_properties
        assert mfpw in dycore.input_properties
        assert len(dycore.input_properties) == 9
    else:
        assert len(dycore.input_properties) == 6

    assert "air_isentropic_density" in dycore.input_tendency_properties
    assert "x_momentum_isentropic" in dycore.input_tendency_properties
    assert "y_momentum_isentropic" in dycore.input_tendency_properties
    if moist:
        assert mfwv in dycore.input_tendency_properties
        assert mfwv in dycore.input_tendency_properties
        assert mfpw in dycore.input_tendency_properties
        assert len(dycore.input_tendency_properties) == 6
    else:
        assert len(dycore.input_tendency_properties) == 3

    assert "air_isentropic_density" in dycore.output_properties
    assert "x_momentum_isentropic" in dycore.output_properties
    assert "x_velocity_at_u_locations" in dycore.output_properties
    assert "y_momentum_isentropic" in dycore.output_properties
    assert "y_velocity_at_v_locations" in dycore.output_properties
    if moist:
        assert mfwv in dycore.output_properties
        assert mfcw in dycore.output_properties
        assert mfpw in dycore.output_properties
        assert len(dycore.output_properties) == 8
    else:
        assert len(dycore.output_properties) == 5

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    state_new = dycore(state, {}, timestep)

    for key in state:
        # with subtests.test(key=key):
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_arrays(state[key].data, state_dc[key].data)

    assert "time" in state_new
    compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 9
    else:
        assert len(state_new) == 6

    raw_state_now = {"time": state["time"]}
    for name, props in dycore.input_properties.items():
        raw_state_now[name] = state[name].to_units(props["units"]).data
    raw_state_now["air_pressure_on_interface_levels"] = (
        state["air_pressure_on_interface_levels"].to_units("Pa").data
    )

    raw_state_new_val = rk3ws_step(
        domain,
        moist,
        timestep,
        raw_state_now,
        {},
        damp,
        damp_at_every_stage,
        vd,
        smooth,
        smooth_at_every_stage,
        hs,
        eps,
        backend=backend,
        storage_options=so,
    )

    for name in state_new:
        # with subtests.test(name=name):
        if name != "time":
            compare_arrays(
                state_new[name].data,
                raw_state_new_val[name],
                slice=(slice(nx), slice(ny), slice(nz)),
                # atol=1e-6,
            )


# @reproduce_failure('4.28.0', b'AXicY2BgZEAGzOh8NhQubyeSPKrKgQIAYKMApw==')
@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize(
    "backend", conf.backend_debug.union(conf.gt_cpu_backend)
)
@pytest.mark.parametrize("dtype", conf.dtype)
def test2(data, backend, dtype, subtests):
    """
    - Slow tendencies: yes
    - Intermediate tendencies: no
    - Intermediate diagnostics: no
    - Sub-stepping: no
    """
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
            xaxis_length=(1, 50),
            yaxis_length=(1, 50),
            zaxis_length=(2, 50),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
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

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    eps = data.draw(st_floats(min_value=0, max_value=1), label="eps")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd"):
        tendencies["air_isentropic_density"] = deepcopy_dataarray(
            state["air_isentropic_density"]
        )
        tendencies["air_isentropic_density"].attrs[
            "units"
        ] = "kg m^-2 K^-1 s^-1"
    if data.draw(hyp_st.booleans(), label="su_tnd"):
        tendencies["x_momentum_isentropic"] = deepcopy_dataarray(
            state["x_momentum_isentropic"]
        )
        tendencies["x_momentum_isentropic"].attrs[
            "units"
        ] = "kg m^-1 K^-1 s^-2"
    if data.draw(hyp_st.booleans(), label="sv_tnd"):
        tendencies["y_momentum_isentropic"] = deepcopy_dataarray(
            state["y_momentum_isentropic"]
        )
        tendencies["y_momentum_isentropic"].attrs[
            "units"
        ] = "kg m^-1 K^-1 s^-2"
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd"):
            tendencies[mfwv] = deepcopy_dataarray(state[mfwv])
            tendencies[mfwv].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qc_tnd"):
            tendencies[mfcw] = deepcopy_dataarray(state[mfcw])
            tendencies[mfcw].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qr_tnd"):
            tendencies[mfpw] = deepcopy_dataarray(state[mfpw])
            tendencies[mfpw].attrs["units"] = "g g^-1 s^-1"

    damp = data.draw(hyp_st.booleans(), label="damp")
    damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="damp_depth"
    )
    damp_at_every_stage = data.draw(
        hyp_st.booleans(), label="damp_at_every_stage"
    )

    smooth = False  # data.draw(hyp_st.booleans(), label="smooth")
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz),
        label="smooth_damp_depth",
    )
    smooth_at_every_stage = data.draw(
        hyp_st.booleans(), label="smooth_at_every_stage"
    )

    # ========================================
    # test bed
    # ========================================
    domain.horizontal_boundary.reference_state = state

    vd = VerticalDamping.factory(
        "rayleigh",
        grid,
        damp_depth,
        0.0002,
        "s",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    hs = HorizontalSmoothing.factory(
        "second_order",
        storage_shape,
        0.03,
        0.24,
        smooth_damp_depth,
        hb.nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    pt_raw = state["air_pressure_on_interface_levels"].data[0, 0, 0]
    pt = DataArray(
        pt_raw,
        attrs={
            "units": state["air_pressure_on_interface_levels"].attrs["units"]
        },
    )

    dycore = IsentropicDynamicalCore(
        domain,
        fast_tendency_component=None,
        fast_diagnostic_component=None,
        superfast_tendency_component=None,
        superfast_diagnostic_component=None,
        moist=moist,
        time_integration_scheme="rk3ws_si",
        horizontal_flux_scheme="fifth_order_upwind",
        time_integration_properties={"pt": pt, "eps": eps},
        damp=damp,
        damp_type="rayleigh",
        damp_depth=damp_depth,
        damp_max=0.0002,
        damp_at_every_stage=damp_at_every_stage,
        smooth=smooth,
        smooth_type="second_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=smooth_damp_depth,
        smooth_at_every_stage=smooth_at_every_stage,
        smooth_moist=smooth,
        smooth_moist_type="second_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=smooth_damp_depth,
        smooth_moist_at_every_stage=smooth_at_every_stage,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    #
    # test properties
    #
    assert "air_isentropic_density" in dycore.input_properties
    assert "montgomery_potential" in dycore.input_properties
    assert "x_momentum_isentropic" in dycore.input_properties
    assert "x_velocity_at_u_locations" in dycore.input_properties
    assert "y_momentum_isentropic" in dycore.input_properties
    assert "y_velocity_at_v_locations" in dycore.input_properties
    if moist:
        assert mfwv in dycore.input_properties
        assert mfcw in dycore.input_properties
        assert mfpw in dycore.input_properties
        assert len(dycore.input_properties) == 9
    else:
        assert len(dycore.input_properties) == 6

    assert "air_isentropic_density" in dycore.input_tendency_properties
    assert "x_momentum_isentropic" in dycore.input_tendency_properties
    assert "y_momentum_isentropic" in dycore.input_tendency_properties
    if moist:
        assert mfwv in dycore.input_tendency_properties
        assert mfwv in dycore.input_tendency_properties
        assert mfpw in dycore.input_tendency_properties
        assert len(dycore.input_tendency_properties) == 6
    else:
        assert len(dycore.input_tendency_properties) == 3

    assert "air_isentropic_density" in dycore.output_properties
    assert "x_momentum_isentropic" in dycore.output_properties
    assert "x_velocity_at_u_locations" in dycore.output_properties
    assert "y_momentum_isentropic" in dycore.output_properties
    assert "y_velocity_at_v_locations" in dycore.output_properties
    if moist:
        assert mfwv in dycore.output_properties
        assert mfcw in dycore.output_properties
        assert mfpw in dycore.output_properties
        assert len(dycore.output_properties) == 8
    else:
        assert len(dycore.output_properties) == 5

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    state_new = dycore(state, tendencies, timestep)

    for key in state:
        # with subtests.test(key=key):
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_arrays(state[key].data, state_dc[key].data)

    assert "time" in state_new
    compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 9
    else:
        assert len(state_new) == 6

    raw_state_now = {"time": state["time"]}
    for name, props in dycore.input_properties.items():
        raw_state_now[name] = state[name].to_units(props["units"]).data
    raw_state_now["air_pressure_on_interface_levels"] = (
        state["air_pressure_on_interface_levels"].to_units("Pa").data
    )

    raw_tendencies = {}
    for name, props in dycore.input_tendency_properties.items():
        if name in tendencies:
            raw_tendencies[name] = (
                tendencies[name].to_units(props["units"]).data
            )

    raw_state_new_val = rk3ws_step(
        domain,
        moist,
        timestep,
        raw_state_now,
        raw_tendencies,
        damp,
        damp_at_every_stage,
        vd,
        smooth,
        smooth_at_every_stage,
        hs,
        eps,
        backend=backend,
        storage_options=so,
    )

    for name in state_new:
        # with subtests.test(name=name):
        if name != "time":
            compare_arrays(
                state_new[name].data,
                raw_state_new_val[name],
                slice=(slice(nx), slice(ny), slice(nz)),
                # atol=1e-6,
            )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize(
    "backend", conf.backend_debug.union(conf.gt_cpu_backend)
)
@pytest.mark.parametrize("dtype", conf.dtype)
def test3(data, backend, dtype, subtests):
    """
    - Slow tendencies: yes
    - Intermediate tendencies: yes
    - Intermediate diagnostics: no
    - Sub-stepping: no
    """
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
            xaxis_length=(1, 25),
            yaxis_length=(1, 25),
            zaxis_length=(2, 15),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
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

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    eps = data.draw(st_floats(min_value=0, max_value=1), label="eps")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd"):
        tendencies["air_isentropic_density"] = deepcopy_dataarray(
            state["air_isentropic_density"]
        )
        tendencies["air_isentropic_density"].attrs[
            "units"
        ] = "kg m^-2 K^-1 s^-1"
    if data.draw(hyp_st.booleans(), label="su_tnd"):
        tendencies["x_momentum_isentropic"] = deepcopy_dataarray(
            state["x_momentum_isentropic"]
        )
        tendencies["x_momentum_isentropic"].attrs[
            "units"
        ] = "kg m^-1 K^-1 s^-2"
    if data.draw(hyp_st.booleans(), label="sv_tnd"):
        tendencies["y_momentum_isentropic"] = deepcopy_dataarray(
            state["y_momentum_isentropic"]
        )
        tendencies["y_momentum_isentropic"].attrs[
            "units"
        ] = "kg m^-1 K^-1 s^-2"
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd"):
            tendencies[mfwv] = deepcopy_dataarray(state[mfwv])
            tendencies[mfwv].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qc_tnd"):
            tendencies[mfcw] = deepcopy_dataarray(state[mfcw])
            tendencies[mfcw].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qr_tnd"):
            tendencies[mfpw] = deepcopy_dataarray(state[mfpw])
            tendencies[mfpw].attrs["units"] = "g g^-1 s^-1"

    damp = data.draw(hyp_st.booleans(), label="damp")
    damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="damp_depth"
    )
    damp_at_every_stage = data.draw(
        hyp_st.booleans(), label="damp_at_every_stage"
    )

    smooth = False  # data.draw(hyp_st.booleans(), label="smooth")
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz),
        label="smooth_damp_depth",
    )
    smooth_at_every_stage = data.draw(
        hyp_st.booleans(), label="smooth_at_every_stage"
    )

    # ========================================
    # test bed
    # ========================================
    domain.horizontal_boundary.reference_state = state
    hb_np = domain.copy(backend="numpy").horizontal_boundary

    vd = VerticalDamping.factory(
        "rayleigh",
        grid,
        damp_depth,
        0.0002,
        "s",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    hs = HorizontalSmoothing.factory(
        "second_order",
        storage_shape,
        0.03,
        0.24,
        smooth_damp_depth,
        hb.nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    cf = IsentropicConservativeCoriolis(
        domain,
        grid_type="numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    cfv = cf._f

    pt_raw = state["air_pressure_on_interface_levels"].data[0, 0, 0]
    pt = DataArray(
        pt_raw,
        attrs={
            "units": state["air_pressure_on_interface_levels"].attrs["units"]
        },
    )

    dycore = IsentropicDynamicalCore(
        domain,
        fast_tendency_component=cf,
        fast_diagnostic_component=None,
        superfast_tendency_component=None,
        superfast_diagnostic_component=None,
        moist=moist,
        time_integration_scheme="rk3ws_si",
        horizontal_flux_scheme="fifth_order_upwind",
        time_integration_properties={"pt": pt, "eps": eps},
        damp=damp,
        damp_type="rayleigh",
        damp_depth=damp_depth,
        damp_max=0.0002,
        damp_at_every_stage=damp_at_every_stage,
        smooth=smooth,
        smooth_type="second_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=smooth_damp_depth,
        smooth_at_every_stage=smooth_at_every_stage,
        smooth_moist=smooth,
        smooth_moist_type="second_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=smooth_damp_depth,
        smooth_moist_at_every_stage=smooth_at_every_stage,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    #
    # test properties
    #
    assert "air_isentropic_density" in dycore.input_properties
    assert "montgomery_potential" in dycore.input_properties
    assert "x_momentum_isentropic" in dycore.input_properties
    assert "x_velocity_at_u_locations" in dycore.input_properties
    assert "y_momentum_isentropic" in dycore.input_properties
    assert "y_velocity_at_v_locations" in dycore.input_properties
    if moist:
        assert mfwv in dycore.input_properties
        assert mfcw in dycore.input_properties
        assert mfpw in dycore.input_properties
        assert len(dycore.input_properties) == 9
    else:
        assert len(dycore.input_properties) == 6

    assert "air_isentropic_density" in dycore.input_tendency_properties
    assert "x_momentum_isentropic" in dycore.input_tendency_properties
    assert "y_momentum_isentropic" in dycore.input_tendency_properties
    if moist:
        assert mfwv in dycore.input_tendency_properties
        assert mfwv in dycore.input_tendency_properties
        assert mfpw in dycore.input_tendency_properties
        assert len(dycore.input_tendency_properties) == 6
    else:
        assert len(dycore.input_tendency_properties) == 3

    assert "air_isentropic_density" in dycore.output_properties
    assert "x_momentum_isentropic" in dycore.output_properties
    assert "x_velocity_at_u_locations" in dycore.output_properties
    assert "y_momentum_isentropic" in dycore.output_properties
    assert "y_velocity_at_v_locations" in dycore.output_properties
    if moist:
        assert mfwv in dycore.output_properties
        assert mfcw in dycore.output_properties
        assert mfpw in dycore.output_properties
        assert len(dycore.output_properties) == 8
    else:
        assert len(dycore.output_properties) == 5

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    state_new = dycore(state, tendencies, timestep)

    for key in state:
        # with subtests.test(key=key):
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_arrays(state[key].data, state_dc[key].data)

    assert "time" in state_new
    compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 9
    else:
        assert len(state_new) == 6

    raw_state_0 = {"time": state["time"]}
    raw_state_0["air_pressure_on_interface_levels"] = (
        state["air_pressure_on_interface_levels"].to_units("Pa").data
    )
    for name, props in dycore.input_properties.items():
        raw_state_0[name] = state[name].to_units(props["units"]).data
    for name, props in dycore.output_properties.items():
        if name not in dycore.input_properties:
            raw_state_0[name] = state[name].to_units(props["units"]).data
    if moist:
        raw_state_0["isentropic_density_of_water_vapor"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )
        raw_state_0["isentropic_density_of_cloud_liquid_water"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )
        raw_state_0["isentropic_density_of_precipitation_water"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )

    raw_tendencies = {}
    for name, props in dycore.input_tendency_properties.items():
        if name in tendencies:
            raw_tendencies[name] = (
                tendencies[name].to_units(props["units"]).data
            )

    if "x_momentum_isentropic" not in raw_tendencies:
        raw_tendencies["x_momentum_isentropic"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )
    if "y_momentum_isentropic" not in raw_tendencies:
        raw_tendencies["y_momentum_isentropic"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )

    raw_tendencies_dc = deepcopy_array_dict(raw_tendencies)

    raw_state_1 = deepcopy_array_dict(raw_state_0)
    raw_state_2 = deepcopy_array_dict(raw_state_0)
    raw_state_3 = deepcopy_array_dict(raw_state_0)

    names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    field_properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "y_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    }
    if moist:
        field_properties.update(
            {
                mfwv: {"units": "g g^-1"},
                mfcw: {"units": "g g^-1"},
                mfpw: {"units": "g g^-1"},
            }
        )

    state_ref = hb.reference_state
    raw_state_ref = {}
    for name in field_properties:
        raw_state_ref[name] = (
            state_ref[name].to_units(field_properties[name]["units"]).data
        )

    #
    # stage 0
    #
    raw_tendencies["x_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["x_momentum_isentropic"])
        + cfv * to_numpy(raw_state_0["y_momentum_isentropic"]),
        storage_options=so,
    )
    raw_tendencies["y_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["y_momentum_isentropic"])
        - cfv * to_numpy(raw_state_0["x_momentum_isentropic"]),
        storage_options=so,
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        0,
        timestep,
        grid,
        raw_state_0,
        raw_state_0,
        raw_state_ref,
        raw_tendencies,
        raw_state_1,
        field_properties,
        hb,
        hb_np,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
        backend=backend,
        storage_options=so,
    )

    #
    # stage 1
    #
    raw_tendencies["x_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["x_momentum_isentropic"])
        + cfv * to_numpy(raw_state_1["y_momentum_isentropic"]),
        storage_options=so,
    )
    raw_tendencies["y_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["y_momentum_isentropic"])
        - cfv * to_numpy(raw_state_1["x_momentum_isentropic"]),
        storage_options=so,
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        1,
        timestep,
        grid,
        raw_state_0,
        raw_state_1,
        raw_state_ref,
        raw_tendencies,
        raw_state_2,
        field_properties,
        hb,
        hb_np,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
        backend=backend,
        storage_options=so,
    )

    #
    # stage 2
    #
    raw_tendencies["x_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["x_momentum_isentropic"])
        + cfv * to_numpy(raw_state_2["y_momentum_isentropic"]),
        storage_options=so,
    )
    raw_tendencies["y_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["y_momentum_isentropic"])
        - cfv * to_numpy(raw_state_2["x_momentum_isentropic"]),
        storage_options=so,
    )

    rk3wssi_stage(
        2,
        timestep,
        grid,
        raw_state_0,
        raw_state_2,
        raw_state_ref,
        raw_tendencies,
        raw_state_3,
        field_properties,
        hb,
        hb_np,
        moist,
        damp,
        vd,
        smooth,
        hs,
        eps,
        backend=backend,
        storage_options=so,
    )

    for name in state_new:
        # with subtests.test(name=name):
        if name != "time":
            compare_arrays(
                state_new[name].data,
                raw_state_3[name],
                slice=(slice(nx), slice(ny), slice(nz)),
                # atol=1e-6,
            )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize(
    "backend", conf.backend_debug.union(conf.gt_cpu_backend)
)
@pytest.mark.parametrize("dtype", conf.dtype)
def test4(data, backend, dtype, subtests):
    """
    - Slow tendencies: yes
    - Intermediate tendencies: yes
    - Intermediate diagnostics: yes
    - Sub-stepping: no
    """
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
            xaxis_length=(1, 50),
            yaxis_length=(1, 50),
            zaxis_length=(2, 50),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
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

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    eps = data.draw(st_floats(min_value=0, max_value=1), label="eps")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd"):
        tendencies["air_isentropic_density"] = deepcopy_dataarray(
            state["air_isentropic_density"]
        )
        tendencies["air_isentropic_density"].attrs[
            "units"
        ] = "kg m^-2 K^-1 s^-1"
    if data.draw(hyp_st.booleans(), label="su_tnd"):
        tendencies["x_momentum_isentropic"] = deepcopy_dataarray(
            state["x_momentum_isentropic"]
        )
        tendencies["x_momentum_isentropic"].attrs[
            "units"
        ] = "kg m^-1 K^-1 s^-2"
    if data.draw(hyp_st.booleans(), label="sv_tnd"):
        tendencies["y_momentum_isentropic"] = deepcopy_dataarray(
            state["y_momentum_isentropic"]
        )
        tendencies["y_momentum_isentropic"].attrs[
            "units"
        ] = "kg m^-1 K^-1 s^-2"
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd"):
            tendencies[mfwv] = deepcopy_dataarray(state[mfwv])
            tendencies[mfwv].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qc_tnd"):
            tendencies[mfcw] = deepcopy_dataarray(state[mfcw])
            tendencies[mfcw].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qr_tnd"):
            tendencies[mfpw] = deepcopy_dataarray(state[mfpw])
            tendencies[mfpw].attrs["units"] = "g g^-1 s^-1"

    damp = data.draw(hyp_st.booleans(), label="damp")
    damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="damp_depth"
    )
    damp_at_every_stage = data.draw(
        hyp_st.booleans(), label="damp_at_every_stage"
    )

    smooth = False  # data.draw(hyp_st.booleans(), label="smooth")
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz),
        label="smooth_damp_depth",
    )
    smooth_at_every_stage = data.draw(
        hyp_st.booleans(), label="smooth_at_every_stage"
    )

    # ========================================
    # test bed
    # ========================================
    domain.horizontal_boundary.reference_state = state
    hb_np = domain.copy(backend="numpy").horizontal_boundary

    vd = VerticalDamping.factory(
        "rayleigh",
        grid,
        damp_depth,
        0.0002,
        "s",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    hs = HorizontalSmoothing.factory(
        "second_order",
        storage_shape,
        0.03,
        0.24,
        smooth_damp_depth,
        hb.nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    cf = IsentropicConservativeCoriolis(
        domain,
        grid_type="numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    cfv = cf._f

    pt_raw = state["air_pressure_on_interface_levels"].data[0, 0, 0]
    pt = DataArray(
        pt_raw,
        attrs={
            "units": state["air_pressure_on_interface_levels"].attrs["units"]
        },
    )

    dv = IsentropicDiagnostics(
        domain,
        "numerical",
        moist,
        pt,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    rdv = RawIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    dycore = IsentropicDynamicalCore(
        domain,
        fast_tendency_component=cf,
        fast_diagnostic_component=dv,
        superfast_tendency_component=None,
        superfast_diagnostic_component=None,
        moist=moist,
        time_integration_scheme="rk3ws_si",
        horizontal_flux_scheme="fifth_order_upwind",
        time_integration_properties={"pt": pt, "eps": eps},
        damp=damp,
        damp_type="rayleigh",
        damp_depth=damp_depth,
        damp_max=0.0002,
        damp_at_every_stage=damp_at_every_stage,
        smooth=smooth,
        smooth_type="second_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=smooth_damp_depth,
        smooth_at_every_stage=smooth_at_every_stage,
        smooth_moist=smooth,
        smooth_moist_type="second_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=smooth_damp_depth,
        smooth_moist_at_every_stage=smooth_at_every_stage,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    #
    # test properties
    #
    assert "air_isentropic_density" in dycore.input_properties
    assert "montgomery_potential" in dycore.input_properties
    assert "x_momentum_isentropic" in dycore.input_properties
    assert "x_velocity_at_u_locations" in dycore.input_properties
    assert "y_momentum_isentropic" in dycore.input_properties
    assert "y_velocity_at_v_locations" in dycore.input_properties
    if moist:
        assert mfwv in dycore.input_properties
        assert mfcw in dycore.input_properties
        assert mfpw in dycore.input_properties
        assert len(dycore.input_properties) == 9
    else:
        assert len(dycore.input_properties) == 6

    assert "air_isentropic_density" in dycore.input_tendency_properties
    assert "x_momentum_isentropic" in dycore.input_tendency_properties
    assert "y_momentum_isentropic" in dycore.input_tendency_properties
    if moist:
        assert mfwv in dycore.input_tendency_properties
        assert mfwv in dycore.input_tendency_properties
        assert mfpw in dycore.input_tendency_properties
        assert len(dycore.input_tendency_properties) == 6
    else:
        assert len(dycore.input_tendency_properties) == 3

    assert "air_isentropic_density" in dycore.output_properties
    assert "air_pressure_on_interface_levels" in dycore.output_properties
    assert "exner_function_on_interface_levels" in dycore.output_properties
    assert "height_on_interface_levels" in dycore.output_properties
    assert "montgomery_potential" in dycore.output_properties
    assert "x_momentum_isentropic" in dycore.output_properties
    assert "x_velocity_at_u_locations" in dycore.output_properties
    assert "y_momentum_isentropic" in dycore.output_properties
    assert "y_velocity_at_v_locations" in dycore.output_properties
    if moist:
        assert "air_density" in dycore.output_properties
        assert "air_temperature" in dycore.output_properties
        assert mfwv in dycore.output_properties
        assert mfcw in dycore.output_properties
        assert mfpw in dycore.output_properties
        assert len(dycore.output_properties) == 14
    else:
        assert len(dycore.output_properties) == 9

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    state_new = dycore(state, tendencies, timestep)

    for key in state:
        # with subtests.test(key=key):
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_arrays(state[key].data, state_dc[key].data)

    assert "time" in state_new
    compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "air_pressure_on_interface_levels" in state_new
    assert "exner_function_on_interface_levels" in state_new
    assert "height_on_interface_levels" in state_new
    assert "montgomery_potential" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert "air_density" in state_new
        assert "air_temperature" in state_new
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 15
    else:
        assert len(state_new) == 10

    raw_state_0 = {"time": state["time"]}
    for name, props in dycore.input_properties.items():
        raw_state_0[name] = state[name].to_units(props["units"]).data
    for name, props in dycore.output_properties.items():
        if name not in dycore.input_properties:
            raw_state_0[name] = state[name].to_units(props["units"]).data
    if moist:
        raw_state_0["isentropic_density_of_water_vapor"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )
        raw_state_0["isentropic_density_of_cloud_liquid_water"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )
        raw_state_0["isentropic_density_of_precipitation_water"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )
    raw_state_0["air_pressure_on_interface_levels"] = (
        state["air_pressure_on_interface_levels"].to_units("Pa").data
    )

    raw_tendencies = {}
    for name, props in dycore.input_tendency_properties.items():
        if name in tendencies:
            raw_tendencies[name] = (
                tendencies[name].to_units(props["units"]).data
            )

    if "x_momentum_isentropic" not in raw_tendencies:
        raw_tendencies["x_momentum_isentropic"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )
    if "y_momentum_isentropic" not in raw_tendencies:
        raw_tendencies["y_momentum_isentropic"] = zeros(
            backend, shape=storage_shape, storage_options=so
        )

    raw_tendencies_dc = deepcopy_array_dict(raw_tendencies)

    raw_state_1 = deepcopy_array_dict(raw_state_0)
    raw_state_2 = deepcopy_array_dict(raw_state_0)
    raw_state_3 = deepcopy_array_dict(raw_state_0)

    names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    field_properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "y_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    }
    if moist:
        field_properties.update(
            {
                mfwv: {"units": "g g^-1"},
                mfcw: {"units": "g g^-1"},
                mfpw: {"units": "g g^-1"},
            }
        )

    state_ref = hb.reference_state
    raw_state_ref = {}
    for name in field_properties:
        raw_state_ref[name] = (
            state_ref[name].to_units(field_properties[name]["units"]).data
        )

    #
    # stage 0
    #
    raw_tendencies["x_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["x_momentum_isentropic"])
        + cfv * to_numpy(raw_state_0["y_momentum_isentropic"]),
        storage_options=so,
    )
    raw_tendencies["y_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["y_momentum_isentropic"])
        - cfv * to_numpy(raw_state_0["x_momentum_isentropic"]),
        storage_options=so,
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        0,
        timestep,
        grid,
        raw_state_0,
        raw_state_0,
        raw_state_ref,
        raw_tendencies,
        raw_state_1,
        field_properties,
        hb,
        hb_np,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
        backend=backend,
        storage_options=so,
    )

    rdv.get_diagnostic_variables(
        raw_state_1["air_isentropic_density"],
        pt_raw,
        raw_state_1["air_pressure_on_interface_levels"],
        raw_state_1["exner_function_on_interface_levels"],
        raw_state_1["montgomery_potential"],
        raw_state_1["height_on_interface_levels"],
    )
    if moist:
        rdv.get_density_and_temperature(
            raw_state_1["air_isentropic_density"],
            raw_state_1["exner_function_on_interface_levels"],
            raw_state_1["height_on_interface_levels"],
            raw_state_1["air_density"],
            raw_state_1["air_temperature"],
        )

    #
    # stage 1
    #
    raw_tendencies["x_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["x_momentum_isentropic"])
        + cfv * to_numpy(raw_state_1["y_momentum_isentropic"]),
        storage_options=so,
    )
    raw_tendencies["y_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["y_momentum_isentropic"])
        - cfv * to_numpy(raw_state_1["x_momentum_isentropic"]),
        storage_options=so,
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        1,
        timestep,
        grid,
        raw_state_0,
        raw_state_1,
        raw_state_ref,
        raw_tendencies,
        raw_state_2,
        field_properties,
        hb,
        hb_np,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
        backend=backend,
        storage_options=so,
    )

    rdv.get_diagnostic_variables(
        raw_state_2["air_isentropic_density"],
        pt_raw,
        raw_state_2["air_pressure_on_interface_levels"],
        raw_state_2["exner_function_on_interface_levels"],
        raw_state_2["montgomery_potential"],
        raw_state_2["height_on_interface_levels"],
    )
    if moist:
        rdv.get_density_and_temperature(
            raw_state_2["air_isentropic_density"],
            raw_state_2["exner_function_on_interface_levels"],
            raw_state_2["height_on_interface_levels"],
            raw_state_2["air_density"],
            raw_state_2["air_temperature"],
        )

    #
    # stage 2
    #
    raw_tendencies["x_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["x_momentum_isentropic"])
        + cfv * to_numpy(raw_state_2["y_momentum_isentropic"]),
        storage_options=so,
    )
    raw_tendencies["y_momentum_isentropic"][...] = as_storage(
        backend,
        data=to_numpy(raw_tendencies_dc["y_momentum_isentropic"])
        - cfv * to_numpy(raw_state_2["x_momentum_isentropic"]),
        storage_options=so,
    )

    rk3wssi_stage(
        2,
        timestep,
        grid,
        raw_state_0,
        raw_state_2,
        raw_state_ref,
        raw_tendencies,
        raw_state_3,
        field_properties,
        hb,
        hb_np,
        moist,
        damp,
        vd,
        smooth,
        hs,
        eps,
        backend=backend,
        storage_options=so,
    )

    rdv.get_diagnostic_variables(
        raw_state_3["air_isentropic_density"],
        pt_raw,
        raw_state_3["air_pressure_on_interface_levels"],
        raw_state_3["exner_function_on_interface_levels"],
        raw_state_3["montgomery_potential"],
        raw_state_3["height_on_interface_levels"],
    )
    if moist:
        rdv.get_density_and_temperature(
            raw_state_3["air_isentropic_density"],
            raw_state_3["exner_function_on_interface_levels"],
            raw_state_3["height_on_interface_levels"],
            raw_state_3["air_density"],
            raw_state_3["air_temperature"],
        )

    for name in state_new:
        # with subtests.test(name=name):
        if name != "time":
            compare_arrays(
                state_new[name].data,
                raw_state_3[name],
                slice=(slice(nx), slice(ny), slice(nz)),
                # atol=1e-6,
            )


class FooTendencyComponent(TendencyComponent):
    def __init__(self, domain):
        super().__init__(domain, "numerical")

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        return_dict = {
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-1",
            },
        }
        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        return_dict = {
            "x_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
            "y_momentum_isentropic": {
                "dims": dims,
                "units": "kg m^-1 K^-1 s^-2",
            },
        }
        return return_dict

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state):
        su = state["x_momentum_isentropic"]
        sv = state["y_momentum_isentropic"]
        out_su = 0.1 * su
        out_sv = 0.1 * sv
        tendencies = {
            "x_momentum_isentropic": out_su,
            "y_momentum_isentropic": out_sv,
        }
        diagnostics = {}
        return tendencies, diagnostics


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def _test5(data, backend, dtype, subtests):
    """
    - Slow tendencies: yes
    - Intermediate tendencies: yes
    - Intermediate diagnostics: yes, but computing tendencies
    - Sub-stepping: no
    """
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 25),
            yaxis_length=(1, 25),
            zaxis_length=(2, 15),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=moist,
            backend=backend,
            aligned_index=aligned_index,
            storage_shape=storage_shape,
        ),
        label="state",
    )
    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    eps = data.draw(st_floats(min_value=0, max_value=1), label="eps")

    tendencies = {}
    if data.draw(hyp_st.booleans(), label="s_tnd"):
        tendencies["air_isentropic_density"] = deepcopy_dataarray(
            state["air_isentropic_density"]
        )
        tendencies["air_isentropic_density"].attrs[
            "units"
        ] = "kg m^-2 K^-1 s^-1"
    if data.draw(hyp_st.booleans(), label="su_tnd"):
        tendencies["x_momentum_isentropic"] = deepcopy_dataarray(
            state["x_momentum_isentropic"]
        )
        tendencies["x_momentum_isentropic"].attrs[
            "units"
        ] = "kg m^-1 K^-1 s^-2"
    if data.draw(hyp_st.booleans(), label="sv_tnd"):
        tendencies["y_momentum_isentropic"] = deepcopy_dataarray(
            state["y_momentum_isentropic"]
        )
        tendencies["y_momentum_isentropic"].attrs[
            "units"
        ] = "kg m^-1 K^-1 s^-2"
    if moist:
        if data.draw(hyp_st.booleans(), label="qv_tnd"):
            tendencies[mfwv] = deepcopy_dataarray(state[mfwv])
            tendencies[mfwv].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qc_tnd"):
            tendencies[mfcw] = deepcopy_dataarray(state[mfcw])
            tendencies[mfcw].attrs["units"] = "g g^-1 s^-1"
        if data.draw(hyp_st.booleans(), label="qr_tnd"):
            tendencies[mfpw] = deepcopy_dataarray(state[mfpw])
            tendencies[mfpw].attrs["units"] = "g g^-1 s^-1"

    damp = data.draw(hyp_st.booleans(), label="damp")
    damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="damp_depth"
    )
    damp_at_every_stage = data.draw(
        hyp_st.booleans(), label="damp_at_every_stage"
    )

    smooth = data.draw(hyp_st.booleans(), label="smooth")
    smooth_damp_depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz),
        label="smooth_damp_depth",
    )
    smooth_at_every_stage = data.draw(
        hyp_st.booleans(), label="smooth_at_every_stage"
    )

    # ========================================
    # test bed
    # ========================================
    domain.horizontal_boundary.reference_state = state

    vd = VerticalDamping.factory(
        "rayleigh",
        grid,
        damp_depth,
        0.0002,
        "s",
        backend=backend,
        dtype=dtype,
        aligned_index=aligned_index,
        storage_shape=storage_shape,
    )
    hs = HorizontalSmoothing.factory(
        "second_order",
        storage_shape,
        0.03,
        0.24,
        smooth_damp_depth,
        hb.nb,
        backend=backend,
        dtype=dtype,
        aligned_index=aligned_index,
    )

    cf = IsentropicConservativeCoriolis(
        domain,
        grid_type="numerical",
        backend=backend,
        dtype=dtype,
        aligned_index=aligned_index,
        storage_shape=storage_shape,
    )
    cfv = cf._f

    foo = FooTendencyComponent(domain)

    dycore = IsentropicDynamicalCore(
        domain,
        fast_tendency_component=cf,
        fast_diagnostic_component=foo,
        superfast_tendency_component=None,
        superfast_diagnostic_component=None,
        moist=moist,
        time_integration_scheme="rk3ws_si",
        horizontal_flux_scheme="fifth_order_upwind",
        time_integration_properties={
            "pt": state["air_pressure_on_interface_levels"][0, 0, 0],
            "eps": eps,
        },
        damp=damp,
        damp_type="rayleigh",
        damp_depth=damp_depth,
        damp_max=0.0002,
        damp_at_every_stage=damp_at_every_stage,
        smooth=smooth,
        smooth_type="second_order",
        smooth_coeff=0.03,
        smooth_coeff_max=0.24,
        smooth_damp_depth=smooth_damp_depth,
        smooth_at_every_stage=smooth_at_every_stage,
        smooth_moist=smooth,
        smooth_moist_type="second_order",
        smooth_moist_coeff=0.03,
        smooth_moist_coeff_max=0.24,
        smooth_moist_damp_depth=smooth_damp_depth,
        smooth_moist_at_every_stage=smooth_at_every_stage,
        backend=backend,
        dtype=dtype,
        aligned_index=aligned_index,
        rebuild=False,
        storage_shape=storage_shape,
    )

    #
    # test properties
    #
    assert "air_isentropic_density" in dycore.input_properties
    assert "montgomery_potential" in dycore.input_properties
    assert "x_momentum_isentropic" in dycore.input_properties
    assert "x_velocity_at_u_locations" in dycore.input_properties
    assert "y_momentum_isentropic" in dycore.input_properties
    assert "y_velocity_at_v_locations" in dycore.input_properties
    if moist:
        assert mfwv in dycore.input_properties
        assert mfcw in dycore.input_properties
        assert mfpw in dycore.input_properties
        assert len(dycore.input_properties) == 9
    else:
        assert len(dycore.input_properties) == 6

    assert "air_isentropic_density" in dycore.input_tendency_properties
    assert "x_momentum_isentropic" in dycore.input_tendency_properties
    assert "y_momentum_isentropic" in dycore.input_tendency_properties
    if moist:
        assert mfwv in dycore.input_tendency_properties
        assert mfwv in dycore.input_tendency_properties
        assert mfpw in dycore.input_tendency_properties
        assert len(dycore.input_tendency_properties) == 6
    else:
        assert len(dycore.input_tendency_properties) == 3

    assert "air_isentropic_density" in dycore.output_properties
    assert "x_momentum_isentropic" in dycore.output_properties
    assert "x_velocity_at_u_locations" in dycore.output_properties
    assert "y_momentum_isentropic" in dycore.output_properties
    assert "y_velocity_at_v_locations" in dycore.output_properties
    if moist:
        assert mfwv in dycore.output_properties
        assert mfcw in dycore.output_properties
        assert mfpw in dycore.output_properties
        assert len(dycore.output_properties) == 8
    else:
        assert len(dycore.output_properties) == 5

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    state_new = dycore(state, tendencies, timestep)

    for key in state:
        # with subtests.test(key=key):
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_arrays(state[key], state_dc[key])

    assert "time" in state_new
    # compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 9
    else:
        assert len(state_new) == 6

    raw_state_0 = {"time": state["time"]}
    raw_state_0["air_pressure_on_interface_levels"] = (
        state["air_pressure_on_interface_levels"].to_units("Pa").data
    )
    for name, props in dycore.input_properties.items():
        raw_state_0[name] = state[name].to_units(props["units"]).data
    for name, props in dycore.output_properties.items():
        if name not in dycore.input_properties:
            raw_state_0[name] = state[name].to_units(props["units"]).data
    if moist:
        raw_state_0["isentropic_density_of_water_vapor"] = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            aligned_index=aligned_index,
        )
        raw_state_0["isentropic_density_of_cloud_liquid_water"] = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            aligned_index=aligned_index,
        )
        raw_state_0["isentropic_density_of_precipitation_water"] = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            aligned_index=aligned_index,
        )

    raw_tendencies = {}
    for name, props in dycore.input_tendency_properties.items():
        if name in tendencies:
            raw_tendencies[name] = (
                tendencies[name].to_units(props["units"]).data
            )

    if "x_momentum_isentropic" not in raw_tendencies:
        raw_tendencies["x_momentum_isentropic"] = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            aligned_index=aligned_index,
        )
    if "y_momentum_isentropic" not in raw_tendencies:
        raw_tendencies["y_momentum_isentropic"] = zeros(
            storage_shape,
            backend=backend,
            dtype=dtype,
            aligned_index=aligned_index,
        )

    tendencies_dc = deepcopy_dataarray_dict(tendencies)
    raw_tendencies_dc = deepcopy_array_dict(raw_tendencies)

    raw_state_1 = deepcopy_array_dict(raw_state_0)
    raw_state_2 = deepcopy_array_dict(raw_state_0)
    raw_state_3 = deepcopy_array_dict(raw_state_0)
    raw_state_4 = deepcopy_array_dict(raw_state_0)
    raw_state_5 = deepcopy_array_dict(raw_state_0)
    raw_state_6 = deepcopy_array_dict(raw_state_0)

    names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        names.append("isentropic_density_of_water_vapor")
        names.append("isentropic_density_of_cloud_liquid_water")
        names.append("isentropic_density_of_precipitation_water")

    field_properties = {
        "air_isentropic_density": {"units": "kg m^-2 K^-1"},
        "x_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
        "y_momentum_isentropic": {"units": "kg m^-1 K^-1 s^-1"},
    }
    if moist:
        field_properties.update(
            {
                mfwv: {"units": "g g^-1"},
                mfcw: {"units": "g g^-1"},
                mfpw: {"units": "g g^-1"},
            }
        )

    state_ref = hb.reference_state
    raw_state_ref = {}
    for name in field_properties:
        raw_state_ref[name] = (
            state_ref[name].to_units(field_properties[name]["units"]).data
        )

    #
    # stage 0
    #
    su0 = raw_state_0["x_momentum_isentropic"]
    sv0 = raw_state_0["y_momentum_isentropic"]
    raw_tendencies["x_momentum_isentropic"][...] = (
        raw_tendencies_dc["x_momentum_isentropic"] + cfv * sv0[...]
    )
    raw_tendencies["y_momentum_isentropic"][...] = (
        raw_tendencies_dc["y_momentum_isentropic"] - cfv * su0[...]
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        0,
        timestep,
        grid,
        raw_state_0,
        raw_state_0,
        raw_state_ref,
        raw_tendencies,
        raw_state_1,
        field_properties,
        hb,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
    )

    #
    # stage 1
    #
    su1 = raw_state_1["x_momentum_isentropic"]
    sv1 = raw_state_1["y_momentum_isentropic"]
    raw_tendencies["x_momentum_isentropic"][...] = (
        raw_tendencies_dc["x_momentum_isentropic"]
        + cfv * sv1[...]
        + 0.1 * su1[...]
    )
    raw_tendencies["y_momentum_isentropic"][...] = (
        raw_tendencies_dc["y_momentum_isentropic"]
        - cfv * su1[...]
        + 0.1 * sv1[...]
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        1,
        timestep,
        grid,
        raw_state_0,
        raw_state_1,
        raw_state_ref,
        raw_tendencies,
        raw_state_2,
        field_properties,
        hb,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
    )

    #
    # stage 2
    #
    su2 = raw_state_2["x_momentum_isentropic"]
    sv2 = raw_state_2["y_momentum_isentropic"]
    raw_tendencies["x_momentum_isentropic"][...] = (
        raw_tendencies_dc["x_momentum_isentropic"]
        + cfv * sv2[...]
        + 0.1 * su2[...]
    )
    raw_tendencies["y_momentum_isentropic"][...] = (
        raw_tendencies_dc["y_momentum_isentropic"]
        - cfv * su2[...]
        + 0.1 * sv2[...]
    )

    rk3wssi_stage(
        2,
        timestep,
        grid,
        raw_state_0,
        raw_state_2,
        raw_state_ref,
        raw_tendencies,
        raw_state_3,
        field_properties,
        hb,
        moist,
        damp,
        vd,
        smooth,
        hs,
        eps,
    )

    for name in state_new:
        # with subtests.test(name=name):
        if name != "time":
            compare_arrays(
                state_new[name].data[:-1, :-1, :-1],
                raw_state_3[name][:-1, :-1, :-1],
                # atol=1e-6,
            )

    state_new = dycore(state, tendencies_dc, timestep)

    for key in state:
        # with subtests.test(key=key):
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_arrays(state[key], state_dc[key])

    assert "time" in state_new
    # compare_datetimes(state_new["time"], state["time"] + timestep)

    assert "air_isentropic_density" in state_new
    assert "x_momentum_isentropic" in state_new
    assert "x_velocity_at_u_locations" in state_new
    assert "y_momentum_isentropic" in state_new
    assert "y_velocity_at_v_locations" in state_new
    if moist:
        assert mfwv in state_new
        assert mfcw in state_new
        assert mfpw in state_new
        assert len(state_new) == 9
    else:
        assert len(state_new) == 6

    #
    # stage 3
    #
    su3 = raw_state_3["x_momentum_isentropic"]
    sv3 = raw_state_3["y_momentum_isentropic"]
    raw_tendencies["x_momentum_isentropic"][...] = (
        raw_tendencies_dc["x_momentum_isentropic"]
        + cfv * sv0[...]
        + 0.1 * su3[...]
    )
    raw_tendencies["y_momentum_isentropic"][...] = (
        raw_tendencies_dc["y_momentum_isentropic"]
        - cfv * su0[...]
        + 0.1 * sv3[...]
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        0,
        timestep,
        grid,
        raw_state_0,
        raw_state_0,
        raw_state_ref,
        raw_tendencies,
        raw_state_4,
        field_properties,
        hb,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
    )

    #
    # stage 4
    #
    su4 = raw_state_4["x_momentum_isentropic"]
    sv4 = raw_state_4["y_momentum_isentropic"]
    raw_tendencies["x_momentum_isentropic"][...] = (
        raw_tendencies_dc["x_momentum_isentropic"]
        + cfv * sv4[...]
        + 0.1 * su4[...]
    )
    raw_tendencies["y_momentum_isentropic"][...] = (
        raw_tendencies_dc["y_momentum_isentropic"]
        - cfv * su4[...]
        + 0.1 * sv4[...]
    )

    _damp = damp and damp_at_every_stage
    _smooth = smooth and smooth_at_every_stage
    rk3wssi_stage(
        1,
        timestep,
        grid,
        raw_state_0,
        raw_state_4,
        raw_state_ref,
        raw_tendencies,
        raw_state_5,
        field_properties,
        hb,
        moist,
        _damp,
        vd,
        _smooth,
        hs,
        eps,
    )

    #
    # stage 5
    #
    su5 = raw_state_5["x_momentum_isentropic"]
    sv5 = raw_state_5["y_momentum_isentropic"]
    raw_tendencies["x_momentum_isentropic"][...] = (
        raw_tendencies_dc["x_momentum_isentropic"]
        + cfv * sv5[...]
        + 0.1 * su5[...]
    )
    raw_tendencies["y_momentum_isentropic"][...] = (
        raw_tendencies_dc["y_momentum_isentropic"]
        - cfv * su5[...]
        + 0.1 * sv5[...]
    )

    rk3wssi_stage(
        2,
        timestep,
        grid,
        raw_state_0,
        raw_state_5,
        raw_state_ref,
        raw_tendencies,
        raw_state_6,
        field_properties,
        hb,
        moist,
        damp,
        vd,
        smooth,
        hs,
        eps,
    )

    for name in state_new:
        # with subtests.test(name=name):
        if name != "time":
            compare_arrays(
                state_new[name].data[:-1, :-1, :-1],
                raw_state_6[name][:-1, :-1, :-1],
                # atol=1e-6,
            )


if __name__ == "__main__":
    pytest.main([__file__])
