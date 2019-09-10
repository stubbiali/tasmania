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
from hypothesis import (
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
from hypothesis.extra.numpy import arrays as st_arrays
import numpy as np
from pint import UnitRegistry
import pytest

import gridtools as gt
from tasmania.python.isentropic.dynamics.diagnostics import (
    IsentropicDiagnostics as DynamicsIsentropicDiagnostics,
)
from tasmania.python.isentropic.physics.diagnostics import (
    IsentropicDiagnostics,
    IsentropicVelocityComponents,
)
from tasmania.python.utils.storage_utils import get_dataarray_3d, zeros

try:
    from .conf import backend as conf_backend, halo as conf_halo
    from .utils import (
        compare_datetimes,
        compare_arrays,
        compare_dataarrays,
        st_floats,
        st_one_of,
        st_domain,
        st_physical_grid,
        st_isentropic_state_f,
        st_raw_field,
    )
except ModuleNotFoundError:
    from conf import backend as conf_backend, halo as conf_halo
    from utils import (
        compare_datetimes,
        compare_arrays,
        compare_dataarrays,
        st_floats,
        st_one_of,
        st_domain,
        st_physical_grid,
        st_isentropic_state_f,
        st_raw_field,
    )


conf_backend = ("gtx86", )
backend_opts = {"max_region_offset": 3, "verbose": False}


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_diagnostic_variables(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(zaxis_name="z"), label="grid")
    assume(grid.nz > 1)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    s = data.draw(
        st_raw_field(
            storage_shape,
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            halo=halo,
        ),
        label="s",
    )
    pt = data.draw(st_floats(min_value=1, max_value=1e5), label="pt")

    # ========================================
    # test bed
    # ========================================
    p = zeros(storage_shape, backend, dtype, halo=halo)
    exn = zeros(storage_shape, backend, dtype, halo=halo)
    mtg = zeros(storage_shape, backend, dtype, halo=halo)
    h = zeros(storage_shape, backend, dtype, halo=halo)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_opts=backend_opts,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )
    did.get_diagnostic_variables(s, pt, p, exn, mtg, h)

    cp = did._pcs["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = did._pcs["air_pressure_at_sea_level"]
    rd = did._pcs["gas_constant_of_dry_air"]
    g = did._pcs["gravitational_acceleration"]

    dz = grid.dz.to_units("K").values.item()
    topo = grid.topography.profile.to_units("m").values

    # pressure
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s[:nx, :ny, k - 1]
    assert np.allclose(p[:nx, :ny, :], p_val)

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)
    assert np.allclose(exn[:nx, :ny, : nz + 1], exn_val)

    # montgomery
    mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
    theta_s = grid.z_on_interface_levels.to_units("K").values[-1]
    mtg_s = theta_s * exn_val[:, :, -1] + g * topo
    mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg_val[:, :, k] = mtg_val[:, :, k + 1] + dz * exn_val[:, :, k + 1]
    assert np.allclose(mtg[:nx, :ny, :nz], mtg_val)

    # height
    theta = grid.z_on_interface_levels.to_units("K").values
    h_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h_val[:, :, -1] = topo
    for k in range(nz - 1, -1, -1):
        h_val[:, :, k] = h_val[:, :, k + 1] - (rd / (cp * g)) * (
            theta[k] * exn_val[:, :, k] + theta[k + 1] * exn_val[:, :, k + 1]
        ) * (p_val[:, :, k] - p_val[:, :, k + 1]) / (p_val[:, :, k] + p_val[:, :, k + 1])
    assert np.allclose(h[:nx, :ny, :], h_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_montgomery(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(zaxis_name="z"), label="grid")
    assume(grid.nz > 1)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    s = data.draw(
        st_raw_field(
            storage_shape,
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            halo=halo,
        ),
        label="s",
    )
    pt = data.draw(st_floats(min_value=1, max_value=1e5), label="pt")

    # ========================================
    # test bed
    # ========================================
    mtg = zeros(storage_shape, backend, dtype, halo=halo)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_opts=backend_opts,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )
    did.get_montgomery_potential(s, pt, mtg)

    cp = did._pcs["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = did._pcs["air_pressure_at_sea_level"]
    rd = did._pcs["gas_constant_of_dry_air"]
    g = did._pcs["gravitational_acceleration"]

    dz = grid.dz.to_units("K").values.item()
    topo = grid.topography.profile.to_units("m").values

    # pressure
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s[:nx, :ny, k - 1]

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)

    # montgomery
    mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
    theta_s = grid.z_on_interface_levels.to_units("K").values[-1]
    mtg_s = theta_s * exn_val[:, :, -1] + g * topo
    mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg_val[:, :, k] = mtg_val[:, :, k + 1] + dz * exn_val[:, :, k + 1]
    assert np.allclose(mtg[:nx, :ny, :nz], mtg_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_height(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(zaxis_name="z"), label="grid")
    assume(grid.nz > 1)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    s = data.draw(
        st_raw_field(
            storage_shape,
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            halo=halo,
        ),
        label="s",
    )
    pt = data.draw(st_floats(min_value=1, max_value=1e5), label="pt")

    # ========================================
    # test bed
    # ========================================
    h = zeros(storage_shape, backend, dtype, halo=halo)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_opts=backend_opts,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )
    did.get_height(s, pt, h)

    cp = did._pcs["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = did._pcs["air_pressure_at_sea_level"]
    rd = did._pcs["gas_constant_of_dry_air"]
    g = did._pcs["gravitational_acceleration"]

    dz = grid.dz.to_units("K").values.item()
    topo = grid.topography.profile.to_units("m").values

    # pressure
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s[:nx, :ny, k - 1]

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)

    # height
    theta = grid.z_on_interface_levels.to_units("K").values
    h_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h_val[:, :, -1] = topo
    for k in range(nz - 1, -1, -1):
        h_val[:, :, k] = h_val[:, :, k + 1] - (rd / (cp * g)) * (
            theta[k] * exn_val[:, :, k] + theta[k + 1] * exn_val[:, :, k + 1]
        ) * (p_val[:, :, k] - p_val[:, :, k + 1]) / (p_val[:, :, k] + p_val[:, :, k + 1])
    assert np.allclose(h[:nx, :ny, :], h_val)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_density_and_temperature(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(zaxis_name="z"), label="grid")
    assume(grid.nz > 1)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    s = data.draw(
        st_raw_field(
            storage_shape,
            min_value=1,
            max_value=1e4,
            backend=backend,
            dtype=dtype,
            halo=halo,
        ),
        label="s",
    )
    pt = data.draw(st_floats(min_value=1, max_value=1e5), label="pt")

    # ========================================
    # test bed
    # ========================================
    p = zeros(storage_shape, backend, dtype, halo=halo)
    exn = zeros(storage_shape, backend, dtype, halo=halo)
    mtg = zeros(storage_shape, backend, dtype, halo=halo)
    h = zeros(storage_shape, backend, dtype, halo=halo)
    rho = zeros(storage_shape, backend, dtype, halo=halo)
    t = zeros(storage_shape, backend, dtype, halo=halo)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_opts=backend_opts,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )
    did.get_diagnostic_variables(s, pt, p, exn, mtg, h)
    did.get_density_and_temperature(s, exn, h, rho, t)

    cp = did._pcs["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = did._pcs["air_pressure_at_sea_level"]
    rd = did._pcs["gas_constant_of_dry_air"]
    g = did._pcs["gravitational_acceleration"]

    dz = grid.dz.to_units("K").values.item()
    topo = grid.topography.profile.to_units("m").values

    # pressure
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s[:nx, :ny, k - 1]
    assert np.allclose(p[:nx, :ny, :], p_val)

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)
    assert np.allclose(exn[:nx, :ny, : nz + 1], exn_val)

    # montgomery
    mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
    theta_s = grid.z_on_interface_levels.to_units("K").values[-1]
    mtg_s = theta_s * exn_val[:, :, -1] + g * topo
    mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg_val[:, :, k] = mtg_val[:, :, k + 1] + dz * exn_val[:, :, k + 1]
    assert np.allclose(mtg[:nx, :ny, :nz], mtg_val)

    # height
    theta = grid.z_on_interface_levels.to_units("K").values
    h_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h_val[:, :, -1] = topo
    for k in range(nz - 1, -1, -1):
        h_val[:, :, k] = h_val[:, :, k + 1] - (rd / (cp * g)) * (
            theta[k] * exn_val[:, :, k] + theta[k + 1] * exn_val[:, :, k + 1]
        ) * (p_val[:, :, k] - p_val[:, :, k + 1]) / (p_val[:, :, k] + p_val[:, :, k + 1])
    assert np.allclose(h[:nx, :ny, :], h_val)

    # density
    theta = theta[np.newaxis, np.newaxis, :]
    rho_val = (
        s[:nx, :ny, :nz]
        * (theta[:, :, :-1] - theta[:, :, 1:])
        / (h_val[:, :, :-1] - h_val[:, :, 1:])
    )
    assert np.allclose(rho[:nx, :ny, :nz], rho_val)

    # temperature
    t_val = (
        0.5
        / cp
        * (theta[:, :, :-1] * exn_val[:, :, :-1] + theta[:, :, 1:] * exn_val[:, :, 1:])
    )
    assert np.allclose(t[:nx, :ny, :nz], t_val)


unit_registry = UnitRegistry()


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_isentropic_diagnostics(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, halo=halo, storage_shape=storage_shape
        ),
        label="state",
    )

    # ========================================
    # test bed
    # ========================================
    #
    # validation data
    #
    p = zeros(storage_shape, backend, dtype, halo=halo)
    exn = zeros(storage_shape, backend, dtype, halo=halo)
    mtg = zeros(storage_shape, backend, dtype, halo=halo)
    h = zeros(storage_shape, backend, dtype, halo=halo)
    rho = zeros(storage_shape, backend, dtype, halo=halo)
    t = zeros(storage_shape, backend, dtype, halo=halo)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_opts=backend_opts,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    pt = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]

    did.get_diagnostic_variables(s, pt, p, exn, mtg, h)
    did.get_density_and_temperature(s, exn, h, rho, t)

    #
    # dry
    #
    pid = IsentropicDiagnostics(
        domain,
        "numerical",
        moist=False,
        pt=state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        backend_opts=backend_opts,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )

    diags = pid(state)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    validation_dict = {
        "air_pressure_on_interface_levels": {
            "storage": p,
            "shape": (nx, ny, nz + 1),
            "units": "Pa",
        },
        "exner_function_on_interface_levels": {
            "storage": exn,
            "shape": (nx, ny, nz + 1),
            "units": "J kg^-1 K^-1",
        },
        "montgomery_potential": {
            "storage": mtg,
            "shape": (nx, ny, nz),
            "units": "m^2 s^-2",
        },
        "height_on_interface_levels": {
            "storage": h,
            "shape": (nx, ny, nz + 1),
            "units": "m",
        },
    }

    for name, props in validation_dict.items():
        assert name in diags
        val = get_dataarray_3d(
            props["storage"],
            grid,
            props["units"],
            name=name,
            grid_shape=props["shape"],
            set_coordinates=False,
        )
        compare_dataarrays(diags[name], val, compare_coordinate_values=False)

    assert len(diags) == len(validation_dict)

    #
    # moist
    #
    pid = IsentropicDiagnostics(
        domain,
        "numerical",
        moist=True,
        pt=state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        backend_opts=backend_opts,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )

    diags = pid(state)

    validation_dict.update(
        {
            "air_density": {"storage": rho, "shape": (nx, ny, nz), "units": "kg m^-3"},
            "air_temperature": {"storage": t, "shape": (nx, ny, nz), "units": "K"},
        }
    )

    for name, props in validation_dict.items():
        assert name in diags
        val = get_dataarray_3d(
            props["storage"],
            grid,
            props["units"],
            name=name,
            grid_shape=props["shape"],
            set_coordinates=False,
        )
        compare_dataarrays(diags[name], val)

    assert len(diags) == len(validation_dict)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_horizontal_velocity(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    grid = domain.numerical_grid
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, halo=halo, storage_shape=storage_shape
        ),
        label="state",
    )

    hb.reference_state = state

    # ========================================
    # test bed
    # ========================================
    ivc = IsentropicVelocityComponents(
        domain,
        backend=backend,
        backend_opts=backend_opts,
        dtype=dtype,
        halo=halo,
        rebuild=False,
        storage_shape=storage_shape,
    )

    diags = ivc(state)

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
    su = (
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[:nx, :ny, :nz]
    )
    sv = (
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[:nx, :ny, :nz]
    )

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    u = np.zeros((nx + 1, ny, nz), dtype=dtype)
    v = np.zeros((nx, ny + 1, nz), dtype=dtype)

    assert "x_velocity_at_u_locations" in diags
    u[1:-1, :] = (su[:-1, :] + su[1:, :]) / (s[:-1, :] + s[1:, :])
    hb.dmn_set_outermost_layers_x(
        u,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"],
    )
    u_val = get_dataarray_3d(u, grid, "m s^-1", name="x_velocity_at_u_locations")
    compare_dataarrays(
        diags["x_velocity_at_u_locations"][: nx + 1, :ny, :nz],
        u_val,
        compare_coordinate_values=False,
    )

    assert "y_velocity_at_v_locations" in diags
    v[:, 1:-1] = (sv[:, :-1] + sv[:, 1:]) / (s[:, :-1] + s[:, 1:])
    hb.dmn_set_outermost_layers_y(
        v,
        field_name="y_velocity_at_v_locations",
        field_units="m s^-1",
        time=state["time"],
    )
    v_val = get_dataarray_3d(v, grid, "m s^-1", name="y_velocity_at_u_locations")
    compare_dataarrays(diags["y_velocity_at_v_locations"][:nx, : ny + 1, :nz], v_val)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_isentropic_diagnostics()
