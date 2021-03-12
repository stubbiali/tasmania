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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
from pint import UnitRegistry
import pytest
import sympl

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.dynamics.diagnostics import (
    IsentropicDiagnostics as DynamicsIsentropicDiagnostics,
)
from tasmania.python.isentropic.physics.diagnostics import (
    IsentropicDiagnostics,
    IsentropicVelocityComponents,
)
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import (
    st_floats,
    st_one_of,
    st_domain,
    st_physical_grid,
    st_isentropic_state_f,
    st_raw_field,
)
from tests.utilities import compare_arrays, compare_dataarrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_diagnostic_variables(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    grid = data.draw(
        st_physical_grid(
            zaxis_name="z", zaxis_length=(2, 20), storage_options=so
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    s = data.draw(
        st_raw_field(
            storage_shape,
            min_value=1,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="s",
    )
    pt = data.draw(st_floats(min_value=1, max_value=1e5), label="pt")

    # ========================================
    # test bed
    # ========================================
    p = zeros(backend, shape=storage_shape, storage_options=so)
    exn = zeros(backend, shape=storage_shape, storage_options=so)
    mtg = zeros(backend, shape=storage_shape, storage_options=so)
    h = zeros(backend, shape=storage_shape, storage_options=so)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    did.get_diagnostic_variables(s, pt, p, exn, mtg, h)

    cp = did.rpc["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = did.rpc["air_pressure_at_sea_level"]
    rd = did.rpc["gas_constant_of_dry_air"]
    g = did.rpc["gravitational_acceleration"]

    dz = grid.dz.to_units("K").data.item()
    topo = grid.topography.profile.to_units("m").data

    # pressure
    s_np = to_numpy(s)
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s_np[:nx, :ny, k - 1]
    compare_arrays(p[:nx, :ny, :], p_val)

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)
    compare_arrays(exn[:nx, :ny, : nz + 1], exn_val)

    # montgomery
    mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
    theta_s = grid.z_on_interface_levels.to_units("K").data[-1]
    mtg_s = theta_s * exn_val[:, :, -1] + g * topo
    mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg_val[:, :, k] = mtg_val[:, :, k + 1] + dz * exn_val[:, :, k + 1]
    compare_arrays(mtg[:nx, :ny, :nz], mtg_val)

    # height
    theta = grid.z_on_interface_levels.to_units("K").data
    h_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h_val[:, :, -1] = topo
    for k in range(nz - 1, -1, -1):
        h_val[:, :, k] = h_val[:, :, k + 1] - (rd / (cp * g)) * (
            theta[k] * exn_val[:, :, k] + theta[k + 1] * exn_val[:, :, k + 1]
        ) * (p_val[:, :, k] - p_val[:, :, k + 1]) / (
            p_val[:, :, k] + p_val[:, :, k + 1]
        )
    compare_arrays(h[:nx, :ny, :], h_val)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_montgomery(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    grid = data.draw(
        st_physical_grid(
            zaxis_name="z", zaxis_length=(2, 20), storage_options=so
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    s = data.draw(
        st_raw_field(
            storage_shape,
            min_value=1,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="s",
    )
    pt = data.draw(st_floats(min_value=1, max_value=1e5), label="pt")

    # ========================================
    # test bed
    # ========================================
    mtg = zeros(backend, shape=storage_shape, storage_options=so)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    did.get_montgomery_potential(s, pt, mtg)

    cp = did.rpc["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = did.rpc["air_pressure_at_sea_level"]
    rd = did.rpc["gas_constant_of_dry_air"]
    g = did.rpc["gravitational_acceleration"]

    dz = grid.dz.to_units("K").data.item()
    topo = grid.topography.profile.to_units("m").data

    # pressure
    s_np = to_numpy(s)
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s_np[:nx, :ny, k - 1]

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)

    # montgomery
    mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
    theta_s = grid.z_on_interface_levels.to_units("K").data[-1]
    mtg_s = theta_s * exn_val[:, :, -1] + g * topo
    mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg_val[:, :, k] = mtg_val[:, :, k + 1] + dz * exn_val[:, :, k + 1]
    compare_arrays(mtg[:nx, :ny, :nz], mtg_val)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_height(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    grid = data.draw(
        st_physical_grid(
            zaxis_name="z", zaxis_length=(2, 20), storage_options=so
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    s = data.draw(
        st_raw_field(
            storage_shape,
            min_value=1,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="s",
    )
    pt = data.draw(st_floats(min_value=1, max_value=1e5), label="pt")

    # ========================================
    # test bed
    # ========================================
    h = zeros(backend, shape=storage_shape, storage_options=so)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    did.get_height(s, pt, h)

    cp = did.rpc["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = did.rpc["air_pressure_at_sea_level"]
    rd = did.rpc["gas_constant_of_dry_air"]
    g = did.rpc["gravitational_acceleration"]

    dz = grid.dz.to_units("K").data.item()
    topo = grid.topography.profile.to_units("m").data

    # pressure
    s_np = to_numpy(s)
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s_np[:nx, :ny, k - 1]

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)

    # height
    theta = grid.z_on_interface_levels.to_units("K").data
    h_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h_val[:, :, -1] = topo
    for k in range(nz - 1, -1, -1):
        h_val[:, :, k] = h_val[:, :, k + 1] - (rd / (cp * g)) * (
            theta[k] * exn_val[:, :, k] + theta[k + 1] * exn_val[:, :, k + 1]
        ) * (p_val[:, :, k] - p_val[:, :, k + 1]) / (
            p_val[:, :, k] + p_val[:, :, k + 1]
        )
    compare_arrays(h[:nx, :ny, :], h_val)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_density_and_temperature(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    grid = data.draw(
        st_physical_grid(
            zaxis_name="z", zaxis_length=(2, 20), storage_options=so
        ),
        label="grid",
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    storage_shape = (nx + dnx, ny + dny, nz + 1)

    s = data.draw(
        st_raw_field(
            storage_shape,
            min_value=1,
            max_value=1e4,
            backend=backend,
            storage_options=so,
        ),
        label="s",
    )
    pt = data.draw(st_floats(min_value=1, max_value=1e5), label="pt")

    # ========================================
    # test bed
    # ========================================
    p = zeros(backend, shape=storage_shape, storage_options=so)
    exn = zeros(backend, shape=storage_shape, storage_options=so)
    mtg = zeros(backend, shape=storage_shape, storage_options=so)
    h = zeros(backend, shape=storage_shape, storage_options=so)
    rho = zeros(backend, shape=storage_shape, storage_options=so)
    t = zeros(backend, shape=storage_shape, storage_options=so)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    did.get_diagnostic_variables(s, pt, p, exn, mtg, h)
    did.get_density_and_temperature(s, exn, h, rho, t)

    cp = did.rpc["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = did.rpc["air_pressure_at_sea_level"]
    rd = did.rpc["gas_constant_of_dry_air"]
    g = did.rpc["gravitational_acceleration"]

    dz = grid.dz.to_units("K").data.item()
    topo = grid.topography.profile.to_units("m").data

    # pressure
    s_np = to_numpy(s)
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s_np[:nx, :ny, k - 1]
    compare_arrays(p[:nx, :ny, :], p_val)

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)
    compare_arrays(exn[:nx, :ny, : nz + 1], exn_val)

    # montgomery
    mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
    theta_s = grid.z_on_interface_levels.to_units("K").data[-1]
    mtg_s = theta_s * exn_val[:, :, -1] + g * topo
    mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg_val[:, :, k] = mtg_val[:, :, k + 1] + dz * exn_val[:, :, k + 1]
    compare_arrays(mtg[:nx, :ny, :nz], mtg_val)

    # height
    theta = grid.z_on_interface_levels.to_units("K").data
    h_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h_val[:, :, -1] = topo
    for k in range(nz - 1, -1, -1):
        h_val[:, :, k] = h_val[:, :, k + 1] - (rd / (cp * g)) * (
            theta[k] * exn_val[:, :, k] + theta[k + 1] * exn_val[:, :, k + 1]
        ) * (p_val[:, :, k] - p_val[:, :, k + 1]) / (
            p_val[:, :, k] + p_val[:, :, k + 1]
        )
    compare_arrays(h[:nx, :ny, :], h_val)

    # density
    theta = theta[np.newaxis, np.newaxis, :]
    rho_val = (
        s_np[:nx, :ny, :nz]
        * (theta[:, :, :-1] - theta[:, :, 1:])
        / (h_val[:, :, :-1] - h_val[:, :, 1:])
    )
    compare_arrays(rho[:nx, :ny, :nz], rho_val)

    # temperature
    t_val = (
        0.5
        / cp
        * (
            theta[:, :, :-1] * exn_val[:, :, :-1]
            + theta[:, :, 1:] * exn_val[:, :, 1:]
        )
    )
    compare_arrays(t[:nx, :ny, :nz], t_val)


unit_registry = UnitRegistry()


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_isentropic_diagnostics(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(backend=backend, backend_options=bo, storage_options=so),
        label="domain",
    )
    grid = domain.numerical_grid

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state",
    )

    # ========================================
    # test bed
    # ========================================
    p = zeros(backend, shape=storage_shape, storage_options=so)
    exn = zeros(backend, shape=storage_shape, storage_options=so)
    mtg = zeros(backend, shape=storage_shape, storage_options=so)
    h = zeros(backend, shape=storage_shape, storage_options=so)
    rho = zeros(backend, shape=storage_shape, storage_options=so)
    t = zeros(backend, shape=storage_shape, storage_options=so)

    did = DynamicsIsentropicDiagnostics(
        grid,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    pt = sympl.DataArray(
        state["air_pressure_on_interface_levels"].to_units("Pa").data[0, 0, 0],
        attrs={"units": "Pa"},
    )

    did.get_diagnostic_variables(s, pt.data.item(), p, exn, mtg, h)
    did.get_density_and_temperature(s, exn, h, rho, t)

    #
    # dry
    #
    pid = IsentropicDiagnostics(
        domain,
        "numerical",
        moist=False,
        pt=pt,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
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
        pt=pt,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    diags = pid(state)

    validation_dict.update(
        {
            "air_density": {
                "storage": rho,
                "shape": (nx, ny, nz),
                "units": "kg m^-3",
            },
            "air_temperature": {
                "storage": t,
                "shape": (nx, ny, nz),
                "units": "K",
            },
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


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_horizontal_velocity(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(backend=backend, backend_options=bo, storage_options=so),
        label="domain",
    )
    hb = domain.horizontal_boundary
    grid = domain.numerical_grid

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
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
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    diags = ivc(state)

    s = to_numpy(state["air_isentropic_density"].to_units("kg m^-2 K^-1").data)
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    sv = to_numpy(
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )

    u = zeros("numpy", shape=storage_shape, storage_options=so)
    v = zeros("numpy", shape=storage_shape, storage_options=so)

    assert "x_velocity_at_u_locations" in diags
    u[1:-1, :] = (su[:-2, :] + su[1:-1, :]) / (s[:-2, :] + s[1:-1, :])
    hb_np = domain.copy(backend="numpy").horizontal_boundary
    hb_np.set_outermost_layers_x(
        u,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"],
    )
    u_val = get_dataarray_3d(
        u,
        grid,
        "m s^-1",
        name="x_velocity_at_u_locations",
        grid_shape=(nx + 1, ny, nz),
        set_coordinates=False,
    )
    compare_dataarrays(
        diags["x_velocity_at_u_locations"],
        u_val,
        compare_coordinate_values=False,
        slice=(slice(nx + 1), slice(ny), slice(nz)),
    )

    assert "y_velocity_at_v_locations" in diags
    v[:, 1:-1] = (sv[:, :-2] + sv[:, 1:-1]) / (s[:, :-2] + s[:, 1:-1])
    hb_np.set_outermost_layers_y(
        v,
        field_name="y_velocity_at_v_locations",
        field_units="m s^-1",
        time=state["time"],
    )
    v_val = get_dataarray_3d(
        v,
        grid,
        "m s^-1",
        name="y_velocity_at_u_locations",
        grid_shape=(nx, ny + 1, nz),
        set_coordinates=False,
    )
    compare_dataarrays(
        diags["y_velocity_at_v_locations"],
        v_val,
        slice=(slice(nx), slice(ny + 1), slice(nz)),
    )


if __name__ == "__main__":
    pytest.main([__file__])
