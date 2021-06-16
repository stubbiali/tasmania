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
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
from property_cached import cached_property
import pytest

from sympl._core.data_array import DataArray

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

from tests import conf
from tests.strategies import (
    st_floats,
    st_one_of,
    st_physical_grid,
    st_isentropic_state_f,
    st_raw_field,
)
from tests.suites.core_components import DiagnosticComponentTestSuite
from tests.suites.domain import DomainSuite
from tests.utilities import compare_arrays, hyp_settings


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


class IsentropicDiagnosticsTestSuite(DiagnosticComponentTestSuite):
    def __init__(self, domain_suite, moist, pt):
        self.moist = moist
        self.pt = pt
        super().__init__(domain_suite)

    @cached_property
    def component(self):
        return IsentropicDiagnostics(
            self.ds.domain,
            self.ds.grid_type,
            moist=self.moist,
            pt=self.pt,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        return self.ds.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=self.moist,
                precipitation=False,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            )
        )

    def get_validation_diagnostics(self, raw_state_np):
        s = raw_state_np["air_isentropic_density"]

        p = self.component.zeros(
            backend="numpy",
            shape=self.component.get_field_storage_shape(
                "air_pressure_on_interface_levels"
            ),
        )
        exn = self.component.zeros(
            backend="numpy",
            shape=self.component.get_field_storage_shape(
                "exner_function_on_interface_levels"
            ),
        )
        mtg = zeros(
            backend="numpy",
            shape=self.component.get_field_storage_shape(
                "montgomery_potential"
            ),
        )
        h = zeros(
            backend="numpy",
            shape=self.component.get_field_storage_shape(
                "height_on_interface_levels"
            ),
        )
        rho = zeros(
            backend="numpy",
            shape=self.component.get_field_storage_shape("air_density"),
        )
        t = zeros(
            backend="numpy",
            shape=self.component.get_field_storage_shape("air_temperature"),
        )

        did = DynamicsIsentropicDiagnostics(
            self.ds.grid,
            backend="numpy",
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )
        did.get_diagnostic_variables(s, self.pt.data.item(), p, exn, mtg, h)
        out = {
            "air_pressure_on_interface_levels": p,
            "exner_function_on_interface_levels": exn,
            "montgomery_potential": mtg,
            "height_on_interface_levels": h,
        }

        if self.moist:
            did.get_density_and_temperature(s, exn, h, rho, t)
            out["air_density"] = rho
            out["air_temperature"] = t

        return out


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_isentropic_diagnostics(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    ds = DomainSuite(data, backend, dtype)
    pt = DataArray(
        data.draw(st_floats(min_value=0, max_value=1e4)), attrs={"units": "Pa"}
    )

    # ========================================
    # test bed
    # ========================================
    # dry
    ts = IsentropicDiagnosticsTestSuite(ds, moist=False, pt=pt)
    ts.run()

    # moist
    ts = IsentropicDiagnosticsTestSuite(ds, moist=True, pt=pt)
    ts.run()


class HorizontalVelocityTestSuite(DiagnosticComponentTestSuite):
    @cached_property
    def component(self):
        return IsentropicVelocityComponents(
            self.ds.domain,
            backend=self.ds.backend,
            backend_options=self.ds.bo,
            storage_shape=self.ds.storage_shape,
            storage_options=self.ds.so,
        )

    def get_state(self):
        return self.ds.hyp_data.draw(
            st_isentropic_state_f(
                self.ds.grid,
                moist=False,
                backend=self.ds.backend,
                storage_shape=self.ds.storage_shape,
                storage_options=self.ds.so,
            )
        )

    def get_validation_diagnostics(self, raw_state_np):
        nx, ny, nz = self.ds.grid.nx, self.ds.grid.ny, self.ds.grid.nz

        s = raw_state_np["air_isentropic_density"]
        su = raw_state_np["x_momentum_isentropic"]
        sv = raw_state_np["y_momentum_isentropic"]

        hb_np = self.ds.domain.copy(backend="numpy").horizontal_boundary

        u = self.component.zeros(
            backend="numpy",
            shape=self.component.get_field_storage_shape(
                "x_velocity_at_u_locations", self.ds.storage_shape
            ),
        )
        u[1:nx, :ny, :nz] = (su[: nx - 1, :ny, :nz] + su[1:nx, :ny, :nz]) / (
            s[: nx - 1, :ny, :nz] + s[1:nx, :ny, :nz]
        )
        hb_np.set_outermost_layers_x(
            u,
            field_name="x_velocity_at_u_locations",
            field_units="m s^-1",
            time=raw_state_np["time"],
        )

        v = self.component.zeros(
            backend="numpy",
            shape=self.component.get_field_storage_shape(
                "y_velocity_at_v_locations", self.ds.storage_shape
            ),
        )
        v[:nx, 1:ny, :nz] = (sv[:nx, : ny - 1, :nz] + sv[:nx, 1:ny, :nz]) / (
            s[:nx, : ny - 1, :nz] + s[:nx, 1:ny, :nz]
        )
        hb_np.set_outermost_layers_y(
            v,
            field_name="y_velocity_at_v_locations",
            field_units="m s^-1",
            time=raw_state_np["time"],
        )

        out = {"x_velocity_at_u_locations": u, "y_velocity_at_v_locations": v}

        return out


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_horizontal_velocity(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    ds = DomainSuite(data, backend, dtype, grid_type="numerical")
    assume(ds.domain.horizontal_boundary.type != "identity")

    # ========================================
    # test bed
    # ========================================
    ts = HorizontalVelocityTestSuite(ds)
    state = ts.get_state()
    ds.domain.horizontal_boundary.reference_state = state
    ts.run()


if __name__ == "__main__":
    pytest.main([__file__])
