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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest
import sympl

from tasmania.python.framework.composite import DiagnosticComponentComposite
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.physics.diagnostics import (
    IsentropicDiagnostics,
)
from tasmania.python.physics.microphysics.kessler import (
    KesslerSaturationAdjustmentDiagnostic,
)
from tasmania.python.utils.meteo import tetens_formula
from tasmania.python.utils.storage import deepcopy_dataarray_dict

from tests import conf
from tests.physics.test_microphysics_kessler import (
    kessler_saturation_adjustment_diagnostic_validation,
)
from tests.strategies import (
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_timedeltas,
)
from tests.utilities import (
    compare_arrays,
    compare_dataarrays,
    compare_datetimes,
    hyp_settings,
)


class KesslerSaturationAdjustment(KesslerSaturationAdjustmentDiagnostic):
    @property
    def tendency_properties(self):
        return {}

    def array_call(self, state, timestep):
        _, diagnostics = super().array_call(state, timestep)
        return {}, diagnostics


def isentropic_diagnostics_validation(grid, state, cp, p_ref, rd, g):
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    theta = grid.z_on_interface_levels.to_units("K").data[
        np.newaxis, np.newaxis, :
    ]
    dz = grid.dz.to_units("K").data.item()
    topo = grid.topography.profile.to_units("m").data

    s = to_numpy(
        state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )[:nx, :ny, :nz]
    dtype = s.dtype
    pt = state["air_pressure_on_interface_levels"].to_units("Pa").data[0, 0, 0]

    # pressure
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s[:, :, k - 1]

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)

    # montgomery
    mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
    theta_s = grid.z_on_interface_levels.to_units("K").data[-1]
    mtg_s = theta_s * exn_val[:, :, -1] + g * topo
    mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg_val[:, :, k] = mtg_val[:, :, k + 1] + dz * exn_val[:, :, k + 1]

    # height
    h_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h_val[:, :, -1] = topo
    for k in range(nz - 1, -1, -1):
        h_val[:, :, k] = h_val[:, :, k + 1] - (rd / (cp * g)) * (
            theta[:, :, k] * exn_val[:, :, k]
            + theta[:, :, k + 1] * exn_val[:, :, k + 1]
        ) * (p_val[:, :, k] - p_val[:, :, k + 1]) / (
            p_val[:, :, k] + p_val[:, :, k + 1]
        )

    # density
    r_val = (
        s
        * (theta[:, :, :-1] - theta[:, :, 1:])
        / (h_val[:, :, :-1] - h_val[:, :, 1:])
    )

    # temperature
    temp_val = (
        0.5
        * (
            theta[:, :, :-1] * exn_val[:, :, :-1]
            + theta[:, :, 1:] * exn_val[:, :, 1:]
        )
        / cp
    )

    return p_val, exn_val, mtg_val, h_val, r_val, temp_val


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_serial(data, backend, dtype, subtests):
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
            zaxis_length=(2, 30),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

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

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    pt = sympl.DataArray(
        state["air_pressure_on_interface_levels"].to_units("Pa").data[0, 0, 0],
        attrs={"units": "Pa"},
    )
    dv = IsentropicDiagnostics(
        domain,
        grid_type,
        True,
        pt,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    sa = KesslerSaturationAdjustment(
        domain,
        grid_type,
        air_pressure_on_interface_levels=True,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    dcc = DiagnosticComponentComposite(dv, sa, execution_policy="serial")

    #
    # test properties
    #
    assert "air_isentropic_density" in dcc.input_properties
    assert "mass_fraction_of_water_vapor_in_air" in dcc.input_properties
    assert "mass_fraction_of_cloud_liquid_water_in_air" in dcc.input_properties
    assert len(dcc.input_properties) == 3

    assert "air_density" in dcc.diagnostic_properties
    assert "air_pressure_on_interface_levels" in dcc.diagnostic_properties
    assert "air_temperature" in dcc.diagnostic_properties
    assert "exner_function_on_interface_levels" in dcc.diagnostic_properties
    assert "height_on_interface_levels" in dcc.diagnostic_properties
    assert "mass_fraction_of_water_vapor_in_air" in dcc.diagnostic_properties
    assert (
        "mass_fraction_of_cloud_liquid_water_in_air"
        in dcc.diagnostic_properties
    )
    assert "montgomery_potential" in dcc.diagnostic_properties
    assert len(dcc.diagnostic_properties) == 8

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    diagnostics = dcc(state, timestep)

    for key in state:
        # with subtests.test(key=key):
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_dataarrays(
                state[key], state_dc[key], compare_coordinate_values=False
            )

    assert len(state) == len(state_dc)

    assert "air_density" in diagnostics
    assert "air_pressure_on_interface_levels" in diagnostics
    assert "air_temperature" in diagnostics
    assert "exner_function_on_interface_levels" in diagnostics
    assert "height_on_interface_levels" in diagnostics
    assert "mass_fraction_of_water_vapor_in_air" in diagnostics
    assert "mass_fraction_of_cloud_liquid_water_in_air" in diagnostics
    assert "montgomery_potential" in diagnostics
    assert len(diagnostics) == 8

    cp = dv._core.rpc["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = dv._core.rpc["air_pressure_at_sea_level"]
    rd = dv._core.rpc["gas_constant_of_dry_air"]
    g = dv._core.rpc["gravitational_acceleration"]

    (
        p_val,
        exn_val,
        mtg_val,
        h_val,
        r_val,
        temp_tmp,
    ) = isentropic_diagnostics_validation(grid, state, cp, p_ref, rd, g)

    compare_arrays(
        diagnostics["air_pressure_on_interface_levels"].to_units("Pa").data,
        p_val,
        slice=(slice(nx), slice(ny), slice(nz + 1)),
    )
    compare_arrays(
        diagnostics["exner_function_on_interface_levels"]
        .to_units("J kg^-1 K^-1")
        .data,
        exn_val,
        slice=(slice(nx), slice(ny), slice(nz + 1)),
    )
    compare_arrays(
        diagnostics["montgomery_potential"].to_units("m^2 s^-2").data,
        mtg_val,
        slice=(slice(nx), slice(ny), slice(nz)),
    )
    compare_arrays(
        diagnostics["height_on_interface_levels"].to_units("m").data,
        h_val,
        slice=(slice(nx), slice(ny), slice(nz + 1)),
    )
    compare_arrays(
        diagnostics["air_density"].to_units("kg m^-3").data,
        r_val,
        slice=(slice(nx), slice(ny), slice(nz)),
    )

    qv_np = to_numpy(
        state["mass_fraction_of_water_vapor_in_air"].to_units("g g^-1").data
    )[:nx, :ny, :nz]
    qc_np = to_numpy(
        state["mass_fraction_of_cloud_liquid_water_in_air"]
        .to_units("g g^-1")
        .data
    )[:nx, :ny, :nz]
    rv = sa.rpc["gas_constant_of_water_vapor"]
    beta = rd / rv
    lhvw = sa.rpc["latent_heat_of_vaporization_of_water"]

    (
        qv_val,
        qc_val,
        temp_val,
        _,
    ) = kessler_saturation_adjustment_diagnostic_validation(
        timestep.total_seconds(),
        p_val,
        temp_tmp,
        exn_val,
        qv_np,
        qc_np,
        tetens_formula,
        beta,
        lhvw,
        cp,
        rv,
    )

    compare_arrays(
        diagnostics["mass_fraction_of_water_vapor_in_air"]
        .to_units("g g^-1")
        .data,
        qv_val,
        slice=(slice(nx), slice(ny), slice(nz)),
    )
    compare_arrays(
        diagnostics["mass_fraction_of_cloud_liquid_water_in_air"]
        .to_units("g g^-1")
        .data,
        qc_val,
        slice=(slice(nx), slice(ny), slice(nz)),
    )
    compare_arrays(
        diagnostics["air_temperature"].to_units("K").data,
        temp_val,
        slice=(slice(nx), slice(ny), slice(nz)),
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_asparallel(data, backend, dtype, subtests):
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
            zaxis_length=(2, 30),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

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

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    pt = sympl.DataArray(
        state["air_pressure_on_interface_levels"].to_units("Pa").data[0, 0, 0],
        attrs={"units": "Pa"},
    )
    dv = IsentropicDiagnostics(
        domain,
        grid_type,
        True,
        pt,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    sa = KesslerSaturationAdjustment(
        domain,
        grid_type,
        air_pressure_on_interface_levels=True,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    dcc = DiagnosticComponentComposite(dv, sa, execution_policy="as_parallel")

    #
    # test properties
    #
    assert "air_isentropic_density" in dcc.input_properties
    assert "air_pressure_on_interface_levels" in dcc.input_properties
    assert "air_temperature" in dcc.input_properties
    assert "exner_function_on_interface_levels" in dcc.input_properties
    assert "mass_fraction_of_water_vapor_in_air" in dcc.input_properties
    assert "mass_fraction_of_cloud_liquid_water_in_air" in dcc.input_properties
    assert len(dcc.input_properties) == 6

    assert "air_density" in dcc.diagnostic_properties
    assert "air_pressure_on_interface_levels" in dcc.diagnostic_properties
    assert "air_temperature" in dcc.diagnostic_properties
    assert "exner_function_on_interface_levels" in dcc.diagnostic_properties
    assert "height_on_interface_levels" in dcc.diagnostic_properties
    assert "mass_fraction_of_water_vapor_in_air" in dcc.diagnostic_properties
    assert (
        "mass_fraction_of_cloud_liquid_water_in_air"
        in dcc.diagnostic_properties
    )
    assert "montgomery_potential" in dcc.diagnostic_properties
    assert len(dcc.diagnostic_properties) == 8

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    diagnostics = dcc(state, timestep)

    for key in state:
        # with subtests.test(key=key):
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_dataarrays(
                state[key], state_dc[key], compare_coordinate_values=False
            )

    assert len(state) == len(state_dc)

    assert "air_density" in diagnostics
    assert "air_pressure_on_interface_levels" in diagnostics
    assert "air_temperature" in diagnostics
    assert "exner_function_on_interface_levels" in diagnostics
    assert "height_on_interface_levels" in diagnostics
    assert "mass_fraction_of_water_vapor_in_air" in diagnostics
    assert "mass_fraction_of_cloud_liquid_water_in_air" in diagnostics
    assert "montgomery_potential" in diagnostics
    assert len(diagnostics) == 8

    cp = dv._core.rpc["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = dv._core.rpc["air_pressure_at_sea_level"]
    rd = dv._core.rpc["gas_constant_of_dry_air"]
    g = dv._core.rpc["gravitational_acceleration"]

    (
        p_val,
        exn_val,
        mtg_val,
        h_val,
        r_val,
        temp_val,
    ) = isentropic_diagnostics_validation(grid, state, cp, p_ref, rd, g)

    compare_arrays(
        diagnostics["air_pressure_on_interface_levels"]
        .to_units("Pa")
        .data[:nx, :ny, : nz + 1],
        p_val,
    )
    compare_arrays(
        diagnostics["exner_function_on_interface_levels"]
        .to_units("J kg^-1 K^-1")
        .data[:nx, :ny, : nz + 1],
        exn_val,
    )
    compare_arrays(
        diagnostics["montgomery_potential"]
        .to_units("m^2 s^-2")
        .data[:nx, :ny, :nz],
        mtg_val,
    )
    compare_arrays(
        diagnostics["height_on_interface_levels"]
        .to_units("m")
        .data[:nx, :ny, : nz + 1],
        h_val,
    )
    compare_arrays(
        diagnostics["air_density"].to_units("kg m^-3").data[:nx, :ny, :nz],
        r_val,
    )

    p_np = to_numpy(
        state["air_pressure_on_interface_levels"].to_units("Pa").data
    )[:nx, :ny, : nz + 1]
    t_np = to_numpy(state["air_temperature"].to_units("K").data)[:nx, :ny, :nz]
    exn_np = to_numpy(
        state["exner_function_on_interface_levels"]
        .to_units("J kg^-1 K^-1")
        .data
    )[:nx, :ny, : nz + 1]
    qv_np = to_numpy(
        state["mass_fraction_of_water_vapor_in_air"].to_units("g g^-1").data
    )[:nx, :ny, :nz]
    qc_np = to_numpy(
        state["mass_fraction_of_cloud_liquid_water_in_air"]
        .to_units("g g^-1")
        .data
    )[:nx, :ny, :nz]
    rv = sa.rpc["gas_constant_of_water_vapor"]
    beta = rd / rv
    lhvw = sa.rpc["latent_heat_of_vaporization_of_water"]

    (
        qv_val,
        qc_val,
        t_val,
        _,
    ) = kessler_saturation_adjustment_diagnostic_validation(
        timestep.total_seconds(),
        p_np,
        t_np,
        exn_np,
        qv_np,
        qc_np,
        tetens_formula,
        beta,
        lhvw,
        cp,
        rv,
    )

    compare_arrays(
        diagnostics["mass_fraction_of_water_vapor_in_air"]
        .to_units("g g^-1")
        .data[:nx, :ny, :nz],
        qv_val,
    )
    compare_arrays(
        diagnostics["mass_fraction_of_cloud_liquid_water_in_air"]
        .to_units("g g^-1")
        .data[:nx, :ny, :nz],
        qc_val,
    )
    compare_arrays(
        diagnostics["air_temperature"].to_units("K").data[:nx, :ny, :nz],
        t_val,
    )


if __name__ == "__main__":
    pytest.main([__file__])
