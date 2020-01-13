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
from hypothesis import (
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import numpy as np
import pytest

import gt4py as gt
from tasmania.python.framework.composite import DiagnosticComponentComposite
from tasmania.python.isentropic.physics.diagnostics import IsentropicDiagnostics
from tasmania.python.physics.microphysics.old_kessler import KesslerSaturationAdjustment
from tasmania.python.utils.meteo_utils import tetens_formula
from tasmania.python.utils.storage_utils import deepcopy_dataarray_dict

from tests.conf import (
    backend as conf_backend,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.utilities import (
    compare_arrays,
    compare_dataarrays,
    compare_datetimes,
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_timedeltas,
)


def isentropic_diagnostics_validation(grid, state, cp, p_ref, rd, g):
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    theta = grid.z_on_interface_levels.to_units("K").values[np.newaxis, np.newaxis, :]
    dz = grid.dz.to_units("K").values.item()
    topo = grid.topography.profile.to_units("m").values
    dtype = grid.x.dtype

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
    pt = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]

    # pressure
    p_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p_val[:, :, 0] = pt
    for k in range(1, nz + 1):
        p_val[:, :, k] = p_val[:, :, k - 1] + g * dz * s[:, :, k - 1]

    # exner
    exn_val = cp * (p_val / p_ref) ** (rd / cp)

    # montgomery
    mtg_val = np.zeros((nx, ny, nz), dtype=dtype)
    theta_s = grid.z_on_interface_levels.to_units("K").values[-1]
    mtg_s = theta_s * exn_val[:, :, -1] + g * topo
    mtg_val[:, :, -1] = mtg_s + 0.5 * dz * exn_val[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg_val[:, :, k] = mtg_val[:, :, k + 1] + dz * exn_val[:, :, k + 1]

    # height
    h_val = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h_val[:, :, -1] = topo
    for k in range(nz - 1, -1, -1):
        h_val[:, :, k] = h_val[:, :, k + 1] - (rd / (cp * g)) * (
            theta[:, :, k] * exn_val[:, :, k] + theta[:, :, k + 1] * exn_val[:, :, k + 1]
        ) * (p_val[:, :, k] - p_val[:, :, k + 1]) / (p_val[:, :, k] + p_val[:, :, k + 1])

    # density
    r_val = (
        s * (theta[:, :, :-1] - theta[:, :, 1:]) / (h_val[:, :, :-1] - h_val[:, :, 1:])
    )

    # temperature
    temp_val = (
        0.5
        * (theta[:, :, :-1] * exn_val[:, :, :-1] + theta[:, :, 1:] * exn_val[:, :, 1:])
        / cp
    )

    return p_val, exn_val, mtg_val, h_val, r_val, temp_val


def old_kessler_saturation_adjustment_validation(p, t, qv, qc, beta, lhvw, cp):
    p = p if p.shape[2] == t.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])

    ps = tetens_formula(t)
    qvs = beta * ps / (p - ps)
    sat = (qvs - qv) / (1.0 + qvs * 4093.0 * lhvw / (cp * ((t - 36.0) ** 2.0)))
    dq = deepcopy(qc)
    dq[sat <= qc] = sat[sat <= qc]

    return qv + dq, qc - dq


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_serial(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(2, 30)),
        label="domain",
    )

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(hours=1)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    dv = IsentropicDiagnostics(
        domain,
        grid_type,
        True,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )
    sa = KesslerSaturationAdjustment(
        domain,
        grid_type,
        air_pressure_on_interface_levels=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
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
    assert "mass_fraction_of_cloud_liquid_water_in_air" in dcc.diagnostic_properties
    assert "montgomery_potential" in dcc.diagnostic_properties
    assert len(dcc.diagnostic_properties) == 8

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    diagnostics = dcc(state, timestep)

    for key in state:
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_dataarrays(state[key], state_dc[key], compare_coordinate_values=False)

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

    cp = dv._core._pcs["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = dv._core._pcs["air_pressure_at_sea_level"]
    rd = dv._core._pcs["gas_constant_of_dry_air"]
    g = dv._core._pcs["gravitational_acceleration"]

    p_val, exn_val, mtg_val, h_val, r_val, temp_val = isentropic_diagnostics_validation(
        grid, state, cp, p_ref, rd, g
    )

    compare_arrays(
        diagnostics["air_pressure_on_interface_levels"]
        .to_units("Pa")
        .values[:nx, :ny, : nz + 1],
        p_val,
    )
    compare_arrays(
        diagnostics["exner_function_on_interface_levels"]
        .to_units("J kg^-1 K^-1")
        .values[:nx, :ny, : nz + 1],
        exn_val,
    )
    compare_arrays(
        diagnostics["montgomery_potential"].to_units("m^2 s^-2").values[:nx, :ny, :nz],
        mtg_val,
    )
    compare_arrays(
        diagnostics["height_on_interface_levels"]
        .to_units("m")
        .values[:nx, :ny, : nz + 1],
        h_val,
    )
    compare_arrays(
        diagnostics["air_density"].to_units("kg m^-3").values[:nx, :ny, :nz], r_val
    )
    compare_arrays(
        diagnostics["air_temperature"].to_units("K").values[:nx, :ny, :nz], temp_val
    )

    qv_in = (
        state["mass_fraction_of_water_vapor_in_air"]
        .to_units("g g^-1")
        .values[:nx, :ny, :nz]
    )
    qc_in = (
        state["mass_fraction_of_cloud_liquid_water_in_air"]
        .to_units("g g^-1")
        .values[:nx, :ny, :nz]
    )
    rd = sa._pcs["gas_constant_of_dry_air"]
    rv = sa._pcs["gas_constant_of_water_vapor"]
    beta = rd / rv
    lhvw = sa._pcs["latent_heat_of_vaporization_of_water"]

    qv_out, qc_out = old_kessler_saturation_adjustment_validation(
        p_val, temp_val, qv_in, qc_in, beta, lhvw, cp
    )

    compare_arrays(
        diagnostics["mass_fraction_of_water_vapor_in_air"]
        .to_units("g g^-1")
        .values[:nx, :ny, :nz],
        qv_out,
    )
    compare_arrays(
        diagnostics["mass_fraction_of_cloud_liquid_water_in_air"]
        .to_units("g g^-1")
        .values[:nx, :ny, :nz],
        qc_out,
    )


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_asparallel(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(
        st_domain(xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(2, 30)),
        label="domain",
    )

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(hours=1)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    dv = IsentropicDiagnostics(
        domain,
        grid_type,
        True,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )
    sa = KesslerSaturationAdjustment(
        domain,
        grid_type,
        air_pressure_on_interface_levels=True,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )

    dcc = DiagnosticComponentComposite(dv, sa, execution_policy="as_parallel")

    #
    # test properties
    #
    assert "air_isentropic_density" in dcc.input_properties
    assert "air_pressure_on_interface_levels" in dcc.input_properties
    assert "air_temperature" in dcc.input_properties
    assert "mass_fraction_of_water_vapor_in_air" in dcc.input_properties
    assert "mass_fraction_of_cloud_liquid_water_in_air" in dcc.input_properties
    assert len(dcc.input_properties) == 5

    assert "air_density" in dcc.diagnostic_properties
    assert "air_pressure_on_interface_levels" in dcc.diagnostic_properties
    assert "air_temperature" in dcc.diagnostic_properties
    assert "exner_function_on_interface_levels" in dcc.diagnostic_properties
    assert "height_on_interface_levels" in dcc.diagnostic_properties
    assert "mass_fraction_of_water_vapor_in_air" in dcc.diagnostic_properties
    assert "mass_fraction_of_cloud_liquid_water_in_air" in dcc.diagnostic_properties
    assert "montgomery_potential" in dcc.diagnostic_properties
    assert len(dcc.diagnostic_properties) == 8

    #
    # test numerics
    #
    state_dc = deepcopy_dataarray_dict(state)

    diagnostics = dcc(state, timestep)

    for key in state:
        if key == "time":
            compare_datetimes(state["time"], state_dc["time"])
        else:
            compare_dataarrays(state[key], state_dc[key], compare_coordinate_values=False)

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

    cp = dv._core._pcs["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = dv._core._pcs["air_pressure_at_sea_level"]
    rd = dv._core._pcs["gas_constant_of_dry_air"]
    g = dv._core._pcs["gravitational_acceleration"]

    p_val, exn_val, mtg_val, h_val, r_val, temp_val = isentropic_diagnostics_validation(
        grid, state, cp, p_ref, rd, g
    )

    compare_arrays(
        diagnostics["air_pressure_on_interface_levels"]
        .to_units("Pa")
        .values[:nx, :ny, : nz + 1],
        p_val,
    )
    compare_arrays(
        diagnostics["exner_function_on_interface_levels"]
        .to_units("J kg^-1 K^-1")
        .values[:nx, :ny, : nz + 1],
        exn_val,
    )
    compare_arrays(
        diagnostics["montgomery_potential"].to_units("m^2 s^-2").values[:nx, :ny, :nz],
        mtg_val,
    )
    compare_arrays(
        diagnostics["height_on_interface_levels"]
        .to_units("m")
        .values[:nx, :ny, : nz + 1],
        h_val,
    )
    compare_arrays(
        diagnostics["air_density"].to_units("kg m^-3").values[:nx, :ny, :nz], r_val
    )
    compare_arrays(
        diagnostics["air_temperature"].to_units("K").values[:nx, :ny, :nz], temp_val
    )

    p = (
        state["air_pressure_on_interface_levels"]
        .to_units("Pa")
        .values[:nx, :ny, : nz + 1]
    )
    temp = state["air_temperature"].to_units("K").values[:nx, :ny, :nz]
    qv_in = (
        state["mass_fraction_of_water_vapor_in_air"]
        .to_units("g g^-1")
        .values[:nx, :ny, :nz]
    )
    qc_in = (
        state["mass_fraction_of_cloud_liquid_water_in_air"]
        .to_units("g g^-1")
        .values[:nx, :ny, :nz]
    )
    rd = sa._pcs["gas_constant_of_dry_air"]
    rv = sa._pcs["gas_constant_of_water_vapor"]
    beta = rd / rv
    lhvw = sa._pcs["latent_heat_of_vaporization_of_water"]

    qv_out, qc_out = old_kessler_saturation_adjustment_validation(
        p, temp, qv_in, qc_in, beta, lhvw, cp
    )

    compare_arrays(
        diagnostics["mass_fraction_of_water_vapor_in_air"]
        .to_units("g g^-1")
        .values[:nx, :ny, :nz],
        qv_out,
    )
    compare_arrays(
        diagnostics["mass_fraction_of_cloud_liquid_water_in_air"]
        .to_units("g g^-1")
        .values[:nx, :ny, :nz],
        qc_out,
    )


if __name__ == "__main__":
    pytest.main([__file__])
