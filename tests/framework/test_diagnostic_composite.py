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

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

import gt4py as gt
from tasmania.python.framework.composite import DiagnosticComponentComposite
from tasmania.python.isentropic.physics.diagnostics import IsentropicDiagnostics
from tasmania.python.physics.microphysics import SaturationAdjustmentKessler
from test_microphysics import saturation_adjustment_kessler_validation


def isentropic_diagnostics_validation(grid, state, cp, p_ref, rd, g):
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    theta = grid.z_on_interface_levels.to_units("K").values[np.newaxis, np.newaxis, :]
    dz = grid.dz.to_units("K").values.item()
    topo = grid.topography.profile.to_units("m").values
    dtype = grid.x.dtype

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
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


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def _test_serial(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")

    grid_type = data.draw(utils.st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(utils.st_isentropic_state_f(grid, moist=True), label="state")

    backend = data.draw(utils.st_one_of(conf.backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    dv = IsentropicDiagnostics(
        domain,
        grid_type,
        True,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=gt.mode.NUMPY,
        dtype=dtype,
    )
    sa = SaturationAdjustmentKessler(
        domain,
        grid_type,
        air_pressure_on_interface_levels=True,
        backend=backend,
        dtype=dtype,
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
    state_dc = deepcopy(state)

    diagnostics = dcc(state)

    for key in state:
        if key == "time":
            assert state["time"] == state_dc["time"]
        else:
            assert np.allclose(state[key], state_dc[key])

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

    cp = dv._core._physical_constants["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = dv._core._physical_constants["air_pressure_at_sea_level"]
    rd = dv._core._physical_constants["gas_constant_of_dry_air"]
    g = dv._core._physical_constants["gravitational_acceleration"]

    p_val, exn_val, mtg_val, h_val, r_val, temp_val = isentropic_diagnostics_validation(
        grid, state, cp, p_ref, rd, g
    )

    assert np.allclose(
        diagnostics["air_pressure_on_interface_levels"], p_val, equal_nan=True
    )
    assert np.allclose(
        diagnostics["exner_function_on_interface_levels"], exn_val, equal_nan=True
    )
    assert np.allclose(diagnostics["montgomery_potential"], mtg_val, equal_nan=True)
    assert np.allclose(diagnostics["height_on_interface_levels"], h_val, equal_nan=True)
    assert np.allclose(diagnostics["air_density"], r_val, equal_nan=True)
    assert np.allclose(diagnostics["air_temperature"], temp_val, equal_nan=True)

    qv_in = state["mass_fraction_of_water_vapor_in_air"].to_units("g g^-1").values
    qc_in = state["mass_fraction_of_cloud_liquid_water_in_air"].to_units("g g^-1").values
    beta = sa._beta
    lhvw = sa._physical_constants["latent_heat_of_vaporization_of_water"]

    qv_out, qc_out = saturation_adjustment_kessler_validation(
        p_val, temp_val, qv_in, qc_in, beta, lhvw, cp
    )

    assert np.allclose(
        diagnostics["mass_fraction_of_water_vapor_in_air"], qv_out, equal_nan=True
    )
    assert np.allclose(
        diagnostics["mass_fraction_of_cloud_liquid_water_in_air"], qc_out, equal_nan=True
    )


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def _test_asparallel(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")

    grid_type = data.draw(utils.st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(utils.st_isentropic_state_f(grid, moist=True), label="state")

    backend = data.draw(utils.st_one_of(conf.backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    dv = IsentropicDiagnostics(
        domain,
        grid_type,
        True,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=gt.mode.NUMPY,
        dtype=dtype,
    )
    sa = SaturationAdjustmentKessler(
        domain,
        grid_type,
        air_pressure_on_interface_levels=True,
        backend=backend,
        dtype=dtype,
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
    state_dc = deepcopy(state)

    diagnostics = dcc(state)

    for key in state:
        if key == "time":
            assert state["time"] == state_dc["time"]
        else:
            assert np.allclose(state[key], state_dc[key])

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

    cp = dv._core._physical_constants["specific_heat_of_dry_air_at_constant_pressure"]
    p_ref = dv._core._physical_constants["air_pressure_at_sea_level"]
    rd = dv._core._physical_constants["gas_constant_of_dry_air"]
    g = dv._core._physical_constants["gravitational_acceleration"]

    p_val, exn_val, mtg_val, h_val, r_val, temp_val = isentropic_diagnostics_validation(
        grid, state, cp, p_ref, rd, g
    )

    assert np.allclose(
        diagnostics["air_pressure_on_interface_levels"], p_val, equal_nan=True
    )
    assert np.allclose(
        diagnostics["exner_function_on_interface_levels"], exn_val, equal_nan=True
    )
    assert np.allclose(diagnostics["montgomery_potential"], mtg_val, equal_nan=True)
    assert np.allclose(diagnostics["height_on_interface_levels"], h_val, equal_nan=True)
    assert np.allclose(diagnostics["air_density"], r_val, equal_nan=True)
    assert np.allclose(diagnostics["air_temperature"], temp_val, equal_nan=True)

    p = state["air_pressure_on_interface_levels"].to_units("Pa").values
    temp = state["air_temperature"].to_units("K").values
    qv_in = state["mass_fraction_of_water_vapor_in_air"].to_units("g g^-1").values
    qc_in = state["mass_fraction_of_cloud_liquid_water_in_air"].to_units("g g^-1").values
    beta = sa._beta
    lhvw = sa._physical_constants["latent_heat_of_vaporization_of_water"]

    qv_out, qc_out = saturation_adjustment_kessler_validation(
        p, temp, qv_in, qc_in, beta, lhvw, cp
    )

    assert np.allclose(
        diagnostics["mass_fraction_of_water_vapor_in_air"], qv_out, equal_nan=True
    )
    assert np.allclose(
        diagnostics["mass_fraction_of_cloud_liquid_water_in_air"], qc_out, equal_nan=True
    )


if __name__ == "__main__":
    pytest.main([__file__])
