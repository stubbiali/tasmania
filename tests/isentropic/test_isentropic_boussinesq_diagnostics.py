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
import numpy as np
import pytest

from tasmania.python.isentropic.physics.boussinesq_diagnostics import (
    IsentropicBoussinesqDiagnostics1,
    IsentropicBoussinesqDiagnostics2,
    IsentropicBoussinesqDiagnostics3,
    IsentropicBoussinesqDiagnostics4,
    IsentropicBoussinesqDiagnostics5,
    IsentropicBoussinesqDiagnostics6,
    IsentropicBoussinesqDiagnostics7,
)

try:
    from .conf import backend as conf_backend
    from .utils import (
        compare_datetimes,
        compare_arrays,
        compare_dataarrays,
        st_floats,
        st_one_of,
        st_domain,
        st_physical_grid,
        st_isentropic_state,
        st_isentropic_boussinesq_state_f,
    )
except ModuleNotFoundError:
    from conf import backend as conf_backend
    from utils import (
        compare_datetimes,
        compare_arrays,
        compare_dataarrays,
        st_floats,
        st_one_of,
        st_domain,
        st_physical_grid,
        st_isentropic_state,
        st_isentropic_boussinesq_state_f,
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_isentropic_boussinesq_diagnostics_1(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_f(grid, moist=moist), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    ibd = IsentropicBoussinesqDiagnostics1(
        domain,
        grid_type,
        moist,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in ibd.input_properties
    assert "dd_montgomery_potential" in ibd.input_properties
    assert len(ibd.input_properties) == 2

    assert "air_pressure_on_interface_levels" in ibd.diagnostic_properties
    assert "exner_function_on_interface_levels" in ibd.diagnostic_properties
    assert "height_on_interface_levels" in ibd.diagnostic_properties
    assert "montgomery_potential" in ibd.diagnostic_properties
    if moist:
        assert "air_density" in ibd.diagnostic_properties
        assert "air_temperature" in ibd.diagnostic_properties
        assert len(ibd.diagnostic_properties) == 6
    else:
        assert len(ibd.diagnostic_properties) == 4

    diags = ibd(state)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    thetabar = grid.z_on_interface_levels.to_units("K").values[-1]
    dtheta = grid.dz.to_units("K").values.item()

    pref = ibd._pref.value
    rd = ibd._rd.value
    g = ibd._g.value
    cp = ibd._cp.value

    # pressure
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    p = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p[:, :, 0] = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]
    for k in range(1, nz + 1):
        p[:, :, k] = p[:, :, k - 1] + dtheta * g * s[:, :, k - 1]
    assert "air_pressure_on_interface_levels" in diags
    compare_arrays(p, diags["air_pressure_on_interface_levels"])

    # exner function
    exn = cp * ((p / pref) ** (rd / cp))
    assert "exner_function_on_interface_levels" in diags
    compare_arrays(exn, diags["exner_function_on_interface_levels"])

    # height
    ddmtg = state["dd_montgomery_potential"].to_units("m^2 K^-2 s^-2").values
    h = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h[:, :, -1] = grid.topography.profile.to_units("m").values
    for k in range(nz - 1, -1, -1):
        h[:, :, k] = h[:, :, k + 1] - dtheta * (thetabar / g) * ddmtg[:, :, k]
    assert "height_on_interface_levels" in diags
    compare_arrays(h, diags["height_on_interface_levels"])

    # montgomery potential
    mtg = np.zeros((nx, ny, nz), dtype=dtype)
    mtg_s = g * h[:, :, -1] + thetabar * exn[:, :, -1]
    mtg[:, :, -1] = mtg_s - 0.5 * dtheta * (g / thetabar) * h[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg[:, :, k] = mtg[:, :, k + 1] - dtheta * (g / thetabar) * h[:, :, k + 1]
    assert "montgomery_potential" in diags
    compare_arrays(mtg, diags["montgomery_potential"])

    if moist:
        # density
        rho = s * dtheta / (h[:, :, :-1] - h[:, :, 1:])
        assert "air_density" in diags
        compare_arrays(rho, diags["air_density"])

        # temperature
        temp = 0.5 * (p[:, :, :-1] + p[:, :, 1:]) / (rd * rho)
        assert "air_temperature" in diags
        compare_arrays(temp, diags["air_temperature"])

        assert len(diags) == 6
    else:
        assert len(diags) == 4


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_isentropic_boussinesq_diagnostics_2(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_f(grid, moist=moist), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    ibd = IsentropicBoussinesqDiagnostics2(
        domain,
        grid_type,
        moist,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in ibd.input_properties
    assert len(ibd.input_properties) == 1

    assert "air_pressure_on_interface_levels" in ibd.diagnostic_properties
    assert "exner_function_on_interface_levels" in ibd.diagnostic_properties
    assert "height_on_interface_levels" in ibd.diagnostic_properties
    assert "montgomery_potential" in ibd.diagnostic_properties
    if moist:
        assert "air_density" in ibd.diagnostic_properties
        assert "air_temperature" in ibd.diagnostic_properties
        assert len(ibd.diagnostic_properties) == 6
    else:
        assert len(ibd.diagnostic_properties) == 4

    diags = ibd(state)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    thetabar = grid.z_on_interface_levels.to_units("K").values[-1]
    dtheta = grid.dz.to_units("K").values.item()

    pref = ibd._pref.value
    rd = ibd._rd.value
    g = ibd._g.value
    cp = ibd._cp.value
    rhoref = ibd._rhoref.value

    # exner function
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    exn = np.zeros((nx, ny, nz + 1), dtype=dtype)
    pt = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]
    exn[:, :, 0] = cp * (pt / pref) ** (rd / cp)
    for k in range(1, nz + 1):
        exn[:, :, k] = exn[:, :, k - 1] + dtheta * g * s[:, :, k - 1] / (
            thetabar * rhoref
        )
    assert "exner_function_on_interface_levels" in diags
    compare_arrays(exn, diags["exner_function_on_interface_levels"])

    # pressure
    p = pref * (exn / cp) ** (cp / rd)
    assert "air_pressure_on_interface_levels" in diags
    compare_arrays(p, diags["air_pressure_on_interface_levels"])

    # montgomery potential
    mtg = np.zeros((nx, ny, nz), dtype=dtype)
    mtg_s = g * grid.topography.profile.to_units("m").values + thetabar * exn[:, :, -1]
    mtg[:, :, -1] = mtg_s + 0.5 * dtheta * exn[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg[:, :, k] = mtg[:, :, k + 1] + dtheta * exn[:, :, k + 1]
    assert "montgomery_potential" in diags
    compare_arrays(mtg, diags["montgomery_potential"])

    # height
    h = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h[:, :, -1] = grid.topography.profile.to_units("m").values
    h[:, :, :-1] = thetabar * (cp - exn[:, :, :-1]) / g
    assert "height_on_interface_levels" in diags
    compare_arrays(h, diags["height_on_interface_levels"])

    if moist:
        # density
        rho = s * dtheta / (h[:, :, :-1] - h[:, :, 1:])
        assert "air_density" in diags
        compare_arrays(rho, diags["air_density"])

        # temperature
        temp = 0.5 * (p[:, :, :-1] + p[:, :, 1:]) / (rd * rho)
        assert "air_temperature" in diags
        compare_arrays(temp, diags["air_temperature"])

        assert len(diags) == 6
    else:
        assert len(diags) == 4


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_isentropic_boussinesq_diagnostics_3(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_f(grid, moist=moist), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    ibd = IsentropicBoussinesqDiagnostics3(
        domain,
        grid_type,
        moist,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in ibd.input_properties
    assert len(ibd.input_properties) == 1

    assert "air_pressure_on_interface_levels" in ibd.diagnostic_properties
    assert "exner_function_on_interface_levels" in ibd.diagnostic_properties
    assert "height_on_interface_levels" in ibd.diagnostic_properties
    assert "montgomery_potential" in ibd.diagnostic_properties
    if moist:
        assert "air_density" in ibd.diagnostic_properties
        assert "air_temperature" in ibd.diagnostic_properties
        assert len(ibd.diagnostic_properties) == 6
    else:
        assert len(ibd.diagnostic_properties) == 4

    diags = ibd(state)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    thetabar = grid.z_on_interface_levels.to_units("K").values[-1]
    dtheta = grid.dz.to_units("K").values.item()

    pref = ibd._pref.value
    rd = ibd._rd.value
    g = ibd._g.value
    cp = ibd._cp.value
    rhoref = ibd._rhoref.value

    # pressure
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    p = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p[:, :, 0] = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]
    for k in range(1, nz + 1):
        p[:, :, k] = p[:, :, k - 1] + dtheta * g * s[:, :, k - 1]
    assert "air_pressure_on_interface_levels" in diags
    compare_arrays(p, diags["air_pressure_on_interface_levels"])

    # exner function
    exn = cp * (p / pref) ** (rd / cp)
    assert "exner_function_on_interface_levels" in diags
    compare_arrays(exn, diags["exner_function_on_interface_levels"])

    # height
    h = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h[:, :, -1] = grid.topography.profile.to_units("m").values
    for k in range(nz - 1, -1, -1):
        h[:, :, k] = h[:, :, k + 1] + dtheta * s[:, :, k] / rhoref
    assert "height_on_interface_levels" in diags
    compare_arrays(h, diags["height_on_interface_levels"])

    # montgomery potential
    mtg = np.zeros((nx, ny, nz), dtype=dtype)
    mtg_s = g * h[:, :, -1] + thetabar * exn[:, :, -1]
    mtg[:, :, -1] = mtg_s - 0.5 * dtheta * (g / thetabar) * h[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg[:, :, k] = mtg[:, :, k + 1] - dtheta * (g / thetabar) * h[:, :, k + 1]
    assert "montgomery_potential" in diags
    compare_arrays(mtg, diags["montgomery_potential"])

    if moist:
        # density
        rho = s * dtheta / (h[:, :, :-1] - h[:, :, 1:])
        assert "air_density" in diags
        compare_arrays(rho, diags["air_density"])

        # temperature
        temp = 0.5 * (p[:, :, :-1] + p[:, :, 1:]) / (rd * rho)
        assert "air_temperature" in diags
        compare_arrays(temp, diags["air_temperature"])

        assert len(diags) == 6
    else:
        assert len(diags) == 4


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_isentropic_boussinesq_diagnostics_4(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_f(grid, moist=moist), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    ibd = IsentropicBoussinesqDiagnostics4(
        domain,
        grid_type,
        moist,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in ibd.input_properties
    assert len(ibd.input_properties) == 1

    assert "air_pressure_on_interface_levels" in ibd.diagnostic_properties
    assert "exner_function_on_interface_levels" in ibd.diagnostic_properties
    assert "height_on_interface_levels" in ibd.diagnostic_properties
    assert "montgomery_potential" in ibd.diagnostic_properties
    if moist:
        assert "air_density" in ibd.diagnostic_properties
        assert "air_temperature" in ibd.diagnostic_properties
        assert len(ibd.diagnostic_properties) == 6
    else:
        assert len(ibd.diagnostic_properties) == 4

    diags = ibd(state)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    thetabar = grid.z_on_interface_levels.to_units("K").values[-1]
    dtheta = grid.dz.to_units("K").values.item()

    pref = ibd._pref.value
    rd = ibd._rd.value
    g = ibd._g.value
    cp = ibd._cp.value

    # pressure
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    p = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p[:, :, 0] = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]
    for k in range(1, nz + 1):
        p[:, :, k] = p[:, :, k - 1] + dtheta * g * s[:, :, k - 1]
    assert "air_pressure_on_interface_levels" in diags
    compare_arrays(p, diags["air_pressure_on_interface_levels"])

    # exner function
    exn = cp * (p / pref) ** (rd / cp)
    assert "exner_function_on_interface_levels" in diags
    compare_arrays(exn, diags["exner_function_on_interface_levels"])

    # height
    h = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h[:, :, -1] = grid.topography.profile.to_units("m").values
    h[:, :, :-1] = thetabar * (cp - exn[:, :, :-1]) / g
    assert "height_on_interface_levels" in diags
    compare_arrays(h, diags["height_on_interface_levels"])

    # montgomery potential
    mtg = np.zeros((nx, ny, nz), dtype=dtype)
    mtg_s = g * h[:, :, -1] + thetabar * exn[:, :, -1]
    mtg[:, :, -1] = mtg_s + 0.5 * dtheta * (cp * thetabar - g * h[:, :, -1]) / thetabar
    for k in range(nz - 2, -1, -1):
        mtg[:, :, k] = (
            mtg[:, :, k + 1] + dtheta * (cp * thetabar - g * h[:, :, k + 1]) / thetabar
        )
    assert "montgomery_potential" in diags
    compare_arrays(mtg, diags["montgomery_potential"])

    if moist:
        # density
        rho = s * dtheta / (h[:, :, :-1] - h[:, :, 1:])
        assert "air_density" in diags
        compare_arrays(rho, diags["air_density"])

        # temperature
        temp = 0.5 * (p[:, :, :-1] + p[:, :, 1:]) / (rd * rho)
        assert "air_temperature" in diags
        compare_arrays(temp, diags["air_temperature"])

        assert len(diags) == 6
    else:
        assert len(diags) == 4


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_isentropic_boussinesq_diagnostics_5(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_f(grid, moist=moist), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    ibd = IsentropicBoussinesqDiagnostics5(
        domain,
        grid_type,
        moist,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in ibd.input_properties
    assert len(ibd.input_properties) == 1

    assert "air_pressure_on_interface_levels" in ibd.diagnostic_properties
    assert "exner_function_on_interface_levels" in ibd.diagnostic_properties
    assert "height_on_interface_levels" in ibd.diagnostic_properties
    assert "montgomery_potential" in ibd.diagnostic_properties
    if moist:
        assert "air_density" in ibd.diagnostic_properties
        assert "air_temperature" in ibd.diagnostic_properties
        assert len(ibd.diagnostic_properties) == 6
    else:
        assert len(ibd.diagnostic_properties) == 4

    diags = ibd(state)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    thetabar = grid.z_on_interface_levels.to_units("K").values[-1]
    dtheta = grid.dz.to_units("K").values.item()

    pref = ibd._pref.value
    rd = ibd._rd.value
    g = ibd._g.value
    cp = ibd._cp.value

    # pressure
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    p = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p[:, :, 0] = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]
    for k in range(1, nz + 1):
        p[:, :, k] = p[:, :, k - 1] + dtheta * g * s[:, :, k - 1]
    assert "air_pressure_on_interface_levels" in diags
    compare_arrays(p, diags["air_pressure_on_interface_levels"])

    # exner function
    exn = cp * (p / pref) ** (rd / cp)
    assert "exner_function_on_interface_levels" in diags
    compare_arrays(exn, diags["exner_function_on_interface_levels"])

    # height
    h = np.zeros((nx, ny, nz + 1), dtype=dtype)
    h[:, :, -1] = grid.topography.profile.to_units("m").values
    h[:, :, :-1] = thetabar * (cp - exn[:, :, :-1]) / g
    assert "height_on_interface_levels" in diags
    compare_arrays(h, diags["height_on_interface_levels"])

    # montgomery potential
    mtg = np.zeros((nx, ny, nz), dtype=dtype)
    mtg_s = g * h[:, :, -1] + thetabar * exn[:, :, -1]
    mtg[:, :, -1] = mtg_s - 0.5 * dtheta * (g / thetabar) * h[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg[:, :, k] = mtg[:, :, k + 1] - dtheta * (g / thetabar) * h[:, :, k + 1]
    assert "montgomery_potential" in diags
    compare_arrays(mtg, diags["montgomery_potential"])

    if moist:
        # density
        rho = s * dtheta / (h[:, :, :-1] - h[:, :, 1:])
        assert "air_density" in diags
        compare_arrays(rho, diags["air_density"])

        # temperature
        temp = 0.5 * (p[:, :, :-1] + p[:, :, 1:]) / (rd * rho)
        assert "air_temperature" in diags
        compare_arrays(temp, diags["air_temperature"])

        assert len(diags) == 6
    else:
        assert len(diags) == 4


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_isentropic_boussinesq_diagnostics_6(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_f(grid, moist=moist), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    ibd = IsentropicBoussinesqDiagnostics6(
        domain,
        grid_type,
        moist,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in ibd.input_properties
    assert len(ibd.input_properties) == 1

    assert "air_pressure_on_interface_levels" in ibd.diagnostic_properties
    assert "exner_function_on_interface_levels" in ibd.diagnostic_properties
    assert "height_on_interface_levels" in ibd.diagnostic_properties
    assert "montgomery_potential" in ibd.diagnostic_properties
    if moist:
        assert "air_density" in ibd.diagnostic_properties
        assert "air_temperature" in ibd.diagnostic_properties
        assert len(ibd.diagnostic_properties) == 6
    else:
        assert len(ibd.diagnostic_properties) == 4

    diags = ibd(state)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    thetabar = grid.z_on_interface_levels.to_units("K").values[-1]
    dtheta = grid.dz.to_units("K").values.item()

    pref = ibd._pref.value
    rd = ibd._rd.value
    g = ibd._g.value
    cp = ibd._cp.value
    rhoref = ibd._rhoref.value

    # pressure
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    p = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p[:, :, 0] = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]
    for k in range(1, nz + 1):
        p[:, :, k] = p[:, :, k - 1] + dtheta * g * s[:, :, k - 1]
    assert "air_pressure_on_interface_levels" in diags
    compare_arrays(p, diags["air_pressure_on_interface_levels"])

    # exner function
    exn = cp * (p / pref) ** (rd / cp)
    assert "exner_function_on_interface_levels" in diags
    compare_arrays(exn, diags["exner_function_on_interface_levels"])

    # dmtg
    hs = grid.topography.profile.to_units("m").values
    dmtg = np.zeros((nx, ny, nz + 1), dtype=dtype)
    dmtg[:, :, -1] = -(g / thetabar) * hs
    for k in range(nz - 1, -1, -1):
        dmtg[:, :, k] = dmtg[:, :, k + 1] - dtheta * g / (rhoref * thetabar) * s[:, :, k]
    compare_arrays(dmtg, ibd._dmtg)

    # height
    h = -(thetabar / g) * dmtg
    assert "height_on_interface_levels" in diags
    compare_arrays(h, diags["height_on_interface_levels"])

    # montgomery potential
    mtg = np.zeros((nx, ny, nz), dtype=dtype)
    mtg_s = g * hs + thetabar * exn[:, :, -1]
    mtg[:, :, -1] = mtg_s + 0.5 * dtheta * dmtg[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg[:, :, k] = mtg[:, :, k + 1] + dtheta * dmtg[:, :, k + 1]
    assert "montgomery_potential" in diags
    compare_arrays(mtg, diags["montgomery_potential"])

    if moist:
        # density
        rho = s * dtheta / (h[:, :, :-1] - h[:, :, 1:])
        assert "air_density" in diags
        compare_arrays(rho, diags["air_density"])

        # temperature
        temp = 0.5 * (p[:, :, :-1] + p[:, :, 1:]) / (rd * rho)
        assert "air_temperature" in diags
        compare_arrays(temp, diags["air_temperature"])

        assert len(diags) == 6
    else:
        assert len(diags) == 4


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_isentropic_boussinesq_diagnostics_7(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    moist = data.draw(hyp_st.booleans(), label="moist")
    state = data.draw(st_isentropic_boussinesq_state_f(grid, moist=moist), label="state")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype

    # ========================================
    # test bed
    # ========================================
    ibd = IsentropicBoussinesqDiagnostics7(
        domain,
        grid_type,
        moist,
        state["air_pressure_on_interface_levels"][0, 0, 0],
        backend=backend,
        dtype=dtype,
    )

    assert "air_isentropic_density" in ibd.input_properties
    assert "dd_montgomery_potential" in ibd.input_properties
    assert len(ibd.input_properties) == 2

    assert "air_pressure_on_interface_levels" in ibd.diagnostic_properties
    assert "exner_function_on_interface_levels" in ibd.diagnostic_properties
    assert "height_on_interface_levels" in ibd.diagnostic_properties
    assert "montgomery_potential" in ibd.diagnostic_properties
    if moist:
        assert "air_density" in ibd.diagnostic_properties
        assert "air_temperature" in ibd.diagnostic_properties
        assert len(ibd.diagnostic_properties) == 6
    else:
        assert len(ibd.diagnostic_properties) == 4

    diags = ibd(state)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    thetabar = grid.z_on_interface_levels.to_units("K").values[-1]
    dtheta = grid.dz.to_units("K").values.item()

    pref = ibd._pref.value
    rd = ibd._rd.value
    g = ibd._g.value
    cp = ibd._cp.value

    # pressure
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    p = np.zeros((nx, ny, nz + 1), dtype=dtype)
    p[:, :, 0] = state["air_pressure_on_interface_levels"].to_units("Pa").values[0, 0, 0]
    for k in range(1, nz + 1):
        p[:, :, k] = p[:, :, k - 1] + dtheta * g * s[:, :, k - 1]
    assert "air_pressure_on_interface_levels" in diags
    compare_arrays(p, diags["air_pressure_on_interface_levels"])

    # exner function
    exn = cp * (p / pref) ** (rd / cp)
    assert "exner_function_on_interface_levels" in diags
    compare_arrays(exn, diags["exner_function_on_interface_levels"])

    # dmtg
    ddmtg = state["dd_montgomery_potential"].to_units("m^2 K^-2 s^-2").values
    hs = grid.topography.profile.to_units("m").values
    dmtg = np.zeros((nx, ny, nz + 1), dtype=dtype)
    dmtg[:, :, -1] = -(g / thetabar) * hs
    for k in range(nz - 1, -1, -1):
        dmtg[:, :, k] = dmtg[:, :, k + 1] + dtheta * ddmtg[:, :, k]
    compare_arrays(dmtg, ibd._dmtg)

    # height
    h = -(thetabar / g) * dmtg
    assert "height_on_interface_levels" in diags
    compare_arrays(h, diags["height_on_interface_levels"])

    # montgomery potential
    mtg = np.zeros((nx, ny, nz), dtype=dtype)
    mtg_s = g * hs + thetabar * exn[:, :, -1]
    mtg[:, :, -1] = mtg_s + 0.5 * dtheta * dmtg[:, :, -1]
    for k in range(nz - 2, -1, -1):
        mtg[:, :, k] = mtg[:, :, k + 1] + dtheta * dmtg[:, :, k + 1]
    assert "montgomery_potential" in diags
    compare_arrays(mtg, diags["montgomery_potential"])

    if moist:
        # density
        rho = s * dtheta / (h[:, :, :-1] - h[:, :, 1:])
        assert "air_density" in diags
        compare_arrays(rho, diags["air_density"])

        # temperature
        temp = 0.5 * (p[:, :, :-1] + p[:, :, 1:]) / (rd * rho)
        assert "air_temperature" in diags
        compare_arrays(temp, diags["air_temperature"])

        assert len(diags) == 6
    else:
        assert len(diags) == 4


if __name__ == "__main__":
    pytest.main([__file__])
