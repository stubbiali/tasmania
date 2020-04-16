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
from sympl import DataArray

import gt4py as gt

from tasmania.python.physics.microphysics.kessler import (
    KesslerFallVelocity,
    KesslerMicrophysics,
    KesslerSaturationAdjustmentDiagnostic,
    KesslerSaturationAdjustmentPrognostic,
    KesslerSedimentation,
)
from tasmania import get_dataarray_3d
from tasmania.python.utils.meteo_utils import goff_gratch_formula, tetens_formula
from tasmania.python.utils.storage_utils import zeros

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.physics.test_microphysics_utils import kessler_sedimentation_validation
from tests.utilities import (
    compare_arrays,
    compare_dataarrays,
    compare_datetimes,
    st_floats,
    st_one_of,
    st_domain,
    st_isentropic_state_f,
)


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def kessler_validation(
    rho, p, t, exn, qv, qc, qr, a, k1, k2, svpf, beta, lhvw, rain_evaporation
):
    p = p if p.shape[2] == rho.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    exn = exn if exn.shape[2] == rho.shape[2] else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])

    ar = k1 * (qc - a) * (qc > a)
    cr = k2 * qc * np.where(qr > 0, qr ** 0.875, 0)

    tnd_qc = -ar - cr
    tnd_qr = ar + cr

    if rain_evaporation:
        ps = svpf(t)
        qvs = beta * ps / p
        er = np.where(
            qr > 0.0, 0.0484794 * (qvs - qv) * (rho * qr) ** (13.0 / 20.0), 0.0
        )
        # er[qr < 0] = 0.0
        tnd_qv = er
        tnd_qr -= er
        tnd_theta = -lhvw * er / exn
    else:
        tnd_qv = 0.0
        tnd_theta = 0.0

    return tnd_qv, tnd_qc, tnd_qr, tnd_theta


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_kessler_microphysics(data, subtests):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    apoif = data.draw(hyp_st.booleans(), label="apoif")
    toaptid = data.draw(hyp_st.booleans(), label="toaptid")
    re = data.draw(hyp_st.booleans(), label="re")
    svpf_type = data.draw(st_one_of(("tetens", "goff_gratch")), label="svpf_type")

    a = data.draw(hyp_st.floats(min_value=0, max_value=10), label="a")
    k1 = data.draw(hyp_st.floats(min_value=0, max_value=10), label="k1")
    k2 = data.draw(hyp_st.floats(min_value=0, max_value=10), label="k2")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    if not apoif:
        p = state["air_pressure_on_interface_levels"].to_units("Pa").values
        p_unstg = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        p_unstg[:, :, :-1] = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
        state["air_pressure"] = get_dataarray_3d(
            p_unstg,
            grid,
            "Pa",
            name="air_pressure",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        exn = state["exner_function_on_interface_levels"].to_units("J kg^-1 K^-1").values
        exn_unstg = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        exn_unstg[:, :, :-1] = 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])
        state["exner_function"] = get_dataarray_3d(
            exn_unstg,
            grid,
            "J kg^-1 K^-1",
            name="exner_function",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

    rd = 287.0
    rv = 461.5
    lhvw = 2.5e6
    beta = rd / rv

    #
    # test properties
    #
    kessler = KesslerMicrophysics(
        domain,
        grid_type,
        air_pressure_on_interface_levels=apoif,
        tendency_of_air_potential_temperature_in_diagnostics=toaptid,
        rain_evaporation=re,
        autoconversion_threshold=DataArray(a, attrs={"units": "g g^-1"}),
        autoconversion_rate=DataArray(k1, attrs={"units": "s^-1"}),
        collection_rate=DataArray(k2, attrs={"units": "hr^-1"}),
        saturation_vapor_pressure_formula=svpf_type,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
        storage_shape=storage_shape,
    )

    assert "air_density" in kessler.input_properties
    assert "air_temperature" in kessler.input_properties
    assert mfwv in kessler.input_properties
    assert mfcw in kessler.input_properties
    assert mfpw in kessler.input_properties
    if apoif:
        assert "air_pressure_on_interface_levels" in kessler.input_properties
        assert "exner_function_on_interface_levels" in kessler.input_properties
    else:
        assert "air_pressure" in kessler.input_properties
        assert "exner_function" in kessler.input_properties
    assert len(kessler.input_properties) == 7

    tendency_names = []
    assert mfcw in kessler.tendency_properties
    tendency_names.append(mfcw)
    assert mfpw in kessler.tendency_properties
    tendency_names.append(mfpw)
    if re:
        assert mfwv in kessler.tendency_properties
        tendency_names.append(mfwv)
        if not toaptid:
            assert "air_potential_temperature" in kessler.tendency_properties
            tendency_names.append("air_potential_temperature")
            assert len(kessler.tendency_properties) == 4
        else:
            assert len(kessler.tendency_properties) == 3
    else:
        assert len(kessler.tendency_properties) == 2

    diagnostic_names = []
    if re and toaptid:
        assert "tendency_of_air_potential_temperature" in kessler.diagnostic_properties
        diagnostic_names.append("tendency_of_air_potential_temperature")
        assert len(kessler.diagnostic_properties) == 1
    else:
        assert len(kessler.diagnostic_properties) == 0

    #
    # test numerics
    #
    tendencies, diagnostics = kessler(state)

    for name in tendency_names:
        with subtests.test(name=name):
            assert name in tendencies
    assert len(tendencies) == len(tendency_names)

    for name in diagnostic_names:
        with subtests.test(name=name):
            assert name in diagnostics
    assert len(diagnostics) == len(diagnostic_names)

    rho = state["air_density"].to_units("kg m^-3").values[:nx, :ny, :nz]
    p = (
        state["air_pressure_on_interface_levels"]
        .to_units("Pa")
        .values[:nx, :ny, : nz + 1]
        if apoif
        else state["air_pressure"].to_units("Pa").values[:nx, :ny, :nz]
    )
    t = state["air_temperature"].to_units("K").values[:nx, :ny, :nz]
    exn = (
        state["exner_function_on_interface_levels"]
        .to_units("J kg^-1 K^-1")
        .values[:nx, :ny, : nz + 1]
        if apoif
        else state["exner_function"].to_units("J kg^-1 K^-1").values[:nx, :ny, :nz]
    )
    qv = state[mfwv].to_units("g g^-1").values[:nx, :ny, :nz]
    qc = state[mfcw].to_units("g g^-1").values[:nx, :ny, :nz]
    qr = state[mfpw].to_units("g g^-1").values[:nx, :ny, :nz]

    # assert kessler._a == a
    # assert kessler._k1 == k1
    # assert np.isclose(kessler._k2, k2/3600.0)

    svpf = goff_gratch_formula if svpf_type == "goff_gratch" else tetens_formula

    tnd_qv, tnd_qc, tnd_qr, tnd_theta = kessler_validation(
        rho, p, t, exn, qv, qc, qr, a, k1, k2 / 3600.0, svpf, beta, lhvw, re
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qc, grid, "g g^-1 s^-1"),
        tendencies[mfcw][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qr, grid, "g g^-1 s^-1"),
        tendencies[mfpw][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )
    if mfwv in tendency_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_qv, grid, "g g^-1 s^-1"),
            tendencies[mfwv][:nx, :ny, :nz],
            compare_coordinate_values=False,
        )
    if "air_potential_temperature" in tendency_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_theta, grid, "K s^-1"),
            tendencies["air_potential_temperature"][:nx, :ny, :nz],
            compare_coordinate_values=False,
        )
    if "tendency_of_air_potential_temperature" in diagnostic_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_theta, grid, "K s^-1"),
            diagnostics["tendency_of_air_potential_temperature"][:nx, :ny, :nz],
            compare_coordinate_values=False,
        )


def kessler_saturation_adjustment_diagnostic_validation(
    timestep, p, t, exn, qv, qc, svpf, beta, lhvw, cp, rv
):
    p = p if p.shape[2] == t.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    exn = exn if exn.shape[2] == t.shape[2] else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])

    ps = svpf(t)
    qvs = beta * ps / p
    sat = (qvs - qv) / (1.0 + qvs * (lhvw ** 2) / (cp * rv * (t ** 2)))
    dq = deepcopy(qc)
    dq[sat <= qc] = sat[sat <= qc]
    dt = -lhvw * dq / cp
    cv = -dq / timestep

    return qv + dq, qc - dq, t + dt, lhvw / exn * cv


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_kessler_saturation_adjustment_diagnostic(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    apoif = data.draw(hyp_st.booleans(), label="apoif")
    svpf_type = data.draw(st_one_of(("tetens", "goff_gratch")), label="svpf_type")

    timestep = data.draw(st_floats(min_value=1e-6, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    if not apoif:
        p = state["air_pressure_on_interface_levels"].to_units("Pa").values
        p_unstg = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        p_unstg[:, :, :-1] = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
        state["air_pressure"] = get_dataarray_3d(
            p_unstg,
            grid,
            "Pa",
            name="air_pressure",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        exn = state["exner_function_on_interface_levels"].to_units("J kg^-1 K^-1").values
        exn_unstg = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        exn_unstg[:, :, :-1] = 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])
        state["exner_function"] = get_dataarray_3d(
            exn_unstg,
            grid,
            "J kg^-1 K^-1",
            name="exner_function",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

    rd = 287.0
    rv = 461.5
    cp = 1004.0
    lhvw = 2.5e6
    beta = rd / rv

    #
    # test properties
    #
    sak = KesslerSaturationAdjustmentDiagnostic(
        domain,
        grid_type,
        air_pressure_on_interface_levels=apoif,
        saturation_vapor_pressure_formula=svpf_type,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        rebuild=False,
        storage_shape=storage_shape,
    )

    assert "air_temperature" in sak.input_properties
    assert mfwv in sak.input_properties
    assert mfcw in sak.input_properties
    if apoif:
        assert "air_pressure_on_interface_levels" in sak.input_properties
        assert "exner_function_on_interface_levels" in sak.input_properties
    else:
        assert "air_pressure" in sak.input_properties
        assert "exner_function" in sak.input_properties
    assert len(sak.input_properties) == 5

    assert "air_potential_temperature" in sak.tendency_properties
    assert len(sak.tendency_properties) == 1

    assert mfwv in sak.diagnostic_properties
    assert mfcw in sak.diagnostic_properties
    assert "air_temperature" in sak.diagnostic_properties
    assert len(sak.diagnostic_properties) == 3

    #
    # test numerics
    #
    dt = timedelta(seconds=timestep)

    tendencies, diagnostics = sak(state, dt)

    assert "air_potential_temperature" in tendencies
    assert len(tendencies) == 1

    assert "air_temperature" in diagnostics
    assert mfwv in diagnostics
    assert mfcw in diagnostics
    assert len(diagnostics) == 3

    p = (
        state["air_pressure_on_interface_levels"]
        .to_units("Pa")
        .values[:nx, :ny, : nz + 1]
        if apoif
        else state["air_pressure"].to_units("Pa").values[:nx, :ny, :nz]
    )
    t = state["air_temperature"].to_units("K").values[:nx, :ny, :nz]
    exn = (
        state["exner_function_on_interface_levels"]
        .to_units("J kg^-1 K^-1")
        .values[:nx, :ny, : nz + 1]
        if apoif
        else state["exner_function"].to_units("J kg^-1 K^-1").values[:nx, :ny, :nz]
    )
    qv = state[mfwv].to_units("g g^-1").values[:nx, :ny, :nz]
    qc = state[mfcw].to_units("g g^-1").values[:nx, :ny, :nz]

    svpf = goff_gratch_formula if svpf_type == "goff_gratch" else tetens_formula

    (
        out_qv,
        out_qc,
        out_t,
        out_theta_tnd,
    ) = kessler_saturation_adjustment_diagnostic_validation(
        dt.total_seconds(), p, t, exn, qv, qc, svpf, beta, lhvw, cp, rv
    )

    compare_dataarrays(
        get_dataarray_3d(out_theta_tnd, grid, "K s^-1"),
        tendencies["air_potential_temperature"][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )

    compare_dataarrays(
        get_dataarray_3d(out_t, grid, "K"),
        diagnostics["air_temperature"][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )
    compare_dataarrays(
        get_dataarray_3d(out_qv, grid, "g g^-1"),
        diagnostics[mfwv][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )
    compare_dataarrays(
        get_dataarray_3d(out_qc, grid, "g g^-1"),
        diagnostics[mfcw][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )


def kessler_saturation_adjustment_prognostic_validation(
    timestep, p, t, exn, qv, qc, svpf, beta, lhvw, cp, rv, sr
):
    p = p if p.shape[2] == t.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    exn = exn if exn.shape[2] == t.shape[2] else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])

    ps = svpf(t)
    qvs = beta * ps / p
    sat = (qvs - qv) / (1.0 + qvs * (lhvw ** 2) / (cp * rv * (t ** 2)))
    dq = deepcopy(qc)
    dq[sat <= qc] = sat[sat <= qc]

    return sr * dq, -sr * dq, -sr * (lhvw / exn) * dq


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_kessler_saturation_adjustment_prognostic(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    apoif = data.draw(hyp_st.booleans(), label="apoif")
    svpf_type = data.draw(st_one_of(("tetens", "goff_gratch")), label="svpf_type")
    sr = data.draw(hyp_st.floats(min_value=0, max_value=1), label="sr")

    timestep = data.draw(st_floats(min_value=1e-6, max_value=3600), label="timestep")

    # ========================================
    # test bed
    # ========================================
    if not apoif:
        p = state["air_pressure_on_interface_levels"].to_units("Pa").values
        p_unstg = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        p_unstg[:, :, :-1] = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
        state["air_pressure"] = get_dataarray_3d(
            p_unstg,
            grid,
            "Pa",
            name="air_pressure",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        exn = state["exner_function_on_interface_levels"].to_units("J kg^-1 K^-1").values
        exn_unstg = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        exn_unstg[:, :, :-1] = 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])
        state["exner_function"] = get_dataarray_3d(
            exn_unstg,
            grid,
            "J kg^-1 K^-1",
            name="exner_function",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

    rd = 287.0
    rv = 461.5
    cp = 1004.0
    lhvw = 2.5e6
    beta = rd / rv

    #
    # test properties
    #
    sak = KesslerSaturationAdjustmentPrognostic(
        domain,
        grid_type,
        air_pressure_on_interface_levels=apoif,
        saturation_vapor_pressure_formula=svpf_type,
        saturation_rate=DataArray(sr, attrs={"units": "s^-1"}),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        rebuild=False,
        storage_shape=storage_shape,
    )

    assert "air_temperature" in sak.input_properties
    assert mfwv in sak.input_properties
    assert mfcw in sak.input_properties
    if apoif:
        assert "air_pressure_on_interface_levels" in sak.input_properties
        assert "exner_function_on_interface_levels" in sak.input_properties
    else:
        assert "air_pressure" in sak.input_properties
        assert "exner_function" in sak.input_properties
    assert len(sak.input_properties) == 5

    assert "air_potential_temperature" in sak.tendency_properties
    assert mfwv in sak.tendency_properties
    assert mfcw in sak.tendency_properties
    assert len(sak.tendency_properties) == 3

    assert len(sak.diagnostic_properties) == 0

    #
    # test numerics
    #
    dt = timedelta(seconds=timestep)

    tendencies, diagnostics = sak(state)

    assert "air_potential_temperature" in tendencies
    assert mfwv in tendencies
    assert mfcw in tendencies
    assert len(tendencies) == 3

    assert len(diagnostics) == 0

    p = (
        state["air_pressure_on_interface_levels"]
        .to_units("Pa")
        .values[:nx, :ny, : nz + 1]
        if apoif
        else state["air_pressure"].to_units("Pa").values[:nx, :ny, :nz]
    )
    t = state["air_temperature"].to_units("K").values[:nx, :ny, :nz]
    exn = (
        state["exner_function_on_interface_levels"]
        .to_units("J kg^-1 K^-1")
        .values[:nx, :ny, : nz + 1]
        if apoif
        else state["exner_function"].to_units("J kg^-1 K^-1").values[:nx, :ny, :nz]
    )
    qv = state[mfwv].to_units("g g^-1").values[:nx, :ny, :nz]
    qc = state[mfcw].to_units("g g^-1").values[:nx, :ny, :nz]

    svpf = goff_gratch_formula if svpf_type == "goff_gratch" else tetens_formula

    tnd_qv, tnd_qc, tnd_theta = kessler_saturation_adjustment_prognostic_validation(
        dt.total_seconds(), p, t, exn, qv, qc, svpf, beta, lhvw, cp, rv, sr
    )

    compare_dataarrays(
        get_dataarray_3d(tnd_theta, grid, "K s^-1"),
        tendencies["air_potential_temperature"][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qv, grid, "g g^-1 s^-1"),
        tendencies[mfwv][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qc, grid, "g g^-1 s^-1"),
        tendencies[mfcw][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )


def kessler_fall_velocity_validation(rho, qr):
    return 36.34 * (0.001 * rho * qr) ** 0.1346 * np.sqrt(rho[:, :, -1:] / rho)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_kessler_fall_velocity(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(gt_powered=gt_powered, backend=backend, dtype=dtype), label="domain"
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    # ========================================
    # test bed
    # ========================================
    #
    # test properties
    #
    rfv = KesslerFallVelocity(
        domain,
        grid_type,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
        storage_shape=storage_shape,
    )

    assert "air_density" in rfv.input_properties
    assert mfpw in rfv.input_properties
    assert len(rfv.input_properties) == 2

    assert "raindrop_fall_velocity" in rfv.diagnostic_properties
    assert len(rfv.diagnostic_properties) == 1

    #
    # test numerics
    #
    diagnostics = rfv(state)

    assert "raindrop_fall_velocity" in diagnostics
    assert len(diagnostics) == 1

    rho = state["air_density"].to_units("kg m^-3").values[:nx, :ny, :nz]
    qr = state[mfpw].to_units("g g^-1").values[:nx, :ny, :nz]

    vt = kessler_fall_velocity_validation(rho, qr)

    compare_dataarrays(
        get_dataarray_3d(vt, grid, "m s^-1"),
        diagnostics["raindrop_fall_velocity"][:nx, :ny, :nz],
        compare_coordinate_values=False,
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
def test_kessler_sedimentation(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            zaxis_length=(3, 20), gt_powered=gt_powered, backend=backend, dtype=dtype
        ),
        label="domain",
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )

    flux_type = data.draw(
        st_one_of(("first_order_upwind", "second_order_upwind")), label="flux_type"
    )
    maxcfl = data.draw(hyp_st.floats(min_value=0, max_value=1), label="maxcfl")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    rfv = KesslerFallVelocity(
        domain,
        grid_type,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
    )
    diagnostics = rfv(state)
    state.update(diagnostics)

    sed = KesslerSedimentation(
        domain,
        grid_type,
        flux_type,
        maxcfl,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=False,
        storage_shape=storage_shape,
    )

    #
    # test properties
    #
    assert "air_density" in sed.input_properties
    assert "height_on_interface_levels" in sed.input_properties
    assert mfpw in sed.input_properties
    assert "raindrop_fall_velocity" in sed.input_properties
    assert len(sed.input_properties) == 4

    assert mfpw in sed.tendency_properties
    assert len(sed.tendency_properties) == 1

    assert len(sed.diagnostic_properties) == 0

    #
    # test numerics
    #
    tendencies, diagnostics = sed(state, timestep)

    assert mfpw in tendencies
    raw_mfpw_val = kessler_sedimentation_validation(
        nx, ny, nz, state, timestep, flux_type, maxcfl
    )

    compare_dataarrays(
        get_dataarray_3d(
            raw_mfpw_val,
            grid,
            "g g^-1 s^-1",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        tendencies[mfpw][:nx, :ny, :nz],
        compare_coordinate_values=False,
    )

    assert len(tendencies) == 1

    assert len(diagnostics) == 0


if __name__ == "__main__":
    # pytest.main([__file__])
    test_kessler_microphysics()
