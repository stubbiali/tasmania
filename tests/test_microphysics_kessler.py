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

from tasmania.python.physics.microphysics.kessler import (
    KesslerFallVelocity,
    KesslerMicrophysics,
    KesslerSaturationAdjustment,
    KesslerSedimentation,
)
from tasmania import get_dataarray_3d
from tasmania.python.utils.meteo_utils import goff_gratch_formula, tetens_formula

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .test_microphysics_utils import kessler_sedimentation_validation
    from .utils import (
        compare_arrays,
        compare_dataarrays,
        compare_datetimes,
        st_floats,
        st_one_of,
        st_domain,
        st_isentropic_state_f,
    )
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from test_microphysics_utils import kessler_sedimentation_validation
    from utils import (
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
    rho, p, t, exn, qv, qc, qr, a, k1, k2, swvf, beta, lhvw, rain_evaporation
):
    p = p if p.shape[2] == rho.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    exn = exn if exn.shape[2] == rho.shape[2] else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])

    p_mbar = 0.01 * p
    rho_gcm3 = 0.001 * rho

    ar = k1 * (qc - a) * (qc > a)
    cr = k2 * qc * qr ** 0.875

    tnd_qc = -ar - cr
    tnd_qr = ar + cr

    if rain_evaporation:
        ps = swvf(t)
        qvs = beta * ps / (p - ps)
        c = 1.6 + 124.9 * (rho_gcm3 * qr) ** 0.2046
        er = (
            (1.0 - qv / qvs)
            * c
            * (rho_gcm3 * qr) ** 0.525
            / (rho_gcm3 * (5.4e5 + 2.55e6 / (p_mbar * qvs)))
        )
        tnd_qv = er
        tnd_qr -= er
        tnd_theta = -lhvw * er / exn
    else:
        tnd_qv = 0.0
        tnd_theta = 0.0

    return tnd_qv, tnd_qc, tnd_qr, tnd_theta


def kessler_validation_bis(
    rho, p, t, exn, qv, qc, qr, a, k1, k2, swvf, beta, lhvw, rain_evaporation
):
    p = p if p.shape[2] == rho.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    exn = exn if exn.shape[2] == rho.shape[2] else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])

    p_mbar = 0.01 * p
    rho_gcm3 = 0.001 * rho

    ar = k1 * (qc - a) * (qc > a)
    cr = np.zeros_like(qv, dtype=qv.dtype)
    k = qr > 0.0
    cr[k] = k2 * qc[k] * qr[k] ** 0.875

    tnd_qc = -ar - cr
    tnd_qr = ar + cr

    if rain_evaporation:
        ps = swvf(t)
        qvs = beta * ps / (p - ps)
        c = np.zeros_like(qv, dtype=qv.dtype)
        c[k] = 1.6 + 124.9 * (rho_gcm3[k] * qr[k]) ** 0.2046
        er = np.zeros_like(qv, dtype=qv.dtype)
        er[k] = (
            (1.0 - qv[k] / qvs[k])
            * c[k]
            * (rho_gcm3[k] * qr[k]) ** 0.525
            / (rho_gcm3[k] * (5.4e5 + 2.55e6 / (p_mbar[k] * qvs[k])))
        )
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
@given(hyp_st.data())
def _test_kessler_microphysics(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

    apoif = data.draw(hyp_st.booleans(), label="apoif")
    toaptid = data.draw(hyp_st.booleans(), label="toaptid")
    re = data.draw(hyp_st.booleans(), label="re")
    swvf_type = data.draw(st_one_of(("tetens", "goff_gratch")), label="swvf_type")

    a = data.draw(hyp_st.floats(min_value=0, max_value=10), label="a")
    k1 = data.draw(hyp_st.floats(min_value=0, max_value=10), label="k1")
    k2 = data.draw(hyp_st.floats(min_value=0, max_value=10), label="k2")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    if not apoif:
        p = state["air_pressure_on_interface_levels"].to_units("Pa").values
        state["air_pressure"] = get_dataarray_3d(
            0.5 * (p[:, :, :-1] + p[:, :, 1:]), grid, "Pa", name="air_pressure"
        )
        exn = state["exner_function_on_interface_levels"].to_units("J kg^-1 K^-1").values
        state["exner_function"] = get_dataarray_3d(
            0.5 * (exn[:, :, :-1] + exn[:, :, 1:]),
            grid,
            "J kg^-1 K^-1",
            name="exner_function",
        )

    dtype = grid.x.dtype

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
        saturation_water_vapor_formula=swvf_type,
        backend=backend,
        dtype=dtype,
        halo=halo,
        rebuild=True,
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
        assert name in tendencies
    assert len(tendencies) == len(tendency_names)

    for name in diagnostic_names:
        assert name in diagnostics
    assert len(diagnostics) == len(diagnostic_names)

    rho = state["air_density"].to_units("kg m^-3").values
    p = (
        state["air_pressure_on_interface_levels"].to_units("Pa").values
        if apoif
        else state["air_pressure"].to_units("Pa").values
    )
    t = state["air_temperature"].to_units("K").values
    exn = (
        state["exner_function_on_interface_levels"].to_units("J kg^-1 K^-1").values
        if apoif
        else state["exner_function"].to_units("J kg^-1 K^-1").values
    )
    qv = state[mfwv].to_units("g g^-1").values
    qc = state[mfcw].to_units("g g^-1").values
    qr = state[mfpw].to_units("g g^-1").values

    # assert kessler._a == a
    # assert kessler._k1 == k1
    # assert np.isclose(kessler._k2, k2/3600.0)

    swvf = goff_gratch_formula if swvf_type == "goff_gratch" else tetens_formula

    tnd_qv, tnd_qc, tnd_qr, tnd_theta = kessler_validation(
        rho, p, t, exn, qv, qc, qr, a, k1, k2 / 3600.0, swvf, beta, lhvw, re
    )
    # tnd_qv, tnd_qc, tnd_qr, tnd_theta = kessler_validation_bis(
    # 	rho, p, t, exn, qv, qc, qr, a, k1, k2/3600.0, swvf, beta, lhvw, re
    # )

    compare_dataarrays(
        get_dataarray_3d(tnd_qc, grid, "g g^-1 s^-1"),
        tendencies[mfcw],
        compare_coordinate_values=False,
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qr, grid, "g g^-1 s^-1"),
        tendencies[mfpw],
        compare_coordinate_values=False,
    )
    if mfwv in tendency_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_qv, grid, "g g^-1 s^-1"),
            tendencies[mfwv],
            compare_coordinate_values=False,
        )
    if "air_potential_temperature" in tendency_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_theta, grid, "K s^-1"),
            tendencies["air_potential_temperature"],
            compare_coordinate_values=False,
        )
    if "tendency_of_air_potential_temperature" in diagnostic_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_theta, grid, "K s^-1"),
            diagnostics["tendency_of_air_potential_temperature"],
            compare_coordinate_values=False,
        )


def kessler_saturation_adjustment_validation(p, t, qv, qc, beta, lhvw, cp):
    p = p if p.shape[2] == t.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    pvs = tetens_formula(t)
    qvs = beta * pvs / (p - pvs)
    d = np.minimum((qvs - qv) / (1.0 + qvs * 4093 * lhvw / (cp * (t - 36) ** 2)), qc)
    return qv + d, qc - d


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_kessler_saturation_adjustment(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

    apoif = data.draw(hyp_st.booleans(), label="apoif")

    backend = data.draw(st_one_of(conf_backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    if not apoif:
        p = state["air_pressure_on_interface_levels"].to_units("Pa").values
        state["air_pressure"] = get_dataarray_3d(
            0.5 * (p[:, :, :-1] + p[:, :, 1:]), grid, "Pa", name="air_pressure"
        )

    dtype = grid.x.dtype

    rd = 287.0
    rv = 461.5
    cp = 1004.0
    lhvw = 2.5e6
    beta = rd / rv

    #
    # test properties
    #
    sak = KesslerSaturationAdjustment(
        domain,
        grid_type,
        air_pressure_on_interface_levels=apoif,
        backend=backend,
        dtype=dtype,
        rebuild=True,
    )

    assert "air_temperature" in sak.input_properties
    assert mfwv in sak.input_properties
    assert mfcw in sak.input_properties
    if apoif:
        assert "air_pressure_on_interface_levels" in sak.input_properties
    else:
        assert "air_pressure" in sak.input_properties
    assert len(sak.input_properties) == 4

    assert mfwv in sak.diagnostic_properties
    assert mfcw in sak.diagnostic_properties
    assert len(sak.diagnostic_properties) == 2

    #
    # test numerics
    #
    diagnostics = sak(state)

    assert mfwv in diagnostics
    assert mfcw in diagnostics
    assert len(diagnostics) == 2

    p = (
        state["air_pressure_on_interface_levels"].to_units("Pa").values
        if apoif
        else state["air_pressure"].to_units("Pa").values
    )
    t = state["air_temperature"].to_units("K").values
    qv = state[mfwv].to_units("g g^-1").values
    qc = state[mfcw].to_units("g g^-1").values

    out_qv, out_qc = kessler_saturation_adjustment_validation(
        p, t, qv, qc, beta, lhvw, cp
    )

    compare_dataarrays(
        get_dataarray_3d(out_qv, grid, "g g^-1"),
        diagnostics[mfwv],
        compare_coordinate_values=False,
    )
    compare_dataarrays(
        get_dataarray_3d(out_qc, grid, "g g^-1"),
        diagnostics[mfcw],
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
def _test_kessler_fall_velocity(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    #
    # test properties
    #
    rfv = KesslerFallVelocity(
        domain, grid_type, backend=backend, dtype=dtype, halo=halo, rebuild=False
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

    rho = state["air_density"].to_units("kg m^-3").values
    qr = state[mfpw].to_units("g g^-1").values

    vt = kessler_fall_velocity_validation(rho, qr)

    compare_dataarrays(
        get_dataarray_3d(vt, grid, "m s^-1"),
        diagnostics["raindrop_fall_velocity"],
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
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(zaxis_length=(3, 20)), label="domain")

    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.physical_grid if grid_type == "physical" else domain.numerical_grid
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

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

    backend = data.draw(st_one_of(conf_backend), label="backend")
    halo = data.draw(st_one_of(conf_halo), label="halo")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    rfv = KesslerFallVelocity(domain, grid_type, backend=backend, dtype=dtype, halo=halo)
    diagnostics = rfv(state)
    state.update(diagnostics)

    sed = KesslerSedimentation(
        domain,
        grid_type,
        flux_type,
        maxcfl,
        backend=backend,
        dtype=dtype,
        halo=halo,
        rebuild=True,
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
    raw_mfpw_val = kessler_sedimentation_validation(state, timestep, flux_type, maxcfl)
    compare_dataarrays(
        get_dataarray_3d(raw_mfpw_val, grid, "g g^-1 s^-1"),
        tendencies[mfpw],
        compare_coordinate_values=False,
    )
    assert len(tendencies) == 1

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
    # test_kessler_microphysics()
