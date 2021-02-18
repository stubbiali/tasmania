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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.framework.allocators import as_storage, zeros
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.physics.microphysics.kessler import (
    KesslerFallVelocity,
    KesslerMicrophysics,
    KesslerSaturationAdjustmentDiagnostic,
    KesslerSaturationAdjustmentPrognostic,
    KesslerSedimentation,
)
from tasmania import get_dataarray_3d
from tasmania.python.utils.meteo import tetens_formula

from tests import conf
from tests.physics.test_microphysics_utils import (
    kessler_sedimentation_validation,
)
from tests.strategies import (
    st_floats,
    st_one_of,
    st_domain,
    st_isentropic_state_f,
)
from tests.utilities import compare_dataarrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def kessler_validation(
    rho, p, t, exn, qv, qc, qr, a, k1, k2, svpf, beta, lhvw, rain_evaporation
):
    p = p if p.shape[2] == rho.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    exn = (
        exn
        if exn.shape[2] == rho.shape[2]
        else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])
    )

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


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend.difference(conf.gtc_backend))
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_microphysics(data, backend, dtype, subtests):
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
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
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

    apoif = data.draw(hyp_st.booleans(), label="apoif")
    toaptid = data.draw(hyp_st.booleans(), label="toaptid")
    re = data.draw(hyp_st.booleans(), label="re")

    a = data.draw(hyp_st.floats(min_value=0, max_value=10), label="a")
    k1 = data.draw(hyp_st.floats(min_value=0, max_value=10), label="k1")
    k2 = data.draw(hyp_st.floats(min_value=0, max_value=10), label="k2")

    # ========================================
    # test bed
    # ========================================
    if not apoif:
        p_np = to_numpy(
            state["air_pressure_on_interface_levels"].to_units("Pa").data
        )
        p_unstg = zeros(backend, shape=storage_shape, storage_options=so)
        p_unstg[:, :, :-1] = as_storage(
            backend,
            data=0.5 * (p_np[:, :, :-1] + p_np[:, :, 1:]),
            storage_options=so,
        )
        state["air_pressure"] = get_dataarray_3d(
            p_unstg,
            grid,
            "Pa",
            name="air_pressure",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        exn_np = to_numpy(
            state["exner_function_on_interface_levels"]
            .to_units("J kg^-1 K^-1")
            .data
        )
        exn_unstg = zeros(backend, shape=storage_shape, storage_options=so)
        exn_unstg[:, :, :-1] = as_storage(
            backend,
            data=0.5 * (exn_np[:, :, :-1] + exn_np[:, :, 1:]),
            storage_options=so,
        )
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
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
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
        assert (
            "tendency_of_air_potential_temperature"
            in kessler.diagnostic_properties
        )
        diagnostic_names.append("tendency_of_air_potential_temperature")
        assert len(kessler.diagnostic_properties) == 1
    else:
        assert len(kessler.diagnostic_properties) == 0

    #
    # test numerics
    #
    tendencies, diagnostics = kessler(state)

    for name in tendency_names:
        # with subtests.test(name=name):
        assert name in tendencies
    assert len(tendencies) == len(tendency_names)

    for name in diagnostic_names:
        # with subtests.test(name=name):
        assert name in diagnostics
    assert len(diagnostics) == len(diagnostic_names)

    rho_np = to_numpy(state["air_density"].to_units("kg m^-3").data)[
        :nx, :ny, :nz
    ]
    p_np = to_numpy(
        state["air_pressure_on_interface_levels" if apoif else "air_pressure"]
        .to_units("Pa")
        .data
    )
    p_np = p_np[:nx, :ny, : nz + 1] if apoif else p_np[:nx, :ny, :nz]
    t_np = to_numpy(state["air_temperature"].to_units("K").data[:nx, :ny, :nz])
    exn_np = to_numpy(
        state[
            "exner_function_on_interface_levels" if apoif else "exner_function"
        ]
        .to_units("J kg^-1 K^-1")
        .data
    )
    exn_np = exn_np[:nx, :ny, : nz + 1] if apoif else exn_np[:nx, :ny, :nz]
    qv_np = to_numpy(state[mfwv].to_units("g g^-1").data)[:nx, :ny, :nz]
    qc_np = to_numpy(state[mfcw].to_units("g g^-1").data)[:nx, :ny, :nz]
    qr_np = to_numpy(state[mfpw].to_units("g g^-1").data)[:nx, :ny, :nz]

    # assert kessler._a == a
    # assert kessler._k1 == k1
    # assert np.isclose(kessler._k2, k2/3600.0)

    tnd_qv, tnd_qc, tnd_qr, tnd_theta = kessler_validation(
        rho_np,
        p_np,
        t_np,
        exn_np,
        qv_np,
        qc_np,
        qr_np,
        a,
        k1,
        k2 / 3600.0,
        tetens_formula,
        beta,
        lhvw,
        re,
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qc, grid, "g g^-1 s^-1"),
        tendencies[mfcw],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qr, grid, "g g^-1 s^-1"),
        tendencies[mfpw],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )
    if mfwv in tendency_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_qv, grid, "g g^-1 s^-1"),
            tendencies[mfwv],
            compare_coordinate_values=False,
            slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
        )
    if "air_potential_temperature" in tendency_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_theta, grid, "K s^-1"),
            tendencies["air_potential_temperature"],
            compare_coordinate_values=False,
            slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
        )
    if "tendency_of_air_potential_temperature" in diagnostic_names:
        compare_dataarrays(
            get_dataarray_3d(tnd_theta, grid, "K s^-1"),
            diagnostics["tendency_of_air_potential_temperature"],
            compare_coordinate_values=False,
            slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
        )


def kessler_saturation_adjustment_diagnostic_validation(
    timestep, p, t, exn, qv, qc, svpf, beta, lhvw, cp, rv
):
    p = p if p.shape[2] == t.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    exn = (
        exn
        if exn.shape[2] == t.shape[2]
        else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])
    )

    ps = svpf(t)
    qvs = beta * ps / p
    sat = (qvs - qv) / (1.0 + qvs * (lhvw ** 2) / (cp * rv * (t ** 2)))
    dq = deepcopy(qc)
    dq[sat <= qc] = sat[sat <= qc]
    dt = -lhvw * dq / cp
    cv = -dq / timestep

    return qv + dq, qc - dq, t + dt, lhvw / exn * cv


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend.difference(conf.gtc_backend))
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_saturation_adjustment_diagnostic(data, backend, dtype):
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
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
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

    apoif = data.draw(hyp_st.booleans(), label="apoif")

    timestep = data.draw(
        st_floats(min_value=1e-6, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    if not apoif:
        p_np = to_numpy(
            state["air_pressure_on_interface_levels"].to_units("Pa").data
        )
        p_unstg = zeros(backend, shape=storage_shape, storage_options=so)
        p_unstg[:, :, :-1] = as_storage(
            backend,
            data=0.5 * (p_np[:, :, :-1] + p_np[:, :, 1:]),
            storage_options=so,
        )
        state["air_pressure"] = get_dataarray_3d(
            p_unstg,
            grid,
            "Pa",
            name="air_pressure",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        exn_np = to_numpy(
            state["exner_function_on_interface_levels"]
            .to_units("J kg^-1 K^-1")
            .data
        )
        exn_unstg = zeros(backend, shape=storage_shape, storage_options=so)
        exn_unstg[:, :, :-1] = as_storage(
            backend,
            data=0.5 * (exn_np[:, :, :-1] + exn_np[:, :, 1:]),
            storage_options=so,
        )
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
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
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

    p_np = to_numpy(
        state["air_pressure_on_interface_levels" if apoif else "air_pressure"]
        .to_units("Pa")
        .data
    )
    p_np = p_np[:nx, :ny, : nz + 1] if apoif else p_np[:nx, :ny, :nz]
    t_np = to_numpy(state["air_temperature"].to_units("K").data)[:nx, :ny, :nz]
    exn_np = to_numpy(
        state[
            "exner_function_on_interface_levels" if apoif else "exner_function"
        ]
        .to_units("J kg^-1 K^-1")
        .data
    )
    exn_np = exn_np[:nx, :ny, : nz + 1] if apoif else exn_np[:nx, :ny, :nz]
    qv_np = to_numpy(state[mfwv].to_units("g g^-1").data)[:nx, :ny, :nz]
    qc_np = to_numpy(state[mfcw].to_units("g g^-1").data)[:nx, :ny, :nz]

    (
        out_qv,
        out_qc,
        out_t,
        out_theta_tnd,
    ) = kessler_saturation_adjustment_diagnostic_validation(
        dt.total_seconds(),
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

    compare_dataarrays(
        get_dataarray_3d(out_theta_tnd, grid, "K s^-1"),
        tendencies["air_potential_temperature"],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )

    compare_dataarrays(
        get_dataarray_3d(out_t, grid, "K"),
        diagnostics["air_temperature"],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )
    compare_dataarrays(
        get_dataarray_3d(out_qv, grid, "g g^-1"),
        diagnostics[mfwv],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )
    compare_dataarrays(
        get_dataarray_3d(out_qc, grid, "g g^-1"),
        diagnostics[mfcw],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )


def kessler_saturation_adjustment_prognostic_validation(
    timestep, p, t, exn, qv, qc, svpf, beta, lhvw, cp, rv, sr
):
    p = p if p.shape[2] == t.shape[2] else 0.5 * (p[:, :, :-1] + p[:, :, 1:])
    exn = (
        exn
        if exn.shape[2] == t.shape[2]
        else 0.5 * (exn[:, :, :-1] + exn[:, :, 1:])
    )

    ps = svpf(t)
    qvs = beta * ps / p
    sat = (qvs - qv) / (1.0 + qvs * (lhvw ** 2) / (cp * rv * (t ** 2)))
    dq = deepcopy(qc)
    dq[sat <= qc] = sat[sat <= qc]

    return sr * dq, -sr * dq, -sr * (lhvw / exn) * dq


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend.difference(conf.gtc_backend))
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_saturation_adjustment_prognostic(data, backend, dtype):
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
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
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

    apoif = data.draw(hyp_st.booleans(), label="apoif")
    sr = data.draw(hyp_st.floats(min_value=0, max_value=1), label="sr")

    timestep = data.draw(
        st_floats(min_value=1e-6, max_value=3600), label="timestep"
    )

    # ========================================
    # test bed
    # ========================================
    if not apoif:
        p_np = to_numpy(
            state["air_pressure_on_interface_levels"].to_units("Pa").data
        )
        p_unstg = zeros(backend, shape=storage_shape, storage_options=so)
        p_unstg[:, :, :-1] = as_storage(
            backend,
            data=0.5 * (p_np[:, :, :-1] + p_np[:, :, 1:]),
            storage_options=so,
        )
        state["air_pressure"] = get_dataarray_3d(
            p_unstg,
            grid,
            "Pa",
            name="air_pressure",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        exn_np = to_numpy(
            state["exner_function_on_interface_levels"]
            .to_units("J kg^-1 K^-1")
            .data
        )
        exn_unstg = zeros(backend, shape=storage_shape, storage_options=so)
        exn_unstg[:, :, :-1] = as_storage(
            backend,
            data=0.5 * (exn_np[:, :, :-1] + exn_np[:, :, 1:]),
            storage_options=so,
        )
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
        saturation_rate=DataArray(sr, attrs={"units": "s^-1"}),
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
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

    p_np = to_numpy(
        state["air_pressure_on_interface_levels" if apoif else "air_pressure"]
        .to_units("Pa")
        .data
    )
    p_np = p_np[:nx, :ny, : nz + 1] if apoif else p_np[:nx, :ny, :nz]
    t_np = to_numpy(state["air_temperature"].to_units("K").data)[:nx, :ny, :nz]
    exn_np = to_numpy(
        state[
            "exner_function_on_interface_levels" if apoif else "exner_function"
        ]
        .to_units("J kg^-1 K^-1")
        .data
    )
    exn_np = exn_np[:nx, :ny, : nz + 1] if apoif else exn_np[:nx, :ny, :nz]
    qv_np = to_numpy(state[mfwv].to_units("g g^-1").data)[:nx, :ny, :nz]
    qc_np = to_numpy(state[mfcw].to_units("g g^-1").data)[:nx, :ny, :nz]

    (
        tnd_qv,
        tnd_qc,
        tnd_theta,
    ) = kessler_saturation_adjustment_prognostic_validation(
        dt.total_seconds(),
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
        sr,
    )

    compare_dataarrays(
        get_dataarray_3d(tnd_theta, grid, "K s^-1"),
        tendencies["air_potential_temperature"],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qv, grid, "g g^-1 s^-1"),
        tendencies[mfwv],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )
    compare_dataarrays(
        get_dataarray_3d(tnd_qc, grid, "g g^-1 s^-1"),
        tendencies[mfcw],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )


def kessler_fall_velocity_validation(rho, qr):
    return 36.34 * (0.001 * rho * qr) ** 0.1346 * np.sqrt(rho[:, :, -1:] / rho)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend.difference(conf.gtc_backend))
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_fall_velocity(data, backend, dtype):
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
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
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
    #
    # test properties
    #
    rfv = KesslerFallVelocity(
        domain,
        grid_type,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
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

    rho_np = to_numpy(state["air_density"].to_units("kg m^-3").data)[
        :nx, :ny, :nz
    ]
    qr_np = to_numpy(state[mfpw].to_units("g g^-1").data)[:nx, :ny, :nz]

    vt = kessler_fall_velocity_validation(rho_np, qr_np)

    compare_dataarrays(
        get_dataarray_3d(vt, grid, "m s^-1"),
        diagnostics["raindrop_fall_velocity"],
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend.difference(conf.gtc_backend))
@pytest.mark.parametrize("dtype", conf.dtype)
def test_kessler_sedimentation(data, backend, dtype):
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
            zaxis_length=(3, 20),
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

    flux_type = data.draw(
        st_one_of(("first_order_upwind", "second_order_upwind")),
        label="flux_type",
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
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    diagnostics = rfv(state)
    state.update(diagnostics)

    sed = KesslerSedimentation(
        domain,
        grid_type,
        flux_type,
        maxcfl,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
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
        tendencies[mfpw],
        compare_coordinate_values=False,
        slice=(slice(nx), slice(ny), slice(nz)),
    )

    assert len(tendencies) == 1

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
    # test_kessler_microphysics("gt4py:gtx86", np.float32, None)
