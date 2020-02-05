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
import pytest

import gt4py as gt

from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
    IsentropicImplicitVerticalAdvectionPrognostic,
)
from tasmania.python.utils.storage_utils import get_dataarray_3d, zeros

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.utils.test_gtscript_utils import thomas_validation
from tests.utilities import (
    compare_arrays,
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_raw_field,
    st_timedeltas,
)


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def setup_tridiagonal_system(gamma, w, phi, a=None, b=None, c=None, d=None):
    ni, nj, nk = phi.shape

    a = deepcopy(phi) if a is None else a
    b = deepcopy(phi) if b is None else b
    c = deepcopy(phi) if c is None else c
    d = deepcopy(phi) if d is None else d

    for i in range(ni):
        for j in range(nj):
            a[i, j, 0] = 0.0
            b[i, j, 0] = 1.0
            c[i, j, 0] = 0.0
            d[i, j, 0] = phi[i, j, 0]

            for k in range(1, nk - 1):
                a[i, j, k] = gamma * w[i, j, k - 1]
                b[i, j, k] = 1.0
                c[i, j, k] = -gamma * w[i, j, k + 1]
                d[i, j, k] = phi[i, j, k] - gamma * (
                    w[i, j, k - 1] * phi[i, j, k - 1] - w[i, j, k + 1] * phi[i, j, k + 1]
                )

            a[i, j, nk - 1] = 0.0
            b[i, j, nk - 1] = 1.0
            c[i, j, nk - 1] = 0.0
            d[i, j, nk - 1] = phi[i, j, nk - 1]

    return a, b, c, d


def validation_diagnostic(
    domain, moist, toaptoil, gt_powered, backend, default_origin, rebuild, state, timestep
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dtype = grid.z.dtype

    storage_shape = state["air_isentropic_density"].shape

    fluxer = IsentropicImplicitVerticalAdvectionDiagnostic(
        domain,
        moist,
        tendency_of_air_potential_temperature_on_interface_levels=toaptoil,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=rebuild,
        storage_shape=storage_shape,
    )

    input_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if toaptoil:
        input_names.append("tendency_of_air_potential_temperature_on_interface_levels")
    else:
        input_names.append("tendency_of_air_potential_temperature")
    if moist:
        input_names.append(mfwv)
        input_names.append(mfcw)
        input_names.append(mfpw)

    output_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        output_names.append(mfwv)
        output_names.append(mfcw)
        output_names.append(mfpw)

    for name in input_names:
        assert name in fluxer.input_properties
    assert len(fluxer.input_properties) == len(input_names)

    assert fluxer.tendency_properties == {}

    for name in output_names:
        assert name in fluxer.diagnostic_properties
    assert len(fluxer.diagnostic_properties) == len(output_names)

    if toaptoil:
        name = "tendency_of_air_potential_temperature_on_interface_levels"
        w_hl = state[name].to_units("K s^-1").values
        w = zeros(
            (nx, ny, nz),
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        w[...] = 0.5 * (w_hl[:nx, :ny, :nz] + w_hl[:nx, :ny, 1 : nz + 1])
    else:
        name = "tendency_of_air_potential_temperature"
        w = state[name].to_units("K s^-1").values[:nx, :ny, :nz]

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
    su = (
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[:nx, :ny, :nz]
    )
    sv = (
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[:nx, :ny, :nz]
    )
    if moist:
        qv = state[mfwv].to_units("g g^-1").values[:nx, :ny, :nz]
        sqv = s * qv
        qc = state[mfcw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqc = s * qc
        qr = state[mfpw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqr = s * qr

    tendencies, diagnostics = fluxer(state, timestep)

    assert tendencies == {}

    dz = grid.dz.to_units("K").values.item()
    dt = timestep.total_seconds()
    gamma = dt / (4.0 * dz)

    a = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    b = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    c = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    d = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    setup_tridiagonal_system(gamma, w, s, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    out_s = deepcopy(out)
    assert "air_isentropic_density" in diagnostics
    compare_arrays(out, diagnostics["air_isentropic_density"].values[:nx, :ny, :nz])

    setup_tridiagonal_system(gamma, w, su, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    assert "x_momentum_isentropic" in diagnostics
    compare_arrays(out, diagnostics["x_momentum_isentropic"].values[:nx, :ny, :nz])

    setup_tridiagonal_system(gamma, w, sv, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    assert "y_momentum_isentropic" in diagnostics
    compare_arrays(out, diagnostics["y_momentum_isentropic"].values[:nx, :ny, :nz])

    if moist:
        setup_tridiagonal_system(gamma, w, sqv, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfwv in diagnostics
        compare_arrays(out, diagnostics[mfwv].values[:nx, :ny, :nz])

        setup_tridiagonal_system(gamma, w, sqc, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfcw in diagnostics
        compare_arrays(out, diagnostics[mfcw].values[:nx, :ny, :nz])

        setup_tridiagonal_system(gamma, w, sqr, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfpw in diagnostics
        compare_arrays(out, diagnostics[mfpw].values[:nx, :ny, :nz])

    assert len(diagnostics) == len(output_names)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_diagnostic_dry(data):
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
    grid = domain.numerical_grid

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape,
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation_diagnostic(
        domain, False, False, gt_powered, backend, default_origin, False, state, timestep
    )
    validation_diagnostic(
        domain, False, True, gt_powered, backend, default_origin, False, state, timestep
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
def test_diagnostic_moist(data):
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
    grid = domain.numerical_grid

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
    field = data.draw(
        st_raw_field(
            storage_shape,
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation_diagnostic(
        domain, True, False, gt_powered, backend, default_origin, False, state, timestep
    )
    validation_diagnostic(
        domain, True, True, gt_powered, backend, default_origin, False, state, timestep
    )


def check_consistency(
    domain, moist, gt_powered, backend, default_origin, rebuild, state, timestep
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dtype = grid.z.dtype

    storage_shape = state["air_isentropic_density"].shape

    fluxer = IsentropicImplicitVerticalAdvectionDiagnostic(
        domain,
        moist,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=rebuild,
        storage_shape=storage_shape,
    )

    input_names = [
        "air_isentropic_density",
        "tendency_of_air_potential_temperature",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        input_names.append(mfwv)
        input_names.append(mfcw)
        input_names.append(mfpw)

    output_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        output_names.append(mfwv)
        output_names.append(mfcw)
        output_names.append(mfpw)

    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        grid,
        units="K s^-1",
        grid_shape=(nx, ny, nz),
        set_coordinates=False,
    )

    tendencies, diagnostics = fluxer(state, timestep)

    compare_arrays(
        diagnostics["air_isentropic_density"]
        .to_units("kg m^-2 K^-1")
        .values[:nx, :ny, :nz],
        state["air_isentropic_density"].to_units("kg m^-2 K^-1").values[:nx, :ny, :nz],
    )
    compare_arrays(
        diagnostics["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values[:nx, :ny, :nz],
        state["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values[:nx, :ny, :nz],
    )
    compare_arrays(
        diagnostics["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values[:nx, :ny, :nz],
        state["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values[:nx, :ny, :nz],
    )
    if moist:
        compare_arrays(
            diagnostics[mfwv].to_units("g g^-1").values[:nx, :ny, :nz],
            state[mfwv].to_units("g g^-1").values[:nx, :ny, :nz],
        )
        compare_arrays(
            diagnostics[mfcw].to_units("g g^-1").values[:nx, :ny, :nz],
            state[mfcw].to_units("g g^-1").values[:nx, :ny, :nz],
        )
        compare_arrays(
            diagnostics[mfpw].to_units("g g^-1").values[:nx, :ny, :nz],
            state[mfpw].to_units("g g^-1").values[:nx, :ny, :nz],
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
def test_diagnostic_consistency(data):
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
    grid = domain.numerical_grid

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

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    check_consistency(
        domain, True, gt_powered, backend, default_origin, False, state, timestep
    )


def validation_prognostic(
    domain, moist, toaptoil, gt_powered, backend, default_origin, rebuild, state, timestep
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dtype = grid.z.dtype

    storage_shape = state["air_isentropic_density"].shape

    fluxer = IsentropicImplicitVerticalAdvectionPrognostic(
        domain,
        moist,
        gt_powered=gt_powered,
        tendency_of_air_potential_temperature_on_interface_levels=toaptoil,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        rebuild=rebuild,
        storage_shape=storage_shape,
    )

    input_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if toaptoil:
        input_names.append("tendency_of_air_potential_temperature_on_interface_levels")
    else:
        input_names.append("tendency_of_air_potential_temperature")
    if moist:
        input_names.append(mfwv)
        input_names.append(mfcw)
        input_names.append(mfpw)

    output_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        output_names.append(mfwv)
        output_names.append(mfcw)
        output_names.append(mfpw)

    for name in input_names:
        assert name in fluxer.input_properties
    assert len(fluxer.input_properties) == len(input_names)

    for name in output_names:
        assert name in fluxer.tendency_properties
    assert len(fluxer.tendency_properties) == len(output_names)

    assert fluxer.diagnostic_properties == {}

    if toaptoil:
        name = "tendency_of_air_potential_temperature_on_interface_levels"
        w_hl = state[name].to_units("K s^-1").values
        w = zeros(
            (nx, ny, nz),
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        w[...] = 0.5 * (w_hl[:nx, :ny, :nz] + w_hl[:nx, :ny, 1 : nz + 1])
    else:
        name = "tendency_of_air_potential_temperature"
        w = state[name].to_units("K s^-1").values[:nx, :ny, :nz]

    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values[:nx, :ny, :nz]
    su = (
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[:nx, :ny, :nz]
    )
    sv = (
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values[:nx, :ny, :nz]
    )
    if moist:
        qv = state[mfwv].to_units("g g^-1").values[:nx, :ny, :nz]
        sqv = s * qv
        qc = state[mfcw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqc = s * qc
        qr = state[mfpw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqr = s * qr

    tendencies, diagnostics = fluxer(state, timestep)

    assert diagnostics == {}

    dz = grid.dz.to_units("K").values.item()
    dt = timestep.total_seconds()
    gamma = dt / (4.0 * dz)

    a = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    b = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    c = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    d = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )
    out = zeros(
        (nx, ny, nz),
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
    )

    setup_tridiagonal_system(gamma, w, s, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    out_s = deepcopy(out)
    out[...] = (out - s) / dt
    assert "air_isentropic_density" in tendencies
    compare_arrays(out, tendencies["air_isentropic_density"].values[:nx, :ny, :nz])

    setup_tridiagonal_system(gamma, w, su, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    out[...] = (out - su) / dt
    assert "x_momentum_isentropic" in tendencies
    compare_arrays(out, tendencies["x_momentum_isentropic"].values[:nx, :ny, :nz])

    setup_tridiagonal_system(gamma, w, sv, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    out[...] = (out - sv) / dt
    assert "y_momentum_isentropic" in tendencies
    compare_arrays(out, tendencies["y_momentum_isentropic"].values[:nx, :ny, :nz])

    if moist:
        setup_tridiagonal_system(gamma, w, sqv, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = (out / out_s - qv) / dt
        assert mfwv in tendencies
        compare_arrays(out, tendencies[mfwv].values[:nx, :ny, :nz])

        setup_tridiagonal_system(gamma, w, sqc, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = (out / out_s - qc) / dt
        assert mfcw in tendencies
        compare_arrays(out, tendencies[mfcw].values[:nx, :ny, :nz])

        setup_tridiagonal_system(gamma, w, sqr, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = (out / out_s - qr) / dt
        assert mfpw in tendencies
        compare_arrays(out, tendencies[mfpw].values[:nx, :ny, :nz])

    assert len(tendencies) == len(output_names)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_prognostic_dry(data):
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
    grid = domain.numerical_grid

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape,
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation_prognostic(
        domain, False, False, gt_powered, backend, default_origin, False, state, timestep
    )
    validation_prognostic(
        domain, False, True, gt_powered, backend, default_origin, False, state, timestep
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
def test_prognostic_moist(data):
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
    grid = domain.numerical_grid

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
    field = data.draw(
        st_raw_field(
            storage_shape,
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state["tendency_of_air_potential_temperature_on_interface_levels"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=1), max_value=timedelta(hours=1)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation_prognostic(
        domain, True, False, gt_powered, backend, default_origin, False, state, timestep
    )
    validation_prognostic(
        domain, True, True, gt_powered, backend, default_origin, False, state, timestep
    )


if __name__ == "__main__":
    pytest.main([__file__])
