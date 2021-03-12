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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
    IsentropicImplicitVerticalAdvectionPrognostic,
)
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import (
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_raw_field,
    st_timedeltas,
)
from tests.utilities import compare_arrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def setup_tridiagonal_system(gamma, w, phi, a=None, b=None, c=None, d=None):
    ni, nj, nk = phi.shape

    a = np.zeros_like(phi) if a is None else a
    b = np.zeros_like(phi) if b is None else b
    c = np.zeros_like(phi) if c is None else c
    d = np.zeros_like(phi) if d is None else d

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
                    w[i, j, k - 1] * phi[i, j, k - 1]
                    - w[i, j, k + 1] * phi[i, j, k + 1]
                )

            a[i, j, nk - 1] = 0.0
            b[i, j, nk - 1] = 1.0
            c[i, j, nk - 1] = 0.0
            d[i, j, nk - 1] = phi[i, j, nk - 1]

    return a, b, c, d


def thomas_validation(a, b, c, d, x=None):
    nx, ny, nz = a.shape

    w = np.zeros_like(b)
    beta = b.copy()
    delta = d.copy()
    for i in range(nx):
        for j in range(ny):
            w[i, j, 0] = 0.0
            for k in range(1, nz):
                w[i, j, k] = (
                    a[i, j, k] / beta[i, j, k - 1]
                    if beta[i, j, k - 1] != 0.0
                    else a[i, j, k]
                )
                beta[i, j, k] = b[i, j, k] - w[i, j, k] * c[i, j, k - 1]
                delta[i, j, k] = d[i, j, k] - w[i, j, k] * delta[i, j, k - 1]

    x = np.zeros_like(b) if x is None else x
    for i in range(nx):
        for j in range(ny):
            x[i, j, -1] = (
                delta[i, j, -1] / beta[i, j, -1]
                if beta[i, j, -1] != 0.0
                else delta[i, j, -1] / b[i, j, -1]
            )
            for k in range(nz - 2, -1, -1):
                x[i, j, k] = (
                    (delta[i, j, k] - c[i, j, k] * x[i, j, k + 1])
                    / beta[i, j, k]
                    if beta[i, j, k] != 0.0
                    else (delta[i, j, k] - c[i, j, k] * x[i, j, k + 1])
                    / b[i, j, k]
                )

    return x


def validation_diagnostic(
    domain,
    moist,
    toaptoil,
    state,
    timestep,
    backend,
    backend_options,
    storage_options,
    *,
    subtests
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    storage_shape = state["air_isentropic_density"].shape

    fluxer = IsentropicImplicitVerticalAdvectionDiagnostic(
        domain,
        moist,
        tendency_of_air_potential_temperature_on_interface_levels=toaptoil,
        backend=backend,
        backend_options=backend_options,
        storage_shape=storage_shape,
        storage_options=storage_options,
    )

    input_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if toaptoil:
        input_names.append(
            "tendency_of_air_potential_temperature_on_interface_levels"
        )
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
        # with subtests.test(name=name):
        assert name in fluxer.input_properties
    assert len(fluxer.input_properties) == len(input_names)

    assert fluxer.tendency_properties == {}

    for name in output_names:
        # with subtests.test(name=name):
        assert name in fluxer.diagnostic_properties
    assert len(fluxer.diagnostic_properties) == len(output_names)

    if toaptoil:
        name = "tendency_of_air_potential_temperature_on_interface_levels"
        w_hl = to_numpy(state[name].to_units("K s^-1").data)
        w = zeros("numpy", shape=(nx, ny, nz), storage_options=storage_options)
        w[...] = 0.5 * (w_hl[:nx, :ny, :nz] + w_hl[:nx, :ny, 1 : nz + 1])
    else:
        name = "tendency_of_air_potential_temperature"
        w = to_numpy(state[name].to_units("K s^-1").data)[:nx, :ny, :nz]

    s = to_numpy(
        state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )[:nx, :ny, :nz]
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )[:nx, :ny, :nz]
    sv = to_numpy(
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )[:nx, :ny, :nz]
    if moist:
        qv = to_numpy(state[mfwv].to_units("g g^-1").data)[:nx, :ny, :nz]
        sqv = s * qv
        qc = to_numpy(state[mfcw].to_units("g g^-1").data)[:nx, :ny, :nz]
        sqc = s * qc
        qr = to_numpy(state[mfpw].to_units("g g^-1").data)[:nx, :ny, :nz]
        sqr = s * qr

    tendencies, diagnostics = fluxer(state, timestep)

    assert tendencies == {}

    dz = grid.dz.to_units("K").values.item()
    dt = timestep.total_seconds()
    gamma = dt / (4.0 * dz)

    slc = (slice(nx), slice(ny), slice(nz))
    out = zeros("numpy", shape=(nx, ny, nz), storage_options=storage_options)

    a, b, c, d = setup_tridiagonal_system(gamma, w, s)
    thomas_validation(a, b, c, d, x=out)
    out_s = out.copy()
    assert "air_isentropic_density" in diagnostics
    compare_arrays(out, diagnostics["air_isentropic_density"].data, slice=slc)

    a, b, c, d = setup_tridiagonal_system(gamma, w, su)
    thomas_validation(a, b, c, d, x=out)
    assert "x_momentum_isentropic" in diagnostics
    compare_arrays(out, diagnostics["x_momentum_isentropic"].data, slice=slc)

    a, b, c, d = setup_tridiagonal_system(gamma, w, sv)
    thomas_validation(a, b, c, d, x=out)
    assert "y_momentum_isentropic" in diagnostics
    compare_arrays(out, diagnostics["y_momentum_isentropic"].data, slice=slc)

    if moist:
        a, b, c, d = setup_tridiagonal_system(gamma, w, sqv)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfwv in diagnostics
        compare_arrays(out, diagnostics[mfwv].data, slice=slc)

        a, b, c, d = setup_tridiagonal_system(gamma, w, sqc)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfcw in diagnostics
        compare_arrays(out, diagnostics[mfcw].data, slice=slc)

        a, b, c, d = setup_tridiagonal_system(gamma, w, sqr)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfpw in diagnostics
        compare_arrays(out, diagnostics[mfpw].data, slice=slc)

    assert len(diagnostics) == len(output_names)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_diagnostic_dry(data, backend, dtype, subtests):
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
            zaxis_length=(3, 50),
            backend=backend,
            backend_options=bo,
            storage_options=so,
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
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape, -1e4, 1e4, backend=backend, storage_options=so
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state[
        "tendency_of_air_potential_temperature_on_interface_levels"
    ] = get_dataarray_3d(
        field,
        grid,
        "K s^-1",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=False,
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=1), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation_diagnostic(
        domain,
        False,
        False,
        state,
        timestep,
        backend,
        bo,
        so,
        subtests=subtests,
    )
    validation_diagnostic(
        domain,
        False,
        True,
        state,
        timestep,
        backend,
        bo,
        so,
        subtests=subtests,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_diagnostic_moist(data, backend, dtype, subtests):
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
            zaxis_length=(3, 50),
            backend=backend,
            backend_options=bo,
            storage_options=so,
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
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape, -1e4, 1e4, backend=backend, storage_options=so
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state[
        "tendency_of_air_potential_temperature_on_interface_levels"
    ] = get_dataarray_3d(
        field,
        grid,
        "K s^-1",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=False,
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=1), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation_diagnostic(
        domain,
        True,
        False,
        state,
        timestep,
        backend,
        bo,
        so,
        subtests=subtests,
    )
    validation_diagnostic(
        domain,
        True,
        True,
        state,
        timestep,
        backend,
        bo,
        so,
        subtests=subtests,
    )


def check_consistency(
    domain, moist, state, timestep, backend, backend_options, storage_options
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    storage_shape = state["air_isentropic_density"].shape

    fluxer = IsentropicImplicitVerticalAdvectionDiagnostic(
        domain,
        moist,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        backend=backend,
        backend_options=backend_options,
        storage_shape=storage_shape,
        storage_options=storage_options,
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
        zeros(backend, shape=storage_shape, storage_options=storage_options),
        grid,
        units="K s^-1",
        grid_shape=(nx, ny, nz),
        set_coordinates=False,
    )

    tendencies, diagnostics = fluxer(state, timestep)

    slc = (slice(nx), slice(ny), slice(nz))
    compare_arrays(
        diagnostics["air_isentropic_density"].to_units("kg m^-2 K^-1").data,
        state["air_isentropic_density"].to_units("kg m^-2 K^-1").data,
        slice=slc,
    )
    compare_arrays(
        diagnostics["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data,
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        slice=slc,
    )
    compare_arrays(
        diagnostics["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data,
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        slice=slc,
    )
    if moist:
        compare_arrays(
            diagnostics[mfwv].to_units("g g^-1").data,
            state[mfwv].to_units("g g^-1").data,
            slice=slc,
        )
        compare_arrays(
            diagnostics[mfcw].to_units("g g^-1").data,
            state[mfcw].to_units("g g^-1").data,
            slice=slc,
        )
        compare_arrays(
            diagnostics[mfpw].to_units("g g^-1").data,
            state[mfpw].to_units("g g^-1").data,
            slice=slc,
        )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_diagnostic_consistency(data, backend, dtype):
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

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=1), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    check_consistency(domain, True, state, timestep, backend, bo, so)


def validation_prognostic(
    domain,
    moist,
    toaptoil,
    state,
    timestep,
    backend,
    backend_options,
    storage_options,
    *,
    subtests
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    storage_shape = state["air_isentropic_density"].shape

    fluxer = IsentropicImplicitVerticalAdvectionPrognostic(
        domain,
        moist,
        tendency_of_air_potential_temperature_on_interface_levels=toaptoil,
        backend=backend,
        backend_options=backend_options,
        storage_shape=storage_shape,
        storage_options=storage_options,
    )

    input_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if toaptoil:
        input_names.append(
            "tendency_of_air_potential_temperature_on_interface_levels"
        )
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
        # with subtests.test(name=name):
        assert name in fluxer.input_properties
    assert len(fluxer.input_properties) == len(input_names)

    for name in output_names:
        # with subtests.test(name=name):
        assert name in fluxer.tendency_properties
    assert len(fluxer.tendency_properties) == len(output_names)

    assert fluxer.diagnostic_properties == {}

    if toaptoil:
        name = "tendency_of_air_potential_temperature_on_interface_levels"
        w_hl = to_numpy(state[name].to_units("K s^-1").data)
        w = zeros("numpy", shape=(nx, ny, nz), storage_options=storage_options)
        w[...] = 0.5 * (w_hl[:nx, :ny, :nz] + w_hl[:nx, :ny, 1 : nz + 1])
    else:
        name = "tendency_of_air_potential_temperature"
        w = to_numpy(state[name].to_units("K s^-1").data)[:nx, :ny, :nz]

    s = to_numpy(
        state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )[:nx, :ny, :nz]
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )[:nx, :ny, :nz]
    sv = to_numpy(
        state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )[:nx, :ny, :nz]

    if moist:
        qv = to_numpy(state[mfwv].to_units("g g^-1").data)[:nx, :ny, :nz]
        sqv = s * qv
        qc = to_numpy(state[mfcw].to_units("g g^-1").data)[:nx, :ny, :nz]
        sqc = s * qc
        qr = to_numpy(state[mfpw].to_units("g g^-1").data)[:nx, :ny, :nz]
        sqr = s * qr

    tendencies, diagnostics = fluxer(state, timestep)

    assert diagnostics == {}

    dz = grid.dz.to_units("K").values.item()
    dt = timestep.total_seconds()
    gamma = dt / (4.0 * dz)

    out = zeros("numpy", shape=(nx, ny, nz), storage_options=storage_options)
    slc = (slice(nx), slice(ny), slice(nz))

    a, b, c, d = setup_tridiagonal_system(gamma, w, s)
    thomas_validation(a, b, c, d, x=out)
    out_s = out.copy()
    out[...] = (out - s) / dt
    assert "air_isentropic_density" in tendencies
    compare_arrays(out, tendencies["air_isentropic_density"].data, slice=slc)

    setup_tridiagonal_system(gamma, w, su, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    out[...] = (out - su) / dt
    assert "x_momentum_isentropic" in tendencies
    compare_arrays(out, tendencies["x_momentum_isentropic"].data, slice=slc)

    setup_tridiagonal_system(gamma, w, sv, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    out[...] = (out - sv) / dt
    assert "y_momentum_isentropic" in tendencies
    compare_arrays(out, tendencies["y_momentum_isentropic"].data, slice=slc)

    if moist:
        setup_tridiagonal_system(gamma, w, sqv, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = (out / out_s - qv) / dt
        assert mfwv in tendencies
        compare_arrays(out, tendencies[mfwv].data, slice=slc)

        setup_tridiagonal_system(gamma, w, sqc, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = (out / out_s - qc) / dt
        assert mfcw in tendencies
        compare_arrays(out, tendencies[mfcw].data, slice=slc)

        setup_tridiagonal_system(gamma, w, sqr, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = (out / out_s - qr) / dt
        assert mfpw in tendencies
        compare_arrays(out, tendencies[mfpw].data, slice=slc)

    assert len(tendencies) == len(output_names)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_prognostic_dry(data, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(zaxis_length=(3, 20), backend=backend, dtype=dtype),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state",
    )
    field = data.draw(
        st_raw_field(
            storage_shape, -1e4, 1e4, backend=backend, storage_options=so
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state[
        "tendency_of_air_potential_temperature_on_interface_levels"
    ] = get_dataarray_3d(
        field,
        grid,
        "K s^-1",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=False,
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=1), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation_prognostic(
        domain,
        False,
        False,
        state,
        timestep,
        backend,
        bo,
        so,
        subtests=subtests,
    )
    validation_prognostic(
        domain,
        False,
        True,
        state,
        timestep,
        backend,
        bo,
        so,
        subtests=subtests,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_prognostic_moist(data, backend, dtype, subtests):
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
    field = data.draw(
        st_raw_field(
            storage_shape, -1e4, 1e4, backend=backend, storage_options=so
        ),
        label="field",
    )
    state["tendency_of_air_potential_temperature"] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
    )
    state[
        "tendency_of_air_potential_temperature_on_interface_levels"
    ] = get_dataarray_3d(
        field,
        grid,
        "K s^-1",
        grid_shape=(nx, ny, nz + 1),
        set_coordinates=False,
    )

    timestep = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=1), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation_prognostic(
        domain,
        True,
        False,
        state,
        timestep,
        backend,
        bo,
        so,
        subtests=subtests,
    )
    validation_prognostic(
        domain,
        True,
        True,
        state,
        timestep,
        backend,
        bo,
        so,
        subtests=subtests,
    )


if __name__ == "__main__":
    pytest.main([__file__])
    # test_diagnostic_dry("numpy", float, None)
