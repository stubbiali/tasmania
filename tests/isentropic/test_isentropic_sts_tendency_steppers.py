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
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

import gt4py

from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
)
from tasmania.python.isentropic.physics.sts_tendency_steppers import (
    IsentropicVerticalAdvection,
)
from tasmania.python.utils.storage_utils import get_dataarray_3d, zeros

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.utils.test_gtscript_utils import thomas_validation
from tests.utilities import (
    compare_arrays,
    compare_datetimes,
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_raw_field,
    st_timedeltas,
)


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def setup_tridiagonal_system(gamma, w, phi, phi_prv, a=None, b=None, c=None, d=None):
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
            d[i, j, 0] = phi_prv[i, j, 0]

            for k in range(1, nk - 1):
                a[i, j, k] = gamma * w[i, j, k - 1]
                b[i, j, k] = 1.0
                c[i, j, k] = -gamma * w[i, j, k + 1]
                d[i, j, k] = phi_prv[i, j, k] - gamma * (
                    w[i, j, k - 1] * phi[i, j, k - 1] - w[i, j, k + 1] * phi[i, j, k + 1]
                )

            a[i, j, nk - 1] = 0.0
            b[i, j, nk - 1] = 1.0
            c[i, j, nk - 1] = 0.0
            d[i, j, nk - 1] = phi_prv[i, j, nk - 1]

    return a, b, c, d


def validation(
    domain,
    moist,
    toaptoil,
    gt_powered,
    backend,
    default_origin,
    rebuild,
    state,
    state_prv,
    timestep,
    *,
    subtests
):
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dtype = grid.z.dtype

    storage_shape = state["air_isentropic_density"].shape

    core = IsentropicImplicitVerticalAdvectionDiagnostic(
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
    stepper = IsentropicVerticalAdvection(
        core, gt_powered=gt_powered, backend=backend, dtype=dtype, rebuild=rebuild
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

    prv_input_names = [
        "air_isentropic_density",
        "x_momentum_isentropic",
        "y_momentum_isentropic",
    ]
    if moist:
        prv_input_names.append(mfwv)
        prv_input_names.append(mfcw)
        prv_input_names.append(mfpw)

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
        with subtests.test(name=name):
            assert name in stepper.input_properties
    assert len(stepper.input_properties) == len(input_names)

    for name in prv_input_names:
        with subtests.test(name=name):
            assert name in stepper.provisional_input_properties
    assert len(stepper.provisional_input_properties) == len(prv_input_names)

    assert stepper.diagnostic_properties == {}

    for name in output_names:
        with subtests.test(name=name):
            assert name in stepper.output_properties
    assert len(stepper.output_properties) == len(output_names)

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
        state["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values[:nx, :ny, :nz]
    )
    sv = (
        state["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values[:nx, :ny, :nz]
    )
    if moist:
        qv = state[mfwv].to_units("g g^-1").values[:nx, :ny, :nz]
        sqv = s * qv
        qc = state[mfcw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqc = s * qc
        qr = state[mfpw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqr = s * qr

    s_prv = (
        state_prv["air_isentropic_density"]
        .to_units("kg m^-2 K^-1")
        .values[:nx, :ny, :nz]
    )
    su_prv = (
        state_prv["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values[:nx, :ny, :nz]
    )
    sv_prv = (
        state_prv["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values[:nx, :ny, :nz]
    )
    if moist:
        qv_prv = state_prv[mfwv].to_units("g g^-1").values[:nx, :ny, :nz]
        sqv_prv = s_prv * qv_prv
        qc_prv = state_prv[mfcw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqc_prv = s_prv * qc_prv
        qr_prv = state_prv[mfpw].to_units("g g^-1").values[:nx, :ny, :nz]
        sqr_prv = s_prv * qr_prv

    diagnostics, state_out = stepper(state, state_prv, timestep)

    assert "time" in diagnostics
    compare_datetimes(state["time"], diagnostics["time"])
    assert len(diagnostics) == 1

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

    setup_tridiagonal_system(gamma, w, s, s_prv, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    out_s = deepcopy(out)
    assert "air_isentropic_density" in state_out
    compare_arrays(out, state_out["air_isentropic_density"].values[:nx, :ny, :nz])

    setup_tridiagonal_system(gamma, w, su, su_prv, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    assert "x_momentum_isentropic" in state_out
    compare_arrays(out, state_out["x_momentum_isentropic"].values[:nx, :ny, :nz])

    setup_tridiagonal_system(gamma, w, sv, sv_prv, a=a, b=b, c=c, d=d)
    thomas_validation(a, b, c, d, x=out)
    assert "y_momentum_isentropic" in state_out
    compare_arrays(out, state_out["y_momentum_isentropic"].values[:nx, :ny, :nz])

    if moist:
        setup_tridiagonal_system(gamma, w, sqv, sqv_prv, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfwv in state_out
        compare_arrays(out, state_out[mfwv].values[:nx, :ny, :nz])

        setup_tridiagonal_system(gamma, w, sqc, sqc_prv, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfcw in state_out
        compare_arrays(out, state_out[mfcw].values[:nx, :ny, :nz])

        setup_tridiagonal_system(gamma, w, sqr, sqr_prv, a=a, b=b, c=c, d=d)
        thomas_validation(a, b, c, d, x=out)
        out[...] = out / out_s
        assert mfpw in state_out
        compare_arrays(out, state_out[mfpw].values[:nx, :ny, :nz])

    assert "time" in state_out
    compare_datetimes(state_out["time"], state["time"] + timestep)

    assert len(state_out) == len(output_names) + 1


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_isentropic_vertical_advection_dry(data, subtests):
    gt4py.storage.prepare_numpy()

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
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
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
    state[
        "tendency_of_air_potential_temperature_on_interface_levels"
    ] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="prv_state",
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        domain,
        False,
        False,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        state_prv,
        timestep,
        subtests=subtests,
    )
    validation(
        domain,
        False,
        True,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        state_prv,
        timestep,
        subtests=subtests,
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_isentropic_vertical_advection_moist(data, subtests):
    gt4py.storage.prepare_numpy()

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
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    storage_shape = (nx + 1, ny + 1, nz + 1)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            precipitation=False,
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
    state[
        "tendency_of_air_potential_temperature_on_interface_levels"
    ] = get_dataarray_3d(
        field, grid, "K s^-1", grid_shape=(nx, ny, nz + 1), set_coordinates=False
    )

    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            precipitation=False,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape,
        ),
        label="prv_state",
    )

    timestep = data.draw(
        st_timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    validation(
        domain,
        True,
        False,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        state_prv,
        timestep,
        subtests=subtests,
    )
    validation(
        domain,
        True,
        True,
        gt_powered,
        backend,
        default_origin,
        False,
        state,
        state_prv,
        timestep,
        subtests=subtests,
    )


if __name__ == "__main__":
    pytest.main([__file__])
