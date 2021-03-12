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
    strategies as hyp_st,
    reproduce_failure,
)
import pytest
from sympl import units_are_same

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper

from tests import conf
from tests.strategies import (
    st_domain,
    st_isentropic_state_f,
    st_one_of,
    st_timedeltas,
)
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test(data, backend, dtype, make_fake_tendency_component_1):
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
    grid = domain.numerical_grid
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="prv_state",
    )

    dt = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    slc = (slice(grid.nx), slice(grid.ny), slice(grid.nz))

    tc1 = make_fake_tendency_component_1(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    fe = STSTendencyStepper.factory(
        "forward_euler",
        tc1,
        execution_policy="serial",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    assert "air_isentropic_density" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["air_isentropic_density"]["units"],
        "kg m^-2 K^-1",
    )
    assert "x_momentum_isentropic" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["x_momentum_isentropic"]["units"],
        "kg m^-1 K^-1 s^-1",
    )
    assert "x_velocity" in fe.provisional_input_properties
    assert units_are_same(
        fe.provisional_input_properties["x_velocity"]["units"], "m s^-1",
    )
    assert len(fe.provisional_input_properties) == 3

    assert "air_isentropic_density" in fe.output_properties
    assert units_are_same(
        fe.output_properties["air_isentropic_density"]["units"], "kg m^-2 K^-1"
    )
    assert "x_momentum_isentropic" in fe.output_properties
    assert units_are_same(
        fe.output_properties["x_momentum_isentropic"]["units"],
        "kg m^-1 K^-1 s^-1",
    )
    assert "x_velocity" in fe.output_properties
    assert units_are_same(
        fe.output_properties["x_velocity"]["units"], "m s^-1"
    )
    assert len(fe.output_properties) == 3

    out_diagnostics, out_state = fe(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s_prv = to_numpy(
        prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    su_prv = to_numpy(
        prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    vx_prv = to_numpy(prv_state["x_velocity"].to_units("m s^-1").data)

    s_tnd = to_numpy(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s_new = s_prv + dt.total_seconds() * s_tnd
    compare_arrays(
        s_new,
        out_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data,
        slice=slc,
    )

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su_new = su_prv + dt.total_seconds() * su_tnd
    compare_arrays(
        su_new,
        out_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        slice=slc,
    )

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx_new = vx_prv + dt.total_seconds() * vx_tnd
    compare_arrays(
        vx_new, out_state["x_velocity"].to_units("m s^-1").data, slice=slc
    )

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].data,
        out_diagnostics["fake_variable"].data,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_hb(data, backend, dtype, make_fake_tendency_component_1):
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
    grid = domain.numerical_grid
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state",
    )
    prv_state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=False,
            precipitation=False,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="prv_state",
    )
    hb = domain.horizontal_boundary
    hb.reference_state = state

    dt = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    slc = (slice(grid.nx), slice(grid.ny), slice(grid.nz))
    hb_np = domain.copy(backend="numpy").horizontal_boundary

    tc1 = make_fake_tendency_component_1(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    fe = STSTendencyStepper.factory(
        "forward_euler",
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    out_diagnostics, out_state = fe(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s_prv = to_numpy(
        prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    su_prv = to_numpy(
        prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    vx_prv = to_numpy(prv_state["x_velocity"].to_units("m s^-1").data)

    s_tnd = to_numpy(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s_new = s_prv + dt.total_seconds() * s_tnd
    hb_np.enforce_field(
        s_new,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
    )
    compare_arrays(s_new, out_state["air_isentropic_density"].data, slice=slc)

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su_new = su_prv + dt.total_seconds() * su_tnd
    hb_np.enforce_field(
        su_new,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
    )
    compare_arrays(su_new, out_state["x_momentum_isentropic"].data, slice=slc)

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx_new = vx_prv + dt.total_seconds() * vx_tnd
    hb_np.enforce_field(
        vx_new,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + dt,
    )
    compare_arrays(vx_new, out_state["x_velocity"].data, slice=slc)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].data,
        out_diagnostics["fake_variable"].data,
    )


if __name__ == "__main__":
    pytest.main([__file__])
