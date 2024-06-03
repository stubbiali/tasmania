# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import (
    BackendOptions,
    StorageOptions,
    TimeIntegrationOptions,
)
from tasmania.python.framework.sequential_tendency_splitting import (
    SequentialTendencySplitting,
)
from tasmania.python.utils.storage import deepcopy_dataarray_dict

from tests import conf
from tests.strategies import st_domain, st_isentropic_state_f, st_one_of
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
def test_properties(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )

    # ========================================
    # test bed
    # ========================================
    tendency1 = make_fake_tendency_component_1(domain, grid_type)
    tendency2 = make_fake_tendency_component_2(domain, grid_type)

    #
    # test 1
    #
    sts = SequentialTendencySplitting(
        TimeIntegrationOptions(tendency2, scheme="forward_euler"),
        TimeIntegrationOptions(tendency1, scheme="forward_euler"),
    )

    assert "air_isentropic_density" in sts.input_properties
    assert "fake_variable" in sts.input_properties
    assert "x_momentum_isentropic" in sts.input_properties
    assert "x_velocity" in sts.input_properties
    assert "x_velocity_at_u_locations" in sts.input_properties
    assert "y_momentum_isentropic" in sts.input_properties
    assert "y_velocity_at_v_locations" in sts.input_properties
    assert len(sts.input_properties) == 7

    assert "air_isentropic_density" in sts.provisional_input_properties
    assert "x_momentum_isentropic" in sts.provisional_input_properties
    assert "x_velocity" in sts.provisional_input_properties
    assert "y_momentum_isentropic" in sts.provisional_input_properties
    assert len(sts.provisional_input_properties) == 4

    assert "air_isentropic_density" in sts.output_properties
    assert "fake_variable" in sts.output_properties
    assert "x_momentum_isentropic" in sts.output_properties
    assert "x_velocity" in sts.output_properties
    assert "x_velocity_at_u_locations" in sts.output_properties
    assert "y_momentum_isentropic" in sts.output_properties
    assert "y_velocity_at_v_locations" in sts.output_properties
    assert len(sts.output_properties) == 7

    assert "air_isentropic_density" in sts.provisional_output_properties
    assert "x_momentum_isentropic" in sts.provisional_output_properties
    assert "x_velocity" in sts.provisional_output_properties
    assert "y_momentum_isentropic" in sts.provisional_output_properties
    assert len(sts.provisional_output_properties) == 4

    #
    # test 2
    #
    sts = SequentialTendencySplitting(
        TimeIntegrationOptions(tendency1, scheme="forward_euler"),
        TimeIntegrationOptions(tendency2, scheme="forward_euler"),
    )

    assert "air_isentropic_density" in sts.input_properties
    assert "x_momentum_isentropic" in sts.input_properties
    assert "x_velocity" in sts.input_properties
    assert "x_velocity_at_u_locations" in sts.input_properties
    assert "y_momentum_isentropic" in sts.input_properties
    assert "y_velocity_at_v_locations" in sts.input_properties
    assert len(sts.input_properties) == 6

    assert "air_isentropic_density" in sts.provisional_input_properties
    assert "x_momentum_isentropic" in sts.provisional_input_properties
    assert "x_velocity" in sts.provisional_input_properties
    assert "y_momentum_isentropic" in sts.provisional_input_properties
    assert len(sts.provisional_input_properties) == 4

    assert "air_isentropic_density" in sts.output_properties
    assert "fake_variable" in sts.output_properties
    assert "x_momentum_isentropic" in sts.output_properties
    assert "x_velocity" in sts.output_properties
    assert "x_velocity_at_u_locations" in sts.output_properties
    assert "y_momentum_isentropic" in sts.output_properties
    assert "y_velocity_at_v_locations" in sts.output_properties
    assert len(sts.output_properties) == 7

    assert "air_isentropic_density" in sts.provisional_output_properties
    assert "x_momentum_isentropic" in sts.provisional_output_properties
    assert "x_velocity" in sts.provisional_output_properties
    assert "y_momentum_isentropic" in sts.provisional_output_properties
    assert len(sts.provisional_output_properties) == 4

    #
    # test 3
    #
    sts = SequentialTendencySplitting(
        TimeIntegrationOptions(tendency1, scheme="forward_euler", substeps=3),
        TimeIntegrationOptions(tendency2, scheme="forward_euler", substeps=2),
    )

    assert "air_isentropic_density" in sts.input_properties
    assert "x_momentum_isentropic" in sts.input_properties
    assert "x_velocity" in sts.input_properties
    assert "x_velocity_at_u_locations" in sts.input_properties
    assert "y_momentum_isentropic" in sts.input_properties
    assert "y_velocity_at_v_locations" in sts.input_properties
    assert len(sts.input_properties) == 6

    assert "air_isentropic_density" in sts.provisional_input_properties
    assert "x_momentum_isentropic" in sts.provisional_input_properties
    assert "x_velocity" in sts.provisional_input_properties
    assert "y_momentum_isentropic" in sts.provisional_input_properties
    assert len(sts.provisional_input_properties) == 4

    assert "air_isentropic_density" in sts.output_properties
    assert "fake_variable" in sts.output_properties
    assert "x_momentum_isentropic" in sts.output_properties
    assert "x_velocity" in sts.output_properties
    assert "x_velocity_at_u_locations" in sts.output_properties
    assert "y_momentum_isentropic" in sts.output_properties
    assert "y_velocity_at_v_locations" in sts.output_properties
    assert len(sts.output_properties) == 7

    assert "air_isentropic_density" in sts.provisional_output_properties
    assert "x_momentum_isentropic" in sts.provisional_output_properties
    assert "x_velocity" in sts.provisional_output_properties
    assert "y_momentum_isentropic" in sts.provisional_output_properties
    assert len(sts.provisional_output_properties) == 4


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_forward_euler(
    data,
    backend,
    dtype,
    make_fake_tendency_component_1,
    make_fake_tendency_component_2,
):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(3, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

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
    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state_prv",
    )

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    slc = (slice(nx), slice(ny), slice(nz))

    tendency1 = make_fake_tendency_component_1(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    tendency2 = make_fake_tendency_component_2(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    sts = SequentialTendencySplitting(
        TimeIntegrationOptions(
            tendency1,
            scheme="forward_euler",
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        TimeIntegrationOptions(
            tendency2,
            scheme="forward_euler",
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
    )
    sts(state, state_prv, timestep)

    assert "fake_variable" in state
    s0 = to_numpy(
        state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    f1 = to_numpy(state["fake_variable"].data)
    compare_arrays(f1, 2 * s0, slice=slc)

    assert "air_isentropic_density" in state_prv
    s1 = to_numpy(
        state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    s2 = s1 + timestep.total_seconds() * 0.001 * s0
    s3 = s2 + timestep.total_seconds() * 0.01 * f1
    compare_arrays(state_prv["air_isentropic_density"].data, s3, slice=slc)

    assert "x_momentum_isentropic" in state_prv
    su0 = to_numpy(
        state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    su1 = to_numpy(
        state_prv_dc["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data
    )
    su2 = su1 + timestep.total_seconds() * 300 * su0
    compare_arrays(
        state_prv["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        su2,
        slice=slc,
    )

    assert "x_velocity" in state_prv
    u0 = to_numpy(
        state_dc["x_velocity_at_u_locations"].to_units("m s^-1").data
    )
    vx1 = to_numpy(state_prv_dc["x_velocity"].to_units("m s^-1").data)
    vx2 = np.zeros_like(vx1)
    vx2[:nx, :ny, :nz] = vx1[:nx, :ny, :nz] + timestep.total_seconds() * 50 * (
        u0[:nx, :ny, :nz] + u0[1 : nx + 1, :ny, :nz]
    )
    compare_arrays(
        state_prv["x_velocity"].to_units("m s^-1").data, vx2, slice=slc
    )

    assert "y_momentum_isentropic" in state_prv
    v0 = to_numpy(
        state_dc["y_velocity_at_v_locations"].to_units("m s^-1").data
    )
    sv1 = to_numpy(
        state_prv_dc["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data
    )
    sv3 = np.zeros_like(sv1)
    sv3[:nx, :ny, :nz] = sv1[
        :nx, :ny, :nz
    ] + timestep.total_seconds() * 0.5 * s0[:nx, :ny, :nz] * (
        v0[:nx, :ny, :nz] + v0[:nx, 1 : ny + 1, :nz]
    )
    compare_arrays(state_prv["y_momentum_isentropic"].data, sv3, slice=slc)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_rk2(
    data,
    backend,
    dtype,
    make_fake_tendency_component_1,
    make_fake_tendency_component_2,
):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(3, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

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
    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        ),
        label="state_prv",
    )

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    # ========================================
    # test bed
    # ========================================
    slc = (slice(nx), slice(ny), slice(nz))

    tendency1 = make_fake_tendency_component_1(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )
    tendency2 = make_fake_tendency_component_2(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    sts = SequentialTendencySplitting(
        TimeIntegrationOptions(
            tendency1,
            scheme="rk2",
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        TimeIntegrationOptions(
            tendency2,
            scheme="rk2",
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
    )

    sts(state, state_prv, timestep)

    assert "fake_variable" in state
    assert "air_isentropic_density" in state_prv
    s0 = to_numpy(
        state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    s1 = state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    s2b = 0.5 * (s0 + s1 + timestep.total_seconds() * 0.001 * s0)
    s2 = s1 + timestep.total_seconds() * 0.001 * s2b
    f2 = to_numpy(state["fake_variable"].to_units("kg m^-2 K^-1").data)
    compare_arrays(f2, 2 * s2b, slice=slc)
    s3b = 0.5 * (s0 + s2 + timestep.total_seconds() * 0.01 * f2)
    s3 = s2 + timestep.total_seconds() * 0.01 * f2
    compare_arrays(
        state_prv["air_isentropic_density"].to_units("kg m^-2 K^-1").data,
        s3,
        slice=slc,
    )

    assert "x_momentum_isentropic" in state_prv
    su0 = to_numpy(
        state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    su1 = to_numpy(
        state_prv_dc["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data
    )
    su2b = 0.5 * (su0 + su1 + timestep.total_seconds() * 300 * su0)
    su2 = su1 + timestep.total_seconds() * 300 * su2b
    compare_arrays(
        state_prv["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        su2,
        slice=slc,
    )

    assert "x_velocity" in state_prv
    u0 = to_numpy(
        state_dc["x_velocity_at_u_locations"].to_units("m s^-1").data
    )
    vx1 = to_numpy(state_prv_dc["x_velocity"].to_units("m s^-1").data)
    vx2 = np.zeros_like(vx1)
    vx2[:nx, :ny, :nz] = vx1[:nx, :ny, :nz] + timestep.total_seconds() * 50 * (
        u0[:nx, :ny, :nz] + u0[1 : nx + 1, :ny, :nz]
    )
    compare_arrays(
        state_prv["x_velocity"].to_units("m s^-1").data, vx2, slice=slc
    )

    assert "y_momentum_isentropic" in state_prv
    v0 = to_numpy(
        state_dc["y_velocity_at_v_locations"].to_units("m s^-1").data
    )
    sv0 = to_numpy(
        state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    sv1 = to_numpy(
        state_prv_dc["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data
    )
    sv3b = np.zeros_like(sv1)
    sv3b[:nx, :ny, :nz] = 0.5 * (
        sv0[:nx, :ny, :nz]
        + sv1[:nx, :ny, :nz]
        + timestep.total_seconds()
        * 0.5
        * s0[:nx, :ny, :nz]
        * (v0[:nx, :ny, :nz] + v0[:nx, 1 : ny + 1, :nz])
    )
    sv3 = np.zeros_like(sv1)
    sv3[:nx, :ny, :nz] = sv1[
        :nx, :ny, :nz
    ] + timestep.total_seconds() * 0.5 * s3b[:nx, :ny, :nz] * (
        v0[:nx, :ny, :nz] + v0[:nx, 1 : ny + 1, :nz]
    )
    compare_arrays(state_prv["y_momentum_isentropic"].data, sv3, slice=slc)


if __name__ == "__main__":
    pytest.main([__file__])
