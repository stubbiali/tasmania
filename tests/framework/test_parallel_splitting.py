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

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import (
    BackendOptions,
    StorageOptions,
    TimeIntegrationOptions,
)
from tasmania.python.framework.parallel_splitting import ParallelSplitting
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
    ps = ParallelSplitting(
        TimeIntegrationOptions(component=tendency1, scheme="forward_euler"),
        TimeIntegrationOptions(component=tendency2, scheme="forward_euler"),
        execution_policy="as_parallel",
    )

    assert "air_isentropic_density" in ps.input_properties
    assert "fake_variable" in ps.input_properties
    assert "x_momentum_isentropic" in ps.input_properties
    assert "x_velocity" in ps.input_properties
    assert "x_velocity_at_u_locations" in ps.input_properties
    assert "y_momentum_isentropic" in ps.input_properties
    assert "y_velocity_at_v_locations" in ps.input_properties
    assert len(ps.input_properties) == 7

    assert "air_isentropic_density" in ps.provisional_input_properties
    assert "x_momentum_isentropic" in ps.provisional_input_properties
    assert "x_velocity" in ps.provisional_input_properties
    assert "y_momentum_isentropic" in ps.provisional_input_properties
    assert len(ps.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.output_properties
    assert "fake_variable" in ps.output_properties
    assert "x_momentum_isentropic" in ps.output_properties
    assert "x_velocity" in ps.output_properties
    assert "x_velocity_at_u_locations" in ps.output_properties
    assert "y_momentum_isentropic" in ps.output_properties
    assert "y_velocity_at_v_locations" in ps.output_properties
    assert len(ps.output_properties) == 7

    assert "air_isentropic_density" in ps.provisional_output_properties
    assert "x_momentum_isentropic" in ps.provisional_output_properties
    assert "x_velocity" in ps.provisional_output_properties
    assert "y_momentum_isentropic" in ps.provisional_output_properties
    assert len(ps.provisional_output_properties) == 4

    #
    # test 2
    #
    ps = ParallelSplitting(
        TimeIntegrationOptions(component=tendency1, scheme="forward_euler"),
        TimeIntegrationOptions(component=tendency2, scheme="forward_euler"),
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=False,
    )

    assert "air_isentropic_density" in ps.input_properties
    assert "x_momentum_isentropic" in ps.input_properties
    assert "x_velocity" in ps.input_properties
    assert "x_velocity_at_u_locations" in ps.input_properties
    assert "y_momentum_isentropic" in ps.input_properties
    assert "y_velocity_at_v_locations" in ps.input_properties
    assert len(ps.input_properties) == 6

    assert "air_isentropic_density" in ps.provisional_input_properties
    assert "x_momentum_isentropic" in ps.provisional_input_properties
    assert "x_velocity" in ps.provisional_input_properties
    assert "y_momentum_isentropic" in ps.provisional_input_properties
    assert len(ps.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.output_properties
    assert "fake_variable" in ps.output_properties
    assert "x_momentum_isentropic" in ps.output_properties
    assert "x_velocity" in ps.output_properties
    assert "x_velocity_at_u_locations" in ps.output_properties
    assert "y_momentum_isentropic" in ps.output_properties
    assert "y_velocity_at_v_locations" in ps.output_properties
    assert len(ps.output_properties) == 7

    assert "air_isentropic_density" in ps.provisional_output_properties
    assert "x_momentum_isentropic" in ps.provisional_output_properties
    assert "x_velocity" in ps.provisional_output_properties
    assert "y_momentum_isentropic" in ps.provisional_output_properties
    assert len(ps.provisional_output_properties) == 4

    #
    # test 3
    #
    ps = ParallelSplitting(
        TimeIntegrationOptions(component=tendency1, scheme="forward_euler"),
        TimeIntegrationOptions(component=tendency2, scheme="forward_euler"),
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=True,
    )

    assert "air_isentropic_density" in ps.input_properties
    assert "x_momentum_isentropic" in ps.input_properties
    assert "x_velocity" in ps.input_properties
    assert "x_velocity_at_u_locations" in ps.input_properties
    assert "y_momentum_isentropic" in ps.input_properties
    assert "y_velocity_at_v_locations" in ps.input_properties
    assert len(ps.input_properties) == 6

    assert "air_isentropic_density" in ps.provisional_input_properties
    assert "x_momentum_isentropic" in ps.provisional_input_properties
    assert "x_velocity" in ps.provisional_input_properties
    assert "y_momentum_isentropic" in ps.provisional_input_properties
    assert len(ps.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.output_properties
    assert "fake_variable" in ps.output_properties
    assert "x_momentum_isentropic" in ps.output_properties
    assert "x_velocity" in ps.output_properties
    assert "x_velocity_at_u_locations" in ps.output_properties
    assert "y_momentum_isentropic" in ps.output_properties
    assert "y_velocity_at_v_locations" in ps.output_properties
    assert len(ps.output_properties) == 7

    assert "air_isentropic_density" in ps.provisional_output_properties
    assert "x_momentum_isentropic" in ps.provisional_output_properties
    assert "x_velocity" in ps.provisional_output_properties
    assert "y_momentum_isentropic" in ps.provisional_output_properties
    assert len(ps.provisional_output_properties) == 4


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

    ts_kwargs = {
        "backend": backend,
        "backend_options": bo,
        "storage_options": so,
    }
    ps_kwargs = {
        "backend": backend,
        "backend_options": bo,
        "storage_options": so,
    }

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf.nb)), label="nb"
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

    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

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
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
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

    ps = ParallelSplitting(
        TimeIntegrationOptions(tendency1, scheme="forward_euler", **ts_kwargs),
        TimeIntegrationOptions(tendency2, scheme="forward_euler", **ts_kwargs),
        execution_policy="serial",
        **ps_kwargs
    )

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    ps(state=state, state_prv=state_prv, timestep=timestep)

    assert "fake_variable" in state
    s = to_numpy(
        state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    f = to_numpy(state["fake_variable"].to_units("kg m^-2 K^-1").data)
    compare_arrays(f, 2 * s, slice=slc)

    assert "air_isentropic_density" in state_prv
    s1 = to_numpy(
        state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    s2 = s + timestep.total_seconds() * 0.001 * s
    s3 = s + timestep.total_seconds() * 0.01 * f
    s_out = s1 + (s2 - s) + (s3 - s)
    compare_arrays(
        state_prv["air_isentropic_density"].to_units("kg m^-2 K^-1").data,
        s_out,
        slice=slc,
    )

    assert "x_momentum_isentropic" in state_prv
    su = to_numpy(
        state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    su1 = to_numpy(
        state_prv_dc["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data
    )
    su2 = su + timestep.total_seconds() * 300 * su
    su_out = su1 + (su2 - su)
    compare_arrays(
        state_prv["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        su_out,
        slice=slc,
    )

    assert "x_velocity" in state_prv
    vx = to_numpy(state_dc["x_velocity"].to_units("m s^-1").data)
    u = to_numpy(state_dc["x_velocity_at_u_locations"].to_units("m s^-1").data)
    vx1 = to_numpy(state_prv_dc["x_velocity"].to_units("m s^-1").data)
    vx2 = np.zeros_like(vx)
    vx2[:nx, :ny, :nz] = vx[:nx, :ny, :nz] + timestep.total_seconds() * 50 * (
        u[:nx, :ny, :nz] + u[1 : nx + 1, :ny, :nz]
    )
    vx_out = vx1 + (vx2 - vx)
    compare_arrays(
        state_prv["x_velocity"].to_units("m s^-1").data, vx_out, slice=slc
    )

    assert "y_momentum_isentropic" in state_prv
    v = to_numpy(state_dc["y_velocity_at_v_locations"].to_units("m s^-1").data)
    sv = to_numpy(
        state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    sv1 = to_numpy(
        state_prv_dc["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data
    )
    sv3 = np.zeros_like(sv)
    sv3[:nx, :ny, :nz] = sv[
        :nx, :ny, :nz
    ] + timestep.total_seconds() * 0.5 * s[:nx, :ny, :nz] * (
        v[:nx, :ny, :nz] + v[:nx, 1 : ny + 1, :nz]
    )
    sv_out = sv1 + (sv3 - sv)
    compare_arrays(
        state_prv["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        sv_out,
        slice=slc,
    )


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

    ts_kwargs = {
        "backend": backend,
        "backend_options": bo,
        "storage_options": so,
    }
    ps_kwargs = {
        "backend": backend,
        "backend_options": bo,
        "storage_options": so,
    }

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf.nb)), label="nb"
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

    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

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
    grid = domain.numerical_grid
    nx, ny, nz = grid.nx, grid.ny, grid.nz
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

    ps = ParallelSplitting(
        TimeIntegrationOptions(tendency1, scheme="rk2", **ts_kwargs),
        TimeIntegrationOptions(tendency2, scheme="rk2", **ts_kwargs),
        execution_policy="serial",
        **ps_kwargs
    )

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    ps(state=state, state_prv=state_prv, timestep=timestep)

    assert "air_isentropic_density" in state_prv
    assert "fake_variable" in state
    s = to_numpy(
        state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    s1 = to_numpy(
        state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    s2b = s + 0.5 * timestep.total_seconds() * 0.001 * s
    s2 = s + timestep.total_seconds() * 0.001 * s2b
    f = to_numpy(state["fake_variable"].to_units("kg m^-2 K^-1").data)
    s3b = s + 0.5 * timestep.total_seconds() * 0.01 * f
    s3 = s + timestep.total_seconds() * 0.01 * f
    s_out = s1 + (s2 - s) + (s3 - s)
    compare_arrays(f, 2 * s2b, slice=slc)
    compare_arrays(
        state_prv["air_isentropic_density"].to_units("kg m^-2 K^-1").data,
        s_out,
        slice=slc,
    )

    assert "x_momentum_isentropic" in state_prv
    su = to_numpy(
        state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    su1 = to_numpy(
        state_prv_dc["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data
    )
    su2b = su + 0.5 * timestep.total_seconds() * 300 * su
    su2 = su + timestep.total_seconds() * 300 * su2b
    su_out = su1 + (su2 - su)
    compare_arrays(
        state_prv["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        su_out,
        slice=slc,
    )

    assert "x_velocity" in state_prv
    vx = to_numpy(state_dc["x_velocity"].to_units("m s^-1").data)
    u = to_numpy(state_dc["x_velocity_at_u_locations"].to_units("m s^-1").data)
    vx1 = to_numpy(state_prv_dc["x_velocity"].to_units("m s^-1").data)
    vx2 = np.zeros_like(vx)
    vx2[:nx, :ny, :nz] = vx[:nx, :ny, :nz] + timestep.total_seconds() * 50 * (
        u[:nx, :ny, :nz] + u[1 : nx + 1, :ny, :nz]
    )
    vx_out = vx1 + (vx2 - vx)
    compare_arrays(
        state_prv["x_velocity"].to_units("m s^-1").data, vx_out, slice=slc
    )

    assert "y_momentum_isentropic" in state_prv
    v = to_numpy(state_dc["y_velocity_at_v_locations"].to_units("m s^-1").data)
    sv = to_numpy(
        state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    sv1 = to_numpy(
        state_prv_dc["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .data
    )
    sv3 = np.zeros_like(sv)
    sv3[:nx, :ny, :nz] = sv[
        :nx, :ny, :nz
    ] + timestep.total_seconds() * 0.5 * s3b[:nx, :ny, :nz] * (
        v[:nx, :ny, :nz] + v[:nx, 1 : ny + 1, :nz]
    )
    sv_out = sv1 + (sv3 - sv)
    compare_arrays(
        state_prv["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data,
        sv_out,
        slice=slc,
    )


if __name__ == "__main__":
    pytest.main([__file__])
    # test_rk2("numpy", float)
