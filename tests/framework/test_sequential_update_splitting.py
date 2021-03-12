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
from tasmania.python.framework.sequential_update_splitting import (
    SequentialUpdateSplitting,
)
from tasmania.python.utils.storage import deepcopy_dataarray_dict
from tasmania.python.utils.backend import is_gt

from tests import conf
from tests.strategies import st_domain, st_isentropic_state_f, st_one_of
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
def test_properties(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2,
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
    sus = SequentialUpdateSplitting(
        TimeIntegrationOptions(tendency2, scheme="forward_euler"),
        TimeIntegrationOptions(tendency1, scheme="forward_euler"),
    )

    assert "air_isentropic_density" in sus.input_properties
    assert "fake_variable" in sus.input_properties
    assert "x_momentum_isentropic" in sus.input_properties
    assert "x_velocity" in sus.input_properties
    assert "x_velocity_at_u_locations" in sus.input_properties
    assert "y_momentum_isentropic" in sus.input_properties
    assert "y_velocity_at_v_locations" in sus.input_properties
    assert len(sus.input_properties) == 7

    assert "air_isentropic_density" in sus.output_properties
    assert "fake_variable" in sus.output_properties
    assert "x_momentum_isentropic" in sus.output_properties
    assert "x_velocity" in sus.output_properties
    assert "x_velocity_at_u_locations" in sus.output_properties
    assert "y_momentum_isentropic" in sus.output_properties
    assert "y_velocity_at_v_locations" in sus.output_properties
    assert len(sus.output_properties) == 7

    #
    # test 2
    #
    sus = SequentialUpdateSplitting(
        TimeIntegrationOptions(tendency1, scheme="forward_euler"),
        TimeIntegrationOptions(tendency2, scheme="forward_euler"),
    )

    assert "air_isentropic_density" in sus.input_properties
    assert "x_momentum_isentropic" in sus.input_properties
    assert "x_velocity" in sus.input_properties
    assert "x_velocity_at_u_locations" in sus.input_properties
    assert "y_momentum_isentropic" in sus.input_properties
    assert "y_velocity_at_v_locations" in sus.input_properties
    assert len(sus.input_properties) == 6

    assert "air_isentropic_density" in sus.output_properties
    assert "fake_variable" in sus.output_properties
    assert "x_momentum_isentropic" in sus.output_properties
    assert "x_velocity" in sus.output_properties
    assert "x_velocity_at_u_locations" in sus.output_properties
    assert "y_momentum_isentropic" in sus.output_properties
    assert "y_velocity_at_v_locations" in sus.output_properties
    assert len(sus.output_properties) == 7

    #
    # test 3
    #
    sus = SequentialUpdateSplitting(
        TimeIntegrationOptions(tendency1, scheme="forward_euler", substeps=2),
        TimeIntegrationOptions(tendency2, scheme="forward_euler", substeps=3),
    )

    assert "air_isentropic_density" in sus.input_properties
    assert "x_momentum_isentropic" in sus.input_properties
    assert "x_velocity" in sus.input_properties
    assert "x_velocity_at_u_locations" in sus.input_properties
    assert "y_momentum_isentropic" in sus.input_properties
    assert "y_velocity_at_v_locations" in sus.input_properties
    assert len(sus.input_properties) == 6

    assert "air_isentropic_density" in sus.output_properties
    assert "fake_variable" in sus.output_properties
    assert "x_momentum_isentropic" in sus.output_properties
    assert "x_velocity" in sus.output_properties
    assert "x_velocity_at_u_locations" in sus.output_properties
    assert "y_momentum_isentropic" in sus.output_properties
    assert "y_velocity_at_v_locations" in sus.output_properties
    assert len(sus.output_properties) == 7


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
        hyp_st.integers(min_value=3, max_value=max(3, conf.nb)), label="nb"
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

    sus = SequentialUpdateSplitting(
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

    state_dc = deepcopy_dataarray_dict(state)

    sus(state, timestep)

    assert "fake_variable" in state
    s1 = to_numpy(
        state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    f2 = to_numpy(state["fake_variable"].data)
    compare_arrays(f2, 2 * s1, slice=slc)

    assert "air_isentropic_density" in state
    s2 = s1 + timestep.total_seconds() * 0.001 * s1
    s3 = s2 + timestep.total_seconds() * 0.01 * f2
    compare_arrays(state["air_isentropic_density"].data, s3, slice=slc)

    assert "x_momentum_isentropic" in state
    su1 = to_numpy(state_dc["x_momentum_isentropic"].data)
    su2 = su1 + timestep.total_seconds() * 300 * su1
    compare_arrays(state["x_momentum_isentropic"].data, su2, slice=slc)

    assert "x_velocity" in state
    vx1 = to_numpy(state_dc["x_velocity"].to_units("m s^-1").data)
    u1 = to_numpy(
        state_dc["x_velocity_at_u_locations"].to_units("m s^-1").data
    )
    vx2 = np.zeros_like(vx1)
    vx2[:nx, :ny, :nz] = vx1[:nx, :ny, :nz] + timestep.total_seconds() * 50 * (
        u1[:nx, :ny, :nz] + u1[1 : nx + 1, :ny, :nz]
    )
    compare_arrays(state["x_velocity"].data, vx2, slice=slc)

    assert "y_momentum_isentropic" in state
    v1 = to_numpy(
        state_dc["y_velocity_at_v_locations"].to_units("m s^-1").data
    )
    sv1 = to_numpy(
        state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    sv3 = np.zeros_like(sv1)
    sv3[:nx, :ny, :nz] = sv1[
        :nx, :ny, :nz
    ] + timestep.total_seconds() * 0.5 * s2[:nx, :ny, :nz] * (
        v1[:nx, :ny, :nz] + v1[:nx, 1 : ny + 1, :nz]
    )
    compare_arrays(state["y_momentum_isentropic"].data, sv3, slice=slc)


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
        hyp_st.integers(min_value=3, max_value=max(3, conf.nb)), label="nb"
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

    sus = SequentialUpdateSplitting(
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

    state_dc = deepcopy_dataarray_dict(state)

    sus(state, timestep)

    assert "air_isentropic_density" in state
    assert "fake_variable" in state
    s1 = to_numpy(
        state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    )
    f2 = to_numpy(state["fake_variable"].data)
    s2b = s1 + 0.5 * timestep.total_seconds() * 0.001 * s1
    s2 = s1 + timestep.total_seconds() * 0.001 * s2b
    compare_arrays(f2, 2 * s2b, slice=slc)
    s3b = s2 + 0.5 * timestep.total_seconds() * 0.01 * f2
    s3 = s2 + timestep.total_seconds() * 0.01 * f2
    compare_arrays(state["air_isentropic_density"].data, s3, slice=slc)

    assert "x_momentum_isentropic" in state
    su1 = to_numpy(state_dc["x_momentum_isentropic"].data)
    su2b = su1 + 0.5 * timestep.total_seconds() * 300 * su1
    su2 = su1 + timestep.total_seconds() * 300 * su2b
    compare_arrays(state["x_momentum_isentropic"].data, su2, slice=slc)

    assert "x_velocity" in state
    vx1 = to_numpy(state_dc["x_velocity"].to_units("m s^-1").data)
    u1 = to_numpy(
        state_dc["x_velocity_at_u_locations"].to_units("m s^-1").data
    )
    vx2 = np.zeros_like(vx1)
    vx2[:nx, :ny, :nz] = vx1[:nx, :ny, :nz] + timestep.total_seconds() * 50 * (
        u1[:nx, :ny, :nz] + u1[1 : nx + 1, :ny, :nz]
    )
    compare_arrays(state["x_velocity"].data, vx2, slice=slc)

    assert "y_momentum_isentropic" in state
    v1 = to_numpy(
        state_dc["y_velocity_at_v_locations"].to_units("m s^-1").data
    )
    sv1 = to_numpy(
        state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    sv3 = np.zeros_like(sv1)
    sv3[:nx, :ny, :nz] = sv1[
        :nx, :ny, :nz
    ] + timestep.total_seconds() * 0.5 * s3b[:nx, :ny, :nz] * (
        v1[:nx, :ny, :nz] + v1[:nx, 1 : ny + 1, :nz]
    )
    compare_arrays(state["y_momentum_isentropic"].data, sv3, slice=slc)


if __name__ == "__main__":
    pytest.main([__file__])
