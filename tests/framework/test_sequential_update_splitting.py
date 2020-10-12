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
import pytest

from tasmania.python.framework.sequential_update_splitting import (
    SequentialUpdateSplitting,
)
from tasmania.python.utils.storage_utils import deepcopy_dataarray_dict
from tasmania.python.utils.utils import is_gt

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.strategies import st_domain, st_isentropic_state_f, st_one_of
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(data=hyp_st.data())
def test_properties(
    data,
    make_fake_tendency_component_1,
    make_fake_tendency_component_2,
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
        {
            "component": tendency2,
            "time_integrator": "forward_euler",
            "substeps": 1,
        },
        {
            "component": tendency1,
            "time_integrator": "forward_euler",
            "substeps": 1,
        },
    )

    assert "air_isentropic_density" in sus.input_properties
    assert "fake_variable" in sus.input_properties
    assert "x_momentum_isentropic" in sus.input_properties
    assert "x_velocity_at_u_locations" in sus.input_properties
    assert "y_momentum_isentropic" in sus.input_properties
    assert "y_velocity_at_v_locations" in sus.input_properties
    assert len(sus.input_properties) == 6

    assert "air_isentropic_density" in sus.output_properties
    assert "fake_variable" in sus.output_properties
    assert "x_momentum_isentropic" in sus.output_properties
    assert "x_velocity_at_u_locations" in sus.output_properties
    assert "y_momentum_isentropic" in sus.output_properties
    assert "y_velocity_at_v_locations" in sus.output_properties
    assert len(sus.output_properties) == 6

    #
    # test 2
    #
    sus = SequentialUpdateSplitting(
        {
            "component": tendency1,
            "time_integrator": "forward_euler",
            "substeps": 1,
        },
        {
            "component": tendency2,
            "time_integrator": "forward_euler",
            "substeps": 1,
        },
    )

    assert "air_isentropic_density" in sus.input_properties
    assert "x_momentum_isentropic" in sus.input_properties
    assert "x_velocity_at_u_locations" in sus.input_properties
    assert "y_momentum_isentropic" in sus.input_properties
    assert "y_velocity_at_v_locations" in sus.input_properties
    assert len(sus.input_properties) == 5

    assert "air_isentropic_density" in sus.output_properties
    assert "fake_variable" in sus.output_properties
    assert "x_momentum_isentropic" in sus.output_properties
    assert "x_velocity_at_u_locations" in sus.output_properties
    assert "y_momentum_isentropic" in sus.output_properties
    assert "y_velocity_at_v_locations" in sus.output_properties
    assert len(sus.output_properties) == 6

    #
    # test 3
    #
    sus = SequentialUpdateSplitting(
        {
            "component": tendency1,
            "time_integrator": "forward_euler",
            "substeps": 3,
        },
        {
            "component": tendency2,
            "time_integrator": "forward_euler",
            "substeps": 2,
        },
    )

    assert "air_isentropic_density" in sus.input_properties
    assert "x_momentum_isentropic" in sus.input_properties
    assert "x_velocity_at_u_locations" in sus.input_properties
    assert "y_momentum_isentropic" in sus.input_properties
    assert "y_velocity_at_v_locations" in sus.input_properties
    assert len(sus.input_properties) == 5

    assert "air_isentropic_density" in sus.output_properties
    assert "fake_variable" in sus.output_properties
    assert "x_momentum_isentropic" in sus.output_properties
    assert "x_velocity_at_u_locations" in sus.output_properties
    assert "y_momentum_isentropic" in sus.output_properties
    assert "y_velocity_at_v_locations" in sus.output_properties
    assert len(sus.output_properties) == 6


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
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
    backend_ts1 = (
        backend
        if data.draw(hyp_st.booleans(), label="same_backend_1")
        else "numpy"
    )
    backend_ts2 = (
        backend
        if data.draw(hyp_st.booleans(), label="same_backend_2")
        else "numpy"
    )
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
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
    tendency1 = make_fake_tendency_component_1(domain, "numerical")
    tendency2 = make_fake_tendency_component_2(domain, "numerical")

    sus = SequentialUpdateSplitting(
        {
            "component": tendency1,
            "time_integrator": "forward_euler",
            "time_integrator_kwargs": {
                "backend": backend_ts1,
                "dtype": dtype,
                "default_origin": default_origin,
            },
        },
        {
            "component": tendency2,
            "time_integrator": "forward_euler",
            "time_integrator_kwargs": {
                "backend": backend_ts2,
                "dtype": dtype,
                "default_origin": default_origin,
            },
        },
    )

    state_dc = deepcopy_dataarray_dict(state)

    sus(state, timestep)

    assert "fake_variable" in state
    s1 = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    f2 = state["fake_variable"].data
    compare_arrays(f2, 2 * s1)

    assert "air_isentropic_density" in state
    s2 = s1 + timestep.total_seconds() * 0.001 * s1
    s3 = s2 + timestep.total_seconds() * 0.01 * f2
    compare_arrays(state["air_isentropic_density"].data, s3)

    assert "x_momentum_isentropic" in state
    su1 = state_dc["x_momentum_isentropic"].data
    su2 = su1 + timestep.total_seconds() * 300 * su1
    compare_arrays(state["x_momentum_isentropic"].data, su2)

    assert "x_velocity_at_u_locations" in state
    u1 = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").data
    u2 = u1 + timestep.total_seconds() * 50 * u1
    compare_arrays(state["x_velocity_at_u_locations"].data, u2)

    assert "y_momentum_isentropic" in state
    v1 = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").data
    sv1 = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    if same_shape or is_gt(backend):
        sv3 = sv1[:, :-1] + timestep.total_seconds() * 0.5 * s2[:, :-1] * (
            v1[:, :-1] + v1[:, 1:]
        )
        compare_arrays(state["y_momentum_isentropic"].data[:, :-1], sv3)
    else:
        sv3 = sv1 + timestep.total_seconds() * 0.5 * s2 * (
            v1[:, :-1] + v1[:, 1:]
        )
        compare_arrays(state["y_momentum_isentropic"].data, sv3)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
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
    backend_ts1 = (
        backend
        if data.draw(hyp_st.booleans(), label="same_backend_1")
        else "numpy"
    )
    backend_ts2 = (
        backend
        if data.draw(hyp_st.booleans(), label="same_backend_2")
        else "numpy"
    )
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
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
    tendency1 = make_fake_tendency_component_1(domain, "numerical")
    tendency2 = make_fake_tendency_component_2(domain, "numerical")

    sus = SequentialUpdateSplitting(
        {
            "component": tendency1,
            "time_integrator": "rk2",
            "time_integrator_kwargs": {
                "backend": backend_ts1,
                "dtype": dtype,
                "default_origin": default_origin,
            },
        },
        {
            "component": tendency2,
            "time_integrator": "rk2",
            "time_integrator_kwargs": {
                "backend": backend_ts2,
                "dtype": dtype,
                "default_origin": default_origin,
            },
        },
    )

    state_dc = deepcopy_dataarray_dict(state)

    sus(state, timestep)

    assert "fake_variable" in state
    s1 = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").data
    f2 = state["fake_variable"].data
    compare_arrays(f2, 2 * s1)

    assert "air_isentropic_density" in state
    s2b = s1 + 0.5 * timestep.total_seconds() * 0.001 * s1
    s2 = s1 + timestep.total_seconds() * 0.001 * s2b
    s3b = s2 + 0.5 * timestep.total_seconds() * 0.01 * f2
    s3 = s2 + timestep.total_seconds() * 0.01 * f2
    compare_arrays(state["air_isentropic_density"].data, s3)

    assert "x_momentum_isentropic" in state
    su1 = state_dc["x_momentum_isentropic"].data
    su2b = su1 + 0.5 * timestep.total_seconds() * 300 * su1
    su2 = su1 + timestep.total_seconds() * 300 * su2b
    compare_arrays(state["x_momentum_isentropic"].data, su2)

    assert "x_velocity_at_u_locations" in state
    u1 = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").data
    u2b = u1 + 0.5 * timestep.total_seconds() * 50 * u1
    u2 = u1 + timestep.total_seconds() * 50 * u2b
    compare_arrays(state["x_velocity_at_u_locations"].data, u2)

    assert "y_momentum_isentropic" in state
    v1 = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").data
    sv1 = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    if same_shape or is_gt(backend):
        sv3 = sv1[:, :-1] + timestep.total_seconds() * 0.5 * s3b[:, :-1] * (
            v1[:, :-1] + v1[:, 1:]
        )
        compare_arrays(state["y_momentum_isentropic"].data[:, :-1], sv3)
    else:
        sv3 = sv1 + timestep.total_seconds() * 0.5 * s3b * (
            v1[:, :-1] + v1[:, 1:]
        )
        compare_arrays(state["y_momentum_isentropic"].data, sv3)


if __name__ == "__main__":
    pytest.main([__file__])
