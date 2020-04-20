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
import pytest

from tasmania.python.framework.parallel_splitting import ParallelSplitting
from tasmania.python.utils.storage_utils import deepcopy_dataarray_dict

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.strategies import st_domain, st_isentropic_state_f, st_one_of
from tests.utilities import compare_arrays


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_properties(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(gt_powered=False), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")

    # ========================================
    # test bed
    # ========================================
    tendency1 = make_fake_tendency_component_1(domain, grid_type)
    tendency2 = make_fake_tendency_component_2(domain, grid_type)

    #
    # test 1
    #
    ps = ParallelSplitting(
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 1},
        {"component": tendency2, "time_integrator": "forward_euler"},
        execution_policy="as_parallel",
    )

    assert "air_isentropic_density" in ps.input_properties
    assert "fake_variable" in ps.input_properties
    assert "x_momentum_isentropic" in ps.input_properties
    assert "x_velocity_at_u_locations" in ps.input_properties
    assert "y_momentum_isentropic" in ps.input_properties
    assert "y_velocity_at_v_locations" in ps.input_properties
    assert len(ps.input_properties) == 6

    assert "air_isentropic_density" in ps.provisional_input_properties
    assert "x_momentum_isentropic" in ps.provisional_input_properties
    assert "x_velocity_at_u_locations" in ps.provisional_input_properties
    assert "y_momentum_isentropic" in ps.provisional_input_properties
    assert len(ps.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.output_properties
    assert "fake_variable" in ps.output_properties
    assert "x_momentum_isentropic" in ps.output_properties
    assert "x_velocity_at_u_locations" in ps.output_properties
    assert "y_momentum_isentropic" in ps.output_properties
    assert "y_velocity_at_v_locations" in ps.output_properties
    assert len(ps.output_properties) == 6

    assert "air_isentropic_density" in ps.provisional_output_properties
    assert "x_momentum_isentropic" in ps.provisional_output_properties
    assert "x_velocity_at_u_locations" in ps.provisional_output_properties
    assert "y_momentum_isentropic" in ps.provisional_output_properties
    assert len(ps.provisional_output_properties) == 4

    #
    # test 2
    #
    ps = ParallelSplitting(
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 1},
        {"component": tendency2, "time_integrator": "forward_euler"},
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=False,
    )

    assert "air_isentropic_density" in ps.input_properties
    assert "x_momentum_isentropic" in ps.input_properties
    assert "x_velocity_at_u_locations" in ps.input_properties
    assert "y_momentum_isentropic" in ps.input_properties
    assert "y_velocity_at_v_locations" in ps.input_properties
    assert len(ps.input_properties) == 5

    assert "air_isentropic_density" in ps.provisional_input_properties
    assert "x_momentum_isentropic" in ps.provisional_input_properties
    assert "x_velocity_at_u_locations" in ps.provisional_input_properties
    assert "y_momentum_isentropic" in ps.provisional_input_properties
    assert len(ps.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.output_properties
    assert "fake_variable" in ps.output_properties
    assert "x_momentum_isentropic" in ps.output_properties
    assert "x_velocity_at_u_locations" in ps.output_properties
    assert "y_momentum_isentropic" in ps.output_properties
    assert "y_velocity_at_v_locations" in ps.output_properties
    assert len(ps.output_properties) == 6

    assert "air_isentropic_density" in ps.provisional_output_properties
    assert "x_momentum_isentropic" in ps.provisional_output_properties
    assert "x_velocity_at_u_locations" in ps.provisional_output_properties
    assert "y_momentum_isentropic" in ps.provisional_output_properties
    assert len(ps.provisional_output_properties) == 4

    #
    # test 3
    #
    ps = ParallelSplitting(
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 1},
        {"component": tendency2, "time_integrator": "forward_euler"},
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=True,
    )

    assert "air_isentropic_density" in ps.input_properties
    assert "x_momentum_isentropic" in ps.input_properties
    assert "x_velocity_at_u_locations" in ps.input_properties
    assert "y_momentum_isentropic" in ps.input_properties
    assert "y_velocity_at_v_locations" in ps.input_properties
    assert len(ps.input_properties) == 5

    assert "air_isentropic_density" in ps.provisional_input_properties
    assert "x_momentum_isentropic" in ps.provisional_input_properties
    assert "x_velocity_at_u_locations" in ps.provisional_input_properties
    assert "y_momentum_isentropic" in ps.provisional_input_properties
    assert len(ps.provisional_input_properties) == 4

    assert "air_isentropic_density" in ps.output_properties
    assert "fake_variable" in ps.output_properties
    assert "x_momentum_isentropic" in ps.output_properties
    assert "x_velocity_at_u_locations" in ps.output_properties
    assert "y_momentum_isentropic" in ps.output_properties
    assert "y_velocity_at_v_locations" in ps.output_properties
    assert len(ps.output_properties) == 6

    assert "air_isentropic_density" in ps.provisional_output_properties
    assert "x_momentum_isentropic" in ps.provisional_output_properties
    assert "x_velocity_at_u_locations" in ps.provisional_output_properties
    assert "y_momentum_isentropic" in ps.provisional_output_properties
    assert len(ps.provisional_output_properties) == 4


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_forward_euler(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    gt_powered_ts1 = gt_powered and data.draw(hyp_st.booleans(), label="gt_powered_ts1")
    gt_powered_ts2 = gt_powered and data.draw(hyp_st.booleans(), label="gt_powered_ts2")
    gt_powered_ps = gt_powered and data.draw(hyp_st.booleans(), label="gt_powered_ps")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    gt_kwargs = {"backend": backend, "dtype": dtype, "default_origin": default_origin}
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=nb,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="state",
    )
    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
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
    tendency1 = make_fake_tendency_component_1(domain, "numerical")
    tendency2 = make_fake_tendency_component_2(domain, "numerical")

    ps = ParallelSplitting(
        {
            "component": tendency1,
            "time_integrator": "forward_euler",
            "gt_powered": gt_powered_ts1,
            "time_integrator_kwargs": gt_kwargs,
        },
        {
            "component": tendency2,
            "time_integrator": "forward_euler",
            "gt_powered": gt_powered_ts2,
            "time_integrator_kwargs": gt_kwargs,
        },
        execution_policy="serial",
        gt_powered=gt_powered_ps,
        **gt_kwargs
    )

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    ps(state=state, state_prv=state_prv, timestep=timestep)

    assert "fake_variable" in state
    s = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f = state["fake_variable"].values
    compare_arrays(f, 2 * s)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].values
    s2 = s + timestep.total_seconds() * 0.001 * s
    s3 = s + timestep.total_seconds() * 0.01 * f
    s_out = s1 + (s2 - s) + (s3 - s)
    compare_arrays(state_prv["air_isentropic_density"].values, s_out)

    assert "x_momentum_isentropic" in state_prv
    su = state_dc["x_momentum_isentropic"].values
    su1 = state_prv_dc["x_momentum_isentropic"].values
    su2 = su + timestep.total_seconds() * 300 * su
    su_out = su1 + (su2 - su)
    compare_arrays(state_prv["x_momentum_isentropic"].values, su_out)

    assert "x_velocity_at_u_locations" in state_prv
    u = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2 = u + timestep.total_seconds() * 50 * u
    u_out = u1 + (u2 - u)
    compare_arrays(state_prv["x_velocity_at_u_locations"].values, u_out)

    assert "y_momentum_isentropic" in state_prv
    v = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    if same_shape:
        sv3 = sv[:, :-1] + timestep.total_seconds() * 0.5 * s[:, :-1] * (
            v[:, :-1] + v[:, 1:]
        )
        sv_out = sv1[:, :-1] + (sv3 - sv[:, :-1])
        compare_arrays(state_prv["y_momentum_isentropic"].values[:, :-1], sv_out)
    else:
        sv3 = sv + timestep.total_seconds() * 0.5 * s * (v[:, :-1] + v[:, 1:])
        sv_out = sv1 + (sv3 - sv)
        compare_arrays(state_prv["y_momentum_isentropic"].values, sv_out)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_rk2(data, make_fake_tendency_component_1, make_fake_tendency_component_2):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    gt_powered_ts1 = gt_powered and data.draw(hyp_st.booleans(), label="gt_powered_ts1")
    gt_powered_ts2 = gt_powered and data.draw(hyp_st.booleans(), label="gt_powered_ts2")
    gt_powered_ps = gt_powered and data.draw(hyp_st.booleans(), label="gt_powered_ps")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    gt_kwargs = {"backend": backend, "dtype": dtype, "default_origin": default_origin}
    same_shape = data.draw(hyp_st.booleans(), label="same_shape")

    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb")
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=nb,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid = domain.numerical_grid
    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
        ),
        label="state",
    )
    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            gt_powered=gt_powered,
            backend=backend,
            default_origin=default_origin,
            storage_shape=storage_shape if same_shape else None,
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
    tendency1 = make_fake_tendency_component_1(domain, "numerical")
    tendency2 = make_fake_tendency_component_2(domain, "numerical")

    ps = ParallelSplitting(
        {
            "component": tendency1,
            "time_integrator": "rk2",
            "gt_powered": gt_powered_ts1,
            "time_integrator_kwargs": gt_kwargs,
        },
        {
            "component": tendency2,
            "time_integrator": "rk2",
            "gt_powered": gt_powered_ts2,
            "time_integrator_kwargs": gt_kwargs,
        },
        execution_policy="serial",
        gt_powered=gt_powered_ps,
        **gt_kwargs
    )

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    ps(state=state, state_prv=state_prv, timestep=timestep)

    assert "fake_variable" in state
    s = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f = state["fake_variable"].values
    compare_arrays(f, 2 * s)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].values
    s2b = s + 0.5 * timestep.total_seconds() * 0.001 * s
    s2 = s + timestep.total_seconds() * 0.001 * s2b
    s3b = s + 0.5 * timestep.total_seconds() * 0.01 * f
    s3 = s + timestep.total_seconds() * 0.01 * f
    s_out = s1 + (s2 - s) + (s3 - s)
    compare_arrays(state_prv["air_isentropic_density"].values, s_out)

    assert "x_momentum_isentropic" in state_prv
    su = state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su1 = state_prv_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su2b = su + 0.5 * timestep.total_seconds() * 300 * su
    su2 = su + timestep.total_seconds() * 300 * su2b
    su_out = su1 + (su2 - su)
    compare_arrays(state_prv["x_momentum_isentropic"].values, su_out)

    assert "x_velocity_at_u_locations" in state_prv
    u = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2b = u + 0.5 * timestep.total_seconds() * 50 * u
    u2 = u + timestep.total_seconds() * 50 * u2b
    u_out = u1 + (u2 - u)
    compare_arrays(state_prv["x_velocity_at_u_locations"].values, u_out)

    assert "y_momentum_isentropic" in state_prv
    v = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    if same_shape:
        sv3 = sv[:, :-1] + timestep.total_seconds() * 0.5 * s3b[:, :-1] * (
            v[:, :-1] + v[:, 1:]
        )
        sv_out = sv1[:, :-1] + (sv3 - sv[:, :-1])
        compare_arrays(state_prv["y_momentum_isentropic"].values[:, :-1], sv_out)
    else:
        sv3 = sv + timestep.total_seconds() * 0.5 * s3b * (v[:, :-1] + v[:, 1:])
        sv_out = sv1 + (sv3 - sv)
        compare_arrays(state_prv["y_momentum_isentropic"].values, sv_out)


if __name__ == "__main__":
    pytest.main([__file__])
