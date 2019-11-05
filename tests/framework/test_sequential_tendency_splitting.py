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
import numpy as np
import pytest

import gridtools as gt

from tasmania.python.framework.sequential_tendency_splitting import (
    SequentialTendencySplitting,
)
from tasmania.python.utils.storage_utils import deepcopy_dataarray_dict

try:
    from .conf import (
        backend as conf_backend,
        default_origin as conf_dorigin,
        nb as conf_nb,
    )
    from .utils import compare_arrays, st_domain, st_isentropic_state_f, st_one_of
except (ImportError, ModuleNotFoundError):
    from conf import (
        backend as conf_backend,
        default_origin as conf_dorigin,
        nb as conf_nb,
    )
    from utils import compare_arrays, st_domain, st_isentropic_state_f, st_one_of


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_properties(data, make_fake_tendency_component_1, make_fake_tendency_component_2):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")

    # ========================================
    # test bed
    # ========================================
    tendency1 = make_fake_tendency_component_1(domain, grid_type)
    tendency2 = make_fake_tendency_component_2(domain, grid_type)

    #
    # test 1
    #
    sts = SequentialTendencySplitting(
        {"component": tendency2, "time_integrator": "forward_euler", "substeps": 1},
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 1},
    )

    assert "air_isentropic_density" in sts.input_properties
    assert "fake_variable" in sts.input_properties
    assert "x_momentum_isentropic" in sts.input_properties
    assert "x_velocity_at_u_locations" in sts.input_properties
    assert "y_momentum_isentropic" in sts.input_properties
    assert "y_velocity_at_v_locations" in sts.input_properties
    assert len(sts.input_properties) == 6

    assert "air_isentropic_density" in sts.provisional_input_properties
    assert "x_momentum_isentropic" in sts.provisional_input_properties
    assert "x_velocity_at_u_locations" in sts.provisional_input_properties
    assert "y_momentum_isentropic" in sts.provisional_input_properties
    assert len(sts.provisional_input_properties) == 4

    assert "air_isentropic_density" in sts.output_properties
    assert "fake_variable" in sts.output_properties
    assert "x_momentum_isentropic" in sts.output_properties
    assert "x_velocity_at_u_locations" in sts.output_properties
    assert "y_momentum_isentropic" in sts.output_properties
    assert "y_velocity_at_v_locations" in sts.output_properties
    assert len(sts.output_properties) == 6

    assert "air_isentropic_density" in sts.provisional_output_properties
    assert "x_momentum_isentropic" in sts.provisional_output_properties
    assert "x_velocity_at_u_locations" in sts.provisional_output_properties
    assert "y_momentum_isentropic" in sts.provisional_output_properties
    assert len(sts.provisional_output_properties) == 4

    #
    # test 2
    #
    sts = SequentialTendencySplitting(
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 1},
        {"component": tendency2, "time_integrator": "forward_euler", "substeps": 1},
    )

    assert "air_isentropic_density" in sts.input_properties
    assert "x_momentum_isentropic" in sts.input_properties
    assert "x_velocity_at_u_locations" in sts.input_properties
    assert "y_momentum_isentropic" in sts.input_properties
    assert "y_velocity_at_v_locations" in sts.input_properties
    assert len(sts.input_properties) == 5

    assert "air_isentropic_density" in sts.provisional_input_properties
    assert "x_momentum_isentropic" in sts.provisional_input_properties
    assert "x_velocity_at_u_locations" in sts.provisional_input_properties
    assert "y_momentum_isentropic" in sts.provisional_input_properties
    assert len(sts.provisional_input_properties) == 4

    assert "air_isentropic_density" in sts.output_properties
    assert "fake_variable" in sts.output_properties
    assert "x_momentum_isentropic" in sts.output_properties
    assert "x_velocity_at_u_locations" in sts.output_properties
    assert "y_momentum_isentropic" in sts.output_properties
    assert "y_velocity_at_v_locations" in sts.output_properties
    assert len(sts.output_properties) == 6

    assert "air_isentropic_density" in sts.provisional_output_properties
    assert "x_momentum_isentropic" in sts.provisional_output_properties
    assert "x_velocity_at_u_locations" in sts.provisional_output_properties
    assert "y_momentum_isentropic" in sts.provisional_output_properties
    assert len(sts.provisional_output_properties) == 4

    #
    # test 3
    #
    sts = SequentialTendencySplitting(
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 3},
        {"component": tendency2, "time_integrator": "forward_euler", "substeps": 2},
    )

    assert "air_isentropic_density" in sts.input_properties
    assert "x_momentum_isentropic" in sts.input_properties
    assert "x_velocity_at_u_locations" in sts.input_properties
    assert "y_momentum_isentropic" in sts.input_properties
    assert "y_velocity_at_v_locations" in sts.input_properties
    assert len(sts.input_properties) == 5

    assert "air_isentropic_density" in sts.provisional_input_properties
    assert "x_momentum_isentropic" in sts.provisional_input_properties
    assert "x_velocity_at_u_locations" in sts.provisional_input_properties
    assert "y_momentum_isentropic" in sts.provisional_input_properties
    assert len(sts.provisional_input_properties) == 4

    assert "air_isentropic_density" in sts.output_properties
    assert "fake_variable" in sts.output_properties
    assert "x_momentum_isentropic" in sts.output_properties
    assert "x_velocity_at_u_locations" in sts.output_properties
    assert "y_momentum_isentropic" in sts.output_properties
    assert "y_velocity_at_v_locations" in sts.output_properties
    assert len(sts.output_properties) == 6

    assert "air_isentropic_density" in sts.provisional_output_properties
    assert "x_momentum_isentropic" in sts.provisional_output_properties
    assert "x_velocity_at_u_locations" in sts.provisional_output_properties
    assert "y_momentum_isentropic" in sts.provisional_output_properties
    assert len(sts.provisional_output_properties) == 4


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
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)))
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 20), nb=nb
        ),
        label="domain",
    )

    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    gt_kwargs = {"backend": backend, "default_origin": default_origin}

    grid = domain.numerical_grid
    state = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, default_origin=default_origin
        ),
        label="state",
    )
    state_prv = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, default_origin=default_origin
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

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    sts = SequentialTendencySplitting(
        {
            "component": tendency1,
            "time_integrator": "forward_euler",
            "time_integrator_kwargs": gt_kwargs,
        },
        {
            "component": tendency2,
            "time_integrator": "forward_euler",
            "time_integrator_kwargs": gt_kwargs,
        },
    )
    sts(state, state_prv, timestep)

    assert "fake_variable" in state
    s0 = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f1 = state["fake_variable"].values
    compare_arrays(f1, 2 * s0)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    s2 = s1 + timestep.total_seconds() * 0.001 * s0
    s3 = s2 + timestep.total_seconds() * 0.01 * f1
    compare_arrays(state_prv["air_isentropic_density"].values, s3)

    assert "x_momentum_isentropic" in state_prv
    su0 = state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su1 = state_prv_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su2 = su1 + timestep.total_seconds() * 300 * su0
    compare_arrays(
        state_prv["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values, su2
    )

    assert "x_velocity_at_u_locations" in state_prv
    u0 = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2 = u1 + timestep.total_seconds() * 50 * u0
    compare_arrays(state_prv["x_velocity_at_u_locations"].to_units("m s^-1").values, u2)

    assert "y_momentum_isentropic" in state_prv
    v0 = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3 = sv1 + timestep.total_seconds() * 0.5 * s0 * (v0[:, :-1, :] + v0[:, 1:, :])
    compare_arrays(state_prv["y_momentum_isentropic"].values, sv3)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_gt_forward_euler(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)))
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 20), yaxis_length=(1, 20), zaxis_length=(1, 20), nb=nb
        ),
        label="domain",
    )

    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    gt_kwargs = {"backend": backend, "default_origin": default_origin}

    grid = domain.numerical_grid
    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(grid.nx + 1, grid.ny + 1, grid.nz + 1),
        ),
        label="state",
    )
    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(grid.nx + 1, grid.ny + 1, grid.nz + 1),
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

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    sts = SequentialTendencySplitting(
        {
            "component": tendency1,
            "time_integrator": "forward_euler",
            "time_integrator_kwargs": gt_kwargs,
        },
        {
            "component": tendency2,
            "time_integrator": "forward_euler",
            "time_integrator_kwargs": gt_kwargs,
        },
    )
    sts(state, state_prv, timestep)

    assert "fake_variable" in state
    s0 = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f1 = state["fake_variable"].values
    compare_arrays(f1, 2 * s0)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    s2 = s1 + timestep.total_seconds() * 0.001 * s0
    s3 = s2 + timestep.total_seconds() * 0.01 * f1
    compare_arrays(state_prv["air_isentropic_density"].values, s3)

    assert "x_momentum_isentropic" in state_prv
    su0 = state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su1 = state_prv_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su2 = su1 + timestep.total_seconds() * 300 * su0
    compare_arrays(
        state_prv["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values, su2
    )

    assert "x_velocity_at_u_locations" in state_prv
    u0 = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2 = u1 + timestep.total_seconds() * 50 * u0
    compare_arrays(state_prv["x_velocity_at_u_locations"].to_units("m s^-1").values, u2)

    assert "y_momentum_isentropic" in state_prv
    v0 = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3 = np.zeros((grid.nx + 1, grid.ny + 1, grid.nz + 1), dtype=sv1.dtype)
    sv3[:, :-1] = sv1[:, :-1] + timestep.total_seconds() * 0.5 * s0[:, :-1] * (
        v0[:, :-1] + v0[:, 1:]
    )
    compare_arrays(state_prv["y_momentum_isentropic"].values, sv3)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_rk2(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)))
    domain = data.draw(
        st_domain(xaxis_length=(1, 20), yaxis_length=(1, 20), nb=nb), label="domain"
    )

    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    gt_kwargs = {"backend": backend, "default_origin": default_origin}

    grid = domain.numerical_grid
    state = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, default_origin=default_origin
        ),
        label="state",
    )
    state_prv = data.draw(
        st_isentropic_state_f(
            grid, moist=True, backend=backend, default_origin=default_origin
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

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    sts = SequentialTendencySplitting(
        {
            "component": tendency1,
            "time_integrator": "rk2",
            "time_integrator_kwargs": gt_kwargs,
        },
        {
            "component": tendency2,
            "time_integrator": "rk2",
            "time_integrator_kwargs": gt_kwargs,
        },
    )

    sts(state, state_prv, timestep)

    assert "fake_variable" in state
    s0 = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f1 = state["fake_variable"].to_units("kg m^-2 K^-1").values
    compare_arrays(f1, 2 * s0)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    s2b = 0.5 * (s0 + s1 + timestep.total_seconds() * 0.001 * s0)
    s2 = s1 + timestep.total_seconds() * 0.001 * s2b
    s3b = 0.5 * (s0 + s2 + timestep.total_seconds() * 0.01 * f1)
    s3 = s2 + timestep.total_seconds() * 0.01 * f1
    compare_arrays(
        state_prv["air_isentropic_density"].to_units("kg m^-2 K^-1").values, s3
    )

    assert "x_momentum_isentropic" in state_prv
    su0 = state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su1 = state_prv_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su2b = 0.5 * (su0 + su1 + timestep.total_seconds() * 300 * su0)
    su2 = su1 + timestep.total_seconds() * 300 * su2b
    compare_arrays(
        state_prv["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values, su2
    )

    assert "x_velocity_at_u_locations" in state_prv
    u0 = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2b = 0.5 * (u0 + u1 + timestep.total_seconds() * 50 * u0)
    u2 = u1 + timestep.total_seconds() * 50 * u2b
    compare_arrays(state_prv["x_velocity_at_u_locations"].to_units("m s^-1").values, u2)

    assert "y_momentum_isentropic" in state_prv
    v0 = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv0 = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3b = 0.5 * (
        sv0 + sv1 + timestep.total_seconds() * 0.5 * s0 * (v0[:, :-1] + v0[:, 1:])
    )
    sv3 = sv1 + timestep.total_seconds() * 0.5 * s3b * (v0[:, :-1] + v0[:, 1:])
    compare_arrays(state_prv["y_momentum_isentropic"].values, sv3)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_gt_rk2(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)))
    domain = data.draw(
        st_domain(xaxis_length=(1, 20), yaxis_length=(1, 20), nb=nb), label="domain"
    )

    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    backend = data.draw(st_one_of(conf_backend), label="backend")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")
    gt_kwargs = {"backend": backend, "default_origin": default_origin}

    grid = domain.numerical_grid
    state = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(grid.nx + 1, grid.ny + 1, grid.nz + 1),
        ),
        label="state",
    )
    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
            backend=backend,
            default_origin=default_origin,
            storage_shape=(grid.nx + 1, grid.ny + 1, grid.nz + 1),
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

    state_dc = deepcopy_dataarray_dict(state)
    state_prv_dc = deepcopy_dataarray_dict(state_prv)

    sts = SequentialTendencySplitting(
        {
            "component": tendency1,
            "time_integrator": "rk2",
            "time_integrator_kwargs": gt_kwargs,
        },
        {
            "component": tendency2,
            "time_integrator": "rk2",
            "time_integrator_kwargs": gt_kwargs,
        },
    )

    sts(state, state_prv, timestep)

    assert "fake_variable" in state
    s0 = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f1 = state["fake_variable"].to_units("kg m^-2 K^-1").values
    compare_arrays(f1, 2 * s0)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    s2b = 0.5 * (s0 + s1 + timestep.total_seconds() * 0.001 * s0)
    s2 = s1 + timestep.total_seconds() * 0.001 * s2b
    s3b = 0.5 * (s0 + s2 + timestep.total_seconds() * 0.01 * f1)
    s3 = s2 + timestep.total_seconds() * 0.01 * f1
    compare_arrays(
        state_prv["air_isentropic_density"].to_units("kg m^-2 K^-1").values, s3
    )

    assert "x_momentum_isentropic" in state_prv
    su0 = state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su1 = state_prv_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su2b = 0.5 * (su0 + su1 + timestep.total_seconds() * 300 * su0)
    su2 = su1 + timestep.total_seconds() * 300 * su2b
    compare_arrays(
        state_prv["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values, su2
    )

    assert "x_velocity_at_u_locations" in state_prv
    u0 = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2b = 0.5 * (u0 + u1 + timestep.total_seconds() * 50 * u0)
    u2 = u1 + timestep.total_seconds() * 50 * u2b
    compare_arrays(state_prv["x_velocity_at_u_locations"].to_units("m s^-1").values, u2)

    assert "y_momentum_isentropic" in state_prv
    v0 = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv0 = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3b = np.zeros((grid.nx + 1, grid.ny + 1, grid.nz + 1), dtype=sv1.dtype)
    sv3b[:, :-1] = 0.5 * (
        sv0[:, :-1]
        + sv1[:, :-1]
        + timestep.total_seconds() * 0.5 * s0[:, :-1] * (v0[:, :-1] + v0[:, 1:])
    )
    sv3 = np.zeros((grid.nx + 1, grid.ny + 1, grid.nz + 1), dtype=sv1.dtype)
    sv3[:, :-1] = sv1[:, :-1] + timestep.total_seconds() * 0.5 * s3b[:, :-1] * (
        v0[:, :-1] + v0[:, 1:]
    )
    compare_arrays(state_prv["y_momentum_isentropic"].values, sv3)


if __name__ == "__main__":
    pytest.main([__file__])
