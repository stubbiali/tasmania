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
import numpy as np
import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf
import utils

from tasmania.python.framework.sequential_update_splitting import (
    SequentialUpdateSplitting,
)
from tasmania.python.isentropic.dynamics.minimal_dycore import (
    IsentropicMinimalDynamicalCore,
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
def test_properties(data, make_fake_tendency_component_1, make_fake_tendency_component_2):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(utils.st_domain(), label="domain")
    grid_type = data.draw(utils.st_one_of(("physical", "numerical")), label="grid_type")

    # ========================================
    # test bed
    # ========================================
    tendency1 = make_fake_tendency_component_1(domain, grid_type)
    tendency2 = make_fake_tendency_component_2(domain, grid_type)

    #
    # test 1
    #
    sus = SequentialUpdateSplitting(
        {"component": tendency2, "time_integrator": "forward_euler", "substeps": 1},
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 1},
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
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 1},
        {"component": tendency2, "time_integrator": "forward_euler", "substeps": 1},
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
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 3},
        {"component": tendency2, "time_integrator": "forward_euler", "substeps": 2},
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


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_numerics_forward_euler(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    # ========================================
    # random data generation
    # ========================================
    nb = (
        3
    )  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf.nb)))
    domain = data.draw(
        utils.st_domain(
            xaxis_length=(2 * nb + 1, 40), yaxis_length=(2 * nb + 1, 40), nb=nb
        ),
        label="domain",
    )

    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    grid = domain.numerical_grid
    state = data.draw(utils.st_isentropic_state_f(grid, moist=True), label="state")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    backend = data.draw(utils.st_one_of(conf.backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    tendency1 = make_fake_tendency_component_1(domain, "numerical")
    tendency2 = make_fake_tendency_component_2(domain, "numerical")

    dycore = IsentropicMinimalDynamicalCore(
        domain,
        time_integration_scheme="rk3ws",
        horizontal_flux_scheme="fifth_order_upwind",
        moist=False,
        damp=False,
        smooth=False,
        backend=backend,
        dtype=dtype,
    )

    hb.reference_state = state

    state_prv = dycore(state, {}, timestep)
    state_prv_dc = deepcopy(state_prv)

    sus = SequentialUpdateSplitting(
        {"component": tendency1, "time_integrator": "forward_euler"},
        {"component": tendency2, "time_integrator": "forward_euler"},
    )
    sus(state_prv, timestep)

    assert "fake_variable" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f2 = state_prv["fake_variable"].values
    assert np.allclose(f2, 2 * s1, equal_nan=True)

    assert "air_isentropic_density" in state_prv
    s2 = s1 + timestep.total_seconds() * 0.001 * s1
    s3 = s2 + timestep.total_seconds() * 0.01 * f2
    assert np.allclose(state_prv["air_isentropic_density"].values, s3, equal_nan=True)

    assert "x_momentum_isentropic" in state_prv
    su1 = state_prv_dc["x_momentum_isentropic"].values
    su2 = su1 + timestep.total_seconds() * 300 * su1
    assert np.allclose(state_prv["x_momentum_isentropic"].values, su2, equal_nan=True)

    assert "x_velocity_at_u_locations" in state_prv
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2 = u1 + timestep.total_seconds() * 50 * u1
    assert np.allclose(state_prv["x_velocity_at_u_locations"].values, u2, equal_nan=True)

    assert "y_momentum_isentropic" in state_prv
    v1 = state_prv_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3 = sv1 + timestep.total_seconds() * 0.5 * s2 * (v1[:, :-1, :] + v1[:, 1:, :])
    assert np.allclose(state_prv["y_momentum_isentropic"].values, sv3, equal_nan=True)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_numerics_rk2(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    # ========================================
    # random data generation
    # ========================================
    nb = (
        3
    )  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf.nb)))
    domain = data.draw(
        utils.st_domain(
            xaxis_length=(2 * nb + 1, 40), yaxis_length=(2 * nb + 1, 40), nb=nb
        ),
        label="domain",
    )

    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    grid = domain.numerical_grid
    state = data.draw(utils.st_isentropic_state_f(grid, moist=True), label="state")

    timestep = data.draw(
        hyp_st.timedeltas(
            min_value=timedelta(seconds=1e-6), max_value=timedelta(hours=1)
        ),
        label="timestep",
    )

    backend = data.draw(utils.st_one_of(conf.backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    tendency1 = make_fake_tendency_component_1(domain, "numerical")
    tendency2 = make_fake_tendency_component_2(domain, "numerical")

    dycore = IsentropicMinimalDynamicalCore(
        domain,
        time_integration_scheme="rk3ws",
        horizontal_flux_scheme="fifth_order_upwind",
        moist=False,
        damp=False,
        smooth=False,
        backend=backend,
        dtype=dtype,
    )

    hb.reference_state = state

    state_prv = dycore(state, {}, timestep)
    state_prv_dc = deepcopy(state_prv)

    sus = SequentialUpdateSplitting(
        {"component": tendency1, "time_integrator": "rk2"},
        {"component": tendency2, "time_integrator": "rk2"},
    )
    sus(state_prv, timestep)

    assert "fake_variable" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f2 = state_prv["fake_variable"].values
    assert np.allclose(f2, 2 * s1, equal_nan=True)

    assert "air_isentropic_density" in state_prv
    s2b = s1 + 0.5 * timestep.total_seconds() * 0.001 * s1
    s2 = s1 + timestep.total_seconds() * 0.001 * s2b
    s3b = s2 + 0.5 * timestep.total_seconds() * 0.01 * f2
    s3 = s2 + timestep.total_seconds() * 0.01 * f2
    assert np.allclose(state_prv["air_isentropic_density"].values, s3, equal_nan=True)

    assert "x_momentum_isentropic" in state_prv
    su1 = state_prv_dc["x_momentum_isentropic"].values
    su2b = su1 + 0.5 * timestep.total_seconds() * 300 * su1
    su2 = su1 + timestep.total_seconds() * 300 * su2b
    assert np.allclose(state_prv["x_momentum_isentropic"].values, su2, equal_nan=True)

    assert "x_velocity_at_u_locations" in state_prv
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2b = u1 + 0.5 * timestep.total_seconds() * 50 * u1
    u2 = u1 + timestep.total_seconds() * 50 * u2b
    assert np.allclose(state_prv["x_velocity_at_u_locations"].values, u2, equal_nan=True)

    assert "y_momentum_isentropic" in state_prv
    v1 = state_prv_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3 = sv1 + timestep.total_seconds() * 0.5 * s3b * (v1[:, :-1, :] + v1[:, 1:, :])
    assert np.allclose(state_prv["y_momentum_isentropic"].values, sv3, equal_nan=True)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test_numerics_substepping(
    data, make_fake_tendency_component_1, make_fake_tendency_component_2
):
    # ========================================
    # random data generation
    # ========================================
    nb = (
        3
    )  # TODO: nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf.nb)))
    domain = data.draw(
        utils.st_domain(
            xaxis_length=(2 * nb + 1, 40), yaxis_length=(2 * nb + 1, 40), nb=nb
        ),
        label="domain",
    )

    hb = domain.horizontal_boundary
    assume(hb.type != "dirichlet")

    grid = domain.numerical_grid
    state = data.draw(utils.st_isentropic_state_f(grid, moist=True), label="state")

    timestep = data.draw(
        hyp_st.timedeltas(min_value=timedelta(seconds=0), max_value=timedelta(hours=1)),
        label="timestep",
    )

    backend = data.draw(utils.st_one_of(conf.backend), label="backend")

    # ========================================
    # test bed
    # ========================================
    dtype = grid.x.dtype

    tendency1 = make_fake_tendency_component_1(domain, "numerical")
    tendency2 = make_fake_tendency_component_2(domain, "numerical")

    dycore = IsentropicMinimalDynamicalCore(
        domain,
        time_integration_scheme="rk3ws",
        horizontal_flux_scheme="fifth_order_upwind",
        moist=False,
        damp=False,
        smooth=False,
        backend=backend,
        dtype=dtype,
    )

    hb.reference_state = state

    state_prv = dycore(state, {}, timestep)
    state_prv_dc = deepcopy(state_prv)

    sus = SequentialUpdateSplitting(
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 4},
        {"component": tendency2, "time_integrator": "rk2", "substeps": 1},
    )
    sus(state_prv, timestep)

    assert "fake_variable" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f2 = state_prv["fake_variable"].values
    assert np.allclose(f2, 2 * s1, equal_nan=True)

    assert "air_isentropic_density" in state_prv
    s2b = s1 + (timestep / 4.0).total_seconds() * 0.001 * s1
    s2c = s2b + (timestep / 4.0).total_seconds() * 0.001 * s2b
    s2d = s2c + (timestep / 4.0).total_seconds() * 0.001 * s2c
    s2 = s2d + (timestep / 4.0).total_seconds() * 0.001 * s2d
    s3b = s2 + 0.5 * timestep.total_seconds() * 0.01 * f2
    s3 = s2 + timestep.total_seconds() * 0.01 * f2
    assert np.allclose(state_prv["air_isentropic_density"].values, s3, equal_nan=True)

    assert "x_momentum_isentropic" in state_prv
    su1 = state_prv_dc["x_momentum_isentropic"].values
    su2b = su1 + (timestep / 4.0).total_seconds() * 300 * su1
    su2c = su2b + (timestep / 4.0).total_seconds() * 300 * su2b
    su2d = su2c + (timestep / 4.0).total_seconds() * 300 * su2c
    su2 = su2d + (timestep / 4.0).total_seconds() * 300 * su2d
    assert np.allclose(state_prv["x_momentum_isentropic"].values, su2, equal_nan=True)

    assert "x_velocity_at_u_locations" in state_prv
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2b = u1 + (timestep / 4.0).total_seconds() * 50 * u1
    u2c = u2b + (timestep / 4.0).total_seconds() * 50 * u2b
    u2d = u2c + (timestep / 4.0).total_seconds() * 50 * u2c
    u2 = u2d + (timestep / 4.0).total_seconds() * 50 * u2d
    assert np.allclose(state_prv["x_velocity_at_u_locations"].values, u2, equal_nan=True)

    assert "y_momentum_isentropic" in state_prv
    v1 = state_prv_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3 = sv1 + timestep.total_seconds() * 0.5 * s3b * (v1[:, :-1, :] + v1[:, 1:, :])
    assert np.allclose(state_prv["y_momentum_isentropic"].values, sv3, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
