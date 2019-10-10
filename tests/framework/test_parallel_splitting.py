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

from tasmania.python.framework.parallel_splitting import ParallelSplitting
from tasmania.python.isentropic.dynamics.minimal_dycore import (
    IsentropicMinimalDynamicalCore,
)

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import st_domain, st_isentropic_state_f, st_one_of
except (ImportError, ModuleNotFoundError):
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import st_domain, st_isentropic_state_f, st_one_of


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

    state_dc = deepcopy(state)
    state_prv = dycore(state, {}, timestep)
    state_prv_dc = deepcopy(state_prv)

    ps = ParallelSplitting(
        {"component": tendency1, "time_integrator": "forward_euler"},
        {"component": tendency2, "time_integrator": "forward_euler"},
        execution_policy="serial",
    )
    ps(state=state, state_prv=state_prv, timestep=timestep)

    assert "fake_variable" in state
    s = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f = state["fake_variable"].values
    assert np.allclose(f, 2 * s, equal_nan=True)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].values
    s2 = s + timestep.total_seconds() * 0.001 * s
    s3 = s + timestep.total_seconds() * 0.01 * f
    s_out = s1 + (s2 - s) + (s3 - s)
    assert np.allclose(state_prv["air_isentropic_density"].values, s_out, equal_nan=True)

    assert "x_momentum_isentropic" in state_prv
    su = state_dc["x_momentum_isentropic"].values
    su1 = state_prv_dc["x_momentum_isentropic"].values
    su2 = su + timestep.total_seconds() * 300 * su
    su_out = su1 + (su2 - su)
    assert np.allclose(state_prv["x_momentum_isentropic"].values, su_out, equal_nan=True)

    assert "x_velocity_at_u_locations" in state_prv
    u = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2 = u + timestep.total_seconds() * 50 * u
    u_out = u1 + (u2 - u)
    assert np.allclose(
        state_prv["x_velocity_at_u_locations"].values, u_out, equal_nan=True
    )

    assert "y_momentum_isentropic" in state_prv
    v = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3 = sv + timestep.total_seconds() * 0.5 * s * (v[:, :-1, :] + v[:, 1:, :])
    sv_out = sv1 + (sv3 - sv)
    assert np.allclose(state_prv["y_momentum_isentropic"].values, sv_out, equal_nan=True)


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

    state_dc = deepcopy(state)
    state_prv = dycore(state, {}, timestep)
    state_prv_dc = deepcopy(state_prv)

    ps = ParallelSplitting(
        {"component": tendency1, "time_integrator": "rk2"},
        {"component": tendency2, "time_integrator": "rk2"},
        execution_policy="serial",
    )
    ps(state=state, state_prv=state_prv, timestep=timestep)

    assert "fake_variable" in state
    s = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f = state["fake_variable"].values
    assert np.allclose(f, 2 * s, equal_nan=True)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].values
    s2b = s + 0.5 * timestep.total_seconds() * 0.001 * s
    s2 = s + timestep.total_seconds() * 0.001 * s2b
    s3b = s + 0.5 * timestep.total_seconds() * 0.01 * f
    s3 = s + timestep.total_seconds() * 0.01 * f
    s_out = s1 + (s2 - s) + (s3 - s)
    assert np.allclose(state_prv["air_isentropic_density"].values, s_out, equal_nan=True)

    assert "x_momentum_isentropic" in state_prv
    su = state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su1 = state_prv_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su2b = su + 0.5 * timestep.total_seconds() * 300 * su
    su2 = su + timestep.total_seconds() * 300 * su2b
    su_out = su1 + (su2 - su)
    assert np.allclose(state_prv["x_momentum_isentropic"].values, su_out, equal_nan=True)

    assert "x_velocity_at_u_locations" in state_prv
    u = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2b = u + 0.5 * timestep.total_seconds() * 50 * u
    u2 = u + timestep.total_seconds() * 50 * u2b
    u_out = u1 + (u2 - u)
    assert np.allclose(
        state_prv["x_velocity_at_u_locations"].values, u_out, equal_nan=True
    )

    assert "y_momentum_isentropic" in state_prv
    v = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3 = sv + timestep.total_seconds() * 0.5 * s3b * (v[:, :-1, :] + v[:, 1:, :])
    sv_out = sv1 + (sv3 - sv)
    assert np.allclose(state_prv["y_momentum_isentropic"].values, sv_out, equal_nan=True)


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

    state_dc = deepcopy(state)
    state_prv = dycore(state, {}, timestep)
    state_prv_dc = deepcopy(state_prv)

    ps = ParallelSplitting(
        {"component": tendency1, "time_integrator": "forward_euler", "substeps": 3},
        {"component": tendency2, "time_integrator": "rk2", "substeps": 1},
        execution_policy="serial",
    )
    ps(state=state, state_prv=state_prv, timestep=timestep)

    assert "fake_variable" in state
    s = state_dc["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    f = state["fake_variable"].values
    assert np.allclose(f, 2 * s, equal_nan=True)

    assert "air_isentropic_density" in state_prv
    s1 = state_prv_dc["air_isentropic_density"].values
    s2b = s + (timestep / 3.0).total_seconds() * 0.001 * s
    s2c = s2b + (timestep / 3.0).total_seconds() * 0.001 * s2b
    s2 = s2c + (timestep / 3.0).total_seconds() * 0.001 * s2c
    s3b = s + 0.5 * timestep.total_seconds() * 0.01 * f
    s3 = s + timestep.total_seconds() * 0.01 * f
    s_out = s1 + (s2 - s) + (s3 - s)
    assert np.allclose(state_prv["air_isentropic_density"].values, s_out, equal_nan=True)

    assert "x_momentum_isentropic" in state_prv
    su = state_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su1 = state_prv_dc["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    su2b = su + (timestep / 3.0).total_seconds() * 300 * su
    su2c = su2b + (timestep / 3.0).total_seconds() * 300 * su2b
    su2 = su2c + (timestep / 3.0).total_seconds() * 300 * su2c
    su_out = su1 + (su2 - su)
    assert np.allclose(state_prv["x_momentum_isentropic"].values, su_out, equal_nan=True)

    assert "x_velocity_at_u_locations" in state_prv
    u = state_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u1 = state_prv_dc["x_velocity_at_u_locations"].to_units("m s^-1").values
    u2b = u + (timestep / 3.0).total_seconds() * 50 * u
    u2c = u2b + (timestep / 3.0).total_seconds() * 50 * u2b
    u2 = u2c + (timestep / 3.0).total_seconds() * 50 * u2c
    u_out = u1 + (u2 - u)
    assert np.allclose(
        state_prv["x_velocity_at_u_locations"].values, u_out, equal_nan=True
    )

    assert "y_momentum_isentropic" in state_prv
    v = state_dc["y_velocity_at_v_locations"].to_units("m s^-1").values
    sv = state_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv1 = state_prv_dc["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv3 = sv + timestep.total_seconds() * 0.5 * s3b * (v[:, :-1, :] + v[:, 1:, :])
    sv_out = sv1 + (sv3 - sv)
    assert np.allclose(state_prv["y_momentum_isentropic"].values, sv_out, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
