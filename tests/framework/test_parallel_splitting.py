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

from tasmania.python.framework.options import (
    BackendOptions,
    StorageOptions,
    TimeIntegrationOptions,
)
from tasmania.python.framework.parallel_splitting import ParallelSplitting
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
        TimeIntegrationOptions(component=tendency1, scheme="forward_euler"),
        TimeIntegrationOptions(component=tendency2, scheme="forward_euler"),
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
        TimeIntegrationOptions(component=tendency1, scheme="forward_euler"),
        TimeIntegrationOptions(component=tendency2, scheme="forward_euler"),
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
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    ts_kwargs = {
        "backend": backend,
        "backend_options": BackendOptions(rebuild=False),
        "storage_options": StorageOptions(
            dtype=dtype, default_origin=default_origin
        ),
    }
    ps_kwargs = {
        "backend": backend if data.draw(hyp_st.booleans()) else "numpy",
        "backend_options": BackendOptions(rebuild=False),
        "storage_options": StorageOptions(
            dtype=dtype, default_origin=default_origin
        ),
    }

    same_shape = data.draw(hyp_st.booleans(), label="same_shape")
    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb"
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
    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
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
        TimeIntegrationOptions(tendency1, scheme="forward_euler", **ts_kwargs),
        TimeIntegrationOptions(tendency2, scheme="forward_euler", **ts_kwargs),
        execution_policy="serial",
        **ps_kwargs
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
    sv1 = (
        state_prv_dc["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values
    )
    if same_shape or is_gt(backend):
        sv3 = sv[:, :-1] + timestep.total_seconds() * 0.5 * s[:, :-1] * (
            v[:, :-1] + v[:, 1:]
        )
        sv_out = sv1[:, :-1] + (sv3 - sv[:, :-1])
        compare_arrays(
            state_prv["y_momentum_isentropic"].values[:, :-1], sv_out
        )
    else:
        sv3 = sv + timestep.total_seconds() * 0.5 * s * (v[:, :-1] + v[:, 1:])
        sv_out = sv1 + (sv3 - sv)
        compare_arrays(state_prv["y_momentum_isentropic"].values, sv_out)


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
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    ts_kwargs = {
        "backend": backend,
        "backend_options": BackendOptions(rebuild=False),
        "storage_options": StorageOptions(
            dtype=dtype, default_origin=default_origin
        ),
    }
    ps_kwargs = {
        "backend": backend if data.draw(hyp_st.booleans()) else "numpy",
        "backend_options": BackendOptions(rebuild=False),
        "storage_options": StorageOptions(
            dtype=dtype, default_origin=default_origin
        ),
    }

    same_shape = data.draw(hyp_st.booleans(), label="same_shape") or is_gt(
        backend
    )
    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb"
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
    state_prv = data.draw(
        st_isentropic_state_f(
            grid,
            moist=True,
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
        TimeIntegrationOptions(tendency1, scheme="rk2", **ts_kwargs),
        TimeIntegrationOptions(tendency2, scheme="rk2", **ts_kwargs),
        execution_policy="serial",
        **ps_kwargs
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
    su1 = (
        state_prv_dc["x_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values
    )
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
    sv1 = (
        state_prv_dc["y_momentum_isentropic"]
        .to_units("kg m^-1 K^-1 s^-1")
        .values
    )
    if same_shape or is_gt(backend):
        sv3 = sv[:, :-1] + timestep.total_seconds() * 0.5 * s3b[:, :-1] * (
            v[:, :-1] + v[:, 1:]
        )
        sv_out = sv1[:, :-1] + (sv3 - sv[:, :-1])
        compare_arrays(
            state_prv["y_momentum_isentropic"].values[:, :-1], sv_out
        )
    else:
        sv3 = sv + timestep.total_seconds() * 0.5 * s3b * (
            v[:, :-1] + v[:, 1:]
        )
        sv_out = sv1 + (sv3 - sv)
        compare_arrays(state_prv["y_momentum_isentropic"].values, sv_out)


if __name__ == "__main__":
    pytest.main([__file__])
    # test_rk2("numpy", float)
