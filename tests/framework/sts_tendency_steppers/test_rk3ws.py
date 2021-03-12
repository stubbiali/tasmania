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

from tasmania.python.framework.allocators import as_storage
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper
from tasmania.python.utils.storage import get_dataarray_dict

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
    nx, ny, nz = grid.nx, grid.ny, grid.nz
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

    rk3 = STSTendencyStepper.factory(
        "rk3ws",
        tc1,
        execution_policy="serial",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    out_diagnostics, out_state = rk3(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = to_numpy(state["air_isentropic_density"].to_units("kg m^-2 K^-1").data)
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    vx = to_numpy(state["x_velocity"].to_units("m s^-1").data)

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
    s1 = 1.0 / 3.0 * (2.0 * s + s_prv + dt.total_seconds() * s_tnd)

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su1 = 1.0 / 3.0 * (2.0 * su + su_prv + dt.total_seconds() * su_tnd)

    raw_state_1 = {
        "time": state["time"] + dt / 3.0,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
    }
    properties = {
        "air_isentropic_density": {
            "units": "kg m^-2 K^-1",
            "grid_shape": (nx, ny, nz),
            "set_coordinates": False,
        },
        "x_momentum_isentropic": {
            "units": "kg m^-1 K^-1 s^-1",
            "grid_shape": (nx, ny, nz),
            "set_coordinates": False,
        },
    }
    state_1 = get_dataarray_dict(raw_state_1, grid, properties)
    state_1["x_velocity_at_u_locations"] = state["x_velocity_at_u_locations"]

    tendencies, _ = tc1(state_1)

    s_tnd = to_numpy(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s2 = 0.5 * (s + s_prv + dt.total_seconds() * s_tnd)

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su2 = 0.5 * (su + su_prv + dt.total_seconds() * su_tnd)

    raw_state_2 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
    }
    state_2 = get_dataarray_dict(raw_state_2, grid, properties)
    state_2["x_velocity_at_u_locations"] = state["x_velocity_at_u_locations"]

    tendencies, _ = tc1(state_2)

    s_tnd = to_numpy(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s3 = s_prv + dt.total_seconds() * s_tnd
    compare_arrays(s3, out_state["air_isentropic_density"].data, slice=slc)

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su3 = su_prv + dt.total_seconds() * su_tnd
    compare_arrays(su3, out_state["x_momentum_isentropic"].data, slice=slc)

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx3 = vx_prv + dt.total_seconds() * vx_tnd
    compare_arrays(vx3, out_state["x_velocity"].data, slice=slc)

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
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)
    hb = domain.horizontal_boundary

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

    rk3 = STSTendencyStepper.factory(
        "rk3ws",
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    out_diagnostics, out_state = rk3(state, prv_state, dt)

    tendencies, diagnostics = tc1(state)

    s = to_numpy(state["air_isentropic_density"].to_units("kg m^-2 K^-1").data)
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    vx = to_numpy(state["x_velocity"].to_units("m s^-1").data)

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
    s1 = 1.0 / 3.0 * (2.0 * s + s_prv + dt.total_seconds() * s_tnd)
    hb_np.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt / 3.0,
    )

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su1 = 1.0 / 3.0 * (2.0 * su + su_prv + dt.total_seconds() * su_tnd)
    hb_np.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt / 3.0,
    )

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx1 = 1.0 / 3.0 * (2.0 * vx + vx_prv + dt.total_seconds() * vx_tnd)
    hb_np.enforce_field(
        vx1,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + dt / 3.0,
    )

    raw_state_1 = {
        "time": state["time"] + dt / 3.0,
        "air_isentropic_density": s1,
        "x_momentum_isentropic": su1,
    }
    properties = {
        "air_isentropic_density": {
            "units": "kg m^-2 K^-1",
            "grid_shape": (nx, ny, nz),
            "set_coordinates": False,
        },
        "x_momentum_isentropic": {
            "units": "kg m^-1 K^-1 s^-1",
            "grid_shape": (nx, ny, nz),
            "set_coordinates": False,
        },
    }
    state_1 = get_dataarray_dict(raw_state_1, grid, properties)
    state_1["x_velocity_at_u_locations"] = state["x_velocity_at_u_locations"]

    tendencies, _ = tc1(state_1)

    s_tnd = to_numpy(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s2 = 0.5 * (s + s_prv + dt.total_seconds() * s_tnd)
    hb_np.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
    )

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su2 = 0.5 * (su + su_prv + dt.total_seconds() * su_tnd)
    hb_np.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
    )

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx2 = 0.5 * (vx + vx_prv + dt.total_seconds() * vx_tnd)
    hb_np.enforce_field(
        vx2,
        field_name="x_velocity_at_u_locations",
        field_units="m s^-1",
        time=state["time"] + 0.5 * dt,
    )

    raw_state_2 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": s2,
        "x_momentum_isentropic": su2,
    }
    state_2 = get_dataarray_dict(raw_state_2, grid, properties)
    state_2["x_velocity_at_u_locations"] = state["x_velocity_at_u_locations"]

    tendencies, _ = tc1(state_2)

    s_tnd = to_numpy(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s3 = s_prv + dt.total_seconds() * s_tnd
    hb_np.enforce_field(
        s3,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
    )
    compare_arrays(s3, out_state["air_isentropic_density"].data, slice=slc)

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su3 = su_prv + dt.total_seconds() * su_tnd
    hb.enforce_field(
        su3,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
    )
    compare_arrays(su3, out_state["x_momentum_isentropic"].data, slice=slc)

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx3 = vx_prv + dt.total_seconds() * vx_tnd
    hb_np.enforce_field(
        vx3,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + dt,
    )
    compare_arrays(vx3, out_state["x_velocity"].data, slice=slc)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].data,
        out_diagnostics["fake_variable"].data,
    )


if __name__ == "__main__":
    pytest.main([__file__])
