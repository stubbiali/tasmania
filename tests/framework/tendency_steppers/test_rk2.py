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
    strategies as hyp_st,
    reproduce_failure,
)
import pytest
from sympl import units_are_same

from tasmania.python.framework.allocators import as_storage
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.tendency_stepper import TendencyStepper
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
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

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

    dt = data.draw(
        st_timedeltas(
            min_value=timedelta(seconds=0), max_value=timedelta(minutes=60)
        ),
        label="dt",
    )

    # ========================================
    # test bed
    # ========================================
    slc = (slice(nx), slice(ny), slice(nz))

    tc1 = make_fake_tendency_component_1(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    rk2 = TendencyStepper.factory(
        "rk2",
        tc1,
        execution_policy="serial",
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    assert "air_isentropic_density" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["air_isentropic_density"]["units"],
        "kg m^-2 K^-1",
    )
    assert "x_momentum_isentropic" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_momentum_isentropic"]["units"],
        "kg m^-1 K^-1 s^-1",
    )
    assert "x_velocity" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_velocity"]["units"], "m s^-1"
    )
    assert len(rk2.output_properties) == 3

    out_diagnostics, out_state = rk2(state, dt)

    tendencies, diagnostics = tc1(state)

    s = to_numpy(state["air_isentropic_density"].to_units("kg m^-2 K^-1").data)
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    vx = to_numpy(state["x_velocity"].to_units("m s^-1").data)

    s_tnd = to_numpy(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s1 = s + 0.5 * dt.total_seconds() * s_tnd

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su1 = su + 0.5 * dt.total_seconds() * su_tnd

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx1 = vx + 0.5 * dt.total_seconds() * vx_tnd

    raw_state_1 = {
        "time": state["time"] + 0.5 * dt,
        "air_isentropic_density": as_storage(
            backend, data=s1, storage_options=so
        ),
        "x_momentum_isentropic": as_storage(
            backend, data=su1, storage_options=so
        ),
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
    s2 = s + dt.total_seconds() * s_tnd
    compare_arrays(s2, out_state["air_isentropic_density"].data, slice=slc)

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su2 = su + dt.total_seconds() * su_tnd
    compare_arrays(su2, out_state["x_momentum_isentropic"].data, slice=slc)

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx2 = vx + dt.total_seconds() * vx_tnd
    compare_arrays(vx2, out_state["x_velocity"].data, slice=slc)

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
    storage_shape = (nx + dnx, ny + dny, nz + dnz)
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
    slc = (slice(nx), slice(ny), slice(nz))
    hb_np = domain.copy(backend="numpy").horizontal_boundary

    tc1 = make_fake_tendency_component_1(
        domain,
        "numerical",
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    rk2 = TendencyStepper.factory(
        "rk2",
        tc1,
        execution_policy="serial",
        enforce_horizontal_boundary=True,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )

    assert "air_isentropic_density" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["air_isentropic_density"]["units"],
        "kg m^-2 K^-1",
    )
    assert "x_momentum_isentropic" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_momentum_isentropic"]["units"],
        "kg m^-1 K^-1 s^-1",
    )
    assert "x_velocity" in rk2.output_properties
    assert units_are_same(
        rk2.output_properties["x_velocity"]["units"], "m s^-1"
    )
    assert len(rk2.output_properties) == 3

    out_diagnostics, out_state = rk2(state, dt)

    tendencies, diagnostics = tc1(state)

    s = to_numpy(state["air_isentropic_density"].to_units("kg m^-2 K^-1").data)
    su = to_numpy(
        state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
    )
    vx = to_numpy(state["x_velocity"].to_units("m s^-1").data)

    s_tnd = to_numpy(
        tendencies["air_isentropic_density"].to_units("kg m^-2 K^-1 s^-1").data
    )
    s1 = s + 0.5 * dt.total_seconds() * s_tnd
    hb_np.enforce_field(
        s1,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + 0.5 * dt,
    )

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su1 = su + 0.5 * dt.total_seconds() * su_tnd
    hb_np.enforce_field(
        su1,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + 0.5 * dt,
    )

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx1 = vx + 0.5 * dt.total_seconds() * vx_tnd
    hb_np.enforce_field(
        vx1,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + 0.5 * dt,
    )

    raw_state_1 = {
        "time": state["time"] + 0.5 * dt,
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
    s2 = s + dt.total_seconds() * s_tnd
    hb_np.enforce_field(
        s2,
        field_name="air_isentropic_density",
        field_units="kg m^-2 K^-1",
        time=state["time"] + dt,
    )
    compare_arrays(s2, out_state["air_isentropic_density"].data, slice=slc)

    su_tnd = to_numpy(
        tendencies["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-2").data
    )
    su2 = su + dt.total_seconds() * su_tnd
    hb_np.enforce_field(
        su2,
        field_name="x_momentum_isentropic",
        field_units="kg m^-1 K^-1 s^-1",
        time=state["time"] + dt,
    )
    compare_arrays(su2, out_state["x_momentum_isentropic"].data, slice=slc)

    vx_tnd = to_numpy(tendencies["x_velocity"].to_units("m s^-2").data)
    vx2 = vx + dt.total_seconds() * vx_tnd
    hb_np.enforce_field(
        vx2,
        field_name="x_velocity",
        field_units="m s^-1",
        time=state["time"] + dt,
    )
    compare_arrays(vx2, out_state["x_velocity"].data, slice=slc)

    assert "fake_variable" in out_diagnostics
    compare_arrays(
        diagnostics["fake_variable"].data,
        out_diagnostics["fake_variable"].data,
    )


if __name__ == "__main__":
    pytest.main([__file__])
