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
from sympl import DataArray

from tasmania.python.burgers.state import ZhaoSolutionFactory, ZhaoStateFactory

try:
    from .utils import st_floats, st_physical_grid
except (ImportError, ModuleNotFoundError):
    from utils import st_floats, st_physical_grid


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_zhao_solution_factory(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(zaxis_length=(1, 1)))
    eps = DataArray(data.draw(st_floats()), attrs={"units": "m^2 s^-1"})

    el0 = data.draw(hyp_st.integers(min_value=0, max_value=grid.nx))
    el1 = data.draw(hyp_st.integers(min_value=0, max_value=grid.nx))
    assume(el0 != el1)
    slice_x = slice(el0, el1) if el0 < el1 else slice(el1, el0)

    el0 = data.draw(hyp_st.integers(min_value=0, max_value=grid.ny))
    el1 = data.draw(hyp_st.integers(min_value=0, max_value=grid.ny))
    assume(el0 != el1)
    slice_y = slice(el0, el1) if el0 < el1 else slice(el1, el0)

    init_time = data.draw(hyp_st.datetimes())
    time = data.draw(hyp_st.datetimes(min_value=init_time))

    # ========================================
    # test
    # ========================================
    zsf = ZhaoSolutionFactory(init_time, eps)

    u = zsf(time, grid, field_name="x_velocity")
    v = zsf(time, grid, field_name="x_velocity")
    assert u.shape == (grid.nx, grid.ny, grid.nz)
    assert v.shape == (grid.nx, grid.ny, grid.nz)

    u = zsf(time, grid, slice_x=slice_x, field_name="x_velocity")
    v = zsf(time, grid, slice_x=slice_x, field_name="x_velocity")
    assert u.shape == (slice_x.stop - slice_x.start, grid.ny, grid.nz)
    assert v.shape == (slice_x.stop - slice_x.start, grid.ny, grid.nz)

    u = zsf(time, grid, slice_y=slice_y, field_name="x_velocity")
    v = zsf(time, grid, slice_y=slice_y, field_name="x_velocity")
    assert u.shape == (grid.nx, slice_y.stop - slice_y.start, grid.nz)
    assert v.shape == (grid.nx, slice_y.stop - slice_y.start, grid.nz)

    u = zsf(time, grid, slice_x=slice_x, slice_y=slice_y, field_name="x_velocity")
    v = zsf(time, grid, slice_x=slice_x, slice_y=slice_y, field_name="x_velocity")
    assert u.shape == (
        slice_x.stop - slice_x.start,
        slice_y.stop - slice_y.start,
        grid.nz,
    )
    assert v.shape == (
        slice_x.stop - slice_x.start,
        slice_y.stop - slice_y.start,
        grid.nz,
    )


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_zhao_state_factory(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(zaxis_length=(1, 1)))

    eps = DataArray(
        data.draw(st_floats(min_value=-1e10, max_value=1e10)),
        attrs={"units": "m^2 s^-1"},
    )

    init_time = data.draw(hyp_st.datetimes())
    time = data.draw(hyp_st.datetimes(min_value=init_time))

    # ========================================
    # test
    # ========================================
    zsf = ZhaoStateFactory(init_time, eps)

    state = zsf(init_time, grid)

    assert "time" in state
    assert "x_velocity" in state
    assert "y_velocity" in state
    assert len(state) == 3

    assert state["time"] == init_time

    x = grid.x.to_units("m").values
    x = np.tile(x[:, np.newaxis, np.newaxis], (1, grid.ny, grid.nz))
    y = grid.y.to_units("m").values
    y = np.tile(y[np.newaxis, :, np.newaxis], (grid.nx, 1, grid.nz))

    e = eps.to_units("m^2 s^-1").values.item()

    u = (
        -4.0
        * e
        * np.pi
        * np.cos(2.0 * np.pi * x)
        * np.sin(np.pi * y)
        / (2.0 + np.sin(2.0 * np.pi * x) * np.sin(np.pi * y))
    )
    assert np.allclose(u, state["x_velocity"])

    v = (
        -2.0
        * e
        * np.pi
        * np.sin(2.0 * np.pi * x)
        * np.cos(np.pi * y)
        / (2.0 + np.sin(2.0 * np.pi * x) * np.sin(np.pi * y))
    )
    assert np.allclose(v, state["y_velocity"])


if __name__ == "__main__":
    pytest.main([__file__])
