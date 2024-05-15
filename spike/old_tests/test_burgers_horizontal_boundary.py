# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
from hypothesis import given, HealthCheck, settings, strategies as hyp_st
import numpy as np
import pytest
from sympl import DataArray

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

from spike.burgers_horizontal_boundary import ZhaoHorizontalBoundary
from tasmania.python.burgers.state import ZhaoSolutionFactory
from spike.horizontal_boundary import HorizontalBoundary


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_zhao_test_case_hb(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(utils.st_grid_xyz(zaxis_length=(1, 1)))
    state = data.draw(utils.st_burgers_state(grid))
    eps = DataArray(
        data.draw(utils.st_floats(min_value=-1e1, max_value=1e1)),
        attrs={"units": "m^2 s^-1"},
    )
    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=int(min(grid.nx, grid.ny) / 2))
    )
    time = data.draw(
        hyp_st.datetimes(
            min_value=state["time"],
            max_value=state["time"] + timedelta(hours=1),
        )
    )

    # ========================================
    # random data generation
    # ========================================
    u = state["x_velocity"].to_units("m s^-1").values
    u_dc = deepcopy(u)
    v = state["y_velocity"].to_units("m s^-1").values
    v_dc = deepcopy(v)

    zsf = ZhaoSolutionFactory(eps)

    hb = HorizontalBoundary.factory(
        "zhao", grid, nb, init_time=state["time"], solution_factory=zsf
    )

    assert isinstance(hb, ZhaoHorizontalBoundary)

    hb.enforce(u, u, field_name="x_velocity", time=time)
    assert np.allclose(u[nb:-nb, nb:-nb, :], u_dc[nb:-nb, nb:-nb, :])
    assert np.allclose(
        u[:nb, :, :],
        zsf(
            grid,
            time - state["time"],
            slice_x=slice(0, nb),
            field_name="x_velocity",
        ),
        equal_nan=True,
    )
    assert np.allclose(
        u[-nb:, :, :],
        zsf(
            grid,
            time - state["time"],
            slice_x=slice(grid.nx - nb, grid.nx),
            field_name="x_velocity",
        ),
        equal_nan=True,
    )
    assert np.allclose(
        u[:, :nb, :],
        zsf(
            grid,
            time - state["time"],
            slice_y=slice(0, nb),
            field_name="x_velocity",
        ),
        equal_nan=True,
    )
    assert np.allclose(
        u[:, -nb:, :],
        zsf(
            grid,
            time - state["time"],
            slice_y=slice(grid.ny - nb, grid.ny),
            field_name="x_velocity",
        ),
        equal_nan=True,
    )

    hb.enforce(v, v, field_name="y_velocity", time=time)
    assert np.allclose(v[nb:-nb, nb:-nb, :], v_dc[nb:-nb, nb:-nb, :])
    assert np.allclose(
        v[:nb, :, :],
        zsf(
            grid,
            time - state["time"],
            slice_x=slice(0, nb),
            field_name="y_velocity",
        ),
        equal_nan=True,
    )
    assert np.allclose(
        v[-nb:, :, :],
        zsf(
            grid,
            time - state["time"],
            slice_x=slice(grid.nx - nb, grid.nx),
            field_name="y_velocity",
        ),
        equal_nan=True,
    )
    assert np.allclose(
        v[:, :nb, :],
        zsf(
            grid,
            time - state["time"],
            slice_y=slice(0, nb),
            field_name="y_velocity",
        ),
        equal_nan=True,
    )
    assert np.allclose(
        v[:, -nb:, :],
        zsf(
            grid,
            time - state["time"],
            slice_y=slice(grid.ny - nb, grid.ny),
            field_name="y_velocity",
        ),
        equal_nan=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
    # test_zhao_test_case_hb()
