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

from tasmania.python.physics.turbulence import Smagorinsky2d
from tasmania import get_dataarray_3d

try:
    from .conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from .utils import compare_dataarrays, st_domain, st_floats, st_one_of, st_raw_field
except (ImportError, ModuleNotFoundError):
    from conf import backend as conf_backend, halo as conf_halo, nb as conf_nb
    from utils import compare_dataarrays, st_domain, st_floats, st_one_of, st_raw_field


def smagorinsky2d_validation(dx, dy, cs, u, v):
    u_tnd = np.zeros_like(u, dtype=u.dtype)
    v_tnd = np.zeros_like(v, dtype=v.dtype)

    s00 = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
    s01 = 0.5 * (
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)
        + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
    )
    s11 = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
    nu = (cs ** 2) * (dx * dy) * (2.0 * s00 ** 2 + 4.0 * s01 ** 2 + 2.0 * s11 ** 2) ** 0.5
    u_tnd[2:-2, 2:-2] = 2.0 * (
        (nu[2:, 1:-1] * s00[2:, 1:-1] - nu[:-2, 1:-1] * s00[:-2, 1:-1]) / (2.0 * dx)
        + (nu[1:-1, 2:] * s01[1:-1, 2:] - nu[1:-1, :-2] * s01[1:-1, :-2]) / (2.0 * dy)
    )
    v_tnd[2:-2, 2:-2] = 2.0 * (
        (nu[2:, 1:-1] * s01[2:, 1:-1] - nu[:-2, 1:-1] * s01[:-2, 1:-1]) / (2.0 * dx)
        + (nu[1:-1, 2:] * s11[1:-1, 2:] - nu[1:-1, :-2] * s11[1:-1, :-2]) / (2.0 * dy)
    )

    return u_tnd, v_tnd


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_smagorinsky2d(data):
    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)), label="nb")
    domain = data.draw(st_domain(nb=nb), label="domain")
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    halo = data.draw(st_one_of(conf_halo), label="halo")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    u = data.draw(
        st_raw_field(storage_shape, -1e3, 1e3, backend=backend, dtype=dtype, halo=halo),
        label="u",
    )
    v = data.draw(
        st_raw_field(storage_shape, -1e3, 1e3, backend=backend, dtype=dtype, halo=halo),
        label="v",
    )

    cs = data.draw(hyp_st.floats(min_value=0, max_value=10), label="cs")

    time = data.draw(hyp_st.datetimes(), label="time")

    # ========================================
    # test bed
    # ========================================
    dx = grid.dx.to_units("m").values.item()
    dy = grid.dy.to_units("m").values.item()

    state = {
        "time": time,
        "x_velocity": get_dataarray_3d(
            u, grid, "m s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
        ),
        "y_velocity": get_dataarray_3d(
            v, grid, "m s^-1", grid_shape=(nx, ny, nz), set_coordinates=False
        ),
    }

    u_tnd, v_tnd = smagorinsky2d_validation(dx, dy, cs, u, v)

    smag = Smagorinsky2d(
        domain,
        smagorinsky_constant=cs,
        backend=backend,
        dtype=dtype,
        halo=halo,
        storage_shape=storage_shape,
    )

    tendencies, diagnostics = smag(state)

    assert "x_velocity" in tendencies
    compare_dataarrays(
        tendencies["x_velocity"][nb : nx - nb, nb : ny - nb, :nz],
        get_dataarray_3d(
            u_tnd, grid, "m s^-2", grid_shape=(nx, ny, nz), set_coordinates=False
        )[nb : nx - nb, nb : ny - nb, :nz],
        compare_coordinate_values=False,
    )
    assert "y_velocity" in tendencies
    compare_dataarrays(
        tendencies["y_velocity"][nb : nx - nb, nb : ny - nb, :nz],
        get_dataarray_3d(
            v_tnd, grid, "m s^-2", grid_shape=(nx, ny, nz), set_coordinates=False
        )[nb : nx - nb, nb : ny - nb, :nz],
        compare_coordinate_values=False,
    )
    assert len(tendencies) == 2

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])