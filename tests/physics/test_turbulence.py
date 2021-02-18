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
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import numpy as np
import pytest

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.physics.turbulence import Smagorinsky2d
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import st_domain, st_one_of, st_raw_field
from tests.utilities import compare_dataarrays, hyp_settings


def smagorinsky2d_validation(dx, dy, cs, u, v):
    u_tnd = np.zeros_like(u)
    v_tnd = np.zeros_like(v)

    s00 = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
    s01 = 0.5 * (
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)
        + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)
    )
    s11 = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
    nu = (
        (cs ** 2)
        * (dx * dy)
        * (2.0 * s00 ** 2 + 4.0 * s01 ** 2 + 2.0 * s11 ** 2) ** 0.5
    )
    u_tnd[2:-2, 2:-2] = 2.0 * (
        (nu[2:, 1:-1] * s00[2:, 1:-1] - nu[:-2, 1:-1] * s00[:-2, 1:-1])
        / (2.0 * dx)
        + (nu[1:-1, 2:] * s01[1:-1, 2:] - nu[1:-1, :-2] * s01[1:-1, :-2])
        / (2.0 * dy)
    )
    v_tnd[2:-2, 2:-2] = 2.0 * (
        (nu[2:, 1:-1] * s01[2:, 1:-1] - nu[:-2, 1:-1] * s01[:-2, 1:-1])
        / (2.0 * dx)
        + (nu[1:-1, 2:] * s11[1:-1, 2:] - nu[1:-1, :-2] * s11[1:-1, :-2])
        / (2.0 * dy)
    )

    return u_tnd, v_tnd


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend.difference(conf.gtc_backend))
@pytest.mark.parametrize("dtype", conf.dtype)
def test_smagorinsky2d(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=2, max_value=max(2, conf.nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid = domain.numerical_grid

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    u = data.draw(
        st_raw_field(
            storage_shape, -1e3, 1e3, backend=backend, storage_options=so
        ),
        label="u",
    )
    v = data.draw(
        st_raw_field(
            storage_shape, -1e3, 1e3, backend=backend, storage_options=so
        ),
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
    u_np, v_np = to_numpy(u), to_numpy(v)

    u_tnd, v_tnd = smagorinsky2d_validation(dx, dy, cs, u_np, v_np)

    smag = Smagorinsky2d(
        domain,
        smagorinsky_constant=cs,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    tendencies, diagnostics = smag(state)

    assert "x_velocity" in tendencies
    compare_dataarrays(
        tendencies["x_velocity"],
        get_dataarray_3d(
            u_tnd,
            grid,
            "m s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
        slice=(slice(nb, nx - nb), slice(nb, ny - nb), slice(0, nz)),
    )
    assert "y_velocity" in tendencies
    compare_dataarrays(
        tendencies["y_velocity"],
        get_dataarray_3d(
            v_tnd,
            grid,
            "m s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
        slice=(slice(nb, nx - nb), slice(nb, ny - nb), slice(0, nz)),
    )
    assert len(tendencies) == 2

    assert len(diagnostics) == 0


if __name__ == "__main__":
    pytest.main([__file__])
