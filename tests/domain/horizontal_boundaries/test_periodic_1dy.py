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
from hypothesis import (
    given,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import StorageOptions

from tests.conf import backend as conf_backend, dtype as conf_dtype
from tests.domain.horizontal_boundaries.test_periodic import validate
from tests.strategies import (
    st_horizontal_boundary_layers,
    st_physical_grid,
    st_raw_field,
)
from tests.utilities import compare_arrays, hyp_settings


@hyp_settings
@given(hyp_st.data())
def test_properties(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)),
        label="grid",
    )
    ny = grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("periodic", grid, nb)

    assert hb.nx == 1
    assert hb.ny == ny
    assert hb.nb == nb
    assert hb.ni == 2 * nb + 1
    assert hb.nj == ny + 2 * nb
    assert hb.type == "periodic"
    assert len(hb.kwargs) == 0


@hyp_settings
@given(hyp_st.data())
def test_axis(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)),
        label="grid",
    )
    ny = grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("periodic", grid, nb)

    # numerical axes - mass points
    validate(hb.get_numerical_yaxis, grid.y, nb)

    # numerical axes - staggered points
    validate(hb.get_numerical_yaxis_staggered, grid.y_at_v_locations, nb)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_field(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    so = StorageOptions(dtype=dtype)

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)),
        label="grid",
    )
    ny, nz = grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    pfield = data.draw(
        st_raw_field(
            (2, ny + 1, nz), -1e4, 1e4, backend=backend, storage_options=so
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "periodic", grid, nb, backend=backend, storage_options=so
    )

    # (1, ny)
    pf = pfield[:-1, :-1]
    cf = zeros(
        backend, shape=(2 * nb + 1, ny + 2 * nb, nz), storage_options=so
    )
    cf[nb : nb + 1, nb : ny + nb] = pf
    cf[nb : nb + 1, :nb] = cf[nb : nb + 1, ny - 1 : ny + nb - 1]
    cf[nb : nb + 1, -nb:] = cf[nb : nb + 1, nb + 1 : 2 * nb + 1]
    cf[:nb, :] = cf[nb : nb + 1, :]
    cf[-nb:, :] = cf[nb : nb + 1, :]
    compare_arrays(hb.get_numerical_field(pf), cf)
    compare_arrays(hb.get_physical_field(cf), pf)

    # (1, ny+1)
    pf = pfield[:-1, :]
    cf = zeros(
        backend, shape=(2 * nb + 1, ny + 1 + 2 * nb, nz), storage_options=so
    )
    cf[nb : nb + 1, nb : ny + 1 + nb] = pf
    cf[nb : nb + 1, :nb] = cf[nb : nb + 1, ny - 1 : ny + nb - 1]
    cf[nb : nb + 1, -nb:] = cf[nb : nb + 1, nb + 2 : 2 * nb + 2]
    cf[:nb, :] = cf[nb : nb + 1, :]
    cf[-nb:, :] = cf[nb : nb + 1, :]
    compare_arrays(hb.get_numerical_field(pf, field_name="at_v_locations"), cf)
    compare_arrays(hb.get_physical_field(cf, field_name="at_v_locations"), pf)

    # (2, ny)
    pf = pfield[:, :-1]
    cf = zeros(
        backend, shape=(2 * nb + 2, ny + 2 * nb, nz), storage_options=so
    )
    cf[nb : nb + 2, nb : ny + nb] = pf
    cf[nb : nb + 2, :nb] = cf[nb : nb + 2, ny - 1 : ny + nb - 1]
    cf[nb : nb + 2, -nb:] = cf[nb : nb + 2, nb + 1 : 2 * nb + 1]
    cf[:nb, :] = cf[nb : nb + 1, :]
    cf[-nb:, :] = cf[nb + 1 : nb + 2, :]
    compare_arrays(hb.get_numerical_field(pf, field_name="at_u_locations"), cf)
    compare_arrays(hb.get_physical_field(cf, field_name="at_u_locations"), pf)

    # (nx+1, 2)
    pf = pfield
    cf = zeros(
        backend, shape=(2 * nb + 2, ny + 1 + 2 * nb, nz), storage_options=so
    )
    cf[nb : nb + 2, nb : ny + 1 + nb] = pf
    cf[nb : nb + 2, :nb] = cf[nb : nb + 2, ny - 1 : ny + nb - 1]
    cf[nb : nb + 2, -nb:] = cf[nb : nb + 2, nb + 2 : 2 * nb + 2]
    cf[:nb, :] = cf[nb : nb + 1, :]
    cf[-nb:, :] = cf[nb + 1 : nb + 2, :]
    compare_arrays(
        hb.get_numerical_field(pf, field_name="at_u_locations_at_v_locations"),
        cf,
    )
    compare_arrays(
        hb.get_physical_field(cf, field_name="at_u_locations_at_v_locations"),
        pf,
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_enforce(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    so = StorageOptions(dtype=dtype)

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)),
        label="grid",
    )
    ny, nz = grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    cfield = data.draw(
        st_raw_field(
            (2 * nb + 2, ny + 2 * nb + 1, nz),
            -1e4,
            1e4,
            backend=backend,
            storage_options=so,
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "periodic", grid, nb, backend=backend, storage_options=so
    )

    # (1, ny)
    cf = deepcopy(cfield[:-1, :-1])
    hb.enforce_field(cf)
    compare_arrays(cf[nb:-nb, :nb], cf[nb:-nb, ny - 1 : ny - 1 + nb])
    compare_arrays(cf[nb:-nb, -nb:], cf[nb:-nb, nb + 1 : 2 * nb + 1])
    compare_arrays(cf[:nb, :], cf[nb : nb + 1, :])
    compare_arrays(cf[-nb:, :], cf[nb : nb + 1, :])

    # (1, ny+1)
    cf = deepcopy(cfield[:-1, :])
    hb.enforce_field(cf, field_name="at_v_locations")
    compare_arrays(cf[nb:-nb, :nb], cf[nb:-nb, ny - 1 : ny - 1 + nb])
    compare_arrays(cf[nb:-nb, -nb:], cf[nb:-nb, nb + 2 : 2 * nb + 2])
    compare_arrays(cf[:nb, :], cf[nb : nb + 1, :])
    compare_arrays(cf[-nb:, :], cf[nb : nb + 1, :])

    # (2, ny)
    cf = deepcopy(cfield[:, :-1])
    hb.enforce_field(cf, field_name="at_u_locations")
    compare_arrays(cf[nb:-nb, :nb], cf[nb:-nb, ny - 1 : ny - 1 + nb])
    compare_arrays(cf[nb:-nb, -nb:], cf[nb:-nb, nb + 1 : 2 * nb + 1])
    compare_arrays(cf[:nb, :], cf[nb : nb + 1, :])
    compare_arrays(cf[-nb:, :], cf[nb + 1 : nb + 2, :])

    # (2, ny+1)
    cf = cfield
    hb.enforce_field(cf, field_name="at_u_locations_at_v_locations")
    compare_arrays(cf[nb:-nb, :nb], cf[nb:-nb, ny - 1 : ny - 1 + nb])
    compare_arrays(cf[nb:-nb, -nb:], cf[nb:-nb, nb + 2 : 2 * nb + 2])
    compare_arrays(cf[:nb, :], cf[nb : nb + 1, :])
    compare_arrays(cf[-nb:, :], cf[nb + 1 : nb + 2, :])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_outermost_layers(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    so = StorageOptions(dtype=dtype)

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)),
        label="grid",
    )
    ny, nz = grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    cfield = data.draw(
        st_raw_field(
            (2 * nb + 2, ny + 2 * nb + 1, nz),
            -1e4,
            1e4,
            backend=backend,
            storage_options=so,
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "periodic", grid, nb, backend=backend, storage_options=so
    )

    # (1, ny+1)
    cf = deepcopy(cfield[:-1, :])
    hb.set_outermost_layers_y(cf)
    compare_arrays(cf[:, 0], cf[:, -2])
    compare_arrays(cf[:, -1], cf[:, 1])

    # (2, ny+1)
    cf = deepcopy(cfield)
    hb.set_outermost_layers_y(cf)
    compare_arrays(cf[:, 0], cf[:, -2])
    compare_arrays(cf[:, -1], cf[:, 1])


if __name__ == "__main__":
    pytest.main([__file__])
