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
from tests.strategies import (
    st_horizontal_boundary_layers,
    st_physical_grid,
    st_raw_field,
)
from tests.utilities import compare_arrays, compare_dataarrays, hyp_settings


@hyp_settings
@given(hyp_st.data())
def test_properties(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("identity", grid, nb)

    assert hb.nx == nx
    assert hb.ny == ny
    assert hb.nb == nb
    assert hb.ni == nx
    assert hb.nj == ny
    assert hb.type == "identity"
    assert len(hb.kwargs) == 0


@hyp_settings
@given(hyp_st.data())
def test_axis(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("identity", grid, nb)

    # numerical axes - mass points
    compare_dataarrays(hb.get_numerical_xaxis(dims=grid.x.dims[0]), grid.x)
    compare_dataarrays(hb.get_numerical_yaxis(dims=grid.y.dims[0]), grid.y)

    # numerical axes - staggered points
    compare_dataarrays(
        hb.get_numerical_xaxis_staggered(dims=grid.x_at_u_locations.dims[0]),
        grid.x_at_u_locations,
    )
    compare_dataarrays(
        hb.get_numerical_yaxis_staggered(dims=grid.y_at_v_locations.dims[0]),
        grid.y_at_v_locations,
    )


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
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    pfield = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz),
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
        "identity", grid, nb, backend=backend, storage_options=so
    )

    # (nx, ny)
    pf = pfield[:-1, :-1]
    compare_arrays(hb.get_numerical_field(pf), pf)
    compare_arrays(hb.get_physical_field(pf), pf)

    # (nx+1, ny)
    pf = pfield[:, :-1]
    compare_arrays(hb.get_numerical_field(pf, field_name="at_u_locations"), pf)
    compare_arrays(hb.get_physical_field(pf, field_name="at_u_locations"), pf)

    # (nx, ny+1)
    pf = pfield[:-1, :]
    compare_arrays(hb.get_numerical_field(pf, field_name="at_v_locations"), pf)
    compare_arrays(hb.get_physical_field(pf, field_name="at_v_locations"), pf)

    # (nx+1, ny+1)
    pf = pfield
    compare_arrays(
        hb.get_numerical_field(pf, field_name="at_u_locations_at_v_locations"),
        pf,
    )
    compare_arrays(
        hb.get_physical_field(pf, field_name="at_u_locations_at_v_locations"),
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
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    storage_shape = (nx + 1, ny + 1, nz + 1)
    nfield = data.draw(
        st_raw_field(
            storage_shape, -1e4, 1e4, backend=backend, storage_options=so
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "identity", grid, nb, backend=backend, storage_options=so
    )

    nfield_val = deepcopy(nfield)
    hb.enforce_field(nfield)
    compare_arrays(nfield, nfield_val)


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
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    storage_shape = (nx + 1, ny + 1, nz + 1)
    nfield = zeros(backend, shape=storage_shape, storage_options=so)

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "identity", grid, nb, backend=backend, storage_options=so
    )

    nfield_val = deepcopy(nfield)
    hb.set_outermost_layers_x(nfield)
    compare_arrays(nfield, nfield_val)
    hb.set_outermost_layers_y(nfield)
    compare_arrays(nfield, nfield_val)


if __name__ == "__main__":
    pytest.main([__file__])
