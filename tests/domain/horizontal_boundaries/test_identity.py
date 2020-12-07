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
from hypothesis import (
    given,
    strategies as hyp_st,
    reproduce_failure,
)
import numpy as np
import pytest

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
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
    hb = HorizontalBoundary.factory("identity", nx, ny, nb)

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
    hb = HorizontalBoundary.factory("identity", nx, ny, nb)

    #
    # get_numerical_axis
    #
    # mass points
    px = grid.x
    cx = hb.get_numerical_xaxis(px, dims=px.dims[0])
    compare_dataarrays(cx, px)
    cx = hb.get_numerical_yaxis(px, dims=px.dims[0])
    compare_dataarrays(cx, px)

    # staggered points
    px = grid.x_at_u_locations
    cx = hb.get_numerical_xaxis(px, dims=px.dims[0])
    compare_dataarrays(cx, px)
    cx = hb.get_numerical_yaxis(px, dims=px.dims[0])
    compare_dataarrays(cx, px)

    #
    # get_physical_axis
    #
    # mass points
    px_val = grid.x
    cx = hb.get_numerical_xaxis(px_val)
    px = hb.get_physical_xaxis(cx)
    compare_dataarrays(px, px_val)
    px = hb.get_physical_yaxis(cx)
    compare_dataarrays(px, px_val)

    # staggered points
    px_val = grid.y_at_v_locations
    cx = hb.get_numerical_xaxis(px_val)
    px = hb.get_physical_xaxis(cx)
    compare_dataarrays(px, px_val)
    px = hb.get_physical_yaxis(cx)
    compare_dataarrays(px, px_val)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_field(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    pfield = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz), -1e4, 1e4, backend=backend, dtype=dtype,
        )
    )

    # ========================================
    # test
    # ========================================
    so = StorageOptions(dtype=dtype)
    hb = HorizontalBoundary.factory(
        "identity", nx, ny, nb, backend=backend, storage_options=so
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
    grid = data.draw(
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    storage_shape = (nx + 1, ny + 1, nz + 1)
    cfield = data.draw(
        st_raw_field(storage_shape, -1e4, 1e4, backend=backend, dtype=dtype)
    )

    # ========================================
    # test
    # ========================================
    so = StorageOptions(dtype=dtype)
    hb = HorizontalBoundary.factory(
        "identity", nx, ny, nb, backend=backend, storage_options=so
    )

    cfield_val = deepcopy(cfield)
    hb.enforce_field(cfield)
    compare_arrays(cfield, cfield_val)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_outermost_layers(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    storage_shape = (nx + 1, ny + 1, nz + 1)
    cfield = np.zeros(storage_shape, dtype=dtype)

    # ========================================
    # test
    # ========================================
    so = StorageOptions(dtype=dtype)
    hb = HorizontalBoundary.factory(
        "identity", nx, ny, nb, backend=backend, storage_options=so
    )

    cfield_val = deepcopy(cfield)
    hb.set_outermost_layers_x(cfield)
    compare_arrays(cfield, cfield_val)
    hb.set_outermost_layers_y(cfield)
    compare_arrays(cfield, cfield_val)


if __name__ == "__main__":
    pytest.main([__file__])
