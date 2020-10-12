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
from tasmania.python.domain.grid import NumericalGrid

from tests.conf import backend as conf_backend, dtype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary_layers,
    st_physical_grid,
    st_raw_field,
)
from tests.utilities import (
    compare_arrays,
    compare_dataarrays,
    hyp_settings,
    pi_function,
)


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
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("dirichlet", nx, ny, nb)

    assert hb.nx == 1
    assert hb.ny == ny
    assert hb.nb == nb
    assert hb.ni == 2 * nb + 1
    assert hb.nj == ny
    assert hb.type == "dirichlet"
    assert "core" in hb.kwargs
    assert len(hb.kwargs) == 1


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
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("dirichlet", nx, ny, nb)

    #
    # get_numerical_axis
    #
    # mass points
    px = grid.y
    cx = hb.get_numerical_yaxis(px, dims=px.dims[0])
    compare_dataarrays(cx, px)

    # staggered points
    px = grid.y_at_v_locations
    cx = hb.get_numerical_yaxis(px, dims=px.dims[0])
    compare_dataarrays(cx, px)

    #
    # get_physical_axis
    #
    # mass points
    px_val = grid.y
    cx = hb.get_numerical_yaxis(px_val)
    px = hb.get_physical_yaxis(cx)
    compare_dataarrays(px, px_val)

    # staggered points
    px_val = grid.y_at_v_locations
    cx = hb.get_numerical_yaxis(px_val)
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
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)),
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
            dtype=dtype,
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "dirichlet", nx, ny, nb, backend=backend, dtype=dtype
    )

    # (1, ny)
    pf = pfield[:-1, :-1]
    cf = hb.get_numerical_field(pf)
    compare_arrays(cf, pf)
    compare_arrays(hb.get_physical_field(cf), pf)

    # (2, ny)
    pf = pfield[:, :-1]
    cf = hb.get_numerical_field(pf, field_name="at_u_locations")
    compare_arrays(cf[: nb + 1, :], pf[:1, :])
    compare_arrays(cf[-nb - 1 :, :], pf[-1:, :])
    compare_arrays(hb.get_physical_field(cf, field_name="at_u_locations"), pf)

    # (1, ny+1)
    pf = pfield[:-1, :]
    cf = hb.get_numerical_field(pf, field_name="at_v_locations")
    compare_arrays(cf, pf)
    compare_arrays(hb.get_physical_field(cf, field_name="at_v_locations"), pf)

    # (2, ny+1)
    pf = pfield
    cf = hb.get_numerical_field(pf, field_name="at_uv_locations")
    compare_arrays(cf[: nb + 1, :], pf[:1, :])
    compare_arrays(cf[-nb - 1 :, :], pf[-1:, :])
    compare_arrays(hb.get_physical_field(cf, field_name="at_uv_locations"), pf)


def enforce(cf_val, hb):
    nx, ny, nb = hb.nx, hb.ny, hb.nb

    cf_val[nb:-nb, :nb] = np.pi
    cf_val[nb:-nb, -nb:] = np.pi
    cf_val[:nb, :] = cf_val[nb : nb + 1, :]
    cf_val[-nb:, :] = cf_val[-nb - 1 : -nb, :]


def validation(cf, cf_val, hb):
    nb = hb.nb

    compare_arrays(cf[nb:-nb, nb:-nb], cf_val[nb:-nb, nb:-nb])
    compare_arrays(cf[nb:-nb, :nb], cf_val[nb:-nb, :nb])
    compare_arrays(cf[nb:-nb, -nb:], cf_val[nb:-nb, -nb:])
    compare_arrays(cf[:nb, :], cf_val[nb : nb + 1, :])
    compare_arrays(cf[-nb:, :], cf_val[-nb - 1 : -nb, :])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_enforce(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    storage_shape = (nx + 2 * nb + 1, ny + 1, nz + 1)
    cfield = data.draw(
        st_raw_field(storage_shape, -1e4, 1e4, backend=backend, dtype=dtype)
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "dirichlet",
        nx,
        ny,
        nb,
        backend=backend,
        dtype=dtype,
        core=pi_function,
    )

    cgrid = NumericalGrid(grid, hb)

    # (1, ny)
    cf = deepcopy(cfield)
    hb.enforce_field(cf, grid=cgrid)
    cf_val = deepcopy(cfield[:-1, :-1])
    enforce(cf_val, hb)
    validation(cf[:-1, :-1], cf_val, hb)

    # (2, ny)
    cf = deepcopy(cfield)
    hb.enforce_field(
        cf, field_name="afield_at_u_locations_on_interface_levels", grid=cgrid
    )
    cf_val = deepcopy(cfield[:, :-1])
    enforce(cf_val, hb)
    validation(cf[:, :-1], cf_val, hb)

    # (1, ny+1)
    cf = deepcopy(cfield)
    hb.enforce_field(cf, field_name="afield_at_v_locations", grid=cgrid)
    cf_val = deepcopy(cfield[:-1, :])
    enforce(cf_val, hb)
    validation(cf[:-1, :], cf_val, hb)

    # (2, ny+1)
    cf = deepcopy(cfield)
    hb.enforce_field(cf, field_name="afield_at_uv_locations", grid=cgrid)
    cf_val = cfield
    enforce(cf_val, hb)
    validation(cf, cf_val, hb)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_outermost_layers(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    storage_shape = (nx + +2 * nb + 1, ny + 1, nz + 1)
    cfield = np.zeros(storage_shape, dtype=dtype)

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "dirichlet",
        nx,
        ny,
        nb,
        backend=backend,
        dtype=dtype,
        core=pi_function,
    )

    cgrid = NumericalGrid(grid, hb)

    # (2, ny)
    cf = deepcopy(cfield)
    hb.set_outermost_layers_x(
        cf, field_name="afield_at_u_locations", grid=cgrid
    )
    compare_arrays(cf[0, :-1], np.pi * np.ones((1, 1, 1), dtype=dtype))
    compare_arrays(cf[-1, :-1], np.pi * np.ones((1, 1, 1), dtype=dtype))

    # (1, ny+1)
    cf = deepcopy(cfield)
    hb.set_outermost_layers_y(
        cf, field_name="afield_at_v_locations", grid=cgrid
    )
    compare_arrays(cf[:-1, 0], np.pi * np.ones((1, 1, 1), dtype=dtype))
    compare_arrays(cf[:-1, -1], np.pi * np.ones((1, 1, 1), dtype=dtype))

    # (2, ny+1)
    cf = deepcopy(cfield)
    hb.set_outermost_layers_x(
        cf, field_name="afield_at_uv_locations", grid=cgrid
    )
    hb.set_outermost_layers_y(
        cf, field_name="afield_at_uv_locations", grid=cgrid
    )
    compare_arrays(cf[0, :], np.pi * np.ones((1, 1, 1), dtype=dtype))
    compare_arrays(cf[-1, :], np.pi * np.ones((1, 1, 1), dtype=dtype))
    compare_arrays(cf[:, 0], np.pi * np.ones((1, 1, 1), dtype=dtype))
    compare_arrays(cf[:, -1], np.pi * np.ones((1, 1, 1), dtype=dtype))


if __name__ == "__main__":
    pytest.main([__file__])
