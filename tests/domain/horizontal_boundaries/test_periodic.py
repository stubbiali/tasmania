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
import numpy as np
import pytest
from sympl import DataArray

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
    hb = HorizontalBoundary.factory("periodic", grid, nb)

    assert hb.nx == nx
    assert hb.ny == ny
    assert hb.nb == nb
    assert hb.ni == nx + 2 * nb
    assert hb.nj == ny + 2 * nb
    assert hb.type == "periodic"
    assert len(hb.kwargs) == 0


def validate(callback, paxis, nb):
    ni = len(paxis)
    di = paxis.values[1] - paxis.values[0]
    naxis_val_values = np.array(
        tuple(paxis.values[0] + i * di for i in range(-nb, ni + nb))
    )
    naxis_val = DataArray(
        naxis_val_values,
        coords={paxis.dims[0]: naxis_val_values},
        dims=paxis.dims,
        attrs=paxis.attrs.copy(),
    )
    naxis = callback(dims=paxis.dims[0])
    compare_dataarrays(naxis, naxis_val)


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
    hb = HorizontalBoundary.factory("periodic", grid, nb)

    # numerical axes - mass points
    validate(hb.get_numerical_xaxis, grid.x, nb)
    validate(hb.get_numerical_yaxis, grid.y, nb)

    # numerical axes - staggered points
    validate(hb.get_numerical_xaxis_staggered, grid.x_at_u_locations, nb)
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
        "periodic", grid, nb, backend=backend, storage_options=so
    )

    # (nx, ny)
    pf = pfield[:-1, :-1]
    nf = zeros(
        backend, shape=(nx + 2 * nb, ny + 2 * nb, nz), storage_options=so
    )
    nf[nb : nx + nb, nb : ny + nb] = pf
    nf[:nb, nb : ny + nb] = nf[nx - 1 : nx + nb - 1, nb : ny + nb]
    nf[-nb:, nb : ny + nb] = nf[nb + 1 : 2 * nb + 1, nb : ny + nb]
    nf[:, :nb] = nf[:, ny - 1 : ny + nb - 1]
    nf[:, -nb:] = nf[:, nb + 1 : 2 * nb + 1]
    compare_arrays(hb.get_numerical_field(pf), nf)
    compare_arrays(hb.get_physical_field(nf), pf)

    # (nx+1, ny)
    pf = pfield[:, :-1]
    nf = zeros(
        backend, shape=(nx + 1 + 2 * nb, ny + 2 * nb, nz), storage_options=so
    )
    nf[nb : nx + 1 + nb, nb : ny + nb] = pf
    nf[:nb, nb : ny + nb] = nf[nx - 1 : nx + nb - 1, nb : ny + nb]
    nf[-nb:, nb : ny + nb] = nf[nb + 2 : 2 * nb + 2, nb : ny + nb]
    nf[:, :nb] = nf[:, ny - 1 : ny + nb - 1]
    nf[:, -nb:] = nf[:, nb + 1 : 2 * nb + 1]
    compare_arrays(hb.get_numerical_field(pf, field_name="at_u_locations"), nf)
    compare_arrays(hb.get_physical_field(nf, field_name="at_u_locations"), pf)

    # (nx, ny+1)
    pf = pfield[:-1, :]
    nf = zeros(
        backend, shape=(nx + 2 * nb, ny + 1 + 2 * nb, nz), storage_options=so
    )
    nf[nb : nx + nb, nb : ny + 1 + nb] = pf
    nf[:nb, nb : ny + 1 + nb] = nf[nx - 1 : nx + nb - 1, nb : ny + 1 + nb]
    nf[-nb:, nb : ny + 1 + nb] = nf[nb + 1 : 2 * nb + 1, nb : ny + 1 + nb]
    nf[:, :nb] = nf[:, ny - 1 : ny + nb - 1]
    nf[:, -nb:] = nf[:, nb + 2 : 2 * nb + 2]
    compare_arrays(hb.get_numerical_field(pf, field_name="at_v_locations"), nf)
    compare_arrays(hb.get_physical_field(nf, field_name="at_v_locations"), pf)

    # (nx+1, ny+1)
    pf = pfield
    nf = zeros(
        backend,
        shape=(nx + 1 + 2 * nb, ny + 1 + 2 * nb, nz),
        storage_options=so,
    )
    nf[nb : nx + 1 + nb, nb : ny + 1 + nb] = pf
    nf[:nb, nb : ny + 1 + nb] = nf[nx - 1 : nx + nb - 1, nb : ny + 1 + nb]
    nf[-nb:, nb : ny + 1 + nb] = nf[nb + 2 : 2 * nb + 2, nb : ny + 1 + nb]
    nf[:, :nb] = nf[:, ny - 1 : ny + nb - 1]
    nf[:, -nb:] = nf[:, nb + 2 : 2 * nb + 2]
    compare_arrays(
        hb.get_numerical_field(pf, field_name="at_u_locations_at_v_locations"),
        nf,
    )
    compare_arrays(
        hb.get_physical_field(nf, field_name="at_u_locations_at_v_locations"),
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

    nfield = data.draw(
        st_raw_field(
            (nx + 2 * nb + 1, ny + 2 * nb + 1, nz),
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

    # (nx, ny)
    nf = deepcopy(nfield[:-1, :-1])
    hb.enforce_field(nf)
    compare_arrays(nf[:nb, nb:-nb], nf[nx - 1 : nx - 1 + nb, nb:-nb])
    compare_arrays(nf[-nb:, nb:-nb], nf[nb + 1 : 2 * nb + 1, nb:-nb])
    compare_arrays(nf[:, :nb], nf[:, ny - 1 : ny - 1 + nb])
    compare_arrays(nf[:, -nb:], nf[:, nb + 1 : 2 * nb + 1])

    # (nx+1, ny)
    nf = deepcopy(nfield[:, :-1])
    hb.enforce_field(nf, field_name="at_u_locations")
    compare_arrays(nf[:nb, nb:-nb], nf[nx - 1 : nx - 1 + nb, nb:-nb])
    compare_arrays(nf[-nb:, nb:-nb], nf[nb + 2 : 2 * nb + 2, nb:-nb])
    compare_arrays(nf[:, :nb], nf[:, ny - 1 : ny - 1 + nb])
    compare_arrays(nf[:, -nb:], nf[:, nb + 1 : 2 * nb + 1])

    # (nx, ny+1)
    nf = deepcopy(nfield[:-1, :])
    hb.enforce_field(nf, field_name="at_v_locations")
    compare_arrays(nf[:nb, nb:-nb], nf[nx - 1 : nx - 1 + nb, nb:-nb])
    compare_arrays(nf[-nb:, nb:-nb], nf[nb + 1 : 2 * nb + 1, nb:-nb])
    compare_arrays(nf[:, :nb], nf[:, ny - 1 : ny - 1 + nb])
    compare_arrays(nf[:, -nb:], nf[:, nb + 2 : 2 * nb + 2])

    # (nx+1, ny+1)
    nf = nfield
    hb.enforce_field(nf, field_name="at_u_locations_at_v_locations")
    compare_arrays(nf[:nb, nb:-nb], nf[nx - 1 : nx - 1 + nb, nb:-nb])
    compare_arrays(nf[-nb:, nb:-nb], nf[nb + 2 : 2 * nb + 2, nb:-nb])
    compare_arrays(nf[:, :nb], nf[:, ny - 1 : ny - 1 + nb])
    compare_arrays(nf[:, -nb:], nf[:, nb + 2 : 2 * nb + 2])


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

    nfield = data.draw(
        st_raw_field(
            (nx + 2 * nb + 1, ny + 2 * nb + 1, nz),
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

    # (nx+1, ny)
    nf = deepcopy(nfield[:, :-1])
    hb.set_outermost_layers_x(nf)
    compare_arrays(nf[0, :], nf[-2, :])
    compare_arrays(nf[-1, :], nf[1, :])

    # (nx, ny+1)
    nf = deepcopy(nfield[:-1, :])
    hb.set_outermost_layers_y(nf)
    compare_arrays(nf[:, 0], nf[:, -2])
    compare_arrays(nf[:, -1], nf[:, 1])


if __name__ == "__main__":
    pytest.main([__file__])
