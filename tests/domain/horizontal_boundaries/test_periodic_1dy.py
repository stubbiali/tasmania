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
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import numpy as np
import pytest
from sympl import DataArray

import gt4py

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary

from tests.conf import backend as conf_backend, datatype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary_layers,
    st_one_of,
    st_physical_grid,
    st_raw_field,
)
from tests.utilities import compare_arrays, compare_dataarrays


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_properties(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)), label="grid"
    )
    ny = grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("periodic", 1, ny, nb)

    assert hb.nx == 1
    assert hb.ny == ny
    assert hb.nb == nb
    assert hb.ni == 2 * nb + 1
    assert hb.nj == ny + 2 * nb
    assert hb.type == "periodic"
    assert len(hb.kwargs) == 0


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_axis(data):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)), label="grid"
    )
    ny = grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("periodic", 1, ny, nb)

    #
    # get_numerical_axis
    #
    # mass points
    px = grid.y
    dx = px.values[1] - px.values[0]
    cx_val_values = np.array(tuple(px.values[0] + i * dx for i in range(-nb, ny + nb)))
    cx_val = DataArray(
        cx_val_values,
        coords={px.dims[0]: cx_val_values},
        dims=px.dims,
        attrs=px.attrs.copy(),
    )
    cx = hb.get_numerical_yaxis(px, dims=px.dims[0])
    compare_dataarrays(cx, cx_val)

    # staggered points
    px = grid.y_at_v_locations
    dx = px.values[1] - px.values[0]
    cx_val_values = np.array(
        tuple(px.values[0] + i * dx for i in range(-nb, ny + 1 + nb))
    )
    cx_val = DataArray(
        cx_val_values,
        coords={px.dims[0]: cx_val_values},
        dims=px.dims,
        attrs=px.attrs.copy(),
    )
    cx = hb.get_numerical_yaxis(px, dims=px.dims[0])
    compare_dataarrays(cx, cx_val)

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


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_field(data):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    if gt_powered:
        gt4py.storage.prepare_numpy()

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)), label="grid"
    )
    ny, nz = grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    pfield = data.draw(
        st_raw_field(
            (2, ny + 1, nz),
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "periodic", 1, ny, nb, gt_powered=gt_powered, backend=backend, dtype=dtype
    )

    # (1, ny)
    pf = pfield[:-1, :-1]
    cf = np.zeros((2 * nb + 1, ny + 2 * nb, nz), dtype=dtype)
    cf[nb : nb + 1, nb : ny + nb] = pf
    cf[nb : nb + 1, :nb] = cf[nb : nb + 1, ny - 1 : ny + nb - 1]
    cf[nb : nb + 1, -nb:] = cf[nb : nb + 1, nb + 1 : 2 * nb + 1]
    cf[:nb, :] = cf[nb : nb + 1, :]
    cf[-nb:, :] = cf[nb : nb + 1, :]
    compare_arrays(hb.get_numerical_field(pf), cf)
    compare_arrays(hb.get_physical_field(cf), pf)

    # (1, ny+1)
    pf = pfield[:-1, :]
    cf = np.zeros((2 * nb + 1, ny + 1 + 2 * nb, nz), dtype=dtype)
    cf[nb : nb + 1, nb : ny + 1 + nb] = pf
    cf[nb : nb + 1, :nb] = cf[nb : nb + 1, ny - 1 : ny + nb - 1]
    cf[nb : nb + 1, -nb:] = cf[nb : nb + 1, nb + 2 : 2 * nb + 2]
    cf[:nb, :] = cf[nb : nb + 1, :]
    cf[-nb:, :] = cf[nb : nb + 1, :]
    compare_arrays(hb.get_numerical_field(pf, field_name="at_v_locations"), cf)
    compare_arrays(hb.get_physical_field(cf, field_name="at_v_locations"), pf)

    # (2, ny)
    pf = pfield[:, :-1]
    cf = np.zeros((2 * nb + 2, ny + 2 * nb, nz), dtype=dtype)
    cf[nb : nb + 2, nb : ny + nb] = pf
    cf[nb : nb + 2, :nb] = cf[nb : nb + 2, ny - 1 : ny + nb - 1]
    cf[nb : nb + 2, -nb:] = cf[nb : nb + 2, nb + 1 : 2 * nb + 1]
    cf[:nb, :] = cf[nb : nb + 1, :]
    cf[-nb:, :] = cf[nb + 1 : nb + 2, :]
    compare_arrays(hb.get_numerical_field(pf, field_name="at_u_locations"), cf)
    compare_arrays(hb.get_physical_field(cf, field_name="at_u_locations"), pf)

    # (nx+1, 2)
    pf = pfield
    cf = np.zeros((2 * nb + 2, ny + 1 + 2 * nb, nz), dtype=dtype)
    cf[nb : nb + 2, nb : ny + 1 + nb] = pf
    cf[nb : nb + 2, :nb] = cf[nb : nb + 2, ny - 1 : ny + nb - 1]
    cf[nb : nb + 2, -nb:] = cf[nb : nb + 2, nb + 2 : 2 * nb + 2]
    cf[:nb, :] = cf[nb : nb + 1, :]
    cf[-nb:, :] = cf[nb + 1 : nb + 2, :]
    compare_arrays(
        hb.get_numerical_field(pf, field_name="at_u_locations_at_v_locations"), cf
    )
    compare_arrays(
        hb.get_physical_field(cf, field_name="at_u_locations_at_v_locations"), pf
    )


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_enforce(data):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    if gt_powered:
        gt4py.storage.prepare_numpy()

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)), label="grid"
    )
    ny, nz = grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    cfield = data.draw(
        st_raw_field(
            (2 * nb + 2, ny + 2 * nb + 1, nz),
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "periodic", 1, ny, nb, gt_powered=gt_powered, backend=backend, dtype=dtype
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


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_outermost_layers(data):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    if gt_powered:
        gt4py.storage.prepare_numpy()

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)), label="grid"
    )
    ny, nz = grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(1, ny), label="nb")

    cfield = data.draw(
        st_raw_field(
            (2 * nb + 2, ny + 2 * nb + 1, nz),
            -1e4,
            1e4,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "periodic", 1, ny, nb, gt_powered=gt_powered, backend=backend, dtype=dtype
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
