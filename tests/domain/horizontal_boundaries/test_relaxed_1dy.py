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

import gt4py as gt

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary

from tests.conf import backend as conf_backend, datatype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary_kwargs,
    st_horizontal_boundary_layers,
    st_one_of,
    st_physical_grid,
    st_raw_field,
    st_state,
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
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb), label="hb_kwargs"
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("relaxed", nx, ny, nb, **hb_kwargs)

    assert hb.nx == 1
    assert hb.ny == ny
    assert hb.nb == nb
    assert hb.ni == 2 * nb + 1
    assert hb.nj == ny
    assert hb.type == "relaxed"
    assert "nr" in hb.kwargs
    assert "nz" in hb.kwargs
    assert len(hb.kwargs) == 2


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
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb), label="hb_kwargs"
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("relaxed", nx, ny, nb, **hb_kwargs)

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
        gt.storage.prepare_numpy()

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)), label="grid"
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb), label="hb_kwargs"
    )

    pfield = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz),
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
        "relaxed",
        nx,
        ny,
        nb,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        **hb_kwargs
    )

    # (1, ny)
    pf = pfield[:-1, :-1]
    cf = hb.get_numerical_field(pf)
    compare_arrays(cf, pf)
    compare_arrays(hb.get_physical_field(cf), pf)

    # (1, ny+1)
    pf = pfield[:-1, :]
    cf = hb.get_numerical_field(pf, field_name="at_v_locations")
    compare_arrays(cf, pf)
    compare_arrays(hb.get_physical_field(cf, field_name="at_v_locations"), pf)

    # (2, ny)
    pf = pfield[:, :-1]
    cf = hb.get_numerical_field(pf, field_name="at_u_locations")
    compare_arrays(cf[: nb + 1, :], pf[:1, :])
    compare_arrays(cf[nb + 1 :, :], pf[1:, :])
    compare_arrays(hb.get_physical_field(cf, field_name="at_u_locations"), pf)

    # (2, ny+1)
    pf = pfield
    cf = hb.get_numerical_field(pf, field_name="at_uv_locations")
    compare_arrays(cf[: nb + 1, :], pf[:1, :])
    compare_arrays(cf[nb + 1 :, :], pf[1:, :])
    compare_arrays(hb.get_physical_field(cf, field_name="at_uv_locations"), pf)


def enforce(cf_val, cf_ref, hb):
    ny, nb, nr = hb.ny, hb.nb, hb.kwargs["nr"]
    mi, mj, mk = cf_val.shape

    i, k = slice(nb, nb + 2 if mi == 2 * nb + 2 else nb + 1), slice(0, mk)
    cf_val[i, :, k] -= hb._gamma[i, :mj, k] * (cf_val[i, :, k] - cf_ref[i, :mj, k])
    cf_val[:nb, :] = cf_val[nb : nb + 1, :]
    cf_val[-nb:, :] = cf_val[-nb - 1 : -nb, :]


def validation(cf, cf_val, hb):
    nb, nr = hb.nb, hb.kwargs["nr"]

    compare_arrays(cf[nb:-nb, :nr], cf_val[nb:-nb, :nr])
    compare_arrays(cf[nb:-nb, nr:-nr], cf_val[nb:-nb, nr:-nr])
    compare_arrays(cf[nb:-nb, -nr:], cf_val[nb:-nb, -nr:])
    compare_arrays(cf[:nb, :], cf_val[nb : nb + 1, :])
    compare_arrays(cf[-nb:, :], cf_val[-nb - 1 : -nb, :])


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
        gt.storage.prepare_numpy()

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)), label="grid"
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb, nz=nz), label="hb_kwargs"
    )

    storage_shape = (nx + 2 * nb + 1, ny + 1, nz + 1)
    cfield = data.draw(
        st_raw_field(
            storage_shape, -1e4, 1e4, gt_powered=gt_powered, backend=backend, dtype=dtype
        )
    )

    ref_state = data.draw(
        st_state(
            grid, gt_powered=gt_powered, backend=backend, storage_shape=storage_shape
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "relaxed",
        nx,
        ny,
        nb,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        **hb_kwargs
    )
    hb.reference_state = ref_state

    # (1, ny)
    cf = deepcopy(cfield)
    units = ref_state["afield"].attrs["units"]
    hb.enforce_field(cf, field_name="afield", field_units=units)
    cf_val = deepcopy(cfield[:-1, :-1, :-1])
    cf_ref = ref_state["afield"].values
    enforce(cf_val, cf_ref, hb)
    validation(cf[:-1, :-1, :-1], cf_val, hb)

    # (1, ny+1)
    cf = deepcopy(cfield)
    units = ref_state["afield_at_v_locations"].attrs["units"]
    hb.enforce_field(cf, field_name="afield_at_v_locations", field_units=units)
    cf_val = deepcopy(cfield[:-1, :, :-1])
    cf_ref = ref_state["afield_at_v_locations"].values
    enforce(cf_val, cf_ref, hb)
    validation(cf[:-1, :, :-1], cf_val, hb)

    # (2, ny)
    cf = deepcopy(cfield)
    units = ref_state["afield_at_u_locations"].attrs["units"]
    hb.enforce_field(cf, field_name="afield_at_u_locations", field_units=units)
    cf_val = deepcopy(cfield[:, :-1, :-1])
    cf_ref = ref_state["afield_at_u_locations"].values
    enforce(cf_val, cf_ref, hb)
    validation(cf[:, :-1, :-1], cf_val, hb)

    # (2, ny+1)
    cf = deepcopy(cfield)
    units = ref_state["afield_at_uv_locations"].attrs["units"]
    hb.enforce_field(cf, field_name="afield_at_uv_locations", field_units=units)
    cf_val = cfield[:, :, :-1]
    cf_ref = ref_state["afield_at_uv_locations"].values
    enforce(cf_val, cf_ref, hb)
    validation(cf[:, :, :-1], cf_val, hb)


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
        gt.storage.prepare_numpy()

    grid = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(2, None)), label="grid"
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb), label="hb_kwargs"
    )

    storage_shape = (nx + 2 * nb + 1, ny + 1, nz + 1)
    cfield = np.zeros(storage_shape, dtype=dtype)

    ref_state = data.draw(
        st_state(
            grid, gt_powered=gt_powered, backend=backend, storage_shape=storage_shape
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "relaxed",
        nx,
        ny,
        nb,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        **hb_kwargs
    )
    hb.reference_state = ref_state

    # (1, ny+1)
    cf = deepcopy(cfield)
    units = ref_state["afield_at_v_locations"].attrs["units"]
    hb.set_outermost_layers_y(cf, field_name="afield_at_v_locations", field_units=units)
    cf_ref = ref_state["afield_at_v_locations"].values
    compare_arrays(cf[:-1, 0], cf_ref[:-1, 0])
    compare_arrays(cf[:-1, -1], cf_ref[:-1, -1])

    # (2, ny)
    cf = deepcopy(cfield)
    units = ref_state["afield_at_u_locations"].attrs["units"]
    hb.set_outermost_layers_x(cf, field_name="afield_at_u_locations", field_units=units)
    cf_ref = ref_state["afield_at_u_locations"].values
    compare_arrays(cf[0, :-1], cf_ref[0, :-1])
    compare_arrays(cf[-1, :-1], cf_ref[-1, :-1])

    # (2, ny+1)
    cf = deepcopy(cfield)
    units = ref_state["afield_at_uv_locations"].attrs["units"]
    hb.set_outermost_layers_x(cf, field_name="afield_at_uv_locations", field_units=units)
    hb.set_outermost_layers_y(cf, field_name="afield_at_uv_locations", field_units=units)
    cf_ref = ref_state["afield_at_uv_locations"].values
    compare_arrays(cf[0, :], cf_ref[0, :])
    compare_arrays(cf[-1, :], cf_ref[-1, :])
    compare_arrays(cf[:, 0], cf_ref[:, 0])
    compare_arrays(cf[:, -1], cf_ref[:, -1])


if __name__ == "__main__":
    pytest.main([__file__])
