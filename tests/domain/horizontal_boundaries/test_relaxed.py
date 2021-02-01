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
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory

from tests.conf import backend as conf_backend, dtype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary_kwargs,
    st_horizontal_boundary_layers,
    st_physical_grid,
    st_raw_field,
    st_state,
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
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb), label="hb_kwargs"
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("relaxed", grid, nb, **hb_kwargs)

    assert hb.nx == nx
    assert hb.ny == ny
    assert hb.nb == nb
    assert hb.ni == nx
    assert hb.nj == ny
    assert hb.type == "relaxed"
    assert "nr" in hb.kwargs
    assert len(hb.kwargs) == 1


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
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb), label="hb_kwargs"
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("relaxed", grid, nb, **hb_kwargs)

    # numerical axes - mass points
    compare_dataarrays(hb.get_numerical_xaxis(grid.x.dims[0]), grid.x)
    compare_dataarrays(hb.get_numerical_yaxis(grid.y.dims[0]), grid.y)

    # numerical axes - staggered points
    compare_dataarrays(
        hb.get_numerical_xaxis_staggered(grid.x_at_u_locations.dims[0]),
        grid.x_at_u_locations,
    )
    compare_dataarrays(
        hb.get_numerical_yaxis_staggered(grid.y_at_v_locations.dims[0]),
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
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb),
        label="hb_kwargs",
    )

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
        "relaxed", grid, nb, backend=backend, storage_options=so, **hb_kwargs
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


def enforce(stencil_irelax, nf_val, nf_ref, hb):
    mi, mj, mk = nf_val.shape
    stencil_irelax(
        in_gamma=hb._gamma,
        in_phi_ref=nf_ref,
        inout_phi=nf_val,
        origin=(0, 0, 0),
        domain=(mi, mj, mk),
    )


def validation(nf, nf_val, hb):
    nr = hb.kwargs["nr"]

    compare_arrays(nf[nr:-nr, nr:-nr], nf_val[nr:-nr, nr:-nr])
    compare_arrays(nf[:nr, :nr], nf_val[:nr, :nr])
    compare_arrays(nf[:nr, nr:-nr], nf_val[:nr, nr:-nr])
    compare_arrays(nf[:nr, -nr:], nf_val[:nr, -nr:])
    compare_arrays(nf[-nr:, :nr], nf_val[-nr:, :nr])
    compare_arrays(nf[-nr:, nr:-nr], nf_val[-nr:, nr:-nr])
    compare_arrays(nf[-nr:, -nr:], nf_val[-nr:, -nr:])
    compare_arrays(nf[nr:-nr, :nr], nf_val[nr:-nr, :nr])
    compare_arrays(nf[nr:-nr, -nr:], nf_val[nr:-nr, -nr:])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_enforce(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype)

    grid = data.draw(
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb),
        label="hb_kwargs",
    )

    dnx = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=1, max_value=3), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    nfield = data.draw(
        st_raw_field(
            storage_shape, -1e4, 1e4, backend=backend, storage_options=so
        )
    )

    ref_state = data.draw(
        st_state(grid, backend=backend, storage_shape=storage_shape,)
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "relaxed",
        grid,
        nb,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
        **hb_kwargs
    )
    hb.reference_state = ref_state

    sf = StencilFactory(
        backend=backend, backend_options=bo, storage_options=so
    )
    stencil_irelax = sf.compile("irelax")

    # (nx, ny)
    nf_val = sf.as_storage(data=nfield)
    units = ref_state["afield"].attrs["units"]
    hb.enforce_field(nfield, field_name="afield", field_units=units)
    nf_ref = ref_state["afield"].data
    enforce(stencil_irelax, nf_val, nf_ref, hb)
    validation(nfield[:nx, :ny, :nz], nf_val[:nx, :ny, :nz], hb)

    # (nx+1, ny)
    nf_val = sf.as_storage(data=nfield)
    units = ref_state["afield_at_u_locations"].attrs["units"]
    hb.enforce_field(
        nfield, field_name="afield_at_u_locations", field_units=units
    )
    nf_ref = ref_state["afield_at_u_locations"].data
    enforce(stencil_irelax, nf_val, nf_ref, hb)
    validation(nfield[: nx + 1, :ny, :nz], nf_val[: nx + 1, :ny, :nz], hb)

    # (nx, ny+1)
    nf_val = sf.as_storage(data=nfield)
    units = ref_state["afield_at_v_locations"].attrs["units"]
    hb.enforce_field(
        nfield, field_name="afield_at_v_locations", field_units=units
    )
    nf_ref = ref_state["afield_at_v_locations"].data
    enforce(stencil_irelax, nf_val, nf_ref, hb)
    validation(nfield[:nx, : ny + 1, :nz], nf_val[:nx, : ny + 1, :nz], hb)

    # (nx+1, ny+1)
    nf_val = sf.as_storage(data=nfield)
    units = ref_state["afield_at_uv_locations"].attrs["units"]
    hb.enforce_field(
        nfield, field_name="afield_at_uv_locations", field_units=units
    )
    nf_ref = ref_state["afield_at_uv_locations"].data
    enforce(stencil_irelax, nf_val, nf_ref, hb)
    validation(
        nfield[: nx + 1, : ny + 1, :nz], nf_val[: nx + 1, : ny + 1, :nz], hb
    )


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_outermost_layers(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype)

    grid = data.draw(
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(2, None)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")
    hb_kwargs = data.draw(
        st_horizontal_boundary_kwargs("relaxed", nx, ny, nb),
        label="hb_kwargs",
    )
    storage_shape = (nx + 1, ny + 1, nz + 1)
    ref_state = data.draw(
        st_state(
            grid,
            backend=backend,
            storage_shape=storage_shape,
            storage_options=so,
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "relaxed",
        grid,
        nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
        **hb_kwargs
    )
    hb.reference_state = ref_state

    sf = StencilFactory(
        backend=backend, backend_options=bo, storage_options=so
    )

    # (nx+1, ny)
    nfield = sf.zeros(shape=storage_shape)
    units = ref_state["afield_at_u_locations"].attrs["units"]
    hb.set_outermost_layers_x(
        nfield, field_name="afield_at_u_locations", field_units=units
    )
    nf_ref = ref_state["afield_at_u_locations"].data
    compare_arrays(nfield[0, :-1], nf_ref[0, :-1])
    compare_arrays(nfield[-1, :-1], nf_ref[-1, :-1])

    # (nx, ny+1)
    nfield = sf.zeros(shape=storage_shape)
    units = ref_state["afield_at_v_locations"].attrs["units"]
    hb.set_outermost_layers_y(
        nfield, field_name="afield_at_v_locations", field_units=units
    )
    nf_ref = ref_state["afield_at_v_locations"].data
    compare_arrays(nfield[:-1, 0], nf_ref[:-1, 0])
    compare_arrays(nfield[:-1, -1], nf_ref[:-1, -1])

    # (nx+1, ny+1)
    nfield = sf.zeros(shape=storage_shape)
    units = ref_state["afield_at_uv_locations"].attrs["units"]
    hb.set_outermost_layers_x(
        nfield, field_name="afield_at_uv_locations", field_units=units
    )
    hb.set_outermost_layers_y(
        nfield, field_name="afield_at_uv_locations", field_units=units
    )
    nf_ref = ref_state["afield_at_uv_locations"].data
    compare_arrays(nfield[0, :], nf_ref[0, :])
    compare_arrays(nfield[-1, :], nf_ref[-1, :])
    compare_arrays(nfield[:, 0], nf_ref[:, 0])
    compare_arrays(nfield[:, -1], nf_ref[:, -1])


if __name__ == "__main__":
    pytest.main([__file__])
