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

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.grid import NumericalGrid
from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import StorageOptions

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
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(1, 1)),
        label="grid",
    )
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("dirichlet", grid, nb)

    assert hb.nx == nx
    assert hb.ny == 1
    assert hb.nb == nb
    assert hb.ni == nx
    assert hb.nj == 2 * nb + 1
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
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(1, 1)),
        label="grid",
    )
    nx, ny = grid.grid_xy.nx, grid.grid_xy.ny
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory("dirichlet", grid, nb)

    # numerical axes - mass points
    compare_dataarrays(hb.get_numerical_xaxis(dims=grid.x.dims[0]), grid.x)

    # numerical axes - staggered points
    compare_dataarrays(
        hb.get_numerical_xaxis_staggered(dims=grid.x_at_u_locations.dims[0]),
        grid.x_at_u_locations,
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
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(1, 1)),
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
        "dirichlet", grid, nb, backend=backend, storage_options=so
    )

    # (nx, 1)
    pf = pfield[:-1, :-1]
    nf = hb.get_numerical_field(pf)
    compare_arrays(nf, pf)
    compare_arrays(hb.get_physical_field(nf), pf)

    # (nx+1, 1)
    pf = pfield[:, :-1]
    nf = hb.get_numerical_field(pf, field_name="at_u_locations")
    compare_arrays(nf, pf)
    compare_arrays(hb.get_physical_field(nf, field_name="at_u_locations"), pf)

    # (nx, 2)
    pf = pfield[:-1, :]
    nf = hb.get_numerical_field(pf, field_name="at_v_locations")
    compare_arrays(nf[:, : nb + 1], pf[:, :1])
    compare_arrays(nf[:, -nb - 1 :], pf[:, -1:])
    compare_arrays(hb.get_physical_field(nf, field_name="at_v_locations"), pf)

    # (nx+1, ny+1)
    pf = pfield
    nf = hb.get_numerical_field(pf, field_name="at_uv_locations")
    compare_arrays(nf[:, : nb + 1], pf[:, :1])
    compare_arrays(nf[:, -nb - 1 :], pf[:, -1:])
    compare_arrays(hb.get_physical_field(nf, field_name="at_uv_locations"), pf)


def enforce(nf_val, hb):
    nx, ny, nb = hb.nx, hb.ny, hb.nb

    nf_val[:nb, nb:-nb] = np.pi
    nf_val[-nb:, nb:-nb] = np.pi
    nf_val[:, :nb] = nf_val[:, nb : nb + 1]
    nf_val[:, -nb:] = nf_val[:, -nb - 1 : -nb]


def validation(nf, nf_val, hb):
    nb = hb.nb

    compare_arrays(nf[nb:-nb, nb:-nb], nf_val[nb:-nb, nb:-nb])
    compare_arrays(nf[:nb, nb:-nb], nf_val[:nb, nb:-nb])
    compare_arrays(nf[-nb:, nb:-nb], nf_val[-nb:, nb:-nb])
    compare_arrays(nf[:, :nb], nf_val[:, nb : nb + 1])
    compare_arrays(nf[:, -nb:], nf_val[:, -nb - 1 : -nb])


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
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(1, 1)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    storage_shape = (nx + 1, ny + 2 * nb + 1, nz + 1)
    nfield = data.draw(
        st_raw_field(
            storage_shape, -1e4, 1e4, backend=backend, storage_options=so
        )
    )

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "dirichlet",
        grid,
        nb,
        backend=backend,
        storage_options=so,
        core=pi_function,
    )

    # (nx, 1)
    nf = deepcopy(nfield)
    hb.enforce_field(nf)
    nf_val = deepcopy(nfield[:-1, :-1])
    enforce(nf_val, hb)
    validation(nf[:-1, :-1], nf_val, hb)

    # (nx+1, 1)
    nf = deepcopy(nfield)
    hb.enforce_field(
        nf, field_name="afield_at_u_locations_on_interface_levels"
    )
    nf_val = deepcopy(nfield[:, :-1])
    enforce(nf_val, hb)
    validation(nf[:, :-1], nf_val, hb)

    # (nx, 2)
    nf = deepcopy(nfield)
    hb.enforce_field(nf, field_name="afield_at_v_locations")
    nf_val = deepcopy(nfield[:-1, :])
    enforce(nf_val, hb)
    validation(nf[:-1, :], nf_val, hb)

    # (nx+1, 2)
    nf = deepcopy(nfield)
    hb.enforce_field(nf, field_name="afield_at_uv_locations")
    nf_val = nfield
    enforce(nf_val, hb)
    validation(nf, nf_val, hb)


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
        st_physical_grid(xaxis_length=(2, None), yaxis_length=(1, 1)),
        label="grid",
    )
    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz
    nb = data.draw(st_horizontal_boundary_layers(nx, ny), label="nb")

    storage_shape = (nx + 1, ny + 2 * nb + 1, nz + 1)
    nfield = zeros(backend, shape=storage_shape, storage_options=so)

    # ========================================
    # test
    # ========================================
    hb = HorizontalBoundary.factory(
        "dirichlet",
        grid,
        nb,
        backend=backend,
        storage_options=so,
        core=pi_function,
    )

    # (nx+1, 1)
    nf = deepcopy(nfield)
    hb.set_outermost_layers_x(nf, field_name="afield_at_u_locations")
    compare_arrays(
        nf[0, :-1], np.pi * np.ones((ny + 2 * nb, nz + 1), dtype=dtype)
    )
    compare_arrays(
        nf[-1, :-1], np.pi * np.ones((ny + 2 * nb, nz + 1), dtype=dtype)
    )

    # (nx, 2)
    nf = deepcopy(nfield)
    hb.set_outermost_layers_y(nf, field_name="afield_at_v_locations")
    compare_arrays(nf[:-1, 0], np.pi * np.ones((nx, nz + 1), dtype=dtype))
    compare_arrays(nf[:-1, -1], np.pi * np.ones((nx, nz + 1), dtype=dtype))

    # (nx+1, 2)
    nf = deepcopy(nfield)
    hb.set_outermost_layers_x(nf, field_name="afield_at_uv_locations")
    hb.set_outermost_layers_y(nf, field_name="afield_at_uv_locations")
    compare_arrays(
        nf[0, :], np.pi * np.ones((ny + 2 * nb + 1, nz + 1), dtype=dtype)
    )
    compare_arrays(
        nf[-1, :], np.pi * np.ones((ny + 2 * nb + 1, nz + 1), dtype=dtype)
    )
    compare_arrays(nf[:, 0], np.pi * np.ones((nx + 1, nz + 1), dtype=dtype))
    compare_arrays(nf[:, -1], np.pi * np.ones((nx + 1, nz + 1), dtype=dtype))


if __name__ == "__main__":
    pytest.main([__file__])
