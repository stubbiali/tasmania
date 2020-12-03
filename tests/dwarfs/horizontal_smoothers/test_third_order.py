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
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.dwarfs.horizontal_smoothing import (
    HorizontalSmoothing as HS,
)
from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.dwarfs.horizontal_smoothers.test_first_order import (
    assert_xyz,
    assert_xz,
    assert_yz,
)
from tests.strategies import st_domain, st_one_of, st_raw_field
from tests.utilities import hyp_settings


def third_order_smoothing_xyz(phi, g):
    ni, nj, nk = phi.shape

    i, j, k = slice(3, ni - 3), slice(3, nj - 3), slice(0, nk)
    im1, ip1 = slice(2, ni - 4), slice(4, ni - 2)
    im2, ip2 = slice(1, ni - 5), slice(5, ni - 1)
    im3, ip3 = slice(0, ni - 6), slice(6, ni)
    jm1, jp1 = slice(2, nj - 4), slice(4, nj - 2)
    jm2, jp2 = slice(1, nj - 5), slice(5, nj - 1)
    jm3, jp3 = slice(0, nj - 6), slice(6, nj)

    phi_smooth = deepcopy(phi)
    phi_smooth[i, j, k] = (1 - 0.625 * g[i, j, k]) * phi[
        i, j, k
    ] + 0.015625 * g[i, j, k] * (
        phi[im3, j, k]
        - 6.0 * phi[im2, j, k]
        + 15.0 * phi[im1, j, k]
        + phi[ip3, j, k]
        - 6.0 * phi[ip2, j, k]
        + 15.0 * phi[ip1, j, k]
        + phi[i, jm3, k]
        - 6.0 * phi[i, jm2, k]
        + 15.0 * phi[i, jm1, k]
        + phi[i, jp3, k]
        - 6.0 * phi[i, jp2, k]
        + 15.0 * phi[i, jp1, k]
    )

    return phi_smooth


def third_order_smoothing_xz(phi, g):
    ni, nj, nk = phi.shape

    i, j, k = slice(3, ni - 3), slice(0, nj), slice(0, nk)
    im1, ip1 = slice(2, ni - 4), slice(4, ni - 2)
    im2, ip2 = slice(1, ni - 5), slice(5, ni - 1)
    im3, ip3 = slice(0, ni - 6), slice(6, ni)

    phi_smooth = deepcopy(phi)
    phi_smooth[i, j, k] = (1 - 0.3125 * g[i, j, k]) * phi[
        i, j, k
    ] + 0.015625 * g[i, j, k] * (
        phi[im3, j, k]
        - 6.0 * phi[im2, j, k]
        + 15.0 * phi[im1, j, k]
        + phi[ip3, j, k]
        - 6.0 * phi[ip2, j, k]
        + 15.0 * phi[ip1, j, k]
    )

    return phi_smooth


def third_order_smoothing_yz(phi, g):
    ni, nj, nk = phi.shape

    i, j, k = slice(0, ni), slice(3, nj - 3), slice(0, nk)
    jm1, jp1 = slice(2, nj - 4), slice(4, nj - 2)
    jm2, jp2 = slice(1, nj - 5), slice(5, nj - 1)
    jm3, jp3 = slice(0, nj - 6), slice(6, nj)

    phi_smooth = deepcopy(phi)
    phi_smooth[i, j, k] = (1 - 0.3125 * g[i, j, k]) * phi[
        i, j, k
    ] + 0.015625 * g[i, j, k] * (
        phi[i, jm3, k]
        - 6.0 * phi[i, jm2, k]
        + 15.0 * phi[i, jm1, k]
        + phi[i, jp3, k]
        - 6.0 * phi[i, jp2, k]
        + 15.0 * phi[i, jp1, k]
    )

    return phi_smooth


def third_order_validation_xyz(
    phi, smooth_depth, nb, backend, backend_options, storage_options
):
    ni, nj, nk = phi.shape
    phi_new = zeros(backend, shape=phi.shape, storage_options=storage_options)

    hs = HS.factory(
        "third_order",
        (ni, nj, nk),
        0.5,
        1.0,
        smooth_depth,
        nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hs(phi, phi_new)

    gamma = hs._gamma

    phi_new_assert = third_order_smoothing_xyz(phi, gamma)
    assert_xyz(phi, phi_new, phi_new_assert, nb)


def third_order_validation_xz(
    phi, smooth_depth, nb, backend, backend_options, storage_options
):
    ni, nj, nk = phi.shape
    phi_new = zeros(backend, shape=phi.shape, storage_options=storage_options)

    hs = HS.factory(
        "third_order_1dx",
        (ni, nj, nk),
        0.5,
        1.0,
        smooth_depth,
        nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hs(phi, phi_new)

    gamma = hs._gamma

    phi_new_assert = third_order_smoothing_xz(phi, gamma)
    assert_xz(phi, phi_new, phi_new_assert, nb)


def third_order_validation_yz(
    phi, smooth_depth, nb, backend, backend_options, storage_options
):
    ni, nj, nk = phi.shape
    phi_new = zeros(backend, shape=phi.shape, storage_options=storage_options)

    hs = HS.factory(
        "third_order_1dy",
        (ni, nj, nk),
        0.5,
        1.0,
        smooth_depth,
        nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hs(phi, phi_new)

    gamma = hs._gamma

    phi_new_assert = third_order_smoothing_yz(phi, gamma)
    assert_yz(phi, phi_new, phi_new_assert, nb)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(
        hyp_st.integers(min_value=3, max_value=max(3, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 30),
            nb=nb,
            backend=backend,
            dtype=dtype,
        ),
        label="grid",
    )
    grid = domain.numerical_grid

    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnz")
    shape = (grid.nx + dnx, grid.ny + dny, grid.nz + dnz)

    phi = data.draw(
        st_raw_field(
            shape,
            min_value=1e-10,
            max_value=1e10,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="phi",
    )

    depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="depth"
    )

    # ========================================
    # test
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    third_order_validation_xyz(phi, depth, nb, backend, bo, so)
    third_order_validation_xz(phi, depth, nb, backend, bo, so)
    third_order_validation_yz(phi, depth, nb, backend, bo, so)


if __name__ == "__main__":
    pytest.main([__file__])
