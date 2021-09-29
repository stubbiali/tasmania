# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions

from tests.conf import (
    aligned_index as conf_aligned_index,
    backend as conf_backend,
    dtype as conf_dtype,
    nb as conf_nb,
)
from tests.strategies import st_domain, st_one_of, st_raw_field
from tests.utilities import compare_arrays, hyp_settings


def assert_xyz(phi, phi_new, phi_new_assert, nb):
    compare_arrays(
        phi_new_assert[nb:-nb, nb:-nb, :], phi_new[nb:-nb, nb:-nb, :]
    )
    compare_arrays(phi[:nb, :, :], phi_new[:nb, :, :])
    compare_arrays(phi[-nb:, :, :], phi_new[-nb:, :, :])
    compare_arrays(phi[nb:-nb, :nb, :], phi_new[nb:-nb, :nb, :])
    compare_arrays(phi[nb:-nb, -nb:, :], phi_new[nb:-nb, -nb:, :])


def assert_xz(phi, phi_new, phi_new_assert, nb):
    compare_arrays(phi_new_assert[nb:-nb, :, :], phi_new[nb:-nb, :, :])
    compare_arrays(phi[:nb, :, :], phi_new[:nb, :, :])
    compare_arrays(phi[-nb:, :, :], phi_new[-nb:, :, :])


def assert_yz(phi, phi_new, phi_new_assert, nb):
    compare_arrays(phi_new_assert[:, nb:-nb, :], phi_new[:, nb:-nb, :])
    compare_arrays(phi[:, :nb, :], phi_new[:, :nb, :])
    compare_arrays(phi[:, -nb:, :], phi_new[:, -nb:, :])


def first_order_smoothing_xyz(phi, g):
    ni, nj, nk = phi.shape

    i, j, k = slice(1, ni - 1), slice(1, nj - 1), slice(0, nk)
    im1, ip1 = slice(0, ni - 2), slice(2, ni)
    jm1, jp1 = slice(0, nj - 2), slice(2, nj)

    phi_smooth = deepcopy(phi)
    phi_smooth[i, j, k] = (1 - g[i, j, k]) * phi[i, j, k] + 0.25 * g[
        i, j, k
    ] * (phi[ip1, j, k] + phi[im1, j, k] + phi[i, jm1, k] + phi[i, jp1, k])

    return phi_smooth


def first_order_smoothing_xz(phi, g):
    ni, nj, nk = phi.shape

    i, j, k = slice(1, ni - 1), slice(0, nj), slice(0, nk)
    im1, ip1 = slice(0, ni - 2), slice(2, ni)

    phi_smooth = deepcopy(phi)
    phi_smooth[i, j, k] = (1 - 0.5 * g[i, j, k]) * phi[i, j, k] + 0.25 * g[
        i, j, k
    ] * (phi[ip1, j, k] + phi[im1, j, k])

    return phi_smooth


def first_order_smoothing_yz(phi, g):
    ni, nj, nk = phi.shape

    i, j, k = slice(0, ni), slice(1, nj - 1), slice(0, nk)
    jm1, jp1 = slice(0, nj - 2), slice(2, nj)

    phi_smooth = deepcopy(phi)
    phi_smooth[i, j, k] = (1 - 0.5 * g[i, j, k]) * phi[i, j, k] + 0.25 * g[
        i, j, k
    ] * (phi[i, jm1, k] + phi[i, jp1, k])

    return phi_smooth


def first_order_validation_xyz(
    phi, smooth_depth, nb, backend, backend_options, storage_options
):
    phi_new = zeros(backend, shape=phi.shape, storage_options=storage_options)

    hs = HS.factory(
        "first_order",
        phi.shape,
        0.5,
        1.0,
        smooth_depth,
        nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hs(phi, phi_new)

    phi_new_assert = first_order_smoothing_xyz(
        to_numpy(phi), to_numpy(hs._gamma)
    )
    assert_xyz(phi, phi_new, phi_new_assert, nb)


def first_order_validation_xz(
    phi, smooth_depth, nb, backend, backend_options, storage_options
):
    phi_new = zeros(backend, shape=phi.shape, storage_options=storage_options)

    hs = HS.factory(
        "first_order_1dx",
        phi.shape,
        0.5,
        1.0,
        smooth_depth,
        nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hs(phi, phi_new)

    phi_new_assert = first_order_smoothing_xz(
        to_numpy(phi), to_numpy(hs._gamma)
    )
    assert_xz(phi, phi_new, phi_new_assert, nb)


def first_order_validation_yz(
    phi, smooth_depth, nb, backend, backend_options, storage_options
):
    phi_new = zeros(backend, shape=phi.shape, storage_options=storage_options)

    hs = HS.factory(
        "first_order_1dy",
        phi.shape,
        0.5,
        1.0,
        smooth_depth,
        nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hs(phi, phi_new)

    phi_new_assert = first_order_smoothing_yz(
        to_numpy(phi), to_numpy(hs._gamma)
    )
    assert_yz(phi, phi_new, phi_new_assert, nb)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf_aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False, cache=True, check_rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    nb = data.draw(
        hyp_st.integers(min_value=1, max_value=max(1, conf_nb)), label="nb"
    )
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 30),
            nb=nb,
            backend=backend,
            backend_options=bo,
            storage_options=so,
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
            storage_options=so,
        ),
        label="phi",
    )

    depth = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz), label="depth"
    )

    # ========================================
    # test
    # ========================================
    first_order_validation_xyz(phi, depth, nb, backend, bo, so)
    first_order_validation_xz(phi, depth, nb, backend, bo, so)
    first_order_validation_yz(phi, depth, nb, backend, bo, so)


if __name__ == "__main__":
    pytest.main([__file__])
