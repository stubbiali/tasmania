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

from tasmania.python.dwarfs.horizontal_diffusion import (
    HorizontalDiffusion as HD,
)
from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.strategies import st_domain, st_one_of, st_raw_field
from tests.utilities import compare_arrays, hyp_settings


def assert_xyz(phi_tnd, phi_tnd_assert, nb):
    compare_arrays(
        phi_tnd_assert[nb:-nb, nb:-nb, :], phi_tnd[nb:-nb, nb:-nb, :]
    )


def assert_xz(phi_tnd, phi_tnd_assert, nb):
    compare_arrays(phi_tnd_assert[nb:-nb, :, :], phi_tnd[nb:-nb, :, :])


def assert_yz(phi_tnd, phi_tnd_assert, nb):
    compare_arrays(phi_tnd_assert[:, nb:-nb, :], phi_tnd[:, nb:-nb, :])


def second_order_laplacian_x(dx, phi):
    out = deepcopy(phi)
    out[1:-1, :, :] = (
        phi[2:, :, :] - 2.0 * phi[1:-1, :, :] + phi[:-2, :, :]
    ) / (dx * dx)
    return out


def second_order_laplacian_y(dy, phi):
    out = deepcopy(phi)
    out[:, 1:-1, :] = (
        phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]
    ) / (dy * dy)
    return out


def second_order_diffusion_xyz(dx, dy, phi):
    return second_order_laplacian_x(dx, phi) + second_order_laplacian_y(
        dy, phi
    )


def second_order_diffusion_xz(dx, phi):
    return second_order_laplacian_x(dx, phi)


def second_order_diffusion_yz(dy, phi):
    return second_order_laplacian_y(dy, phi)


def second_order_validation_xyz(
    phi, grid, diffusion_depth, nb, backend, backend_options, storage_options
):
    ni, nj, nk = phi.shape
    phi_tnd = zeros(
        backend, shape=(ni, nj, nk), storage_options=storage_options
    )

    dx = grid.dx.values.item()
    dy = grid.dy.values.item()

    hd = HD.factory(
        "second_order",
        (ni, nj, nk),
        dx,
        dy,
        0.5,
        1.0,
        diffusion_depth,
        nb=nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hd(phi, phi_tnd)

    gamma = hd._gamma  # np.tile(hd._gamma, (ni, nj, 1))

    phi_tnd_assert = gamma * second_order_diffusion_xyz(dx, dy, phi)
    assert_xyz(phi_tnd, phi_tnd_assert, nb)


def second_order_validation_xz(
    phi, grid, diffusion_depth, nb, backend, backend_options, storage_options
):
    ni, nj, nk = phi.shape
    phi_tnd = zeros(
        backend, shape=(ni, nj, nk), storage_options=storage_options
    )

    dx = grid.dx.values.item()
    dy = grid.dy.values.item()

    hd = HD.factory(
        "second_order_1dx",
        (ni, nj, nk),
        dx,
        dy,
        0.5,
        1.0,
        diffusion_depth,
        nb=nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hd(phi, phi_tnd)

    gamma = hd._gamma  # np.tile(hd._gamma, (ni, nj, 1))

    phi_tnd_assert = gamma * second_order_diffusion_xz(dx, phi)
    assert_xz(phi_tnd, phi_tnd_assert, nb)


def second_order_validation_yz(
    phi, grid, diffusion_depth, nb, backend, backend_options, storage_options
):
    ni, nj, nk = phi.shape
    phi_tnd = zeros(
        backend, shape=(ni, nj, nk), storage_options=storage_options
    )

    dx = grid.dx.values.item()
    dy = grid.dy.values.item()

    hd = HD.factory(
        "second_order_1dy",
        (ni, nj, nk),
        dx,
        dy,
        0.5,
        1.0,
        diffusion_depth,
        nb=nb,
        backend=backend,
        backend_options=backend_options,
        storage_options=storage_options,
    )
    hd(phi, phi_tnd)

    gamma = hd._gamma  # np.tile(hd._gamma, (ni, nj, 1))

    phi_tnd_assert = gamma * second_order_diffusion_yz(dy, phi)
    assert_yz(phi_tnd, phi_tnd_assert, nb)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)))
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
            min_value=-1e10,
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

    second_order_validation_xyz(phi, grid, depth, nb, backend, bo, so)
    second_order_validation_xz(phi, grid, depth, nb, backend, bo, so)
    second_order_validation_yz(phi, grid, depth, nb, backend, bo, so)


if __name__ == "__main__":
    pytest.main([__file__])
