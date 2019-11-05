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
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    seed,
    settings,
    strategies as hyp_st,
)
import numpy as np
import pytest

import gridtools as gt

from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion as HD
from tasmania.python.utils.storage_utils import zeros

try:
    from .conf import (
        backend as conf_backend,
        default_origin as conf_dorigin,
        nb as conf_nb,
    )
    from .utils import compare_arrays, st_domain, st_floats, st_one_of, st_raw_field
except (ImportError, ModuleNotFoundError):
    from conf import (
        backend as conf_backend,
        default_origin as conf_dorigin,
        nb as conf_nb,
    )
    from utils import compare_arrays, st_domain, st_floats, st_one_of, st_raw_field


def assert_xyz(phi_tnd, phi_tnd_assert, nb):
    compare_arrays(phi_tnd_assert[nb:-nb, nb:-nb, :], phi_tnd[nb:-nb, nb:-nb, :])


def assert_xz(phi_tnd, phi_tnd_assert, nb):
    compare_arrays(phi_tnd_assert[nb:-nb, :, :], phi_tnd[nb:-nb, :, :])


def assert_yz(phi_tnd, phi_tnd_assert, nb):
    compare_arrays(phi_tnd_assert[:, nb:-nb, :], phi_tnd[:, nb:-nb, :])


def second_order_laplacian_x(dx, phi):
    out = deepcopy(phi)
    out[1:-1, :, :] = (phi[2:, :, :] - 2.0 * phi[1:-1, :, :] + phi[:-2, :, :]) / (dx * dx)
    return out


def second_order_laplacian_y(dy, phi):
    out = deepcopy(phi)
    out[:, 1:-1, :] = (phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]) / (dy * dy)
    return out


def second_order_diffusion_xyz(dx, dy, phi):
    return second_order_laplacian_x(dx, phi) + second_order_laplacian_y(dy, phi)


def second_order_diffusion_xz(dx, phi):
    return second_order_laplacian_x(dx, phi)


def second_order_diffusion_yz(dy, phi):
    return second_order_laplacian_y(dy, phi)


def second_order_validation(phi, grid, diffusion_depth, nb, backend, default_origin):
    ni, nj, nk = phi.shape
    dtype = phi.dtype
    phi_tnd = zeros((ni, nj, nk), backend, dtype, default_origin)

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
        dtype=phi.dtype,
        default_origin=default_origin,
        rebuild=False,
    )
    hd(phi, phi_tnd)

    gamma = hd._gamma  # np.tile(hd._gamma, (ni, nj, 1))

    if ni < 3:
        phi_tnd_assert = gamma * second_order_diffusion_yz(dy, phi)
        assert_yz(phi_tnd, phi_tnd_assert, nb)
    elif nj < 3:
        phi_tnd_assert = gamma * second_order_diffusion_xz(dx, phi)
        assert_xz(phi_tnd, phi_tnd_assert, nb)
    else:
        phi_tnd_assert = gamma * second_order_diffusion_xyz(dx, dy, phi)
        assert_xyz(phi_tnd, phi_tnd_assert, nb)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_second_order(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=1, max_value=max(1, conf_nb)))
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30), nb=nb
        ),
        label="grid",
    )
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

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

    depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz), label="depth")

    # ========================================
    # test
    # ========================================
    second_order_validation(phi, grid, depth, nb, backend, default_origin)


def fourth_order_laplacian_x(dx, phi):
    out = deepcopy(phi)
    out[2:-2, :, :] = (
        -phi[:-4, :, :]
        + 16.0 * phi[1:-3, :, :]
        - 30.0 * phi[2:-2, :, :]
        + 16.0 * phi[3:-1, :, :]
        - phi[4:, :, :]
    ) / (12.0 * dx * dx)
    return out


def fourth_order_laplacian_y(dy, phi):
    out = deepcopy(phi)
    out[:, 2:-2, :] = (
        -phi[:, :-4, :]
        + 16.0 * phi[:, 1:-3, :]
        - 30.0 * phi[:, 2:-2, :]
        + 16.0 * phi[:, 3:-1, :]
        - phi[:, 4:, :]
    ) / (12.0 * dy * dy)
    return out


def fourth_order_diffusion_xyz(dx, dy, phi):
    return fourth_order_laplacian_x(dx, phi) + fourth_order_laplacian_y(dy, phi)


def fourth_order_diffusion_xz(dx, phi):
    return fourth_order_laplacian_x(dx, phi)


def fourth_order_diffusion_yz(dy, phi):
    return fourth_order_laplacian_y(dy, phi)


def fourth_order_validation(phi, grid, diffusion_depth, nb, backend, default_origin):
    ni, nj, nk = phi.shape
    dtype = phi.dtype
    phi_tnd = zeros((ni, nj, nk), backend, dtype, default_origin)

    dx = grid.dx.values.item()
    dy = grid.dy.values.item()

    hd = HD.factory(
        "fourth_order",
        (ni, nj, nk),
        dx,
        dy,
        0.5,
        1.0,
        diffusion_depth,
        nb=nb,
        backend=backend,
        dtype=phi.dtype,
        default_origin=default_origin,
        rebuild=False,
    )
    hd(phi, phi_tnd)

    if ni < 5:
        phi_tnd_assert = hd._gamma * fourth_order_diffusion_yz(dy, phi)
        assert_yz(phi_tnd, phi_tnd_assert, nb)
    elif nj < 5:
        phi_tnd_assert = hd._gamma * fourth_order_diffusion_xz(dx, phi)
        assert_xz(phi_tnd, phi_tnd_assert, nb)
    else:
        phi_tnd_assert = hd._gamma * fourth_order_diffusion_xyz(dx, dy, phi)
        assert_xyz(phi_tnd, phi_tnd_assert, nb)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_fourth_order(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    nb = data.draw(hyp_st.integers(min_value=2, max_value=max(2, conf_nb)))
    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30), yaxis_length=(1, 30), zaxis_length=(1, 30), nb=nb
        ),
        label="grid",
    )
    grid = domain.numerical_grid

    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = grid.x.dtype
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

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
        label="cphi_rnd",
    )

    depth = data.draw(hyp_st.integers(min_value=0, max_value=grid.nz), label="depth")

    # ========================================
    # test
    # ========================================
    fourth_order_validation(phi, grid, depth, nb, backend, default_origin)


if __name__ == "__main__":
    pytest.main([__file__])
