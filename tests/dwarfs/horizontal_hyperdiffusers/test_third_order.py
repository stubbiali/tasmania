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
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.dwarfs.horizontal_hyperdiffusion import (
    HorizontalHyperDiffusion as HHD,
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
from tests.dwarfs.horizontal_hyperdiffusers.test_first_order import (
    assert_xyz,
    assert_xz,
    assert_yz,
    laplacian_x,
    laplacian_y,
    laplacian2d,
)
from tests.strategies import st_domain, st_one_of, st_raw_field
from tests.utilities import hyp_settings


def third_order_diffusion_xyz(dx, dy, phi):
    lap0 = laplacian2d(dx, dy, phi)
    lap1 = laplacian2d(dx, dy, lap0)
    lap2 = laplacian2d(dx, dy, lap1)
    return lap2


def third_order_diffusion_xz(dx, phi):
    lap0 = laplacian_x(dx, phi)
    lap1 = laplacian_x(dx, lap0)
    lap2 = laplacian_x(dx, lap1)
    return lap2


def third_order_diffusion_yz(dy, phi):
    lap0 = laplacian_y(dy, phi)
    lap1 = laplacian_y(dy, lap0)
    lap2 = laplacian_y(dy, lap1)
    return lap2


def third_order_validation_xyz(
    phi, grid, diffusion_depth, nb, backend, bo, so
):
    ni, nj, nk = phi.shape
    phi_tnd = zeros(backend, shape=(ni, nj, nk), storage_options=so)

    dx = grid.dx.values.item()
    dy = grid.dy.values.item()

    hhd = HHD.factory(
        "third_order",
        (ni, nj, nk),
        dx,
        dy,
        0.5,
        1.0,
        diffusion_depth,
        nb=nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    hhd(phi, phi_tnd)

    gamma = to_numpy(hhd._gamma)  # np.tile(hd._gamma, (ni, nj, 1))
    phi_tnd_assert = gamma * third_order_diffusion_xyz(dx, dy, to_numpy(phi))

    assert_xyz(phi_tnd, phi_tnd_assert, nb)


def third_order_validation_xz(phi, grid, diffusion_depth, nb, backend, bo, so):
    ni, nj, nk = phi.shape
    phi_tnd = zeros(backend, shape=(ni, nj, nk), storage_options=so)

    dx = grid.dx.values.item()
    dy = grid.dy.values.item()

    hhd = HHD.factory(
        "third_order_1dx",
        (ni, nj, nk),
        dx,
        dy,
        0.5,
        1.0,
        diffusion_depth,
        nb=nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    hhd(phi, phi_tnd)

    gamma = to_numpy(hhd._gamma)  # np.tile(hd._gamma, (ni, nj, 1))
    phi_tnd_assert = gamma * third_order_diffusion_xz(dx, to_numpy(phi))

    assert_xz(phi_tnd, phi_tnd_assert, nb)


def third_order_validation_yz(phi, grid, diffusion_depth, nb, backend, bo, so):
    ni, nj, nk = phi.shape
    phi_tnd = zeros(backend, shape=(ni, nj, nk), storage_options=so)

    dx = grid.dx.values.item()
    dy = grid.dy.values.item()

    hhd = HHD.factory(
        "third_order_1dy",
        (ni, nj, nk),
        dx,
        dy,
        0.5,
        1.0,
        diffusion_depth,
        nb=nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
    )
    hhd(phi, phi_tnd)

    gamma = to_numpy(hhd._gamma)  # np.tile(hd._gamma, (ni, nj, 1))
    phi_tnd_assert = gamma * third_order_diffusion_yz(dy, to_numpy(phi))

    assert_yz(phi_tnd, phi_tnd_assert, nb)


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

    nb = data.draw(hyp_st.integers(min_value=3, max_value=max(3, conf_nb)))
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
            min_value=-1e10,
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
    third_order_validation_xyz(phi, grid, depth, nb, backend, bo, so)
    third_order_validation_xz(phi, grid, depth, nb, backend, bo, so)
    third_order_validation_yz(phi, grid, depth, nb, backend, bo, so)


if __name__ == "__main__":
    pytest.main([__file__])
