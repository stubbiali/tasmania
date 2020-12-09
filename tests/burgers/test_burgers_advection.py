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
import numpy as np
import pytest

from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, interval

from tasmania.python.burgers.dynamics.advection import BurgersAdvection
from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import StorageOptions
from tasmania.python.utils.utils import get_gt_backend, is_gt

from tests.conf import (
    backend as conf_backend,
    dtype as conf_dtype,
    default_origin as conf_dorigin,
)
from tests.strategies import st_burgers_state, st_one_of, st_physical_grid
from tests.utilities import compare_arrays, hyp_settings


class WrappingStencil:
    def __init__(self, advection, backend, dtype):
        self.nb = advection.extent
        decorator = gtscript.stencil(
            backend,
            rebuild=False,
            dtypes={"dtype": dtype},
            externals={"call_func": advection.stencil_subroutine("advection")},
        )
        self.stencil = decorator(self.stencil_defs)

    def __call__(self, dx, dy, u, v, adv_u_x, adv_u_y, adv_v_x, adv_v_y):
        mi, mj, mk = u.shape
        nb = self.nb
        self.stencil(
            in_u=u,
            in_v=v,
            out_adv_u_x=adv_u_x,
            out_adv_u_y=adv_u_y,
            out_adv_v_x=adv_v_x,
            out_adv_v_y=adv_v_y,
            dx=dx,
            dy=dy,
            origin=(nb, nb, 0),
            domain=(mi - 2 * nb, mj - 2 * nb, mk),
        )

    @staticmethod
    def stencil_defs(
        in_u: gtscript.Field["dtype"],
        in_v: gtscript.Field["dtype"],
        out_adv_u_x: gtscript.Field["dtype"],
        out_adv_u_y: gtscript.Field["dtype"],
        out_adv_v_x: gtscript.Field["dtype"],
        out_adv_v_y: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float
    ):
        from __externals__ import call_func

        with computation(PARALLEL), interval(...):
            out_adv_u_x, out_adv_u_y, out_adv_v_x, out_adv_v_y = call_func(
                dx=dx, dy=dy, u=in_u, v=in_v
            )


def first_order_advection(dx, dy, u, v, phi):
    adv_x = deepcopy(phi)
    adv_x[1:-1, :, :] = u[1:-1, :, :] / (2.0 * dx) * (
        phi[2:, :, :] - phi[:-2, :, :]
    ) - np.abs(u)[1:-1, :, :] / (2.0 * dx) * (
        phi[2:, :, :] - 2.0 * phi[1:-1, :, :] + phi[:-2, :, :]
    )
    adv_y = deepcopy(phi)
    adv_y[:, 1:-1, :] = v[:, 1:-1, :] / (2.0 * dy) * (
        phi[:, 2:, :] - phi[:, :-2, :]
    ) - np.abs(v)[:, 1:-1, :] / (2.0 * dy) * (
        phi[:, 2:, :] - 2.0 * phi[:, 1:-1, :] + phi[:, :-2, :]
    )
    return adv_x, adv_y


def second_order_advection(dx, dy, u, v, phi):
    adv_x = deepcopy(phi)
    adv_x[1:-1, :, :] = (
        u[1:-1, :, :] / (2.0 * dx) * (phi[2:, :, :] - phi[:-2, :, :])
    )
    adv_y = deepcopy(phi)
    adv_y[:, 1:-1, :] = (
        v[:, 1:-1, :] / (2.0 * dy) * (phi[:, 2:, :] - phi[:, :-2, :])
    )
    return adv_x, adv_y


def third_order_advection(dx, dy, u, v, phi):
    adv_x = deepcopy(phi)
    adv_x[2:-2, :, :] = u[2:-2, :, :] / (12.0 * dx) * (
        8.0 * (phi[3:-1, :, :] - phi[1:-3, :, :])
        - (phi[4:, :, :] - phi[:-4, :, :])
    ) + np.abs(u)[2:-2, :, :] / (12.0 * dx) * (
        (phi[4:, :, :] + phi[:-4, :, :])
        - 4.0 * (phi[3:-1, :, :] + phi[1:-3, :, :])
        + 6.0 * phi[2:-2, :, :]
    )
    adv_y = deepcopy(phi)
    adv_y[:, 2:-2, :] = v[:, 2:-2, :] / (12.0 * dy) * (
        8.0 * (phi[:, 3:-1, :] - phi[:, 1:-3, :])
        - (phi[:, 4:, :] - phi[:, :-4, :])
    ) + np.abs(v)[:, 2:-2, :] / (12.0 * dy) * (
        (phi[:, 4:, :] + phi[:, :-4, :])
        - 4.0 * (phi[:, 3:-1, :] + phi[:, 1:-3, :])
        + 6.0 * phi[:, 2:-2, :]
    )
    return adv_x, adv_y


def fourth_order_advection(dx, dy, u, v, phi):
    adv_x = deepcopy(phi)
    adv_x[2:-2, :, :] = (
        u[2:-2, :, :]
        / (12.0 * dx)
        * (
            8.0 * (phi[3:-1, :, :] - phi[1:-3, :, :])
            - (phi[4:, :, :] - phi[:-4, :, :])
        )
    )
    adv_y = deepcopy(phi)
    adv_y[:, 2:-2, :] = (
        v[:, 2:-2, :]
        / (12.0 * dy)
        * (
            8.0 * (phi[:, 3:-1, :] - phi[:, 1:-3, :])
            - (phi[:, 4:, :] - phi[:, :-4, :])
        )
    )
    return adv_x, adv_y


def fifth_order_advection(dx, dy, u, v, phi):
    adv_x = deepcopy(phi)
    adv_x[3:-3, :, :] = u[3:-3, :, :] / (60.0 * dx) * (
        45.0 * (phi[4:-2, :, :] - phi[2:-4, :, :])
        - 9.0 * (phi[5:-1, :, :] - phi[1:-5, :, :])
        + (phi[6:, :, :] - phi[:-6, :, :])
    ) - np.abs(u)[3:-3, :, :] / (60.0 * dx) * (
        (phi[6:, :, :] + phi[:-6, :, :])
        - 6.0 * (phi[5:-1, :, :] + phi[1:-5, :, :])
        + 15.0 * (phi[4:-2, :, :] + phi[2:-4, :, :])
        - 20.0 * phi[3:-3, :, :]
    )
    adv_y = deepcopy(phi)
    adv_y[:, 3:-3, :] = v[:, 3:-3, :] / (60.0 * dy) * (
        45.0 * (phi[:, 4:-2, :] - phi[:, 2:-4, :])
        - 9.0 * (phi[:, 5:-1, :] - phi[:, 1:-5, :])
        + (phi[:, 6:, :] - phi[:, :-6, :])
    ) - np.abs(v)[:, 3:-3, :] / (60.0 * dy) * (
        (phi[:, 6:, :] + phi[:, :-6, :])
        - 6.0 * (phi[:, 5:-1, :] + phi[:, 1:-5, :])
        + 15.0 * (phi[:, 4:-2, :] + phi[:, 2:-4, :])
        - 20.0 * phi[:, 3:-3, :]
    )
    return adv_x, adv_y


def sixth_order_advection(dx, dy, u, v, phi):
    adv_x = deepcopy(phi)
    adv_x[3:-3, :, :] = (
        u[3:-3, :, :]
        / (60.0 * dx)
        * (
            45.0 * (phi[4:-2, :, :] - phi[2:-4, :, :])
            - 9.0 * (phi[5:-1, :, :] - phi[1:-5, :, :])
            + (phi[6:, :, :] - phi[:-6, :, :])
        )
    )

    adv_y = deepcopy(phi)
    adv_y[:, 3:-3, :] = (
        v[:, 3:-3, :]
        / (60.0 * dy)
        * (
            45.0 * (phi[:, 4:-2, :] - phi[:, 2:-4, :])
            - 9.0 * (phi[:, 5:-1, :] - phi[:, 1:-5, :])
            + (phi[:, 6:, :] - phi[:, :-6, :])
        )
    )

    return adv_x, adv_y


validation_functions = {
    "first_order": first_order_advection,
    "second_order": second_order_advection,
    "third_order": third_order_advection,
    "fourth_order": fourth_order_advection,
    "fifth_order": fifth_order_advection,
    "sixth_order": sixth_order_advection,
}


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("order", validation_functions.keys())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, order, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    advection = BurgersAdvection.factory(order, backend)
    nb = advection.extent

    grid = data.draw(
        st_physical_grid(
            xaxis_length=(2 * nb + 1, 40),
            yaxis_length=(2 * nb + 1, 40),
            zaxis_length=(1, 1),
            dtype=dtype,
        ),
        label="grid",
    )

    state = data.draw(
        st_burgers_state(grid, backend=backend, default_origin=default_origin),
        label="state",
    )

    # ========================================
    # test bed
    # ========================================
    so = StorageOptions(dtype=dtype, default_origin=default_origin)

    dx = grid.grid_xy.dx.to_units("m").values.item()
    dy = grid.grid_xy.dy.to_units("m").values.item()
    u = state["x_velocity"].to_units("m s^-1").data
    v = state["y_velocity"].to_units("m s^-1").data

    grid_shape = u.shape
    adv_u_x = zeros(backend, shape=grid_shape, storage_options=so)
    adv_u_y = zeros(backend, shape=grid_shape, storage_options=so)
    adv_v_x = zeros(backend, shape=grid_shape, storage_options=so)
    adv_v_y = zeros(backend, shape=grid_shape, storage_options=so)

    if is_gt(backend):
        ws = WrappingStencil(advection, get_gt_backend(backend), dtype)
        ws(dx, dy, u, v, adv_u_x, adv_u_y, adv_v_x, adv_v_y)
    else:
        (
            adv_u_x[nb:-nb, nb:-nb],
            adv_u_y[nb:-nb, nb:-nb],
            adv_v_x[nb:-nb, nb:-nb],
            adv_v_y[nb:-nb, nb:-nb],
        ) = advection.stencil_subroutine("advection")(dx, dy, u, v)

    adv_u_x_val, adv_u_y_val = validation_functions[order](dx, dy, u, v, u)
    adv_v_x_val, adv_v_y_val = validation_functions[order](dx, dy, u, v, v)

    compare_arrays(adv_u_x[nb:-nb, nb:-nb], adv_u_x_val[nb:-nb, nb:-nb])
    compare_arrays(adv_u_y[nb:-nb, nb:-nb], adv_u_y_val[nb:-nb, nb:-nb])
    compare_arrays(adv_v_x[nb:-nb, nb:-nb], adv_v_x_val[nb:-nb, nb:-nb])
    compare_arrays(adv_v_y[nb:-nb, nb:-nb], adv_v_y_val[nb:-nb, nb:-nb])


if __name__ == "__main__":
    pytest.main([__file__])
