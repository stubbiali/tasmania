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
import numpy as np

from gt4py import gtscript

from tasmania.python.framework.allocators import as_storage, zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import (
    stencil_definition,
    subroutine_definition,
)

from tests.utilities import compare_arrays


class ThirdOrder(StencilFactory):
    def __init__(self, backend, backend_options, storage_options):
        super().__init__(backend, backend_options, storage_options)
        self.nb = 2
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.stencil = self.compile_stencil("burgers_advection")

    def __call__(self, dx, dy, u, v, adv_u_x, adv_u_y, adv_v_x, adv_v_y):
        nb = self.nb
        mi, mj, mk = u.shape
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
            validate_args=self.backend_options.validate_args,
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="burgers_advection")
    def burgers_advection_gt4py(
        in_u: gtscript.Field["dtype"],
        in_v: gtscript.Field["dtype"],
        out_adv_u_x: gtscript.Field["dtype"],
        out_adv_u_y: gtscript.Field["dtype"],
        out_adv_v_x: gtscript.Field["dtype"],
        out_adv_v_y: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
    ):
        with computation(PARALLEL), interval(...):
            abs_u = in_u if in_u > 0 else -in_u
            abs_v = in_v if in_v > 0 else -in_v

            out_adv_u_x = in_u[0, 0, 0] / (12.0 * dx) * (
                8.0 * (in_u[+1, 0, 0] - in_u[-1, 0, 0])
                - (in_u[+2, 0, 0] - in_u[-2, 0, 0])
            ) + abs_u[0, 0, 0] / (12.0 * dx) * (
                in_u[+2, 0, 0]
                + in_u[-2, 0, 0]
                - 4.0 * (in_u[+1, 0, 0] + in_u[-1, 0, 0])
                + 6.0 * in_u[0, 0, 0]
            )
            out_adv_u_y = in_v[0, 0, 0] / (12.0 * dy) * (
                8.0 * (in_u[0, +1, 0] - in_u[0, -1, 0])
                - (in_u[0, +2, 0] - in_u[0, -2, 0])
            ) + abs_v[0, 0, 0] / (12.0 * dy) * (
                in_u[0, +2, 0]
                + in_u[0, -2, 0]
                - 4.0 * (in_u[0, +1, 0] + in_u[0, -1, 0])
                + 6.0 * in_u[0, 0, 0]
            )
            out_adv_v_x = in_u[0, 0, 0] / (12.0 * dx) * (
                8.0 * (in_v[+1, 0, 0] - in_v[-1, 0, 0])
                - (in_v[+2, 0, 0] - in_v[-2, 0, 0])
            ) + abs_u[0, 0, 0] / (12.0 * dx) * (
                in_v[+2, 0, 0]
                + in_v[-2, 0, 0]
                - 4.0 * (in_v[+1, 0, 0] + in_v[-1, 0, 0])
                + 6.0 * in_v[0, 0, 0]
            )
            out_adv_v_y = in_v[0, 0, 0] / (12.0 * dy) * (
                8.0 * (in_v[0, +1, 0] - in_v[0, -1, 0])
                - (in_v[0, +2, 0] - in_v[0, -2, 0])
            ) + abs_v[0, 0, 0] / (12.0 * dy) * (
                in_v[0, +2, 0]
                + in_v[0, -2, 0]
                - 4.0 * (in_v[0, +1, 0] + in_v[0, -1, 0])
                + 6.0 * in_v[0, 0, 0]
            )


class ThirdOrderSubroutine(StencilFactory):
    def __init__(self, backend, backend_options, storage_options):
        super().__init__(backend, backend_options, storage_options)
        self.nb = 2
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "advection": self.stencil_subroutine("advection")
        }
        self.stencil = self.compile_stencil("burgers_advection")

    def __call__(self, dx, dy, u, v, adv_u_x, adv_u_y, adv_v_x, adv_v_y):
        nb = self.nb
        mi, mj, mk = u.shape
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
            validate_args=self.backend_options.validate_args,
        )

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="advection")
    @gtscript.function
    def advection_gt4py(u, v, dx, dy):
        abs_u = u if u > 0 else -u
        abs_v = v if v > 0 else -v

        adv_u_x = u[0, 0, 0] / (12.0 * dx) * (
            8.0 * (u[+1, 0, 0] - u[-1, 0, 0]) - (u[+2, 0, 0] - u[-2, 0, 0])
        ) + abs_u[0, 0, 0] / (12.0 * dx) * (
            u[+2, 0, 0]
            + u[-2, 0, 0]
            - 4.0 * (u[+1, 0, 0] + u[-1, 0, 0])
            + 6.0 * u[0, 0, 0]
        )
        adv_u_y = v[0, 0, 0] / (12.0 * dy) * (
            8.0 * (u[0, +1, 0] - u[0, -1, 0]) - (u[0, +2, 0] - u[0, -2, 0])
        ) + abs_v[0, 0, 0] / (12.0 * dy) * (
            u[0, +2, 0]
            + u[0, -2, 0]
            - 4.0 * (u[0, +1, 0] + u[0, -1, 0])
            + 6.0 * u[0, 0, 0]
        )
        adv_v_x = u[0, 0, 0] / (12.0 * dx) * (
            8.0 * (v[+1, 0, 0] - v[-1, 0, 0]) - (v[+2, 0, 0] - v[-2, 0, 0])
        ) + abs_u[0, 0, 0] / (12.0 * dx) * (
            v[+2, 0, 0]
            + v[-2, 0, 0]
            - 4.0 * (v[+1, 0, 0] + v[-1, 0, 0])
            + 6.0 * v[0, 0, 0]
        )
        adv_v_y = v[0, 0, 0] / (12.0 * dy) * (
            8.0 * (v[0, +1, 0] - v[0, -1, 0]) - (v[0, +2, 0] - v[0, -2, 0])
        ) + abs_v[0, 0, 0] / (12.0 * dy) * (
            v[0, +2, 0]
            + v[0, -2, 0]
            - 4.0 * (v[0, +1, 0] + v[0, -1, 0])
            + 6.0 * v[0, 0, 0]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="burgers_advection")
    def burgers_advection_gt4py(
        in_u: gtscript.Field["dtype"],
        in_v: gtscript.Field["dtype"],
        out_adv_u_x: gtscript.Field["dtype"],
        out_adv_u_y: gtscript.Field["dtype"],
        out_adv_v_x: gtscript.Field["dtype"],
        out_adv_v_y: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
    ):
        from __externals__ import advection

        with computation(PARALLEL), interval(...):
            out_adv_u_x, out_adv_u_y, out_adv_v_x, out_adv_v_y = advection(
                in_u, in_v, dx, dy
            )


def third_order_advection(dx, dy, u, v, phi):
    adv_x = np.zeros_like(phi)
    adv_x[2:-2, :, :] = u[2:-2, :, :] / (12.0 * dx) * (
        8.0 * (phi[3:-1, :, :] - phi[1:-3, :, :])
        - (phi[4:, :, :] - phi[:-4, :, :])
    ) + np.abs(u)[2:-2, :, :] / (12.0 * dx) * (
        (phi[4:, :, :] + phi[:-4, :, :])
        - 4.0 * (phi[3:-1, :, :] + phi[1:-3, :, :])
        + 6.0 * phi[2:-2, :, :]
    )
    adv_y = np.zeros_like(phi)
    adv_y[:, 2:-2, :] = v[:, 2:-2, :] / (12.0 * dy) * (
        8.0 * (phi[:, 3:-1, :] - phi[:, 1:-3, :])
        - (phi[:, 4:, :] - phi[:, :-4, :])
    ) + np.abs(v)[:, 2:-2, :] / (12.0 * dy) * (
        (phi[:, 4:, :] + phi[:, :-4, :])
        - 4.0 * (phi[:, 3:-1, :] + phi[:, 1:-3, :])
        + 6.0 * phi[:, 2:-2, :]
    )
    return adv_x, adv_y


def main():
    backend = "gt4py:gtcuda"
    # backend = "gt4py:gtc:gt:gpu"
    dtype = float
    aligned_index = (0, 0, 0)

    bo = BackendOptions(cache=True, nopython=True, rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    shape = (5, 5, 1)
    u_np = np.zeros(shape, dtype=dtype)
    v_np = 0.08333333333333529 * np.ones(shape, dtype=dtype)
    dx = 2.5002222514558532e-08
    dy = 3.330669073875471e-11

    u = zeros(backend, shape=shape, storage_options=so)
    u[...] = as_storage(backend, data=u_np)
    v = zeros(backend, shape=shape, storage_options=so)
    v[...] = as_storage(backend, data=v_np)

    adv_u_x = zeros(backend, shape=shape, storage_options=so)
    adv_u_y = zeros(backend, shape=shape, storage_options=so)
    adv_v_x = zeros(backend, shape=shape, storage_options=so)
    adv_v_y = zeros(backend, shape=shape, storage_options=so)

    # ws = ThirdOrder(backend, bo, so)
    ws = ThirdOrderSubroutine(backend, bo, so)
    ws(dx, dy, u, v, adv_u_x, adv_u_y, adv_v_x, adv_v_y)

    adv_u_x_val, adv_u_y_val = third_order_advection(dx, dy, u_np, v_np, u_np)
    adv_v_x_val, adv_v_y_val = third_order_advection(dx, dy, u_np, v_np, v_np)

    nb = 2
    compare_arrays(adv_u_x[nb:-nb, nb:-nb], adv_u_x_val[nb:-nb, nb:-nb])
    compare_arrays(adv_u_y[nb:-nb, nb:-nb], adv_u_y_val[nb:-nb, nb:-nb])
    compare_arrays(adv_v_x[nb:-nb, nb:-nb], adv_v_x_val[nb:-nb, nb:-nb])
    compare_arrays(adv_v_y[nb:-nb, nb:-nb], adv_v_y_val[nb:-nb, nb:-nb])


if __name__ == "__main__":
    main()
