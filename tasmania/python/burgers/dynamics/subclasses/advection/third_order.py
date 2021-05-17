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
import numba
import numpy as np

from gt4py import gtscript

from tasmania.python.burgers.dynamics.advection import BurgersAdvection
from tasmania.python.framework.register import register
from tasmania.python.framework.tag import stencil_subroutine


@register("third_order")
class ThirdOrder(BurgersAdvection):
    extent = 2

    @staticmethod
    @stencil_subroutine(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="advection"
    )
    def call_numpy(dx, dy, u, v):
        abs_u = np.abs(u[2:-2, 2:-2])
        abs_v = np.abs(v[2:-2, 2:-2])

        adv_u_x = u[2:-2, 2:-2] / (12.0 * dx) * (
            8.0 * (u[3:-1, 2:-2] - u[1:-3, 2:-2])
            - (u[4:, 2:-2] - u[:-4, 2:-2])
        ) + abs_u / (12.0 * dx) * (
            u[4:, 2:-2]
            + u[:-4, 2:-2]
            - 4.0 * (u[3:-1, 2:-2] + u[1:-3, 2:-2])
            + 6.0 * u[2:-2, 2:-2]
        )
        adv_u_y = v[2:-2, 2:-2] / (12.0 * dy) * (
            8.0 * (u[2:-2, 3:-1] - u[2:-2, 1:-3])
            - (u[2:-2, 4:] - u[2:-2, :-4])
        ) + abs_v / (12.0 * dy) * (
            u[2:-2, 4:]
            + u[2:-2, :-4]
            - 4.0 * (u[2:-2, 3:-1] + u[2:-2, 1:-3])
            + 6.0 * u[2:-2, 2:-2]
        )
        adv_v_x = u[2:-2, 2:-2] / (12.0 * dx) * (
            8.0 * (v[3:-1, 2:-2] - v[1:-3, 2:-2])
            - (v[4:, 2:-2] - v[:-4, 2:-2])
        ) + abs_u / (12.0 * dx) * (
            v[4:, 2:-2]
            + v[:-4, 2:-2]
            - 4.0 * (v[3:-1, 2:-2] + v[1:-3, 2:-2])
            + 6.0 * v[2:-2, 2:-2]
        )
        adv_v_y = v[2:-2, 2:-2] / (12.0 * dy) * (
            8.0 * (v[2:-2, 3:-1] - v[2:-2, 1:-3])
            - (v[2:-2, 4:] - v[2:-2, :-4])
        ) + abs_v / (12.0 * dy) * (
            v[2:-2, 4:]
            + v[2:-2, :-4]
            - 4.0 * (v[2:-2, 3:-1] + v[2:-2, 1:-3])
            + 6.0 * v[2:-2, 2:-2]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="advection")
    @gtscript.function
    def call_gt4py(dx, dy, u, v):
        abs_u = abs(u)  # u if u > 0 else -u
        abs_v = abs(v)  # v if v > 0 else -v

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
    @stencil_subroutine(backend="numba:cpu:stencil", stencil="advection")
    def call_numba_cpu(dx, dy, u, v):
        # >>> stencil definitions
        def absolute_def(phi):
            return phi[0, 0, 0] if phi[0, 0, 0] > 0 else -phi[0, 0, 0]

        def advection_x_def(u, abs_u, phi, dx):
            return u[0, 0, 0] / (12.0 * dx) * (
                8.0 * (phi[+1, 0, 0] - phi[-1, 0, 0])
                - (phi[+2, 0, 0] - phi[-2, 0, 0])
            ) + abs_u[0, 0, 0] / (12.0 * dx) * (
                phi[+2, 0, 0]
                + phi[-2, 0, 0]
                - 4.0 * (phi[+1, 0, 0] + phi[-1, 0, 0])
                + 6.0 * phi[0, 0, 0]
            )

        def advection_y_def(v, abs_v, phi, dy):
            return v[0, 0, 0] / (12.0 * dy) * (
                8.0 * (phi[0, +1, 0] - phi[0, -1, 0])
                - (phi[0, +2, 0] - phi[0, -2, 0])
            ) + abs_v[0, 0, 0] / (12.0 * dy) * (
                phi[0, +2, 0]
                + phi[0, -2, 0]
                - 4.0 * (phi[0, +1, 0] + phi[0, -1, 0])
                + 6.0 * phi[0, 0, 0]
            )

        # >>> stencil compilations
        absolute = numba.stencil(absolute_def)
        advection_x = numba.stencil(advection_x_def)
        advection_y = numba.stencil(advection_y_def)

        # >>> calculations
        abs_u = absolute(u)
        abs_v = absolute(v)
        adv_u_x = advection_x(u, abs_u, u, dx)
        adv_u_y = advection_y(v, abs_v, u, dy)
        adv_v_x = advection_x(u, abs_u, v, dx)
        adv_v_y = advection_y(v, abs_v, v, dy)

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y
