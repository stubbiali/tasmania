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


@register("fifth_order")
class FifthOrder(BurgersAdvection):
    extent = 3

    @staticmethod
    @stencil_subroutine(
        backend=("numpy", "cupy", "numba:cpu"), stencil="advection"
    )
    def call_numpy(dx, dy, u, v):
        abs_u = np.abs(u[3:-3, 3:-3])
        abs_v = np.abs(v[3:-3, 3:-3])

        adv_u_x = u[3:-3, 3:-3] / (60.0 * dx) * (
            +45.0 * (u[4:-2, 3:-3] - u[2:-4, 3:-3])
            - 9.0 * (u[5:-1, 3:-3] - u[1:-5, 3:-3])
            + (u[6:, 3:-3] - u[:-6, 3:-3])
        ) - abs_u / (60.0 * dx) * (
            +(u[6:, 3:-3] + u[:-6, 3:-3])
            - 6.0 * (u[5:-1, 3:-3] + u[1:-5, 3:-3])
            + 15.0 * (u[4:-2, 3:-3] + u[2:-4, 3:-3])
            - 20.0 * u[3:-3, 3:-3]
        )
        adv_u_y = v[3:-3, 3:-3] / (60.0 * dy) * (
            +45.0 * (u[3:-3, 4:-2] - u[3:-3, 2:-4])
            - 9.0 * (u[3:-3, 5:-1] - u[3:-3, 1:-5])
            + (u[3:-3, 6:] - u[3:-3, :-6])
        ) - abs_v / (60.0 * dy) * (
            +(u[3:-3, 6:] + u[3:-3, :-6])
            - 6.0 * (u[3:-3, 5:-1] + u[3:-3, 1:-5])
            + 15.0 * (u[3:-3, 4:-2] + u[3:-3, 2:-4])
            - 20.0 * u[3:-3, 3:-3]
        )
        adv_v_x = u[3:-3, 3:-3] / (60.0 * dx) * (
            +45.0 * (v[4:-2, 3:-3] - v[2:-4, 3:-3])
            - 9.0 * (v[5:-1, 3:-3] - v[1:-5, 3:-3])
            + (v[6:, 3:-3] - v[:-6, 3:-3])
        ) - abs_u / (60.0 * dx) * (
            +(v[6:, 3:-3] + v[:-6, 3:-3])
            - 6.0 * (v[5:-1, 3:-3] + v[1:-5, 3:-3])
            + 15.0 * (v[4:-2, 3:-3] + v[2:-4, 3:-3])
            - 20.0 * v[3:-3, 3:-3]
        )
        adv_v_y = v[3:-3, 3:-3] / (60.0 * dy) * (
            +45.0 * (v[3:-3, 4:-2] - v[3:-3, 2:-4])
            - 9.0 * (v[3:-3, 5:-1] - v[3:-3, 1:-5])
            + (v[3:-3, 6:] - v[3:-3, :-6])
        ) - abs_v / (60.0 * dy) * (
            +(v[3:-3, 6:] + v[3:-3, :-6])
            - 6.0 * (v[3:-3, 5:-1] + v[3:-3, 1:-5])
            + 15.0 * (v[3:-3, 4:-2] + v[3:-3, 2:-4])
            - 20.0 * v[3:-3, 3:-3]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="advection")
    @gtscript.function
    def call_gt4py(dx, dy, u, v):
        abs_u = abs(u)  # u if u > 0 else -u
        abs_v = abs(v)  # v if v > 0 else -v

        adv_u_x = u[0, 0, 0] / (60.0 * dx) * (
            +45.0 * (u[+1, 0, 0] - u[-1, 0, 0])
            - 9.0 * (u[+2, 0, 0] - u[-2, 0, 0])
            + (u[+3, 0, 0] - u[-3, 0, 0])
        ) - abs_u[0, 0, 0] / (60.0 * dx) * (
            +(u[+3, 0, 0] + u[-3, 0, 0])
            - 6.0 * (u[+2, 0, 0] + u[-2, 0, 0])
            + 15.0 * (u[+1, 0, 0] + u[-1, 0, 0])
            - 20.0 * u[0, 0, 0]
        )
        adv_u_y = v[0, 0, 0] / (60.0 * dy) * (
            +45.0 * (u[0, +1, 0] - u[0, -1, 0])
            - 9.0 * (u[0, +2, 0] - u[0, -2, 0])
            + (u[0, +3, 0] - u[0, -3, 0])
        ) - abs_v[0, 0, 0] / (60.0 * dy) * (
            +(u[0, +3, 0] + u[0, -3, 0])
            - 6.0 * (u[0, +2, 0] + u[0, -2, 0])
            + 15.0 * (u[0, +1, 0] + u[0, -1, 0])
            - 20.0 * u[0, 0, 0]
        )
        adv_v_x = u[0, 0, 0] / (60.0 * dx) * (
            +45.0 * (v[+1, 0, 0] - v[-1, 0, 0])
            - 9.0 * (v[+2, 0, 0] - v[-2, 0, 0])
            + (v[+3, 0, 0] - v[-3, 0, 0])
        ) - abs_u[0, 0, 0] / (60.0 * dx) * (
            +(v[+3, 0, 0] + v[-3, 0, 0])
            - 6.0 * (v[+2, 0, 0] + v[-2, 0, 0])
            + 15.0 * (v[+1, 0, 0] + v[-1, 0, 0])
            - 20.0 * v[0, 0, 0]
        )
        adv_v_y = v[0, 0, 0] / (60.0 * dy) * (
            +45.0 * (v[0, +1, 0] - v[0, -1, 0])
            - 9.0 * (v[0, +2, 0] - v[0, -2, 0])
            + (v[0, +3, 0] - v[0, -3, 0])
        ) - abs_v[0, 0, 0] / (60.0 * dy) * (
            +(v[0, +3, 0] + v[0, -3, 0])
            - 6.0 * (v[0, +2, 0] + v[0, -2, 0])
            + 15.0 * (v[0, +1, 0] + v[0, -1, 0])
            - 20.0 * v[0, 0, 0]
        )

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    # @staticmethod
    # @stencil_subroutine(backend="numba:cpu", stencil="advection")
    # def call_numba_cpu(dx, dy, u, v):
    #     # >>> stencil definitions
    #     def absolute_def(phi):
    #         return phi[0, 0, 0] if phi[0, 0, 0] > 0 else -phi[0, 0, 0]
    #
    #     def advection_x_def(u, abs_u, phi, dx):
    #         return u[0, 0, 0] / (60.0 * dx) * (
    #             +45.0 * (phi[+1, 0, 0] - phi[-1, 0, 0])
    #             - 9.0 * (phi[+2, 0, 0] - phi[-2, 0, 0])
    #             + (phi[+3, 0, 0] - phi[-3, 0, 0])
    #         ) - abs_u[0, 0, 0] / (60.0 * dx) * (
    #             +(phi[+3, 0, 0] + phi[-3, 0, 0])
    #             - 6.0 * (phi[+2, 0, 0] + phi[-2, 0, 0])
    #             + 15.0 * (phi[+1, 0, 0] + phi[-1, 0, 0])
    #             - 20.0 * phi[0, 0, 0]
    #         )
    #
    #     def advection_y_def(v, abs_v, phi, dy):
    #         return v[0, 0, 0] / (60.0 * dy) * (
    #             +45.0 * (phi[0, +1, 0] - phi[0, -1, 0])
    #             - 9.0 * (phi[0, +2, 0] - phi[0, -2, 0])
    #             + (phi[0, +3, 0] - phi[0, -3, 0])
    #         ) - abs_v[0, 0, 0] / (60.0 * dy) * (
    #             +(phi[0, +3, 0] + phi[0, -3, 0])
    #             - 6.0 * (phi[0, +2, 0] + phi[0, -2, 0])
    #             + 15.0 * (phi[0, +1, 0] + phi[0, -1, 0])
    #             - 20.0 * phi[0, 0, 0]
    #         )
    #
    #     # >>> stencil compilations
    #     absolute = numba.stencil(absolute_def)
    #     advection_x = numba.stencil(advection_x_def)
    #     advection_y = numba.stencil(advection_y_def)
    #
    #     # >>> calculations
    #     abs_u = absolute(u)
    #     abs_v = absolute(v)
    #     adv_u_x = advection_x(u, abs_u, u, dx)
    #     adv_u_y = advection_y(v, abs_v, u, dy)
    #     adv_v_x = advection_x(u, abs_u, v, dx)
    #     adv_v_y = advection_y(v, abs_v, v, dy)
    #
    #     return adv_u_x, adv_u_y, adv_v_x, adv_v_y
