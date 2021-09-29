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
import numba

from gt4py import gtscript

from tasmania.python.burgers.dynamics.advection import BurgersAdvection
from tasmania.python.framework.register import register
from tasmania.python.framework.tag import subroutine_definition


@register("second_order")
class SecondOrder(BurgersAdvection):
    extent = 1

    @staticmethod
    @subroutine_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="advection"
    )
    def call_numpy(dx, dy, u, v):
        adv_u_x = u[1:-1, 1:-1] / (2.0 * dx) * (u[2:, 1:-1] - u[:-2, 1:-1])
        adv_u_y = v[1:-1, 1:-1] / (2.0 * dy) * (u[1:-1, 2:] - u[1:-1, :-2])
        adv_v_x = u[1:-1, 1:-1] / (2.0 * dx) * (v[2:, 1:-1] - v[:-2, 1:-1])
        adv_v_y = v[1:-1, 1:-1] / (2.0 * dy) * (v[1:-1, 2:] - v[1:-1, :-2])

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="advection")
    @gtscript.function
    def call_gt4py(dx, dy, u, v):
        adv_u_x = u[0, 0, 0] / (2.0 * dx) * (u[+1, 0, 0] - u[-1, 0, 0])
        adv_u_y = v[0, 0, 0] / (2.0 * dy) * (u[0, +1, 0] - u[0, -1, 0])
        adv_v_x = u[0, 0, 0] / (2.0 * dx) * (v[+1, 0, 0] - v[-1, 0, 0])
        adv_v_y = v[0, 0, 0] / (2.0 * dy) * (v[0, +1, 0] - v[0, -1, 0])

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y

    @staticmethod
    @subroutine_definition(backend="numba:cpu:stencil", stencil="advection")
    def call_numba_cpu(dx, dy, u, v):
        # >>> stencil definitions
        def advection_x_def(u, phi, dx):
            return u[0, 0, 0] / (2.0 * dx) * (phi[+1, 0, 0] - phi[-1, 0, 0])

        def advection_y_def(v, phi, dy):
            return v[0, 0, 0] / (2.0 * dy) * (phi[0, +1, 0] - phi[0, -1, 0])

        # >>> stencil compilations
        advection_x = numba.stencil(advection_x_def)
        advection_y = numba.stencil(advection_y_def)

        # >>> calculations
        adv_u_x = advection_x(u, u, dx)
        adv_u_y = advection_y(v, u, dy)
        adv_v_x = advection_x(u, v, dx)
        adv_v_y = advection_y(v, v, dy)

        return adv_u_x, adv_u_y, adv_v_x, adv_v_y
