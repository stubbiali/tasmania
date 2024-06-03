# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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

from gt4py.cartesian import gtscript

from tasmania.framework.tag import subroutine_definition
from tasmania.physics.microphysics.utils import SedimentationFlux


class FirstOrderUpwind(SedimentationFlux):
    """The standard, first-order accurate upwind method."""

    name = "first_order_upwind"
    nb = 1

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="flux")
    def call_numpy(rho, h, q, vt):
        dfdz = (
            rho[:, :, :-1] * q[:, :, :-1] * vt[:, :, :-1]
            - rho[:, :, 1:] * q[:, :, 1:] * vt[:, :, 1:]
        ) / (h[:, :, :-1] - h[:, :, 1:])
        return dfdz

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux")
    @gtscript.function
    def call_gt4py(rho, h, q, vt):
        dfdz = (
            rho[0, 0, -1] * q[0, 0, -1] * vt[0, 0, -1] - rho[0, 0, 0] * q[0, 0, 0] * vt[0, 0, 0]
        ) / (h[0, 0, -1] - h[0, 0, 0])
        return dfdz
