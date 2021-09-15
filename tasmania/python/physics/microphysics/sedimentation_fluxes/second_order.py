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
from gt4py import gtscript

from tasmania.python.framework.tag import subroutine_definition
from tasmania.python.physics.microphysics.utils import SedimentationFlux


class SecondOrderUpwind(SedimentationFlux):
    """The second-order accurate upwind method."""

    name = "second_order_upwind"
    nb = 2

    @staticmethod
    @subroutine_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="flux"
    )
    def call_numpy(rho, h, q, vt):
        # evaluate the space-dependent coefficients occurring in the
        # second-order upwind finite difference approximation of the
        # vertical derivative of the flux
        tmp_a = (2.0 * h[:, :, 2:] - h[:, :, 1:-1] - h[:, :, :-2]) / (
            (h[:, :, 1:-1] - h[:, :, 2:]) * (h[:, :, :-2] - h[:, :, 2:])
        )
        tmp_b = (h[:, :, :-2] - h[:, :, 2:]) / (
            (h[:, :, 1:-1] - h[:, :, 2:]) * (h[:, :, :-2] - h[:, :, 1:-1])
        )
        tmp_c = (h[:, :, 2:] - h[:, :, 1:-1]) / (
            (h[:, :, :-2] - h[:, :, 2:]) * (h[:, :, :-2] - h[:, :, 1:-1])
        )

        # calculate the vertical derivative of the sedimentation flux
        dfdz = (
            tmp_a * rho[:, :, 2:] * q[:, :, 2:] * vt[:, :, 2:]
            + tmp_b * rho[:, :, 1:-1] * q[:, :, 1:-1] * vt[:, :, 1:-1]
            + tmp_c * rho[:, :, :-2] * q[:, :, :-2] * vt[:, :, :-2]
        )

        return dfdz

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux")
    @gtscript.function
    def call_gt4py(rho, h, q, vt):
        # evaluate the space-dependent coefficients occurring in the
        # second-order upwind finite difference approximation of the
        # vertical derivative of the flux
        tmp_a = (2.0 * h[0, 0, 0] - h[0, 0, -1] - h[0, 0, -2]) / (
            (h[0, 0, -1] - h[0, 0, 0]) * (h[0, 0, -2] - h[0, 0, 0])
        )
        tmp_b = (h[0, 0, -2] - h[0, 0, 0]) / (
            (h[0, 0, -1] - h[0, 0, 0]) * (h[0, 0, -2] - h[0, 0, -1])
        )
        tmp_c = (h[0, 0, 0] - h[0, 0, -1]) / (
            (h[0, 0, -2] - h[0, 0, 0]) * (h[0, 0, -2] - h[0, 0, -1])
        )

        # calculate the vertical derivative of the sedimentation flux
        dfdz = (
            tmp_a[0, 0, 0] * rho[0, 0, 0] * q[0, 0, 0] * vt[0, 0, 0]
            + tmp_b[0, 0, 0] * rho[0, 0, -1] * q[0, 0, -1] * vt[0, 0, -1]
            + tmp_c[0, 0, 0] * rho[0, 0, -2] * q[0, 0, -2] * vt[0, 0, -2]
        )

        return dfdz
