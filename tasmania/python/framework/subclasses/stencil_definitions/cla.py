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
import numpy as np

from tasmania.third_party import cupy, gt4py, numba

from tasmania.python.framework.stencil import stencil_definition


@stencil_definition.register(backend="numpy", stencil="thomas")
def thomas_numpy(a, b, c, d, out, *, origin, domain):
    """The Thomas' algorithm to solve a tridiagonal system of equations."""
    i, j = [slice(o, o + d) for o, d in zip(origin[:2], domain[:2])]
    kstart, kstop = origin[2], origin[2] + domain[2]

    beta = deepcopy(b)
    delta = deepcopy(d)
    for k in range(kstart + 1, kstop):
        w = np.where(
            beta[i, j, k - 1] != 0.0,
            a[i, j, k] / beta[i, j, k - 1],
            a[i, j, k],
        )
        beta[i, j, k] -= w * c[i, j, k - 1]
        delta[i, j, k] -= w * delta[i, j, k - 1]

    out[i, j, kstop - 1] = np.where(
        beta[i, j, kstop - 1] != 0.0,
        delta[i, j, kstop - 1] / beta[i, j, kstop - 1],
        delta[i, j, kstop - 1] / b[i, j, kstop - 1],
    )
    for k in range(kstop - 2, kstart - 1, -1):
        out[i, j, k] = np.where(
            beta[i, j, k] != 0.0,
            (delta[i, j, k] - c[i, j, k] * out[i, j, k + 1]) / beta[i, j, k],
            (delta[i, j, k] - c[i, j, k] * out[i, j, k + 1]) / b[i, j, k],
        )


if True:  # cupy:
    stencil_definition.register(thomas_numpy, "cupy", "thomas")


if gt4py:
    from gt4py import gtscript

    @stencil_definition.register(backend="gt4py*", stencil="thomas")
    def thomas_gt4py(
        a: gtscript.Field["dtype"],
        b: gtscript.Field["dtype"],
        c: gtscript.Field["dtype"],
        d: gtscript.Field["dtype"],
        x: gtscript.Field["dtype"],
    ) -> None:
        # """The Thomas' algorithm to solve a tridiagonal system of equations."""
        with computation(FORWARD), interval(0, 1):
            w = 0.0
            beta = b[0, 0, 0]
            delta = d[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            w = (
                a[0, 0, 0] / beta[0, 0, -1]
                if beta[0, 0, -1] != 0.0
                else a[0, 0, 0]
            )
            beta = b[0, 0, 0] - w[0, 0, 0] * c[0, 0, -1]
            delta = d[0, 0, 0] - w[0, 0, 0] * delta[0, 0, -1]

        with computation(BACKWARD), interval(-1, None):
            x = (
                delta[0, 0, 0] / beta[0, 0, 0]
                if beta[0, 0, 0] != 0.0
                else delta[0, 0, 0] / b[0, 0, 0]
            )
        with computation(BACKWARD), interval(0, -1):
            x = (
                (delta[0, 0, 0] - c[0, 0, 0] * x[0, 0, 1]) / beta[0, 0, 0]
                if beta[0, 0, 0] != 0.0
                else (delta[0, 0, 0] - c[0, 0, 0] * x[0, 0, 1]) / b[0, 0, 0]
            )


if numba:
    stencil_definition.register(thomas_numpy, "numba:cpu:numpy", "thomas")
