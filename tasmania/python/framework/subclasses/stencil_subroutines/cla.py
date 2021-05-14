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
from typing import Optional, Tuple, Union

from tasmania.third_party import cupy, gt4py

from tasmania.python.framework.stencil import stencil_subroutine
from tasmania.python.utils import typingx


@stencil_subroutine.register(backend="numpy", stencil="thomas")
def thomas_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    out: np.ndarray,
    beta: Optional[np.ndarray] = None,
    delta: Optional[np.ndarray] = None,
    *,
    i: Union[int, slice],
    j: Union[int, slice],
    kstart: int,
    kstop: int
):
    """The Thomas' algorithm to solve a tridiagonal system of equations."""
    beta = beta if beta is not None else np.copy(b, subok=True)
    delta = delta if delta is not None else np.copy(d, subok=True)
    beta[i, j, kstart] = b[i, j, kstart]
    delta[i, j, kstart] = d[i, j, kstart]
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


@stencil_subroutine.register(backend="numpy", stencil="setup_thomas")
@stencil_subroutine.register(backend="numpy", stencil="setup_thomas_bc")
def setup_tridiagonal_system_numpy(
    gamma: float,
    w: np.ndarray,
    phi: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    i: Union[int, slice],
    j: Union[int, slice],
    kstart: int,
    kstop: int
) -> None:
    a[i, j, kstart + 1 : kstop - 1] = gamma * w[i, j, kstart : kstop - 2]
    a[i, j, kstop - 1] = 0.0

    c[i, j, kstart] = 0.0
    c[i, j, kstart + 1 : kstop - 1] = -gamma * w[i, j, kstart + 2 : kstop]

    d[i, j, kstart] = phi[i, j, kstart]
    d[i, j, kstart + 1 : kstop - 1] = phi[
        i, j, kstart + 1 : kstop - 1
    ] - gamma * (
        w[i, j, kstart : kstop - 2] * phi[i, j, kstart : kstop - 2]
        - w[i, j, kstart + 2 : kstop] * phi[i, j, kstart + 2 : kstop]
    )
    d[i, j, kstop - 1] = phi[i, j, kstop - 1]


if cupy:
    stencil_subroutine.register(thomas_numpy, "cupy", "thomas")
    stencil_subroutine.register(
        setup_tridiagonal_system_numpy,
        "cupy",
        ("setup_thomas", "setup_thomas_bc"),
    )


if gt4py:
    from gt4py import gtscript

    @stencil_subroutine.register(backend="gt4py*", stencil="setup_thomas")
    @gtscript.function
    def setup_tridiagonal_system_gt4py(
        gamma: float, w: typingx.GTField, phi: typingx.GTField
    ) -> "Tuple[typingx.GTField, typingx.GTField, typingx.GTField]":
        a = gamma * w[0, 0, -1]
        c = -gamma * w[0, 0, 1]
        d = phi[0, 0, 0] - gamma * (
            w[0, 0, -1] * phi[0, 0, -1] - w[0, 0, 1] * phi[0, 0, 1]
        )
        return a, c, d

    @stencil_subroutine.register(backend="gt4py*", stencil="setup_thomas_bc")
    @gtscript.function
    def setup_tridiagonal_system_bc_gt4py(
        phi: typingx.GTField,
    ) -> "Tuple[typingx.GTField, typingx.GTField, typingx.GTField]":
        a = 0.0
        c = 0.0
        d = phi[0, 0, 0]
        return a, c, d
