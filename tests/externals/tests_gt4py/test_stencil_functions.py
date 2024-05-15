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
import numpy as np

from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, interval

from tasmania.python.utils.storage import zeros


@gtscript.function
def _stage_laplacian_x(dx, phi):
    lap = (phi[-1, 0, 0] - 2.0 * phi[0, 0, 0] + phi[1, 0, 0]) / (dx * dx)
    return lap


@gtscript.function
def _stage_laplacian_y(dy, phi):
    lap = (phi[0, -1, 0] - 2.0 * phi[0, 0, 0] + phi[0, 1, 0]) / (dy * dy)
    return lap


@gtscript.function
def _stage_laplacian(dx, dy, phi):
    from __externals__ import stage_laplacian_x, stage_laplacian_y

    lap_x = stage_laplacian_x(dx=dx, phi=phi)
    lap_y = stage_laplacian_y(dy=dy, phi=phi)
    lap = lap_x + lap_y
    return lap


def hyperdiffusion_defs(
    in_phi: gtscript.Field["dtype"],
    in_gamma: gtscript.Field["dtype"],
    out_phi: gtscript.Field["dtype"],
    *,
    dx: float,
    dy: float
):
    from __externals__ import (
        stage_laplacian,
        stage_laplacian_x,
        stage_laplacian_y,
    )

    with computation(PARALLEL), interval(...):
        lap = stage_laplacian(dx=dx, dy=dy, phi=in_phi)
        lap1 = stage_laplacian(dx=dx, dy=dy, phi=lap)
        out_phi = in_gamma * lap1


if __name__ == "__main__":
    dtype = np.float64

    decorator = gtscript.stencil(
        "numpy",
        dtypes={"dtype": dtype},
        externals={
            "stage_laplacian": _stage_laplacian,
            "stage_laplacian_x": _stage_laplacian_x,
            "stage_laplacian_y": _stage_laplacian_y,
        },
        rebuild=False,
    )
    hyperdiffusion = decorator(hyperdiffusion_defs)

    in_phi = zeros((30, 30, 10), gt_powered=True, backend="numpy", dtype=dtype)
    in_gamma = zeros(
        (30, 30, 10), gt_powered=True, backend="numpy", dtype=dtype
    )
    out_phi = zeros(
        (30, 30, 10), gt_powered=True, backend="numpy", dtype=dtype
    )

    hyperdiffusion(
        in_phi=in_phi,
        in_gamma=in_gamma,
        out_phi=out_phi,
        dx=1.0,
        dy=1.0,
        origin=(2, 2, 0),
        domain=(26, 26, 10),
    )
