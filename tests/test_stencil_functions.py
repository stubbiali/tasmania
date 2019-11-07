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

from gridtools import __externals__
from gridtools import gtscript

# from gridtools.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.utils.storage_utils import zeros


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
    lap = lap_x[0, 0, 0] + lap_y[0, 0, 0]
    return lap


def hyperdiffusion_defs(
    in_phi: gtscript.Field[np.float64],
    in_gamma: gtscript.Field[np.float64],
    out_phi: gtscript.Field[np.float64],
    *,
    dx: float,
    dy: float
):
    from __externals__ import stage_laplacian, stage_laplacian_x, stage_laplacian_y

    with computation(PARALLEL), interval(...):
        lap = stage_laplacian(dx=dx, dy=dy, phi=in_phi)
        lap1 = stage_laplacian(dx=dx, dy=dy, phi=lap)
        out_phi = in_gamma[0, 0, 0] * lap1[0, 0, 0]


if __name__ == "__main__":
    decorator = gtscript.stencil(
        "numpy",
        externals={
            "stage_laplacian": _stage_laplacian,
            "stage_laplacian_x": _stage_laplacian_x,
            "stage_laplacian_y": _stage_laplacian_y,
        },
        rebuild=True,
    )
    hyperdiffusion = decorator(hyperdiffusion_defs)

    in_phi = zeros((30, 30, 10), "numpy", np.float64)
    in_gamma = zeros((30, 30, 10), "numpy", np.float64)
    out_phi = zeros((30, 30, 10), "numpy", np.float64)

    hyperdiffusion(
        in_phi=in_phi,
        in_gamma=in_gamma,
        out_phi=out_phi,
        dx=1.0,
        dy=1.0,
        origin=(2, 2, 0),
        domain=(26, 26, 10)
    )