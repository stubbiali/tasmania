# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2012-2019, ETH Zurich
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
import matplotlib.pyplot as plt
import numpy as np
import tasmania as taz

from example01 import gradx, grady, laplacian
from utils import copy_defs, get_timer, update_halo


def diffusion_defs(in_phi: gtscript.Field["dtype"], out_phi: gtscript.Field["dtype"]):
    from __gtscript__ import PARALLEL, computation, interval
    from __externals__ import ALPHA, gradx, grady, laplacian

    with computation(PARALLEL), interval(...):
        # compute the laplacian-of-laplacian
        lap = laplacian(in_phi)
        bilap = laplacian(lap)

        # compute the x- and y-flux
        flux_x = gradx(bilap)
        flux_y = grady(bilap)

        # compute the gradient of the input field
        grad_x = gradx(in_phi)
        grad_y = grady(in_phi)

        # apply a simple flux limiter
        fx = flux_x if flux_x * grad_x > 0 else 0
        fy = flux_y if flux_y * grad_y > 0 else 0

        # update the field
        out_phi = in_phi + ALPHA * (gradx(fx[-1, 0, 0]) + grady(fy[0, -1, 0]))


if __name__ == "__main__":
    # gt4py settings
    backend = "gtcuda"
    dtype = np.float64
    verbose = True

    # domain size
    nx = 256
    ny = 256
    nz = 128

    # boundary layers: no smaller than 3
    nb = 3

    # diffusion coefficient
    alpha = 1.0 / 1024.0

    # number of iterations
    nt = 100

    # allocate storages
    shape = (nx + 2 * nb, ny + 2 * nb, nz)
    in_phi = taz.zeros(shape, gt_powered=True, backend=backend, dtype=dtype)
    out_phi = taz.zeros(shape, gt_powered=True, backend=backend, dtype=dtype)

    # set initial conditions
    in_phi[nb + nx // 4 : nb + 3 * nx // 4, nb + ny // 4 : nb + 3 * ny // 4] = 1.0

    # compile stencils
    copy = gtscript.stencil(backend, copy_defs, dtypes={"dtype": dtype}, verbose=verbose)
    diffusion = gtscript.stencil(
        backend,
        diffusion_defs,
        dtypes={"dtype": dtype},
        externals={
            "ALPHA": alpha,
            "gradx": gradx,
            "grady": grady,
            "laplacian": laplacian,
        },
        verbose=verbose,
    )

    # warm up caches
    diffusion(in_phi, out_phi, origin=(nb, nb, 0), domain=(nx, ny, nz))

    # start timing
    tic = get_timer()

    # time integration
    for _ in range(nt):
        # apply diffusion
        diffusion(in_phi, out_phi, origin=(nb, nb, 0), domain=(nx, ny, nz))

        # set boundaries
        update_halo(copy, out_phi, nb)

        # swap fields
        in_phi, out_phi = out_phi, in_phi

    # stop timing
    toc = get_timer()

    print(f"Execution time: {toc - tic} s")

    # plot
    # plt.ioff()
    # plt.imshow(np.asarray(out_phi[nb:-nb, nb:-nb, 0]), origin="lower")
    # plt.colorbar()
    # plt.show()
