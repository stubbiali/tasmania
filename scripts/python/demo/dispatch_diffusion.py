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
import matplotlib.pyplot as plt
import numpy as np

import tasmania as taz


# configuration
backend = "gt4py:gtmc"
backend_opts = {"rebuild": False, "dtypes": {"dtype": float}}
storage_shape = (128, 128, 64)
storage_opts = {"dtype": float}
num_iter = 32

# initialize fields
in_field = taz.zeros(backend=backend, shape=storage_shape, **storage_opts)
nx, ny, nz = storage_shape
in_field[nx // 4 : 3 * nx // 4, ny // 4 : 3 * ny // 4] = 1.0
out_field = taz.zeros(backend=backend, shape=storage_shape, **storage_opts)
alpha = 1 / 1024

# compile stencil
diffusion = taz.stencil_compiler(
    backend=backend, stencil="diffusion", **backend_opts
)

# run
for _ in range(num_iter):
    diffusion(
        in_field,
        out_field,
        alpha=alpha,
        origin=(3, 3, 0),
        domain=(nx - 6, ny - 6, nz),
    )
    in_field, out_field = out_field, in_field

# plot
# plt.ioff()
# plt.imshow(np.asarray(out_field[:, :, 0]), origin="lower")
# plt.colorbar()
# plt.show()
