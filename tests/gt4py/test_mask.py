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

from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, interval

from tasmania.python.utils.storage_utils import zeros


def stencil_defs(
    in_a: gtscript.Field["dtype"],
    in_b: gtscript.Field["dtype"],
    out_c: gtscript.Field["dtype"],
):
    with computation(PARALLEL), interval(...):
        out_c = in_a + in_b


if __name__ == "__main__":
    nx = 25
    ny = 31
    nz = 16

    backend = "numpy"
    dtype = np.float64

    a = zeros(
        (nx, ny, 1),
        gt_powered=True,
        backend=backend,
        dtype=dtype,
        mask=(True, True, False),
    )
    a[...] = np.random.rand(*a.shape)
    b = zeros((nx, ny, nz), gt_powered=True, backend=backend, dtype=dtype)
    b[...] = np.random.rand(*b.shape)
    c = zeros((nx, ny, nz), gt_powered=True, backend=backend, dtype=dtype)

    stencil = gtscript.stencil(
        definition=stencil_defs, backend=backend, dtypes={"dtype": dtype}, rebuild=False
    )

    stencil(
        a,
        b,
        c,
        origin={"a": (0, 0), "b": (0, 0, 0), "c": (0, 0, 0)},
        domain=(nx, ny, nz),
    )

    print("Completed successfully.")
