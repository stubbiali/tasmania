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
from gt4py.gtscript import IJ

from tasmania.python.framework.allocators import as_storage, zeros
from tasmania.python.framework.options import StorageOptions


def fma_defs(
    a: gtscript.Field[float],
    b: gtscript.Field[float],
    c: gtscript.Field[IJ, float],
    out: gtscript.Field[float],
):
    with computation(PARALLEL), interval(...):
        out = a + b * c


if __name__ == "__main__":
    fma = gtscript.stencil("numpy", fma_defs)

    so_ij = StorageOptions(dtype=float, aligned_index=(0, 0))
    so_ijk = StorageOptions(dtype=float, aligned_index=(0, 0, 0))

    tmp_np = np.random.rand(10, 10, 10)
    a = as_storage("gt4py:numpy", data=tmp_np, storage_options=so_ijk)
    b = as_storage("gt4py:numpy", data=tmp_np, storage_options=so_ijk)
    c = as_storage("gt4py:numpy", data=tmp_np[:, :, 1], storage_options=so_ij)
    out = zeros("gt4py:numpy", shape=(10, 10, 10), storage_options=so_ijk)

    fma(a, b, c, out, origin=(0, 0, 0), domain=(10, 10, 10))

    assert np.allclose(a + b * c[:, :, np.newaxis], out)
