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
import numpy as np

from gt4py import gtscript

from tasmania.python.framework.allocators import as_storage, zeros
from tasmania.python.framework.options import StorageOptions


def stencil_numpy(a, b, *, x):
    a_avg = np.zeros_like(a)
    a_avg[:, :, 1:-1] = 0.5 * (a[:, :, :-2] + a[:, :, 2:])
    c = a + x * b ** 2
    d = np.exp(a_avg) * np.sin(b)
    out = c / d
    return out


def stencil_def(
    a: gtscript.Field["dtype"],
    b: gtscript.Field["dtype"],
    out: gtscript.Field["dtype"],
    *,
    x: float
):
    with computation(PARALLEL):
        with interval(0, 1):
            a_avg = 0.0
        with interval(1, -1):
            a_avg = 0.5 * (a[0, 0, -1] + a[0, 0, 1])
        with interval(-1, None):
            a_avg = 0.0

    with computation(PARALLEL), interval(...):
        c = a + x * b ** 2
        d = exp(a_avg) * sin(b)
        out = c / d


def main(backend, storage_shape, storage_options):
    stencil = gtscript.stencil(
        backend, stencil_def, dtypes={"dtype": storage_options.dtype}
    )

    a_np = np.zeros(storage_shape, dtype=storage_options.dtype)
    a_np[...] = np.random.rand(*storage_shape)
    a = as_storage(
        "gt4py:" + backend, data=a_np, storage_options=storage_options
    )
    b_np = np.zeros(storage_shape, dtype=storage_options.dtype)
    b_np[...] = np.random.rand(*storage_shape)
    b = as_storage(
        "gt4py:" + backend, data=b_np, storage_options=storage_options
    )

    out = zeros(
        "gt4py:" + backend,
        shape=storage_shape,
        storage_options=storage_options,
    )

    out_np = stencil_numpy(a_np, b_np, x=0.333)
    nx, ny, nz = storage_shape
    stencil(a, b, out, x=0.333, origin=(0, 0, 0), domain=storage_shape)

    assert np.all(np.equal(out, out_np))


if __name__ == "__main__":
    main("numpy", (5, 5, 1), StorageOptions(dtype=np.float64))
    main("numpy", (5, 5, 1), StorageOptions(dtype=np.float32))
