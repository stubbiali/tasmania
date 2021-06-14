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

from tasmania.third_party import cupy as cp, gt4py as gt, numba

from tasmania.python.framework.allocators import ones
from tasmania.python.framework.options import StorageOptions


@ones.register(backend="numpy")
def ones_numpy(shape, *, storage_options=None):
    so = storage_options or StorageOptions
    return np.ones(shape, dtype=so.dtype)


if numba:
    ones.register(ones_numpy, backend="numba:cpu*")


if cp:

    @ones.register(backend="cupy")
    def ones_cupy(shape, *, storage_options=None):
        so = storage_options or StorageOptions
        return cp.ones(shape, dtype=so.dtype)

    if numba:
        ones.register(ones_cupy, backend="numba:gpu")


if gt:
    from tasmania.python.utils.backend import get_gt_backend

    @ones.register(backend="gt4py*")
    def ones_gt4py(shape, *, storage_options=None):
        backend = ones_gt4py.__tasmania_runtime__["backend"]
        defaults = get_gt_backend(backend)
        so = storage_options or StorageOptions
        # >>> old storage
        # return gt.storage.ones(
        #     defaults,
        #     so.aligned_index,
        #     shape,
        #     dtype=so.dtype,
        # )
        # <<< new storage
        return gt.storage.ones(
            shape,
            dtype=so.dtype,
            aligned_index=so.aligned_index,
            defaults=defaults,
            halo=so.halo,
            managed=so.managed,
        )
        # <<<
