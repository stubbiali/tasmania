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
    ones.register(ones_numpy, backend="numba:cpu")


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
        gt_backend = get_gt_backend(backend)
        so = storage_options or StorageOptions
        default_origin = so.default_origin or (0,) * len(shape)
        return gt.storage.ones(
            gt_backend,
            default_origin,
            shape,
            so.dtype,
            mask=so.mask,
            managed_memory=so.managed_memory,
        )
