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

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import StorageOptions


@zeros.register(backend="numpy")
def zeros_numpy(shape, *, storage_options=None):
    so = storage_options or StorageOptions()
    return np.zeros(shape, dtype=so.dtype)


if numba:
    zeros.register(zeros_numpy, backend="numba:cpu*")


if cp:

    @zeros.register(backend="cupy")
    def zeros_cupy(shape, *, storage_options=None):
        so = storage_options or StorageOptions()
        return cp.zeros(shape, dtype=so.dtype)

    if numba:
        zeros.register(zeros_cupy, backend="numba:gpu")


if gt:
    from tasmania.python.utils.backend import get_gt_backend

    @zeros.register(backend="gt4py*")
    def zeros_gt4py(shape, *, storage_options=None):
        backend = zeros_gt4py.__tasmania_runtime__["backend"]
        defaults = get_gt_backend(backend)
        so = storage_options or StorageOptions()
        # >>> old storage
        if len(shape) == 1:
            mask = (False, False, True)
        elif len(shape) == 2:
            mask = (True, True, False)
        else:
            mask = (True, True, True)
        return gt.storage.zeros(
            defaults, so.aligned_index, shape, dtype=so.dtype, mask=mask
        )
        # <<< new storage
        # return gt.storage.zeros(
        #     shape,
        #     dtype=so.dtype,
        #     aligned_index=so.aligned_index,
        #     defaults=defaults,
        #     halo=so.halo,
        #     managed=so.managed,
        # )


# if ti:
#     @zeros.register(backend="taichi:*")
#     def zeros_taichi(shape, *, storage_options=None):
#         backend = zeros_taichi.__tasmania_runtime__["backend"]
#         exec(f"ti.init(arch=ti.{get_ti_arch(backend)})")
#         so = storage_options or StorageOptions()
#         return ti.field(so.dtype, shape=shape)
