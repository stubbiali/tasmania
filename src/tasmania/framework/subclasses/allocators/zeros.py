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

from gt4py import storage as gt_storage

from tasmania.externals import cupy as cp, numba
from tasmania.framework.allocators import zeros
from tasmania.framework.options import StorageOptions
from tasmania.utils.gt4pyx import get_gt_backend


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


@zeros.register(backend="gt4py*")
def zeros_gt4py(shape, *, storage_options=None):
    backend = zeros_gt4py.__tasmania_runtime__["backend"]
    gt_backend = get_gt_backend(backend)
    so = storage_options or StorageOptions()
    return gt_storage.zeros(shape, dtype=so.dtype, backend=gt_backend)
