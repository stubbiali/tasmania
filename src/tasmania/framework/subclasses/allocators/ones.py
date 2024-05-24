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
from tasmania.framework.allocators import ones
from tasmania.framework.options import StorageOptions
from tasmania.utils.gt4pyx import get_gt_backend


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


@ones.register(backend="gt4py*")
def ones_gt4py(shape, *, storage_options=None):
    backend = ones_gt4py.__tasmania_runtime__["backend"]
    gt_backend = get_gt_backend(backend)
    so = storage_options or StorageOptions
    return gt_storage.ones(shape, dtype=so.dtype, backend=gt_backend)
