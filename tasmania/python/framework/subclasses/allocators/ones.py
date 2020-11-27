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

try:
    import cupy as cp
except ImportError:
    cp = np

import gt4py as gt

from tasmania.python.framework.allocators import ones
from tasmania.python.utils.utils import get_gt_backend


@ones.register(backend=("numpy", "numba:cpu"))
def ones_numpy(shape, dtype, **kwargs):
    return np.ones(shape, dtype=dtype)


@ones.register(backend=("cupy", "numba:gpu"))
def ones_cupy(shape, dtype, **kwargs):
    return cp.ones(shape, dtype=dtype)


@ones.register(backend="gt4py*")
def ones_gt4py(
    shape,
    dtype,
    default_origin=None,
    mask=None,
    managed_memory=False,
    **kwargs
):
    backend = ones_gt4py.__tasmania_runtime__["backend"]
    gt_backend = get_gt_backend(backend)
    default_origin = default_origin or (0,) * len(shape)
    return gt.storage.ones(
        gt_backend,
        default_origin,
        shape,
        dtype,
        mask=mask,
        managed_memory=managed_memory,
    )
