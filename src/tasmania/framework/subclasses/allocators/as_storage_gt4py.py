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

from __future__ import annotations
from functools import singledispatch
import numpy as np
from typing import TYPE_CHECKING

from gt4py import storage as gt_storage

from tasmania.externals import cupy as cp
from tasmania.framework.allocators import as_storage
from tasmania.framework.options import StorageOptions
from tasmania.utils.gt4pyx import get_gt_backend

if TYPE_CHECKING:
    from typing import Optional

    from tasmania.utils.typingx import NDArray


@as_storage.register(backend="gt4py*")
@singledispatch
def as_storage_gt4py(data: NDArray, *, storage_options: Optional[StorageOptions] = None) -> NDArray:
    pass


@as_storage_gt4py.register
def _(data: np.ndarray, *, storage_options: Optional[StorageOptions] = None) -> NDArray:
    backend = as_storage_gt4py.__tasmania_runtime__["backend"]
    gt_backend = get_gt_backend(backend)
    so = storage_options or StorageOptions
    return gt_storage.from_array(
        data, dtype=so.dtype, backend=gt_backend, aligned_index=so.aligned_index
    )


if cp:

    @as_storage_gt4py.register
    def _(data: cp.ndarray, *, storage_options: Optional[StorageOptions] = None) -> NDArray:
        backend = as_storage_gt4py.__tasmania_runtime__["backend"]
        gt_backend = get_gt_backend(backend)
        so = storage_options or StorageOptions
        return gt_storage.from_array(
            data, dtype=so.dtype, backend=gt_backend, aligned_index=so.aligned_index
        )
