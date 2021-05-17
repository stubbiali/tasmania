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
from functools import singledispatch
import numpy as np
from typing import Optional, TYPE_CHECKING

from tasmania.third_party import cupy as cp, gt4py as gt, numba

from tasmania.python.framework.allocators import as_storage
from tasmania.python.framework.options import StorageOptions

if TYPE_CHECKING:
    from tasmania.python.utils.typingx import Storage


@as_storage.register(backend="numpy")
@singledispatch
def as_storage_numpy(
    data: "Storage", *, storage_options: Optional[StorageOptions] = None
) -> np.ndarray:
    pass


@as_storage_numpy.register
def _(
    data: np.ndarray, *, storage_options: Optional[StorageOptions] = None
) -> np.ndarray:
    return data


if numba:
    as_storage.register(as_storage_numpy, backend="numba:cpu")


if cp:

    @as_storage_numpy.register
    def _(
        data: cp.ndarray, *, storage_options: Optional[StorageOptions] = None
    ) -> np.ndarray:
        return data.get()


if gt:

    @as_storage_numpy.register
    def _(
        data: gt.storage.Storage,
        *,
        storage_options: Optional[StorageOptions] = None
    ) -> np.ndarray:
        data.synchronize()
        # >>> old storage
        return data
        # <<< new storage
        # return data.to_numpy()
