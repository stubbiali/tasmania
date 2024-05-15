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
from functools import singledispatch
import numpy as np
from typing import Optional, TYPE_CHECKING

from tasmania.third_party import cupy as cp, gt4py as gt

from tasmania.python.framework.allocators import as_storage
from tasmania.python.framework.options import StorageOptions
from tasmania.python.utils.backend import get_gt_backend

if TYPE_CHECKING:
    from tasmania.python.utils.typingx import Storage


if gt:

    # >>> old storage
    # <<< new storage
    # from gt4py.storage.definitions import SyncState

    # <<<

    @as_storage.register(backend="gt4py*")
    @singledispatch
    def as_storage_gt4py(
        data: "Storage", *, storage_options: Optional[StorageOptions] = None
    ) -> gt.storage.Storage:
        pass

    @as_storage_gt4py.register
    def _(
        data: np.ndarray, *, storage_options: Optional[StorageOptions] = None
    ) -> gt.storage.Storage:
        backend = as_storage_gt4py.__tasmania_runtime__["backend"]
        defaults = get_gt_backend(backend)
        so = storage_options or StorageOptions
        # >>> old storage
        return gt.storage.from_array(data, defaults, so.aligned_index)
        # <<< new storage
        # sync_state = SyncState()
        # sync_state.state = 1
        # return gt.storage.as_storage(
        #     data=data,
        #     dtype=so.dtype,
        #     aligned_index=so.aligned_index,
        #     defaults=defaults,
        #     halo=so.halo,
        #     managed=so.managed,
        #     sync_state=sync_state,
        # )
        # <<<

    @as_storage_gt4py.register
    def _(
        data: gt.storage.Storage,
        *,
        storage_options: Optional[StorageOptions] = None
    ) -> gt.storage.Storage:
        backend = as_storage_gt4py.__tasmania_runtime__["backend"]
        defaults = get_gt_backend(backend)
        so = storage_options or StorageOptions
        # >>> old storage
        return gt.storage.from_array(data, defaults, so.aligned_index)
        # <<< new storage
        # return gt.storage.as_storage(
        #     data=data,
        #     dtype=so.dtype,
        #     aligned_index=so.aligned_index,
        #     defaults=defaults,
        #     halo=so.halo,
        #     managed=so.managed,
        # )
        # <<<

    if cp:

        @as_storage_gt4py.register
        def _(
            data: cp.ndarray,
            *,
            storage_options: Optional[StorageOptions] = None
        ) -> gt.storage.Storage:
            backend = as_storage_gt4py.__tasmania_runtime__["backend"]
            defaults = get_gt_backend(backend)
            so = storage_options or StorageOptions
            # >>> old storage
            return gt.storage.from_array(data, defaults, so.aligned_index)
            # <<< new storage
            # sync_state = SyncState()
            # sync_state.state = 2
            # return gt.storage.as_storage(
            #     device_data=data,
            #     dtype=so.dtype,
            #     aligned_index=so.aligned_index,
            #     defaults=defaults,
            #     halo=so.halo,
            #     managed=so.managed,
            #     sync_state=sync_state,
            # )
            # <<<
