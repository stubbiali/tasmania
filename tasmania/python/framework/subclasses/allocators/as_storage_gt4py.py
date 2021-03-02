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
from typing import Optional, TYPE_CHECKING

from tasmania.third_party import gt4py as gt

from tasmania.python.framework.allocators import as_storage
from tasmania.python.framework.options import StorageOptions
from tasmania.python.utils.backend import get_gt_backend

if TYPE_CHECKING:
    from tasmania.python.utils.typing import Storage


if gt:

    @as_storage.register(backend="gt4py*")
    def as_storage_gt4py(
        data: "Storage", *, storage_options: Optional[StorageOptions] = None
    ) -> gt.storage.Storage:
        backend = as_storage_gt4py.__tasmania_runtime__["backend"]
        gt_backend = get_gt_backend(backend)
        so = storage_options or StorageOptions
        default_origin = so.default_origin or (0,) * len(data.shape)
        return gt.storage.from_array(
            data,
            backend=gt_backend,
            default_origin=default_origin,
            dtype=so.dtype,
        )
