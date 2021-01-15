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

from tasmania.python.framework.asarray import asarray


@asarray.register(backend="numpy")
def asarray_numpy():
    return np.asarray


if cp:

    @asarray.register(backend="cupy")
    def asarray_cupy():
        return cp.asarray


if gt:
    from tasmania.python.utils.backend import get_gt_backend

    @asarray.register(backend="gt4py*")
    def asarray_gt4py():
        backend = asarray_gt4py.__tasmania_runtime__["backend"]
        gt_backend = get_gt_backend(backend)
        device = gt.backend.from_name(gt_backend).storage_info["device"]
        return cp.asarray if device == "gpu" else np.asarray


if numba:
    asarray.register(asarray_numpy, "numba:cpu")
