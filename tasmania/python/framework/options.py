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
from dataclasses import dataclass
import numpy as np
from typing import Any, Mapping, Sequence, Type


@dataclass
class BackendOptions:
    # gt4py
    backend_opts: Mapping[str, Any] = None
    build_info: Mapping[str, Any] = None
    dtypes: Mapping[str, Type] = None
    exec_info: Mapping[str, Any] = None
    externals: Mapping[str, Any] = None
    rebuild: bool = False

    # numba
    parallel: bool = True


@dataclass
class StorageOptions:
    # generic
    dtype: Type = np.float64

    # gt4py
    default_origin: Sequence[int] = None
    managed_memory: bool = False
    mask: Sequence[bool] = None
