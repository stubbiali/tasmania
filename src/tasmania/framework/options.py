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
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import DTypeLike
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Union

    from sympl._core.core_components import (
        DiagnosticComponent,
        ImplicitTendencyComponent,
        TendencyComponent,
    )
    from sympl._core.composite import (
        DiagnosticComponentComposite,
        ImplicitTendencyComponentComposite,
        TendencyComponentComposite,
    )

    from tasmania.framework.concurrent_coupling import ConcurrentCoupling


@dataclass
class BackendOptions:
    # gt4py
    backend_opts: dict[str, Any] = None
    build_info: dict[str, Any] = None
    dtypes: dict[str, DTypeLike] = field(default_factory=dict)
    exec_info: dict[str, Any] = None
    externals: dict[str, Any] = field(default_factory=dict)
    rebuild: bool = False
    validate_args: bool = False
    verbose: bool = True

    # numba
    cache: bool = True
    check_rebuild: bool = True
    fastmath: bool = False
    inline: str = "always"
    nopython: bool = True
    parallel: bool = True

    # numba-gpu
    blockspergrid: Sequence[int] = None
    threadsperblock: Sequence[int] = None


@dataclass
class StorageOptions:
    # generic
    dtype: DTypeLike = np.float64

    # gt4py
    aligned_index: Sequence[int] = (0, 0, 0)
    halo: Sequence[int] = None
    managed: Union[bool, str] = "gt4py"


@dataclass
class TimeIntegrationOptions:
    # mandatory
    component: Union[
        DiagnosticComponent,
        DiagnosticComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        TendencyComponent,
        TendencyComponentComposite,
        ConcurrentCoupling,
    ] = None

    # optional
    scheme: str = None
    enforce_horizontal_boundary: bool = False
    substeps: int = 1
    enable_checks: bool = True
    backend: str = "numpy"
    backend_options: BackendOptions = field(default_factory=BackendOptions)
    storage_options: StorageOptions = field(default_factory=StorageOptions)
    kwargs: dict[str, Any] = field(default_factory=dict)
