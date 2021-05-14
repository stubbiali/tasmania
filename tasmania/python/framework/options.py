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
from dataclasses import dataclass, field
import numpy as np
from typing import Any, Mapping, Sequence, TYPE_CHECKING, Type, Union

if TYPE_CHECKING:
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

    from tasmania.python.framework.concurrent_coupling import (
        ConcurrentCoupling,
    )


@dataclass
class BackendOptions:
    # gt4py
    backend_opts: Mapping[str, Any] = None
    build_info: Mapping[str, Any] = None
    dtypes: Mapping[str, Type] = field(default_factory=dict)
    exec_info: Mapping[str, Any] = None
    externals: Mapping[str, Any] = field(default_factory=dict)
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
    dtype: Type = np.float64

    # gt4py
    aligned_index: Sequence[int] = None
    halo: Sequence[int] = None
    managed: Union[bool, str] = "gt4py"


@dataclass
class TimeIntegrationOptions:
    # mandatory
    component: Union[
        "DiagnosticComponent",
        "DiagnosticComponentComposite",
        "ImplicitTendencyComponent",
        "ImplicitTendencyComponentComposite",
        "TendencyComponent",
        "TendencyComponentComposite",
        "ConcurrentCoupling",
    ] = None

    # optional
    scheme: str = None
    enforce_horizontal_boundary: bool = False
    substeps: int = 1
    backend: str = "numpy"
    backend_options: BackendOptions = field(default_factory=BackendOptions)
    storage_options: StorageOptions = field(default_factory=StorageOptions)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
