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
import datetime as dt
import numpy as np
import pandas as pd
from sympl import (
    DataArray,
    DiagnosticComponent,
    DiagnosticComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
)
from typing import (
    Any,
    Dict,
    Mapping,
    MutableMapping,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)

try:
    import cupy as cp
except ImportError:
    cp = np

from gt4py import gtscript
from gt4py.storage.storage import Storage

if TYPE_CHECKING:
    from tasmania.python.framework._base import (
        BaseConcurrentCoupling,
        BaseDiagnostic2Tendency,
        BaseDiagnosticComponentComposite,
        BaseTendency2Diagnostic,
    )


array_t = Union[np.ndarray, cp.ndarray, Storage]
dataarray_t = DataArray
gtstorage_t = Storage
timedelta_t = Union[dt.timedelta, pd.Timedelta]

array_dict_t = Dict[str, Union[timedelta_t, array_t]]
dataarray_dict_t = Dict[str, Union[timedelta_t, dataarray_t]]
datetime_t = Union[dt.datetime, pd.Timestamp]
diagnostic_component_t = Union[
    DiagnosticComponent, DiagnosticComponentComposite, "BaseDiagnosticComponentComposite"
]
dtype_t = type
gtfield_t = gtscript.Field["dtype"]
gtstorage_dict_t = Dict[str, Union[timedelta_t, gtstorage_t]]
mutable_array_dict_t = Dict[str, Union[timedelta_t, array_t]]
mutable_dataarray_dict_t = Dict[str, Union[timedelta_t, dataarray_t]]
mutable_gtstorage_dict_t = Dict[str, Union[timedelta_t, gtstorage_t]]
mutable_options_dict_t = Dict[str, Any]
options_dict_t = Dict[str, Any]
pair_int_t = Tuple[int, int]
promoter_component_t = Union["BaseDiagnostic2Tendency", "BaseTendency2Diagnostic"]
properties_dict_t = Dict[str, Any]
properties_mapping_t = Union[Mapping[str, Any], MutableMapping[str, Any]]
tendency_component_t = Union[
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    "BaseConcurrentCoupling",
]
triplet_bool_t = Union[Tuple[bool, bool, bool], Sequence[bool]]
triplet_int_t = Union[Tuple[int, int, int], Sequence[int]]

component_t = Union[diagnostic_component_t, promoter_component_t, tendency_component_t]
