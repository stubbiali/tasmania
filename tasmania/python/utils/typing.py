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
    TypeVar,
    Union,
)

try:
    import cupy as cp
except ImportError:
    cp = np

from gt4py import gtscript, storage as gt_storage

if TYPE_CHECKING:
    from tasmania.python.framework._base import (
        BaseConcurrentCoupling,
        BaseDiagnostic2Tendency,
        BaseDiagnosticComponentComposite,
        BaseTendency2Diagnostic,
    )
    from tasmania.python.framework.sts_tendency_stepper import (
        STSTendencyStepper,
    )
    from tasmania.python.framework.tendency_stepper import TendencyStepper

Datatype = TypeVar("Datatype")
Datetime = Union[dt.datetime, pd.Timestamp]
Storage = TypeVar("Storage")
StorageDict = Dict[str, Union[Datetime, Storage]]
DataArrayDict = Dict[str, Union[Datetime, DataArray]]
TimeDelta = Union[dt.timedelta, pd.Timedelta]
PropertiesDict = Dict[str, Dict[str, Any]]

array_t = Union[np.ndarray, cp.ndarray, gt_storage.Storage]
dataarray_t = DataArray
gtstorage_t = gt_storage.Storage

DiagnosticComponent = Union[
    DiagnosticComponent,
    DiagnosticComponentComposite,
    "BaseDiagnosticComponentComposite",
]
dtype_t = type
gtfield_t = gtscript.Field["dtype"]
gtstorage_dict_t = Dict[str, Union[TimeDelta, gtstorage_t]]
mutable_array_dict_t = Dict[str, Union[TimeDelta, array_t]]
mutable_dataarray_dict_t = Dict[str, Union[TimeDelta, dataarray_t]]
mutable_gtstorage_dict_t = Dict[str, Union[TimeDelta, gtstorage_t]]
mutable_options_dict_t = Dict[str, Any]
options_dict_t = Dict[str, Any]
PairInt = Tuple[int, int]
PromoterComponent = Union["BaseDiagnostic2Tendency", "BaseTendency2Diagnostic"]
properties_mapping_t = Union[Mapping[str, Any], MutableMapping[str, Any]]
TendencyComponent = Union[
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    "BaseConcurrentCoupling",
]
TripletBool = Union[Tuple[bool, bool, bool], Sequence[bool]]
TripletInt = Union[Tuple[int, int, int], Sequence[int]]

Component = Union[
    DiagnosticComponent,
    PromoterComponent,
    "STSTendencyStepper",
    TendencyComponent,
    "TendencyStepper",
]
