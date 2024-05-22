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
import datetime as dt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Any, Sequence, Union

try:
    import cupy as cp
except ImportError:
    cp = np

from sympl import (
    DataArray,
    DiagnosticComponent,
    DiagnosticComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
)

from gt4py.cartesian import gtscript


Datetime = Union[dt.datetime, pd.Timestamp]
TimeDelta = Union[dt.timedelta, pd.Timedelta]
NDArray = Union[NDArray, cp.ndarray]
NDArrayDict = dict[str, Union[Datetime, NDArray]]
DataArrayDict = dict[str, Union[Datetime, DataArray]]
PropertyDict = dict[str, dict[str, Any]]

DiagnosticComponent = Union[
    DiagnosticComponent,
    DiagnosticComponentComposite,
    "BaseDiagnosticComponentComposite",
]
PromoterComponent = Union["BaseFromDiagnosticToTendency", "BaseFromTendencyToDiagnostic"]
TendencyComponent = Union[
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    "BaseConcurrentCoupling",
]
Component = Union["DynamicalCore", PromoterComponent, "STSTendencyStepper", "SymplComponent"]

PairInt = tuple[int, int]
TripletBool = Union[tuple[bool, bool, bool], Sequence[bool]]
TripletInt = Union[tuple[int, int, int], Sequence[int]]

GTField = gtscript.Field["dtype"]
