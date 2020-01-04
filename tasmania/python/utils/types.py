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
from typing import Any, Mapping, MutableMapping, Tuple, Union

from gt4py import gtscript
from gt4py.storage.storage import Storage

from tasmania.python.framework._base import (
    BaseConcurrentCoupling,
    BaseDiagnosticComponentComposite,
)
from tasmania.python.framework.promoters import Diagnostic2Tendency, Tendency2Diagnostic
from tasmania.python.grids.domain import Domain
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary
from tasmania.python.grids.horizontal_grid import HorizontalGrid
from tasmania.python.grids.grid import Grid


dataarray_t = DataArray
gtstorage_t = Storage
timedelta_t = Union[dt.timedelta, pd.Timedelta]

array_dict_t = Mapping[str, Union[timedelta_t, np.ndarray]]
dataarray_dict_t = Mapping[str, Union[timedelta_t, dataarray_t]]
datetime_t = Union[dt.datetime, pd.Timestamp]
diagnostic_component_t = Union[
    DiagnosticComponent, DiagnosticComponentComposite, BaseDiagnosticComponentComposite
]
domain_t = Domain
dtype_t = type
field_t = gtscript.Field["dtype"]
grid_t = Grid
gtstorage_dict_t = Mapping[str, Union[timedelta_t, gtstorage_t]]
horizontal_boundary_t = HorizontalBoundary
horizontal_grid_t = HorizontalGrid
mutable_array_dict_t = MutableMapping[str, Union[timedelta_t, np.ndarray]]
mutable_dataarray_dict_t = MutableMapping[str, Union[timedelta_t, dataarray_t]]
mutable_gtstorage_dict_t = MutableMapping[str, Union[timedelta_t, gtstorage_t]]
mutable_options_dict_t = Mapping[str, Any]
options_dict_t = Mapping[str, Any]
promoter_component_t = Union[Diagnostic2Tendency, Tendency2Diagnostic]
properties_dict_t = Mapping[str, Any]
tendency_component_t = Union[
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    BaseConcurrentCoupling,
]
triplet_int_t = Tuple[int, int, int]
