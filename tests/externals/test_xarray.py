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
import pytest
import xarray as xr

from gt4py.storage.storage import CPUStorage

from tasmania.python.utils.storage_utils import zeros


def test_xarray_gt4py_compatibility_gtmc():
    x = zeros((5, 5, 2), gt_powered=True, backend="gtmc", dtype=np.float64)
    x_da = xr.DataArray(x, dims=["x", "y", "z"], attrs={"units": "kg"})

    assert isinstance(x_da.data, CPUStorage)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_xarray_gt4py_compatibility_gtmc()
