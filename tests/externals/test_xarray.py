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
import pytest
import xarray as xr

from gt4py.storage.definitions import Storage

from tasmania.python.framework.allocators import zeros


def test_xarray_gt4py_compatibility_gtmc():
    x = zeros("gt4py:gtmc", shape=(5, 5, 2))
    x_da = xr.DataArray(x, dims=["x", "y", "z"], attrs={"units": "kg"})

    assert isinstance(x_da.data, Storage)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_xarray_gt4py_compatibility_gtmc()
