# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
import sympl
import xarray as xr

from tasmania.python.framework.allocators import as_storage, zeros

from tests.conf import backend as conf_backend


@pytest.mark.parametrize("backend", conf_backend)
def test_compatibility(backend):
    shape = (15, 20, 25)
    x = zeros(backend, shape=shape)
    x_xr = xr.DataArray(x, attrs={"units": "kg"})
    assert isinstance(x_xr.data, type(x))


def test_indexing(backend="gt4py:gtmc"):
    shape = (15, 20, 25)
    x_np = np.random.rand(*shape)
    y_gt = as_storage(backend, data=x_np)

    x_xr = xr.DataArray(x_np, attrs={"units": "m"})
    y_xr = xr.DataArray(y_gt, attrs={"units": "m"})

    _ = x_xr[...]
    print("numpy read: pass")

    x_xr[...] = 1
    print("numpy write: pass")

    _ = y_xr[...]
    print("gt4py read: pass")

    y_xr[...] = 1
    print("gt4py write: pass")


def test_units_conversion_pint(backend="gt4py:gtmc"):
    import pint_xarray

    shape = (15, 20, 25)
    x_np = np.random.rand(*shape)
    y_gt = as_storage(backend, data=x_np)

    x_xr = xr.DataArray(x_np, attrs={"units": "m"})
    y_xr = xr.DataArray(y_gt, attrs={"units": "m"})

    _ = x_xr.pint.to("km")
    print("numpy: pass")


def test_units_conversion_sympl(backend="gt4py:gtmc"):
    shape = (15, 20, 25)
    x_np = np.random.rand(*shape)
    y_gt = as_storage(backend, data=x_np)

    x_da = sympl.DataArray(x_np, attrs={"units": "m"})
    y_da = sympl.DataArray(y_gt, attrs={"units": "m"})

    _ = x_da.to_units("km")
    print("numpy: pass")

    _ = y_da.to_units("km")
    print("gt4py: pass")


if __name__ == "__main__":
    pytest.main([__file__])
