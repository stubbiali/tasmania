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
from hypothesis import given, strategies as hyp_st
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.domain.topography import PhysicalTopography
from tasmania.python.utils.storage import (
    deepcopy_dataarray,
    get_dataarray_2d,
)

from tests.strategies import st_physical_horizontal_grid, st_topography_kwargs
from tests.utilities import compare_dataarrays, hyp_settings


@hyp_settings
@given(hyp_st.data())
def test_compute_steady_profile(data):
    # ========================================
    # random data generation
    # ========================================
    pgrid = data.draw(st_physical_horizontal_grid(), label="pgrid")
    kwargs = data.draw(st_topography_kwargs(pgrid.x, pgrid.y), label="kwargs")

    # ========================================
    # test bed
    # ========================================
    keys = (
        "time",
        "smooth",
        "max_height",
        "center_x",
        "center_y",
        "width_x",
        "width_y",
    )
    topo = PhysicalTopography.factory(
        "schaer", pgrid, **{key: kwargs[key] for key in keys}
    )

    profile_val = get_dataarray_2d(
        np.zeros((pgrid.nx, pgrid.ny), dtype=pgrid.x.dtype), pgrid, "m"
    )
    flat = deepcopy_dataarray(profile_val)

    x, y = pgrid.x, pgrid.y
    xv, yv = x.values, y.values

    hmax = kwargs["max_height"] or DataArray(500.0, attrs={"units": "m"})
    hmax = hmax.to_units("m").values.item()

    wx = kwargs["width_x"] or DataArray(1.0, attrs={"units": x.attrs["units"]})
    wx = wx.to_units(x.attrs["units"]).values.item()

    wy = kwargs["width_y"] or DataArray(1.0, attrs={"units": y.attrs["units"]})
    wy = wy.to_units(y.attrs["units"]).values.item()

    cx = (
        kwargs["center_x"].to_units(x.attrs["units"]).values.item()
        if kwargs["center_x"] is not None
        else 0.5 * (xv[0] + xv[-1])
    )

    cy = (
        kwargs["center_y"].to_units(y.attrs["units"]).values.item()
        if kwargs["center_y"] is not None
        else 0.5 * (yv[0] + yv[-1])
    )

    for i in range(pgrid.nx):
        for j in range(pgrid.ny):
            profile_val.values[i, j] = (
                hmax
                / (1 + ((xv[i] - cx) / wx) ** 2 + ((yv[j] - cy) / wy) ** 2)
                ** 1.5
            )

    if kwargs["smooth"]:
        tmp = deepcopy_dataarray(profile_val)
        for i in range(1, pgrid.nx - 1):
            for j in range(1, pgrid.ny - 1):
                profile_val.values[i, j] += 0.125 * (
                    tmp.values[i - 1, j]
                    + tmp.values[i + 1, j]
                    + tmp.values[i, j - 1]
                    + tmp.values[i, j + 1]
                    - 4.0 * tmp.values[i, j]
                )

    compare_dataarrays(topo.steady_profile, profile_val)

    if kwargs["time"] is None or kwargs["time"].total_seconds() == 0.0:
        compare_dataarrays(topo.profile, profile_val)
    else:
        compare_dataarrays(topo.profile, flat)


if __name__ == "__main__":
    pytest.main([__file__])
