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
from hypothesis import given, strategies as hyp_st
import numpy as np
import pytest

from tasmania.python.domain.topography import PhysicalTopography
from tasmania.python.utils.storage_utils import get_dataarray_2d

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
    keys = ("time", "smooth")
    topo = PhysicalTopography.factory(
        "flat", pgrid, **{key: kwargs[key] for key in keys}
    )

    profile_val = get_dataarray_2d(
        np.zeros((pgrid.nx, pgrid.ny), dtype=pgrid.x.dtype), pgrid, "m"
    )

    compare_dataarrays(topo.steady_profile, profile_val)
    compare_dataarrays(topo.profile, profile_val)


if __name__ == "__main__":
    pytest.main([__file__])
