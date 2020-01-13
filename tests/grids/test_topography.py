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
from copy import deepcopy
from hypothesis import (
    assume,
    given,
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import numpy as np
from pandas import Timedelta
import pytest

from tasmania.python.grids.horizontal_grid import NumericalHorizontalGrid
from tasmania.python.grids.topography import (
    Topography,
    PhysicalTopography,
    NumericalTopography,
)

from tests.utilities import (
    compare_dataarrays,
    st_floats,
    st_horizontal_boundary,
    st_horizontal_field,
    st_physical_horizontal_grid,
    st_topography_kwargs,
)


@given(hyp_st.data())
def test_topography(data):
    # ========================================
    # random data generation
    # ========================================
    pgrid = data.draw(st_physical_horizontal_grid(), label="pgrid")

    steady_profile = data.draw(
        st_horizontal_field(pgrid, 0, 3000, "m", "sprof"), label="sprof"
    )

    kwargs = data.draw(st_topography_kwargs(pgrid.x, pgrid.y), label="kwargs")

    # ========================================
    # test bed
    # ========================================
    topo_type = kwargs.pop("type")
    topo = Topography(topo_type, steady_profile, **kwargs)

    assert topo.type == topo_type
    compare_dataarrays(steady_profile, topo.steady_profile)

    topo_time = kwargs.get("time", Timedelta(seconds=0.0))

    if topo_time.total_seconds() == 0.0:
        kwargs["time"] = topo_time
        assert kwargs == topo.kwargs
        compare_dataarrays(steady_profile, topo.profile)
    else:
        profile = deepcopy(steady_profile)
        profile.values[...] = 0.0
        compare_dataarrays(profile, topo.profile)


@given(hyp_st.data())
def test_update_topography(data):
    # ========================================
    # random data generation
    # ========================================
    pgrid = data.draw(st_physical_horizontal_grid(), label="pgrid")

    steady_profile = data.draw(
        st_horizontal_field(pgrid, 0, 3000, "m", "sprof"), label="sprof"
    )

    kwargs = data.draw(st_topography_kwargs(pgrid.x, pgrid.y), label="kwargs")

    fact1 = data.draw(st_floats(min_value=1e-6, max_value=1.0), label="fact1")
    fact2 = data.draw(st_floats(min_value=fact1, max_value=1.0), label="fact2")

    # ========================================
    # test bed
    # ========================================
    topo_type = kwargs.pop("type")
    topo = Topography(topo_type, steady_profile, **kwargs)

    topo_time = kwargs.get("time", Timedelta(seconds=0.0))

    if topo_time.total_seconds() == 0.0:
        compare_dataarrays(steady_profile, topo.profile)
    else:
        profile = deepcopy(steady_profile)
        profile.values[...] = 0.0
        compare_dataarrays(profile, topo.profile)

        time1 = fact1 * topo_time
        topo.update(time1)
        profile.values[...] = time1 / topo_time * steady_profile.values[...]
        compare_dataarrays(profile, topo.profile)
        compare_dataarrays(steady_profile, topo.steady_profile)

        time2 = fact2 * topo_time
        topo.update(time2)
        profile.values[...] = time2 / topo_time * steady_profile.values[...]
        compare_dataarrays(profile, topo.profile)
        compare_dataarrays(steady_profile, topo.steady_profile)


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_physical_topography(data):
    # ========================================
    # random data generation
    # ========================================
    pgrid = data.draw(st_physical_horizontal_grid(), label="pgrid")
    kwargs = data.draw(st_topography_kwargs(pgrid.x, pgrid.y), label="kwargs")

    # ========================================
    # test bed
    # ========================================
    topo_type = kwargs.pop("type")
    topo = PhysicalTopography(pgrid, topo_type, **kwargs)

    assert topo.type == topo_type

    topo_time = kwargs.get("time", Timedelta(seconds=0.0))

    if topo_time.total_seconds() == 0.0:
        compare_dataarrays(topo.steady_profile, topo.profile)
    else:
        profile = deepcopy(topo.steady_profile)
        profile.values[...] = 0.0
        compare_dataarrays(profile, topo.profile)


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_numerical_topography(data):
    # ========================================
    # random data generation
    # ========================================
    pgrid = data.draw(st_physical_horizontal_grid(), label="pgrid")
    hb = data.draw(st_horizontal_boundary(pgrid.nx, pgrid.ny), label="hb")
    cgrid = NumericalHorizontalGrid(pgrid, hb)
    kwargs = data.draw(st_topography_kwargs(pgrid.x, pgrid.y), label="kwargs")

    # ========================================
    # test bed
    # ========================================
    topo_type = kwargs.pop("type")
    ptopo = PhysicalTopography(pgrid, topo_type, **kwargs)
    ctopo = NumericalTopography(cgrid, ptopo, hb)

    assert ctopo.type == topo_type

    assert np.allclose(ctopo.profile.values, hb.get_numerical_field(ptopo.profile.values))
    assert np.allclose(
        ctopo.steady_profile.values, hb.get_numerical_field(ptopo.steady_profile.values)
    )

    from tasmania.python.grids._horizontal_boundary import Relaxed

    if isinstance(hb, Relaxed):
        assert id(ptopo.profile.values) == id(ctopo.profile.values)
        assert id(ptopo.steady_profile.values) == id(ctopo.steady_profile.values)


if __name__ == "__main__":
    pytest.main([__file__])
