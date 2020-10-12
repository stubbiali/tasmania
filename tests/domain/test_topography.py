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
    given,
    strategies as hyp_st,
)
from pandas import Timedelta
import pytest

from tasmania.python.domain.horizontal_grid import NumericalHorizontalGrid
from tasmania.python.domain.subclasses.topographies import Flat
from tasmania.python.domain.subclasses.topographies.gaussian import Gaussian
from tasmania.python.domain.subclasses.topographies import Schaer
from tasmania.python.domain.topography import (
    Topography,
    PhysicalTopography,
    NumericalTopography,
)

from tests.strategies import (
    st_floats,
    st_horizontal_boundary,
    st_horizontal_field,
    st_physical_horizontal_grid,
    st_topography_kwargs,
)
from tests.utilities import (
    compare_arrays,
    compare_dataarrays,
    compare_datetimes,
    hyp_settings,
)


@hyp_settings
@given(hyp_st.data())
def test_topography_properties(data):
    # ========================================
    # random data generation
    # ========================================
    pgrid = data.draw(st_physical_horizontal_grid(), label="pgrid")

    steady_profile = data.draw(
        st_horizontal_field(pgrid, 0, 10000, "m", "sprof"),
        label="sprof",
    )

    kwargs = data.draw(st_topography_kwargs(pgrid.x, pgrid.y), label="kwargs")

    # ========================================
    # test bed
    # ========================================
    topo = Topography(steady_profile, time=kwargs["time"])

    # steady_profile
    compare_dataarrays(steady_profile, topo.steady_profile)

    # time
    topo_time = kwargs["time"] or Timedelta(seconds=0.0)
    compare_datetimes(topo.time, topo_time)

    # profile
    if topo_time.total_seconds() == 0.0:
        compare_dataarrays(steady_profile, topo.profile)
    else:
        profile = deepcopy(steady_profile)
        profile.values[...] = 0.0
        compare_dataarrays(profile, topo.profile)


@hyp_settings
@given(hyp_st.data())
def test_topography_update(data):
    # ========================================
    # random data generation
    # ========================================
    pgrid = data.draw(st_physical_horizontal_grid(), label="pgrid")

    steady_profile = data.draw(
        st_horizontal_field(pgrid, 0, 10000, "m", "sprof"),
        label="sprof",
    )

    kwargs = data.draw(st_topography_kwargs(pgrid.x, pgrid.y), label="kwargs")

    fact1 = data.draw(st_floats(min_value=1e-6, max_value=1.0), label="fact1")
    fact2 = data.draw(st_floats(min_value=fact1, max_value=1.0), label="fact2")

    # ========================================
    # test bed
    # ========================================
    topo = Topography(steady_profile, time=kwargs["time"])

    topo_time = kwargs["time"] or Timedelta(seconds=0.0)

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


def test_physical_topography_registry():
    # flat
    assert "flat" in PhysicalTopography.registry
    assert PhysicalTopography.registry["flat"] == Flat

    # gaussian
    assert "gaussian" in PhysicalTopography.registry
    assert PhysicalTopography.registry["gaussian"] == Gaussian

    # schaer
    assert "schaer" in PhysicalTopography.registry
    assert PhysicalTopography.registry["schaer"] == Schaer


@hyp_settings
@given(hyp_st.data())
def test_physical_topography_factory(data):
    # ========================================
    # random data generation
    # ========================================
    pgrid = data.draw(st_physical_horizontal_grid(), label="pgrid")

    # ========================================
    # test bed
    # ========================================
    # flat
    obj = PhysicalTopography.factory("flat", pgrid)
    assert isinstance(obj, Flat)

    # gaussian
    obj = PhysicalTopography.factory("gaussian", pgrid)
    assert isinstance(obj, Gaussian)

    # schaer
    obj = PhysicalTopography.factory("schaer", pgrid)
    assert isinstance(obj, Schaer)


@hyp_settings
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
    keys = (
        "time",
        "smooth",
        "max_height",
        "center_x",
        "center_y",
        "width_x",
        "width_y",
    )
    ptopo = PhysicalTopography.factory(
        "gaussian", pgrid, **{key: kwargs[key] for key in keys}
    )
    ctopo = NumericalTopography(cgrid, ptopo, hb)

    assert ctopo.type == "gaussian"

    # profile
    compare_arrays(
        ctopo.profile.values, hb.get_numerical_field(ptopo.profile.values)
    )

    # steady_profile
    compare_arrays(
        ctopo.steady_profile.values,
        hb.get_numerical_field(ptopo.steady_profile.values),
    )

    # time
    compare_datetimes(ptopo.time, ctopo.time)


if __name__ == "__main__":
    pytest.main([__file__])
