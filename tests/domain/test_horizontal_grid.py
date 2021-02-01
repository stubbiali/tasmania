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
from hypothesis import (
    assume,
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.domain.horizontal_grid import (
    HorizontalGrid,
    PhysicalHorizontalGrid,
    NumericalHorizontalGrid,
)
from tasmania.python.framework.options import StorageOptions

from tests.conf import dtype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary,
    st_interval,
    st_length,
    st_physical_grid,
)
from tests.utilities import (
    compare_dataarrays,
    get_xaxis,
    get_yaxis,
    hyp_settings,
)


@hyp_settings
@pytest.mark.parametrize("dtype", conf_dtype)
@given(data=hyp_st.data())
def test_grid(data, dtype):
    # ========================================
    # random data generation
    # ========================================
    nx = data.draw(st_length(axis_name="x"), label="nx")
    ny = data.draw(st_length(axis_name="y"), label="ny")

    assume(not (nx == 1 and ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"))
    domain_y = data.draw(st_interval(axis_name="y"))

    so = StorageOptions(dtype=dtype)

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, storage_options=so)
    y, yv, dy = get_yaxis(domain_y, ny, storage_options=so)

    #
    # test #1
    #
    grid = HorizontalGrid(x, y, xu, yv, storage_options=so)

    compare_dataarrays(x, grid.x)
    compare_dataarrays(xu, grid.x_at_u_locations)
    compare_dataarrays(dx, grid.dx)
    assert grid.nx == nx

    compare_dataarrays(y, grid.y)
    compare_dataarrays(yv, grid.y_at_v_locations)
    compare_dataarrays(dy, grid.dy)
    assert grid.ny == ny

    #
    # test #2
    #
    grid = HorizontalGrid(x, y, x_at_u_locations=xu, storage_options=so)

    compare_dataarrays(x, grid.x)
    compare_dataarrays(xu, grid.x_at_u_locations)
    compare_dataarrays(dx, grid.dx)
    assert grid.nx == nx

    compare_dataarrays(y, grid.y)
    compare_dataarrays(yv, grid.y_at_v_locations)
    compare_dataarrays(dy, grid.dy)
    assert grid.ny == ny

    #
    # test #3
    #
    grid = HorizontalGrid(x, y, y_at_v_locations=yv, storage_options=so)

    compare_dataarrays(x, grid.x)
    compare_dataarrays(xu, grid.x_at_u_locations)
    compare_dataarrays(dx, grid.dx)
    assert grid.nx == nx

    compare_dataarrays(y, grid.y)
    compare_dataarrays(yv, grid.y_at_v_locations)
    compare_dataarrays(dy, grid.dy)
    assert grid.ny == ny

    #
    # test #4
    #
    grid = HorizontalGrid(x, y)

    compare_dataarrays(x, grid.x)
    compare_dataarrays(xu, grid.x_at_u_locations)
    compare_dataarrays(dx, grid.dx)
    assert grid.nx == nx

    compare_dataarrays(y, grid.y)
    compare_dataarrays(yv, grid.y_at_v_locations)
    compare_dataarrays(dy, grid.dy)
    assert grid.ny == ny


@hyp_settings
@pytest.mark.parametrize("dtype", conf_dtype)
@given(data=hyp_st.data())
def test_physical_grid(data, dtype):
    # ========================================
    # random data generation
    # ========================================
    nx = data.draw(st_length(axis_name="x"), label="nx")
    ny = data.draw(st_length(axis_name="y"), label="ny")

    assume(not (nx == 1 and ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"))
    domain_y = data.draw(st_interval(axis_name="y"))

    so = StorageOptions(dtype=dtype)

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, storage_options=so)
    y, yv, dy = get_yaxis(domain_y, ny, storage_options=so)

    grid = PhysicalHorizontalGrid(
        domain_x, nx, domain_y, ny, storage_options=so
    )

    compare_dataarrays(x, grid.x)
    compare_dataarrays(xu, grid.x_at_u_locations)
    compare_dataarrays(dx, grid.dx)
    assert grid.nx == nx

    compare_dataarrays(y, grid.y)
    compare_dataarrays(yv, grid.y_at_v_locations)
    compare_dataarrays(dy, grid.dy)
    assert grid.ny == ny


@hyp_settings
@pytest.mark.parametrize("dtype", conf_dtype)
@given(data=hyp_st.data())
def test_numerical_grid(data, dtype):
    # ========================================
    # random data generation
    # ========================================
    so = StorageOptions(dtype=dtype)

    pgrid = data.draw(st_physical_grid(), label="pgrid")
    assume(not (pgrid.nx == 1 and pgrid.ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"))
    domain_y = data.draw(st_interval(axis_name="y"))

    hb = data.draw(st_horizontal_boundary(pgrid))

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, pgrid.nx, storage_options=so)
    y, yv, dy = get_yaxis(domain_y, pgrid.ny, storage_options=so)

    grid = NumericalHorizontalGrid(hb)

    compare_dataarrays(hb.get_numerical_xaxis(dims="c_x"), grid.x)
    compare_dataarrays(
        hb.get_numerical_xaxis_staggered(dims="c_x_at_u_locations"),
        grid.x_at_u_locations,
    )
    # compare_dataarrays(dx, grid.dx)
    assert grid.nx == hb.ni

    compare_dataarrays(hb.get_numerical_yaxis(dims="c_y"), grid.y)
    compare_dataarrays(
        hb.get_numerical_yaxis_staggered(dims="c_y_at_v_locations"),
        grid.y_at_v_locations,
    )
    # compare_dataarrays(dy, grid.dy)
    assert grid.ny == hb.nj


if __name__ == "__main__":
    pytest.main([__file__])
