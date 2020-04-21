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
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import pytest

from tasmania.python.domain.horizontal_grid import (
    HorizontalGrid,
    PhysicalHorizontalGrid,
    NumericalHorizontalGrid,
)

from tests.conf import datatype as conf_dtype
from tests.strategies import st_horizontal_boundary, st_interval, st_length, st_one_of
from tests.utilities import compare_dataarrays, get_xaxis, get_yaxis


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_grid(data):
    # ========================================
    # random data generation
    # ========================================
    nx = data.draw(st_length(axis_name="x"), label="nx")
    ny = data.draw(st_length(axis_name="y"), label="ny")

    assume(not (nx == 1 and ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"))
    domain_y = data.draw(st_interval(axis_name="y"))

    dtype = data.draw(st_one_of(conf_dtype))

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, dtype)
    y, yv, dy = get_yaxis(domain_y, ny, dtype)

    #
    # test #1
    #
    grid = HorizontalGrid(x, y, xu, yv)

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
    grid = HorizontalGrid(x, y, x_at_u_locations=xu)

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
    grid = HorizontalGrid(x, y, y_at_v_locations=yv)

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


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_physical_grid(data):
    # ========================================
    # random data generation
    # ========================================
    nx = data.draw(st_length(axis_name="x"), label="nx")
    ny = data.draw(st_length(axis_name="y"), label="ny")

    assume(not (nx == 1 and ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"))
    domain_y = data.draw(st_interval(axis_name="y"))

    dtype = data.draw(st_one_of(conf_dtype))

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, dtype)
    y, yv, dy = get_yaxis(domain_y, ny, dtype)

    grid = PhysicalHorizontalGrid(domain_x, nx, domain_y, ny, dtype=dtype)

    compare_dataarrays(x, grid.x)
    compare_dataarrays(xu, grid.x_at_u_locations)
    compare_dataarrays(dx, grid.dx)
    assert grid.nx == nx

    compare_dataarrays(y, grid.y)
    compare_dataarrays(yv, grid.y_at_v_locations)
    compare_dataarrays(dy, grid.dy)
    assert grid.ny == ny


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_numerical_grid(data):
    # ========================================
    # random data generation
    # ========================================
    nx = data.draw(st_length(axis_name="x"), label="nx")
    ny = data.draw(st_length(axis_name="y"), label="ny")

    assume(not (nx == 1 and ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"))
    domain_y = data.draw(st_interval(axis_name="y"))

    hb = data.draw(st_horizontal_boundary(nx, ny))

    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, dtype)
    y, yv, dy = get_yaxis(domain_y, ny, dtype)

    pgrid = PhysicalHorizontalGrid(domain_x, nx, domain_y, ny, dtype=dtype)

    grid = NumericalHorizontalGrid(pgrid, hb)

    compare_dataarrays(hb.get_numerical_xaxis(x, dims="c_x"), grid.x)
    compare_dataarrays(
        hb.get_numerical_xaxis(xu, dims="c_x_at_u_locations"), grid.x_at_u_locations
    )
    compare_dataarrays(dx, grid.dx)
    assert grid.nx == hb.ni

    compare_dataarrays(hb.get_numerical_yaxis(y, dims="c_y"), grid.y)
    compare_dataarrays(
        hb.get_numerical_yaxis(yv, dims="c_y_at_v_locations"), grid.y_at_v_locations
    )
    compare_dataarrays(dy, grid.dy)
    assert grid.ny == hb.nj


if __name__ == "__main__":
    pytest.main([__file__])