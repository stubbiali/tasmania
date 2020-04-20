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

from tasmania.python.domain.grid import Grid, PhysicalGrid, NumericalGrid
from tasmania.python.domain.topography import PhysicalTopography

from tests.conf import datatype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary,
    st_interface,
    st_interval,
    st_length,
    st_one_of,
    st_physical_horizontal_grid,
    st_topography_kwargs,
)
from tests.utilities import compare_dataarrays, get_xaxis, get_yaxis, get_zaxis


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
    grid_xy = data.draw(st_physical_horizontal_grid())

    nz = data.draw(st_length(axis_name="z"))
    domain_z = data.draw(st_interval(axis_name="z"))
    zi = data.draw(st_interface(domain_z))

    topo_kwargs = data.draw(st_topography_kwargs(grid_xy.x, grid_xy.y))
    topo_type = topo_kwargs.pop("type")
    topo_kwargs = {"time": topo_kwargs["time"], "smooth": topo_kwargs["smooth"]}
    topo = PhysicalTopography.factory(topo_type, grid_xy, **topo_kwargs)

    # ========================================
    # test bed
    # ========================================
    z, zhl, dz = get_zaxis(domain_z, nz, grid_xy.x.dtype)

    grid = Grid(grid_xy, z, zhl, zi, topo)

    compare_dataarrays(grid_xy.x, grid.grid_xy.x)
    compare_dataarrays(grid_xy.x_at_u_locations, grid.grid_xy.x_at_u_locations)
    compare_dataarrays(grid_xy.y, grid.grid_xy.y)
    compare_dataarrays(grid_xy.y_at_v_locations, grid.grid_xy.y_at_v_locations)
    compare_dataarrays(z, grid.z)
    compare_dataarrays(zhl, grid.z_on_interface_levels)
    compare_dataarrays(dz, grid.dz)
    assert nz == grid.nz


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
    nz = data.draw(st_length(axis_name="z"), label="nz")

    assume(not (nx == 1 and ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"), label="x")
    domain_y = data.draw(st_interval(axis_name="y"), label="y")
    domain_z = data.draw(st_interval(axis_name="z"), label="z")

    zi = data.draw(st_interface(domain_z), label="zi")

    topo_kwargs = data.draw(st_topography_kwargs(domain_x, domain_y), label="kwargs")
    topo_type = topo_kwargs.pop("type")
    topo_kwargs = {"time": topo_kwargs["time"], "smooth": topo_kwargs["smooth"]}

    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, dtype)
    y, yv, dy = get_yaxis(domain_y, ny, dtype)
    z, zhl, dz = get_zaxis(domain_z, nz, dtype)

    grid = PhysicalGrid(
        domain_x, nx, domain_y, ny, domain_z, nz, zi, topo_type, topo_kwargs, dtype=dtype
    )

    compare_dataarrays(x, grid.grid_xy.x)
    compare_dataarrays(xu, grid.grid_xy.x_at_u_locations)
    compare_dataarrays(dx, grid.grid_xy.dx)
    compare_dataarrays(y, grid.grid_xy.y)
    compare_dataarrays(yv, grid.grid_xy.y_at_v_locations)
    compare_dataarrays(dy, grid.grid_xy.dy)
    compare_dataarrays(z, grid.z)
    compare_dataarrays(zhl, grid.z_on_interface_levels)
    compare_dataarrays(dz, grid.dz)
    assert nz == grid.nz


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
    nz = data.draw(st_length(axis_name="z"), label="nz")

    assume(not (nx == 1 and ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"), label="x")
    domain_y = data.draw(st_interval(axis_name="y"), label="y")
    domain_z = data.draw(st_interval(axis_name="z"), label="z")

    zi = data.draw(st_interface(domain_z), label="zi")

    topo_kwargs = data.draw(st_topography_kwargs(domain_x, domain_y), label="kwargs")
    topo_type = topo_kwargs["type"]
    topo_kwargs = {"time": topo_kwargs["time"], "smooth": topo_kwargs["smooth"]}

    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    hb = data.draw(st_horizontal_boundary(nx, ny))

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, dtype)
    y, yv, dy = get_yaxis(domain_y, ny, dtype)
    z, zhl, dz = get_zaxis(domain_z, nz, dtype)

    pgrid = PhysicalGrid(
        domain_x, nx, domain_y, ny, domain_z, nz, zi, topo_type, topo_kwargs, dtype
    )
    grid = NumericalGrid(pgrid, hb)

    compare_dataarrays(hb.get_numerical_xaxis(x, dims="c_" + x.dims[0]), grid.grid_xy.x)
    compare_dataarrays(
        hb.get_numerical_xaxis(xu, dims="c_" + xu.dims[0]), grid.grid_xy.x_at_u_locations
    )
    compare_dataarrays(dx, grid.grid_xy.dx)
    compare_dataarrays(hb.get_numerical_yaxis(y, dims="c_" + y.dims[0]), grid.grid_xy.y)
    compare_dataarrays(
        hb.get_numerical_yaxis(yv, dims="c_" + yv.dims[0]), grid.grid_xy.y_at_v_locations
    )
    compare_dataarrays(dy, grid.grid_xy.dy)
    compare_dataarrays(z, grid.z)
    compare_dataarrays(zhl, grid.z_on_interface_levels)
    compare_dataarrays(dz, grid.dz)
    assert nz == grid.nz


if __name__ == "__main__":
    pytest.main([__file__])
