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
from hypothesis import (
    assume,
    given,
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.domain.grid import Grid, PhysicalGrid, NumericalGrid
from tasmania.python.domain.topography import PhysicalTopography
from tasmania.python.framework.options import StorageOptions

from tests.conf import dtype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary,
    st_interface,
    st_interval,
    st_length,
    st_physical_horizontal_grid,
    st_topography_kwargs,
)
from tests.utilities import (
    compare_dataarrays,
    get_xaxis,
    get_yaxis,
    get_zaxis,
    hyp_settings,
)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("dtype", conf_dtype)
def test_grid(data, dtype):
    # ========================================
    # random data generation
    # ========================================
    so = StorageOptions(dtype=dtype)

    grid_xy = data.draw(st_physical_horizontal_grid(storage_options=so))

    nz = data.draw(st_length(axis_name="z"))
    domain_z = data.draw(st_interval(axis_name="z"))
    zi = data.draw(st_interface(domain_z))

    topo_kwargs = data.draw(st_topography_kwargs(grid_xy.x, grid_xy.y))
    topo_type = topo_kwargs.pop("type")
    topo_kwargs = {
        "time": topo_kwargs["time"],
        "smooth": topo_kwargs["smooth"],
    }
    topo = PhysicalTopography.factory(topo_type, grid_xy, **topo_kwargs)

    # ========================================
    # test bed
    # ========================================
    z, zhl, dz = get_zaxis(domain_z, nz, storage_options=so)

    grid = Grid(grid_xy, z, zhl, zi, topo)

    compare_dataarrays(grid_xy.x, grid.grid_xy.x)
    compare_dataarrays(grid_xy.x_at_u_locations, grid.grid_xy.x_at_u_locations)
    compare_dataarrays(grid_xy.y, grid.grid_xy.y)
    compare_dataarrays(grid_xy.y_at_v_locations, grid.grid_xy.y_at_v_locations)
    compare_dataarrays(z, grid.z)
    compare_dataarrays(zhl, grid.z_on_interface_levels)
    compare_dataarrays(dz, grid.dz)
    assert nz == grid.nz


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("dtype", conf_dtype)
def test_physical_grid(data, dtype):
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

    topo_kwargs = data.draw(
        st_topography_kwargs(domain_x, domain_y), label="kwargs"
    )
    topo_type = topo_kwargs.pop("type")
    topo_kwargs = {
        "time": topo_kwargs["time"],
        "smooth": topo_kwargs["smooth"],
    }

    so = StorageOptions(dtype=dtype)

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, storage_options=so)
    y, yv, dy = get_yaxis(domain_y, ny, storage_options=so)
    z, zhl, dz = get_zaxis(domain_z, nz, storage_options=so)

    grid = PhysicalGrid(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        zi,
        topo_type,
        topo_kwargs,
        storage_options=so,
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


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("dtype", conf_dtype)
def test_numerical_grid(data, dtype):
    # ========================================
    # random data generation
    # ========================================
    so = StorageOptions(dtype=dtype)

    nx = data.draw(st_length(axis_name="x"), label="nx")
    ny = data.draw(st_length(axis_name="y"), label="ny")
    nz = data.draw(st_length(axis_name="z"), label="nz")

    assume(not (nx == 1 and ny == 1))

    domain_x = data.draw(st_interval(axis_name="x"), label="x")
    domain_y = data.draw(st_interval(axis_name="y"), label="y")
    domain_z = data.draw(st_interval(axis_name="z"), label="z")

    zi = data.draw(st_interface(domain_z), label="zi")

    topo_kwargs = data.draw(
        st_topography_kwargs(domain_x, domain_y), label="kwargs"
    )
    topo_type = topo_kwargs["type"]
    topo_kwargs = {
        "time": topo_kwargs["time"],
        "smooth": topo_kwargs["smooth"],
    }

    pgrid = PhysicalGrid(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        zi,
        topo_type,
        topo_kwargs,
        storage_options=so,
    )

    hb = data.draw(st_horizontal_boundary(pgrid, storage_options=so))

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, storage_options=so)
    y, yv, dy = get_yaxis(domain_y, ny, storage_options=so)
    z, zhl, dz = get_zaxis(domain_z, nz, storage_options=so)

    grid = NumericalGrid(hb)

    compare_dataarrays(
        hb.get_numerical_xaxis(dims="c_" + x.dims[0]), grid.grid_xy.x
    )
    compare_dataarrays(
        hb.get_numerical_xaxis_staggered(dims="c_" + xu.dims[0]),
        grid.grid_xy.x_at_u_locations,
    )
    # compare_dataarrays(dx, grid.grid_xy.dx)
    compare_dataarrays(
        hb.get_numerical_yaxis(dims="c_" + y.dims[0]), grid.grid_xy.y
    )
    compare_dataarrays(
        hb.get_numerical_yaxis_staggered(dims="c_" + yv.dims[0]),
        grid.grid_xy.y_at_v_locations,
    )
    # compare_dataarrays(dy, grid.grid_xy.dy)
    compare_dataarrays(z, grid.z)
    compare_dataarrays(zhl, grid.z_on_interface_levels)
    compare_dataarrays(dz, grid.dz)
    assert nz == grid.nz


if __name__ == "__main__":
    pytest.main([__file__])
