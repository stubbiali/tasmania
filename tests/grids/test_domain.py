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
import pytest

from tasmania.python.grids.domain import Domain
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary

from tests.conf import datatype as conf_dtype
from tests.utilities import (
    compare_dataarrays,
    get_xaxis,
    get_yaxis,
    get_zaxis,
    st_horizontal_boundary_kwargs,
    st_horizontal_boundary_layers,
    st_horizontal_boundary_type,
    st_interface,
    st_interval,
    st_isentropic_state,
    st_length,
    st_one_of,
    st_topography_kwargs,
)


@settings(
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
    deadline=None,
)
@given(hyp_st.data())
def test_domain(data):
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

    hb_type = data.draw(st_horizontal_boundary_type())
    nb = data.draw(st_horizontal_boundary_layers(nx, ny))
    hb_kwargs = data.draw(st_horizontal_boundary_kwargs(hb_type, nx, ny, nb))

    topo_kwargs = data.draw(st_topography_kwargs(domain_x, domain_y), label="kwargs")
    topo_type = topo_kwargs["type"]

    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, dtype)
    y, yv, dy = get_yaxis(domain_y, ny, dtype)
    z, zhl, dz = get_zaxis(domain_z, nz, dtype)

    hb = HorizontalBoundary.factory(hb_type, nx, ny, nb, **hb_kwargs)

    domain = Domain(
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        zi,
        horizontal_boundary_type=hb_type,
        nb=nb,
        horizontal_boundary_kwargs=hb_kwargs,
        topography_type=topo_type,
        topography_kwargs=topo_kwargs,
        dtype=dtype,
    )

    grid = domain.physical_grid
    compare_dataarrays(x, grid.grid_xy.x)
    compare_dataarrays(xu, grid.grid_xy.x_at_u_locations)
    compare_dataarrays(dx, grid.grid_xy.dx)
    compare_dataarrays(y, grid.grid_xy.y)
    compare_dataarrays(yv, grid.grid_xy.y_at_v_locations)
    compare_dataarrays(dy, grid.grid_xy.dy)
    compare_dataarrays(z, grid.z)
    compare_dataarrays(zhl, grid.z_on_interface_levels)
    compare_dataarrays(dz, grid.dz)

    grid = domain.numerical_grid
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

    dmn_hb = domain.horizontal_boundary

    assert hasattr(dmn_hb, "dmn_enforce_field")
    assert hasattr(dmn_hb, "dmn_enforce_raw")
    assert hasattr(dmn_hb, "dmn_enforce")

    if hb_type != "dirichlet":
        state = data.draw(st_isentropic_state(grid, moist=False, precipitation=False))
        state_dmn = deepcopy(state)

        hb.reference_state = state
        dmn_hb.reference_state = state

        # enforce_field
        hb.enforce_field(
            state["air_isentropic_density"].values,
            field_name="air_isentropic_density",
            field_units=state["air_isentropic_density"].attrs["units"],
            time=state["time"],
            grid=grid,
        )
        dmn_hb.enforce_field(
            state_dmn["air_isentropic_density"].values,
            field_name="air_isentropic_density",
            field_units=state_dmn["air_isentropic_density"].attrs["units"],
            time=state_dmn["time"],
        )
        assert np.allclose(
            state["air_isentropic_density"], state_dmn["air_isentropic_density"]
        )

        # enforce_raw
        raw_state = {"time": state["time"]}
        for name in state:
            if name != "time":
                raw_state[name] = state[name].values
        raw_state_dmn = {"time": state_dmn["time"]}
        for name in state_dmn:
            if name != "time":
                raw_state_dmn[name] = state_dmn[name].values
        field_properties = {
            name: {"units": state[name].attrs["units"]}
            for name in state
            if name != "time"
        }
        hb.enforce_raw(raw_state, field_properties=field_properties, grid=grid)
        dmn_hb.enforce_raw(raw_state_dmn, field_properties=field_properties)
        for name in raw_state:
            if name != "time":
                assert np.allclose(raw_state[name], raw_state_dmn[name])

        # enforce
        hb.enforce(state, grid=grid)
        dmn_hb.enforce(state_dmn)
        for name in state:
            if name != "time":
                assert np.allclose(state[name], state_dmn[name])


if __name__ == "__main__":
    pytest.main([__file__])
