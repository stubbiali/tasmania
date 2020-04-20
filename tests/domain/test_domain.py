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

import gt4py

from tasmania.python.domain.domain import Domain
from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.utils.storage_utils import (
    deepcopy_array_dict,
    deepcopy_dataarray_dict,
)

from tests.conf import backend as conf_backend, datatype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary_kwargs,
    st_horizontal_boundary_layers,
    st_horizontal_boundary_type,
    st_interface,
    st_interval,
    st_length,
    st_one_of,
    st_state,
    st_topography_kwargs,
)
from tests.utilities import (
    compare_arrays,
    compare_dataarrays,
    get_xaxis,
    get_yaxis,
    get_zaxis,
)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(data=hyp_st.data())
def test(data, subtests):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    if gt_powered:
        gt4py.storage.prepare_numpy()

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
    topo_type = topo_kwargs.pop("type")

    dtype = data.draw(st_one_of(conf_dtype), label="dtype")

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, dtype)
    y, yv, dy = get_yaxis(domain_y, ny, dtype)
    z, zhl, dz = get_zaxis(domain_z, nz, dtype)

    hb = HorizontalBoundary.factory(
        hb_type,
        nx,
        ny,
        nb,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        **hb_kwargs
    )

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
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
    )

    # physical_grid
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

    # numerical_grid
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

    # horizontal_boundary
    dmn_hb = domain.horizontal_boundary
    assert hasattr(dmn_hb, "dmn_enforce_field")
    assert hasattr(dmn_hb, "dmn_enforce_raw")
    assert hasattr(dmn_hb, "dmn_enforce")
    assert hasattr(dmn_hb, "dmn_set_outermost_layers_x")
    assert hasattr(dmn_hb, "dmn_set_outermost_layers_y")

    state = data.draw(st_state(grid, gt_powered=gt_powered, backend=backend))
    state_dc = deepcopy_dataarray_dict(state)

    hb.reference_state = state
    dmn_hb.reference_state = state

    # enforce_field
    key = "afield_at_u_locations_on_interface_levels"
    hb.enforce_field(
        state[key].values,
        field_name=key,
        field_units=state[key].attrs["units"],
        time=state["time"],
        grid=grid,
    )
    dmn_hb.enforce_field(
        state_dc[key].values,
        field_name=key,
        field_units=state_dc[key].attrs["units"],
        time=state_dc["time"],
    )
    compare_dataarrays(state[key], state_dc[key])

    # enforce_raw
    raw_state = {"time": state["time"]}
    raw_state.update({key: state[key].values for key in state if key != "time"})
    raw_state_dc = deepcopy_array_dict(raw_state)

    field_properties = {}
    for key in state:
        if key != "time" and data.draw(hyp_st.booleans()):
            field_properties[key] = {"units": state[key].attrs["units"]}
    hb.enforce_raw(raw_state, field_properties=field_properties, grid=grid)
    dmn_hb.enforce_raw(raw_state_dc, field_properties=field_properties)

    for name in raw_state:
        # with subtests.test(name=name):
        if name != "time":
            compare_arrays(raw_state[name], raw_state_dc[name])

    # enforce
    field_names = tuple(key for key in field_properties)
    hb.enforce(state, field_names=field_names, grid=grid)
    dmn_hb.enforce(state_dc, field_names=field_names)

    for name in state:
        # with subtests.test(name=name):
        if name != "time":
            compare_dataarrays(state[name], state_dc[name])


if __name__ == "__main__":
    pytest.main([__file__])
