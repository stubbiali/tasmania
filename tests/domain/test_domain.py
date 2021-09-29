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

from tasmania.python.domain.domain import Domain
from tasmania.python.domain.grid import PhysicalGrid
from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.utils.storage import (
    deepcopy_array_dict,
    deepcopy_dataarray_dict,
)

from tests.conf import backend as conf_backend, dtype as conf_dtype
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
    hyp_settings,
)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test(data, backend, dtype, subtests):
    # ========================================
    # random data generation
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype)

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

    topo_kwargs = data.draw(
        st_topography_kwargs(domain_x, domain_y), label="kwargs"
    )
    topo_type = topo_kwargs.pop("type")

    # ========================================
    # test bed
    # ========================================
    x, xu, dx = get_xaxis(domain_x, nx, storage_options=so)
    y, yv, dy = get_yaxis(domain_y, ny, storage_options=so)
    z, zhl, dz = get_zaxis(domain_z, nz, storage_options=so)

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

    hb = HorizontalBoundary.factory(
        hb_type,
        pgrid,
        nb,
        backend=backend,
        backend_options=bo,
        storage_options=so,
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
        backend=backend,
        backend_options=bo,
        storage_options=so,
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
    compare_dataarrays(
        hb.get_numerical_xaxis(dims="c_" + x.dims[0]), grid.grid_xy.x
    )
    compare_dataarrays(
        hb.get_numerical_xaxis_staggered(dims="c_" + xu.dims[0]),
        grid.grid_xy.x_at_u_locations,
    )
    compare_dataarrays(dx, grid.grid_xy.dx)
    compare_dataarrays(
        hb.get_numerical_yaxis(dims="c_" + y.dims[0]), grid.grid_xy.y
    )
    compare_dataarrays(
        hb.get_numerical_yaxis_staggered(dims="c_" + yv.dims[0]),
        grid.grid_xy.y_at_v_locations,
    )
    compare_dataarrays(dy, grid.grid_xy.dy)
    compare_dataarrays(z, grid.z)
    compare_dataarrays(zhl, grid.z_on_interface_levels)
    compare_dataarrays(dz, grid.dz)

    state = data.draw(st_state(grid, backend=backend, storage_options=so))
    state_dc = deepcopy_dataarray_dict(state)
    hb.reference_state = state
    dmn_hb = domain.horizontal_boundary
    dmn_hb.reference_state = state

    # enforce_field
    key = "afield_at_u_locations_on_interface_levels"
    hb.enforce_field(
        state[key].values,
        field_name=key,
        field_units=state[key].attrs["units"],
        time=state["time"],
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
    raw_state.update(
        {key: state[key].values for key in state if key != "time"}
    )
    raw_state_dc = deepcopy_array_dict(raw_state)

    field_properties = {}
    for key in state:
        if key != "time" and data.draw(hyp_st.booleans()):
            field_properties[key] = {"units": state[key].attrs["units"]}
    hb.enforce_raw(raw_state, field_properties=field_properties)
    dmn_hb.enforce_raw(raw_state_dc, field_properties=field_properties)

    for name in raw_state:
        # with subtests.test(name=name):
        if name != "time":
            compare_arrays(raw_state[name], raw_state_dc[name])

    # enforce
    field_names = tuple(key for key in field_properties)
    hb.enforce(state, field_names=field_names)
    dmn_hb.enforce(state_dc, field_names=field_names)

    for name in state:
        # with subtests.test(name=name):
        if name != "time":
            compare_dataarrays(state[name], state_dc[name])


if __name__ == "__main__":
    pytest.main([__file__])
