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
    HealthCheck,
    reproduce_failure,
    settings,
    strategies as hyp_st,
)
import numpy as np
import pytest
from sympl import DataArray

from tasmania.python.plot.retrievers import (
    DataRetriever,
    DataRetrieverComposite,
)
from tasmania import get_dataarray_3d

from tests.strategies import st_domain, st_isentropic_state_f, st_one_of


units = {
    "air_density": ("kg m^-3", "g cm^-3", None),
    "air_isentropic_density": ("kg m^-2 K^-1", "g km^-2 K^-1", None),
    "air_pressure_on_interface_levels": ("Pa", "kPa", "atm", None),
    "air_temperature": ("K", None),
    "exner_function_on_interface_levels": ("J kg^-1 K^-1", None),
    "height_on_interface_levels": ("m", "km", None),
    "montgomery_potential": ("m^2 s^-2", None),
    "x_momentum_isentropic": ("kg m^-1 K^-1 s^-1", None),
    "x_velocity_at_u_locations": ("m s^-1", "km hr^-1", None),
    "y_momentum_isentropic": ("kg m^-1 K^-1 s^-1", None),
    "y_velocity_at_v_locations": ("m s^-1", "km hr^-1", None),
}


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_field(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    state = data.draw(
        st_isentropic_state_f(grid, moist=True, precipitation=True),
        label="state",
    )
    field_name = data.draw(st_one_of(units.keys()), label="field_name")
    field_units = data.draw(st_one_of(units[field_name]), label="field_units")
    xmin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nx - 1), label="xmin"
    )
    xmax = data.draw(
        hyp_st.integers(min_value=xmin + 1, max_value=grid.nx), label="xmax"
    )
    ymin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.ny - 1), label="ymin"
    )
    ymax = data.draw(
        hyp_st.integers(min_value=ymin + 1, max_value=grid.ny), label="ymax"
    )
    zmin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz - 1), label="zmin"
    )
    zmax = data.draw(
        hyp_st.integers(min_value=zmin + 1, max_value=grid.nz), label="zmax"
    )

    # ========================================
    # test bed
    # ========================================
    x = slice(xmin, xmax)
    y = slice(ymin, ymax)
    z = slice(zmin, zmax)

    data_val = (
        state[field_name].to_units(field_units).values
        if field_units is not None
        else state[field_name].values
    )

    # x
    dr = DataRetriever(grid, field_name, field_units, x=x)
    data = dr(state)
    assert data.shape[0] == xmax - xmin
    assert data.shape[1] == state[field_name].shape[1]
    assert data.shape[2] == state[field_name].shape[2]
    assert np.allclose(data, data_val[x, :, :])

    # y
    dr = DataRetriever(grid, field_name, field_units, y=y)
    data = dr(state)
    assert data.shape[0] == state[field_name].shape[0]
    assert data.shape[1] == ymax - ymin
    assert data.shape[2] == state[field_name].shape[2]
    assert np.allclose(data, data_val[:, y, :])

    # z
    dr = DataRetriever(grid, field_name, field_units, z=z)
    data = dr(state)
    assert data.shape[0] == state[field_name].shape[0]
    assert data.shape[1] == state[field_name].shape[1]
    assert data.shape[2] == zmax - zmin
    assert np.allclose(data, data_val[:, :, z])

    # x, y
    dr = DataRetriever(grid, field_name, field_units, x=x, y=y)
    data = dr(state)
    assert data.shape[0] == xmax - xmin
    assert data.shape[1] == ymax - ymin
    assert data.shape[2] == state[field_name].shape[2]
    assert np.allclose(data, data_val[x, y, :])

    # x, z
    dr = DataRetriever(grid, field_name, field_units, x=x, z=z)
    data = dr(state)
    assert data.shape[0] == xmax - xmin
    assert data.shape[1] == state[field_name].shape[1]
    assert data.shape[2] == zmax - zmin
    assert np.allclose(data, data_val[x, :, z])

    # y, z
    dr = DataRetriever(grid, field_name, field_units, y=y, z=z)
    data = dr(state)
    assert data.shape[0] == state[field_name].shape[0]
    assert data.shape[1] == ymax - ymin
    assert data.shape[2] == zmax - zmin
    assert np.allclose(data, data_val[:, y, z])

    # x, y, z
    dr = DataRetriever(grid, field_name, field_units, x=x, y=y, z=z)
    data = dr(state)
    assert data.shape[0] == xmax - xmin
    assert data.shape[1] == ymax - ymin
    assert data.shape[2] == zmax - zmin
    assert np.allclose(data, data_val[x, y, z])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_horizontal_velocity(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    state = data.draw(st_isentropic_state_f(grid), label="state")
    field_units = data.draw(
        st_one_of(units["x_velocity_at_u_locations"]), label="field_units"
    )
    xmin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nx - 1), label="xmin"
    )
    xmax = data.draw(
        hyp_st.integers(min_value=xmin + 1, max_value=grid.nx), label="xmax"
    )
    ymin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.ny - 1), label="ymin"
    )
    ymax = data.draw(
        hyp_st.integers(min_value=ymin + 1, max_value=grid.ny), label="ymax"
    )
    zmin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz - 1), label="zmin"
    )
    zmax = data.draw(
        hyp_st.integers(min_value=zmin + 1, max_value=grid.nz), label="zmax"
    )

    # ========================================
    # test bed
    # ========================================
    x = slice(xmin, xmax)
    y = slice(ymin, ymax)
    z = slice(zmin, zmax)

    dr = DataRetriever(grid, "horizontal_velocity", field_units, x=x, y=y, z=z)

    data = dr(state)

    assert data.shape[0] == xmax - xmin
    assert data.shape[1] == ymax - ymin
    assert data.shape[2] == zmax - zmin
    s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
    su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    sv = state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
    hv = np.sqrt((su / s) ** 2 + (sv / s) ** 2)
    data_val = (
        hv[x, y, z]
        if field_units is None
        else get_dataarray_3d(hv, grid, "m s^-1")[x, y, z]
        .to_units(field_units)
        .values
    )
    assert np.allclose(data, data_val, equal_nan=True)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def _test_height(data):
    # ========================================
    # random data generation
    # ========================================
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    state = data.draw(st_isentropic_state_f(grid), label="state")
    field_units = data.draw(
        st_one_of(units["height_on_interface_levels"]), label="field_units"
    )
    xmin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nx - 1), label="xmin"
    )
    xmax = data.draw(
        hyp_st.integers(min_value=xmin + 1, max_value=grid.nx), label="xmax"
    )
    ymin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.ny - 1), label="ymin"
    )
    ymax = data.draw(
        hyp_st.integers(min_value=ymin + 1, max_value=grid.ny), label="ymax"
    )
    zmin = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz - 1), label="zmin"
    )
    zmax = data.draw(
        hyp_st.integers(min_value=zmin + 1, max_value=grid.nz), label="zmax"
    )

    # ========================================
    # test bed
    # ========================================
    x = slice(xmin, xmax)
    y = slice(ymin, ymax)
    z = slice(zmin, zmax)

    dr = DataRetriever(grid, "height", field_units, x=x, y=y, z=z)

    data = dr(state)

    assert data.shape[0] == xmax - xmin
    assert data.shape[1] == ymax - ymin
    assert data.shape[2] == zmax - zmin
    factor = (
        DataArray(1, attrs={"units": "m"}).to_units(field_units).values.item()
        if field_units is not None
        else 1.0
    )
    h = state["height_on_interface_levels"].to_units("m").values
    data_val = (
        factor * 0.5 * (h[x, y, zmin:zmax] + h[x, y, zmin + 1 : zmax + 1])
    )
    assert np.allclose(data, data_val, equal_nan=True)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_composite(data):
    domain = data.draw(st_domain(), label="domain")
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.physical_grid
        if grid_type == "physical"
        else domain.numerical_grid
    )
    state = data.draw(st_isentropic_state_f(grid, moist=True), label="state")

    field1_name = data.draw(st_one_of(units.keys()), label="field1_name")
    field1_units = data.draw(
        st_one_of(units[field1_name]), label="field1_units"
    )
    xmin1 = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nx - 1), label="xmin1"
    )
    xmax1 = data.draw(
        hyp_st.integers(min_value=xmin1 + 1, max_value=grid.nx), label="xmax1"
    )
    ymin1 = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.ny - 1), label="ymin1"
    )
    ymax1 = data.draw(
        hyp_st.integers(min_value=ymin1 + 1, max_value=grid.ny), label="ymax1"
    )
    zmin1 = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz - 1), label="zmin1"
    )
    zmax1 = data.draw(
        hyp_st.integers(min_value=zmin1 + 1, max_value=grid.nz), label="zmax1"
    )

    field2_name = data.draw(st_one_of(units.keys()), label="field2_name")
    field2_units = data.draw(
        st_one_of(units[field2_name]), label="field2_units"
    )
    xmin2 = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nx - 1), label="xmin2"
    )
    xmax2 = data.draw(
        hyp_st.integers(min_value=xmin2 + 1, max_value=grid.nx), label="xmax2"
    )
    ymin2 = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.ny - 1), label="ymin2"
    )
    ymax2 = data.draw(
        hyp_st.integers(min_value=ymin2 + 1, max_value=grid.ny), label="ymax2"
    )
    zmin2 = data.draw(
        hyp_st.integers(min_value=0, max_value=grid.nz - 1), label="zmin2"
    )
    zmax2 = data.draw(
        hyp_st.integers(min_value=zmin2 + 1, max_value=grid.nz), label="zmax2"
    )

    # ========================================
    # test bed
    # ========================================
    x1, x2 = slice(xmin1, xmax1), slice(xmin2, xmax2)
    y1, y2 = slice(ymin1, ymax1), slice(ymin2, ymax2)
    z1, z2 = slice(zmin1, zmax1), slice(zmin2, zmax2)

    # one input state
    drc = DataRetrieverComposite(grid, field1_name, field1_units, x1, y1, z1)
    data = drc(state)
    assert data[0].shape[0] == xmax1 - xmin1
    assert data[0].shape[1] == ymax1 - ymin1
    assert data[0].shape[2] == zmax1 - zmin1
    f1_units = (
        field1_units
        if field1_units is not None
        else state[field1_name].attrs["units"]
    )
    assert np.allclose(
        data[0], state[field1_name][x1, y1, z1].to_units(f1_units)
    )

    # two input states
    drc = DataRetrieverComposite(
        (grid, grid),
        ((field1_name,), (field2_name,)),
        ((field1_units,), (field2_units,)),
        ((x1,), (x2,)),
        ((y1,), (y2,)),
        ((z1,), (z2,)),
    )
    data = drc(state, state)
    assert data[0].shape[0] == xmax1 - xmin1
    assert data[0].shape[1] == ymax1 - ymin1
    assert data[0].shape[2] == zmax1 - zmin1
    assert np.allclose(
        data[0], state[field1_name][x1, y1, z1].to_units(f1_units)
    )
    assert data[1].shape[0] == xmax2 - xmin2
    assert data[1].shape[1] == ymax2 - ymin2
    assert data[1].shape[2] == zmax2 - zmin2
    f2_units = (
        field2_units
        if field2_units is not None
        else state[field2_name].attrs["units"]
    )
    assert np.allclose(
        data[1], state[field2_name][x2, y2, z2].to_units(f2_units)
    )


if __name__ == "__main__":
    pytest.main([__file__])
