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
    given,
    HealthCheck,
    settings,
    strategies as hyp_st,
    reproduce_failure,
)
import pytest

import gt4py

from tasmania.python.domain.grid import NumericalGrid
from tasmania.python.domain.horizontal_boundaries.dirichlet import (
    Dirichlet,
    Dirichlet1DX,
    Dirichlet1DY,
    dispatch as dispatch_dirichlet,
)
from tasmania.python.domain.horizontal_boundaries.identity import (
    Identity,
    Identity1DX,
    Identity1DY,
    dispatch as dispatch_identity,
)
from tasmania.python.domain.horizontal_boundaries.periodic import (
    Periodic,
    Periodic1DX,
    Periodic1DY,
    dispatch as dispatch_periodic,
)
from tasmania.python.domain.horizontal_boundaries.relaxed import (
    Relaxed,
    Relaxed1DX,
    Relaxed1DY,
    dispatch as dispatch_relaxed,
)
from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.utils.storage_utils import (
    deepcopy_array_dict,
    deepcopy_dataarray_dict,
)

from tests.conf import backend as conf_backend
from tests.strategies import (
    st_horizontal_boundary,
    st_one_of,
    st_physical_grid,
    st_state,
)
from tests.utilities import compare_arrays, compare_dataarrays


def test_register():
    # dirichlet
    assert "dirichlet" in HorizontalBoundary.register
    assert HorizontalBoundary.register["dirichlet"] == dispatch_dirichlet

    # identity
    assert "identity" in HorizontalBoundary.register
    assert HorizontalBoundary.register["identity"] == dispatch_identity

    # periodic
    assert "periodic" in HorizontalBoundary.register
    assert HorizontalBoundary.register["periodic"] == dispatch_periodic

    # relaxed
    assert "relaxed" in HorizontalBoundary.register
    assert HorizontalBoundary.register["relaxed"] == dispatch_relaxed


def test_factory():
    # dirichlet
    obj = HorizontalBoundary.factory("dirichlet", 10, 20, 3)
    assert isinstance(obj, Dirichlet)
    obj = HorizontalBoundary.factory("dirichlet", 10, 1, 3)
    assert isinstance(obj, Dirichlet1DX)
    obj = HorizontalBoundary.factory("dirichlet", 1, 20, 3)
    assert isinstance(obj, Dirichlet1DY)

    # identity
    obj = HorizontalBoundary.factory("identity", 10, 20, 3)
    assert isinstance(obj, Identity)
    obj = HorizontalBoundary.factory("identity", 10, 1, 3)
    assert isinstance(obj, Identity1DX)
    obj = HorizontalBoundary.factory("identity", 1, 20, 3)
    assert isinstance(obj, Identity1DY)

    # periodic
    obj = HorizontalBoundary.factory("periodic", 10, 20, 3)
    assert isinstance(obj, Periodic)
    obj = HorizontalBoundary.factory("periodic", 10, 1, 3)
    assert isinstance(obj, Periodic1DX)
    obj = HorizontalBoundary.factory("periodic", 1, 20, 3)
    assert isinstance(obj, Periodic1DY)

    # relaxed
    obj = HorizontalBoundary.factory("relaxed", 10, 20, 1, nr=2)
    assert isinstance(obj, Relaxed)
    obj = HorizontalBoundary.factory("relaxed", 10, 1, 1, nr=2)
    assert isinstance(obj, Relaxed1DX)
    obj = HorizontalBoundary.factory("relaxed", 1, 20, 1, nr=2)
    assert isinstance(obj, Relaxed1DY)


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_enforce_raw(data):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    if gt_powered:
        gt4py.storage.prepare_numpy()

    pgrid = data.draw(st_physical_grid(), label="grid")
    hb = data.draw(st_horizontal_boundary(pgrid.nx, pgrid.ny), label="hb")
    ngrid = NumericalGrid(pgrid, hb)

    state = data.draw(st_state(ngrid, gt_powered=gt_powered, backend=backend))

    field_properties = {}
    for key in state:
        if key is not "time" and data.draw(hyp_st.booleans()):
            field_properties[key] = {"units": state[key].attrs["units"]}

    # ========================================
    # test
    # ========================================
    hb.reference_state = state

    raw_state = {"time": state["time"]}
    raw_state.update({key: state[key].values for key in state if key is not "time"})
    raw_state_dc = deepcopy_array_dict(raw_state)

    hb.enforce_raw(raw_state, field_properties, ngrid)

    for key in state:
        if key is not "time":
            if key in field_properties:
                hb.enforce_field(
                    raw_state_dc[key],
                    field_name=key,
                    field_units=state[key].attrs["units"],
                    time=state["time"],
                    grid=ngrid,
                )
            compare_arrays(raw_state[key], raw_state_dc[key])


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_enforce(data):
    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")

    if gt_powered:
        gt4py.storage.prepare_numpy()

    pgrid = data.draw(st_physical_grid(), label="grid")
    hb = data.draw(st_horizontal_boundary(pgrid.nx, pgrid.ny), label="hb")
    ngrid = NumericalGrid(pgrid, hb)

    state = data.draw(st_state(ngrid, gt_powered=gt_powered, backend=backend))

    field_names = []
    for key in state:
        if key is not "time" and data.draw(hyp_st.booleans()):
            field_names.append(key)

    # ========================================
    # test
    # ========================================
    hb.reference_state = state

    state_dc = deepcopy_dataarray_dict(state)

    hb.enforce(state, field_names, ngrid)

    for key in state:
        if key is not "time":
            if key in field_names:
                hb.enforce_field(
                    state_dc[key].values,
                    field_name=key,
                    field_units=state[key].attrs["units"],
                    time=state["time"],
                    grid=ngrid,
                )
            compare_dataarrays(state[key], state_dc[key])


if __name__ == "__main__":
    pytest.main([__file__])
