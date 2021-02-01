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
    strategies as hyp_st,
)
import pytest

import gt4py as gt

from tasmania.python.domain.grid import NumericalGrid
from tasmania.python.domain.subclasses.horizontal_boundaries.dirichlet import (
    Dirichlet,
    Dirichlet1DX,
    Dirichlet1DY,
    dispatch as dispatch_dirichlet,
)
from tasmania.python.domain.subclasses.horizontal_boundaries.identity import (
    Identity,
    Identity1DX,
    Identity1DY,
    dispatch as dispatch_identity,
)
from tasmania.python.domain.subclasses.horizontal_boundaries.periodic import (
    Periodic,
    Periodic1DX,
    Periodic1DY,
    dispatch as dispatch_periodic,
)
from tasmania.python.domain.subclasses.horizontal_boundaries.relaxed import (
    Relaxed,
    Relaxed1DX,
    Relaxed1DY,
    dispatch as dispatch_relaxed,
)
from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.utils.storage import (
    deepcopy_array_dict,
    deepcopy_dataarray_dict,
)

from tests.conf import backend as conf_backend, dtype as conf_dtype
from tests.strategies import (
    st_horizontal_boundary,
    st_physical_grid,
    st_state,
)
from tests.utilities import compare_arrays, compare_dataarrays, hyp_settings


def test_registry():
    # dirichlet
    assert "dirichlet" in HorizontalBoundary.registry
    assert HorizontalBoundary.registry["dirichlet"] == dispatch_dirichlet

    # identity
    assert "identity" in HorizontalBoundary.registry
    assert HorizontalBoundary.registry["identity"] == dispatch_identity

    # periodic
    assert "periodic" in HorizontalBoundary.registry
    assert HorizontalBoundary.registry["periodic"] == dispatch_periodic

    # relaxed
    assert "relaxed" in HorizontalBoundary.registry
    assert HorizontalBoundary.registry["relaxed"] == dispatch_relaxed


@given(data=hyp_st.data())
def test_factory(data):
    grid_xy = data.draw(
        st_physical_grid(xaxis_length=(3, 20), yaxis_length=(3, 20))
    )
    grid_x = data.draw(
        st_physical_grid(xaxis_length=(3, 20), yaxis_length=(1, 1))
    )
    grid_y = data.draw(
        st_physical_grid(xaxis_length=(1, 1), yaxis_length=(3, 20))
    )

    # dirichlet
    obj = HorizontalBoundary.factory("dirichlet", grid_xy, 1)
    assert isinstance(obj, Dirichlet)
    obj = HorizontalBoundary.factory("dirichlet", grid_x, 1)
    assert isinstance(obj, Dirichlet1DX)
    obj = HorizontalBoundary.factory("dirichlet", grid_y, 1)
    assert isinstance(obj, Dirichlet1DY)

    # identity
    obj = HorizontalBoundary.factory("identity", grid_xy, 1)
    assert isinstance(obj, Identity)
    obj = HorizontalBoundary.factory("identity", grid_x, 1)
    assert isinstance(obj, Identity1DX)
    obj = HorizontalBoundary.factory("identity", grid_y, 1)
    assert isinstance(obj, Identity1DY)

    # periodic
    obj = HorizontalBoundary.factory("periodic", grid_xy, 1)
    assert isinstance(obj, Periodic)
    obj = HorizontalBoundary.factory("periodic", grid_x, 1)
    assert isinstance(obj, Periodic1DX)
    obj = HorizontalBoundary.factory("periodic", grid_y, 1)
    assert isinstance(obj, Periodic1DY)

    # relaxed
    obj = HorizontalBoundary.factory("relaxed", grid_xy, 1, nr=1)
    assert isinstance(obj, Relaxed)
    obj = HorizontalBoundary.factory("relaxed", grid_x, 1, nr=1)
    assert isinstance(obj, Relaxed1DX)
    obj = HorizontalBoundary.factory("relaxed", grid_y, 1, nr=1)
    assert isinstance(obj, Relaxed1DY)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_enforce_raw(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype)

    pgrid = data.draw(st_physical_grid(storage_options=so), label="grid")
    hb = data.draw(
        st_horizontal_boundary(
            pgrid, backend=backend, backend_options=bo, storage_options=so
        ),
        label="hb",
    )
    ngrid = NumericalGrid(hb)

    state = data.draw(st_state(ngrid, backend=backend, storage_options=so))

    field_properties = {}
    for key in state:
        if key != "time" and data.draw(hyp_st.booleans()):
            field_properties[key] = {"units": state[key].attrs["units"]}

    # ========================================
    # test
    # ========================================
    hb.reference_state = state

    raw_state = {"time": state["time"]}
    raw_state.update({key: state[key].data for key in state if key != "time"})
    raw_state_dc = deepcopy_array_dict(raw_state)

    hb.enforce_raw(raw_state, field_properties)

    for key in state:
        if key != "time":
            if key in field_properties:
                hb.enforce_field(
                    raw_state_dc[key],
                    field_name=key,
                    field_units=state[key].attrs["units"],
                    time=state["time"],
                )
            compare_arrays(raw_state[key], raw_state_dc[key])


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf_backend)
@pytest.mark.parametrize("dtype", conf_dtype)
def test_enforce(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype)

    pgrid = data.draw(st_physical_grid(storage_options=so), label="grid")
    hb = data.draw(
        st_horizontal_boundary(pgrid, backend=backend, storage_options=so),
        label="hb",
    )
    ngrid = NumericalGrid(hb)

    state = data.draw(st_state(ngrid, backend=backend, storage_options=so))

    field_names = []
    for key in state:
        if key != "time" and data.draw(hyp_st.booleans()):
            field_names.append(key)

    # ========================================
    # test
    # ========================================
    hb.reference_state = state

    state_dc = deepcopy_dataarray_dict(state)

    hb.enforce(state, field_names)

    for key in state:
        if key != "time":
            if key in field_names:
                hb.enforce_field(
                    state_dc[key].values,
                    field_name=key,
                    field_units=state[key].attrs["units"],
                    time=state["time"],
                )
            compare_dataarrays(state[key], state_dc[key])


if __name__ == "__main__":
    pytest.main([__file__])
