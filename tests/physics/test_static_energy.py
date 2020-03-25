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
import pytest

import gt4py as gt

from tasmania.python.physics.static_energy import DryStaticEnergy, MoistStaticEnergy
from tasmania import get_dataarray_3d

from tests.conf import (
    backend as conf_backend,
    datatype as conf_dtype,
    default_origin as conf_dorigin,
    nb as conf_nb,
)
from tests.utilities import (
    compare_dataarrays,
    st_domain,
    st_floats,
    st_one_of,
    st_raw_field,
)


mfwv = "mass_fraction_of_water_vapor_in_air"


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_dry(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.numerical_grid if grid_type == "numerical" else domain.physical_grid

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = 1
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    t = data.draw(
        st_raw_field(
            storage_shape,
            -1e3,
            1e3,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="dse",
    )
    h = data.draw(
        st_raw_field(
            storage_shape,
            -1e3,
            1e3,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="qv",
    )

    time = data.draw(hyp_st.datetimes(), label="time")

    # ========================================
    # test bed
    # ========================================
    state = {
        "time": time,
        "air_temperature": get_dataarray_3d(
            t, grid, "K", grid_shape=(nx, ny, nz), set_coordinates=False
        ),
        "height": get_dataarray_3d(
            h, grid, "m", grid_shape=(nx, ny, nz), set_coordinates=False
        ),
        "height_on_interface_levels": get_dataarray_3d(
            h, grid, "m", grid_shape=(nx, ny, nz + 1), set_coordinates=False
        ),
    }

    #
    # height
    #
    comp = DryStaticEnergy(
        domain,
        grid_type,
        height_on_interface_levels=False,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
        rebuild=False,
    )

    diagnostics = comp(state)

    assert "montgomery_potential" in diagnostics
    compare_dataarrays(
        diagnostics["montgomery_potential"][:nx, :ny, :nz],
        get_dataarray_3d(
            comp._cp * t[:nx, :ny, :nz] + comp._g * h[:nx, :ny, :nz],
            grid,
            "m^2 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
    )

    assert len(diagnostics) == 1

    #
    # height_on_interface_levels
    #
    comp = DryStaticEnergy(
        domain,
        grid_type,
        height_on_interface_levels=True,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
        rebuild=False,
    )

    diagnostics = comp(state)

    assert "montgomery_potential" in diagnostics
    compare_dataarrays(
        diagnostics["montgomery_potential"][:nx, :ny, :nz],
        get_dataarray_3d(
            comp._cp * t[:nx, :ny, :nz]
            + comp._g * 0.5 * (h[:nx, :ny, :nz] + h[:nx, :ny, 1 : nz + 1]),
            grid,
            "m^2 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
    )

    assert len(diagnostics) == 1


@settings(
    suppress_health_check=(
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ),
    deadline=None,
)
@given(hyp_st.data())
def test_moist(data):
    gt.storage.prepare_numpy()

    # ========================================
    # random data generation
    # ========================================
    gt_powered = data.draw(hyp_st.booleans(), label="gt_powered")
    backend = data.draw(st_one_of(conf_backend), label="backend")
    dtype = data.draw(st_one_of(conf_dtype), label="dtype")
    default_origin = data.draw(st_one_of(conf_dorigin), label="default_origin")

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
        ),
        label="domain",
    )
    grid_type = data.draw(st_one_of(("physical", "numerical")), label="grid_type")
    grid = domain.numerical_grid if grid_type == "numerical" else domain.physical_grid

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    dse = data.draw(
        st_raw_field(
            storage_shape,
            -1e3,
            1e3,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="dse",
    )
    qv = data.draw(
        st_raw_field(
            storage_shape,
            -1e3,
            1e3,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        ),
        label="qv",
    )

    time = data.draw(hyp_st.datetimes(), label="time")

    # ========================================
    # test bed
    # ========================================
    state = {
        "time": time,
        "montgomery_potential": get_dataarray_3d(
            dse, grid, "m^2 s^-2", grid_shape=(nx, ny, nz), set_coordinates=False
        ),
        mfwv: get_dataarray_3d(
            qv, grid, "g g^-1", grid_shape=(nx, ny, nz), set_coordinates=False
        ),
    }

    comp = MoistStaticEnergy(
        domain,
        grid_type,
        gt_powered=gt_powered,
        backend=backend,
        dtype=dtype,
        default_origin=default_origin,
        storage_shape=storage_shape,
        rebuild=False,
    )

    diagnostics = comp(state)

    assert "moist_static_energy" in diagnostics
    compare_dataarrays(
        diagnostics["moist_static_energy"][:nx, :ny, :nz],
        get_dataarray_3d(
            dse + comp._lhvw * qv,
            grid,
            "m^2 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )[:nx, :ny, :nz],
        compare_coordinate_values=False,
    )

    assert len(diagnostics) == 1


if __name__ == "__main__":
    pytest.main([__file__])
