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
    reproduce_failure,
    strategies as hyp_st,
)
import pytest

from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.physics.static_energy import (
    DryStaticEnergy,
    MoistStaticEnergy,
)
from tasmania.python.utils.storage import get_dataarray_3d

from tests import conf
from tests.strategies import st_domain, st_one_of, st_raw_field
from tests.utilities import compare_dataarrays, hyp_settings


mfwv = "mass_fraction_of_water_vapor_in_air"


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_dry(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.numerical_grid
        if grid_type == "numerical"
        else domain.physical_grid
    )

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = 1
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    t = data.draw(
        st_raw_field(
            storage_shape, -1e3, 1e3, backend=backend, storage_options=so
        ),
        label="dse",
    )
    h = data.draw(
        st_raw_field(
            storage_shape, -1e3, 1e3, backend=backend, storage_options=so
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
    t_np, h_np = to_numpy(t), to_numpy(h)

    #
    # height
    #
    comp = DryStaticEnergy(
        domain,
        grid_type,
        height_on_interface_levels=False,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    diagnostics = comp(state)

    assert "montgomery_potential" in diagnostics
    compare_dataarrays(
        diagnostics["montgomery_potential"],
        get_dataarray_3d(
            comp._cp * t_np[:nx, :ny, :nz] + comp._g * h_np[:nx, :ny, :nz],
            grid,
            "m^2 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )

    assert len(diagnostics) == 1

    #
    # height_on_interface_levels
    #
    comp = DryStaticEnergy(
        domain,
        grid_type,
        height_on_interface_levels=True,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    diagnostics = comp(state)

    assert "montgomery_potential" in diagnostics
    compare_dataarrays(
        diagnostics["montgomery_potential"],
        get_dataarray_3d(
            comp._cp * t_np[:nx, :ny, :nz]
            + comp._g
            * 0.5
            * (h_np[:nx, :ny, :nz] + h_np[:nx, :ny, 1 : nz + 1]),
            grid,
            "m^2 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )

    assert len(diagnostics) == 1


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("backend", conf.backend)
@pytest.mark.parametrize("dtype", conf.dtype)
def test_moist(data, backend, dtype):
    # ========================================
    # random data generation
    # ========================================
    aligned_index = data.draw(
        st_one_of(conf.aligned_index), label="aligned_index"
    )
    bo = BackendOptions(rebuild=False)
    so = StorageOptions(dtype=dtype, aligned_index=aligned_index)

    domain = data.draw(
        st_domain(
            xaxis_length=(1, 30),
            yaxis_length=(1, 30),
            zaxis_length=(1, 20),
            backend=backend,
            backend_options=bo,
            storage_options=so,
        ),
        label="domain",
    )
    grid_type = data.draw(
        st_one_of(("physical", "numerical")), label="grid_type"
    )
    grid = (
        domain.numerical_grid
        if grid_type == "numerical"
        else domain.physical_grid
    )

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dnx = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnx")
    dny = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dny")
    dnz = data.draw(hyp_st.integers(min_value=0, max_value=1), label="dnz")
    storage_shape = (nx + dnx, ny + dny, nz + dnz)

    dse = data.draw(
        st_raw_field(
            storage_shape, -1e3, 1e3, backend=backend, storage_options=so
        ),
        label="dse",
    )
    qv = data.draw(
        st_raw_field(
            storage_shape, -1e3, 1e3, backend=backend, storage_options=so
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
            dse,
            grid,
            "m^2 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        mfwv: get_dataarray_3d(
            qv, grid, "g g^-1", grid_shape=(nx, ny, nz), set_coordinates=False
        ),
    }
    dse_np, qv_np = to_numpy(dse), to_numpy(qv)

    comp = MoistStaticEnergy(
        domain,
        grid_type,
        backend=backend,
        backend_options=bo,
        storage_shape=storage_shape,
        storage_options=so,
    )

    diagnostics = comp(state)

    assert "moist_static_energy" in diagnostics
    compare_dataarrays(
        diagnostics["moist_static_energy"],
        get_dataarray_3d(
            dse_np + comp._lhvw * qv_np,
            grid,
            "m^2 s^-2",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        compare_coordinate_values=False,
        slice=(slice(0, nx), slice(0, ny), slice(0, nz)),
    )

    assert len(diagnostics) == 1


if __name__ == "__main__":
    pytest.main([__file__])
