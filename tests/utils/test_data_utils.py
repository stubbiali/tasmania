# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
from sympl import DataArray

from tasmania.python.utils import data as dutils, storage as sutils
from tasmania.python.utils.exceptions import ConstantNotFoundError

from tests.conf import dtype as conf_dtype
from tests.strategies import (
    st_physical_grid,
    st_physical_horizontal_grid,
    st_raw_field,
)
from tests.utilities import compare_arrays, hyp_settings


def test_get_constant():
    u = dutils.get_constant("gravitational_acceleration", "m s^-2")
    assert u == 9.80665

    v = dutils.get_constant(
        "foo", "m", default_value=DataArray(10.0, attrs={"units": "km"})
    )
    assert v == 10000.0

    w = dutils.get_constant(
        "pippo", "1", default_value=DataArray(10.0, attrs={"units": "1"})
    )
    assert w == 10.0

    try:
        _ = dutils.get_constant("foo", "K")
    except ValueError:
        assert True

    try:
        _ = dutils.get_constant("bar", "K")
    except ConstantNotFoundError:
        assert True


def test_get_physical_constants():
    d_physical_constants = {
        "gravitational_acceleration": DataArray(
            9.80665e-3, attrs={"units": "km s^-2"}
        ),
        "gas_constant_of_dry_air": DataArray(
            287.05, attrs={"units": "J K^-1 kg^-1"}
        ),
        "gas_constant_of_water_vapor": DataArray(
            461.52, attrs={"units": "hJ K^-1 g^-1"}
        ),
        "latent_heat_of_vaporization_of_water": DataArray(
            2.5e6, attrs={"units": "J kg^-1"}
        ),
        "foo_constant": DataArray(1, attrs={"units": "1"}),
    }

    physical_constants = {
        "latent_heat_of_vaporization_of_water": DataArray(
            1.5e3, attrs={"units": "kJ kg^-1"}
        )
    }

    raw_constants = dutils.get_physical_constants(
        d_physical_constants, physical_constants
    )

    assert raw_constants["gravitational_acceleration"] == 9.80665e-3
    assert raw_constants["gas_constant_of_dry_air"] == 287.0
    assert raw_constants["gas_constant_of_water_vapor"] == 461.5e-5
    assert raw_constants["latent_heat_of_vaporization_of_water"] == 1.5e6
    assert raw_constants["foo_constant"] == 1.0


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("dtype", conf_dtype)
def test_get_dataarray_2d(data, dtype):
    grid = data.draw(st_physical_horizontal_grid(dtype=dtype))

    #
    # nx, ny
    #
    raw_array = data.draw(
        st_raw_field(
            (grid.nx, grid.ny), min_value=-1e5, max_value=1e5, dtype=dtype
        )
    )
    units = data.draw(hyp_st.text(max_size=10))
    name = data.draw(hyp_st.text(max_size=10))

    array = sutils.get_dataarray_2d(raw_array, grid, units, name)

    assert array.shape == (grid.nx, grid.ny)
    assert array.dims == (grid.x.dims[0], grid.y.dims[0])
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx+1, ny
    #
    raw_array = data.draw(
        st_raw_field(
            (grid.nx + 1, grid.ny), min_value=-1e5, max_value=1e5, dtype=dtype
        )
    )
    units = data.draw(hyp_st.text(max_size=10))
    name = data.draw(hyp_st.text(max_size=10))

    array = sutils.get_dataarray_2d(raw_array, grid, units, name)

    assert array.shape == (grid.nx + 1, grid.ny)
    assert array.dims == (grid.x_at_u_locations.dims[0], grid.y.dims[0])
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx, ny+1
    #
    raw_array = data.draw(
        st_raw_field(
            (grid.nx, grid.ny + 1), min_value=-1e5, max_value=1e5, dtype=dtype
        )
    )
    units = data.draw(hyp_st.text(max_size=10))
    name = data.draw(hyp_st.text(max_size=10))

    array = sutils.get_dataarray_2d(raw_array, grid, units, name)

    assert array.shape == (grid.nx, grid.ny + 1)
    assert array.dims == (grid.x.dims[0], grid.y_at_v_locations.dims[0])
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx+1, ny+1
    #
    raw_array = data.draw(
        st_raw_field(
            (grid.nx + 1, grid.ny + 1),
            min_value=-1e5,
            max_value=1e5,
            dtype=dtype,
        )
    )
    units = data.draw(hyp_st.text(max_size=10))
    name = data.draw(hyp_st.text(max_size=10))

    array = sutils.get_dataarray_2d(raw_array, grid, units, name)

    assert array.shape == (grid.nx + 1, grid.ny + 1)
    assert array.dims == (
        grid.x_at_u_locations.dims[0],
        grid.y_at_v_locations.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)


@hyp_settings
@given(data=hyp_st.data())
@pytest.mark.parametrize("dtype", conf_dtype)
def test_get_dataarray_3d(data, dtype):
    # ========================================
    # random data generation
    # ========================================
    grid = data.draw(st_physical_grid(dtype=dtype))

    nx, ny, nz = grid.grid_xy.nx, grid.grid_xy.ny, grid.nz

    raw_array_ = data.draw(
        st_raw_field(
            (nx + 1, ny + 1, nz + 1),
            min_value=-1e5,
            max_value=1e5,
            dtype=dtype,
        )
    )
    units = data.draw(hyp_st.text(max_size=10))
    name = data.draw(hyp_st.text(max_size=10))

    #
    # nx, ny, nz
    #
    raw_array = raw_array_[:-1, :-1, :-1]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx, ny, nz)
    assert array.dims == (
        grid.grid_xy.x.dims[0],
        grid.grid_xy.y.dims[0],
        grid.z.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx+1, ny, nz
    #
    raw_array = raw_array_[:, :-1, :-1]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx + 1, ny, nz)
    assert array.dims == (
        grid.grid_xy.x_at_u_locations.dims[0],
        grid.grid_xy.y.dims[0],
        grid.z.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx, ny+1, nz
    #
    raw_array = raw_array_[:-1, :, :-1]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx, ny + 1, nz)
    assert array.dims == (
        grid.grid_xy.x.dims[0],
        grid.grid_xy.y_at_v_locations.dims[0],
        grid.z.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx+1, ny+1, nz
    #
    raw_array = raw_array_[:, :, :-1]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx + 1, ny + 1, nz)
    assert array.dims == (
        grid.grid_xy.x_at_u_locations.dims[0],
        grid.grid_xy.y_at_v_locations.dims[0],
        grid.z.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx, ny, nz+1
    #
    raw_array = raw_array_[:-1, :-1, :]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx, ny, nz + 1)
    assert array.dims == (
        grid.grid_xy.x.dims[0],
        grid.grid_xy.y.dims[0],
        grid.z_on_interface_levels.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx+1, ny, nz+1
    #
    raw_array = raw_array_[:, :-1, :]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx + 1, ny, nz + 1)
    assert array.dims == (
        grid.grid_xy.x_at_u_locations.dims[0],
        grid.grid_xy.y.dims[0],
        grid.z_on_interface_levels.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx, ny+1, nz+1
    #
    raw_array = raw_array_[:-1, :, :]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx, ny + 1, nz + 1)
    assert array.dims == (
        grid.grid_xy.x.dims[0],
        grid.grid_xy.y_at_v_locations.dims[0],
        grid.z_on_interface_levels.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx+1, ny+1, nz+1
    #
    raw_array = raw_array_[...]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx + 1, ny + 1, nz + 1)
    assert array.dims == (
        grid.grid_xy.x_at_u_locations.dims[0],
        grid.grid_xy.y_at_v_locations.dims[0],
        grid.z_on_interface_levels.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx+1, ny+1, 1
    #
    raw_array = raw_array_[:, :, 0:1]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx + 1, ny + 1, 1)
    assert array.dims == (
        grid.grid_xy.x_at_u_locations.dims[0],
        grid.grid_xy.y_at_v_locations.dims[0],
        grid.z.dims[0] + "_at_surface_level" if nz > 1 else grid.z.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # 1, ny, nz
    #
    raw_array = raw_array_[0:1, :-1, :-1]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (1, ny, nz)
    assert array.dims == (
        grid.grid_xy.x.dims[0] + "_gp" if nx > 1 else grid.grid_xy.x.dims[0],
        grid.grid_xy.y.dims[0],
        grid.z.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # nx+1, 1, nz
    #
    raw_array = raw_array_[:, 0:1, :-1]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (nx + 1, 1, nz)
    assert array.dims == (
        grid.grid_xy.x_at_u_locations.dims[0],
        grid.grid_xy.y.dims[0] + "_gp" if ny > 1 else grid.grid_xy.y.dims[0],
        grid.z.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)

    #
    # 1, 1, nz+1
    #
    raw_array = raw_array_[0:1, 0:1, :]
    array = sutils.get_dataarray_3d(raw_array, grid, units, name)

    assert array.shape == (1, 1, nz + 1)
    assert array.dims == (
        grid.grid_xy.x.dims[0] + "_gp" if nx > 1 else grid.grid_xy.x.dims[0],
        grid.grid_xy.y.dims[0] + "_gp" if ny > 1 else grid.grid_xy.y.dims[0],
        grid.z_on_interface_levels.dims[0],
    )
    assert array.attrs["units"] == units
    assert array.name == name
    compare_arrays(raw_array, array.data)
    # assert id(raw_array) == id(array.data)


if __name__ == "__main__":
    pytest.main([__file__])
