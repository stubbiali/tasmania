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
import numpy as np
from pint import UnitRegistry
from sympl import DataArray

import gt4py as gt


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


default_physical_constants = {
    "gas_constant_of_dry_air": DataArray(287.05, attrs={"units": "J K^-1 kg^-1"}),
    "gravitational_acceleration": DataArray(9.81, attrs={"units": "m s^-2"}),
    "reference_air_pressure": DataArray(1.0e5, attrs={"units": "Pa"}),
    "specific_heat_of_dry_air_at_constant_pressure": DataArray(
        1004.0, attrs={"units": "J K^-1 kg^-1"}
    ),
}


unit_registry = UnitRegistry()


def compare_datetimes(td1, td2):
    assert abs(td1 - td2).total_seconds() <= 1e-6


def compare_arrays(field_a, field_b, atol=1e-8, rtol=1e-5):
    # to prevent segfaults
    # gt.storage.restore_numpy()

    # field_a[np.isinf(field_a)] = np.nan
    # field_b[np.isinf(field_b)] = np.nan

    try:
        assert np.allclose(field_a, field_b, equal_nan=True, atol=atol, rtol=rtol)
    except (NotImplementedError, RuntimeError):
        try:
            assert np.allclose(
                np.asarray(field_a),
                np.asarray(field_b),
                equal_nan=True,
                atol=atol,
                rtol=rtol,
            )
        except AttributeError:
            assert False

    # gt.storage.prepare_numpy()


def compare_dataarrays(da1, da2, compare_coordinate_values=True):
    """ Assert whether two :class:`sympl.DataArray`\s are equal. """
    assert len(da1.dims) == len(da2.dims)

    assert all([dim1 == dim2 for dim1, dim2 in zip(da1.dims, da2.dims)])

    try:
        assert all(
            [
                da1.coords[key].attrs["units"] == da2.coords[key].attrs["units"]
                for key in da1.coords
            ]
        )
    except KeyError:
        pass

    if compare_coordinate_values:
        assert all(
            [
                np.allclose(da1.coords[key].values, da2.coords[key].values)
                for key in da1.coords
            ]
        )

    assert unit_registry(da1.attrs["units"]) == unit_registry(da2.attrs["units"])

    compare_arrays(da1.values, da2.values)


def get_float_width(dtype):
    """ Get the number of bits used by ``dtype``. """
    if dtype == np.float16:
        return 16
    elif dtype == np.float32:
        return 32
    else:
        return 64


def get_interval(el0, el1, dims, units, increasing):
    """ Generate a 2-elements DataArray representing a domain interval. """
    invert = ((el0 > el1) and increasing) or ((el0 < el1) and not increasing)
    return DataArray(
        [el1, el0] if invert else [el0, el1], dims=dims, attrs={"units": units}
    )


def get_nanoseconds(secs):
    """ Convert seconds in nanoseconds. """
    return int((secs - int(secs * 1e9) * 1e-9) * 1e12)


def get_xaxis(domain_x, nx, dtype):
    x_v = (
        np.linspace(domain_x.values[0], domain_x.values[1], nx, dtype=dtype)
        if nx > 1
        else np.array([0.5 * (domain_x.values[0] + domain_x.values[1])], dtype=dtype)
    )
    x = DataArray(
        x_v, coords=[x_v], dims=domain_x.dims, attrs={"units": domain_x.attrs["units"]}
    )

    dx_v = 1.0 if nx == 1 else (domain_x.values[-1] - domain_x.values[0]) / (nx - 1)
    dx_v = 1.0 if dx_v == 0.0 else dx_v
    dx = DataArray(dx_v, attrs={"units": domain_x.attrs["units"]})

    xu_v = np.linspace(x_v[0] - 0.5 * dx_v, x_v[-1] + 0.5 * dx_v, nx + 1, dtype=dtype)
    xu = DataArray(
        xu_v,
        coords=[xu_v],
        dims=(domain_x.dims[0] + "_at_u_locations"),
        attrs={"units": domain_x.attrs["units"]},
    )

    return x, xu, dx


def get_yaxis(domain_y, ny, dtype):
    y_v = (
        np.linspace(domain_y.values[0], domain_y.values[1], ny, dtype=dtype)
        if ny > 1
        else np.array([0.5 * (domain_y.values[0] + domain_y.values[1])], dtype=dtype)
    )
    y = DataArray(
        y_v, coords=[y_v], dims=domain_y.dims, attrs={"units": domain_y.attrs["units"]}
    )

    dy_v = 1.0 if ny == 1 else (domain_y.values[-1] - domain_y.values[0]) / (ny - 1)
    dy_v = 1.0 if dy_v == 0.0 else dy_v
    dy = DataArray(dy_v, attrs={"units": domain_y.attrs["units"]})

    yv_v = np.linspace(y_v[0] - 0.5 * dy_v, y_v[-1] + 0.5 * dy_v, ny + 1, dtype=dtype)
    yv = DataArray(
        yv_v,
        coords=[yv_v],
        dims=(domain_y.dims[0] + "_at_v_locations"),
        attrs={"units": domain_y.attrs["units"]},
    )

    return y, yv, dy


def get_zaxis(domain_z, nz, dtype):
    zhl_v = np.linspace(domain_z.values[0], domain_z.values[1], nz + 1, dtype=dtype)
    zhl = DataArray(
        zhl_v,
        coords=[zhl_v],
        dims=(domain_z.dims[0] + "_on_interface_levels"),
        attrs={"units": domain_z.attrs["units"]},
    )

    dz_v = (
        (domain_z.values[1] - domain_z.values[0]) / nz
        if domain_z.values[1] > domain_z.values[0]
        else (domain_z.values[0] - domain_z.values[1]) / nz
    )
    dz = DataArray(dz_v, attrs={"units": domain_z.attrs["units"]})

    z_v = 0.5 * (zhl_v[1:] + zhl_v[:-1])
    z = DataArray(
        z_v, coords=[z_v], dims=domain_z.dims, attrs={"units": domain_z.attrs["units"]}
    )

    return z, zhl, dz


def pi_function(time, grid, slice_x, slice_y, field_name, field_units):
    if slice_x is not None:
        li = slice_x.stop - slice_x.start
    else:
        li = (
            grid.nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else grid.nx
        )

    if slice_y is not None:
        lj = slice_y.stop - slice_y.start
    else:
        lj = (
            grid.ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else grid.ny
        )

    out = np.pi * np.ones((li, lj, 1), dtype=grid.x.dtype)

    return out


def get_grid_shape(name, grid):
    return (
        grid.nx + int("at_u_locations" in name or "at_uv_locations" in name),
        grid.ny + int("at_v_locations" in name or "at_uv_locations" in name),
        grid.nz + int("on_interface_levels" in name),
    )
