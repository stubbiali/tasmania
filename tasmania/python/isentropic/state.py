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
from sympl import DataArray
from typing import Mapping, Optional, TYPE_CHECKING

try:
    import cupy
except (ImportError, ModuleNotFoundError):
    cupy = np

from tasmania.python.utils import taz_types
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.meteo_utils import convert_relative_humidity_to_water_vapor
from tasmania.python.utils.storage_utils import (
    get_dataarray_3d,
    get_storage_shape,
    ones,
    zeros,
)

if TYPE_CHECKING:
    from tasmania.python.domain.grid import Grid


_d_physical_constants = {
    "gas_constant_of_dry_air": DataArray(287.05, attrs={"units": "J K^-1 kg^-1"}),
    "gravitational_acceleration": DataArray(9.81, attrs={"units": "m s^-2"}),
    "reference_air_pressure": DataArray(1.0e5, attrs={"units": "Pa"}),
    "specific_heat_of_dry_air_at_constant_pressure": DataArray(
        1004.0, attrs={"units": "J K^-1 kg^-1"}
    ),
}


# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def get_isentropic_state_from_brunt_vaisala_frequency(
    grid: "Grid",
    time: taz_types.datetime_t,
    x_velocity: DataArray,
    y_velocity: DataArray,
    brunt_vaisala: DataArray,
    moist: bool = False,
    precipitation: bool = False,
    relative_humidity: float = 0.5,
    physical_constants: Optional[Mapping[str, DataArray]] = None,
    gt_powered: bool = True,
    *,
    backend: str = "numpy",
    dtype: taz_types.dtype_t = np.float64,
    default_origin: Optional[taz_types.triplet_int_t] = None,
    storage_shape: Optional[taz_types.triplet_int_t] = None,
    managed_memory: bool = False
) -> taz_types.dataarray_dict_t:
    """
    Compute a valid state for the isentropic model given
    the Brunt-Vaisala frequency.

    Parameters
    ----------
    grid : tasmania.Grid
        The underlying grid.
    time : datetime.datetime
        The time instant at which the state is defined.
    x_velocity : sympl.DataArray
        1-item :class:`sympl.DataArray` representing the uniform
        background x-velocity, in units compatible with [m s^-1].
    y_velocity : sympl.DataArray
        1-item :class:`sympl.DataArray` representing the uniform
        background y-velocity, in units compatible with [m s^-1].
    brunt_vaisala : sympl.DataArray
        1-item :class:`sympl.DataArray` representing the uniform
        Brunt-Vaisala frequency, in units compatible with [s^-1].
    moist : `bool`, optional
        `True` to include some water species in the model state,
        `False` for a fully dry configuration. Defaults to `False`.
    precipitation : `bool`, optional
        `True` if the model takes care of precipitation,
        `False` otherwise. Defaults to `False`.
    relative_humidity : `float`, optional
        The relative humidity in decimals. Defaults to 0.5.
    physical_constants : `dict[str, sympl.DataArray]`, optional
        Dictionary whose keys are strings indicating physical constants used
        within this object, and whose values are :class:`sympl.DataArray`\s
        storing the values and units of those constants. The constants might be:

            * 'gas_constant_of_dry_air', in units compatible with [J kg^-1 K^-1];
            * 'gravitational_acceleration', in units compatible with [m s^-2];
            * 'reference_air_pressure', in units compatible with [Pa];
            * 'specific_heat_of_dry_air_at_constant_pressure', \
                in units compatible with [J kg^-1 K^-1].

    gt_powered : `bool`, optional
        TODO
    backend : `str`, optional
        The GT4Py backend.
    dtype : `data-type`, optional
        Data type of the storages.
    default_origin : `tuple[int]`, optional
        Storage default origin.
    storage_shape : `tuple[int]`, optional
        Shape of the storages.
    managed_memory : `bool`, optional
        `True` to allocate the storages as managed memory, `False` otherwise.

    Return
    ------
    dict[str, sympl.DataArray]
        The model state dictionary.
    """
    # shortcuts
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz = grid.dz.to_units("K").values.item()
    hs = grid.topography.profile.to_units("m").values
    bv = brunt_vaisala.to_units("s^-1").values.item()

    # get needed physical constants
    pcs = get_physical_constants(_d_physical_constants, physical_constants)
    Rd = pcs["gas_constant_of_dry_air"]
    g = pcs["gravitational_acceleration"]
    pref = pcs["reference_air_pressure"]
    cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

    # get storage shape and define the allocator
    storage_shape = get_storage_shape(storage_shape, (nx + 1, ny + 1, nz + 1))

    def allocate():
        return zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

    # initialize the velocity components
    u = allocate()
    u[: nx + 1, :ny, :nz] = x_velocity.to_units("m s^-1").values.item()
    v = allocate()
    v[:nx, : ny + 1, :nz] = y_velocity.to_units("m s^-1").values.item()

    # compute the geometric height of the half levels
    theta1d = grid.z.to_units("K").values[np.newaxis, np.newaxis, :]
    h = allocate()
    h[:nx, :ny, nz] = hs
    for k in range(nz - 1, -1, -1):
        h[:nx, :ny, k : k + 1] = h[:nx, :ny, k + 1 : k + 2] + g * dz / (
            (bv ** 2) * theta1d[:, :, k : k + 1]
        )

    # initialize the Exner function
    exn = allocate()
    exn[:nx, :ny, nz] = cp
    for k in range(nz - 1, -1, -1):
        exn[:nx, :ny, k : k + 1] = exn[:nx, :ny, k + 1 : k + 2] - dz * (g ** 2) / (
            (bv ** 2) * (theta1d[:, :, k : k + 1] ** 2)
        )

    # diagnose the air pressure
    p = allocate()
    p[:nx, :ny, : nz + 1] = pref * ((exn[:nx, :ny, : nz + 1] / cp) ** (cp / Rd))

    # diagnose the Montgomery potential
    mtg_s = (
        g * h[:, :, nz : nz + 1]
        + grid.z_on_interface_levels.to_units("K").values[-1] * exn[:, :, nz : nz + 1]
    )
    mtg = allocate()
    mtg[:nx, :ny, nz - 1] = mtg_s[:nx, :ny, 0] + 0.5 * dz * exn[:nx, :ny, nz]
    for k in range(nz - 2, -1, -1):
        mtg[:nx, :ny, k] = mtg[:nx, :ny, k + 1] + dz * exn[:nx, :ny, k + 1]

    # diagnose the isentropic density and the momenta
    s = allocate()
    s[:nx, :ny, :nz] = -(p[:nx, :ny, :nz] - p[:nx, :ny, 1 : nz + 1]) / (g * dz)
    su = allocate()
    su[:nx, :ny, :nz] = (
        0.5 * s[:nx, :ny, :nz] * (u[:nx, :ny, :nz] + u[1 : nx + 1, :ny, :nz])
    )
    sv = allocate()
    sv[:nx, :ny, :nz] = (
        0.5 * s[:nx, :ny, :nz] * (v[:nx, :ny, :nz] + v[:nx, 1 : ny + 1, :nz])
    )

    # instantiate the return state
    state = {
        "time": time,
        "air_isentropic_density": get_dataarray_3d(
            s,
            grid,
            "kg m^-2 K^-1",
            name="air_isentropic_density",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        "air_pressure_on_interface_levels": get_dataarray_3d(
            p,
            grid,
            "Pa",
            name="air_pressure_on_interface_levels",
            grid_shape=(nx, ny, nz + 1),
            set_coordinates=False,
        ),
        "exner_function_on_interface_levels": get_dataarray_3d(
            exn,
            grid,
            "J K^-1 kg^-1",
            name="exner_function_on_interface_levels",
            grid_shape=(nx, ny, nz + 1),
            set_coordinates=False,
        ),
        "height_on_interface_levels": get_dataarray_3d(
            h,
            grid,
            "m",
            name="height_on_interface_levels",
            grid_shape=(nx, ny, nz + 1),
            set_coordinates=False,
        ),
        "montgomery_potential": get_dataarray_3d(
            mtg,
            grid,
            "J kg^-1",
            name="montgomery_potential",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        "x_momentum_isentropic": get_dataarray_3d(
            su,
            grid,
            "kg m^-1 K^-1 s^-1",
            name="x_momentum_isentropic",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        "x_velocity_at_u_locations": get_dataarray_3d(
            u,
            grid,
            "m s^-1",
            name="x_velocity_at_u_locations",
            grid_shape=(nx + 1, ny, nz),
            set_coordinates=False,
        ),
        "y_momentum_isentropic": get_dataarray_3d(
            sv,
            grid,
            "kg m^-1 K^-1 s^-1",
            name="y_momentum_isentropic",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        "y_velocity_at_v_locations": get_dataarray_3d(
            v,
            grid,
            "m s^-1",
            name="y_velocity_at_v_locations",
            grid_shape=(nx, ny + 1, nz),
            set_coordinates=False,
        ),
    }

    if moist:
        # diagnose the air density and temperature
        rho = allocate()
        rho[:nx, :ny, :nz] = (
            s[:nx, :ny, :nz] * dz / (h[:nx, :ny, :nz] - h[:nx, :ny, 1 : nz + 1])
        )
        state["air_density"] = get_dataarray_3d(
            rho,
            grid,
            "kg m^-3",
            name="air_density",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )
        temp = allocate()
        temp[:nx, :ny, :nz] = (
            0.5 * (exn[:nx, :ny, :nz] + exn[:nx, :ny, 1 : nz + 1]) * theta1d / cp
        )
        state["air_temperature"] = get_dataarray_3d(
            temp,
            grid,
            "K",
            name="air_temperature",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        # initialize the relative humidity
        rh = relative_humidity * ones(
            storage_shape,
            gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
        )
        rh_ = get_dataarray_3d(rh, grid, "1")

        # interpolate the pressure at the main levels
        p_unstg = allocate()
        p_unstg[:nx, :ny, :nz] = 0.5 * (p[:nx, :ny, :nz] + p[:nx, :ny, 1 : nz + 1])
        p_unstg_ = get_dataarray_3d(
            p_unstg, grid, "Pa", grid_shape=(nx, ny, nz), set_coordinates=False
        )

        # diagnose the mass fraction of water vapor
        qv = convert_relative_humidity_to_water_vapor(
            "tetens", p_unstg_, state["air_temperature"], rh_
        )
        state[mfwv] = get_dataarray_3d(
            qv, grid, "g g^-1", name=mfwv, grid_shape=(nx, ny, nz), set_coordinates=False
        )

        # initialize the mass fraction of cloud liquid water and precipitation water
        qc = allocate()
        state[mfcw] = get_dataarray_3d(
            qc, grid, "g g^-1", name=mfcw, grid_shape=(nx, ny, nz), set_coordinates=False
        )
        qr = allocate()
        state[mfpw] = get_dataarray_3d(
            qr, grid, "g g^-1", name=mfpw, grid_shape=(nx, ny, nz), set_coordinates=False
        )

        # precipitation and accumulated precipitation
        if precipitation:
            state["precipitation"] = get_dataarray_3d(
                zeros(
                    (storage_shape[0], storage_shape[1], 1),
                    gt_powered=gt_powered,
                    backend=backend,
                    dtype=dtype,
                    default_origin=default_origin,
                ),
                grid,
                "mm hr^-1",
                name="precipitation",
                grid_shape=(nx, ny, 1),
                set_coordinates=False,
            )
            state["accumulated_precipitation"] = get_dataarray_3d(
                zeros(
                    (storage_shape[0], storage_shape[1], 1),
                    gt_powered=gt_powered,
                    backend=backend,
                    dtype=dtype,
                    default_origin=default_origin,
                ),
                grid,
                "mm",
                name="accumulated_precipitation",
                grid_shape=(nx, ny, 1),
                set_coordinates=False,
            )

    return state


def get_isentropic_state_from_temperature(
    grid: "Grid",
    time: taz_types.datetime_t,
    x_velocity: DataArray,
    y_velocity: DataArray,
    background_temperature: DataArray,
    bubble_center_x: Optional[DataArray] = None,
    bubble_center_y: Optional[DataArray] = None,
    bubble_center_height: Optional[DataArray] = None,
    bubble_radius: Optional[DataArray] = None,
    bubble_maximum_perturbation: Optional[DataArray] = None,
    moist: bool = False,
    precipitation: bool = False,
    physical_constants: Optional[Mapping[str, DataArray]] = None,
    gt_powered: bool = True,
    *,
    backend: str = "numpy",
    dtype: taz_types.dtype_t = np.float64,
    default_origin: Optional[taz_types.triplet_int_t] = None,
    storage_shape: Optional[taz_types.triplet_int_t] = None,
    managed_memory: bool = False
) -> taz_types.dataarray_dict_t:
    """
    Compute a valid state for the isentropic model given
    the air temperature.

    Parameters
    ----------
    grid : tasmania.Grid
        The underlying grid.
    time : datetime.datetime
        The time instant at which the state is defined.
    x_velocity : sympl.DataArray
        1-item :class:`sympl.DataArray` representing the uniform
        background x-velocity, in units compatible with [m s^-1].
    y_velocity : sympl.DataArray
        1-item :class:`sympl.DataArray` representing the uniform
        background y-velocity, in units compatible with [m s^-1].
    background_temperature : sympl.DataArray
        1-item :class:`sympl.DataArray` representing the background
        temperature, in units compatible with [K].
    bubble_center_x : `sympl.DataArray`, optional
        1-item :class:`sympl.DataArray` representing the x-location
        of the center of the warm/cool bubble.
    bubble_center_y : `sympl.DataArray`, optional
        1-item :class:`sympl.DataArray` representing the y-location
        of the center of the warm/cool bubble.
    bubble_center_height : `sympl.DataArray`, optional
        1-item :class:`sympl.DataArray` representing the height
        of the center of the warm/cool bubble.
    bubble_radius : `sympl.DataArray`, optional
        1-item :class:`sympl.DataArray` representing the radius
        of the warm/cool bubble.
    bubble_maximum_perturbation : `sympl.DataArray`, optional
        1-item :class:`sympl.DataArray` representing the temperature
        perturbation in the center of the warm/cool bubble with respect
        to the ambient conditions.
    moist : `bool`, optional
        `True` to include some water species in the model state,
        `False` for a fully dry configuration. Defaults to `False`.
    precipitation : `bool`, optional
        `True` if the model takes care of precipitation,
        `False` otherwise. Defaults to `False`.
    physical_constants : `dict[str, sympl.DataArray]`, optional
        Dictionary whose keys are strings indicating physical constants used
        within this object, and whose values are :class:`sympl.DataArray`\s
        storing the values and units of those constants. The constants might be:

            * 'gas_constant_of_dry_air', in units compatible with [J kg^-1 K^-1];
            * 'gravitational_acceleration', in units compatible with [m s^-2];
            * 'reference_air_pressure', in units compatible with [Pa];
            * 'specific_heat_of_dry_air_at_constant_pressure', \
                in units compatible with [J kg^-1 K^-1].

    gt_powered : `bool`, optional
        TODO
    backend : `str`, optional
        The GT4Py backend.
    dtype : `data-type`, optional
        Data type of the storages.
    default_origin : `tuple[int]`, optional
        Storage default origin.
    storage_shape : `tuple[int]`, optional
        Shape of the storages.
    managed_memory : `bool`, optional
        `True` to allocate the storages as managed memory, `False` otherwise.

    Return
    ------
    dict[str, sympl.DataArray]
        The model state dictionary.
    """
    # shortcuts
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz = grid.dz.to_units("K").values.item()

    # get needed physical constants
    pcs = get_physical_constants(_d_physical_constants, physical_constants)
    Rd = pcs["gas_constant_of_dry_air"]
    g = pcs["gravitational_acceleration"]
    pref = pcs["reference_air_pressure"]
    cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

    # get storage shape and define the allocator
    storage_shape = get_storage_shape(storage_shape, (nx + 1, ny + 1, nz + 1))

    def allocate():
        return zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

    # initialize the air pressure
    theta1d = grid.z_on_interface_levels.to_units("K").values
    theta = allocate()
    theta[:nx, :ny, : nz + 1] = theta1d[np.newaxis, np.newaxis, :]
    temp = background_temperature.to_units("K").values.item()
    p = allocate()
    p[:nx, :ny, : nz + 1] = pref * ((temp / theta[:nx, :ny, : nz + 1]) ** (cp / Rd))

    # initialize the Exner function
    exn = allocate()
    exn[:nx, :ny, : nz + 1] = cp * temp / theta[:nx, :ny, : nz + 1]

    # diagnose the height of the half levels
    hs = grid.topography.profile.to_units("m").values
    h = allocate()
    h[:nx, :ny, nz] = hs
    for k in range(nz - 1, -1, -1):
        h[:nx, :ny, k] = h[:nx, :ny, k + 1] - Rd / (cp * g) * (
            theta[:nx, :ny, k] * exn[:nx, :ny, k]
            + theta[:nx, :ny, k + 1] * exn[:nx, :ny, k + 1]
        ) * (p[:nx, :ny, k] - p[:nx, :ny, k + 1]) / (p[:nx, :ny, k] + p[:nx, :ny, k + 1])

    # warm/cool bubble
    if bubble_maximum_perturbation is not None:
        x = grid.x.to_units("m").values[:, np.newaxis, np.newaxis]
        y = grid.y.to_units("m").values[np.newaxis, :, np.newaxis]
        cx = bubble_center_x.to_units("m").values.item()
        cy = bubble_center_y.to_units("m").values.item()
        ch = bubble_center_height.to_units("m").values.item()
        r = bubble_radius.to_units("m").values.item()
        delta = bubble_maximum_perturbation.to_units("K").values.item()

        d = np.sqrt(((x - cx) ** 2 + (y - cy) ** 2 + (h - ch) ** 2) / r ** 2)
        t = temp * np.ones((nx, ny, nz + 1), dtype=dtype) + delta * (
            np.cos(0.5 * np.pi * d)
        ) ** 2 * (d <= 1.0)
    else:
        t = allocate()
        t[:nx, :ny, : nz + 1] = temp

    # diagnose the air pressure
    p[:nx, :ny, : nz + 1] = pref * (
        (t[:nx, :ny, : nz + 1] / theta[:nx, :ny, : nz + 1]) ** (cp / Rd)
    )

    # diagnose the Exner function
    exn[:nx, :ny, : nz + 1] = cp * temp / theta[:nx, :ny, : nz + 1]

    # diagnose the Montgomery potential
    hs = grid.topography.profile.to_units("m").values
    mtg_s = cp * temp + g * hs
    mtg = allocate()
    mtg[:nx, :ny, nz - 1] = mtg_s + 0.5 * dz * exn[:nx, :ny, nz]
    for k in range(nz - 2, -1, -1):
        mtg[:nx, :ny, k] = mtg[:nx, :ny, k + 1] + dz * exn[:nx, :ny, k + 1]

    # initialize the velocity components
    u = allocate()
    u[: nx + 1, :ny, :nz] = x_velocity.to_units("m s^-1").values.item()
    v = allocate()
    v[:nx, : ny + 1, :nz] = y_velocity.to_units("m s^-1").values.item()

    # diagnose the isentropic density and the momenta
    s = allocate()
    s[:nx, :ny, :nz] = -(p[:nx, :ny, :nz] - p[:nx, :ny, 1 : nz + 1]) / (g * dz)
    su = allocate()
    su[:nx, :ny, :nz] = (
        0.5 * s[:nx, :ny, :nz] * (u[:nx, :ny, :nz] + u[1 : nx + 1, :ny, :nz])
    )
    sv = allocate()
    sv[:nx, :ny, :nz] = (
        0.5 * s[:nx, :ny, :nz] * (v[:nx, :ny, :nz] + v[:nx, 1 : ny + 1, :nz])
    )

    # instantiate the return state
    state = {
        "time": time,
        "air_isentropic_density": get_dataarray_3d(
            s,
            grid,
            "kg m^-2 K^-1",
            name="air_isentropic_density",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        "air_pressure_on_interface_levels": get_dataarray_3d(
            p,
            grid,
            "Pa",
            name="air_pressure_on_interface_levels",
            grid_shape=(nx, ny, nz + 1),
            set_coordinates=False,
        ),
        "exner_function_on_interface_levels": get_dataarray_3d(
            exn,
            grid,
            "J K^-1 kg^-1",
            name="exner_function_on_interface_levels",
            grid_shape=(nx, ny, nz + 1),
            set_coordinates=False,
        ),
        "height_on_interface_levels": get_dataarray_3d(
            h,
            grid,
            "m",
            name="height_on_interface_levels",
            grid_shape=(nx, ny, nz + 1),
            set_coordinates=False,
        ),
        "montgomery_potential": get_dataarray_3d(
            mtg,
            grid,
            "J kg^-1",
            name="montgomery_potential",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        "x_momentum_isentropic": get_dataarray_3d(
            su,
            grid,
            "kg m^-1 K^-1 s^-1",
            name="x_momentum_isentropic",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        "x_velocity_at_u_locations": get_dataarray_3d(
            u,
            grid,
            "m s^-1",
            name="x_velocity_at_u_locations",
            grid_shape=(nx + 1, ny, nz),
            set_coordinates=False,
        ),
        "y_momentum_isentropic": get_dataarray_3d(
            sv,
            grid,
            "kg m^-1 K^-1 s^-1",
            name="y_momentum_isentropic",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        ),
        "y_velocity_at_v_locations": get_dataarray_3d(
            v,
            grid,
            "m s^-1",
            name="y_velocity_at_v_locations",
            grid_shape=(nx, ny + 1, nz),
            set_coordinates=False,
        ),
    }

    if moist:
        raise NotImplementedError()

        # diagnose the air density and temperature
        rho = s * dz / (h[:, :, :-1] - h[:, :, 1:])
        state["air_density"] = get_dataarray_3d(rho, grid, "kg m^-3", name="air_density")
        state["air_temperature"] = get_dataarray_3d(
            0.5 * (t[:, :, :-1] + t[:, :, 1:]), grid, "K", name="air_temperature"
        )

        # initialize the relative humidity
        rhmax, L, kc = 0.98, 10, 11
        k = (nz - 1) - np.arange(kc - L + 1, kc + L)
        rh = np.zeros((nx, ny, nz), dtype=dtype)
        rh[:, :, k] = rhmax * (np.cos(abs(k - kc) * np.pi / (2.0 * L))) ** 2
        rh_ = get_dataarray_3d(rh, grid, "1")

        # interpolate the pressure at the main levels
        p_unstg = 0.5 * (p[:, :, :-1] + p[:, :, 1:])
        p_unstg_ = get_dataarray_3d(p_unstg, grid, "Pa")

        # diagnose the mass fraction fo water vapor
        qv = convert_relative_humidity_to_water_vapor(
            "goff_gratch", p_unstg_, state["air_temperature"], rh_
        )
        state[mfwv] = get_dataarray_3d(qv, grid, "g g^-1", name=mfwv)

        # initialize the mass fraction of cloud liquid water and precipitation water
        qc = np.zeros((nx, ny, nz), dtype=dtype)
        state[mfcw] = get_dataarray_3d(qc, grid, "g g^-1", name=mfcw)
        qr = np.zeros((nx, ny, nz), dtype=dtype)
        state[mfpw] = get_dataarray_3d(qr, grid, "g g^-1", name=mfpw)

        # precipitation and accumulated precipitation
        if precipitation:
            state["precipitation"] = get_dataarray_3d(
                np.zeros((nx, ny), dtype=dtype), grid, "mm hr^-1", name="precipitation"
            )
            state["accumulated_precipitation"] = get_dataarray_3d(
                np.zeros((nx, ny), dtype=dtype),
                grid,
                "mm",
                name="accumulated_precipitation",
            )

    return state
