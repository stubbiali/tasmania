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
import numpy as np
from sympl import DataArray
from typing import Mapping, Optional, Sequence, TYPE_CHECKING

try:
    import cupy
except (ImportError, ModuleNotFoundError):
    cupy = np

from tasmania.python.framework.allocators import as_storage, ones, zeros
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.utils import typingx
from tasmania.python.utils.data import get_physical_constants
from tasmania.python.utils.meteo import (
    convert_relative_humidity_to_water_vapor,
)
from tasmania.python.utils.storage import get_dataarray_3d, get_storage_shape

if TYPE_CHECKING:
    from tasmania.python.domain.grid import Grid
    from tasmania.python.framework.options import StorageOptions


default_physical_constants = {
    "gas_constant_of_dry_air": DataArray(
        287.05, attrs={"units": "J K^-1 kg^-1"}
    ),
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
    time: typingx.Datetime,
    x_velocity: DataArray,
    y_velocity: DataArray,
    brunt_vaisala: DataArray,
    moist: bool = False,
    precipitation: bool = False,
    relative_humidity: float = 0.5,
    physical_constants: Optional[Mapping[str, DataArray]] = None,
    *,
    backend: str = "numpy",
    storage_shape: Optional[Sequence[int]] = None,
    storage_options: Optional["StorageOptions"] = None,
) -> typingx.DataArrayDict:
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
        ``True`` to include some water species in the model state,
        ``False`` for a fully dry configuration. Defaults to ``False``.
    precipitation : `bool`, optional
        ``True`` if the model takes care of precipitation,
        ``False`` otherwise. Defaults to ``False``.
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

    backend : `str`, optional
        The backend.
    storage_shape : `Sequence[int]`, optional
        The shape of the storages allocated within the class.
    storage_options : `StorageOptions`, optional
        Storage-related options.

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
    pcs = get_physical_constants(
        default_physical_constants, physical_constants
    )
    Rd = pcs["gas_constant_of_dry_air"]
    g = pcs["gravitational_acceleration"]
    pref = pcs["reference_air_pressure"]
    cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

    # get storage shape and define the allocator
    storage_shape = get_storage_shape(storage_shape, (nx + 1, ny + 1, nz + 1))

    def allocate():
        return zeros(
            backend, shape=storage_shape, storage_options=storage_options
        )

    def allocate_numpy():
        return zeros(
            "numpy", shape=storage_shape, storage_options=storage_options
        )

    def from_numpy(data):
        return as_storage(backend, data=data, storage_options=storage_options)

    # initialize the velocity components
    u = allocate()
    u[: nx + 1, :ny, :nz] = x_velocity.to_units("m s^-1").values.item()
    u_np = to_numpy(u)
    v = allocate()
    v[:nx, : ny + 1, :nz] = y_velocity.to_units("m s^-1").values.item()
    v_np = to_numpy(v)

    # compute the geometric height of the half levels
    theta1d = grid.z.to_units("K").values[np.newaxis, np.newaxis, :]
    h_np = allocate_numpy()
    h_np[:nx, :ny, nz] = hs
    for k in range(nz - 1, -1, -1):
        h_np[:nx, :ny, k : k + 1] = h_np[:nx, :ny, k + 1 : k + 2] + g * dz / (
            (bv ** 2) * theta1d[:, :, k : k + 1]
        )
    h = from_numpy(h_np)

    # initialize the Exner function
    exn_np = allocate_numpy()
    exn_np[:nx, :ny, nz] = cp
    for k in range(nz - 1, -1, -1):
        exn_np[:nx, :ny, k : k + 1] = exn_np[:nx, :ny, k + 1 : k + 2] - dz * (
            g ** 2
        ) / ((bv ** 2) * (theta1d[:, :, k : k + 1] ** 2))
    exn = from_numpy(exn_np)

    # diagnose the air pressure
    p_np = allocate_numpy()
    p_np[:nx, :ny, : nz + 1] = pref * (
        (exn_np[:nx, :ny, : nz + 1] / cp) ** (cp / Rd)
    )
    p = from_numpy(p_np)

    # diagnose the Montgomery potential
    mtg_s = (
        g * h_np[:, :, nz : nz + 1]
        + grid.z_on_interface_levels.to_units("K").values[-1]
        * exn_np[:, :, nz : nz + 1]
    )
    mtg_np = allocate_numpy()
    mtg_np[:nx, :ny, nz - 1] = (
        mtg_s[:nx, :ny, 0] + 0.5 * dz * exn_np[:nx, :ny, nz]
    )
    for k in range(nz - 2, -1, -1):
        mtg_np[:nx, :ny, k] = (
            mtg_np[:nx, :ny, k + 1] + dz * exn_np[:nx, :ny, k + 1]
        )
    mtg = from_numpy(mtg_np)

    # diagnose the isentropic density and the momenta
    s_np = allocate_numpy()
    s_np[:nx, :ny, :nz] = -(
        p_np[:nx, :ny, :nz] - p_np[:nx, :ny, 1 : nz + 1]
    ) / (g * dz)
    s = from_numpy(s_np)
    su_np = allocate_numpy()
    su_np[:nx, :ny, :nz] = (
        0.5
        * s_np[:nx, :ny, :nz]
        * (u_np[:nx, :ny, :nz] + u_np[1 : nx + 1, :ny, :nz])
    )
    su = from_numpy(su_np)
    sv_np = allocate_numpy()
    sv_np[:nx, :ny, :nz] = (
        0.5
        * s_np[:nx, :ny, :nz]
        * (v_np[:nx, :ny, :nz] + v_np[:nx, 1 : ny + 1, :nz])
    )
    sv = from_numpy(sv_np)

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
            "m^2 s^-2",
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
        rho_np = allocate_numpy()
        rho_np[:nx, :ny, :nz] = (
            s_np[:nx, :ny, :nz]
            * dz
            / (h_np[:nx, :ny, :nz] - h_np[:nx, :ny, 1 : nz + 1])
        )
        rho = from_numpy(rho_np)
        state["air_density"] = get_dataarray_3d(
            rho,
            grid,
            "kg m^-3",
            name="air_density",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )
        temp_np = allocate_numpy()
        temp_np[:nx, :ny, :nz] = (
            0.5
            * (exn_np[:nx, :ny, :nz] + exn_np[:nx, :ny, 1 : nz + 1])
            * theta1d[:, :, :nz]
            / cp
        )
        temp = from_numpy(temp_np)
        state["air_temperature"] = get_dataarray_3d(
            temp,
            grid,
            "K",
            name="air_temperature",
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        # initialize the relative humidity
        rh = allocate()
        rh[...] = relative_humidity
        rh_da = get_dataarray_3d(rh, grid, "1")

        # interpolate the pressure at the main levels
        p_unstg_np = allocate_numpy()
        p_unstg_np[:nx, :ny, :nz] = 0.5 * (
            p_np[:nx, :ny, :nz] + p_np[:nx, :ny, 1 : nz + 1]
        )
        p_unstg = from_numpy(p_unstg_np)
        p_unstg_da = get_dataarray_3d(
            p_unstg, grid, "Pa", grid_shape=(nx, ny, nz), set_coordinates=False
        )

        # diagnose the mass fraction of water vapor
        qv_np = convert_relative_humidity_to_water_vapor(
            "tetens", p_unstg_da, state["air_temperature"], rh_da
        )
        qv = from_numpy(qv_np)
        # qv = allocate()
        state[mfwv] = get_dataarray_3d(
            qv,
            grid,
            "g g^-1",
            name=mfwv,
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        # initialize the mass fraction of cloud liquid water and
        # precipitation water
        qc = allocate()
        state[mfcw] = get_dataarray_3d(
            qc,
            grid,
            "g g^-1",
            name=mfcw,
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )
        qr = allocate()
        state[mfpw] = get_dataarray_3d(
            qr,
            grid,
            "g g^-1",
            name=mfpw,
            grid_shape=(nx, ny, nz),
            set_coordinates=False,
        )

        # precipitation and accumulated precipitation
        if precipitation:
            state["precipitation"] = get_dataarray_3d(
                zeros(
                    backend,
                    shape=(storage_shape[0], storage_shape[1], 1),
                    storage_options=storage_options,
                ),
                grid,
                "mm hr^-1",
                name="precipitation",
                grid_shape=(nx, ny, 1),
                set_coordinates=False,
            )
            state["accumulated_precipitation"] = get_dataarray_3d(
                zeros(
                    backend,
                    shape=(storage_shape[0], storage_shape[1], 1),
                    storage_options=storage_options,
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
    time: typingx.Datetime,
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
    *,
    backend: str = "numpy",
    storage_shape: Optional[Sequence[int]] = None,
    storage_options: Optional["StorageOptions"] = None,
) -> typingx.DataArrayDict:
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
        ``True`` to include some water species in the model state,
        ``False`` for a fully dry configuration. Defaults to ``False``.
    precipitation : `bool`, optional
        ``True`` if the model takes care of precipitation,
        ``False`` otherwise. Defaults to ``False``.
    physical_constants : `dict[str, sympl.DataArray]`, optional
        Dictionary whose keys are strings indicating physical constants used
        within this object, and whose values are :class:`sympl.DataArray`\s
        storing the values and units of those constants. The constants might be:

            * 'gas_constant_of_dry_air', in units compatible with [J kg^-1 K^-1];
            * 'gravitational_acceleration', in units compatible with [m s^-2];
            * 'reference_air_pressure', in units compatible with [Pa];
            * 'specific_heat_of_dry_air_at_constant_pressure', \
                in units compatible with [J kg^-1 K^-1].

    backend : `str`, optional
        The backend.
    storage_shape : `Sequence[int]`, optional
        The shape of the storages allocated within the class.
    storage_options : `StorageOptions`, optional
        Storage-related options.

    Return
    ------
    dict[str, sympl.DataArray]
        The model state dictionary.
    """
    # shortcuts
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dz = grid.dz.to_units("K").values.item()

    # get needed physical constants
    pcs = get_physical_constants(
        default_physical_constants, physical_constants
    )
    Rd = pcs["gas_constant_of_dry_air"]
    g = pcs["gravitational_acceleration"]
    pref = pcs["reference_air_pressure"]
    cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

    # get storage shape and define the allocator
    storage_shape = get_storage_shape(storage_shape, (nx + 1, ny + 1, nz + 1))

    def allocate():
        return zeros(
            backend, shape=storage_shape, storage_options=storage_options
        )

    def allocate_numpy():
        return zeros(
            "numpy", shape=storage_shape, storage_options=storage_options
        )

    def from_numpy(data):
        return as_storage(backend, data=data, storage_options=storage_options)

    # initialize the air pressure
    theta1d = grid.z_on_interface_levels.to_units("K").values[
        np.newaxis, np.newaxis, :
    ]
    temp = background_temperature.to_units("K").values.item()
    p_np = allocate_numpy()
    p_np[:nx, :ny, : nz + 1] = pref * (
        (temp / theta1d[:, :, : nz + 1]) ** (cp / Rd)
    )

    # initialize the Exner function
    exn_np = allocate_numpy()
    exn_np[:nx, :ny, : nz + 1] = cp * temp / theta1d[:, :, : nz + 1]
    exn = from_numpy(exn_np)

    # diagnose the height of the half levels
    hs = grid.topography.profile.to_units("m").values
    h_np = allocate_numpy()
    h_np[:nx, :ny, nz] = hs
    for k in range(nz - 1, -1, -1):
        h_np[:nx, :ny, k] = h_np[:nx, :ny, k + 1] - Rd / (cp * g) * (
            theta1d[:, :, k] * exn_np[:nx, :ny, k]
            + theta1d[:, :, k + 1] * exn_np[:nx, :ny, k + 1]
        ) * (p_np[:nx, :ny, k] - p_np[:nx, :ny, k + 1]) / (
            p_np[:nx, :ny, k] + p_np[:nx, :ny, k + 1]
        )
    h = from_numpy(h_np)

    # warm/cool bubble
    if bubble_maximum_perturbation is not None:
        x = grid.x.to_units("m").values[:, np.newaxis, np.newaxis]
        y = grid.y.to_units("m").values[np.newaxis, :, np.newaxis]
        cx = bubble_center_x.to_units("m").values.item()
        cy = bubble_center_y.to_units("m").values.item()
        ch = bubble_center_height.to_units("m").values.item()
        r = bubble_radius.to_units("m").values.item()
        delta = bubble_maximum_perturbation.to_units("K").values.item()

        d = np.sqrt(
            ((x - cx) ** 2 + (y - cy) ** 2 + (h_np - ch) ** 2) / r ** 2
        )
        t = allocate_numpy()
        t[:nx, :ny, : nz + 1] = temp + delta * (
            np.cos(0.5 * np.pi * d)
        ) ** 2 * (d <= 1.0)
    else:
        t = allocate_numpy()
        t[:nx, :ny, : nz + 1] = temp

    # diagnose the air pressure
    p_np[:nx, :ny, : nz + 1] = pref * (
        (t[:nx, :ny, : nz + 1] / theta1d[:, :, : nz + 1]) ** (cp / Rd)
    )
    p = from_numpy(p_np)

    # diagnose the Exner function
    exn_np = allocate_numpy()
    exn_np[:nx, :ny, : nz + 1] = cp * temp / theta1d[:, :, : nz + 1]
    exn = from_numpy(exn_np)

    # diagnose the Montgomery potential
    hs = grid.topography.profile.to_units("m").values
    mtg_s = cp * temp + g * hs
    mtg_np = allocate_numpy()
    mtg_np[:nx, :ny, nz - 1] = mtg_s + 0.5 * dz * exn_np[:nx, :ny, nz]
    for k in range(nz - 2, -1, -1):
        mtg_np[:nx, :ny, k] = (
            mtg_np[:nx, :ny, k + 1] + dz * exn_np[:nx, :ny, k + 1]
        )
    mtg = from_numpy(mtg_np)

    # initialize the velocity components
    u = allocate()
    u[: nx + 1, :ny, :nz] = x_velocity.to_units("m s^-1").values.item()
    u_np = to_numpy(u)
    v = allocate()
    v[:nx, : ny + 1, :nz] = y_velocity.to_units("m s^-1").values.item()
    v_np = to_numpy(v)

    # diagnose the isentropic density and the momenta
    s_np = allocate_numpy()
    s_np[:nx, :ny, :nz] = -(
        p_np[:nx, :ny, :nz] - p_np[:nx, :ny, 1 : nz + 1]
    ) / (g * dz)
    s = from_numpy(s_np)
    su_np = allocate_numpy()
    su_np[:nx, :ny, :nz] = (
        0.5
        * s_np[:nx, :ny, :nz]
        * (u_np[:nx, :ny, :nz] + u_np[1 : nx + 1, :ny, :nz])
    )
    su = from_numpy(su_np)
    sv_np = allocate_numpy()
    sv_np[:nx, :ny, :nz] = (
        0.5
        * s_np[:nx, :ny, :nz]
        * (v_np[:nx, :ny, :nz] + v_np[:nx, 1 : ny + 1, :nz])
    )
    sv = from_numpy(sv_np)

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

    return state
