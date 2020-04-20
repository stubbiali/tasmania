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
import numpy as np
from sympl import DataArray
from typing import Mapping, Optional, TYPE_CHECKING

from gt4py import gtscript, __externals__

# from gt4py.__gtscript__ import computation, interval, PARALLEL, FORWARD, BACKWARD

from tasmania.python.utils import taz_types
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import zeros

if TYPE_CHECKING:
    from tasmania.python.domain.grid import Grid


class IsentropicDiagnostics:
    """
    Class implementing the diagnostic steps of the three-dimensional
    isentropic dynamical core using GT4Py stencils.
    """

    # Default values for the physical constants used in the class
    _d_physical_constants = {
        "air_pressure_at_sea_level": DataArray(1e5, attrs={"units": "Pa"}),
        "gas_constant_of_dry_air": DataArray(287.05, attrs={"units": "J K^-1 kg^-1"}),
        "gravitational_acceleration": DataArray(9.80665, attrs={"units": "m s^-2"}),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }

    def __init__(
        self,
        grid: "Grid",
        physical_constants: Optional[Mapping[str, DataArray]] = None,
        gt_powered: bool = True,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        physical_constants : `dict[str, sympl.DataArray]`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`sympl.DataArray`\s
            storing the values and units of those constants. The constants might be:

                * 'air_pressure_at_sea_level', in units compatible with [Pa];
                * 'gas_constant_of_dry_air', in units compatible with \
                    [J K^-1 kg^-1];
                * 'gravitational acceleration', in units compatible with [m s^-2].
                * 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
                    with [J K^-1 kg^-1].

            Please refer to
            :func:`tasmania.utils.data_utils.get_physical_constants` and
            :obj:`tasmania.IsentropicDiagnostics._d_physical_constants`
            for the default values.
        gt_powered : `bool`, True
            `True` to harness GT4Py, `False` for a vanilla Numpy implementation.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        storage_shape : `tuple[int]`, optional
            Shape of the storages.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.
        """
        # store the input arguments needed at run-time
        self._grid = grid
        self._backend = backend
        self._dtype = dtype
        self._exec_info = exec_info

        # set the values of the physical constants
        pcs = get_physical_constants(self._d_physical_constants, physical_constants)
        self._pcs = pcs

        # set storage shape
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        storage_shape = (
            (nx + 1, ny + 1, nz + 1) if storage_shape is None else storage_shape
        )
        error_msg = "storage_shape must be larger or equal than {}.".format(
            (nx, ny, nz + 1)
        )
        assert storage_shape[0] >= nx, error_msg
        assert storage_shape[1] >= ny, error_msg
        assert storage_shape[2] >= nz + 1, error_msg

        # allocate auxiliary fields
        self._theta = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            mask=[True, True, True],
            managed_memory=managed_memory,
        )
        self._theta[:nx, :ny, : nz + 1] = grid.z_on_interface_levels.to_units(
            "K"
        ).values[np.newaxis, np.newaxis, :]
        self._topo = zeros(
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        # gather external symbols
        externals = {
            "pref": pcs["air_pressure_at_sea_level"],
            "rd": pcs["gas_constant_of_dry_air"],
            "g": pcs["gravitational_acceleration"],
            "cp": pcs["specific_heat_of_dry_air_at_constant_pressure"],
        }

        # instantiate the underlying stencils
        if gt_powered:
            self._stencil_diagnostic_variables = gtscript.stencil(
                definition=self._stencil_diagnostic_variables_gt_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                externals=externals,
                **(backend_opts or {})
            )
            self._stencil_density_and_temperature = gtscript.stencil(
                definition=self._stencil_density_and_temperature_gt_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                externals=externals,
                **(backend_opts or {})
            )
            self._stencil_montgomery = gtscript.stencil(
                definition=self._stencil_montgomery_gt_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                externals=externals,
                **(backend_opts or {})
            )
            self._stencil_height = gtscript.stencil(
                definition=self._stencil_height_gt_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                externals=externals,
                **(backend_opts or {})
            )
        else:
            self._stencil_diagnostic_variables = self._stencil_diagnostic_variables_numpy
            self._stencil_density_and_temperature = (
                self._stencil_density_and_temperature_numpy
            )
            self._stencil_montgomery = self._stencil_montgomery_numpy
            self._stencil_height = self._stencil_height_numpy

    def get_diagnostic_variables(
        self,
        s: taz_types.gtstorage_t,
        pt: float,
        p: taz_types.gtstorage_t,
        exn: taz_types.gtstorage_t,
        mtg: taz_types.gtstorage_t,
        h: taz_types.gtstorage_t,
    ) -> None:
        """
        With the help of the isentropic density and the upper boundary
        condition on the pressure distribution, diagnose the pressure,
        the Exner function, the Montgomery potential, and the geometric
        height of the half-levels.

        Parameters
        ----------
        s : gt4py.storage.storage.Storage
            The isentropic density, in units of [kg m^-2 K^-1].
        pt : float
            The upper boundary condition on the pressure distribution,
            in units of [Pa].
        p : gt4py.storage.storage.Storage
            The buffer for the pressure at the interface levels, in units of [Pa].
        exn : gt4py.storage.storage.Storage
            The buffer for the Exner function at the interface levels,
            in units of [J K^-1 kg^-1].
        mtg : gt4py.storage.storage.Storage
            The buffer for the Montgomery potential, in units of [J kg^-1].
        h : gt4py.storage.storage.Storage
            The buffer for the geometric height of the interface levels,
            in units of [m].
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dz = self._grid.dz.to_units("K").values.item()

        # set the topography
        self._topo[:nx, :ny, nz] = self._grid.topography.profile.to_units("m").values[
            ...
        ]

        # retrieve all the diagnostic variables
        self._stencil_diagnostic_variables(
            in_theta=self._theta,
            in_hs=self._topo,
            in_s=s,
            inout_p=p,
            out_exn=exn,
            inout_mtg=mtg,
            inout_h=h,
            dz=dz,
            pt=pt,
            origin=(0, 0, 0),
            domain=(nx, ny, nz + 1),
            exec_info=self._exec_info,
        )

    def get_montgomery_potential(
        self, s: taz_types.gtstorage_t, pt: float, mtg: taz_types.gtstorage_t
    ) -> None:
        """
        With the help of the isentropic density and the upper boundary
        condition on the pressure distribution, diagnose the Montgomery
        potential.

        Parameters
        ----------
        s : gt4py.storage.storage.Storage
            The isentropic density, in units of [kg m^-2 K^-1].
        pt : float
            The upper boundary condition on the pressure distribution,
            in units of [Pa].
        mtg : gt4py.storage.storage.Storage
            The buffer for the Montgomery potential, in units of [J kg^-1].
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dz = self._grid.dz.to_units("K").values.item()
        theta_s = self._grid.z_on_interface_levels.to_units("K").values[-1]

        # set the topography
        self._topo[:nx, :ny, nz] = self._grid.topography.profile.to_units("m").values[
            ...
        ]

        # run the stencil
        self._stencil_montgomery(
            in_hs=self._topo,
            in_s=s,
            inout_mtg=mtg,
            dz=dz,
            pt=pt,
            theta_s=theta_s,
            origin=(0, 0, 0),
            domain=(nx, ny, nz + 1),
            exec_info=self._exec_info,
        )

    def get_height(
        self, s: taz_types.gtstorage_t, pt: float, h: taz_types.gtstorage_t
    ) -> None:
        """
        With the help of the isentropic density and the upper boundary
        condition on the pressure distribution, diagnose the geometric
        height of the half-levels.

        Parameters
        ----------
        s : gt4py.storage.storage.Storage
            The isentropic density, in units of [kg m^-2 K^-1].
        pt : float
            The upper boundary condition on the pressure distribution,
            in units of [Pa].
        h : gt4py.storage.storage.Storage
            The buffer for the geometric height of the interface levels,
            in units of [m].
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
        dz = self._grid.dz.to_units("K").values.item()

        # set the topography
        self._topo[:nx, :ny, nz] = self._grid.topography.profile.to_units("m").values[
            ...
        ]

        # run the stencil
        self._stencil_height(
            in_theta=self._theta,
            in_hs=self._topo,
            in_s=s,
            inout_h=h,
            dz=dz,
            pt=pt,
            origin=(0, 0, 0),
            domain=(nx, ny, nz + 1),
            exec_info=self._exec_info,
        )

    def get_density_and_temperature(
        self,
        s: taz_types.gtstorage_t,
        exn: taz_types.gtstorage_t,
        h: taz_types.gtstorage_t,
        rho: taz_types.gtstorage_t,
        t: taz_types.gtstorage_t,
    ) -> None:
        """
        With the help of the isentropic density and the geometric height
        of the interface levels, diagnose the air density and temperature.

        Parameters
        ----------
        s : gt4py.storage.storage.Storage
            The isentropic density, in units of [kg m^-2 K^-1].
        exn : gt4py.storage.storage.Storage
            The buffer for the Exner function at the interface levels,
            in units of [J K^-1 kg^-1].
        h : gt4py.storage.storage.Storage
            The geometric height of the interface levels, in units of [m].
        rho : gt4py.storage.storage.Storage
            The buffer for the air density, in units of [kg m^-3].
        t : gt4py.storage.storage.Storage
            The buffer for the air temperature, in units of [K].
        """
        # shortcuts
        nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz

        # run the stencil
        self._stencil_density_and_temperature(
            in_theta=self._theta,
            in_s=s,
            in_exn=exn,
            in_h=h,
            out_rho=rho,
            out_t=t,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

    def _stencil_diagnostic_variables_numpy(
        self,
        in_theta: np.ndarray,
        in_hs: np.ndarray,
        in_s: np.ndarray,
        inout_p: np.ndarray,
        out_exn: np.ndarray,
        inout_mtg: np.ndarray,
        inout_h: np.ndarray,
        *,
        dz: float,
        pt: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        pref = self._pcs["air_pressure_at_sea_level"]
        rd = self._pcs["gas_constant_of_dry_air"]
        g = self._pcs["gravitational_acceleration"]
        cp = self._pcs["specific_heat_of_dry_air_at_constant_pressure"]

        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        kstart, kstop = origin[2], origin[2] + domain[2]

        # retrieve the pressure
        inout_p[i, j, kstart] = pt
        for k in range(kstart + 1, kstop):
            inout_p[i, j, k] = inout_p[i, j, k - 1] + g * dz * in_s[i, j, k - 1]

        # compute the Exner function
        out_exn[i, j, kstart:kstop] = cp * (inout_p[i, j, kstart:kstop] / pref) ** (
            rd / cp
        )

        # compute the Montgomery potential
        mtg_s = (
            in_theta[i, j, kstop - 1] * out_exn[i, j, kstop - 1]
            + g * in_hs[i, j, kstop - 1]
        )
        inout_mtg[i, j, kstop - 2] = mtg_s + 0.5 * dz * out_exn[i, j, kstop - 1]
        for k in range(kstop - 3, kstart - 1, -1):
            inout_mtg[i, j, k] = inout_mtg[i, j, k + 1] + dz * out_exn[i, j, k + 1]

        # compute the geometric height of the isentropes
        inout_h[i, j, kstop - 1] = in_hs[i, j, kstop - 1]
        for k in range(kstop - 2, kstart - 1, -1):
            inout_h[i, j, k] = inout_h[i, j, k + 1] - rd * (
                in_theta[i, j, k] * out_exn[i, j, k]
                + in_theta[i, j, k + 1] * out_exn[i, j, k + 1]
            ) * (inout_p[i, j, k] - inout_p[i, j, k + 1]) / (
                cp * g * (inout_p[i, j, k] + inout_p[i, j, k + 1])
            )

    @staticmethod
    def _stencil_diagnostic_variables_gt_defs(
        in_theta: gtscript.Field["dtype"],
        in_hs: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        inout_p: gtscript.Field["dtype"],
        out_exn: gtscript.Field["dtype"],
        inout_mtg: gtscript.Field["dtype"],
        inout_h: gtscript.Field["dtype"],
        *,
        dz: float,
        pt: float
    ) -> None:
        from __externals__ import cp, g, pref, rd

        # retrieve the pressure
        with computation(FORWARD), interval(0, 1):
            inout_p = pt
        with computation(FORWARD), interval(1, None):
            inout_p = inout_p[0, 0, -1] + g * dz * in_s[0, 0, -1]

        # compute the Exner function
        with computation(PARALLEL), interval(...):
            out_exn = cp * (inout_p[0, 0, 0] / pref) ** (rd / cp)

        # compute the Montgomery potential
        with computation(BACKWARD), interval(-2, -1):
            mtg_s = in_theta[0, 0, 1] * out_exn[0, 0, 1] + g * in_hs[0, 0, 1]
            inout_mtg = mtg_s + 0.5 * dz * out_exn[0, 0, 1]
        with computation(BACKWARD), interval(0, -2):
            inout_mtg = inout_mtg[0, 0, 1] + dz * out_exn[0, 0, 1]

        # compute the geometric height of the isentropes
        with computation(BACKWARD), interval(-1, None):
            inout_h = in_hs[0, 0, 0]
        with computation(BACKWARD), interval(0, -1):
            inout_h = inout_h[0, 0, 1] - rd * (
                in_theta[0, 0, 0] * out_exn[0, 0, 0]
                + in_theta[0, 0, 1] * out_exn[0, 0, 1]
            ) * (inout_p[0, 0, 0] - inout_p[0, 0, 1]) / (
                cp * g * (inout_p[0, 0, 0] + inout_p[0, 0, 1])
            )

    def _stencil_montgomery_numpy(
        self,
        in_hs: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        inout_mtg: gtscript.Field["dtype"],
        *,
        dz: float,
        pt: float,
        theta_s: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        pref = self._pcs["air_pressure_at_sea_level"]
        rd = self._pcs["gas_constant_of_dry_air"]
        g = self._pcs["gravitational_acceleration"]
        cp = self._pcs["specific_heat_of_dry_air_at_constant_pressure"]

        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        kstart, kstop = origin[2], origin[2] + domain[2]

        # retrieve the pressure
        p = deepcopy(in_s)
        p[i, j, kstart] = pt
        for k in range(kstart + 1, kstop):
            p[i, j, k] = p[i, j, k - 1] + g * dz * in_s[i, j, k - 1]

        # compute the Exner function
        exn = cp * (p / pref) ** (rd / cp)

        # compute the Montgomery potential
        mtg_s = theta_s * exn[i, j, kstop - 1] + g * in_hs[i, j, kstop - 1]
        inout_mtg[i, j, kstop - 2] = mtg_s + 0.5 * dz * exn[i, j, kstop - 1]
        for k in range(kstop - 3, kstart - 1, -1):
            inout_mtg[i, j, k] = inout_mtg[i, j, k + 1] + dz * exn[i, j, k + 1]

    @staticmethod
    def _stencil_montgomery_gt_defs(
        in_hs: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        inout_mtg: gtscript.Field["dtype"],
        *,
        dz: float,
        pt: float,
        theta_s: float
    ) -> None:
        from __externals__ import cp, g, pref, rd

        # retrieve the pressure
        with computation(FORWARD), interval(0, 1):
            inout_p = pt
        with computation(FORWARD), interval(1, None):
            inout_p = inout_p[0, 0, -1] + g * dz * in_s[0, 0, -1]

        # compute the Exner function
        with computation(PARALLEL), interval(...):
            out_exn = cp * (inout_p / pref) ** (rd / cp)

        # compute the Montgomery potential
        with computation(BACKWARD), interval(-2, -1):
            mtg_s = theta_s * out_exn[0, 0, 1] + g * in_hs[0, 0, 1]
            inout_mtg = mtg_s + 0.5 * dz * out_exn[0, 0, 1]
        with computation(BACKWARD), interval(0, -2):
            inout_mtg = inout_mtg[0, 0, 1] + dz * out_exn[0, 0, 1]

    def _stencil_height_numpy(
        self,
        in_theta: np.ndarray,
        in_hs: np.ndarray,
        in_s: np.ndarray,
        inout_h: np.ndarray,
        *,
        dz: float,
        pt: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        pref = self._pcs["air_pressure_at_sea_level"]
        rd = self._pcs["gas_constant_of_dry_air"]
        g = self._pcs["gravitational_acceleration"]
        cp = self._pcs["specific_heat_of_dry_air_at_constant_pressure"]

        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        kstart, kstop = origin[2], origin[2] + domain[2]

        # retrieve the pressure
        p = deepcopy(in_s)
        p[i, j, kstart] = pt
        for k in range(kstart + 1, kstop):
            p[i, j, k] = p[i, j, k - 1] + g * dz * in_s[i, j, k - 1]

        # compute the Exner function
        exn = cp * (p[i, j, kstart:kstop] / pref) ** (rd / cp)

        # compute the geometric height of the isentropes
        inout_h[i, j, kstop - 1] = in_hs[i, j, kstop - 1]
        for k in range(kstop - 2, kstart - 1, -1):
            inout_h[i, j, k] = inout_h[i, j, k + 1] - rd * (
                in_theta[i, j, k] * exn[i, j, k]
                + in_theta[i, j, k + 1] * exn[i, j, k + 1]
            ) * (p[i, j, k] - p[i, j, k + 1]) / (cp * g * (p[i, j, k] + p[i, j, k + 1]))

    @staticmethod
    def _stencil_height_gt_defs(
        in_theta: gtscript.Field["dtype"],
        in_hs: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        inout_h: gtscript.Field["dtype"],
        *,
        dz: float,
        pt: float
    ) -> None:
        from __externals__ import cp, g, rd, pref

        # retrieve the pressure
        with computation(FORWARD), interval(0, 1):
            inout_p = pt
        with computation(FORWARD), interval(1, None):
            inout_p = inout_p[0, 0, -1] + g * dz * in_s[0, 0, -1]

        # compute the Exner function
        with computation(PARALLEL), interval(...):
            out_exn = cp * (inout_p[0, 0, 0] / pref) ** (rd / cp)

        # compute the geometric height of the isentropes
        with computation(BACKWARD), interval(-1, None):
            inout_h = in_hs[0, 0, 0]
        with computation(BACKWARD), interval(0, -1):
            inout_h = inout_h[0, 0, 1] - rd * (
                in_theta[0, 0, 0] * out_exn[0, 0, 0]
                + in_theta[0, 0, 1] * out_exn[0, 0, 1]
            ) * (inout_p[0, 0, 0] - inout_p[0, 0, 1]) / (
                cp * g * (inout_p[0, 0, 0] + inout_p[0, 0, 1])
            )

    def _stencil_density_and_temperature_numpy(
        self,
        in_theta: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        in_exn: gtscript.Field["dtype"],
        in_h: gtscript.Field["dtype"],
        out_rho: gtscript.Field["dtype"],
        out_t: gtscript.Field["dtype"],
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        cp = self._pcs["specific_heat_of_dry_air_at_constant_pressure"]

        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])
        kp1 = slice(origin[2] + 1, origin[2] + domain[2] + 1)

        # compute the air density
        out_rho[i, j, k] = (
            in_s[i, j, k]
            * (in_theta[i, j, k] - in_theta[i, j, kp1])
            / (in_h[i, j, k] - in_h[i, j, kp1])
        )

        # compute the air temperature
        out_t[i, j, k] = (
            0.5
            / cp
            * (
                in_theta[i, j, k] * in_exn[i, j, k]
                + in_theta[i, j, kp1] * in_exn[i, j, kp1]
            )
        )

    @staticmethod
    def _stencil_density_and_temperature_gt_defs(
        in_theta: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        in_exn: gtscript.Field["dtype"],
        in_h: gtscript.Field["dtype"],
        out_rho: gtscript.Field["dtype"],
        out_t: gtscript.Field["dtype"],
    ) -> None:
        from __externals__ import cp

        with computation(PARALLEL), interval(...):
            # compute the air density
            out_rho = (
                in_s[0, 0, 0]
                * (in_theta[0, 0, 0] - in_theta[0, 0, 1])
                / (in_h[0, 0, 0] - in_h[0, 0, 1])
            )

            # compute the air temperature
            out_t = (
                0.5
                / cp
                * (
                    in_theta[0, 0, 0] * in_exn[0, 0, 0]
                    + in_theta[0, 0, 1] * in_exn[0, 0, 1]
                )
            )
