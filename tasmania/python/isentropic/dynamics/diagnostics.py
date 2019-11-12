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

from gt4py import gtscript, __externals__

# from gt4py.__gtscript__ import computation, interval, PARALLEL, FORWARD, BACKWARD

from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import zeros

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


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
        grid,
        physical_constants=None,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None,
        managed_memory=False
    ):
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
            backend,
            dtype,
            default_origin=default_origin,
            mask=[True, True, True],
            managed_memory=managed_memory,
        )
        self._theta[:nx, :ny, : nz + 1] = grid.z_on_interface_levels.to_units("K").values[
            np.newaxis, np.newaxis, :
        ]
        self._topo = zeros(
            storage_shape,
            backend,
            dtype,
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

        # instantiate the underlying gt4py stencils
        self._stencil_diagnostic_variables = gtscript.stencil(
            definition=self._stencil_diagnostic_variables_defs,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            externals=externals,
            **(backend_opts or {})
        )
        self._stencil_density_and_temperature = gtscript.stencil(
            definition=self._stencil_density_and_temperature_defs,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            externals=externals,
            **(backend_opts or {})
        )
        self._stencil_montgomery = gtscript.stencil(
            definition=self._stencil_montgomery_defs,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            externals=externals,
            **(backend_opts or {})
        )
        self._stencil_height = gtscript.stencil(
            definition=self._stencil_height_defs,
            backend=backend,
            build_info=build_info,
            rebuild=rebuild,
            externals=externals,
            **(backend_opts or {})
        )

    def get_diagnostic_variables(self, s, pt, p, exn, mtg, h):
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
        self._topo[:nx, :ny, -1] = self._grid.topography.profile.to_units("m").values[...]

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
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz + 1),
            exec_info=self._exec_info,
        )

    def get_montgomery_potential(self, s, pt, mtg):
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
        self._topo[:nx, :ny, -1] = self._grid.topography.profile.to_units("m").values[...]

        # run the stencil
        self._stencil_montgomery(
            in_hs=self._topo,
            in_s=s,
            inout_mtg=mtg,
            dz=dz,
            pt=pt,
            theta_s=theta_s,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz + 1),
            exec_info=self._exec_info,
        )

    def get_height(self, s, pt, h):
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
        self._topo[:nx, :ny, -1] = self._grid.topography.profile.to_units("m").values[...]

        # run the stencil
        self._stencil_height(
            in_theta=self._theta,
            in_hs=self._topo,
            in_s=s,
            inout_h=h,
            dz=dz,
            pt=pt,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz + 1),
            exec_info=self._exec_info,
        )

    def get_density_and_temperature(self, s, exn, h, rho, t):
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
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_diagnostic_variables_defs(
        in_theta: gtscript.Field[np.float64],
        in_hs: gtscript.Field[np.float64],
        in_s: gtscript.Field[np.float64],
        inout_p: gtscript.Field[np.float64],
        out_exn: gtscript.Field[np.float64],
        inout_mtg: gtscript.Field[np.float64],
        inout_h: gtscript.Field[np.float64],
        *,
        dz: float,
        pt: float
    ):
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

    @staticmethod
    def _stencil_montgomery_defs(
        in_hs: gtscript.Field[np.float64],
        in_s: gtscript.Field[np.float64],
        inout_mtg: gtscript.Field[np.float64],
        *,
        dz: float,
        pt: float,
        theta_s: float
    ):
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
            mtg_s = theta_s * out_exn[0, 0, 1] + g * in_hs[0, 0, 1]
            inout_mtg = mtg_s + 0.5 * dz * out_exn[0, 0, 1]
        with computation(BACKWARD), interval(0, -2):
            inout_mtg = inout_mtg[0, 0, 1] + dz * out_exn[0, 0, 1]

    @staticmethod
    def _stencil_height_defs(
        in_theta: gtscript.Field[np.float64],
        in_hs: gtscript.Field[np.float64],
        in_s: gtscript.Field[np.float64],
        inout_h: gtscript.Field[np.float64],
        *,
        dz: float,
        pt: float
    ):
        from __externals__ import cp, g, rd, pref

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

    @staticmethod
    def _stencil_density_and_temperature_defs(
        in_theta: gtscript.Field[np.float64],
        in_s: gtscript.Field[np.float64],
        in_exn: gtscript.Field[np.float64],
        in_h: gtscript.Field[np.float64],
        out_rho: gtscript.Field[np.float64],
        out_t: gtscript.Field[np.float64],
    ):
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
