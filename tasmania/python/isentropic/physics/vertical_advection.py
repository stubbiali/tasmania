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
from typing import Optional, TYPE_CHECKING, Tuple

from gt4py import gtscript, __externals__

# from gt4py.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.framework.base_components import (
    ImplicitTendencyComponent,
    TendencyComponent,
)
from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)
from tasmania.python.utils import taz_types
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import zeros

if TYPE_CHECKING:
    from tasmania.python.grids.domain import Domain


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@gtscript.function
def first_order_boundary(
    dz: float, w: taz_types.gtfield_t, phi: taz_types.gtfield_t
) -> taz_types.gtfield_t:
    out = (w[0, 0, -1] * phi[0, 0, -1] - w[0, 0, 0] * phi[0, 0, 0]) / dz
    return out


@gtscript.function
def second_order_boundary(
    dz: float, w: taz_types.gtfield_t, phi: taz_types.gtfield_t
) -> taz_types.gtfield_t:
    out = (
        0.5
        * (
            -3.0 * w[0, 0, 0] * phi[0, 0, 0]
            + 4.0 * w[0, 0, -1] * phi[0, 0, -1]
            - w[0, 0, -2] * phi[0, 0, -2]
        )
        / dz
    )
    return out


class IsentropicVerticalAdvection(TendencyComponent):
    """
    This class inherits :class:`tasmania.TendencyComponent` to calculate
    the vertical derivative of the conservative vertical advection flux
    in isentropic coordinates for any prognostic variable included in
    the isentropic model. The class is always instantiated over the
    numerical grid of the underlying domain.
    """

    def __init__(
        self,
        domain: "Domain",
        flux_scheme: str = "upwind",
        moist: bool = False,
        tendency_of_air_potential_temperature_on_interface_levels: bool = False,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        flux_scheme : `str`, optional
            The numerical flux scheme to implement. Defaults to 'upwind'.
            See :class:`~tasmania.IsentropicMinimalVerticalFlux` for all
            available options.
        moist : `bool`, optional
            `True` if water species are included in the model,
            `False` otherwise. Defaults to `False`.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            `True` if the input tendency of air potential temperature
            is defined at the interface levels, `False` otherwise.
            Defaults to `False`.
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
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent class.
        """
        # keep track of the input arguments needed at run-time
        self._moist = moist
        self._stgz = tendency_of_air_potential_temperature_on_interface_levels
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, "numerical", **kwargs)

        # instantiate the object calculating the flux
        self._vflux = IsentropicMinimalVerticalFlux.factory(flux_scheme)

        # set the storage shape
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = (nx, ny, nz + 1) if storage_shape is None else storage_shape
        error_msg = "storage_shape must be larger or equal than {}.".format(
            (nx, ny, nz + 1)
        )
        assert storage_shape[0] >= nx, error_msg
        assert storage_shape[1] >= ny, error_msg
        assert storage_shape[2] >= nz + 1, error_msg

        # allocate the gt4py storages collecting the stencil outputs
        self._out_s = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._out_su = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._out_sv = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        if moist:
            self._out_qv = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
            self._out_qc = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
            self._out_qr = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )

        # instantiate the underlying stencil object
        externals = {
            "compute_boundary": first_order_boundary
            if self._vflux.order == 1
            else second_order_boundary,
            "moist": moist,
            "vflux": self._vflux.__call__,
            "vflux_end": -self._vflux.extent + 1 if self._vflux.extent > 1 else None,
            "vflux_extent": self._vflux.extent,
            "vstaggering": self._stgz,
        }
        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            backend=backend,
            build_info=build_info,
            dtypes={"dtype": dtype},
            externals=externals,
            rebuild=rebuild,
            **(backend_opts or {})
        )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }

        if self._stgz:
            dims_stgz = (
                grid.x.dims[0],
                grid.y.dims[0],
                grid.z_on_interface_levels.dims[0],
            )
            return_dict["tendency_of_air_potential_temperature_on_interface_levels"] = {
                "dims": dims_stgz,
                "units": "K s^-1",
            }
        else:
            return_dict["tendency_of_air_potential_temperature"] = {
                "dims": dims,
                "units": "K s^-1",
            }

        if self._moist:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_precipitation_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }

        return return_dict

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }
        if self._moist:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }
            return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }
            return_dict["mass_fraction_of_precipitation_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {}

    def array_call(
        self, state: taz_types.gtstorage_dict_t
    ) -> Tuple[taz_types.gtstorage_dict_t, taz_types.gtstorage_dict_t]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dz = self.grid.dz.to_units("K").values.item()
        nb = self._vflux.extent

        # grab the required model variables
        in_w = (
            state["tendency_of_air_potential_temperature_on_interface_levels"]
            if self._stgz
            else state["tendency_of_air_potential_temperature"]
        )
        in_s = state["air_isentropic_density"]
        in_su = state["x_momentum_isentropic"]
        in_sv = state["y_momentum_isentropic"]
        if self._moist:
            in_qv = state[mfwv]
            in_qc = state[mfcw]
            in_qr = state[mfpw]

        # set the stencil's arguments
        stencil_args = {
            "dz": dz,
            "in_w": in_w,
            "in_s": in_s,
            "out_s": self._out_s,
            "in_su": in_su,
            "out_su": self._out_su,
            "in_sv": in_sv,
            "out_sv": self._out_sv,
        }
        if self._moist:
            stencil_args.update(
                {
                    "in_qv": in_qv,
                    "out_qv": self._out_qv,
                    "in_qc": in_qc,
                    "out_qc": self._out_qc,
                    "in_qr": in_qr,
                    "out_qr": self._out_qr,
                }
            )

        # in_w[in_w > 0.1 * 0.025] = 0.1 * 0.025

        # run the stencil
        self._stencil(
            **stencil_args,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info
        )

        # collect the output arrays in a dictionary
        tendencies = {
            "air_isentropic_density": self._out_s,
            "x_momentum_isentropic": self._out_su,
            "y_momentum_isentropic": self._out_sv,
        }
        if self._moist:
            tendencies[mfwv] = self._out_qv
            tendencies[mfcw] = self._out_qc
            tendencies[mfpw] = self._out_qr

        return tendencies, {}

    @staticmethod
    def _stencil_defs(
        in_w: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        in_su: gtscript.Field["dtype"],
        in_sv: gtscript.Field["dtype"],
        out_s: gtscript.Field["dtype"],
        out_su: gtscript.Field["dtype"],
        out_sv: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"] = None,
        in_qc: gtscript.Field["dtype"] = None,
        in_qr: gtscript.Field["dtype"] = None,
        out_qv: gtscript.Field["dtype"] = None,
        out_qc: gtscript.Field["dtype"] = None,
        out_qr: gtscript.Field["dtype"] = None,
        *,
        dt: float = 0.0,
        dz: float
    ) -> None:
        from __externals__ import (
            compute_boundary,
            moist,
            vflux,
            vflux_end,
            vflux_extent,
            vstaggering,
        )

        # interpolate the velocity on the interface levels
        with computation(FORWARD), interval(0, 1):
            w = 0.0
        with computation(PARALLEL), interval(1, None):
            if __INLINED(vstaggering):  # compile-time if
                w = in_w
            else:
                w = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, -1])

        # interpolate the velocity on the main levels
        with computation(PARALLEL), interval(0, None):
            if __INLINED(vstaggering):
                wc = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, 1])
            else:
                wc = in_w

        # compute the isentropic density of the water species
        if __INLINED(moist):  # compile-time if
            with computation(PARALLEL), interval(0, None):
                sqv = in_s * in_qv
                sqc = in_s * in_qc
                sqr = in_s * in_qr

        # compute the fluxes
        with computation(PARALLEL), interval(vflux_extent, vflux_end):
            if __INLINED(not moist):  # compile-time if
                flux_s, flux_su, flux_sv = vflux(
                    dt=dt, dz=dz, w=w, s=in_s, su=in_su, sv=in_sv
                )
            else:
                flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr = vflux(
                    dt=dt,
                    dz=dz,
                    w=w,
                    s=in_s,
                    su=in_su,
                    sv=in_sv,
                    sqv=sqv,
                    sqc=sqc,
                    sqr=sqr,
                )

        # calculate the tendencies
        with computation(PARALLEL), interval(0, vflux_extent):
            out_s = 0.0
            out_su = 0.0
            out_sv = 0.0
            if __INLINED(moist):  # compile-time if
                out_qv = 0.0
                out_qc = 0.0
                out_qr = 0.0
        with computation(PARALLEL), interval(vflux_extent, -vflux_extent):
            out_s = (flux_s[0, 0, 1] - flux_s[0, 0, 0]) / dz
            out_su = (flux_su[0, 0, 1] - flux_su[0, 0, 0]) / dz
            out_sv = (flux_sv[0, 0, 1] - flux_sv[0, 0, 0]) / dz
            if __INLINED(moist):  # compile-time if
                out_qv = (flux_sqv[0, 0, 1] - flux_sqv[0, 0, 0]) / (in_s[0, 0, 0] * dz)
                out_qc = (flux_sqc[0, 0, 1] - flux_sqc[0, 0, 0]) / (in_s[0, 0, 0] * dz)
                out_qr = (flux_sqr[0, 0, 1] - flux_sqr[0, 0, 0]) / (in_s[0, 0, 0] * dz)
        with computation(PARALLEL), interval(-vflux_extent, None):
            out_s = 0.0  # compute_boundary(dz=dz, w=wc, phi=in_s)
            out_su = 0.0  # compute_boundary(dz=dz, w=wc, phi=in_su)
            out_sv = 0.0  # compute_boundary(dz=dz, w=wc, phi=in_sv)
            if __INLINED(moist):  # compile-time if
                tmp_qv = 0.0  # compute_boundary(dz=dz, w=wc, phi=sqv)
                out_qv = tmp_qv / in_s
                tmp_qc = 0.0  # compute_boundary(dz=dz, w=wc, phi=sqc)
                out_qc = tmp_qc / in_s
                tmp_qr = 0.0  # compute_boundary(dz=dz, w=wc, phi=sqr)
                out_qr = tmp_qr / in_s


@gtscript.function
def setup_tridiagonal_system(
    gamma: float, w: taz_types.gtfield_t, phi: taz_types.gtfield_t
) -> "Tuple[taz_types.gtfield_t, taz_types.gtfield_t, taz_types.gtfield_t]":
    a = gamma * w[0, 0, -1]
    c = -gamma * w[0, 0, 1]
    d = gamma * (w[0, 0, 1] * phi[0, 0, 1] - w[0, 0, -1] * phi[0, 0, -1])
    return a, c, d


@gtscript.function
def setup_tridiagonal_system_bc(
    phi: taz_types.gtfield_t
) -> "Tuple[taz_types.gtfield_t, taz_types.gtfield_t, taz_types.gtfield_t]":
    a = 0.0
    c = 0.0
    d = phi[0, 0, 0]
    return a, c, d


class IsentropicImplicitVerticalAdvection(ImplicitTendencyComponent):
    """
    Combine the Crank-Nicholson scheme with centered finite difference in space
    to integrated the vertical advection flux.
    """

    def __init__(
        self,
        domain: "Domain",
        moist: bool = False,
        tendency_of_air_potential_temperature_on_interface_levels: bool = False,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        moist : `bool`, optional
            `True` if water species are included in the model,
            `False` otherwise. Defaults to `False`.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            `True` if the input tendency of air potential temperature
            is defined at the interface levels, `False` otherwise.
            Defaults to `False`.
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
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent class.
        """
        # keep track of the input arguments needed at run-time
        self._moist = moist
        self._stgz = tendency_of_air_potential_temperature_on_interface_levels
        self._exec_info = exec_info

        # call parent's constructor
        super().__init__(domain, "numerical", **kwargs)

        # set the storage shape
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = (nx, ny, nz + 1) if storage_shape is None else storage_shape
        error_msg = "storage_shape must be larger or equal than {}.".format(
            (nx, ny, nz + 1)
        )
        assert storage_shape[0] >= nx, error_msg
        assert storage_shape[1] >= ny, error_msg
        assert storage_shape[2] >= nz + 1, error_msg

        # allocate the gt4py storages collecting the stencil outputs
        self._tnd_s = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._tnd_su = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._tnd_sv = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        if moist:
            self._tnd_qv = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
            self._tnd_qc = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
            self._tnd_qr = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )

        # instantiate the underlying stencil object
        externals = {
            "moist": moist,
            "vstaggering": self._stgz,
            "setup_tridiagonal_system": setup_tridiagonal_system,
            "setup_tridiagonal_system_bc": setup_tridiagonal_system_bc,
        }
        self._stencil = gtscript.stencil(
            definition=self._stencil_defs,
            backend=backend,
            build_info=build_info,
            dtypes={"dtype": dtype},
            externals=externals,
            rebuild=rebuild,
            **(backend_opts or {})
        )

    @property
    def input_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }

        if self._stgz:
            dims_stgz = (
                grid.x.dims[0],
                grid.y.dims[0],
                grid.z_on_interface_levels.dims[0],
            )
            return_dict["tendency_of_air_potential_temperature_on_interface_levels"] = {
                "dims": dims_stgz,
                "units": "K s^-1",
            }
        else:
            return_dict["tendency_of_air_potential_temperature"] = {
                "dims": dims,
                "units": "K s^-1",
            }

        if self._moist:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }
            return_dict["mass_fraction_of_precipitation_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1",
            }

        return return_dict

    @property
    def tendency_properties(self) -> taz_types.properties_dict_t:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }
        if self._moist:
            return_dict["mass_fraction_of_water_vapor_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }
            return_dict["mass_fraction_of_cloud_liquid_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }
            return_dict["mass_fraction_of_precipitation_water_in_air"] = {
                "dims": dims,
                "units": "g g^-1 s^-1",
            }

        return return_dict

    @property
    def diagnostic_properties(self) -> taz_types.properties_dict_t:
        return {}

    def array_call(
        self, state: taz_types.gtstorage_dict_t, timestep: taz_types.timedelta_t
    ) -> Tuple[taz_types.gtstorage_dict_t, taz_types.gtstorage_dict_t]:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dz = self.grid.dz.to_units("K").values.item()

        # grab the required model variables
        in_w = (
            state["tendency_of_air_potential_temperature_on_interface_levels"]
            if self._stgz
            else state["tendency_of_air_potential_temperature"]
        )
        in_s = state["air_isentropic_density"]
        in_su = state["x_momentum_isentropic"]
        in_sv = state["y_momentum_isentropic"]
        if self._moist:
            in_qv = state[mfwv]
            in_qc = state[mfcw]
            in_qr = state[mfpw]

        # set the stencil's arguments
        stencil_args = {
            "dt": timestep.total_seconds(),
            "gamma": timestep.total_seconds() / (4.0 * dz),
            "in_w": in_w,
            "in_s": in_s,
            "tnd_s": self._tnd_s,
            "in_su": in_su,
            "tnd_su": self._tnd_su,
            "in_sv": in_sv,
            "tnd_sv": self._tnd_sv,
        }
        if self._moist:
            stencil_args.update(
                {
                    "in_qv": in_qv,
                    "tnd_qv": self._tnd_qv,
                    "in_qc": in_qc,
                    "tnd_qc": self._tnd_qc,
                    "in_qr": in_qr,
                    "tnd_qr": self._tnd_qr,
                }
            )

        # run the stencil
        self._stencil(
            **stencil_args,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info
        )

        # collect the output arrays in a dictionary
        tendencies = {
            "air_isentropic_density": self._tnd_s,
            "x_momentum_isentropic": self._tnd_su,
            "y_momentum_isentropic": self._tnd_sv,
        }
        if self._moist:
            tendencies[mfwv] = self._tnd_qv
            tendencies[mfcw] = self._tnd_qc
            tendencies[mfpw] = self._tnd_qr

        return tendencies, {}

    @staticmethod
    def _stencil_defs(
        in_w: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        in_su: gtscript.Field["dtype"],
        in_sv: gtscript.Field["dtype"],
        tnd_s: gtscript.Field["dtype"],
        tnd_su: gtscript.Field["dtype"],
        tnd_sv: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"] = None,
        in_qc: gtscript.Field["dtype"] = None,
        in_qr: gtscript.Field["dtype"] = None,
        tnd_qv: gtscript.Field["dtype"] = None,
        tnd_qc: gtscript.Field["dtype"] = None,
        tnd_qr: gtscript.Field["dtype"] = None,
        *,
        dt: float,
        gamma: float
    ) -> None:
        from __externals__ import (
            moist,
            setup_tridiagonal_system,
            setup_tridiagonal_system_bc,
            vstaggering,
        )

        # interpolate the velocity on the main levels
        with computation(PARALLEL), interval(0, None):
            if __INLINED(vstaggering):  # compile-time if
                w = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, 1])
            else:
                w = in_w

        # compute the isentropic density of the water species
        if __INLINED(moist):  # compile-time if
            with computation(PARALLEL), interval(0, None):
                sqv = in_s * in_qv
                sqc = in_s * in_qc
                sqr = in_s * in_qr

        #
        # isentropic density
        #
        # set up the tridiagonal system
        with computation(PARALLEL), interval(0, 1):
            a_s, c_s, d_s = setup_tridiagonal_system_bc(in_s)
        with computation(PARALLEL), interval(1, -1):
            a_s, c_s, d_s = setup_tridiagonal_system(gamma, w, in_s)
        with computation(PARALLEL), interval(-1, None):
            a_s, c_s, d_s = setup_tridiagonal_system_bc(in_s)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_s = 0.0
            beta_s = 1.0
            delta_s = d_s[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_s = (
                a_s[0, 0, 0] / beta_s[0, 0, -1]
                if beta_s[0, 0, -1] != 0.0
                else a_s[0, 0, 0]
            )
            beta_s = 1.0 - omega_s[0, 0, 0] * c_s[0, 0, -1]
            delta_s = d_s[0, 0, 0] - omega_s[0, 0, 0] * delta_s[0, 0, -1]
        with computation(BACKWARD), interval(-1, None):
            out_s = (
                delta_s[0, 0, 0] / beta_s[0, 0, 0]
                if beta_s[0, 0, 0] != 0.0
                else delta_s[0, 0, 0]
            )
        with computation(BACKWARD), interval(0, -1):
            out_s = (
                (delta_s[0, 0, 0] - c_s[0, 0, 0] * out_s[0, 0, 1]) / beta_s[0, 0, 0]
                if beta_s[0, 0, 0] != 0.0
                else (delta_s[0, 0, 0] - c_s[0, 0, 0] * out_s[0, 0, 1])
            )

        #
        # x-momentum
        #
        # set up the tridiagonal system
        with computation(PARALLEL), interval(0, 1):
            a_su, c_su, d_su = setup_tridiagonal_system_bc(in_su)
        with computation(PARALLEL), interval(1, -1):
            a_su, c_su, d_su = setup_tridiagonal_system(gamma, w, in_su)
        with computation(PARALLEL), interval(-1, None):
            a_su, c_su, d_su = setup_tridiagonal_system_bc(in_su)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_su = 0.0
            beta_su = 1.0
            delta_su = d_su[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_su = (
                a_su[0, 0, 0] / beta_su[0, 0, -1]
                if beta_su[0, 0, -1] != 0.0
                else a_su[0, 0, 0]
            )
            beta_su = 1.0 - omega_su[0, 0, 0] * c_su[0, 0, -1]
            delta_su = d_su[0, 0, 0] - omega_su[0, 0, 0] * delta_su[0, 0, -1]
        with computation(BACKWARD), interval(-1, None):
            out_su = (
                delta_su[0, 0, 0] / beta_su[0, 0, 0]
                if beta_su[0, 0, 0] != 0.0
                else delta_su[0, 0, 0]
            )
        with computation(BACKWARD), interval(0, -1):
            out_su = (
                (delta_su[0, 0, 0] - c_su[0, 0, 0] * out_su[0, 0, 1]) / beta_su[0, 0, 0]
                if beta_su[0, 0, 0] != 0.0
                else (delta_su[0, 0, 0] - c_su[0, 0, 0] * out_su[0, 0, 1])
            )

        #
        # y-momentum
        #
        # set up the tridiagonal system
        with computation(PARALLEL), interval(0, 1):
            a_sv, c_sv, d_sv = setup_tridiagonal_system_bc(in_sv)
        with computation(PARALLEL), interval(1, -1):
            a_sv, c_sv, d_sv = setup_tridiagonal_system(gamma, w, in_sv)
        with computation(PARALLEL), interval(-1, None):
            a_sv, c_sv, d_sv = setup_tridiagonal_system_bc(in_sv)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_sv = 0.0
            beta_sv = 1.0
            delta_sv = d_sv[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_sv = (
                a_sv[0, 0, 0] / beta_sv[0, 0, -1]
                if beta_sv[0, 0, -1] != 0.0
                else a_sv[0, 0, 0]
            )
            beta_sv = 1.0 - omega_sv[0, 0, 0] * c_sv[0, 0, -1]
            delta_sv = d_sv[0, 0, 0] - omega_sv[0, 0, 0] * delta_sv[0, 0, -1]
        with computation(BACKWARD), interval(-1, None):
            out_sv = (
                delta_sv[0, 0, 0] / beta_sv[0, 0, 0]
                if beta_sv[0, 0, 0] != 0.0
                else delta_sv[0, 0, 0]
            )
        with computation(BACKWARD), interval(0, -1):
            out_sv = (
                (delta_sv[0, 0, 0] - c_sv[0, 0, 0] * out_sv[0, 0, 1]) / beta_sv[0, 0, 0]
                if beta_sv[0, 0, 0] != 0.0
                else (delta_sv[0, 0, 0] - c_sv[0, 0, 0] * out_sv[0, 0, 1])
            )

        if __INLINED(moist):
            #
            # isentropic density of water vapor
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqv, c_sqv, d_sqv = setup_tridiagonal_system_bc(sqv)
            with computation(PARALLEL), interval(1, -1):
                a_sqv, c_sqv, d_sqv = setup_tridiagonal_system(gamma, w, sqv)
            with computation(PARALLEL), interval(-1, None):
                a_sqv, c_sqv, d_sqv = setup_tridiagonal_system_bc(sqv)

            # solve the tridiagonal system
            with computation(FORWARD), interval(0, 1):
                omega_sqv = 0.0
                beta_sqv = 1.0
                delta_sqv = d_sqv[0, 0, 0]
            with computation(FORWARD), interval(1, None):
                omega_sqv = (
                    a_sqv[0, 0, 0] / beta_sqv[0, 0, -1]
                    if beta_sqv[0, 0, -1] != 0.0
                    else a_sqv[0, 0, 0]
                )
                beta_sqv = 1.0 - omega_sqv[0, 0, 0] * c_sqv[0, 0, -1]
                delta_sqv = d_sqv[0, 0, 0] - omega_sqv[0, 0, 0] * delta_sqv[
                    0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqv = (
                    delta_sqv[0, 0, 0] / beta_sqv[0, 0, 0]
                    if beta_sqv[0, 0, 0] != 0.0
                    else delta_sqv[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqv = (
                    (delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1]) /
                    beta_sqv[0, 0, 0]
                    if beta_sqv[0, 0, 0] != 0.0
                    else (delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1])
                )

            #
            # isentropic density of cloud liquid water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqc, c_sqc, d_sqc = setup_tridiagonal_system_bc(sqc)
            with computation(PARALLEL), interval(1, -1):
                a_sqc, c_sqc, d_sqc = setup_tridiagonal_system(gamma, w, sqc)
            with computation(PARALLEL), interval(-1, None):
                a_sqc, c_sqc, d_sqc = setup_tridiagonal_system_bc(sqc)

            # solve the tridiagonal system
            with computation(FORWARD), interval(0, 1):
                omega_sqc = 0.0
                beta_sqc = 1.0
                delta_sqc = d_sqc[0, 0, 0]
            with computation(FORWARD), interval(1, None):
                omega_sqc = (
                    a_sqc[0, 0, 0] / beta_sqc[0, 0, -1]
                    if beta_sqc[0, 0, -1] != 0.0
                    else a_sqc[0, 0, 0]
                )
                beta_sqc = 1.0 - omega_sqc[0, 0, 0] * c_sqc[0, 0, -1]
                delta_sqc = d_sqc[0, 0, 0] - omega_sqc[0, 0, 0] * delta_sqc[
                    0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqc = (
                    delta_sqc[0, 0, 0] / beta_sqc[0, 0, 0]
                    if beta_sqc[0, 0, 0] != 0.0
                    else delta_sqc[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqc = (
                    (delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1]) /
                    beta_sqc[0, 0, 0]
                    if beta_sqc[0, 0, 0] != 0.0
                    else (delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1])
                )

            #
            # isentropic density of precipitation water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqr, c_sqr, d_sqr = setup_tridiagonal_system_bc(sqr)
            with computation(PARALLEL), interval(1, -1):
                a_sqr, c_sqr, d_sqr = setup_tridiagonal_system(gamma, w, sqr)
            with computation(PARALLEL), interval(-1, None):
                a_sqr, c_sqr, d_sqr = setup_tridiagonal_system_bc(sqr)

            # solve the tridiagonal system
            with computation(FORWARD), interval(0, 1):
                omega_sqr = 0.0
                beta_sqr = 1.0
                delta_sqr = d_sqr[0, 0, 0]
            with computation(FORWARD), interval(1, None):
                omega_sqr = (
                    a_sqr[0, 0, 0] / beta_sqr[0, 0, -1]
                    if beta_sqr[0, 0, -1] != 0.0
                    else a_sqr[0, 0, 0]
                )
                beta_sqr = 1.0 - omega_sqr[0, 0, 0] * c_sqr[0, 0, -1]
                delta_sqr = d_sqr[0, 0, 0] - omega_sqr[0, 0, 0] * delta_sqr[
                    0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqr = (
                    delta_sqr[0, 0, 0] / beta_sqr[0, 0, 0]
                    if beta_sqr[0, 0, 0] != 0.0
                    else delta_sqr[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqr = (
                    (delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1]) /
                    beta_sqr[0, 0, 0]
                    if beta_sqr[0, 0, 0] != 0.0
                    else (delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1])
                )


        # compute the tendencies
        with computation(PARALLEL), interval(...):
            tnd_s = (out_s - in_s) / dt
            tnd_su = (out_su - in_su) / dt
            tnd_sv = (out_sv - in_sv) / dt
            if __INLINED(moist):
                tnd_qv = (out_sqv / out_s - sqv) / dt
                tnd_qc = (out_sqc / out_s - sqc) / dt
                tnd_qr = (out_sqr / out_s - sqr) / dt


class PrescribedSurfaceHeating(TendencyComponent):
    """
    Calculate the variation in air potential temperature as prescribed
    in the reference paper, namely

        .. math::
            \dot{\theta} =
            \Biggl \lbrace
            {
                \\frac{\theta \, R_d \, \alpha(t)}{p \, C_p}
                \exp[\left( - \alpha(t) \left( z - h_s \\right) \\right]}
                \left[ F_0^{sw}(t) \sin{\left( \omega^{sw} (t - t_0) \\right)}
                + F_0^{fw}(t) \sin{\left( \omega^{fw} (t - t_0) \\right)} \\right]
                \text{if} {r = \sqrt{x^2 + y^2} < R}
                \atop
                0 \text{otherwise}
            } .

    The class is always instantiated over the numerical grid of the
    underlying domain.

    References
    ----------
    Reisner, J. M., and P. K. Smolarkiewicz. (1994). \
        Thermally forced low Froude number flow past three-dimensional obstacles. \
        *Journal of Atmospheric Sciences*, *51*(1):117-133.
    """

    # Default values for the physical constants used in the class
    _d_physical_constants = {
        "gas_constant_of_dry_air": DataArray(287.0, attrs={"units": "J K^-1 kg^-1"}),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }

    def __init__(
        self,
        domain,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        tendency_of_air_potential_temperature_on_interface_levels=False,
        air_pressure_on_interface_levels=True,
        amplitude_at_day_sw=None,
        amplitude_at_day_fw=None,
        amplitude_at_night_sw=None,
        amplitude_at_night_fw=None,
        frequency_sw=None,
        frequency_fw=None,
        attenuation_coefficient_at_day=None,
        attenuation_coefficient_at_night=None,
        characteristic_length=None,
        starting_time=None,
        backend="numpy",
        physical_constants=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
            `True` to place the calculated tendency of air
            potential temperature in the ``diagnostics`` output
            dictionary, `False` to regularly place it in the
            `tendencies` dictionary. Defaults to `False`.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            `True` (respectively, `False`) if the tendency
            of air potential temperature should be calculated at the
            interface (resp., main) vertical levels. Defaults to `False`.
        air_pressure_on_interface_levels : `bool`, optional
            `True` (respectively, `False`) if the input
            air potential pressure is defined at the interface
            (resp., main) vertical levels. Defaults to `True`.
        amplitude_at_day_sw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`F_0^{sw}` at day,
            in units compatible with [W m^-2].
        amplitude_at_day_fw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`F_0^{fw}` at day,
            in units compatible with [W m^-2].
        amplitude_at_night_sw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`F_0^{sw}` at night,
            in units compatible with [W m^-2].
        amplitude_at_night_fw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`F_0^{fw}` at night,
            in units compatible with [W m^-2].
        frequency_sw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`\omega^{sw}`,
            in units compatible with [s^-1].
        frequency_fw : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`\omega^{fw}`,
            in units compatible with [s^-1].
        attenuation_coefficient_at_day : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`\alpha` at day,
            in units compatible with [m^-1].
        attenuation_coefficient_at_night : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`\alpha` at night,
            in units compatible with [m^-1].
        characteristic_length : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing :math:`R`,
            in units compatible with [m].
        starting_time : `datetime`, optional
            The time :math:`t_0` when surface heating/cooling is triggered.
        backend : `obj`, optional
            TODO
        physical_constants : `dict[str, sympl.DataArray]`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`~sympl.DataArray`\s
            storing the values and units of those constants. The constants might be:

                * 'gas_constant_of_dry_air', in units compatible with \
                    [J K^-1 kg^-1];
                * 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
                    with [J K^-1 kg^-1].

        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            :class:`sympl.TendencyComponent`.
        """
        self._tid = tendency_of_air_potential_temperature_in_diagnostics
        self._apil = air_pressure_on_interface_levels
        self._aptil = (
            tendency_of_air_potential_temperature_on_interface_levels
            and air_pressure_on_interface_levels
        )
        self._backend = backend

        super().__init__(domain, "numerical", **kwargs)

        self._f0d_sw = (
            amplitude_at_day_sw.to_units("W m^-2").values.item()
            if amplitude_at_day_sw is not None
            else 800.0
        )
        self._f0d_fw = (
            amplitude_at_day_fw.to_units("W m^-2").values.item()
            if amplitude_at_day_fw is not None
            else 400.0
        )
        self._f0n_sw = (
            amplitude_at_night_sw.to_units("W m^-2").values.item()
            if amplitude_at_night_sw is not None
            else -75.0
        )
        self._f0n_fw = (
            amplitude_at_night_fw.to_units("W m^-2").values.item()
            if amplitude_at_night_fw is not None
            else -37.5
        )
        self._w_sw = (
            frequency_sw.to_units("hr^-1").values.item()
            if frequency_sw is not None
            else np.pi / 12.0
        )
        self._w_fw = (
            frequency_fw.to_units("hr^-1").values.item()
            if frequency_fw is not None
            else np.pi
        )
        self._ad = (
            attenuation_coefficient_at_day.to_units("m^-1").values.item()
            if attenuation_coefficient_at_day is not None
            else 1.0 / 600.0
        )
        self._an = (
            attenuation_coefficient_at_night.to_units("m^-1").values.item()
            if attenuation_coefficient_at_night is not None
            else 1.0 / 75.0
        )
        self._cl = (
            characteristic_length.to_units("m").values.item()
            if characteristic_length is not None
            else 25000.0
        )
        self._t0 = starting_time

        pcs = get_physical_constants(self._d_physical_constants, physical_constants)
        self._rd = pcs["gas_constant_of_dry_air"]
        self._cp = pcs["specific_heat_of_dry_air_at_constant_pressure"]

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stgz = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return_dict = {"height_on_interface_levels": {"dims": dims_stgz, "units": "m"}}

        if self._apil:
            return_dict["air_pressure_on_interface_levels"] = {
                "dims": dims_stgz,
                "units": "Pa",
            }
        else:
            return_dict["air_pressure"] = {"dims": dims, "units": "Pa"}

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid

        return_dict = {}

        if not self._tid:
            if self._aptil:
                dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
                return_dict["air_potential_temperature_on_interface_levels"] = {
                    "dims": dims,
                    "units": "K s^-1",
                }
            else:
                dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
                return_dict["air_potential_temperature"] = {
                    "dims": dims,
                    "units": "K s^-1",
                }

        return return_dict

    @property
    def diagnostic_properties(self):
        g = self.grid

        return_dict = {}

        if self._tid:
            if self._aptil:
                dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
                return_dict[
                    "tendency_of_air_potential_temperature_on_interface_levels"
                ] = {"dims": dims, "units": "K s^-1"}
            else:
                dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
                return_dict["tendency_of_air_potential_temperature"] = {
                    "dims": dims,
                    "units": "K s^-1",
                }

        return return_dict

    def array_call(self, state):
        g = self.grid
        mi, mj = g.nx, g.ny
        mk = g.nz + 1 if self._aptil else g.nz

        t = state["time"]
        dt = (t - self._t0).total_seconds() / 3600.0 if self._t0 is not None else t.hour

        if dt <= 0.0:
            out = np.zeros((mi, mj, mk), dtype=state["height_on_interface_levels"].dtype)
        else:
            x = g.x.to_units("m").values[:, np.newaxis, np.newaxis]
            y = g.y.to_units("m").values[np.newaxis, :, np.newaxis]
            theta1d = (
                g.z_on_interface_levels.to_units("K").values
                if self._aptil
                else g.z.to_units("K").values
            )
            theta = theta1d[np.newaxis, np.newaxis, :]

            pv = (
                state["air_pressure_on_interface_levels"]
                if self._apil
                else state["air_pressure"]
            )
            p = pv if pv.shape[2] == mk else 0.5 * (pv[:, :, 1:] + pv[:, :, :-1])
            zv = state["height_on_interface_levels"]
            z = zv if self._aptil else 0.5 * (zv[:, :, 1:] + zv[:, :, :-1])
            h = zv[:, :, -1:]

            w_sw = self._w_sw
            w_fw = self._w_fw
            cl = self._cl

            t_in_seconds = t.hour * 60 * 60 + t.minute * 60 + t.second
            t_sw = (2 * np.pi / w_sw) * 60 * 60
            day = int(t_in_seconds / t_sw) % 2 == 0
            f0_sw = self._f0d_sw if day else self._f0n_sw
            f0_fw = self._f0d_fw if day else self._f0n_fw
            a = self._ad if day else self._an

            out = (
                theta
                * self._rd
                * a
                / (p * self._cp)
                * np.exp(-a * (z - h))
                * (f0_sw * np.sin(w_sw * dt) + f0_fw * np.sin(w_fw * dt))
                * (x ** 2 + y ** 2 < cl ** 2)
            )

        tendencies = {}
        if not self._tid:
            if self._aptil:
                tendencies["air_potential_temperature_on_interface_levels"] = out
            else:
                tendencies["air_potential_temperature"] = out

        diagnostics = {}
        if self._tid:
            if self._aptil:
                diagnostics[
                    "tendency_of_air_potential_temperature_on_interface_levels"
                ] = out
            else:
                diagnostics["tendency_of_air_potential_temperature"] = out

        return tendencies, diagnostics
