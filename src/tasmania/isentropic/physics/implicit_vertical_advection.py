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

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript
from sympl._core.time import Timer

from tasmania.framework.core_components import ImplicitTendencyComponent
from tasmania.framework.tag import stencil_definition

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Optional

    from tasmania.domain.domain import Domain
    from tasmania.framework.options import BackendOptions, StorageOptions
    from tasmania.utils.typingx import NDArray, NDArrayDict, PropertyDict, TimeDelta, TripletInt


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicImplicitVerticalAdvectionDiagnostic(ImplicitTendencyComponent):
    """
    Combine the Crank-Nicholson scheme with centered finite difference in space
    to integrated the vertical advection flux.
    """

    def __init__(
        self,
        domain: Domain,
        moist: bool = False,
        tendency_of_air_potential_temperature_on_interface_levels: bool = False,
        *,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional[BackendOptions] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional[StorageOptions] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        moist : `bool`, optional
            ``True`` if water species are included in the model,
            ``False`` otherwise. Defaults to ``False``.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            ``True`` if the input tendency of air potential temperature
            is defined at the interface levels, ``False`` otherwise.
            Defaults to ``False``.
        enable_checks : `bool`, optional
            TODO
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            class.
        """
        # keep track of the input arguments needed at run-time
        self._moist = moist
        self._stgz = tendency_of_air_potential_temperature_on_interface_levels

        # call parent's constructor
        super().__init__(
            domain,
            "numerical",
            enable_checks=enable_checks,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
            **kwargs,
        )

        # instantiate the underlying stencil object
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "moist": moist,
            "staggering": self._stgz,
            "setup": self.get_subroutine_definition("setup_thomas"),
            "setup_bc": self.get_subroutine_definition("setup_thomas_bc"),
        }
        self._stencil = self.compile_stencil("implicit_vertical_advection")

    @property
    def input_properties(self) -> PropertyDict:
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
            return_dict["tendency_of_air_potential_temperature"] = {"dims": dims, "units": "K s^-1"}

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def tendency_properties(self) -> PropertyDict:
        return {}

    @property
    def diagnostic_properties(self) -> PropertyDict:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }
        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    def array_call(
        self,
        state: NDArrayDict,
        timestep: TimeDelta,
        out_tendencies: NDArrayDict,
        out_diagnostics: NDArrayDict,
        overwrite_tendencies: dict[str, bool],
    ) -> None:
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        dz = self.grid.dz.to_units("K").values.item()

        # set the stencil's arguments
        stencil_args = {
            "gamma": timestep.total_seconds() / (4.0 * dz),
            "in_w": (
                state["tendency_of_air_potential_temperature_on_interface_levels"]
                if self._stgz
                else state["tendency_of_air_potential_temperature"]
            ),
            "in_s": state["air_isentropic_density"],
            "out_s": out_diagnostics["air_isentropic_density"],
            "in_su": state["x_momentum_isentropic"],
            "out_su": out_diagnostics["x_momentum_isentropic"],
            "in_sv": state["y_momentum_isentropic"],
            "out_sv": out_diagnostics["y_momentum_isentropic"],
        }
        if self._moist:
            stencil_args.update(
                {
                    "in_qv": state[mfwv],
                    "out_qv": out_diagnostics[mfwv],
                    "in_qc": state[mfcw],
                    "out_qc": out_diagnostics[mfcw],
                    "in_qr": state[mfpw],
                    "out_qr": out_diagnostics[mfpw],
                }
            )

        # run the stencil
        Timer.start(label="stencil")
        self._stencil(
            **stencil_args,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @stencil_definition(backend=("numpy", "cupy"), stencil="implicit_vertical_advection")
    def _stencil_numpy(
        self,
        in_w: NDArray,
        in_s: NDArray,
        in_su: NDArray,
        in_sv: NDArray,
        out_s: NDArray,
        out_su: NDArray,
        out_sv: NDArray,
        in_qv: Optional[NDArray] = None,
        in_qc: Optional[NDArray] = None,
        in_qr: Optional[NDArray] = None,
        out_qv: Optional[NDArray] = None,
        out_qc: Optional[NDArray] = None,
        out_qr: Optional[NDArray] = None,
        *,
        gamma: float,
        origin: TripletInt,
        domain: TripletInt,
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        kstart, kstop = origin[2], origin[2] + domain[2]

        # interpolate the velocity on the main levels
        if staggering:
            w = np.zeros_like(in_w)
            w[i, j, kstart:kstop] = 0.5 * (
                in_w[i, j, kstart:kstop] + in_w[i, j, kstart + 1 : kstop + 1]
            )
        else:
            w = in_w

        # compute the isentropic density of the water species
        if moist:
            sqv = self.zeros(shape=in_qv.shape)
            sqv[i, j, kstart:kstop] = in_s[i, j, kstart:kstop] * in_qv[i, j, kstart:kstop]
            sqc = self.zeros(shape=in_qc.shape)
            sqc[i, j, kstart:kstop] = in_s[i, j, kstart:kstop] * in_qc[i, j, kstart:kstop]
            sqr = self.zeros(shape=in_qr.shape)
            sqr[i, j, kstart:kstop] = in_s[i, j, kstart:kstop] * in_qr[i, j, kstart:kstop]
        else:
            sqv = sqc = sqr = None

        #
        # isentropic density
        #
        # set up the tridiagonal system
        a = self.zeros(shape=in_s.shape)
        b = self.ones(shape=in_s.shape)
        c = self.zeros(shape=in_s.shape)
        d = self.zeros(shape=in_s.shape)
        setup = self.get_subroutine_definition("setup_thomas")
        setup(gamma, w, in_s, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

        # solve the tridiagonal system
        thomas = self.get_subroutine_definition("thomas")
        thomas(a, b, c, d, out_s, i=i, j=j, kstart=kstart, kstop=kstop)

        #
        # x-momentum
        #
        # set up the tridiagonal system
        setup(gamma, w, in_su, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

        # solve the tridiagonal system
        thomas(a, b, c, d, out_su, i=i, j=j, kstart=kstart, kstop=kstop)

        #
        # y-momentum
        #
        # set up the tridiagonal system
        setup(gamma, w, in_sv, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

        # solve the tridiagonal system
        thomas(a, b, c, d, out_sv, i=i, j=j, kstart=kstart, kstop=kstop)

        if moist:
            #
            # isentropic density of water vapor
            #
            # set up the tridiagonal system
            setup(gamma, w, sqv, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

            # solve the tridiagonal system
            out_sqv = self.zeros(shape=sqv.shape)
            thomas(a, b, c, d, out_sqv, i=i, j=j, kstart=kstart, kstop=kstop)

            #
            # isentropic density of cloud liquid water
            #
            # set up the tridiagonal system
            setup(gamma, w, sqc, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

            # solve the tridiagonal system
            out_sqc = self.zeros(shape=sqc.shape)
            thomas(a, b, c, d, out_sqc, i=i, j=j, kstart=kstart, kstop=kstop)

            #
            # isentropic density of precipitation water
            #
            # set up the tridiagonal system
            setup(gamma, w, sqr, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

            # solve the tridiagonal system
            out_sqr = self.zeros(shape=sqr.shape)
            thomas(a, b, c, d, out_sqr, i=i, j=j, kstart=kstart, kstop=kstop)

            #
            # mass fraction of the water species
            #
            out_qv[i, j, kstart:kstop] = out_sqv[i, j, kstart:kstop] / out_s[i, j, kstart:kstop]
            out_qc[i, j, kstart:kstop] = out_sqc[i, j, kstart:kstop] / out_s[i, j, kstart:kstop]
            out_qr[i, j, kstart:kstop] = out_sqr[i, j, kstart:kstop] / out_s[i, j, kstart:kstop]

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="implicit_vertical_advection")
    def _stencil_gt4py(
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
        gamma: float,
    ) -> None:
        from __externals__ import moist, setup, setup_bc, staggering

        # interpolate the velocity on the main levels
        with computation(PARALLEL), interval(0, None):
            if __INLINED(staggering):  # compile-time if
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
            a_s, c_s, d_s = setup_bc(in_s)
        with computation(PARALLEL), interval(1, -1):
            a_s, c_s, d_s = setup(gamma, w, in_s)
        with computation(PARALLEL), interval(-1, None):
            a_s, c_s, d_s = setup_bc(in_s)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_s = 0.0
            beta_s = 1.0
            delta_s = d_s[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_s = a_s[0, 0, 0] / beta_s[0, 0, -1] if beta_s[0, 0, -1] != 0.0 else a_s[0, 0, 0]
            beta_s = 1.0 - omega_s[0, 0, 0] * c_s[0, 0, -1]
            delta_s = d_s[0, 0, 0] - omega_s[0, 0, 0] * delta_s[0, 0, -1]
        with computation(BACKWARD), interval(-1, None):
            out_s = (
                delta_s[0, 0, 0] / beta_s[0, 0, 0] if beta_s[0, 0, 0] != 0.0 else delta_s[0, 0, 0]
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
            a_su, c_su, d_su = setup_bc(in_su)
        with computation(PARALLEL), interval(1, -1):
            a_su, c_su, d_su = setup(gamma, w, in_su)
        with computation(PARALLEL), interval(-1, None):
            a_su, c_su, d_su = setup_bc(in_su)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_su = 0.0
            beta_su = 1.0
            delta_su = d_su[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_su = (
                a_su[0, 0, 0] / beta_su[0, 0, -1] if beta_su[0, 0, -1] != 0.0 else a_su[0, 0, 0]
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
            a_sv, c_sv, d_sv = setup_bc(in_sv)
        with computation(PARALLEL), interval(1, -1):
            a_sv, c_sv, d_sv = setup(gamma, w, in_sv)
        with computation(PARALLEL), interval(-1, None):
            a_sv, c_sv, d_sv = setup_bc(in_sv)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_sv = 0.0
            beta_sv = 1.0
            delta_sv = d_sv[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_sv = (
                a_sv[0, 0, 0] / beta_sv[0, 0, -1] if beta_sv[0, 0, -1] != 0.0 else a_sv[0, 0, 0]
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

        if moist:
            #
            # isentropic density of water vapor
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqv, c_sqv, d_sqv = setup_bc(sqv)
            with computation(PARALLEL), interval(1, -1):
                a_sqv, c_sqv, d_sqv = setup(gamma, w, sqv)
            with computation(PARALLEL), interval(-1, None):
                a_sqv, c_sqv, d_sqv = setup_bc(sqv)

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
                delta_sqv = d_sqv[0, 0, 0] - omega_sqv[0, 0, 0] * delta_sqv[0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqv = (
                    delta_sqv[0, 0, 0] / beta_sqv[0, 0, 0]
                    if beta_sqv[0, 0, 0] != 0.0
                    else delta_sqv[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqv = (
                    (delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1]) / beta_sqv[0, 0, 0]
                    if beta_sqv[0, 0, 0] != 0.0
                    else (delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1])
                )

            #
            # isentropic density of cloud liquid water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqc, c_sqc, d_sqc = setup_bc(sqc)
            with computation(PARALLEL), interval(1, -1):
                a_sqc, c_sqc, d_sqc = setup(gamma, w, sqc)
            with computation(PARALLEL), interval(-1, None):
                a_sqc, c_sqc, d_sqc = setup_bc(sqc)

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
                delta_sqc = d_sqc[0, 0, 0] - omega_sqc[0, 0, 0] * delta_sqc[0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqc = (
                    delta_sqc[0, 0, 0] / beta_sqc[0, 0, 0]
                    if beta_sqc[0, 0, 0] != 0.0
                    else delta_sqc[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqc = (
                    (delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1]) / beta_sqc[0, 0, 0]
                    if beta_sqc[0, 0, 0] != 0.0
                    else (delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1])
                )

            #
            # isentropic density of precipitation water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqr, c_sqr, d_sqr = setup_bc(sqr)
            with computation(PARALLEL), interval(1, -1):
                a_sqr, c_sqr, d_sqr = setup(gamma, w, sqr)
            with computation(PARALLEL), interval(-1, None):
                a_sqr, c_sqr, d_sqr = setup_bc(sqr)

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
                delta_sqr = d_sqr[0, 0, 0] - omega_sqr[0, 0, 0] * delta_sqr[0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqr = (
                    delta_sqr[0, 0, 0] / beta_sqr[0, 0, 0]
                    if beta_sqr[0, 0, 0] != 0.0
                    else delta_sqr[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqr = (
                    (delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1]) / beta_sqr[0, 0, 0]
                    if beta_sqr[0, 0, 0] != 0.0
                    else (delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1])
                )

        # calculate the output mass fraction of the water species
        if moist:
            with computation(PARALLEL), interval(...):
                out_qv = out_sqv / out_s
                out_qc = out_sqc / out_s
                out_qr = out_sqr / out_s


class IsentropicImplicitVerticalAdvectionPrognostic(ImplicitTendencyComponent):
    """
    Combine the Crank-Nicholson scheme with centered finite difference in space
    to integrated the vertical advection flux.
    """

    def __init__(
        self,
        domain: Domain,
        moist: bool = False,
        tendency_of_air_potential_temperature_on_interface_levels: bool = False,
        *,
        backend: str = "numpy",
        backend_options: Optional[BackendOptions] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional[StorageOptions] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        moist : `bool`, optional
            ``True`` if water species are included in the model,
            ``False`` otherwise. Defaults to ``False``.
        tendency_of_air_potential_temperature_on_interface_levels : `bool`, optional
            ``True`` if the input tendency of air potential temperature
            is defined at the interface levels, ``False`` otherwise.
            Defaults to ``False``.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        **kwargs :
            Additional keyword arguments to be directly forwarded to the parent
            class.
        """
        # keep track of the input arguments needed at run-time
        self._moist = moist
        self._stgz = tendency_of_air_potential_temperature_on_interface_levels

        # call parent's constructor
        super().__init__(
            domain,
            "numerical",
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
            **kwargs,
        )

        # set the storage shape
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        storage_shape = self.get_field_storage_shape(storage_shape, (nx, ny, nz + 1))

        # allocate the storages collecting the stencil outputs
        self._tnd_s = self.zeros(shape=storage_shape)
        self._tnd_su = self.zeros(shape=storage_shape)
        self._tnd_sv = self.zeros(shape=storage_shape)
        if moist:
            self._tnd_qv = self.zeros(shape=storage_shape)
            self._tnd_qc = self.zeros(shape=storage_shape)
            self._tnd_qr = self.zeros(shape=storage_shape)

        # instantiate the underlying stencil object
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "moist": moist,
            "vstaggering": self._stgz,
            "setup": self.get_subroutine_definition("setup_thomas"),
            "setup_bc": self.get_subroutine_definition("setup_thomas_bc"),
        }
        self._stencil = self.compile_stencil("stencil")

    @property
    def input_properties(self) -> PropertyDict:
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
            return_dict["tendency_of_air_potential_temperature"] = {"dims": dims, "units": "K s^-1"}

        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1"}

        return return_dict

    @property
    def tendency_properties(self) -> PropertyDict:
        grid = self.grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }
        if self._moist:
            return_dict[mfwv] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfcw] = {"dims": dims, "units": "g g^-1 s^-1"}
            return_dict[mfpw] = {"dims": dims, "units": "g g^-1 s^-1"}

        return return_dict

    @property
    def diagnostic_properties(self) -> PropertyDict:
        return {}

    def array_call(
        self, state: NDArrayDict, timestep: TimeDelta
    ) -> tuple[NDArrayDict, NDArrayDict]:
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
        Timer.start(label="stencil")
        self._stencil(
            **stencil_args,
            origin=(0, 0, 0),
            domain=(nx, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

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

    @stencil_definition(backend=("numpy", "cupy"), stencil="stencil")
    def _stencil_numpy(
        self,
        in_w: NDArray,
        in_s: NDArray,
        in_su: NDArray,
        in_sv: NDArray,
        tnd_s: NDArray,
        tnd_su: NDArray,
        tnd_sv: NDArray,
        in_qv: Optional[NDArray] = None,
        in_qc: Optional[NDArray] = None,
        in_qr: Optional[NDArray] = None,
        tnd_qv: Optional[NDArray] = None,
        tnd_qc: Optional[NDArray] = None,
        tnd_qr: Optional[NDArray] = None,
        *,
        dt: float,
        gamma: float,
        origin: TripletInt,
        domain: TripletInt,
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        kstart, kstop = origin[2], origin[2] + domain[2]

        # interpolate the velocity on the main levels
        if vstaggering:
            w = self.zeros(shape=in_w.shape)
            w[i, j, kstart:kstop] = 0.5 * (
                in_w[i, j, kstart:kstop] + in_w[i, j, kstart + 1 : kstop + 1]
            )
        else:
            w = in_w

        # compute the isentropic density of the water species
        if moist:
            sqv = self.zeros(shape=in_qv.shape)
            sqv[i, j, kstart:kstop] = in_s[i, j, kstart:kstop] * in_qv[i, j, kstart:kstop]
            sqc = self.zeros(shape=in_qc.shape)
            sqc[i, j, kstart:kstop] = in_s[i, j, kstart:kstop] * in_qc[i, j, kstart:kstop]
            sqr = self.zeros(shape=in_qr.shape)
            sqr[i, j, kstart:kstop] = in_s[i, j, kstart:kstop] * in_qr[i, j, kstart:kstop]
        else:
            sqv = sqc = sqr = None

        #
        # isentropic density
        #
        # set up the tridiagonal system
        a = self.zeros(shape=in_s.shape)
        b = self.ones(shape=in_s.shape)
        c = self.zeros(shape=in_s.shape)
        d = self.zeros(shape=in_s.shape)
        setup = self.get_subroutine_definition("setup_thomas")
        setup(gamma, w, in_s, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

        # solve the tridiagonal system
        thomas = self.get_subroutine_definition("thomas")
        out_s = self.zeros(shape=in_s.shape)
        thomas(a, b, c, d, out_s, i=i, j=j, kstart=kstart, kstop=kstop)

        #
        # x-momentum
        #
        # set up the tridiagonal system
        setup(gamma, w, in_su, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

        # solve the tridiagonal system
        out_su = self.zeros(shape=in_su.shape)
        thomas(a, b, c, d, out_su, i=i, j=j, kstart=kstart, kstop=kstop)

        #
        # y-momentum
        #
        # set up the tridiagonal system
        setup(gamma, w, in_sv, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

        # solve the tridiagonal system
        out_sv = self.zeros(shape=in_sv.shape)
        thomas(a, b, c, d, out_sv, i=i, j=j, kstart=kstart, kstop=kstop)

        if moist:
            #
            # isentropic density of water vapor
            #
            # set up the tridiagonal system
            setup(gamma, w, sqv, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

            # solve the tridiagonal system
            out_sqv = self.zeros(shape=sqv.shape)
            thomas(a, b, c, d, out_sqv, i=i, j=j, kstart=kstart, kstop=kstop)

            #
            # isentropic density of cloud liquid water
            #
            # set up the tridiagonal system
            setup(gamma, w, sqc, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

            # solve the tridiagonal system
            out_sqc = self.zeros(shape=sqc.shape)
            thomas(a, b, c, d, out_sqc, i=i, j=j, kstart=kstart, kstop=kstop)

            #
            # isentropic density of precipitation water
            #
            # set up the tridiagonal system
            setup(gamma, w, sqr, a, c, d, i=i, j=j, kstart=kstart, kstop=kstop)

            # solve the tridiagonal system
            out_sqr = self.zeros(shape=sqr.shape)
            thomas(a, b, c, d, out_sqr, i=i, j=j, kstart=kstart, kstop=kstop)

        # compute the tendencies
        tnd_s[i, j, kstart:kstop] = (out_s[i, j, kstart:kstop] - in_s[i, j, kstart:kstop]) / dt
        tnd_su[i, j, kstart:kstop] = (out_su[i, j, kstart:kstop] - in_su[i, j, kstart:kstop]) / dt
        tnd_sv[i, j, kstart:kstop] = (out_sv[i, j, kstart:kstop] - in_sv[i, j, kstart:kstop]) / dt
        if moist:
            tnd_qv[i, j, kstart:kstop] = (
                out_sqv[i, j, kstart:kstop] / out_s[i, j, kstart:kstop] - in_qv[i, j, kstart:kstop]
            ) / dt
            tnd_qc[i, j, kstart:kstop] = (
                out_sqc[i, j, kstart:kstop] / out_s[i, j, kstart:kstop] - in_qc[i, j, kstart:kstop]
            ) / dt
            tnd_qr[i, j, kstart:kstop] = (
                out_sqr[i, j, kstart:kstop] / out_s[i, j, kstart:kstop] - in_qr[i, j, kstart:kstop]
            ) / dt

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil")
    def _stencil_gt4py(
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
        gamma: float,
    ) -> None:
        from __externals__ import moist, setup, setup_bc, vstaggering

        # interpolate the velocity on the main levels
        with computation(PARALLEL), interval(0, None):
            if vstaggering:  # compile-time if
                w = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, 1])
            else:
                w = in_w

        # compute the isentropic density of the water species
        if moist:  # compile-time if
            with computation(PARALLEL), interval(0, None):
                sqv = in_s * in_qv
                sqc = in_s * in_qc
                sqr = in_s * in_qr

        #
        # isentropic density
        #
        # set up the tridiagonal system
        with computation(PARALLEL), interval(0, 1):
            a_s, c_s, d_s = setup_bc(in_s)
        with computation(PARALLEL), interval(1, -1):
            a_s, c_s, d_s = setup(gamma, w, in_s)
        with computation(PARALLEL), interval(-1, None):
            a_s, c_s, d_s = setup_bc(in_s)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_s = 0.0
            beta_s = 1.0
            delta_s = d_s[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_s = a_s[0, 0, 0] / beta_s[0, 0, -1] if beta_s[0, 0, -1] != 0.0 else a_s[0, 0, 0]
            beta_s = 1.0 - omega_s[0, 0, 0] * c_s[0, 0, -1]
            delta_s = d_s[0, 0, 0] - omega_s[0, 0, 0] * delta_s[0, 0, -1]
        with computation(BACKWARD), interval(-1, None):
            out_s = (
                delta_s[0, 0, 0] / beta_s[0, 0, 0] if beta_s[0, 0, 0] != 0.0 else delta_s[0, 0, 0]
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
            a_su, c_su, d_su = setup_bc(in_su)
        with computation(PARALLEL), interval(1, -1):
            a_su, c_su, d_su = setup(gamma, w, in_su)
        with computation(PARALLEL), interval(-1, None):
            a_su, c_su, d_su = setup_bc(in_su)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_su = 0.0
            beta_su = 1.0
            delta_su = d_su[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_su = (
                a_su[0, 0, 0] / beta_su[0, 0, -1] if beta_su[0, 0, -1] != 0.0 else a_su[0, 0, 0]
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
            a_sv, c_sv, d_sv = setup_bc(in_sv)
        with computation(PARALLEL), interval(1, -1):
            a_sv, c_sv, d_sv = setup(gamma, w, in_sv)
        with computation(PARALLEL), interval(-1, None):
            a_sv, c_sv, d_sv = setup_bc(in_sv)

        # solve the tridiagonal system
        with computation(FORWARD), interval(0, 1):
            omega_sv = 0.0
            beta_sv = 1.0
            delta_sv = d_sv[0, 0, 0]
        with computation(FORWARD), interval(1, None):
            omega_sv = (
                a_sv[0, 0, 0] / beta_sv[0, 0, -1] if beta_sv[0, 0, -1] != 0.0 else a_sv[0, 0, 0]
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

        if moist:
            #
            # isentropic density of water vapor
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqv, c_sqv, d_sqv = setup_bc(sqv)
            with computation(PARALLEL), interval(1, -1):
                a_sqv, c_sqv, d_sqv = setup(gamma, w, sqv)
            with computation(PARALLEL), interval(-1, None):
                a_sqv, c_sqv, d_sqv = setup_bc(sqv)

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
                delta_sqv = d_sqv[0, 0, 0] - omega_sqv[0, 0, 0] * delta_sqv[0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqv = (
                    delta_sqv[0, 0, 0] / beta_sqv[0, 0, 0]
                    if beta_sqv[0, 0, 0] != 0.0
                    else delta_sqv[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqv = (
                    (delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1]) / beta_sqv[0, 0, 0]
                    if beta_sqv[0, 0, 0] != 0.0
                    else (delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1])
                )

            #
            # isentropic density of cloud liquid water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqc, c_sqc, d_sqc = setup_bc(sqc)
            with computation(PARALLEL), interval(1, -1):
                a_sqc, c_sqc, d_sqc = setup(gamma, w, sqc)
            with computation(PARALLEL), interval(-1, None):
                a_sqc, c_sqc, d_sqc = setup_bc(sqc)

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
                delta_sqc = d_sqc[0, 0, 0] - omega_sqc[0, 0, 0] * delta_sqc[0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqc = (
                    delta_sqc[0, 0, 0] / beta_sqc[0, 0, 0]
                    if beta_sqc[0, 0, 0] != 0.0
                    else delta_sqc[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqc = (
                    (delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1]) / beta_sqc[0, 0, 0]
                    if beta_sqc[0, 0, 0] != 0.0
                    else (delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1])
                )

            #
            # isentropic density of precipitation water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqr, c_sqr, d_sqr = setup_bc(sqr)
            with computation(PARALLEL), interval(1, -1):
                a_sqr, c_sqr, d_sqr = setup(gamma, w, sqr)
            with computation(PARALLEL), interval(-1, None):
                a_sqr, c_sqr, d_sqr = setup_bc(sqr)

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
                delta_sqr = d_sqr[0, 0, 0] - omega_sqr[0, 0, 0] * delta_sqr[0, 0, -1]
            with computation(BACKWARD), interval(-1, None):
                out_sqr = (
                    delta_sqr[0, 0, 0] / beta_sqr[0, 0, 0]
                    if beta_sqr[0, 0, 0] != 0.0
                    else delta_sqr[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqr = (
                    (delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1]) / beta_sqr[0, 0, 0]
                    if beta_sqr[0, 0, 0] != 0.0
                    else (delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1])
                )

        # compute the tendencies
        with computation(PARALLEL), interval(...):
            tnd_s = (out_s - in_s) / dt
            tnd_su = (out_su - in_su) / dt
            tnd_sv = (out_sv - in_sv) / dt
            if moist:
                tnd_qv = (out_sqv / out_s - in_qv) / dt
                tnd_qc = (out_sqc / out_s - in_qc) / dt
                tnd_qr = (out_sqr / out_s - in_qr) / dt
