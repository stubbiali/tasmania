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
from typing import Optional, Tuple, Union

from gt4py import gtscript

from tasmania.python.framework.register import register
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper
from tasmania.python.framework.subclasses.stencil_definitions.cla import (
    thomas_numpy,
)
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
)
from tasmania.python.utils import typing


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def setup_tridiagonal_system_numpy(
    gamma: float,
    w: np.ndarray,
    phi: np.ndarray,
    phi_prv: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    i: Union[int, slice],
    j: Union[int, slice],
    kstart: int,
    kstop: int
) -> None:
    a[i, j, kstart + 1 : kstop - 1] = gamma * w[i, j, kstart : kstop - 2]
    a[i, j, kstop - 1] = 0.0

    c[i, j, kstart] = 0.0
    c[i, j, kstart + 1 : kstop - 1] = -gamma * w[i, j, kstart + 2 : kstop]

    d[i, j, kstart] = phi_prv[i, j, kstart]
    d[i, j, kstart + 1 : kstop - 1] = phi_prv[
        i, j, kstart + 1 : kstop - 1
    ] - gamma * (
        w[i, j, kstart : kstop - 2] * phi[i, j, kstart : kstop - 2]
        - w[i, j, kstart + 2 : kstop] * phi[i, j, kstart + 2 : kstop]
    )
    d[i, j, kstop - 1] = phi_prv[i, j, kstop - 1]


@gtscript.function
def setup_tridiagonal_system(
    gamma: float,
    w: typing.gtfield_t,
    phi: typing.gtfield_t,
    phi_prv: typing.gtfield_t,
) -> "Tuple[typing.gtfield_t, typing.gtfield_t, typing.gtfield_t]":
    a = gamma * w[0, 0, -1]
    c = -gamma * w[0, 0, 1]
    d = phi_prv[0, 0, 0] - gamma * (
        w[0, 0, -1] * phi[0, 0, -1] - w[0, 0, 1] * phi[0, 0, 1]
    )
    return a, c, d


@gtscript.function
def setup_tridiagonal_system_bc(
    phi_prv: typing.gtfield_t,
) -> "Tuple[typing.gtfield_t, typing.gtfield_t, typing.gtfield_t]":
    a = 0.0
    c = 0.0
    d = phi_prv[0, 0, 0]
    return a, c, d


@register(name="isentropic_vertical_advection")
class IsentropicVerticalAdvection(STSTendencyStepper, StencilFactory):
    """Couple the Crank-Nicholson integrator with centered finite differences
    in space to discretize the vertical advection in the isentropic model."""

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        backend_options=None,
        storage_options=None,
        **kwargs
    ):
        core = None
        for arg in args:
            core = (
                arg
                if isinstance(
                    arg, IsentropicImplicitVerticalAdvectionDiagnostic
                )
                else None
            )
        assert core is not None

        super().__init__(
            core,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )
        super(STSTendencyStepper, self).__init__(
            backend, backend_options, storage_options
        )

        # overwrite properties
        self.input_properties = core.input_properties.copy()
        self.provisional_input_properties = core.input_properties.copy()
        self.provisional_input_properties.pop(
            "tendency_of_air_potential_temperature", None
        )
        self.provisional_input_properties.pop(
            "tendency_of_air_potential_temperature_on_interface_levels", None
        )
        self.diagnostic_properties = {}
        self.output_properties = core.diagnostic_properties.copy()

        # set flags
        self._moist = mfwv in self.input_properties
        self._stgz = (
            "tendency_of_air_potential_temperature_on_interface_levels"
            in self.input_properties
        )

        # extract grid properties
        self._nx = core.grid.nx
        self._ny = core.grid.ny
        self._nz = core.grid.nz
        self._dz = core.grid.dz.to_units("K").values.item()

        # instantiate stencil object
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "moist": self._moist,
            "vstaggering": self._stgz,
            "setup_tridiagonal_system": setup_tridiagonal_system,
            "setup_tridiagonal_system_bc": setup_tridiagonal_system_bc,
        }
        self._stencil = self.compile("stencil")

    def _call(self, state, prv_state, timestep):
        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # grab arrays from input state
        in_w = (
            (
                state[
                    "tendency_of_air_potential_temperature_on_interface_levels"
                ]
                if self._stgz
                else state["tendency_of_air_potential_temperature"]
            )
            .to_units("K s^-1")
            .data
        )
        in_s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
        in_su = (
            state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
        )
        in_sv = (
            state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").data
        )
        if self._moist:
            in_qv = state[mfwv].to_units("g g^-1").data
            in_qc = state[mfcw].to_units("g g^-1").data
            in_qr = state[mfpw].to_units("g g^-1").data

        # grab arrays from provisional state
        in_s_prv = (
            prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").data
        )
        in_su_prv = (
            prv_state["x_momentum_isentropic"]
            .to_units("kg m^-1 K^-1 s^-1")
            .data
        )
        in_sv_prv = (
            prv_state["y_momentum_isentropic"]
            .to_units("kg m^-1 K^-1 s^-1")
            .data
        )
        if self._moist:
            in_qv_prv = prv_state[mfwv].to_units("g g^-1").data
            in_qc_prv = prv_state[mfcw].to_units("g g^-1").data
            in_qr_prv = prv_state[mfpw].to_units("g g^-1").data

        # grab output arrays
        out_s = out_state["air_isentropic_density"].data
        out_su = out_state["x_momentum_isentropic"].data
        out_sv = out_state["y_momentum_isentropic"].data
        if self._moist:
            out_qv = out_state[mfwv].data
            out_qc = out_state[mfcw].data
            out_qr = out_state[mfpw].data

        # set the stencil's arguments
        stencil_args = {
            "gamma": timestep.total_seconds() / (4.0 * self._dz),
            "in_w": in_w,
            "in_s": in_s,
            "in_s_prv": in_s_prv,
            "out_s": out_s,
            "in_su": in_su,
            "in_su_prv": in_su_prv,
            "out_su": out_su,
            "in_sv": in_sv,
            "in_sv_prv": in_sv_prv,
            "out_sv": out_sv,
        }
        if self._moist:
            stencil_args.update(
                {
                    "in_qv": in_qv,
                    "in_qv_prv": in_qv_prv,
                    "out_qv": out_qv,
                    "in_qc": in_qc,
                    "in_qc_prv": in_qc_prv,
                    "out_qc": out_qc,
                    "in_qr": in_qr,
                    "in_qr_prv": in_qr_prv,
                    "out_qr": out_qr,
                }
            )

        # run the stencil
        self._stencil(
            **stencil_args,
            origin=(0, 0, 0),
            domain=(self._nx, self._ny, self._nz),
            validate_args=False
        )

        return {}, out_state

    @stencil_definition(backend=("numpy", "cupy"), stencil="stencil")
    def _stencil_numpy(
        self,
        in_w: np.ndarray,
        in_s: np.ndarray,
        in_s_prv: np.ndarray,
        out_s: np.ndarray,
        in_su: np.ndarray,
        in_su_prv: np.ndarray,
        out_su: np.ndarray,
        in_sv: np.ndarray,
        in_sv_prv: np.ndarray,
        out_sv: np.ndarray,
        in_qv: Optional[np.ndarray] = None,
        in_qv_prv: Optional[np.ndarray] = None,
        out_qv: Optional[np.ndarray] = None,
        in_qc: Optional[np.ndarray] = None,
        in_qc_prv: Optional[np.ndarray] = None,
        out_qc: Optional[np.ndarray] = None,
        in_qr: Optional[np.ndarray] = None,
        in_qr_prv: Optional[np.ndarray] = None,
        out_qr: Optional[np.ndarray] = None,
        *,
        gamma: float,
        origin: typing.triplet_int_t,
        domain: typing.triplet_int_t,
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        kstart, kstop = origin[2], origin[2] + domain[2]

        # interpolate the velocity on the main levels
        if self._stgz:
            w = np.zeros_like(in_w)
            w[i, j, kstart:kstop] = 0.5 * (
                in_w[i, j, kstart:kstop] + in_w[i, j, kstart + 1 : kstop + 1]
            )
        else:
            w = in_w

        # compute the isentropic density of the water species
        if self._moist:
            sqv = np.zeros_like(in_qv)
            sqv_prv = np.zeros_like(in_qv)
            sqc = np.zeros_like(in_qc)
            sqc_prv = np.zeros_like(in_qv)
            sqr = np.zeros_like(in_qr)
            sqr_prv = np.zeros_like(in_qv)

            sqv[i, j, kstart:kstop] = (
                in_s[i, j, kstart:kstop] * in_qv[i, j, kstart:kstop]
            )
            sqv_prv[i, j, kstart:kstop] = (
                in_s_prv[i, j, kstart:kstop] * in_qv_prv[i, j, kstart:kstop]
            )
            sqc[i, j, kstart:kstop] = (
                in_s[i, j, kstart:kstop] * in_qc[i, j, kstart:kstop]
            )
            sqc_prv[i, j, kstart:kstop] = (
                in_s_prv[i, j, kstart:kstop] * in_qc_prv[i, j, kstart:kstop]
            )
            sqr[i, j, kstart:kstop] = (
                in_s[i, j, kstart:kstop] * in_qr[i, j, kstart:kstop]
            )
            sqr_prv[i, j, kstart:kstop] = (
                in_s_prv[i, j, kstart:kstop] * in_qr_prv[i, j, kstart:kstop]
            )
        else:
            sqv = sqv_prv = sqc = sqc_prv = sqr = sqr_prv = None

        #
        # isentropic density
        #
        # set up the tridiagonal system
        a = np.zeros_like(in_s)
        b = np.ones_like(in_s)
        c = np.zeros_like(in_s)
        d = np.zeros_like(in_s)
        setup_tridiagonal_system_numpy(
            gamma,
            w,
            in_s,
            in_s_prv,
            a,
            c,
            d,
            i=i,
            j=j,
            kstart=kstart,
            kstop=kstop,
        )

        # solve the tridiagonal system
        thomas_numpy(a, b, c, d, out_s, i=i, j=j, kstart=kstart, kstop=kstop)

        #
        # x-momentum
        #
        # set up the tridiagonal system
        setup_tridiagonal_system_numpy(
            gamma,
            w,
            in_su,
            in_su_prv,
            a,
            c,
            d,
            i=i,
            j=j,
            kstart=kstart,
            kstop=kstop,
        )

        # solve the tridiagonal system
        thomas_numpy(a, b, c, d, out_su, i=i, j=j, kstart=kstart, kstop=kstop)

        #
        # y-momentum
        #
        # set up the tridiagonal system
        setup_tridiagonal_system_numpy(
            gamma,
            w,
            in_sv,
            in_sv_prv,
            a,
            c,
            d,
            i=i,
            j=j,
            kstart=kstart,
            kstop=kstop,
        )

        # solve the tridiagonal system
        thomas_numpy(a, b, c, d, out_sv, i=i, j=j, kstart=kstart, kstop=kstop)

        if self._moist:
            #
            # isentropic density of water vapor
            #
            # set up the tridiagonal system
            setup_tridiagonal_system_numpy(
                gamma,
                w,
                sqv,
                sqv_prv,
                a,
                c,
                d,
                i=i,
                j=j,
                kstart=kstart,
                kstop=kstop,
            )

            # solve the tridiagonal system
            out_sqv = np.zeros_like(sqv)
            thomas_numpy(
                a, b, c, d, out_sqv, i=i, j=j, kstart=kstart, kstop=kstop
            )

            #
            # isentropic density of cloud liquid water
            #
            # set up the tridiagonal system
            setup_tridiagonal_system_numpy(
                gamma,
                w,
                sqc,
                sqc_prv,
                a,
                c,
                d,
                i=i,
                j=j,
                kstart=kstart,
                kstop=kstop,
            )

            # solve the tridiagonal system
            out_sqc = np.zeros_like(sqc)
            thomas_numpy(
                a, b, c, d, out_sqc, i=i, j=j, kstart=kstart, kstop=kstop
            )

            #
            # isentropic density of precipitation water
            #
            # set up the tridiagonal system
            setup_tridiagonal_system_numpy(
                gamma,
                w,
                sqr,
                sqr_prv,
                a,
                c,
                d,
                i=i,
                j=j,
                kstart=kstart,
                kstop=kstop,
            )

            # solve the tridiagonal system
            out_sqr = np.zeros_like(sqr)
            thomas_numpy(
                a, b, c, d, out_sqr, i=i, j=j, kstart=kstart, kstop=kstop
            )

            #
            # mass fraction of the water species
            #
            out_qv[i, j, kstart:kstop] = (
                out_sqv[i, j, kstart:kstop] / out_s[i, j, kstart:kstop]
            )
            out_qc[i, j, kstart:kstop] = (
                out_sqc[i, j, kstart:kstop] / out_s[i, j, kstart:kstop]
            )
            out_qr[i, j, kstart:kstop] = (
                out_sqr[i, j, kstart:kstop] / out_s[i, j, kstart:kstop]
            )

    @staticmethod
    @stencil_definition(backend="gt4py:*", stencil="stencil")
    def _stencil_gt4py(
        in_w: gtscript.Field["dtype"],
        in_s: gtscript.Field["dtype"],
        in_s_prv: gtscript.Field["dtype"],
        out_s: gtscript.Field["dtype"],
        in_su: gtscript.Field["dtype"],
        in_su_prv: gtscript.Field["dtype"],
        out_su: gtscript.Field["dtype"],
        in_sv: gtscript.Field["dtype"],
        in_sv_prv: gtscript.Field["dtype"],
        out_sv: gtscript.Field["dtype"],
        in_qv: gtscript.Field["dtype"] = None,
        in_qv_prv: gtscript.Field["dtype"] = None,
        out_qv: gtscript.Field["dtype"] = None,
        in_qc: gtscript.Field["dtype"] = None,
        in_qc_prv: gtscript.Field["dtype"] = None,
        out_qc: gtscript.Field["dtype"] = None,
        in_qr: gtscript.Field["dtype"] = None,
        in_qr_prv: gtscript.Field["dtype"] = None,
        out_qr: gtscript.Field["dtype"] = None,
        *,
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
                sqv_prv = in_s_prv * in_qv_prv
                sqc = in_s * in_qc
                sqc_prv = in_s_prv * in_qc_prv
                sqr = in_s * in_qr
                sqr_prv = in_s_prv * in_qr_prv

        #
        # isentropic density
        #
        # set up the tridiagonal system
        with computation(PARALLEL), interval(0, 1):
            a_s, c_s, d_s = setup_tridiagonal_system_bc(in_s_prv)
        with computation(PARALLEL), interval(1, -1):
            a_s, c_s, d_s = setup_tridiagonal_system(gamma, w, in_s, in_s_prv)
        with computation(PARALLEL), interval(-1, None):
            a_s, c_s, d_s = setup_tridiagonal_system_bc(in_s_prv)

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
                (delta_s[0, 0, 0] - c_s[0, 0, 0] * out_s[0, 0, 1])
                / beta_s[0, 0, 0]
                if beta_s[0, 0, 0] != 0.0
                else (delta_s[0, 0, 0] - c_s[0, 0, 0] * out_s[0, 0, 1])
            )

        #
        # x-momentum
        #
        # set up the tridiagonal system
        with computation(PARALLEL), interval(0, 1):
            a_su, c_su, d_su = setup_tridiagonal_system_bc(in_su_prv)
        with computation(PARALLEL), interval(1, -1):
            a_su, c_su, d_su = setup_tridiagonal_system(
                gamma, w, in_su, in_su_prv
            )
        with computation(PARALLEL), interval(-1, None):
            a_su, c_su, d_su = setup_tridiagonal_system_bc(in_su_prv)

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
                (delta_su[0, 0, 0] - c_su[0, 0, 0] * out_su[0, 0, 1])
                / beta_su[0, 0, 0]
                if beta_su[0, 0, 0] != 0.0
                else (delta_su[0, 0, 0] - c_su[0, 0, 0] * out_su[0, 0, 1])
            )

        #
        # y-momentum
        #
        # set up the tridiagonal system
        with computation(PARALLEL), interval(0, 1):
            a_sv, c_sv, d_sv = setup_tridiagonal_system_bc(in_sv_prv)
        with computation(PARALLEL), interval(1, -1):
            a_sv, c_sv, d_sv = setup_tridiagonal_system(
                gamma, w, in_sv, in_sv_prv
            )
        with computation(PARALLEL), interval(-1, None):
            a_sv, c_sv, d_sv = setup_tridiagonal_system_bc(in_sv_prv)

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
                (delta_sv[0, 0, 0] - c_sv[0, 0, 0] * out_sv[0, 0, 1])
                / beta_sv[0, 0, 0]
                if beta_sv[0, 0, 0] != 0.0
                else (delta_sv[0, 0, 0] - c_sv[0, 0, 0] * out_sv[0, 0, 1])
            )

        if __INLINED(moist):
            #
            # isentropic density of water vapor
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqv, c_sqv, d_sqv = setup_tridiagonal_system_bc(sqv_prv)
            with computation(PARALLEL), interval(1, -1):
                a_sqv, c_sqv, d_sqv = setup_tridiagonal_system(
                    gamma, w, sqv, sqv_prv
                )
            with computation(PARALLEL), interval(-1, None):
                a_sqv, c_sqv, d_sqv = setup_tridiagonal_system_bc(sqv_prv)

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
                delta_sqv = (
                    d_sqv[0, 0, 0] - omega_sqv[0, 0, 0] * delta_sqv[0, 0, -1]
                )
            with computation(BACKWARD), interval(-1, None):
                out_sqv = (
                    delta_sqv[0, 0, 0] / beta_sqv[0, 0, 0]
                    if beta_sqv[0, 0, 0] != 0.0
                    else delta_sqv[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqv = (
                    (delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1])
                    / beta_sqv[0, 0, 0]
                    if beta_sqv[0, 0, 0] != 0.0
                    else (
                        delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1]
                    )
                )

            #
            # isentropic density of cloud liquid water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqc, c_sqc, d_sqc = setup_tridiagonal_system_bc(sqc_prv)
            with computation(PARALLEL), interval(1, -1):
                a_sqc, c_sqc, d_sqc = setup_tridiagonal_system(
                    gamma, w, sqc, sqc_prv
                )
            with computation(PARALLEL), interval(-1, None):
                a_sqc, c_sqc, d_sqc = setup_tridiagonal_system_bc(sqc_prv)

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
                delta_sqc = (
                    d_sqc[0, 0, 0] - omega_sqc[0, 0, 0] * delta_sqc[0, 0, -1]
                )
            with computation(BACKWARD), interval(-1, None):
                out_sqc = (
                    delta_sqc[0, 0, 0] / beta_sqc[0, 0, 0]
                    if beta_sqc[0, 0, 0] != 0.0
                    else delta_sqc[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqc = (
                    (delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1])
                    / beta_sqc[0, 0, 0]
                    if beta_sqc[0, 0, 0] != 0.0
                    else (
                        delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1]
                    )
                )

            #
            # isentropic density of precipitation water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqr, c_sqr, d_sqr = setup_tridiagonal_system_bc(sqr_prv)
            with computation(PARALLEL), interval(1, -1):
                a_sqr, c_sqr, d_sqr = setup_tridiagonal_system(
                    gamma, w, sqr, sqr_prv
                )
            with computation(PARALLEL), interval(-1, None):
                a_sqr, c_sqr, d_sqr = setup_tridiagonal_system_bc(sqr_prv)

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
                delta_sqr = (
                    d_sqr[0, 0, 0] - omega_sqr[0, 0, 0] * delta_sqr[0, 0, -1]
                )
            with computation(BACKWARD), interval(-1, None):
                out_sqr = (
                    delta_sqr[0, 0, 0] / beta_sqr[0, 0, 0]
                    if beta_sqr[0, 0, 0] != 0.0
                    else delta_sqr[0, 0, 0]
                )
            with computation(BACKWARD), interval(0, -1):
                out_sqr = (
                    (delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1])
                    / beta_sqr[0, 0, 0]
                    if beta_sqr[0, 0, 0] != 0.0
                    else (
                        delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1]
                    )
                )

        # calculate the output mass fraction of the water species
        if __INLINED(moist):
            with computation(PARALLEL), interval(...):
                out_qv = out_sqv / out_s
                out_qc = out_sqc / out_s
                out_qr = out_sqr / out_s
