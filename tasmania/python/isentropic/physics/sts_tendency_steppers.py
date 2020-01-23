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
from typing import Tuple

from gt4py import gtscript

from tasmania.python.framework.sts_tendency_steppers import STSTendencyStepper, registry
from tasmania.python.isentropic.physics.implicit_vertical_advection import (
    IsentropicImplicitVerticalAdvectionDiagnostic,
)
from tasmania.python.utils import taz_types


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


@gtscript.function
def setup_tridiagonal_system(
    gamma: float,
    w: taz_types.gtfield_t,
    phi: taz_types.gtfield_t,
    phi_prv: taz_types.gtfield_t,
) -> "Tuple[taz_types.gtfield_t, taz_types.gtfield_t, taz_types.gtfield_t]":
    a = gamma * w[0, 0, -1]
    c = -gamma * w[0, 0, 1]
    d = phi_prv[0, 0, 0] - gamma * (
        w[0, 0, -1] * phi[0, 0, -1] - w[0, 0, 1] * phi[0, 0, 1]
    )
    return a, c, d


@gtscript.function
def setup_tridiagonal_system_bc(
    phi_prv: taz_types.gtfield_t
) -> "Tuple[taz_types.gtfield_t, taz_types.gtfield_t, taz_types.gtfield_t]":
    a = 0.0
    c = 0.0
    d = phi_prv[0, 0, 0]
    return a, c, d


@registry(scheme_name="isentropic_vertical_advection")
class IsentropicVerticalAdvection(STSTendencyStepper):
    """ Couple the Crank-Nicholson integrator with centered finite differences
    in space to discretize the vertical advection in the isentropic model. """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        gt_powered=False,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=np.float64,
        rebuild=False,
        **kwargs
    ):
        core = None
        for arg in args:
            core = (
                arg
                if isinstance(arg, IsentropicImplicitVerticalAdvectionDiagnostic)
                else None
            )
        assert core is not None

        super().__init__(
            core,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            rebuild=rebuild,
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
        externals = {
            "moist": self._moist,
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

    def _call(self, state, prv_state, timestep):
        # initialize the output state
        self._out_state = self._out_state or self._allocate_output_state(state)
        out_state = self._out_state

        # grab arrays from input state
        in_w = (
            (
                state["tendency_of_air_potential_temperature_on_interface_levels"]
                if self._stgz
                else state["tendency_of_air_potential_temperature"]
            )
            .to_units("K s^-1")
            .values
        )
        in_s = state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
        in_su = state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
        in_sv = state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
        if self._moist:
            in_qv = state[mfwv].to_units("g g^-1").values
            in_qc = state[mfcw].to_units("g g^-1").values
            in_qr = state[mfpw].to_units("g g^-1").values

        # grab arrays from provisional state
        in_s_prv = prv_state["air_isentropic_density"].to_units("kg m^-2 K^-1").values
        in_su_prv = (
            prv_state["x_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
        )
        in_sv_prv = (
            prv_state["y_momentum_isentropic"].to_units("kg m^-1 K^-1 s^-1").values
        )
        if self._moist:
            in_qv_prv = prv_state[mfwv].to_units("g g^-1").values
            in_qc_prv = prv_state[mfcw].to_units("g g^-1").values
            in_qr_prv = prv_state[mfpw].to_units("g g^-1").values

        # grab output arrays
        out_s = out_state["air_isentropic_density"].values
        out_su = out_state["x_momentum_isentropic"].values
        out_sv = out_state["y_momentum_isentropic"].values
        if self._moist:
            out_qv = out_state[mfwv].values
            out_qc = out_state[mfcw].values
            out_qr = out_state[mfpw].values

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
            origin={"_all_": (0, 0, 0)},
            domain=(self._nx, self._ny, self._nz),
        )

        return {}, out_state

    @staticmethod
    def _stencil_defs(
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
                (delta_s[0, 0, 0] - c_s[0, 0, 0] * out_s[0, 0, 1]) / beta_s[0, 0, 0]
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
            a_su, c_su, d_su = setup_tridiagonal_system(gamma, w, in_su, in_su_prv)
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
                (delta_su[0, 0, 0] - c_su[0, 0, 0] * out_su[0, 0, 1]) / beta_su[0, 0, 0]
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
            a_sv, c_sv, d_sv = setup_tridiagonal_system(gamma, w, in_sv, in_sv_prv)
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
                a_sqv, c_sqv, d_sqv = setup_tridiagonal_system_bc(sqv_prv)
            with computation(PARALLEL), interval(1, -1):
                a_sqv, c_sqv, d_sqv = setup_tridiagonal_system(gamma, w, sqv, sqv_prv)
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
                delta_sqv = d_sqv[0, 0, 0] - omega_sqv[0, 0, 0] * delta_sqv[0, 0, -1]
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
                    else (delta_sqv[0, 0, 0] - c_sqv[0, 0, 0] * out_sqv[0, 0, 1])
                )

            #
            # isentropic density of cloud liquid water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqc, c_sqc, d_sqc = setup_tridiagonal_system_bc(sqc_prv)
            with computation(PARALLEL), interval(1, -1):
                a_sqc, c_sqc, d_sqc = setup_tridiagonal_system(gamma, w, sqc, sqc_prv)
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
                delta_sqc = d_sqc[0, 0, 0] - omega_sqc[0, 0, 0] * delta_sqc[0, 0, -1]
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
                    else (delta_sqc[0, 0, 0] - c_sqc[0, 0, 0] * out_sqc[0, 0, 1])
                )

            #
            # isentropic density of precipitation water
            #
            # set up the tridiagonal system
            with computation(PARALLEL), interval(0, 1):
                a_sqr, c_sqr, d_sqr = setup_tridiagonal_system_bc(sqr_prv)
            with computation(PARALLEL), interval(1, -1):
                a_sqr, c_sqr, d_sqr = setup_tridiagonal_system(gamma, w, sqr, sqr_prv)
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
                delta_sqr = d_sqr[0, 0, 0] - omega_sqr[0, 0, 0] * delta_sqr[0, 0, -1]
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
                    else (delta_sqr[0, 0, 0] - c_sqr[0, 0, 0] * out_sqr[0, 0, 1])
                )

        # calculate the output mass fraction of the water species
        if __INLINED(moist):
            with computation(PARALLEL), interval(...):
                out_qv = out_sqv / out_s
                out_qc = out_sqc / out_s
                out_qr = out_sqr / out_s
