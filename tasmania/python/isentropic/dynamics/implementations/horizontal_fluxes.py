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
from gridtools import gtscript, __externals__

from tasmania.python.isentropic.dynamics.horizontal_fluxes import IsentropicHorizontalFlux


@gtscript.function
def get_upwind_flux_x(u, phi):
    flux = u[1, 0, 0] * (
        (u[1, 0, 0] > 0.0) * phi[0, 0, 0] + (u[1, 0, 0] < 0.0) * phi[1, 0, 0]
    )
    return flux


@gtscript.function
def get_upwind_flux_y(v, phi):
    flux = v[0, 1, 0] * (
        (v[0, 1, 0] > 0.0) * phi[0, 0, 0] + (v[0, 1, 0] < 0.0) * phi[0, 1, 0]
    )
    return flux


class Upwind(IsentropicHorizontalFlux):
    """ Upwind scheme. """

    extent = 1
    order = 1
    externals = {
        "get_upwind_flux_x": get_upwind_flux_x,
        "get_upwind_flux_y": get_upwind_flux_y,
    }

    @staticmethod
    @gtscript.function
    def __call__(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        mtg,
        su,
        sv,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        from __externals__ import get_upwind_flux_x, get_upwind_flux_y, moist

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_upwind_flux_x(u=u, phi=s)
        flux_s_y = get_upwind_flux_y(v=v, phi=s)
        flux_su_x = get_upwind_flux_x(u=u, phi=su)
        flux_su_y = get_upwind_flux_y(v=v, phi=su)
        flux_sv_x = get_upwind_flux_x(u=u, phi=sv)
        flux_sv_y = get_upwind_flux_y(v=v, phi=sv)

        if not moist:  # compile-time if
            return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
        else:
            # compute fluxes for the water constituents
            flux_sqv_x = get_upwind_flux_x(u=u, phi=sqv)
            flux_sqv_y = get_upwind_flux_y(v=v, phi=sqv)
            flux_sqc_x = get_upwind_flux_x(u=u, phi=sqc)
            flux_sqc_y = get_upwind_flux_y(v=v, phi=sqc)
            flux_sqr_x = get_upwind_flux_x(u=u, phi=sqr)
            flux_sqr_y = get_upwind_flux_y(v=v, phi=sqr)

            return (
                flux_s_x,
                flux_s_y,
                flux_su_x,
                flux_su_y,
                flux_sv_x,
                flux_sv_y,
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            )


@gtscript.function
def get_centered_flux_x(u, phi):
    flux = u[1, 0, 0] * 0.5 * (phi[0, 0, 0] + phi[1, 0, 0])
    return flux


@gtscript.function
def get_centered_flux_y(v, phi):
    flux = v[0, 1, 0] * 0.5 * (phi[0, 0, 0] + phi[0, 1, 0])
    return flux


class Centered(IsentropicHorizontalFlux):
    """ Centered scheme. """

    extent = 1
    order = 2
    externals = {
        "get_centered_flux_x": get_centered_flux_x,
        "get_centered_flux_y": get_centered_flux_y,
    }

    @staticmethod
    @gtscript.function
    def __call__(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        mtg,
        su,
        sv,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        from __externals__ import get_centered_flux_x, get_centered_flux_y, moist

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_centered_flux_x(u=u, phi=s)
        flux_s_y = get_centered_flux_y(v=v, phi=s)
        flux_su_x = get_centered_flux_x(u=u, phi=su)
        flux_su_y = get_centered_flux_y(v=v, phi=su)
        flux_sv_x = get_centered_flux_x(u=u, phi=sv)
        flux_sv_y = get_centered_flux_y(v=v, phi=sv)

        if not moist:  # compile-time if
            return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
        else:
            # compute fluxes for the water constituents
            flux_sqv_x = get_centered_flux_x(u=u, phi=sqv)
            flux_sqv_y = get_centered_flux_y(v=v, phi=sqv)
            flux_sqc_x = get_centered_flux_x(u=u, phi=sqc)
            flux_sqc_y = get_centered_flux_y(v=v, phi=sqc)
            flux_sqr_x = get_centered_flux_x(u=u, phi=sqr)
            flux_sqr_y = get_centered_flux_y(v=v, phi=sqr)

            return (
                flux_s_x,
                flux_s_y,
                flux_su_x,
                flux_su_y,
                flux_sv_x,
                flux_sv_y,
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            )


@gtscript.function
def get_maccormack_predicted_value_s(dt, dx, dy, s, su, sv):
    s_prd = s[0, 0, 0] - dt * (
        (su[1, 0, 0] - su[0, 0, 0]) / dx + (sv[0, 1, 0] - sv[0, 0, 0]) / dy
    )
    return s_prd


@gtscript.function
def get_maccormack_predicted_value_su(dt, dx, dy, s, u_unstg, v_unstg, mtg, su, su_tnd):
    from __externals__ import su_tnd_on

    if su_tnd_on:  # compile-time if
        su_prd = su[0, 0, 0] - dt * (
            (u_unstg[1, 0, 0] * su[1, 0, 0] - u_unstg[0, 0, 0] * su[0, 0, 0]) / dx
            + (v_unstg[0, 1, 0] * su[0, 1, 0] - v_unstg[0, 0, 0] * su[0, 0, 0]) / dy
            + s[0, 0, 0] * (mtg[1, 0, 0] - mtg[0, 0, 0]) / dx
        )
    else:
        su_prd = su[0, 0, 0] - dt * (
            (u_unstg[1, 0, 0] * su[1, 0, 0] - u_unstg[0, 0, 0] * su[0, 0, 0]) / dx
            + (v_unstg[0, 1, 0] * su[0, 1, 0] - v_unstg[0, 0, 0] * su[0, 0, 0]) / dy
            + s[0, 0, 0] * (mtg[1, 0, 0] - mtg[0, 0, 0]) / dx
            - su_tnd[0, 0, 0]
        )

    return su_prd


@gtscript.function
def get_maccormack_predicted_value_sv(dt, dx, dy, s, u_unstg, v_unstg, mtg, sv, sv_tnd):
    from __externals__ import sv_tnd_on

    if sv_tnd_on is None:  # compile-time if
        sv_prd = sv[0, 0, 0] - dt * (
            (u_unstg[1, 0, 0] * sv[1, 0, 0] - u_unstg[0, 0, 0] * sv[0, 0, 0]) / dx
            + (v_unstg[0, 1, 0] * sv[0, 1, 0] - v_unstg[0, 0, 0] * sv[0, 0, 0]) / dy
            + s[0, 0, 0] * (mtg[0, 1, 0] - mtg[0, 0, 0]) / dy
        )
    else:
        sv_prd = sv[0, 0, 0] - dt * (
            (u_unstg[1, 0, 0] * sv[1, 0, 0] - u_unstg[0, 0, 0] * sv[0, 0, 0]) / dx
            + (v_unstg[0, 1, 0] * sv[0, 1, 0] - v_unstg[0, 0, 0] * sv[0, 0, 0]) / dy
            + s[0, 0, 0] * (mtg[0, 1, 0] - mtg[0, 0, 0]) / dy
            - sv_tnd[0, 0, 0]
        )

    return sv_prd


@gtscript.function
def get_maccormack_predicted_value_sq(
    dt, dx, dy, s, u_unstg, v_unstg, sq, q_tnd_on, q_tnd
):
    if q_tnd_on is None:  # compile-time if
        sq_prd = sq[0, 0, 0] - dt * (
            (u_unstg[1, 0, 0] * sq[1, 0, 0] - u_unstg[0, 0, 0] * sq[0, 0, 0]) / dx
            + (v_unstg[0, 1, 0] * sq[0, 1, 0] - v_unstg[0, 0, 0] * sq[0, 0, 0]) / dy
        )
    else:
        sq_prd = sq[0, 0, 0] - dt * (
            (u_unstg[1, 0, 0] * sq[1, 0, 0] - u_unstg[0, 0, 0] * sq[0, 0, 0]) / dx
            + (v_unstg[0, 1, 0] * sq[0, 1, 0] - v_unstg[0, 0, 0] * sq[0, 0, 0]) / dy
            - s[0, 0, 0] * q_tnd[0, 0, 0]
        )
    return sq_prd


@gtscript.function
def get_maccormack_flux_x(u_unstg, phi, u_prd_unstg, phi_prd):
    flux = 0.5 * (
        u_unstg[1, 0, 0] * phi[1, 0, 0] + u_prd_unstg[0, 0, 0] * phi_prd[0, 0, 0]
    )
    return flux


@gtscript.function
def get_maccormack_flux_x_s(su, su_prd):
    flux_s_x = 0.5 * (su[1, 0, 0] + su_prd[0, 0, 0])
    return flux_s_x


@gtscript.function
def get_maccormack_flux_y(v_unstg, phi, v_prd_unstg, phi_prd):
    flux = 0.5 * (
        v_unstg[0, 1, 0] * phi[0, 1, 0] + v_prd_unstg[0, 0, 0] * phi_prd[0, 0, 0]
    )

    return flux


@gtscript.function
def get_maccormack_flux_y_s(sv, sv_prd):
    flux_s_y = 0.5 * (sv[0, 1, 0] + sv_prd[0, 0, 0])
    return flux_s_y


class MacCormack(IsentropicHorizontalFlux):
    """	MacCormack scheme. """

    extent = 1
    order = 2
    externals = {
        "get_maccormack_predicted_value_s": get_maccormack_predicted_value_s,
        "get_maccormack_predicted_value_su": get_maccormack_predicted_value_su,
        "get_maccormack_predicted_value_sv": get_maccormack_predicted_value_sv,
        "get_maccormack_predicted_value_sq": get_maccormack_predicted_value_sq,
        "get_maccormack_flux_x": get_maccormack_flux_x,
        "get_maccormack_flux_x_s": get_maccormack_flux_x_s,
        "get_maccormack_flux_y": get_maccormack_flux_y,
        "get_maccormack_flux_y_s": get_maccormack_flux_y_s,
    }

    @staticmethod
    @gtscript.function
    def __call__(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        mtg,
        su,
        sv,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        from __externals__ import (
            get_maccormack_predicted_values_s,
            get_maccormack_predicted_values_su,
            get_maccormack_predicted_values_sv,
            get_maccormack_predicted_value_sq,
        )

        # diagnose the velocity components at the mass points
        u_unstg = su[0, 0, 0] / s[0, 0, 0]
        v_unstg = sv[0, 0, 0] / s[0, 0, 0]

        # compute the predicted values for the isentropic density and the momenta
        s_prd = get_maccormack_predicted_value_s(dt=dt, dx=dx, dy=dy, s=s, su=su, sv=sv)
        su_prd = get_maccormack_predicted_value_su(
            dt=dt,
            dx=dx,
            dy=dy,
            s=s,
            u_unstg=u_unstg,
            v_unstg=v_unstg,
            mtg=mtg,
            su=su,
            su_tnd=su_tnd,
        )
        sv_prd = get_maccormack_predicted_value_sv(
            dt=dt,
            dx=dx,
            dy=dy,
            s=s,
            u_unstg=u_unstg,
            v_unstg=v_unstg,
            mtg=mtg,
            sv=sv,
            sv_tnd=sv_tnd,
        )

        if moist:  # compile-time if
            # compute the predicted values for the water constituents
            sqv_prd = get_maccormack_predicted_value_sq(
                dt=dt,
                dx=dx,
                dy=dy,
                s=s,
                u_unstg=u_unstg,
                v_unstg=v_unstg,
                sq=sqv,
                q_tnd_on=qv_tnd_on,
                q_tnd=qv_tnd,
            )
            sqc_prd = get_maccormack_predicted_value_sq(
                dt=dt,
                dx=dx,
                dy=dy,
                s=s,
                u_unstg=u_unstg,
                v_unstg=v_unstg,
                sq=sqc,
                q_tnd_on=qc_tnd_on,
                q_tnd=qc_tnd,
            )
            sqr_prd = get_maccormack_predicted_value_sq(
                dt=dt,
                dx=dx,
                dy=dy,
                s=s,
                u_unstg=u_unstg,
                v_unstg=v_unstg,
                sq=sqr,
                q_tnd_on=qr_tnd_on,
                q_tnd=qr_tnd,
            )

        # diagnose the predicted values for the velocity components
        # at the mass points
        u_prd_unstg = su_prd[0, 0, 0] / s_prd[0, 0, 0]
        v_prd_unstg = sv_prd[0, 0, 0] / s_prd[0, 0, 0]

        # compute the fluxes for the isentropic density and the momenta
        flux_s_x = get_maccormack_flux_x_s(su=su, su_prd=su_prd)
        flux_s_y = get_maccormack_flux_y_s(sv=sv, sv_prd=sv_prd)
        flux_su_x = get_maccormack_flux_x(
            u_unstg=u_unstg, phi=su, u_prd_unstg=u_prd_unstg, phi_prd=su_prd
        )
        flux_su_y = get_maccormack_flux_y(
            v_unstg=v_unstg, phi=su, v_prd_unstg=v_prd_unstg, phi_prd=su_prd
        )
        flux_sv_x = get_maccormack_flux_x(
            u_unstg=u_unstg, phi=sv, u_prd_unstg=u_prd_unstg, phi_prd=sv_prd
        )
        flux_sv_y = get_maccormack_flux_y(
            v_unstg=v_unstg, phi=sv, v_prd_unstg=v_prd_unstg, phi_prd=sv_prd
        )

        if not moist:  # compile-time if
            return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
        if moist:  # compile-time if
            # compute the fluxes for the water constituents
            flux_sqv_x = get_maccormack_flux_x(
                u_unstg=u_unstg, phi=sqv, u_prd_unstg=u_prd_unstg, phi_prd=sqv_prd
            )
            flux_sqv_y = get_maccormack_flux_y(
                v_unstg=v_unstg, phi=sqv, v_prd_unstg=v_prd_unstg, phi_prd=sqv_prd
            )
            flux_sqc_x = get_maccormack_flux_x(
                u_unstg=u_unstg, phi=sqc, u_prd_unstg=u_prd_unstg, phi_prd=sqc_prd
            )
            flux_sqc_y = get_maccormack_flux_y(
                v_unstg=v_unstg, phi=sqc, v_prd_unstg=v_prd_unstg, phi_prd=sqc_prd
            )
            flux_sqr_x = get_maccormack_flux_x(
                u_unstg=u_unstg, phi=sqr, u_prd_unstg=u_prd_unstg, phi_prd=sqr_prd
            )
            flux_sqr_y = get_maccormack_flux_y(
                v_unstg=v_unstg, phi=sqr, v_prd_unstg=v_prd_unstg, phi_prd=sqr_prd
            )

            return (
                flux_s_x,
                flux_s_y,
                flux_su_x,
                flux_su_y,
                flux_sv_x,
                flux_sv_y,
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            )


@gtscript.function
def get_fourth_order_centered_flux_x(u, phi):
    flux = (
        u[1, 0, 0]
        / 12.0
        * (7.0 * (phi[1, 0, 0] + phi[0, 0, 0]) - (phi[2, 0, 0] + phi[-1, 0, 0]))
    )
    return flux


@gtscript.function
def get_third_order_upwind_flux_x(u, phi):
    from __externals__ import get_fourth_order_centered_flux_x

    flux4 = get_fourth_order_centered_flux_x(u=u, phi=phi)
    flux = flux4[0, 0, 0] - (
        (u[1, 0, 0] > 0.0) * u[1, 0, 0] - (u[1, 0, 0] < 0.0) * u[1, 0, 0]
    ) / 12.0 * (3.0 * (phi[1, 0, 0] - phi[0, 0, 0]) - (phi[2, 0, 0] - phi[-1, 0, 0]))

    return flux


@gtscript.function
def get_fourth_order_centered_flux_y(v, phi):
    flux = (
        v[0, 1, 0]
        / 12.0
        * (7.0 * (phi[0, 1, 0] + phi[0, 0, 0]) - (phi[0, 2, 0] + phi[0, -1, 0]))
    )
    return flux


@gtscript.function
def get_third_order_upwind_flux_y(v, phi):
    from __externals__ import get_fourth_order_centered_flux_y

    flux4 = get_fourth_order_centered_flux_y(v=v, phi=phi)
    flux = flux4[0, 0, 0] - (
        (v[0, 1, 0] > 0.0) * v[0, 1, 0] - (v[0, 1, 0] < 0.0) * v[0, 1, 0]
    ) / 12.0 * (3.0 * (phi[0, 1, 0] - phi[0, 0, 0]) - (phi[0, 2, 0] - phi[0, -1, 0]))

    return flux


class ThirdOrderUpwind(IsentropicHorizontalFlux):
    """ Third-order scheme. """

    extent = 2
    order = 3
    externals = {
        "get_fourth_order_centered_flux_x": get_fourth_order_centered_flux_x,
        "get_third_order_upwind_flux_x": get_third_order_upwind_flux_x,
        "get_fourth_order_centered_flux_y": get_fourth_order_centered_flux_y,
        "get_third_order_upwind_flux_y": get_third_order_upwind_flux_y,
    }

    @staticmethod
    @gtscript.function
    def __call__(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        mtg,
        su,
        sv,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        from __externals__ import (
            get_third_order_upwind_flux_x,
            get_third_order_upwind_flux_y,
            moist,
        )

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_third_order_upwind_flux_x(u=u, phi=s)
        flux_s_y = get_third_order_upwind_flux_y(v=v, phi=s)
        flux_su_x = get_third_order_upwind_flux_x(u=u, phi=su)
        flux_su_y = get_third_order_upwind_flux_y(v=v, phi=su)
        flux_sv_x = get_third_order_upwind_flux_x(u=u, phi=sv)
        flux_sv_y = get_third_order_upwind_flux_y(v=v, phi=sv)

        if not moist:  # compile-time if
            return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
        else:
            # compute fluxes for the water constituents
            flux_sqv_x = get_third_order_upwind_flux_x(u=u, phi=sqv)
            flux_sqv_y = get_third_order_upwind_flux_y(v=v, phi=sqv)
            flux_sqc_x = get_third_order_upwind_flux_x(u=u, phi=sqc)
            flux_sqc_y = get_third_order_upwind_flux_y(v=v, phi=sqc)
            flux_sqr_x = get_third_order_upwind_flux_x(u=u, phi=sqr)
            flux_sqr_y = get_third_order_upwind_flux_y(v=v, phi=sqr)

            return (
                flux_s_x,
                flux_s_y,
                flux_su_x,
                flux_su_y,
                flux_sv_x,
                flux_sv_y,
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            )


@gtscript.function
def get_sixth_order_centered_flux_x(u, phi):
    flux = (
        u[1, 0, 0]
        / 60.0
        * (
            37.0 * (phi[1, 0, 0] + phi[0, 0, 0])
            - 8.0 * (phi[2, 0, 0] + phi[-1, 0, 0])
            + (phi[3, 0, 0] + phi[-2, 0, 0])
        )
    )
    return flux


@gtscript.function
def get_fifth_order_upwind_flux_x(u, phi):
    from __externals__ import get_sixth_order_centered_flux_x

    flux6 = get_sixth_order_centered_flux_x(u=u, phi=phi)
    flux = flux6[0, 0, 0] - (
        (u[1, 0, 0] > 0.0) * u[1, 0, 0] - (u[1, 0, 0] < 0.0) * u[1, 0, 0]
    ) / 60.0 * (
        10.0 * (phi[1, 0, 0] - phi[0, 0, 0])
        - 5.0 * (phi[2, 0, 0] - phi[-1, 0, 0])
        + (phi[3, 0, 0] - phi[-2, 0, 0])
    )

    return flux


@gtscript.function
def get_sixth_order_centered_flux_y(v, phi):
    flux = (
        v[0, 1, 0]
        / 60.0
        * (
            37.0 * (phi[0, 1, 0] + phi[0, 0, 0])
            - 8.0 * (phi[0, 2, 0] + phi[0, -1, 0])
            + (phi[0, 3, 0] + phi[0, -2, 0])
        )
    )
    return flux


@gtscript.function
def get_fifth_order_upwind_flux_y(v, phi):
    from __externals__ import get_sixth_order_centered_flux_y

    flux6 = get_sixth_order_centered_flux_y(v=v, phi=phi)
    flux = flux6[0, 0, 0] - (
        (v[0, 1, 0] > 0.0) * v[0, 1, 0] - (v[0, 1, 0] < 0.0) * v[0, 1, 0]
    ) / 60.0 * (
        10.0 * (phi[0, 1, 0] - phi[0, 0, 0])
        - 5.0 * (phi[0, 2, 0] - phi[0, -1, 0])
        + (phi[0, 3, 0] - phi[0, -2, 0])
    )

    return flux


class FifthOrderUpwind(IsentropicHorizontalFlux):
    """ Fifth-order scheme. """

    extent = 3
    order = 5
    externals = {
        "get_sixth_order_centered_flux_x": get_sixth_order_centered_flux_x,
        "get_fifth_order_upwind_flux_x": get_fifth_order_upwind_flux_x,
        "get_sixth_order_centered_flux_y": get_sixth_order_centered_flux_y,
        "get_fifth_order_upwind_flux_y": get_fifth_order_upwind_flux_y,
    }

    @staticmethod
    @gtscript.function
    def __call__(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        mtg,
        su,
        sv,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
    ):
        from __externals__ import (
            get_fifth_order_upwind_flux_x,
            get_fifth_order_upwind_flux_y,
            moist,
        )

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_fifth_order_upwind_flux_x(u=u, phi=s)
        flux_s_y = get_fifth_order_upwind_flux_y(v=v, phi=s)
        flux_su_x = get_fifth_order_upwind_flux_x(u=u, phi=su)
        flux_su_y = get_fifth_order_upwind_flux_y(v=v, phi=su)
        flux_sv_x = get_fifth_order_upwind_flux_x(u=u, phi=sv)
        flux_sv_y = get_fifth_order_upwind_flux_y(v=v, phi=sv)

        if not moist:  # compile-time if
            return flux_s_x, flux_s_y, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y
        else:
            # compute fluxes for the water constituents
            flux_sqv_x = get_fifth_order_upwind_flux_x(u=u, phi=sqv)
            flux_sqv_y = get_fifth_order_upwind_flux_y(v=v, phi=sqv)
            flux_sqc_x = get_fifth_order_upwind_flux_x(u=u, phi=sqc)
            flux_sqc_y = get_fifth_order_upwind_flux_y(v=v, phi=sqc)
            flux_sqr_x = get_fifth_order_upwind_flux_x(u=u, phi=sqr)
            flux_sqr_y = get_fifth_order_upwind_flux_y(v=v, phi=sqr)

            return (
                flux_s_x,
                flux_s_y,
                flux_su_x,
                flux_su_y,
                flux_sv_x,
                flux_sv_y,
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            )
