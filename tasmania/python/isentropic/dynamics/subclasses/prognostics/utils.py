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
from typing import Union

from gt4py import gtscript

from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicHorizontalFlux,
    IsentropicMinimalHorizontalFlux,
)
from tasmania.python.utils import taz_types


# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


def step_forward_euler_numpy(
    s_now: gtscript.Field["dtype"],
    s_int: gtscript.Field["dtype"],
    s_new: gtscript.Field["dtype"],
    u_int: gtscript.Field["dtype"],
    v_int: gtscript.Field["dtype"],
    su_int: gtscript.Field["dtype"] = None,
    sv_int: gtscript.Field["dtype"] = None,
    mtg_int: gtscript.Field["dtype"] = None,
    sqv_now: gtscript.Field["dtype"] = None,
    sqv_int: gtscript.Field["dtype"] = None,
    sqv_new: gtscript.Field["dtype"] = None,
    sqc_now: gtscript.Field["dtype"] = None,
    sqc_int: gtscript.Field["dtype"] = None,
    sqc_new: gtscript.Field["dtype"] = None,
    sqr_now: gtscript.Field["dtype"] = None,
    sqr_int: gtscript.Field["dtype"] = None,
    sqr_new: gtscript.Field["dtype"] = None,
    s_tnd: gtscript.Field["dtype"] = None,
    qv_tnd: gtscript.Field["dtype"] = None,
    qc_tnd: gtscript.Field["dtype"] = None,
    qr_tnd: gtscript.Field["dtype"] = None,
    *,
    dt: float,
    dx: float,
    dy: float,
    fluxer: Union[IsentropicHorizontalFlux, IsentropicMinimalHorizontalFlux],
    origin: taz_types.triplet_int_t,
    domain: taz_types.triplet_int_t,
    **kwargs  # catch-all
) -> None:
    i = slice(origin[0], origin[0] + domain[0])
    j = slice(origin[1], origin[1] + domain[1])
    k = slice(origin[2], origin[2] + domain[2])

    ne = fluxer.extent
    ip_f = slice(origin[0] - ne + 1, origin[0] - ne + domain[0] + 1)
    im_f = slice(origin[0] - ne, origin[0] - ne + domain[0])
    jp_f = slice(origin[1] - ne + 1, origin[1] - ne + domain[1] + 1)
    jm_f = slice(origin[1] - ne, origin[1] - ne + domain[1])

    fluxes = fluxer.call(
        s=s_int,
        u=u_int,
        v=v_int,
        su=su_int,
        sv=sv_int,
        mtg=mtg_int,
        sqv=sqv_int,
        sqc=sqc_int,
        sqr=sqr_int,
        s_tnd=s_tnd,
        qv_tnd=qv_tnd,
        qc_tnd=qc_tnd,
        qr_tnd=qr_tnd,
        dt=dt,
        dx=dx,
        dy=dy,
        compute_density_fluxes=True,
        compute_momentum_fluxes=False,
        compute_water_species_fluxes=True,
    )

    flux_s_x, flux_s_y = fluxes[0:2]
    s_new[i, j, k] = s_now[i, j, k] - dt * (
        (flux_s_x[ip_f, j, k] - flux_s_x[im_f, j, k]) / dx
        + (flux_s_y[i, jp_f, k] - flux_s_y[i, jm_f, k]) / dy
        - (s_tnd[i, j, k] if s_tnd is not None else 0.0)
    )

    if len(fluxes) > 2:  # moist
        flux_sqv_x, flux_sqv_y = fluxes[2:4]
        sqv_new[i, j, k] = sqv_now[i, j, k] - dt * (
            (flux_sqv_x[ip_f, j, k] - flux_sqv_x[im_f, j, k]) / dx
            + (flux_sqv_y[i, jp_f, k] - flux_sqv_y[i, jm_f, k]) / dy
            - (s_int[i, j, k] * qv_tnd[i, j, k] if qv_tnd is not None else 0.0)
        )

        flux_sqc_x, flux_sqc_y = fluxes[4:6]
        sqc_new[i, j, k] = sqc_now[i, j, k] - dt * (
            (flux_sqc_x[ip_f, j, k] - flux_sqc_x[im_f, j, k]) / dx
            + (flux_sqc_y[i, jp_f, k] - flux_sqc_y[i, jm_f, k]) / dy
            - (s_int[i, j, k] * qc_tnd[i, j, k] if qc_tnd is not None else 0.0)
        )

        flux_sqr_x, flux_sqr_y = fluxes[6:8]
        sqr_new[i, j, k] = sqr_now[i, j, k] - dt * (
            (flux_sqr_x[ip_f, j, k] - flux_sqr_x[im_f, j, k]) / dx
            + (flux_sqr_y[i, jp_f, k] - flux_sqr_y[i, jm_f, k]) / dy
            - (s_int[i, j, k] * qr_tnd[i, j, k] if qr_tnd is not None else 0.0)
        )


def step_forward_euler_momentum_numpy(
    s_now: gtscript.Field["dtype"],
    s_int: gtscript.Field["dtype"],
    s_new: gtscript.Field["dtype"],
    u_int: gtscript.Field["dtype"],
    v_int: gtscript.Field["dtype"],
    su_now: gtscript.Field["dtype"],
    su_int: gtscript.Field["dtype"],
    su_new: gtscript.Field["dtype"],
    sv_now: gtscript.Field["dtype"],
    sv_int: gtscript.Field["dtype"],
    sv_new: gtscript.Field["dtype"],
    mtg_now: gtscript.Field["dtype"],
    mtg_new: gtscript.Field["dtype"],
    mtg_int: gtscript.Field["dtype"] = None,
    su_tnd: gtscript.Field["dtype"] = None,
    sv_tnd: gtscript.Field["dtype"] = None,
    *,
    dt: float,
    dx: float,
    dy: float,
    eps: float,
    fluxer: Union[IsentropicHorizontalFlux, IsentropicMinimalHorizontalFlux],
    origin: taz_types.triplet_int_t,
    domain: taz_types.triplet_int_t,
    **kwargs  # catch-all
) -> None:
    i = slice(origin[0], origin[0] + domain[0])
    im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
    ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
    j = slice(origin[1], origin[1] + domain[1])
    jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
    jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
    k = slice(origin[2], origin[2] + domain[2])

    ne = fluxer.extent
    ip_f = slice(origin[0] - ne + 1, origin[0] - ne + domain[0] + 1)
    im_f = slice(origin[0] - ne, origin[0] - ne + domain[0])
    jp_f = slice(origin[1] - ne + 1, origin[1] - ne + domain[1] + 1)
    jm_f = slice(origin[1] - ne, origin[1] - ne + domain[1])

    flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = fluxer.call(
        dt=dt,
        dx=dx,
        dy=dy,
        s=s_int,
        u=u_int,
        v=v_int,
        su=su_int,
        sv=sv_int,
        mtg=mtg_int,
        su_tnd=su_tnd,
        sv_tnd=sv_tnd,
        compute_density_fluxes=False,
        compute_momentum_fluxes=True,
        compute_water_species_fluxes=False,
    )

    su_new[i, j, k] = su_now[i, j, k] - dt * (
        (flux_su_x[ip_f, j, k] - flux_su_x[im_f, j, k]) / dx
        + (flux_su_y[i, jp_f, k] - flux_su_y[i, jm_f, k]) / dy
        + (1.0 - eps)
        * s_now[i, j, k]
        * (mtg_now[ip1, j, k] - mtg_now[im1, j, k])
        / (2.0 * dx)
        + eps
        * s_new[i, j, k]
        * (mtg_new[ip1, j, k] - mtg_new[im1, j, k])
        / (2.0 * dx)
        - (su_tnd[i, j, k] if su_tnd is not None else 0.0)
    )

    sv_new[i, j, k] = sv_now[i, j, k] - dt * (
        (flux_sv_x[ip_f, j, k] - flux_sv_x[im_f, j, k]) / dx
        + (flux_sv_y[i, jp_f, k] - flux_sv_y[i, jm_f, k]) / dy
        + (1.0 - eps)
        * s_now[i, j, k]
        * (mtg_now[i, jp1, k] - mtg_now[i, jm1, k])
        / (2.0 * dy)
        + eps
        * s_new[i, j, k]
        * (mtg_new[i, jp1, k] - mtg_new[i, jm1, k])
        / (2.0 * dy)
        - (sv_tnd[i, j, k] if sv_tnd is not None else 0.0)
    )


def step_forward_euler_gt(
    s_now: gtscript.Field["dtype"],
    s_int: gtscript.Field["dtype"],
    s_new: gtscript.Field["dtype"],
    u_int: gtscript.Field["dtype"],
    v_int: gtscript.Field["dtype"],
    su_int: gtscript.Field["dtype"] = None,
    sv_int: gtscript.Field["dtype"] = None,
    mtg_int: gtscript.Field["dtype"] = None,
    sqv_now: gtscript.Field["dtype"] = None,
    sqv_int: gtscript.Field["dtype"] = None,
    sqv_new: gtscript.Field["dtype"] = None,
    sqc_now: gtscript.Field["dtype"] = None,
    sqc_int: gtscript.Field["dtype"] = None,
    sqc_new: gtscript.Field["dtype"] = None,
    sqr_now: gtscript.Field["dtype"] = None,
    sqr_int: gtscript.Field["dtype"] = None,
    sqr_new: gtscript.Field["dtype"] = None,
    s_tnd: gtscript.Field["dtype"] = None,
    qv_tnd: gtscript.Field["dtype"] = None,
    qc_tnd: gtscript.Field["dtype"] = None,
    qr_tnd: gtscript.Field["dtype"] = None,
    *,
    dt: float,
    dx: float,
    dy: float
) -> None:
    from __externals__ import (
        fluxer,
        moist,
        qc_tnd_on,
        qr_tnd_on,
        qv_tnd_on,
        s_tnd_on,
    )

    with computation(PARALLEL), interval(...):
        if __INLINED(not moist):  # compile-time if
            flux_s_x, flux_s_y, _, _, _, _ = fluxer(
                s=s_int,
                u=u_int,
                v=v_int,
                su=su_int,
                sv=sv_int,
                mtg=mtg_int,
                s_tnd=s_tnd,
                dt=dt,
                dx=dx,
                dy=dy,
            )
        else:
            (
                flux_s_x,
                flux_s_y,
                _,
                _,
                _,
                _,
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            ) = fluxer(
                s=s_int,
                u=u_int,
                v=v_int,
                su=su_int,
                sv=sv_int,
                mtg=mtg_int,
                sqv=sqv_int,
                sqc=sqc_int,
                sqr=sqr_int,
                s_tnd=s_tnd,
                qv_tnd=qv_tnd,
                qc_tnd=qc_tnd,
                qr_tnd=qr_tnd,
                dt=dt,
                dx=dx,
                dy=dy,
            )

        if __INLINED(s_tnd_on):  # compile-time if
            s_new = s_now[0, 0, 0] - dt * (
                (flux_s_x[1, 0, 0] - flux_s_x[0, 0, 0]) / dx
                + (flux_s_y[0, 1, 0] - flux_s_y[0, 0, 0]) / dy
                - s_tnd[0, 0, 0]
            )
        else:
            s_new = s_now[0, 0, 0] - dt * (
                (flux_s_x[1, 0, 0] - flux_s_x[0, 0, 0]) / dx
                + (flux_s_y[0, 1, 0] - flux_s_y[0, 0, 0]) / dy
            )

        if __INLINED(moist):  # compile-time if
            if __INLINED(qv_tnd_on):  # compile-time if
                sqv_new = sqv_now[0, 0, 0] - dt * (
                    (flux_sqv_x[1, 0, 0] - flux_sqv_x[0, 0, 0]) / dx
                    + (flux_sqv_y[0, 1, 0] - flux_sqv_y[0, 0, 0]) / dy
                    - s_int[0, 0, 0] * qv_tnd[0, 0, 0]
                )
            else:
                sqv_new = sqv_now[0, 0, 0] - dt * (
                    (flux_sqv_x[1, 0, 0] - flux_sqv_x[0, 0, 0]) / dx
                    + (flux_sqv_y[0, 1, 0] - flux_sqv_y[0, 0, 0]) / dy
                )

            if __INLINED(qc_tnd_on):  # compile-time if
                sqc_new = sqc_now[0, 0, 0] - dt * (
                    (flux_sqc_x[1, 0, 0] - flux_sqc_x[0, 0, 0]) / dx
                    + (flux_sqc_y[0, 1, 0] - flux_sqc_y[0, 0, 0]) / dy
                    - s_int[0, 0, 0] * qc_tnd[0, 0, 0]
                )
            else:
                sqc_new = sqc_now[0, 0, 0] - dt * (
                    (flux_sqc_x[1, 0, 0] - flux_sqc_x[0, 0, 0]) / dx
                    + (flux_sqc_y[0, 1, 0] - flux_sqc_y[0, 0, 0]) / dy
                )

            if __INLINED(qr_tnd_on):  # compile-time if
                sqr_new = sqr_now[0, 0, 0] - dt * (
                    (flux_sqr_x[1, 0, 0] - flux_sqr_x[0, 0, 0]) / dx
                    + (flux_sqr_y[0, 1, 0] - flux_sqr_y[0, 0, 0]) / dy
                    - s_int[0, 0, 0] * qr_tnd[0, 0, 0]
                )
            else:
                sqr_new = sqr_now[0, 0, 0] - dt * (
                    (flux_sqr_x[1, 0, 0] - flux_sqr_x[0, 0, 0]) / dx
                    + (flux_sqr_y[0, 1, 0] - flux_sqr_y[0, 0, 0]) / dy
                )


def step_forward_euler_momentum_gt(
    s_now: gtscript.Field["dtype"],
    s_int: gtscript.Field["dtype"],
    s_new: gtscript.Field["dtype"],
    u_int: gtscript.Field["dtype"],
    v_int: gtscript.Field["dtype"],
    su_now: gtscript.Field["dtype"],
    su_int: gtscript.Field["dtype"],
    su_new: gtscript.Field["dtype"],
    sv_now: gtscript.Field["dtype"],
    sv_int: gtscript.Field["dtype"],
    sv_new: gtscript.Field["dtype"],
    mtg_now: gtscript.Field["dtype"],
    mtg_new: gtscript.Field["dtype"],
    mtg_int: gtscript.Field["dtype"] = None,
    su_tnd: gtscript.Field["dtype"] = None,
    sv_tnd: gtscript.Field["dtype"] = None,
    *,
    dt: float,
    dx: float,
    dy: float,
    eps: float
) -> None:
    from __externals__ import fluxer, moist, su_tnd_on, sv_tnd_on

    with computation(PARALLEL), interval(...):
        _, _, flux_su_x, flux_su_y, flux_sv_x, flux_sv_y = fluxer(
            dt=dt,
            dx=dx,
            dy=dy,
            s=s_int,
            u=u_int,
            v=v_int,
            su=su_int,
            sv=sv_int,
            mtg=mtg_int,
            su_tnd=su_tnd,
            sv_tnd=sv_tnd,
        )

        if __INLINED(su_tnd_on):  # compile-time if
            su_new = su_now[0, 0, 0] - dt * (
                (flux_su_x[1, 0, 0] - flux_su_x[0, 0, 0]) / dx
                + (flux_su_y[0, 1, 0] - flux_su_y[0, 0, 0]) / dy
                + (1.0 - eps)
                * s_now[0, 0, 0]
                * (mtg_now[1, 0, 0] - mtg_now[-1, 0, 0])
                / (2.0 * dx)
                + eps
                * s_new[0, 0, 0]
                * (mtg_new[1, 0, 0] - mtg_new[-1, 0, 0])
                / (2.0 * dx)
                - su_tnd[0, 0, 0]
            )
        else:
            su_new = su_now[0, 0, 0] - dt * (
                (flux_su_x[1, 0, 0] - flux_su_x[0, 0, 0]) / dx
                + (flux_su_y[0, 1, 0] - flux_su_y[0, 0, 0]) / dy
                + (1.0 - eps)
                * s_now[0, 0, 0]
                * (mtg_now[1, 0, 0] - mtg_now[-1, 0, 0])
                / (2.0 * dx)
                + eps
                * s_new[0, 0, 0]
                * (mtg_new[1, 0, 0] - mtg_new[-1, 0, 0])
                / (2.0 * dx)
            )

        if __INLINED(sv_tnd_on):  # compile-time if
            sv_new = sv_now[0, 0, 0] - dt * (
                (flux_sv_x[1, 0, 0] - flux_sv_x[0, 0, 0]) / dx
                + (flux_sv_y[0, 1, 0] - flux_sv_y[0, 0, 0]) / dy
                + (1.0 - eps)
                * s_now[0, 0, 0]
                * (mtg_now[0, 1, 0] - mtg_now[0, -1, 0])
                / (2.0 * dy)
                + eps
                * s_new[0, 0, 0]
                * (mtg_new[0, 1, 0] - mtg_new[0, -1, 0])
                / (2.0 * dy)
                - sv_tnd[0, 0, 0]
            )
        else:
            sv_new = sv_now[0, 0, 0] - dt * (
                (flux_sv_x[1, 0, 0] - flux_sv_x[0, 0, 0]) / dx
                + (flux_sv_y[0, 1, 0] - flux_sv_y[0, 0, 0]) / dy
                + (1.0 - eps)
                * s_now[0, 0, 0]
                * (mtg_now[0, 1, 0] - mtg_now[0, -1, 0])
                / (2.0 * dy)
                + eps
                * s_new[0, 0, 0]
                * (mtg_new[0, 1, 0] - mtg_new[0, -1, 0])
                / (2.0 * dy)
            )