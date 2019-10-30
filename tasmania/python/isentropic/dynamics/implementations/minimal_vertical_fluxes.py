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
from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)


def get_upwind_flux(w, phi):
    flux = w[0, 0, 0] * (
        (w[0, 0, 0] > 0.0) * phi[0, 0, 0] + (w[0, 0, 0] < 0.0) * phi[0, 0, -1]
    )
    return flux


class Upwind(IsentropicMinimalVerticalFlux):
    """ Upwind scheme. """

    extent = 1
    order = 1
    externals = {"get_upwind_flux": get_upwind_flux}

    @staticmethod
    def __call__(dt, dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
    # def __call__(dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
        flux_s = get_upwind_flux(w=w, phi=s)
        flux_su = get_upwind_flux(w=w, phi=su)
        flux_sv = get_upwind_flux(w=w, phi=sv)

        if not moist:
            return flux_s, flux_su, flux_sv
        else:
            flux_sqv = get_upwind_flux(w=w, phi=sqv)
            flux_sqc = get_upwind_flux(w=w, phi=sqc)
            flux_sqr = get_upwind_flux(w=w, phi=sqr)

            return flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr


def get_centered_flux(w, phi):
    flux = w[0, 0, 0] * 0.5 * (phi[0, 0, 0] + phi[0, 0, -1])
    return flux


class Centered(IsentropicMinimalVerticalFlux):
    """	Centered scheme. """

    extent = 1
    order = 2
    externals = {"get_centered_flux": get_centered_flux}

    @staticmethod
    def __call__(dt, dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
    # def __call__(dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
        flux_s = get_centered_flux(w=w, phi=s)
        flux_su = get_centered_flux(w=w, phi=su)
        flux_sv = get_centered_flux(w=w, phi=sv)

        if not moist:
            return flux_s, flux_su, flux_sv
        else:
            flux_sqv = get_centered_flux(w=w, phi=sqv)
            flux_sqc = get_centered_flux(w=w, phi=sqc)
            flux_sqr = get_centered_flux(w=w, phi=sqr)

            return flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr


def get_third_order_upwind_flux(w, phi):
    flux = w[0, 0, 0] / 12.0 * (
        7.0 * (phi[0, 0, -1] + phi[0, 0, 0]) - 1.0 * (phi[0, 0, -2] + phi[0, 0, 1])
    ) - (w[0, 0, 0] * (w[0, 0, 0] > 0.0) - w[0, 0, 0] * (w[0, 0, 0] < 0.0)) / 12.0 * (
        3.0 * (phi[0, 0, -1] - phi[0, 0, 0]) - 1.0 * (phi[0, 0, -2] - phi[0, 0, 1])
    )
    return flux


class ThirdOrderUpwind(IsentropicMinimalVerticalFlux):
    """	Third-order upwind scheme. """

    extent = 2
    order = 3
    externals = {"get_third_order_upwind_flux": get_third_order_upwind_flux}

    @staticmethod
    def __call__(dt, dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
    # def __call__(dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
        flux_s = get_third_order_upwind_flux(w=w, phi=s)
        flux_su = get_third_order_upwind_flux(w=w, phi=su)
        flux_sv = get_third_order_upwind_flux(w=w, phi=sv)

        if not moist:
            return flux_s, flux_su, flux_sv
        else:
            flux_sqv = get_third_order_upwind_flux(w=w, phi=sqv)
            flux_sqc = get_third_order_upwind_flux(w=w, phi=sqc)
            flux_sqr = get_third_order_upwind_flux(w=w, phi=sqr)

            return flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr


def get_fifth_order_upwind_flux(w, phi):
    flux = w[0, 0, 0] / 60.0 * (
        37.0 * (phi[0, 0, -1] + phi[0, 0, 0])
        - 8.0 * (phi[0, 0, -2] + phi[0, 0, 1])
        + 1.0 * (phi[0, 0, -3] + phi[0, 0, 2])
    ) - (w[0, 0, 0] * (w[0, 0, 0] > 0.0) - w[0, 0, 0] * (w[0, 0, 0] < 0.0)) / 60.0 * (
        10.0 * (phi[0, 0, -1] - phi[0, 0, 0])
        - 5.0 * (phi[0, 0, -2] - phi[0, 0, 1])
        + 1.0 * (phi[0, 0, -3] - phi[0, 0, 2])
    )
    return flux


class FifthOrderUpwind(IsentropicMinimalVerticalFlux):
    """	Fifth-order upwind scheme. """

    extent = 3
    order = 5
    externals = {"get_fifth_order_upwind_flux": get_fifth_order_upwind_flux}

    @staticmethod
    def __call__(dt, dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
    # def __call__(dz, w, s, su, sv, sqv=None, sqc=None, sqr=None):
        flux_s = get_fifth_order_upwind_flux(w=w, phi=s)
        flux_su = get_fifth_order_upwind_flux(w=w, phi=su)
        flux_sv = get_fifth_order_upwind_flux(w=w, phi=sv)

        if not moist:
            return flux_s, flux_su, flux_sv
        else:
            flux_sqv = get_fifth_order_upwind_flux(w=w, phi=sqv)
            flux_sqc = get_fifth_order_upwind_flux(w=w, phi=sqc)
            flux_sqr = get_fifth_order_upwind_flux(w=w, phi=sqr)

            return flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr
