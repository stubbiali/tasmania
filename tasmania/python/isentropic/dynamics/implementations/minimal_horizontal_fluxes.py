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
from gt4py import gtscript, __externals__

from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicMinimalHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.implementations.horizontal_fluxes import (
    get_centered_flux_x,
    get_centered_flux_x_numpy,
    get_centered_flux_y,
    get_centered_flux_y_numpy,
    get_fifth_order_upwind_flux_x,
    get_fifth_order_upwind_flux_x_numpy,
    get_fifth_order_upwind_flux_y,
    get_fifth_order_upwind_flux_y_numpy,
    get_fourth_order_centered_flux_x,
    get_fourth_order_centered_flux_y,
    get_sixth_order_centered_flux_x,
    get_sixth_order_centered_flux_y,
    get_third_order_upwind_flux_x,
    get_third_order_upwind_flux_x_numpy,
    get_third_order_upwind_flux_y,
    get_third_order_upwind_flux_y_numpy,
    get_upwind_flux_x,
    get_upwind_flux_x_numpy,
    get_upwind_flux_y,
    get_upwind_flux_y_numpy,
)


class Upwind(IsentropicMinimalHorizontalFlux):
    """	Upwind scheme. """

    extent = 1
    order = 1
    externals = {
        "get_upwind_flux_x": get_upwind_flux_x,
        "get_upwind_flux_y": get_upwind_flux_y,
    }

    def __init__(self, moist, gt_powered):
        super().__init__(moist, gt_powered)

    def call_numpy(
        self,
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
        *,
        compute_density_fluxes=True,
        compute_momentum_fluxes=True,
        compute_water_species_fluxes=True
    ):
        return_list = []

        if compute_density_fluxes:
            flux_s_x = get_upwind_flux_x_numpy(u, s)
            flux_s_y = get_upwind_flux_y_numpy(v, s)

            return_list += [flux_s_x, flux_s_y]

        if compute_momentum_fluxes:
            flux_su_x = get_upwind_flux_x_numpy(u, su)
            flux_su_y = get_upwind_flux_y_numpy(v, su)
            flux_sv_x = get_upwind_flux_x_numpy(u, sv)
            flux_sv_y = get_upwind_flux_y_numpy(v, sv)

            return_list += [flux_su_x, flux_su_y, flux_sv_x, flux_sv_y]

        if self.moist and compute_water_species_fluxes:
            flux_sqv_x = get_upwind_flux_x_numpy(u, sqv)
            flux_sqv_y = get_upwind_flux_y_numpy(v, sqv)
            flux_sqc_x = get_upwind_flux_x_numpy(u, sqc)
            flux_sqc_y = get_upwind_flux_y_numpy(v, sqc)
            flux_sqr_x = get_upwind_flux_x_numpy(u, sqr)
            flux_sqr_y = get_upwind_flux_y_numpy(v, sqr)

            return_list += [
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            ]

        return return_list

    @staticmethod
    @gtscript.function
    def call_gt(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
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
        from __externals__ import moist

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_upwind_flux_x(u=u, phi=s)
        flux_s_y = get_upwind_flux_y(v=v, phi=s)
        flux_su_x = get_upwind_flux_x(u=u, phi=su)
        flux_su_y = get_upwind_flux_y(v=v, phi=su)
        flux_sv_x = get_upwind_flux_x(u=u, phi=sv)
        flux_sv_y = get_upwind_flux_y(v=v, phi=sv)

        if __INLINED(not moist):  # compile-time if
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


class Centered(IsentropicMinimalHorizontalFlux):
    """ Centered scheme. """

    extent = 1
    order = 2
    externals = {
        "get_centered_flux_x": get_centered_flux_x,
        "get_centered_flux_y": get_centered_flux_y,
    }

    def __init__(self, moist, gt_powered):
        super().__init__(moist, gt_powered)

    def call_numpy(
        self,
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
        *,
        compute_density_fluxes=True,
        compute_momentum_fluxes=True,
        compute_water_species_fluxes=True
    ):
        return_list = []

        if compute_density_fluxes:
            flux_s_x = get_centered_flux_x_numpy(u, s)
            flux_s_y = get_centered_flux_y_numpy(v, s)

            return_list += [flux_s_x, flux_s_y]

        if compute_momentum_fluxes:
            flux_su_x = get_centered_flux_x_numpy(u, su)
            flux_su_y = get_centered_flux_y_numpy(v, su)
            flux_sv_x = get_centered_flux_x_numpy(u, sv)
            flux_sv_y = get_centered_flux_y_numpy(v, sv)

            return_list += [flux_su_x, flux_su_y, flux_sv_x, flux_sv_y]

        if self.moist and compute_water_species_fluxes:
            flux_sqv_x = get_centered_flux_x_numpy(u, sqv)
            flux_sqv_y = get_centered_flux_y_numpy(v, sqv)
            flux_sqc_x = get_centered_flux_x_numpy(u, sqc)
            flux_sqc_y = get_centered_flux_y_numpy(v, sqc)
            flux_sqr_x = get_centered_flux_x_numpy(u, sqr)
            flux_sqr_y = get_centered_flux_y_numpy(v, sqr)

            return_list += [
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            ]

        return return_list

    @staticmethod
    @gtscript.function
    def call_gt(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
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
        from __externals__ import moist

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_centered_flux_x(u=u, phi=s)
        flux_s_y = get_centered_flux_y(v=v, phi=s)
        flux_su_x = get_centered_flux_x(u=u, phi=su)
        flux_su_y = get_centered_flux_y(v=v, phi=su)
        flux_sv_x = get_centered_flux_x(u=u, phi=sv)
        flux_sv_y = get_centered_flux_y(v=v, phi=sv)

        if __INLINED(not moist):  # compile-time if
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


class ThirdOrderUpwind(IsentropicMinimalHorizontalFlux):
    """	Third-order upwind scheme. """

    extent = 2
    order = 3
    externals = {
        "get_fourth_order_centered_flux_x": get_fourth_order_centered_flux_x,
        "get_third_order_upwind_flux_x": get_third_order_upwind_flux_x,
        "get_fourth_order_centered_flux_y": get_fourth_order_centered_flux_y,
        "get_third_order_upwind_flux_y": get_third_order_upwind_flux_y,
    }

    def __init__(self, moist, gt_powered):
        super().__init__(moist, gt_powered)

    def call_numpy(
        self,
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
        *,
        compute_density_fluxes=True,
        compute_momentum_fluxes=True,
        compute_water_species_fluxes=True
    ):
        return_list = []

        if compute_density_fluxes:
            flux_s_x = get_third_order_upwind_flux_x_numpy(u, s)
            flux_s_y = get_third_order_upwind_flux_y_numpy(v, s)

            return_list += [flux_s_x, flux_s_y]

        if compute_momentum_fluxes:
            flux_su_x = get_third_order_upwind_flux_x_numpy(u, su)
            flux_su_y = get_third_order_upwind_flux_y_numpy(v, su)
            flux_sv_x = get_third_order_upwind_flux_x_numpy(u, sv)
            flux_sv_y = get_third_order_upwind_flux_y_numpy(v, sv)

            return_list += [flux_su_x, flux_su_y, flux_sv_x, flux_sv_y]

        if self.moist and compute_water_species_fluxes:
            flux_sqv_x = get_third_order_upwind_flux_x_numpy(u, sqv)
            flux_sqv_y = get_third_order_upwind_flux_y_numpy(v, sqv)
            flux_sqc_x = get_third_order_upwind_flux_x_numpy(u, sqc)
            flux_sqc_y = get_third_order_upwind_flux_y_numpy(v, sqc)
            flux_sqr_x = get_third_order_upwind_flux_x_numpy(u, sqr)
            flux_sqr_y = get_third_order_upwind_flux_y_numpy(v, sqr)

            return_list += [
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            ]

        return return_list

    @staticmethod
    @gtscript.function
    def call_gt(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
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
        from __externals__ import moist

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_third_order_upwind_flux_x(u=u, phi=s)
        flux_s_y = get_third_order_upwind_flux_y(v=v, phi=s)
        flux_su_x = get_third_order_upwind_flux_x(u=u, phi=su)
        flux_su_y = get_third_order_upwind_flux_y(v=v, phi=su)
        flux_sv_x = get_third_order_upwind_flux_x(u=u, phi=sv)
        flux_sv_y = get_third_order_upwind_flux_y(v=v, phi=sv)

        if __INLINED(not moist):  # compile-time if
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


class FifthOrderUpwind(IsentropicMinimalHorizontalFlux):
    """ Fifth-order upwind scheme. """

    extent = 3
    order = 5
    externals = {
        "get_sixth_order_centered_flux_x": get_sixth_order_centered_flux_x,
        "get_fifth_order_upwind_flux_x": get_fifth_order_upwind_flux_x,
        "get_sixth_order_centered_flux_y": get_sixth_order_centered_flux_y,
        "get_fifth_order_upwind_flux_y": get_fifth_order_upwind_flux_y,
    }

    def __init__(self, moist, gt_powered):
        super().__init__(moist, gt_powered)

    def call_numpy(
        self,
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
        sqv=None,
        sqc=None,
        sqr=None,
        s_tnd=None,
        su_tnd=None,
        sv_tnd=None,
        qv_tnd=None,
        qc_tnd=None,
        qr_tnd=None,
        *,
        compute_density_fluxes=True,
        compute_momentum_fluxes=True,
        compute_water_species_fluxes=True
    ):
        return_list = []

        if compute_density_fluxes:
            flux_s_x = get_fifth_order_upwind_flux_x_numpy(u, s)
            flux_s_y = get_fifth_order_upwind_flux_y_numpy(v, s)

            return_list += [flux_s_x, flux_s_y]

        if compute_momentum_fluxes:
            flux_su_x = get_fifth_order_upwind_flux_x_numpy(u, su)
            flux_su_y = get_fifth_order_upwind_flux_y_numpy(v, su)
            flux_sv_x = get_fifth_order_upwind_flux_x_numpy(u, sv)
            flux_sv_y = get_fifth_order_upwind_flux_y_numpy(v, sv)

            return_list += [flux_su_x, flux_su_y, flux_sv_x, flux_sv_y]

        if self.moist and compute_water_species_fluxes:
            flux_sqv_x = get_fifth_order_upwind_flux_x_numpy(u, sqv)
            flux_sqv_y = get_fifth_order_upwind_flux_y_numpy(v, sqv)
            flux_sqc_x = get_fifth_order_upwind_flux_x_numpy(u, sqc)
            flux_sqc_y = get_fifth_order_upwind_flux_y_numpy(v, sqc)
            flux_sqr_x = get_fifth_order_upwind_flux_x_numpy(u, sqr)
            flux_sqr_y = get_fifth_order_upwind_flux_y_numpy(v, sqr)

            return_list += [
                flux_sqv_x,
                flux_sqv_y,
                flux_sqc_x,
                flux_sqc_y,
                flux_sqr_x,
                flux_sqr_y,
            ]

        return return_list

    @staticmethod
    @gtscript.function
    def call_gt(
        dt,
        dx,
        dy,
        s,
        u,
        v,
        su,
        sv,
        mtg=None,
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
        from __externals__ import moist

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_fifth_order_upwind_flux_x(u=u, phi=s)
        flux_s_y = get_fifth_order_upwind_flux_y(v=v, phi=s)
        flux_su_x = get_fifth_order_upwind_flux_x(u=u, phi=su)
        flux_su_y = get_fifth_order_upwind_flux_y(v=v, phi=su)
        flux_sv_x = get_fifth_order_upwind_flux_x(u=u, phi=sv)
        flux_sv_y = get_fifth_order_upwind_flux_y(v=v, phi=sv)

        if __INLINED(not moist):  # compile-time if
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
