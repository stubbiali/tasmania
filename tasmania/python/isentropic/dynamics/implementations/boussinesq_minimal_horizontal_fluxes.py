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
"""
This module contains:
    Upwind(IsentropicBoussinesqMinimalHorizontalFlux)
    Centered(IsentropicBoussinesqMinimalHorizontalFlux)
    MacCormack(IsentropicBoussinesqMinimalHorizontalFlux)
    FifthOrderUpwind(IsentropicBoussinesqMinimalHorizontalFlux)
"""
from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicBoussinesqMinimalHorizontalFlux,
)
from tasmania.python.isentropic.dynamics.implementations.horizontal_fluxes import (
    get_centered_flux_x,
    get_centered_flux_y,
    get_fifth_order_upwind_flux_x,
    get_fifth_order_upwind_flux_y,
    get_third_order_upwind_flux_x,
    get_third_order_upwind_flux_y,
    get_upwind_flux_x,
    get_upwind_flux_y,
)
from tasmania.python.isentropic.dynamics.implementations.minimal_horizontal_fluxes import (
    Upwind as CoreUpwind,
    Centered as CoreCentered,
    ThirdOrderUpwind as CoreThirdOrderUpwind,
    FifthOrderUpwind as CoreFifthOrderUpwind,
)


class Upwind(IsentropicBoussinesqMinimalHorizontalFlux):
    """
    Upwind scheme.
    """

    extent = 1
    order = 1

    def __init__(self, grid, moist):
        super().__init__(grid, moist)
        self._core = CoreUpwind(grid, moist)

    def __call__(
        self,
        i,
        j,
        dt,
        s,
        u,
        v,
        su,
        sv,
        ddmtg,
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
        # compute fluxes for the isentropic density, the momenta and
        # the water constituents
        return_list = self._core(
            i,
            j,
            dt,
            s,
            u,
            v,
            su,
            sv,
            sqv,
            sqc,
            sqr,
            s_tnd,
            su_tnd,
            sv_tnd,
            qv_tnd,
            qc_tnd,
            qr_tnd,
        )

        # compute fluxes for the derivative of the Montgomery potential
        return_list.insert(6, get_upwind_flux_x(i, j, u, ddmtg))
        return_list.insert(7, get_upwind_flux_y(i, j, v, ddmtg))

        return return_list


class Centered(IsentropicBoussinesqMinimalHorizontalFlux):
    """
    Centered scheme.
    """

    extent = 1
    order = 2

    def __init__(self, grid, moist):
        super().__init__(grid, moist)
        self._core = CoreCentered(grid, moist)

    def __call__(
        self,
        i,
        j,
        dt,
        s,
        u,
        v,
        su,
        sv,
        ddmtg,
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
        # compute fluxes for the isentropic density, the momenta and
        # the water constituents
        return_list = self._core(
            i,
            j,
            dt,
            s,
            u,
            v,
            su,
            sv,
            sqv,
            sqc,
            sqr,
            s_tnd,
            su_tnd,
            sv_tnd,
            qv_tnd,
            qc_tnd,
            qr_tnd,
        )

        # compute fluxes for the derivative of the Montgomery potential
        return_list.insert(6, get_centered_flux_x(i, j, u, ddmtg))
        return_list.insert(7, get_centered_flux_y(i, j, v, ddmtg))

        return return_list


class ThirdOrderUpwind(IsentropicBoussinesqMinimalHorizontalFlux):
    """
    Third-order upwind scheme.
    """

    extent = 2
    order = 3

    def __init__(self, grid, moist):
        super().__init__(grid, moist)
        self._core = CoreThirdOrderUpwind(grid, moist)

    def __call__(
        self,
        i,
        j,
        dt,
        s,
        u,
        v,
        su,
        sv,
        ddmtg,
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
        # compute fluxes for the isentropic density, the momenta and
        # the water constituents
        return_list = self._core(
            i,
            j,
            dt,
            s,
            u,
            v,
            su,
            sv,
            sqv,
            sqc,
            sqr,
            s_tnd,
            su_tnd,
            sv_tnd,
            qv_tnd,
            qc_tnd,
            qr_tnd,
        )

        # compute fluxes for the derivative of the Montgomery potential
        return_list.insert(6, get_third_order_upwind_flux_x(i, j, u, ddmtg))
        return_list.insert(7, get_third_order_upwind_flux_y(i, j, v, ddmtg))

        return return_list


class FifthOrderUpwind(IsentropicBoussinesqMinimalHorizontalFlux):
    """
    Fifth-order upwind scheme.
    """

    extent = 3
    order = 5

    def __init__(self, grid, moist):
        super().__init__(grid, moist)
        self._core = CoreFifthOrderUpwind(grid, moist)

    def __call__(
        self,
        i,
        j,
        dt,
        s,
        u,
        v,
        su,
        sv,
        ddmtg,
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
        # compute fluxes for the isentropic density, the momenta and
        # the water constituents
        return_list = self._core(
            i,
            j,
            dt,
            s,
            u,
            v,
            su,
            sv,
            sqv,
            sqc,
            sqr,
            s_tnd,
            su_tnd,
            sv_tnd,
            qv_tnd,
            qc_tnd,
            qr_tnd,
        )

        # compute fluxes for the derivative of the Montgomery potential
        return_list.insert(6, get_fifth_order_upwind_flux_x(i, j, u, ddmtg))
        return_list.insert(7, get_fifth_order_upwind_flux_y(i, j, v, ddmtg))

        return return_list
