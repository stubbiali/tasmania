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
from typing import List, Optional, Tuple

from gt4py import gtscript

from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
    IsentropicHorizontalFlux,
)
from tasmania.python.utils import taz_types
from tasmania.python.utils.framework_utils import register


def get_upwind_flux_x_numpy(u: np.ndarray, phi: np.ndarray) -> np.ndarray:
    flux = u[1:-1, :] * np.where(u[1:-1, :] > 0.0, phi[:-2, :], phi[1:-1, :])
    return flux


def get_upwind_flux_y_numpy(v: np.ndarray, phi: np.ndarray) -> np.ndarray:
    flux = v[:, 1:-1] * np.where(v[:, 1:-1] > 0.0, phi[:, :-2], phi[:, 1:-1])
    return flux


@gtscript.function
def get_upwind_flux_x(
    u: taz_types.gtfield_t, phi: taz_types.gtfield_t
) -> taz_types.gtfield_t:
    flux = u[0, 0, 0] * (phi[-1, 0, 0] if u[0, 0, 0] > 0 else phi[0, 0, 0])
    return flux


@gtscript.function
def get_upwind_flux_y(
    v: taz_types.gtfield_t, phi: taz_types.gtfield_t
) -> taz_types.gtfield_t:
    flux = v[0, 0, 0] * (phi[0, -1, 0] if v[0, 0, 0] > 0 else phi[0, 0, 0])
    return flux


@register(name="upwind")
class Upwind(IsentropicHorizontalFlux):
    """ Upwind scheme. """

    extent = 1
    order = 1
    externals = {
        "get_upwind_flux_x": get_upwind_flux_x,
        "get_upwind_flux_y": get_upwind_flux_y,
    }

    def __init__(self, moist, backend):
        super().__init__(moist, backend)

    def call_numpy(
        self,
        dt: float,
        dx: float,
        dy: float,
        s: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        su: np.ndarray,
        sv: np.ndarray,
        mtg: np.ndarray = None,
        sqv: Optional[np.ndarray] = None,
        sqc: Optional[np.ndarray] = None,
        sqr: Optional[np.ndarray] = None,
        s_tnd: Optional[np.ndarray] = None,
        su_tnd: Optional[np.ndarray] = None,
        sv_tnd: Optional[np.ndarray] = None,
        qv_tnd: Optional[np.ndarray] = None,
        qc_tnd: Optional[np.ndarray] = None,
        qr_tnd: Optional[np.ndarray] = None,
        *,
        compute_density_fluxes: bool = True,
        compute_momentum_fluxes: bool = True,
        compute_water_species_fluxes: bool = True
    ) -> List[np.ndarray]:
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
        dt: float,
        dx: float,
        dy: float,
        s: taz_types.gtfield_t,
        u: taz_types.gtfield_t,
        v: taz_types.gtfield_t,
        su: taz_types.gtfield_t,
        sv: taz_types.gtfield_t,
        mtg: "Optional[taz_types.gtfield_t]" = None,
        sqv: "Optional[taz_types.gtfield_t]" = None,
        sqc: "Optional[taz_types.gtfield_t]" = None,
        sqr: "Optional[taz_types.gtfield_t]" = None,
        s_tnd: "Optional[taz_types.gtfield_t]" = None,
        su_tnd: "Optional[taz_types.gtfield_t]" = None,
        sv_tnd: "Optional[taz_types.gtfield_t]" = None,
        qv_tnd: "Optional[taz_types.gtfield_t]" = None,
        qc_tnd: "Optional[taz_types.gtfield_t]" = None,
        qr_tnd: "Optional[taz_types.gtfield_t]" = None,
    ) -> "Tuple[taz_types.gtfield_t, ...]":
        from __externals__ import moist

        # compute fluxes for the isentropic density and the momenta
        flux_s_x = get_upwind_flux_x(u=u, phi=s)
        flux_s_y = get_upwind_flux_y(v=v, phi=s)
        flux_su_x = get_upwind_flux_x(u=u, phi=su)
        flux_su_y = get_upwind_flux_y(v=v, phi=su)
        flux_sv_x = get_upwind_flux_x(u=u, phi=sv)
        flux_sv_y = get_upwind_flux_y(v=v, phi=sv)

        if __INLINED(not moist):  # compile-time if
            return (
                flux_s_x,
                flux_s_y,
                flux_su_x,
                flux_su_y,
                flux_sv_x,
                flux_sv_y,
            )
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


