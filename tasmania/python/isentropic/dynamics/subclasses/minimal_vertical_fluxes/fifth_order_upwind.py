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

from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)
from tasmania.python.utils import taz_types
from tasmania.python.utils.framework_utils import register
from tasmania.python.utils.gtscript_utils import absolute


def get_fifth_order_upwind_flux_numpy(
    w: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    flux = w[:, :, 3:-3] / 60.0 * (
        37.0 * (phi[:, :, 2:-4] + phi[:, :, 3:-3])
        - 8.0 * (phi[:, :, 1:-5] + phi[:, :, 4:-2])
        + (phi[:, :, :-6] + phi[:, :, 5:-1])
    ) - np.abs(w[:, :, 3:-3]) / 60.0 * (
        10.0 * (phi[:, :, 2:-4] - phi[:, :, 3:-3])
        - 5.0 * (phi[:, :, 1:-5] - phi[:, :, 4:-2])
        + (phi[:, :, :-6] - phi[:, :, 5:-1])
    )
    return flux


@gtscript.function
def get_fifth_order_upwind_flux(
    w: taz_types.gtfield_t, phi: taz_types.gtfield_t
) -> taz_types.gtfield_t:
    flux = w[0, 0, 0] / 60.0 * (
        37.0 * (phi[0, 0, -1] + phi[0, 0, 0])
        - 8.0 * (phi[0, 0, -2] + phi[0, 0, 1])
        + (phi[0, 0, -3] + phi[0, 0, 2])
    ) - absolute(w)[0, 0, 0] / 60.0 * (
        10.0 * (phi[0, 0, -1] - phi[0, 0, 0])
        - 5.0 * (phi[0, 0, -2] - phi[0, 0, 1])
        + (phi[0, 0, -3] - phi[0, 0, 2])
    )
    return flux


@register(name="fifth_order_upwind")
class FifthOrderUpwind(IsentropicMinimalVerticalFlux):
    """	Fifth-order upwind scheme. """

    extent = 3
    order = 5
    externals = {
        "get_fifth_order_upwind_flux": get_fifth_order_upwind_flux,
        "absolute": absolute,
    }

    def __init__(self, moist, backend):
        super().__init__(moist, backend)

    def call_numpy(
        self,
        dt: float,
        dz: float,
        w: np.ndarray,
        s: np.ndarray,
        su: np.ndarray,
        sv: np.ndarray,
        sqv: Optional[np.ndarray] = None,
        sqc: Optional[np.ndarray] = None,
        sqr: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        flux_s = get_fifth_order_upwind_flux_numpy(w, s)
        flux_su = get_fifth_order_upwind_flux_numpy(w, su)
        flux_sv = get_fifth_order_upwind_flux_numpy(w, sv)

        return_list = [flux_s, flux_su, flux_sv]

        if self.moist:
            flux_sqv = get_fifth_order_upwind_flux_numpy(w, sqv)
            flux_sqc = get_fifth_order_upwind_flux_numpy(w, sqc)
            flux_sqr = get_fifth_order_upwind_flux_numpy(w, sqr)

            return_list += [flux_sqv, flux_sqc, flux_sqr]

        return return_list

    @staticmethod
    @gtscript.function
    def call_gt(
        dt: float,
        dz: float,
        w: taz_types.gtfield_t,
        s: taz_types.gtfield_t,
        su: taz_types.gtfield_t,
        sv: taz_types.gtfield_t,
        sqv: "Optional[taz_types.gtfield_t]" = None,
        sqc: "Optional[taz_types.gtfield_t]" = None,
        sqr: "Optional[taz_types.gtfield_t]" = None,
    ) -> "Tuple[taz_types.gtfield_t, ...]":

        from __externals__ import moist

        flux_s = get_fifth_order_upwind_flux(w=w, phi=s)
        flux_su = get_fifth_order_upwind_flux(w=w, phi=su)
        flux_sv = get_fifth_order_upwind_flux(w=w, phi=sv)

        if __INLINED(not moist):  # compile-time if
            return flux_s, flux_su, flux_sv
        else:
            flux_sqv = get_fifth_order_upwind_flux(w=w, phi=sqv)
            flux_sqc = get_fifth_order_upwind_flux(w=w, phi=sqc)
            flux_sqr = get_fifth_order_upwind_flux(w=w, phi=sqr)

            return flux_s, flux_su, flux_sv, flux_sqv, flux_sqc, flux_sqr
