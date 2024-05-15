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
import numpy as np

from gt4py import gtscript

from tasmania.python.framework.tag import subroutine_definition
from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicMinimalVerticalFlux,
)


def get_upwind_flux_numpy(w, phi):
    flux = w[:, :, 1:-1] * np.where(
        w[:, :, 1:-1] > 0.0, phi[:, :, 1:], phi[:, :, :-1]
    )
    return flux


@gtscript.function
def get_upwind_flux_gt4py(w, phi):
    flux = w[0, 0, 0] * (phi[0, 0, 0] if w[0, 0, 0] > 0 else phi[0, 0, -1])
    return flux


class Upwind(IsentropicMinimalVerticalFlux):
    """Upwind scheme."""

    name = "upwind"
    extent = 1
    order = 1
    externals = {"get_upwind_flux_gt4py": get_upwind_flux_gt4py}

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy"), stencil="flux_dry")
    def flux_dry_numpy(dt, dz, w, s, su, sv):
        flux_s = get_upwind_flux_numpy(w, s)
        flux_su = get_upwind_flux_numpy(w, su)
        flux_sv = get_upwind_flux_numpy(w, sv)
        return flux_s, flux_su, flux_sv

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy"), stencil="flux_moist")
    def flux_moist_numpy(dt, dz, w, sqv, sqc, sqr):
        flux_sqv = get_upwind_flux_numpy(w, sqv)
        flux_sqc = get_upwind_flux_numpy(w, sqc)
        flux_sqr = get_upwind_flux_numpy(w, sqr)
        return flux_sqv, flux_sqc, flux_sqr

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux_dry")
    @gtscript.function
    def flux_dry_gt4py(dt, dz, w, s, su, sv):
        flux_s = get_upwind_flux_gt4py(w=w, phi=s)
        flux_su = get_upwind_flux_gt4py(w=w, phi=su)
        flux_sv = get_upwind_flux_gt4py(w=w, phi=sv)
        return flux_s, flux_su, flux_sv

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux_moist")
    @gtscript.function
    def flux_moist_gt4py(dt, dz, w, sqv, sqc, sqr):
        flux_sqv = get_upwind_flux_gt4py(w=w, phi=sqv)
        flux_sqc = get_upwind_flux_gt4py(w=w, phi=sqc)
        flux_sqr = get_upwind_flux_gt4py(w=w, phi=sqr)
        return flux_sqv, flux_sqc, flux_sqr
