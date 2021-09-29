# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from gt4py.gtscript import PARALLEL, __INLINED, computation, interval

from tasmania.python.utils.storage import zeros


@gtscript.function
def get_upwind_flux(w, phi):
    flux = w[0, 0, 0] * (
        (w[0, 0, 0] > 0.0) * phi[0, 0, 0] + (w[0, 0, 0] < 0.0) * phi[0, 0, -1]
    )
    return flux


@gtscript.function
def vflux(w, s, su, sv):
    from __externals__ import get_upwind_flux

    flux_s = get_upwind_flux(w=w, phi=s)
    flux_su = get_upwind_flux(w=w, phi=su)
    flux_sv = get_upwind_flux(w=w, phi=sv)
    return flux_s, flux_su, flux_sv


def stencil_defs(
    in_w: gtscript.Field["dtype"],
    in_s: gtscript.Field["dtype"],
    in_su: gtscript.Field["dtype"],
    in_sv: gtscript.Field["dtype"],
    out_s: gtscript.Field["dtype"],
    out_su: gtscript.Field["dtype"],
    out_sv: gtscript.Field["dtype"],
    *,
    dz: float
):
    from __externals__ import get_upwind_flux, vflux, vstaggering

    with computation(PARALLEL), interval(0, 1):
        w = 0.0
    with computation(PARALLEL), interval(1, None):  # ... interval(1, None):
        if __INLINED(vstaggering):
            w = in_w
        else:
            w = 0.5 * (in_w[0, 0, 0] + in_w[0, 0, -1])

    with computation(PARALLEL), interval(1, None):
        flux_s, flux_su, flux_sv = vflux(w=w, s=in_s, su=in_su, sv=in_sv)

    with computation(PARALLEL), interval(0, 1):
        out_s = 0.0
        out_su = 0.0
        out_sv = 0.0

    with computation(PARALLEL), interval(1, -1):
        out_s = (flux_s[0, 0, 1] - flux_s[0, 0, 0]) / dz
        out_su = (flux_su[0, 0, 1] - flux_su[0, 0, 0]) / dz
        out_sv = (flux_sv[0, 0, 1] - flux_sv[0, 0, 0]) / dz

    with computation(PARALLEL), interval(-1, None):
        out_s = 0.0
        out_su = 0.0
        out_sv = 0.0


if __name__ == "__main__":
    gt_powered = True
    backend = "numpy"
    vstaggering = True
    dtype = np.float64

    decorator = gtscript.stencil(
        backend,
        dtypes={"dtype": dtype},
        externals={
            "get_upwind_flux": get_upwind_flux,
            "vflux": vflux,
            "vstaggering": vstaggering,
        },
        rebuild=False,
    )
    stencil = decorator(stencil_defs)

    in_w = zeros(
        (30, 30, 11), gt_powered=gt_powered, backend=backend, dtype=dtype
    )
    in_s = zeros(
        (30, 30, 11), gt_powered=gt_powered, backend=backend, dtype=dtype
    )
    in_su = zeros(
        (30, 30, 11), gt_powered=gt_powered, backend=backend, dtype=dtype
    )
    in_sv = zeros(
        (30, 30, 11), gt_powered=gt_powered, backend=backend, dtype=dtype
    )
    out_s = zeros(
        (30, 30, 11), gt_powered=gt_powered, backend=backend, dtype=dtype
    )
    out_su = zeros(
        (30, 30, 11), gt_powered=gt_powered, backend=backend, dtype=dtype
    )
    out_sv = zeros(
        (30, 30, 11), gt_powered=gt_powered, backend=backend, dtype=dtype
    )
    dz = 1.0

    stencil(
        in_w=in_w,
        in_s=in_s,
        in_su=in_su,
        in_sv=in_sv,
        out_s=out_s,
        out_su=out_su,
        out_sv=out_sv,
        dz=dz,
        origin=(0, 0, 0),
        domain=(30, 30, 10),
    )
