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
from gt4py import gtscript

from tasmania.python.framework.stencil import stencil_definition


@stencil_definition.register(backend=("numpy", "cupy"), stencil="diffusion")
def diffusion_numpy(in_phi, out_phi, *, alpha, origin, domain, **kwargs):
    ib, jb, kb = origin
    ie, je, ke = tuple(origin[i] + domain[i] for i in range(3))

    # compute the laplacian-of-laplacian
    lap = (
        -4 * in_phi[ib - 2 : ie + 2, jb - 2 : je + 2, kb:ke]
        + in_phi[ib - 3 : ie + 1, jb - 2 : je + 2, kb:ke]
        + in_phi[ib - 1 : ie + 3, jb - 2 : je + 2, kb:ke]
        + in_phi[ib - 2 : ie + 2, jb - 3 : je + 1, kb:ke]
        + in_phi[ib - 2 : ie + 2, jb - 1 : je + 3, kb:ke]
    )
    bilap = (
        -4 * lap[1:-1, 1:-1]
        + lap[:-2, 1:-1]
        + lap[2:, 1:-1]
        + lap[1:-1, :-2]
        + lap[1:-1, 2:]
    )

    # compute the x- and y-flux
    flux_x = bilap[1:, 1:-1] - bilap[:-1, 1:-1]
    flux_y = bilap[1:-1, 1:] - bilap[1:-1, :-1]

    # update the field
    out_phi[ib:ie, jb:je, kb:ke] = in_phi[ib:ie, jb:je, kb:ke] + alpha * (
        flux_x[1:, :] - flux_x[:-1, :] + flux_y[:, 1:] - flux_y[:, :-1]
    )


@stencil_definition.register(backend="gt4py*", stencil="diffusion")
def diffusion_gt4py(
    in_phi: gtscript.Field["dtype"],
    out_phi: gtscript.Field["dtype"],
    *,
    alpha: "dtype"
) -> None:
    with computation(PARALLEL), interval(...):
        # compute the laplacian-of-laplacian
        lap = (
            -4 * in_phi[0, 0, 0]
            + in_phi[-1, 0, 0]
            + in_phi[1, 0, 0]
            + in_phi[0, -1, 0]
            + in_phi[0, 1, 0]
        )
        bilap = (
            -4 * lap[0, 0, 0]
            + lap[-1, 0, 0]
            + lap[1, 0, 0]
            + lap[0, -1, 0]
            + lap[0, 1, 0]
        )

        # compute the x- and y-flux
        flux_x = bilap[1, 0, 0] - bilap[0, 0, 0]
        flux_y = bilap[0, 1, 0] - bilap[0, 0, 0]

        # update the field
        out_phi = in_phi + alpha * (
            flux_x[0, 0, 0]
            - flux_x[-1, 0, 0]
            + flux_y[0, 0, 0]
            - flux_y[0, -1, 0]
        )
