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

from gt4py import gtscript

from tasmania.python.framework.stencil_compilers import stencil_definition


@stencil_definition.register(backend=("numpy", "cupy"), stencil="laplacian")
def laplacian(in_field, out_field, *, origin, domain, **kwargs):
    ib, jb, kb = origin
    ie, je, ke = tuple(origin[i] + domain[i] for i in range(3))
    i, j, k = np.arange(ib, ie), np.arange(jb, je), np.arange(kb, ke)

    out_field[i, j, k] = (
        -4.0 * in_field[i, j, k]
        + in_field[i - 1, j, k]
        + in_field[i + 1, j, k]
        + in_field[i, j - 1, k]
        + in_field[i, j + 1, k]
    )


@stencil_definition.register(backend="gt4py*", stencil="laplacian")
def laplacian(
    in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"],
):
    with computation(PARALLEL), interval(...):
        out_field = (
            -4.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[+1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, +1, 0]
        )
