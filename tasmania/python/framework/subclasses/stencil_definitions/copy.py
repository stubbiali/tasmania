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
from tasmania.third_party import cupy, gt4py, numba

from tasmania.python.framework.stencil import stencil_definition


@stencil_definition.register(backend="numpy", stencil="copy")
def copy_numpy(src, dst, *, origin, domain):
    ib, jb, kb = origin
    ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
    dst[ib:ie, jb:je, kb:ke] = src[ib:ie, jb:je, kb:ke]


@stencil_definition.register(backend="numpy", stencil="copychange")
def copychange_numpy(src, dst, *, origin, domain):
    ib, jb, kb = origin
    ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
    dst[ib:ie, jb:je, kb:ke] = -src[ib:ie, jb:je, kb:ke]


if True:  # cupy:
    stencil_definition.register(copy_numpy, "cupy", "copy")
    stencil_definition.register(copychange_numpy, "cupy", "copychange")


if gt4py:
    from gt4py import gtscript

    @stencil_definition.register(backend="gt4py*", stencil="copy")
    def copy_gt4py(
        src: gtscript.Field["dtype"], dst: gtscript.Field["dtype"]
    ) -> None:
        with computation(PARALLEL), interval(...):
            dst = src

    @stencil_definition.register(backend="gt4py*", stencil="copychange")
    def copychange_gt4py(
        src: gtscript.Field["dtype"], dst: gtscript.Field["dtype"]
    ) -> None:
        with computation(PARALLEL), interval(...):
            dst = -src


if numba:
    stencil_definition.register(copy_numpy, "numba:cpu:numpy", "copy")
    stencil_definition.register(
        copychange_numpy, "numba:cpu:numpy", "copychange"
    )
