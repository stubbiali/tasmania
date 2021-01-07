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


@stencil_definition.register(backend="numpy", stencil="copy")
@stencil_definition.register(backend="cupy", stencil="copy")
@stencil_definition.register(backend="numba:cpu", stencil="copy")
def copy_numpy(src, dst, *, origin, domain):
    ib, jb, kb = origin
    ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
    dst[ib:ie, jb:je, kb:ke] = src[ib:ie, jb:je, kb:ke]


@stencil_definition.register(backend="gt4py*", stencil="copy")
def copy_gt4py(
    src: gtscript.Field["dtype"], dst: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        dst = src


@stencil_definition.register(backend="numpy", stencil="copychange")
@stencil_definition.register(backend="cupy", stencil="copychange")
@stencil_definition.register(backend="numba:cpu", stencil="copychange")
def copychange_numpy(src, dst, *, origin, domain):
    ib, jb, kb = origin
    ie, je, ke = ib + domain[0], jb + domain[1], kb + domain[2]
    dst[ib:ie, jb:je, kb:ke] = -src[ib:ie, jb:je, kb:ke]


@stencil_definition.register(backend="gt4py*", stencil="copychange")
def copychange_gt4py(
    src: gtscript.Field["dtype"], dst: gtscript.Field["dtype"]
) -> None:
    with computation(PARALLEL), interval(...):
        dst = -src
