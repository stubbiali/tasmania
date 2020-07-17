# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2012-2019, ETH Zurich
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
import timeit

try:
    import cupy as cp
except ImportError:
    import numpy as cp


def copy_defs(src: gtscript.Field["dtype"], dst: gtscript.Field["dtype"]):
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
        dst = src


def update_halo(copy_stencil, field, nb):
    nx = field.shape[0] - 2 * nb
    ny = field.shape[1] - 2 * nb
    nz = field.shape[2]

    # bottom edge (without corners)
    copy_stencil(
        src=field,
        dst=field,
        origin={"src": (nb, ny, 0), "dst": (nb, 0, 0)},
        domain=(nx, nb, nz),
    )

    # top edge (without corners)
    copy_stencil(
        src=field,
        dst=field,
        origin={"src": (nb, nb, 0), "dst": (nb, ny + nb, 0)},
        domain=(nx, nb, nz),
    )

    # left edge (including corners)
    copy_stencil(
        src=field,
        dst=field,
        origin={"src": (nx, 0, 0), "dst": (0, 0, 0)},
        domain=(nb, ny + 2 * nb, nz),
    )

    # right edge (including corners)
    copy_stencil(
        src=field,
        dst=field,
        origin={"src": (nb, 0, 0), "dst": (nx + nb, 0, 0)},
        domain=(nb, ny + 2 * nb, nz),
    )


def get_timer():
    try:
        cp.cuda.Device(0).synchronize()
    except AttributeError:
        pass

    return timeit.default_timer()
