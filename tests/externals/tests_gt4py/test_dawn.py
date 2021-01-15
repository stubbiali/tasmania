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
import gt4py as gt

try:
    import dawn4py
except ImportError:
    dawn4py = None


assert dawn4py is not None


backend = "dawn:cxxopt"


@gt.gtscript.stencil(backend=backend, verbose=True)
def laplacian(
    in_field: gt.gtscript.Field[float], lap_field: gt.gtscript.Field[float]
):
    with computation(PARALLEL), interval(...):
        lap_field = (
            -4.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, 1, 0]
        )


shape = (32, 32, 16)


in_field = gt.storage.zeros(backend, (0, 0, 0), shape, float)
lap_field = gt.storage.zeros(backend, (0, 0, 0), shape, float)

laplacian(in_field, lap_field, origin=(1, 1, 0))

print("All right!")
