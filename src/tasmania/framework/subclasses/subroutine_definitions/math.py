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

from tasmania.third_party import cupy as cp, gt4py, numba

from tasmania.python.framework.stencil import subroutine_definition


@subroutine_definition.register(backend="numpy", stencil="absolute")
def absolute_numpy(phi):
    return np.abs(phi)


@subroutine_definition.register(backend="numpy", stencil="positive")
def positive_numpy(phi):
    return np.where(phi > 0, phi, 0)


@subroutine_definition.register(backend="numpy", stencil="negative")
def negative_numpy(phi):
    return np.where(phi < 0, -phi, 0)


if cp:

    @subroutine_definition.register(backend="cupy", stencil="absolute")
    def absolute_cupy(phi):
        return cp.abs(phi)

    @subroutine_definition.register(backend="cupy", stencil="positive")
    def positive_cupy(phi):
        return cp.where(phi > 0, phi, 0)

    @subroutine_definition.register(backend="cupy", stencil="negative")
    def negative_cupy(phi):
        return cp.where(phi < 0, -phi, 0)


if gt4py:
    from gt4py import gtscript

    @subroutine_definition.register(backend="gt4py*", stencil="absolute")
    @gtscript.function
    def absolute_gt4py(phi):
        return phi if phi > 0 else -phi

    @subroutine_definition.register(backend="gt4py*", stencil="positive")
    @gtscript.function
    def positive_gt4py(phi):
        return phi if phi > 0 else 0

    @subroutine_definition.register(backend="gt4py*", stencil="negative")
    @gtscript.function
    def negative_gt4py(phi):
        return -phi if phi < 0 else 0


if numba:
    subroutine_definition.register(absolute_numpy, "numba:cpu:numpy", "absolute")
    subroutine_definition.register(positive_numpy, "numba:cpu:numpy", "positive")
    subroutine_definition.register(negative_numpy, "numba:cpu:numpy", "negative")

    @subroutine_definition.register(backend="numba:cpu:stencil", stencil="absolute")
    def absolute_numba_cpu(phi):
        def core_def(field):
            return field[0, 0, 0] if field[0, 0, 0] > 0 else -field[0, 0, 0]

        core = numba.stencil(core_def)

        return core(phi)

    @subroutine_definition.register(backend="numba:cpu:stencil", stencil="positive")
    def positive_numba_cpu(phi):
        def core_def(field):
            return field[0, 0, 0] if field[0, 0, 0] > 0 else 0

        core = numba.stencil(core_def)

        return core(phi)

    @subroutine_definition.register(backend="numba:cpu:stencil", stencil="negative")
    def negative_numba_cpu(phi):
        def core_def(field):
            return -field[0, 0, 0] if field[0, 0, 0] < 0 else 0

        core = numba.stencil(core_def)

        return core(phi)
