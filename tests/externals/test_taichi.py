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
import taichi as ti

from gt4py import gtscript

from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania import Timer


class Laplacian(StencilFactory):
    @staticmethod
    @stencil_definition(backend="numpy", stencil="lap")
    def laplacian_numpy(in_phi, out_lap, *, origin, domain):
        ib, jb, kb = origin
        ie, je, ke = tuple(o + d for o, d in zip(origin, domain))

        out_lap[ib:ie, jb:je, kb:ke] = (
            -4 * in_phi[ib:ie, jb:je, kb:ke]
            + in_phi[ib - 1 : ie - 1, jb:je, kb:ke]
            + in_phi[ib + 1 : ie + 1, jb:je, kb:ke]
            + in_phi[ib:ie, jb - 1 : je - 1, kb:ke]
            + in_phi[ib:ie, jb + 1 : je + 1, kb:ke]
        )

    @staticmethod
    @stencil_definition(backend="taichi:*", stencil="lap")
    @ti.kernel
    def laplacian_taichi(in_phi: ti.template(), out_lap: ti.template()):
        for i, j, k in out_lap:
            out_lap[i, j, k] = (
                -4 * in_phi[i, j, k]
                + in_phi[i - 1, j, k]
                + in_phi[i + 1, j, k]
                + in_phi[i, j - 1, k]
                + in_phi[i, j + 1, k]
            )

    @staticmethod
    @stencil_definition(backend="gt4py:*", stencil="lap")
    def laplacian_gt4py(
        in_phi: gtscript.Field["dtype"], out_lap: gtscript.Field["dtype"]
    ):
        with computation(PARALLEL), interval(...):
            out_lap = (
                -4 * in_phi[0, 0, 0]
                + in_phi[-1, 0, 0]
                + in_phi[1, 0, 0]
                + in_phi[0, -1, 0]
                + in_phi[0, 1, 0]
            )


def run(backend):
    ti.init(arch=ti.gpu)

    nx = ny = nz = 128
    nt = 1000

    so = StorageOptions(dtype=float)
    bo = BackendOptions(dtypes={"dtype": so.dtype})

    laplacian = Laplacian(backend, bo, so)

    shape = (nx, ny, nz)
    phi = ti.field(float, shape=shape)  # laplacian.zeros(shape=shape)
    lap = ti.field(float, shape=shape)  # laplacian.zeros(shape=shape)

    stencil = laplacian.compile_stencil("lap")

    stencil(phi, lap)

    for _ in range(nt):
        Timer.start(label=backend)
        stencil(phi, lap)
        Timer.stop(label=backend)

    Timer.print(label=backend)


def run_taichi():
    nx = ny = nz = 128
    nt = 1000

    so = StorageOptions(dtype=float)
    bo = BackendOptions(dtypes={"dtype": so.dtype})

    laplacian = Laplacian("taichi:gpu", bo, so)

    ti.init(arch=ti.gpu)

    shape = (nx, ny, nz)
    phi = laplacian.zeros(shape=shape)
    lap = laplacian.zeros(shape=shape)

    @ti.kernel
    def core(in_phi: ti.template(), out_lap: ti.template()):
        for i, j, k in out_lap:
            out_lap[i, j, k] = (
                -4 * in_phi[i, j, k]
                + in_phi[i - 1, j, k]
                + in_phi[i + 1, j, k]
                + in_phi[i, j - 1, k]
                + in_phi[i, j + 1, k]
            )

    core(phi, lap)

    for _ in range(nt):
        Timer.start(label="taichi")
        core(phi, lap)
        Timer.stop(label="taichi")

    Timer.print(label="taichi")


def main():
    import taichi as ti

    ti.init(arch=ti.gpu)

    n = 320
    pixels = ti.field(dtype=float, shape=(n * 2, n, n))

    @ti.func
    def complex_sqr(z):
        return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])

    @ti.kernel
    def paint(t: float):
        for i, j, k in pixels:  # Parallized over all pixels
            c = ti.Vector([-0.8, ti.cos(t) * 0.2])
            z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
            iterations = 0
            while z.norm() < 20 and iterations < 50:
                z = complex_sqr(z) + c
                iterations += 1
            pixels[i, j, k] = 1 - iterations * 0.02

    # gui = ti.GUI("Julia Set", res=(n * 2, n))

    for i in range(1000):
        paint(i * 0.03)


if __name__ == "__main__":
    # ti.init(arch=ti.gpu)
    # run("numpy")
    # run("taichi:gpu")
    # run("gt4py:gtx86")
    run("taichi:gpu")
    # run_taichi()
    # main()
    # run_taichi()
