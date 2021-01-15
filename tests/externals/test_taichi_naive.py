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
from numba import cuda
import numpy as np
import taichi as ti

from gt4py import gtscript

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.utils.backend import is_ti, get_ti_arch
from tasmania import Timer


@ti.data_oriented
class TestClass(StencilFactory):
    @staticmethod
    @stencil_definition(backend="taichi:*", stencil="lap")
    def run_taichi(
        in_phi: ti.template(),
        out_lap: ti.template(),
        origin: ti.template(),
        domain: ti.template(),
    ):
        for i, j, k in ti.ndrange(
            (origin[0], origin[0] + domain[0]),
            (origin[1], origin[1] + domain[1]),
            (origin[2], origin[2] + domain[2]),
        ):
            out_lap[i, j, k] = (
                -4 * in_phi[i, j, k]
                + in_phi[i - 1, j, k]
                + in_phi[i + 1, j, k]
                + in_phi[i, j - 1, k]
                + in_phi[i, j + 1, k]
            )

    # @staticmethod
    # @stencil_definition(backend="taichi:*", stencil="lap")
    # def run_taichi(
    #     in_phi,
    #     out_lap,
    #     origin,
    #     domain,
    # ):
    #     @ti.kernel
    #     def core():
    #         for i, j, k in out_lap:
    #             out_lap[i, j, k] = (
    #                 -4 * in_phi[i, j, k]
    #                 + in_phi[i - 1, j, k]
    #                 + in_phi[i + 1, j, k]
    #                 + in_phi[i, j - 1, k]
    #                 + in_phi[i, j + 1, k]
    #             )
    #
    #     core()

    @staticmethod
    @stencil_definition(backend="gt4py:*", stencil="lap")
    def run_gt4py(
        in_phi: gtscript.Field[float], out_lap: gtscript.Field[float]
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
    if is_ti(backend):
        exec(f"ti.init(arch=ti.{get_ti_arch(backend)})")

    so = StorageOptions(dtype=float)
    bo = BackendOptions(dtypes={"dtype": so.dtype})
    tc = TestClass(backend, bo, so)

    nx = ny = nz = 128
    nt = 10000

    shape = (nx, ny, nz)
    phi = tc.zeros(shape=shape)
    lap = tc.zeros(shape=shape)

    stencil = tc.compile("lap")

    for _ in range(nt):
        Timer.start(label=backend)
        stencil(phi, lap, origin=(1, 1, 0), domain=(nx - 2, ny - 2, nz))
        Timer.stop(label=backend)

    Timer.print(label=backend)


def run_taichi(arch):
    exec(f"ti.init(arch=ti.{arch})")

    nx = ny = nz = 128
    nt = 10000

    shape = (nx, ny, nz)
    phi = ti.field(float, shape=shape)
    lap = ti.field(float, shape=shape)

    @ti.kernel
    def stencil(in_phi: ti.template(), out_lap: ti.template()):
        for i, j, k in out_lap:
            out_lap[i, j, k] = (
                -4 * in_phi[i, j, k]
                + in_phi[i - 1, j, k]
                + in_phi[i + 1, j, k]
                + in_phi[i, j - 1, k]
                + in_phi[i, j + 1, k]
            )

    for _ in range(nt):
        Timer.start(label="run")
        stencil(phi, lap)  # , origin=(1, 1, 0), domain=(nx - 2, ny - 2, nz))
        Timer.stop(label="run")

    Timer.print(label="run")


if __name__ == "__main__":
    # run("taichi:cpu")
    # run("gt4py:gtx86")
    # run("gt4py:gtmc")
    # run("gt4py:gtcuda")
    # run_taichi("gpu")

    ti.init(arch=ti.cpu)

    nx = ny = nz = 128
    nt = 1000

    shape = (nx, ny, nz)
    phi = ti.field(float, shape=shape)
    lap = ti.field(float, shape=shape)

    lap_np = np.asarray(lap)

    # @ti.kernel
    # def stencil():
    #     for i, j, k in lap:
    #         lap[i, j, k] = (
    #             -4 * phi[i, j, k]
    #             + phi[i - 1, j, k]
    #             + phi[i + 1, j, k]
    #             + phi[i, j - 1, k]
    #             + phi[i, j + 1, k]
    #         )
    #
    # for _ in range(nt):
    #     Timer.start(label="run")
    #     stencil()
    #     Timer.stop(label="run")
    #
    # Timer.print(label="run")
