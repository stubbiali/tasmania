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
import numba
from numba import cuda
import numpy as np
import pytest

from gt4py import gtscript

from tasmania.python.framework.allocators import zeros
from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania import Timer


def laplacian_numpy(phi):
    return (
        -4 * phi[1:-1, 1:-1]
        + phi[:-2, 1:-1]
        + phi[2:, 1:-1]
        + phi[1:-1, :-2]
        + phi[1:-1, 2:]
    )


def laplacian_numba_def(phi, out):
    def core(field):
        return (
            -4 * field[0, 0, 0]
            + field[-1, 0, 0]
            + field[1, 0, 0]
            + field[0, -1, 0]
            + field[0, 1, 0]
        )

    numba.stencil(core)(phi, out=out)
    # return out


def test_laplacian():
    nx, ny, nz = 32, 32, 32

    phi = np.random.rand(nx, ny, nz)
    out = np.zeros_like(phi)

    laplacian_numba = numba.njit(laplacian_numba_def)
    laplacian_numba(phi, out)

    out_val = laplacian_numpy(phi)

    np.testing.assert_allclose(out[1:-1, 1:-1], out_val)


def lapoflap_numpy(phi):
    lap = laplacian_numpy(phi)
    out = laplacian_numpy(lap)
    return out


def lapoflap_numba_def(phi, out):
    def core1(field):
        return (
            -4 * field[0, 0, 0]
            + field[-1, 0, 0]
            + field[1, 0, 0]
            + field[0, -1, 0]
            + field[0, 1, 0]
        )

    def core2(field):
        return (
            -4 * field[0, 0, 0]
            + field[-1, 0, 0]
            + field[1, 0, 0]
            + field[0, -1, 0]
            + field[0, 1, 0]
        )

    core_stencil = numba.stencil(core1)
    lap = core_stencil(phi)
    core_stencil = numba.stencil(core2)
    out = core_stencil(lap, out=out)
    # return out


def test_lapoflap():
    nx, ny, nz = 32, 32, 32

    phi = np.random.rand(nx, ny, nz)
    out = np.zeros_like(phi)

    lapoflap_numba = numba.njit(lapoflap_numba_def)
    lapoflap_numba(phi, out)

    out_val = lapoflap_numpy(phi)

    np.testing.assert_allclose(out[2:-2, 2:-2], out_val)


def test_performance():
    nx = ny = nz = 128
    nt = 100

    shape = (nx, ny, nz)
    bo = BackendOptions(
        cache=True,
        check_rebuild=False,
        nopython=True,
        parallel=True,
        rebuild=False,
        validate_args=False,
    )
    so = StorageOptions(dtype=float)
    gamma = np.random.rand(*shape)

    def run(backend, stencil):
        sf = StencilFactory(backend, bo, so)

        in_gamma = sf.zeros(shape=shape)
        in_phi_ref = sf.zeros(shape=shape)
        inout_phi = sf.zeros(shape=shape)
        in_gamma[...] = sf.asarray()(gamma)

        dtype = sf.storage_options.dtype
        sf.backend_options.dtypes = {"dtype": dtype}
        obj = sf.compile_stencil(stencil)

        Timer.start(label=backend + "-" + stencil)
        for _ in range(nt):
            obj(
                in_gamma,
                in_phi_ref,
                inout_phi,
                origin=(0, 0, 0),
                domain=(shape[0], shape[1], shape[2]),
                validate_args=bo.validate_args,
            )
            # assert np.allclose(inout_phi[:, :, 10:], 0.0)
        Timer.stop()
        Timer.print(label=backend + "-" + stencil, units="ms")

    run("numpy", "irelax")
    run("gt4py:gtx86", "irelax")
    run("numba:cpu", "irelax")
    run("cupy", "irelax")
    run("gt4py:gtcuda", "irelax")


def test_gpu():
    class Laplacian(StencilFactory):
        @staticmethod
        @stencil_definition(backend="numba:cpu", stencil="lap")
        def laplacian_numba_cpu(phi, lap):
            def core_def(phi):
                return (
                    -4 * phi[0, 0, 0]
                    + phi[-1, 0, 0]
                    + phi[1, 0, 0]
                    + phi[0, -1, 0]
                    + phi[0, 1, 0]
                    + 1
                )

            core = numba.stencil(core_def)
            core(phi, out=lap)

        @staticmethod
        @stencil_definition(backend="numba:gpu", stencil="lap")
        def laplacian_numba_gpu(phi, lap, halo):
            mi, mj, mk = phi.shape
            i, j, k = cuda.grid(3)
            if (
                halo[0] <= i < mi - halo[0]
                and halo[1] <= j < mj - halo[1]
                and k < mk
            ):
                lap[i, j, k] = (
                    -4 * phi[i, j, k]
                    + phi[i - 1, j, k]
                    + phi[i + 1, j, k]
                    + phi[i, j - 1, k]
                    + phi[i, j + 1, k]
                    + 1
                )

        @staticmethod
        @stencil_definition(backend="gt4py*", stencil="lap")
        def laplacian_gt4py(
            phi: gtscript.Field["dtype"], lap: gtscript.Field["dtype"]
        ):
            with computation(PARALLEL), interval(...):
                lap = (
                    -4.0 * phi[0, 0, 0]
                    + phi[-1, 0, 0]
                    + phi[1, 0, 0]
                    + phi[0, -1, 0]
                    + phi[0, 1, 0]
                    + 1
                )

    nx = ny = nz = 512
    nt = 100
    shape = (nx, ny, nz)
    so = StorageOptions(dtype=float)

    bo = BackendOptions(cache=True, check_rebuild=False)
    sf = Laplacian("numba:cpu", backend_options=bo, storage_options=so)

    phi = sf.zeros(shape=shape)
    lap = sf.zeros(shape=shape)

    laplacian_numba_cpu = sf.compile_stencil("lap")

    # Timer.start(label="numba-cpu")
    # for _ in range(nt):
    #     laplacian_numba_cpu(phi, lap)
    # Timer.stop()
    # Timer.print(label="numba-cpu")

    sf = Laplacian("numba:gpu", storage_options=so)

    phi = sf.zeros(shape=shape)
    lap = sf.zeros(shape=shape)

    laplacian_numba = cuda.jit(Laplacian.laplacian_numba_gpu)
    threadsperblock = (8, 8, 8)
    blockspergrid = tuple(
        (n + tpb - 1) // tpb for n, tpb in zip(shape, threadsperblock)
    )

    Timer.start(label="numba-gpu")
    for _ in range(nt):
        laplacian_numba[blockspergrid, threadsperblock](
            phi,
            lap,
            [1, 1, 0],
        )
    Timer.stop()
    Timer.print(label="numba-gpu")

    # import ipdb
    # ipdb.set_trace()

    bo = BackendOptions(
        rebuild=False, dtypes={"dtype": so.dtype}, validate_args=True
    )
    sf = Laplacian("gt4py:gtcuda", backend_options=bo, storage_options=so)

    phi = sf.zeros(shape=shape)
    lap = sf.zeros(shape=shape)

    laplacian_gt4py = sf.compile_stencil("lap")

    Timer.start(label="gt4py")
    for _ in range(nt):
        laplacian_gt4py(
            phi, lap, origin=(1, 1, 0), domain=(nx - 2, ny - 2, nz)
        )
    Timer.stop()
    Timer.print(label="gt4py")
    #
    # # import ipdb
    # # ipdb.set_trace()


if __name__ == "__main__":
    # pytest.main([__file__])
    test_gpu()
