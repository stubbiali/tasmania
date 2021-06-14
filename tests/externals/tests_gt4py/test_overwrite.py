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

from tasmania.python.framework.options import BackendOptions, StorageOptions
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.utils.time import Timer


@gtscript.function
def set_output(out, val, ow):
    return val if ow else out + val


class SF(StencilFactory):
    def __init__(self, backend, backend_options, storage_options):
        super().__init__(
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self.stencil1 = self.compile_stencil("stencil1")
        self.stencil2 = self.compile_stencil("stencil2")

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil1")
    def stencil1(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
        ow_out_phi: bool
    ) -> None:
        with computation(PARALLEL), interval(...):
            tmp = in_gamma[0, 0, 0] * (
                (
                    -in_phi[-2, 0, 0]
                    + 16.0 * in_phi[-1, 0, 0]
                    - 30.0 * in_phi[0, 0, 0]
                    + 16.0 * in_phi[1, 0, 0]
                    - in_phi[2, 0, 0]
                )
                / (12.0 * dx * dx)
                + (
                    -in_phi[0, -2, 0]
                    + 16.0 * in_phi[0, -1, 0]
                    - 30.0 * in_phi[0, 0, 0]
                    + 16.0 * in_phi[0, 1, 0]
                    - in_phi[0, 2, 0]
                )
                / (12.0 * dy * dy)
            )

            # if ow_out_phi:
            #     out_phi = tmp
            # else:
            #     out_phi += tmp

            out_phi = set_output(out_phi, tmp, ow_out_phi)

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="stencil2")
    def stencil2(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
        ow_out_phi: bool
    ) -> None:
        with computation(PARALLEL), interval(...):
            if ow_out_phi:
                out_phi = in_gamma[0, 0, 0] * (
                    (
                        -in_phi[-2, 0, 0]
                        + 16.0 * in_phi[-1, 0, 0]
                        - 30.0 * in_phi[0, 0, 0]
                        + 16.0 * in_phi[1, 0, 0]
                        - in_phi[2, 0, 0]
                    )
                    / (12.0 * dx * dx)
                    + (
                        -in_phi[0, -2, 0]
                        + 16.0 * in_phi[0, -1, 0]
                        - 30.0 * in_phi[0, 0, 0]
                        + 16.0 * in_phi[0, 1, 0]
                        - in_phi[0, 2, 0]
                    )
                    / (12.0 * dy * dy)
                )
            else:
                out_phi += in_gamma[0, 0, 0] * (
                    (
                        -in_phi[-2, 0, 0]
                        + 16.0 * in_phi[-1, 0, 0]
                        - 30.0 * in_phi[0, 0, 0]
                        + 16.0 * in_phi[1, 0, 0]
                        - in_phi[2, 0, 0]
                    )
                    / (12.0 * dx * dx)
                    + (
                        -in_phi[0, -2, 0]
                        + 16.0 * in_phi[0, -1, 0]
                        - 30.0 * in_phi[0, 0, 0]
                        + 16.0 * in_phi[0, 1, 0]
                        - in_phi[0, 2, 0]
                    )
                    / (12.0 * dy * dy)
                )


def main(backend, shape, dtype=float):
    bo = BackendOptions(
        cache=True, check_rebuild=True, nopython=True, rebuild=False
    )
    so = StorageOptions(dtype=dtype)
    sf = SF(backend, bo, so)

    phi = sf.as_storage(data=np.random.rand(*shape))
    gamma = sf.as_storage(data=np.random.rand(*shape))
    out = sf.zeros(shape=shape)

    sf.stencil1(
        phi, gamma, out, dx=1.0, dy=1.0, ow_out_phi=True, origin=(2, 2, 0)
    )
    sf.stencil2(
        phi, gamma, out, dx=1.0, dy=1.0, ow_out_phi=True, origin=(2, 2, 0)
    )

    Timer.start(label="stencil1")
    for _ in range(50):
        sf.stencil1(
            phi, gamma, out, dx=1.0, dy=1.0, ow_out_phi=True, origin=(2, 2, 0)
        )
    Timer.stop()

    Timer.start(label="stencil2")
    for _ in range(50):
        sf.stencil2(
            phi, gamma, out, dx=1.0, dy=1.0, ow_out_phi=True, origin=(2, 2, 0)
        )
    Timer.stop()

    Timer.start(label="stencil1")
    for _ in range(50):
        sf.stencil1(
            phi, gamma, out, dx=1.0, dy=1.0, ow_out_phi=True, origin=(2, 2, 0)
        )
    Timer.stop()

    Timer.start(label="stencil2")
    for _ in range(50):
        sf.stencil2(
            phi, gamma, out, dx=1.0, dy=1.0, ow_out_phi=True, origin=(2, 2, 0)
        )
    Timer.stop()

    Timer.print("stencil1", "s")
    Timer.print("stencil2", "s")


if __name__ == "__main__":
    main("gt4py:gtx86", (512, 512, 256))
