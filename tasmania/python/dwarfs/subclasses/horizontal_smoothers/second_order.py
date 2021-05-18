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

from tasmania.python.dwarfs.horizontal_smoothing import HorizontalSmoothing
from tasmania.python.framework.tag import stencil_definition


class SecondOrder(HorizontalSmoothing):
    """Two-dimensional second-order smoothing."""

    name = "second_order"

    def __init__(
        self,
        shape,
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
        nb=None,
        *,
        backend="numpy",
        backend_options=None,
        storage_options=None,
    ):
        nb = 2 if (nb is None or nb < 2) else nb
        super().__init__(
            shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            backend,
            backend_options,
            storage_options,
        )

    def __call__(self, phi, phi_out):
        # shortcuts
        nb = self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil_smooth(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_out,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

        # set the outermost lateral layers of the output field,
        # not affected by the stencil
        self._stencil_copy(
            src=phi,
            dst=phi_out,
            origin=(0, 0, 0),
            domain=(nb, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        self._stencil_copy(
            src=phi,
            dst=phi_out,
            origin=(nx - nb, 0, 0),
            domain=(nb, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        self._stencil_copy(
            src=phi,
            dst=phi_out,
            origin=(nb, 0, 0),
            domain=(nx - 2 * nb, nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        self._stencil_copy(
            src=phi,
            dst=phi_out,
            origin=(nb, ny - nb, 0),
            domain=(nx - 2 * nb, nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="smoothing"
    )
    def _smoothing_numpy(in_phi, in_gamma, out_phi, *, origin, domain):
        i = slice(origin[0], origin[0] + domain[0])
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        j = slice(origin[1], origin[1] + domain[1])
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.75 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.0625 * in_gamma[i, j, k] * (
            -in_phi[im2, j, k]
            + 4.0 * in_phi[im1, j, k]
            - in_phi[ip2, j, k]
            + 4.0 * in_phi[ip1, j, k]
            - in_phi[i, jm2, k]
            + 4.0 * in_phi[i, jm1, k]
            - in_phi[i, jp2, k]
            + 4.0 * in_phi[i, jp1, k]
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="smoothing")
    def _smoothing_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.75 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.0625 * in_gamma[0, 0, 0] * (
                -in_phi[-2, 0, 0]
                + 4.0 * in_phi[-1, 0, 0]
                - in_phi[+2, 0, 0]
                + 4.0 * in_phi[+1, 0, 0]
                - in_phi[0, -2, 0]
                + 4.0 * in_phi[0, -1, 0]
                - in_phi[0, +2, 0]
                + 4.0 * in_phi[0, +1, 0]
            )


class SecondOrder1DX(HorizontalSmoothing):
    """One-dimensional second-order smoothing along the x-direction."""

    name = "second_order_1dx"

    def __init__(
        self,
        shape,
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
        nb=None,
        *,
        backend="numpy",
        backend_options=None,
        storage_options=None,
    ):
        nb = 2 if (nb is None or nb < 2) else nb
        super().__init__(
            shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            backend,
            backend_options,
            storage_options,
        )

    def __call__(self, phi, phi_out):
        # shortcuts
        nb = self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil_smooth(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_out,
            origin=(nb, 0, 0),
            domain=(nx - 2 * nb, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

        # set the outermost lateral layers of the output field,
        # not affected by the stencil
        self._stencil_copy(
            src=phi,
            dst=phi_out,
            origin=(0, 0, 0),
            domain=(nb, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        self._stencil_copy(
            src=phi,
            dst=phi_out,
            origin=(nx - nb, 0, 0),
            domain=(nb, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="smoothing"
    )
    def _smoothing_numpy(in_phi, in_gamma, out_phi, *, origin, domain):
        i = slice(origin[0], origin[0] + domain[0])
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.375 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.0625 * in_gamma[i, j, k] * (
            -in_phi[im2, j, k]
            + 4.0 * in_phi[im1, j, k]
            - in_phi[ip2, j, k]
            + 4.0 * in_phi[ip1, j, k]
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="smoothing")
    def _smoothing_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.375 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.0625 * in_gamma[0, 0, 0] * (
                -in_phi[-2, 0, 0]
                + 4.0 * in_phi[-1, 0, 0]
                - in_phi[+2, 0, 0]
                + 4.0 * in_phi[+1, 0, 0]
            )


class SecondOrder1DY(HorizontalSmoothing):
    """One-dimensional second-order smoothing along the y-direction."""

    name = "second_order_1dy"

    def __init__(
        self,
        shape,
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
        nb=None,
        *,
        backend="numpy",
        backend_options=None,
        storage_options=None,
    ):
        nb = 2 if (nb is None or nb < 2) else nb
        super().__init__(
            shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            backend,
            backend_options,
            storage_options,
        )

    def __call__(self, phi, phi_out):
        # shortcuts
        nb = self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil_smooth(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_out,
            origin=(0, nb, 0),
            domain=(nx, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

        # set the outermost lateral layers of the output field,
        # not affected by the stencil
        self._stencil_copy(
            src=phi,
            dst=phi_out,
            origin=(0, 0, 0),
            domain=(nx, nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        self._stencil_copy(
            src=phi,
            dst=phi_out,
            origin=(0, ny - nb, 0),
            domain=(nx, nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="smoothing"
    )
    def _smoothing_numpy(in_phi, in_gamma, out_phi, *, origin, domain):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.375 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.0625 * in_gamma[i, j, k] * (
            -in_phi[i, jm2, k]
            + 4.0 * in_phi[i, jm1, k]
            - in_phi[i, jp2, k]
            + 4.0 * in_phi[i, jp1, k]
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="smoothing")
    def _smoothing_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.375 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.0625 * in_gamma[0, 0, 0] * (
                -in_phi[0, -2, 0]
                + 4.0 * in_phi[0, -1, 0]
                - in_phi[0, +2, 0]
                + 4.0 * in_phi[0, +1, 0]
            )
