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
from tasmania.python.utils.framework_utils import register


@register(name="first_order")
class FirstOrder(HorizontalSmoothing):
    """ Two-dimensional first-order smoothing. """

    def __init__(
        self,
        shape,
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
        nb,
        gt_powered,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        managed_memory,
    ):
        nb = 1 if (nb is None or nb < 1) else nb
        super().__init__(
            shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            gt_powered,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )

    def __call__(self, phi, phi_out):
        # shortcuts
        nb = self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_out,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
        )

        # set the outermost lateral layers of the output field,
        # not affected by the stencil
        phi_out[:nb, :] = phi[:nb, :]
        phi_out[-nb:, :] = phi[-nb:, :]
        phi_out[nb:-nb, :nb] = phi[nb:-nb, :nb]
        phi_out[nb:-nb, -nb:] = phi[nb:-nb, -nb:]

    @staticmethod
    def _stencil_numpy(in_phi, in_gamma, out_phi, *, origin, domain, **kwargs):
        i = slice(origin[0], origin[0] + domain[0])
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        j = slice(origin[1], origin[1] + domain[1])
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - in_gamma[i, j, k]) * in_phi[i, j, k] + 0.25 * in_gamma[
            i, j, k
        ] * (
            in_phi[im1, j, k] + in_phi[ip1, j, k] + in_phi[i, jm1, k] + in_phi[i, jp1, k]
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - in_gamma[0, 0, 0]) * in_phi[0, 0, 0] + 0.25 * in_gamma[
                0, 0, 0
            ] * (in_phi[-1, 0, 0] + in_phi[1, 0, 0] + in_phi[0, -1, 0] + in_phi[0, 1, 0])


@register(name="first_order_1dx")
class FirstOrder1DX(HorizontalSmoothing):
    """ One-dimensional first-order smoothing along the x-direction. """

    def __init__(
        self,
        shape,
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
        nb,
        gt_powered,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        managed_memory,
    ):
        nb = 1 if (nb is None or nb < 1) else nb
        super().__init__(
            shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            gt_powered,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )

    def __call__(self, phi, phi_out):
        # shortcuts
        nb = self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_out,
            origin=(nb, 0, 0),
            domain=(nx - 2 * nb, ny, nz),
            exec_info=self._exec_info,
        )

        # set the outermost lateral layers of the output field,
        # not affected by the stencil
        phi_out[:nb, :] = phi[:nb, :]
        phi_out[-nb:, :] = phi[-nb:, :]

    @staticmethod
    def _stencil_numpy(in_phi, in_gamma, out_phi, *, origin, domain, **kwargs):
        i = slice(origin[0], origin[0] + domain[0])
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.5 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.25 * in_gamma[i, j, k] * (in_phi[im1, j, k] + in_phi[ip1, j, k])

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.5 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.25 * in_gamma[0, 0, 0] * (in_phi[-1, 0, 0] + in_phi[1, 0, 0])


@register(name="first_order_1dy")
class FirstOrder1DY(HorizontalSmoothing):
    """ One-dimensional first-order smoothing along the y-direction. """

    def __init__(
        self,
        shape,
        smooth_coeff,
        smooth_coeff_max,
        smooth_damp_depth,
        nb,
        gt_powered,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        managed_memory,
    ):
        nb = 1 if (nb is None or nb < 1) else nb
        super().__init__(
            shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            gt_powered,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )

    def __call__(self, phi, phi_out):
        # shortcuts
        nb = self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_out,
            origin=(0, nb, 0),
            domain=(nx, ny - 2 * nb, nz),
            exec_info=self._exec_info,
        )

        # set the outermost lateral layers of the output field,
        # not affected by the stencil
        phi_out[:, :nb] = phi[:, :nb]
        phi_out[:, -nb:] = phi[:, -nb:]

    @staticmethod
    def _stencil_numpy(in_phi, in_gamma, out_phi, *, origin, domain, **kwargs):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.5 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.25 * in_gamma[i, j, k] * (in_phi[i, jm1, k] + in_phi[i, jp1, k])

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.5 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.25 * in_gamma[0, 0, 0] * (in_phi[0, -1, 0] + in_phi[0, 1, 0])
