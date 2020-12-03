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

from tasmania.python.framework import tag
from tasmania.python.framework.register import register
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion


@register(name="second_order")
class SecondOrder(HorizontalDiffusion):
    """ Two-dimensional second-order diffusion. """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb,
        backend,
        backend_options,
        storage_options,
    ):
        nb = 1 if (nb is None or nb < 1) else nb
        lb = 2 * nb + 1
        assert shape[0] >= lb and shape[1] >= lb, (
            f"\n\tProvided: shape[0] = {shape[0]} and shape[1] = {shape[1]}."
            f"\n\tRequirements: shape[0] >= {lb} and shape[1] >= {lb}."
        )

        super().__init__(
            shape,
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
            nb,
            backend,
            backend_options,
            storage_options,
        )

    def __call__(self, phi, phi_tnd):
        # shortcuts
        dx, dy, nb = self._dx, self._dy, self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_tnd,
            dx=dx,
            dy=dy,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=False,
        )

    @staticmethod
    @tag.stencil_definition(backend=("numpy", "cupy"), stencil="diffusion")
    def _stencil_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, origin, domain, **kwargs
    ):
        i = slice(origin[0], origin[0] + domain[0])
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        j = slice(origin[1], origin[1] + domain[1])
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)

        out_phi[i, j] = in_gamma[i, j] * (
            (in_phi[im1, j] - 2.0 * in_phi[i, j] + in_phi[ip1, j]) / (dx * dx)
            + (in_phi[i, jm1] - 2.0 * in_phi[i, j] + in_phi[i, jp1])
            / (dy * dy)
        )

    @staticmethod
    @tag.stencil_definition(backend="gt4py*", stencil="diffusion")
    def _stencil_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = in_gamma[0, 0, 0] * (
                (in_phi[-1, 0, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[1, 0, 0])
                / (dx * dx)
                + (in_phi[0, -1, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[0, 1, 0])
                / (dy * dy)
            )


@register(name="second_order_1dx")
class SecondOrder1DX(HorizontalDiffusion):
    """ One-dimensional second-order diffusion along the x-direction. """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb,
        backend,
        backend_options,
        storage_options,
    ):
        nb = 1 if (nb is None or nb < 1) else nb
        lb = 2 * nb + 1
        assert shape[0] >= lb, (
            f"\n\tProvided: shape[0] = {shape[0]}."
            f"\n\tRequirement: shape[0] >= {lb}."
        )

        super().__init__(
            shape,
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
            nb,
            backend,
            backend_options,
            storage_options,
        )

    def __call__(self, phi, phi_tnd):
        # shortcuts
        dx, dy, nb = self._dx, self._dy, self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_tnd,
            dx=dx,
            dy=dy,
            origin=(nb, 0, 0),
            domain=(nx - 2 * nb, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=False,
        )

    @staticmethod
    @tag.stencil_definition(backend=("numpy", "cupy"), stencil="diffusion")
    def _stencil_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, origin, domain, **kwargs
    ):
        i = slice(origin[0], origin[0] + domain[0])
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        j = slice(origin[1], origin[1] + domain[1])

        out_phi[i, j] = (
            in_gamma[i, j]
            * (in_phi[im1, j] - 2.0 * in_phi[i, j] + in_phi[ip1, j])
            / (dx * dx)
        )

    @staticmethod
    @tag.stencil_definition(backend="gt4py*", stencil="diffusion")
    def _stencil_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float = 0.0
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (
                in_gamma[0, 0, 0]
                * (in_phi[-1, 0, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[1, 0, 0])
                / (dx * dx)
            )


@register(name="second_order_1dy")
class SecondOrder1DY(HorizontalDiffusion):
    """ One-dimensional second-order diffusion along the y-direction. """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb,
        backend,
        backend_options,
        storage_options,
    ):
        nb = 1 if (nb is None or nb < 1) else nb
        lb = 2 * nb + 1
        assert shape[1] >= lb, (
            f"\n\tProvided: shape[1] = {shape[1]}."
            f"\n\tRequirement: shape[1] >= {lb}."
        )

        super().__init__(
            shape,
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
            nb,
            backend,
            backend_options,
            storage_options,
        )

    def __call__(self, phi, phi_tnd):
        # shortcuts
        dx, dy, nb = self._dx, self._dy, self._nb
        nx, ny, nz = self._shape

        # run the stencil
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_tnd,
            dx=dx,
            dy=dy,
            origin=(0, nb, 0),
            domain=(nx, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=False,
        )

    @staticmethod
    @tag.stencil_definition(backend=("numpy", "cupy"), stencil="diffusion")
    def _stencil_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, origin, domain, **kwargs
    ):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)

        out_phi[i, j] = (
            in_gamma[i, j]
            * (in_phi[i, jm1] - 2.0 * in_phi[i, j] + in_phi[i, jp1])
            / (dy * dy)
        )

    @staticmethod
    @tag.stencil_definition(backend="gt4py*", stencil="diffusion")
    def _stencil_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float = 0.0,
        dy: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (
                in_gamma[0, 0, 0]
                * (in_phi[0, -1, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[0, 1, 0])
                / (dy * dy)
            )
