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

from tasmania.python.dwarfs.horizontal_hyperdiffusion import (
    HorizontalHyperDiffusion,
)
from tasmania.python.framework.register import register
from tasmania.python.framework.tag import stencil_definition


@register(name="second_order")
class SecondOrder(HorizontalHyperDiffusion):
    """Two-dimensional second-order hyper-diffusion."""

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
        nb = 2 if (nb is None or nb < 2) else nb
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
            validate_args=self.backend_options.validate_args,
        )

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="hyperdiffusion"
    )
    def _hyperdiffusion_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, origin, domain
    ):
        ib, ie = origin[0], origin[0] + domain[0]
        jb, je = origin[1], origin[1] + domain[1]
        k = slice(origin[2], origin[2] + domain[2])

        lap0 = laplacian(dx, dy, in_phi[ib - 2 : ie + 2, jb - 2 : je + 2, k])
        out_phi[ib:ie, jb:je, k] = in_gamma[ib:ie, jb:je, k] * laplacian(
            dx, dy, lap0
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="hyperdiffusion")
    def _hyperdiffusion_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float
    ) -> None:
        from __externals__ import laplacian, laplacian_x, laplacian_y

        with computation(PARALLEL), interval(...):
            lap0 = laplacian(dx=dx, dy=dy, phi=in_phi)
            lap1 = laplacian(dx=dx, dy=dy, phi=lap0)
            out_phi = in_gamma * lap1


@register(name="second_order_1dx")
class SecondOrder1DX(HorizontalHyperDiffusion):
    """One-dimensional second-order hyper-diffusion along the x-direction."""

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
        nb = 2 if (nb is None or nb < 2) else nb
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
            validate_args=self.backend_options.validate_args,
        )

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="hyperdiffusion"
    )
    def _hyperdiffusion_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, origin, domain
    ):
        ib, ie = origin[0], origin[0] + domain[0]
        jb, je = origin[1], origin[1] + domain[1]
        k = slice(origin[2], origin[2] + domain[2])

        lap0 = laplacian_x(dx, in_phi[ib - 2 : ie + 2, jb:je, k])
        out_phi[ib:ie, jb:je, k] = in_gamma[ib:ie, jb:je, k] * laplacian_x(
            dx, lap0
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="hyperdiffusion")
    def _hyperdiffusion_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float = 0.0
    ) -> None:
        from __externals__ import laplacian_x

        with computation(PARALLEL), interval(...):
            lap0 = laplacian_x(dx=dx, phi=in_phi)
            lap1 = laplacian_x(dx=dx, phi=lap0)
            out_phi = in_gamma * lap1


@register(name="second_order_1dy")
class SecondOrder1DY(HorizontalHyperDiffusion):
    """One-dimensional second-order hyper-diffusion along the y-direction."""

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
        nb = 2 if (nb is None or nb < 2) else nb
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
            validate_args=self.backend_options.validate_args,
        )

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="hyperdiffusion"
    )
    def _hyperdiffusion_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, origin, domain
    ):
        ib, ie = origin[0], origin[0] + domain[0]
        jb, je = origin[1], origin[1] + domain[1]
        k = slice(origin[2], origin[2] + domain[2])

        lap0 = laplacian_y(dy, in_phi[ib:ie, jb - 2 : je + 2, k])
        out_phi[ib:ie, jb:je, k] = in_gamma[ib:ie, jb:je, k] * laplacian_y(
            dy, lap0
        )

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="hyperdiffusion")
    def _hyperdiffusion_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float = 0.0,
        dy: float
    ) -> None:
        from __externals__ import laplacian_y

        with computation(PARALLEL), interval(...):
            lap0 = laplacian_y(dy=dy, phi=in_phi)
            lap1 = laplacian_y(dy=dy, phi=lap0)
            out_phi = in_gamma * lap1
