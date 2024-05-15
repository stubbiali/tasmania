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
import numba

from sympl._core.time import Timer

from gt4py import gtscript

from tasmania.python.framework.tag import stencil_definition
from tasmania.python.dwarfs.horizontal_diffusion import HorizontalDiffusion


class FourthOrder(HorizontalDiffusion):
    """Two-dimensional fourth-order diffusion."""

    name = "fourth_order"

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb=None,
        *,
        backend="numpy",
        backend_options=None,
        storage_options=None,
    ):
        nb = 2 if (nb is None or nb < 2) else nb
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

    def __call__(self, phi, phi_tnd, *, overwrite_output=True):
        # shortcuts
        dx, dy, nb = self._dx, self._dy, self._nb
        nx, ny, nz = self._shape

        # run the stencil
        Timer.start(label="stencil")
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_tnd,
            dx=dx,
            dy=dy,
            ow_out_phi=overwrite_output,
            origin=(nb, nb, 0),
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="diffusion"
    )
    def _diffusion_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, ow_out_phi, origin, domain
    ):
        i = slice(origin[0], origin[0] + domain[0])
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        j = slice(origin[1], origin[1] + domain[1])
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)

        tmp = in_gamma[i, j] * (
            (
                -in_phi[im2, j]
                + 16.0 * in_phi[im1, j]
                - 30.0 * in_phi[i, j]
                + 16.0 * in_phi[ip1, j]
                - in_phi[ip2, j]
            )
            / (12.0 * dx * dx)
            + (
                -in_phi[i, jm2]
                + 16.0 * in_phi[i, jm1]
                - 30.0 * in_phi[i, j]
                + 16.0 * in_phi[i, jp1]
                - in_phi[i, jp2]
            )
            / (12.0 * dy * dy)
        )
        set_output(out_phi[i, j], tmp, ow_out_phi)

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="diffusion")
    def _diffusion_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
        ow_out_phi: bool
    ) -> None:
        from __externals__ import set_output

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
            out_phi = set_output(out_phi, tmp, ow_out_phi)

    @staticmethod
    @stencil_definition(backend="numba:cpu:stencil", stencil="diffusion")
    def _diffusion_numba_cpu(
        in_phi, in_gamma, out_phi, *, dx, dy, ow_out_phi, origin, domain
    ):
        def core_def(phi, gamma, dx, dy):
            return gamma[0, 0, 0] * (
                (
                    -phi[-2, 0, 0]
                    + 16.0 * phi[-1, 0, 0]
                    - 30.0 * phi[0, 0, 0]
                    + 16.0 * phi[1, 0, 0]
                    - phi[2, 0, 0]
                )
                / (12.0 * dx * dx)
                + (
                    -phi[0, -2, 0]
                    + 16.0 * phi[0, -1, 0]
                    - 30.0 * phi[0, 0, 0]
                    + 16.0 * phi[0, 1, 0]
                    - phi[0, 2, 0]
                )
                / (12.0 * dy * dy)
            )

        core = numba.stencil(core_def)

        ib, jb, kb = origin[0] - 2, origin[1] - 2, origin[2]
        ie, je, ke = ib + domain[0] + 4, jb + domain[1] + 4, kb + domain[2]
        tmp = core(
            in_phi[ib:ie, jb:je, kb:ke],
            in_gamma[ib:ie, jb:je, kb:ke],
            dx,
            dy,
        )
        set_output(out_phi[ib:ie, jb:je, kb:ke], tmp, ow_out_phi)


class FourthOrder1DX(HorizontalDiffusion):
    """One-dimensional fourth-order diffusion along the x-direction."""

    name = "fourth_order_1dx"

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb=None,
        *,
        backend="numpy",
        backend_options=None,
        storage_options=None,
    ):
        nb = 2 if (nb is None or nb < 2) else nb
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

    def __call__(self, phi, phi_tnd, *, overwrite_output=True):
        # shortcuts
        dx, dy, nb = self._dx, self._dy, self._nb
        nx, ny, nz = self._shape

        # run the stencil
        Timer.start(label="stencil")
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_tnd,
            dx=dx,
            dy=dy,
            ow_out_phi=overwrite_output,
            origin=(nb, 0, 0),
            domain=(nx - 2 * nb, ny, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="diffusion"
    )
    def _diffusion_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, ow_out_phi, origin, domain
    ):
        i = slice(origin[0], origin[0] + domain[0])
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        j = slice(origin[1], origin[1] + domain[1])

        tmp = (
            in_gamma[i, j]
            * (
                -in_phi[im2, j]
                + 16.0 * in_phi[im1, j]
                - 30.0 * in_phi[i, j]
                + 16.0 * in_phi[ip1, j]
                - in_phi[ip2, j]
            )
            / (12.0 * dx * dx)
        )
        set_output(out_phi[i, j], tmp, ow_out_phi)

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="diffusion")
    def _diffusion_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float = 0.0,
        ow_out_phi: bool
    ) -> None:
        from __externals__ import set_output

        with computation(PARALLEL), interval(...):
            tmp = (
                in_gamma[0, 0, 0]
                * (
                    -in_phi[-2, 0, 0]
                    + 16.0 * in_phi[-1, 0, 0]
                    - 30.0 * in_phi[0, 0, 0]
                    + 16.0 * in_phi[1, 0, 0]
                    - in_phi[2, 0, 0]
                )
                / (12.0 * dx * dx)
            )
            out_phi = set_output(out_phi, tmp, ow_out_phi)

    @staticmethod
    @stencil_definition(backend="numba:cpu:stencil", stencil="diffusion")
    def _diffusion_numba_cpu(
        in_phi, in_gamma, out_phi, *, dx, dy=0.0, ow_out_phi, origin, domain
    ):
        def core_def(phi, gamma, dx):
            return (
                gamma[0, 0, 0]
                * (
                    -phi[-2, 0, 0]
                    + 16.0 * phi[-1, 0, 0]
                    - 30.0 * phi[0, 0, 0]
                    + 16.0 * phi[1, 0, 0]
                    - phi[2, 0, 0]
                )
                / (12.0 * dx * dx)
            )

        core = numba.stencil(core_def)

        ib, jb, kb = origin[0] - 2, origin[1], origin[2]
        ie, je, ke = ib + domain[0] + 4, jb + domain[1], kb + domain[2]
        tmp = core(
            in_phi[ib:ie, jb:je, kb:ke],
            in_gamma[ib:ie, jb:je, kb:ke],
            dx,
        )
        set_output(out_phi[ib:ie, jb:je, kb:ke], tmp, ow_out_phi)


class FourthOrder1DY(HorizontalDiffusion):
    """One-dimensional fourth-order diffusion along the y-direction."""

    name = "fourth_order_1dy"

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb=None,
        *,
        backend="numpy",
        backend_options=None,
        storage_options=None,
    ):
        nb = 2 if (nb is None or nb < 2) else nb
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

    def __call__(self, phi, phi_tnd, *, overwrite_output=True):
        # shortcuts
        dx, dy, nb = self._dx, self._dy, self._nb
        nx, ny, nz = self._shape

        # run the stencil
        Timer.start(label="stencil")
        self._stencil(
            in_phi=phi,
            in_gamma=self._gamma,
            out_phi=phi_tnd,
            dx=dx,
            dy=dy,
            ow_out_phi=overwrite_output,
            origin=(0, nb, 0),
            domain=(nx, ny - 2 * nb, nz),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="diffusion"
    )
    def _diffusion_numpy(
        in_phi, in_gamma, out_phi, *, dx, dy, ow_out_phi, origin, domain
    ):
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)

        tmp = (
            in_gamma[i, j]
            * (
                -in_phi[i, jm2]
                + 16.0 * in_phi[i, jm1]
                - 30.0 * in_phi[i, j]
                + 16.0 * in_phi[i, jp1]
                - in_phi[i, jp2]
            )
            / (12.0 * dy * dy)
        )
        set_output(out_phi[i, j], tmp, ow_out_phi)

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="diffusion")
    def _diffusion_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float = 0.0,
        dy: float,
        ow_out_phi: bool
    ) -> None:
        from __externals__ import set_output

        with computation(PARALLEL), interval(...):
            tmp = (
                in_gamma[0, 0, 0]
                * (
                    -in_phi[0, -2, 0]
                    + 16.0 * in_phi[0, -1, 0]
                    - 30.0 * in_phi[0, 0, 0]
                    + 16.0 * in_phi[0, 1, 0]
                    - in_phi[0, 2, 0]
                )
                / (12.0 * dy * dy)
            )
            out_phi = set_output(out_phi, tmp, ow_out_phi)

    @staticmethod
    @stencil_definition(backend="numba:cpu:stencil", stencil="diffusion")
    def _diffusion_numba_cpu(
        in_phi, in_gamma, out_phi, *, dx=0.0, dy, ow_out_phi, origin, domain
    ):
        def core_def(phi, gamma, dy):
            return (
                gamma[0, 0, 0]
                * (
                    -phi[0, -2, 0]
                    + 16.0 * phi[0, -1, 0]
                    - 30.0 * phi[0, 0, 0]
                    + 16.0 * phi[0, 1, 0]
                    - phi[0, 2, 0]
                )
                / (12.0 * dy * dy)
            )

        core = numba.stencil(core_def)

        ib, jb, kb = origin[0], origin[1] - 2, origin[2]
        ie, je, ke = ib + domain[0], jb + domain[1] + 4, kb + domain[2]
        tmp = core(
            in_phi[ib:ie, jb:je, kb:ke],
            in_gamma[ib:ie, jb:je, kb:ke],
            dy,
        )
        set_output(
            out_phi[ib:ie, jb:je, kb:ke],
            tmp,
            ow_out_phi,
        )
