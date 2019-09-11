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
import abc
import math
import numpy as np

import gridtools as gt


datatype = np.float64


def get_storage_descriptor(storage_shape, dtype, halo=None, mask=(True, True, True)):
    halo = (0, 0, 0) if halo is None else halo
    halo = tuple(halo[i] if storage_shape[i] > 2 * halo[i] else 0 for i in range(3))
    domain = tuple(storage_shape[i] - 2 * halo[i] for i in range(3))
    descriptor = gt.storage.StorageDescriptor(
        dtype, mask=mask, halo=halo, iteration_domain=domain
    )
    return descriptor


class HorizontalDiffusion:
    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb,
        *,
        backend,
        dtype,
        halo,
        rebuild
    ):
        # store input arguments needed at run-time
        self._shape = shape
        nb = 1 if (nb is None or nb < 1) else nb
        self._nb = nb
        self._dx = dx
        self._dy = dy

        # initialize the diffusivity
        gamma = diffusion_coeff * np.ones((1, 1, shape[2]), dtype=dtype)

        # the diffusivity is monotonically increased towards the top of the model,
        # so to mimic the effect of a short-length wave absorber
        n = diffusion_damp_depth
        if n > 0:
            pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
            gamma[:, :, :n] += (diffusion_coeff_max - diffusion_coeff) * pert

        # convert diffusivity to a gt4py storage
        descriptor = get_storage_descriptor(
            shape, dtype, halo, mask=(True, True, True)
        )  # mask=(False, False, True)
        self._gamma = gt.storage.from_array(gamma, descriptor, backend=backend)

        # initialize the underlying stencil
        decorator = gt.stencil(backend, rebuild=rebuild)
        self._stencil = decorator(self._stencil_defs)

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
            origin={"_all_": (nb, nb, 0)},
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
        )

    @staticmethod
    def _stencil_defs(
        in_phi: gt.storage.f64_sd,
        in_gamma: gt.storage.f64_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dx: float,
        dy: float
    ):
        out_phi = in_gamma[0, 0, 0] * (
            (in_phi[-1, 0, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[1, 0, 0]) / (dx * dx)
            + (in_phi[0, -1, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[0, 1, 0]) / (dy * dy)
        )


if __name__ == "__main__":
    shape = (20, 30, 15)
    dx = 1.0
    dy = 1.0
    diffusion_coeff = 1.0
    diffusion_coeff_max = 2.0
    diffusion_damp_depth = 0
    nb = 3
    backend = "numpy"
    dtype = np.float64
    halo = (0, 0, 0)
    rebuild = True

    phi_rnd = np.random.rand(*shape)
    descriptor = get_storage_descriptor(shape, dtype, halo=halo)
    phi = phi_rnd  # gt.storage.from_array(phi_rnd, descriptor, backend=backend)
    phi_out = np.zeros(
        shape, dtype=dtype
    )  # gt.storage.zeros(descriptor, backend=backend)

    diffuser = HorizontalDiffusion(
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb,
        backend=backend,
        dtype=dtype,
        halo=halo,
        rebuild=rebuild,
    )
    diffuser(phi, phi_out)
