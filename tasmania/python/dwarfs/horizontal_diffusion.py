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
import abc
import math
import numpy as np
from typing import TYPE_CHECKING

from sympl._core.factory import AbstractFactory

from gt4py import gtscript

from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLike

    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.utils.typingx import TripletInt


class HorizontalDiffusion(AbstractFactory, StencilFactory):
    """Calculate the tendency due to horizontal diffusion."""

    def __init__(
        self: "HorizontalDiffusion",
        shape: "TripletInt",
        dx: float,
        dy: float,
        diffusion_coeff: float,
        diffusion_coeff_max: float,
        diffusion_damp_depth: int,
        nb: int,
        backend: str,
        backend_options: "BackendOptions",
        storage_options: "StorageOptions",
    ) -> None:
        """
        Parameters
        ----------
        shape : tuple[int]
            Shape of the 3-D arrays for which tendencies should be computed.
        dx : float
            The grid spacing along the first horizontal dimension.
        dy : float
            The grid spacing along the second horizontal dimension.
        diffusion_coeff : float
            Value for the diffusion coefficient far from the top boundary.
        diffusion_coeff_max : float
            Maximum value for the diffusion coefficient.
        diffusion_damp_depth : int
            Depth of, i.e., number of vertical regions in the damping region.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        backend_options : BackendOptions
            Backend-specific options.
        storage_options : StorageOptions
            Storage-related options.
        """
        super().__init__(backend, backend_options, storage_options)

        # store input arguments needed at run-time
        self._shape = shape
        self._nb = nb
        self._dx = dx
        self._dy = dy

        # initialize the diffusivity
        dtype = self.storage_options.dtype
        gamma = diffusion_coeff * np.ones((1, 1, shape[2]), dtype=dtype)

        # the diffusivity is monotonically increased towards the top of the
        # model, so to mimic the effect of a short-length wave absorber
        n = diffusion_damp_depth
        if n > 0:
            pert = (
                np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n)
                ** 2
            )
            gamma[:, :, :n] += (diffusion_coeff_max - diffusion_coeff) * pert

        # convert diffusivity to proper storage
        self._gamma = self.zeros(backend, shape=shape)
        self._gamma[...] = self.as_storage(data=gamma)

        # initialize the underlying stencil
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "set_output": self.get_subroutine_definition("set_output")
        }
        self._stencil = self.compile_stencil("diffusion")

    @abc.abstractmethod
    def __call__(
        self: "HorizontalDiffusion",
        phi: "NDArrayLike",
        phi_tnd: "NDArrayLike",
        *,
        overwrite_output: bool = True
    ) -> None:
        """Calculate the tendency.

        Parameters
        ----------
        phi : array-like
            A 3-D prognostic field.
        phi_tnd : array-like
            Output buffer in which to place the result.
        overwrite_output : `bool`, optional
            TODO
        """
        pass

    @staticmethod
    @stencil_definition(
        backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="diffusion"
    )
    @abc.abstractmethod
    def _diffusion_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        ow_out_phi: bool,
        origin: "TripletInt",
        domain: "TripletInt"
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="diffusion")
    @abc.abstractmethod
    def _diffusion_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
        ow_out_phi: bool
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="numba:cpu:stencil", stencil="diffusion")
    @abc.abstractmethod
    def _diffusion_numba_cpu(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        ow_out_phi: bool,
        origin: "TripletInt",
        domain: "TripletInt"
    ) -> None:
        pass

    # @staticmethod
    # @stencil_definition(backend="taichi:*", stencil="diffusion")
    # @abc.abstractmethod
    # def _diffusion_taichi(
    #     in_phi: "taichi.template()",
    #     in_gamma: "taichi.template()",
    #     out_phi: "taichi.template()",
    #     dx: float,
    #     dy: float,
    #     origin: taz_types.triplet_int_t,
    #     domain: taz_types.triplet_int_t,
    # ) -> None:
    #     pass
