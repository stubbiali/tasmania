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
from typing import Optional, TYPE_CHECKING

from gt4py import gtscript

from tasmania.python.framework.register import factorize
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import (
    stencil_definition,
    subroutine_definition,
)
from tasmania.python.utils import typingx as ty

if TYPE_CHECKING:
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class HorizontalHyperDiffusion(StencilFactory, abc.ABC):
    """Calculate the tendency due to horizontal hyper-diffusion."""

    registry = {}

    def __init__(
        self: "HorizontalHyperDiffusion",
        shape: ty.TripletInt,
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

        # initialize the diffusion coefficient
        dtype = self.storage_options.dtype
        gamma = diffusion_coeff * np.ones((1, 1, shape[2]), dtype=dtype)

        # the diffusivity is monotonically increased towards the top of
        # the domain, so to mimic the effect of a short-length wave absorber
        n = diffusion_damp_depth
        if n > 0:
            pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
            gamma[:, :, :n] += (diffusion_coeff_max - diffusion_coeff) * pert

        # convert diffusivity to proper storage
        self._gamma = self.zeros(backend, shape=shape)
        self._gamma[...] = self.as_storage(data=gamma)

        # compile the underlying stencil
        self.backend_options.dtypes = {"dtype": dtype}
        self.backend_options.externals = {
            "laplacian": self.get_subroutine_definition("laplacian"),
            "laplacian_x": self.get_subroutine_definition("laplacian_x"),
            "laplacian_y": self.get_subroutine_definition("laplacian_y"),
        }
        self._stencil = self.compile_stencil("hyperdiffusion")

    @abc.abstractmethod
    def __call__(self, phi: ty.Storage, phi_tnd: ty.Storage) -> None:
        """Calculate the tendency.

        Parameters
        ----------
        phi : array-like
            The 3-D prognostic field.
        phi_tnd : array-like
            Output buffer into which to place the computed tendency.
        """

    @staticmethod
    def factory(
        diffusion_type: str,
        shape: ty.TripletInt,
        dx: float,
        dy: float,
        diffusion_coeff: float,
        diffusion_coeff_max: float,
        diffusion_damp_depth: int,
        nb: Optional[int] = None,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None,
    ) -> "HorizontalHyperDiffusion":
        """
        Static method returning an instance of the derived class
        calculating the tendency due to horizontal hyper-diffusion of type
        `diffusion_type`.

        Parameters
        ----------
        diffusion_type : str
            String specifying the diffusion technique to implement. Either:

            * 'first_order';
            * 'first_order_1dx';
            * 'first_order_1dy';
            * 'second_order';
            * 'second_order_1dx';
            * 'second_order_1dy';
            * 'third_order';
            * 'third_order_1dx';
            * 'third_order_1dy'.

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
        nb : `int`, optional
            Number of boundary layers. If not specified, this is derived
            from the extent of the underlying stencil.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.

        Return
        ------
        obj :
            Instance of the appropriate derived class.
        """
        args = (
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
        obj = factorize(diffusion_type, HorizontalHyperDiffusion, args)
        return obj

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="hyperdiffusion")
    @abc.abstractmethod
    def _hyperdiffusion_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: ty.TripletInt,
        domain: ty.TripletInt,
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="hyperdiffusion")
    @abc.abstractmethod
    def _hyperdiffusion_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float,
    ) -> None:
        pass

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="laplacian_x")
    def stage_laplacian_x_numpy(dx: float, phi: np.ndarray) -> np.ndarray:
        lap = (phi[:-2] - 2.0 * phi[1:-1] + phi[2:]) / (dx * dx)
        return lap

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="laplacian_y")
    def stage_laplacian_y_numpy(dy: float, phi: np.ndarray) -> np.ndarray:
        lap = (phi[:, :-2] - 2.0 * phi[:, 1:-1] + phi[:, 2:]) / (dy * dy)
        return lap

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="laplacian")
    def stage_laplacian_numpy(dx: float, dy: float, phi: np.ndarray) -> np.ndarray:
        lap = (phi[:-2, 1:-1] - 2.0 * phi[1:-1, 1:-1] + phi[2:, 1:-1]) / (dx * dx) + (
            phi[1:-1, :-2] - 2.0 * phi[1:-1, 1:-1] + phi[1:-1, 2:]
        ) / (dy * dy)
        return lap

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="laplacian_x")
    @gtscript.function
    def stage_laplacian_x(dx: float, phi: ty.GTField) -> ty.GTField:
        lap = (phi[-1, 0, 0] - 2.0 * phi[0, 0, 0] + phi[1, 0, 0]) / (dx * dx)
        return lap

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="laplacian_y")
    @gtscript.function
    def stage_laplacian_y(dy: float, phi: ty.GTField) -> ty.GTField:
        lap = (phi[0, -1, 0] - 2.0 * phi[0, 0, 0] + phi[0, 1, 0]) / (dy * dy)
        return lap

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="laplacian")
    @gtscript.function
    def stage_laplacian(dx: float, dy: float, phi: ty.GTField) -> ty.GTField:
        lap = (phi[-1, 0, 0] - 2.0 * phi[0, 0, 0] + phi[1, 0, 0]) / (dx * dx) + (
            phi[0, -1, 0] - 2.0 * phi[0, 0, 0] + phi[0, 1, 0]
        ) / (dy * dy)
        return lap
