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
from typing import Optional, TYPE_CHECKING

from gt4py import gtscript

from tasmania.python.framework.register import factorize
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.utils import taz_types

if TYPE_CHECKING:
    import taichi as ti

    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class HorizontalDiffusion(StencilFactory, abc.ABC):
    """Calculate the tendency due to horizontal diffusion."""

    registry = {}

    def __init__(
        self: "HorizontalDiffusion",
        shape: taz_types.triplet_int_t,
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
        gamma = self.zeros(
            backend,
            shape=shape,
            # (1, 1, shape[2]), backend, dtype, default_origin,
            # mask=[False, False, True]
        )
        gamma[...] = diffusion_coeff
        self._gamma = gamma

        # the diffusivity is monotonically increased towards the top of the
        # model, so to mimic the effect of a short-length wave absorber
        dtype = self.storage_options.dtype
        n = diffusion_damp_depth
        if n > 0:
            pert = (
                np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n)
                ** 2
            )
            gamma[:, :, :n] = (
                gamma[:, :, :n]
                + (diffusion_coeff_max - diffusion_coeff) * pert
            )

        # initialize the underlying stencil
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil = self.compile("diffusion")

    @abc.abstractmethod
    def __call__(
        self: "HorizontalDiffusion",
        phi: taz_types.array_t,
        phi_tnd: taz_types.array_t,
    ) -> None:
        """Calculate the tendency.

        Parameters
        ----------
        phi : array-like
            A 3-D prognostic field.
        phi_tnd : array-like
            Output buffer in which to place the result.
        """
        pass

    @staticmethod
    def factory(
        diffusion_type: str,
        shape: taz_types.triplet_int_t,
        dx: float,
        dy: float,
        diffusion_coeff: float,
        diffusion_coeff_max: float,
        diffusion_damp_depth: int,
        nb: Optional[int] = None,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> "HorizontalDiffusion":
        """
        Parameters
        ----------
        diffusion_type : str
            String specifying the diffusion technique to implement.
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
        obj = factorize(diffusion_type, HorizontalDiffusion, args)
        return obj

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="diffusion")
    @abc.abstractmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="diffusion")
    @abc.abstractmethod
    def _stencil_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="numba:cpu", stencil="diffusion")
    @abc.abstractmethod
    def _stencil_numba_cpu(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="numba:cpu", stencil="diffusion")
    @abc.abstractmethod
    def _stencil_numba_cpu(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="taichi:*", stencil="diffusion")
    @abc.abstractmethod
    def _stencil_taichi(
        in_phi: "ti.template()",
        in_gamma: "ti.template()",
        out_phi: "ti.template()",
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
    ) -> None:
        pass
