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
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_definition
from tasmania.python.utils import taz_types

if TYPE_CHECKING:
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class HorizontalSmoothing(StencilFactory, abc.ABC):
    """Apply horizontal numerical smoothing to a generic (prognostic) field."""

    registry = {}

    def __init__(
        self: "HorizontalSmoothing",
        shape: taz_types.triplet_int_t,
        smooth_coeff: float,
        smooth_coeff_max: float,
        smooth_damp_depth: int,
        nb: int,
        backend: str,
        backend_options: "BackendOptions",
        storage_options: "StorageOptions",
    ) -> None:
        """
        Parameters
        ----------
        shape : tuple[int]
            Shape of the 3-D arrays which should be filtered.
        smooth_coeff : float
            Value for the smoothing coefficient far from the top boundary.
        smooth_coeff_max : float
            Maximum value for the smoothing coefficient.
        smooth_damp_depth : int
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
        self._backend = backend
        self._shape = shape
        self._nb = nb

        # initialize the diffusivity
        dtype = self.storage_options.dtype
        gamma = smooth_coeff * np.ones((1, 1, shape[2]), dtype=dtype)

        # the diffusivity is monotonically increased towards the top of the
        # model, so to mimic the effect of a short-length wave absorber
        n = smooth_damp_depth
        if n > 0:
            pert = (
                np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n)
                ** 2
            )
            gamma[:, :, :n] += (smooth_coeff_max - smooth_coeff) * pert

        # convert diffusivity to proper storage
        self._gamma = self.zeros(backend, shape=shape)
        asarray = self.asarray(backend)
        self._gamma[...] = asarray(gamma)

        # initialize the underlying stencil
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil = self.compile("smoothing")
        self._stencil_copy = self.compile("copy")

    @abc.abstractmethod
    def __call__(
        self, phi: taz_types.array_t, phi_out: taz_types.array_t
    ) -> None:
        """Apply horizontal smoothing to a prognostic field.

        Parameters
        ----------
        phi : array-like
            The 3-D field to filter.
        phi_out : array-like
            Output buffer in which to place the computed field.
        """
        pass

    @staticmethod
    def factory(
        smooth_type: str,
        shape: taz_types.triplet_int_t,
        smooth_coeff: float,
        smooth_coeff_max: float,
        smooth_damp_depth: int,
        nb: Optional[int] = None,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> "HorizontalSmoothing":
        """Get an instance of a registered derived class.

        Parameters
        ----------
        smooth_type : str
            String specifying the smoothing technique to implement. Either:

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
            Shape of the 3-D arrays which should be filtered.
        smooth_coeff : float
            Value for the smoothing coefficient far from the top boundary.
        smooth_coeff_max : float
            Maximum value for the smoothing coefficient.
        smooth_damp_depth : int
            Depth of, i.e., number of vertical regions in the damping region.
        nb : `int`, optional
            Number of boundary layers.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.

        Return
        ------
        obj :
            Instance of the suitable derived class.
        """
        args = (
            shape,
            smooth_coeff,
            smooth_coeff_max,
            smooth_damp_depth,
            nb,
            backend,
            backend_options,
            storage_options,
        )
        obj = factorize(smooth_type, HorizontalSmoothing, args)
        return obj

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy"), stencil="smoothing")
    @abc.abstractmethod
    def _smoothing_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="smoothing")
    @abc.abstractmethod
    def _smoothing_gt4py(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        pass
