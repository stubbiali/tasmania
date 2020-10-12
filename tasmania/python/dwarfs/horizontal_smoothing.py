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
from typing import Optional

from gt4py import gtscript

from tasmania.python.utils import taz_types
from tasmania.python.utils.dict_utils import stencil_copy_defs
from tasmania.python.utils.framework_utils import factorize
from tasmania.python.utils.storage_utils import get_asarray_function, zeros
from tasmania.python.utils.utils import get_gt_backend, is_gt


class HorizontalSmoothing(abc.ABC):
    """ Apply horizontal numerical smoothing to a generic (prognostic) field. """

    registry = {}

    def __init__(
        self,
        shape: taz_types.triplet_int_t,
        smooth_coeff: float,
        smooth_coeff_max: float,
        smooth_damp_depth: int,
        nb: int,
        backend: str,
        backend_opts: taz_types.options_dict_t,
        dtype: taz_types.dtype_t,
        build_info: taz_types.options_dict_t,
        exec_info: taz_types.mutable_options_dict_t,
        default_origin: taz_types.triplet_int_t,
        rebuild: bool,
        managed_memory: bool,
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
        backend_opts : dict
            Dictionary of backend-specific options.
        build_info : dict
            Dictionary of building options.
        dtype : data-type
            Data type of the storages.
        exec_info : dict
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : tuple[int]
            Storage default origin.
        rebuild : bool
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.
        """
        # store input arguments needed at run-time
        self._backend = backend
        self._shape = shape
        self._nb = nb
        self._exec_info = exec_info

        # initialize the diffusivity
        gamma = smooth_coeff * np.ones((1, 1, shape[2]), dtype=dtype)

        # the diffusivity is monotonically increased towards the top of the model,
        # so to mimic the effect of a short-length wave absorber
        n = smooth_damp_depth
        if n > 0:
            pert = (
                np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n)
                ** 2
            )
            gamma[:, :, :n] += (smooth_coeff_max - smooth_coeff) * pert

        # convert diffusivity to gt4py storage
        self._gamma = zeros(
            shape,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            mask=(True, True, True),
            managed_memory=managed_memory,
        )
        asarray = get_asarray_function(backend)
        self._gamma[...] = asarray(gamma)

        # initialize the underlying stencil
        if is_gt(backend):
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                name=self.__class__.__name__,
                backend=get_gt_backend(backend),
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                **(backend_opts or {})
            )
            self._stencil_copy = gtscript.stencil(
                definition=stencil_copy_defs,
                backend=get_gt_backend(backend),
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @abc.abstractmethod
    def __call__(
        self, phi: taz_types.array_t, phi_out: taz_types.array_t
    ) -> None:
        """Apply horizontal smoothing to a prognostic field.

        Parameters
        ----------
        phi : gt4py.storage.storage.Storage
            The 3-D field to filter.
        phi_out : gt4py.storage.storage.Storage
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
        backend_opts: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        build_info: Optional[taz_types.options_dict_t] = None,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        managed_memory: bool = False
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
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        dtype : `data-type`, optional
            Data type of the storages.
        build_info : `dict`, optional
            Dictionary of building options.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.

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
            backend_opts,
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            managed_memory,
        )
        obj = factorize(smooth_type, HorizontalSmoothing, args)
        return obj

    @staticmethod
    @abc.abstractmethod
    def _stencil_numpy(
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
    @abc.abstractmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        pass
