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

# from gt4py.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import get_asarray_function, zeros
from tasmania.python.utils.dict_utils import stencil_copy_defs


class HorizontalSmoothing(abc.ABC):
    """
    Abstract base class whose derived classes apply horizontal
    numerical smoothing to a generic (prognostic) field.
    """

    def __init__(
        self,
        shape: taz_types.triplet_int_t,
        smooth_coeff: float,
        smooth_coeff_max: float,
        smooth_damp_depth: int,
        nb: int,
        gt_powered: bool,
        backend: str,
        backend_opts: taz_types.options_dict_t,
        build_info: taz_types.options_dict_t,
        dtype: taz_types.dtype_t,
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
        gt_powered : bool
            `True` to harness GT4Py, `False` for a vanilla Numpy implementation.
        backend : str
            The GT4Py backend.
        backend_opts : dict
            Dictionary of backend-specific options.
        build_info : dict
            Dictionary of building options.
        dtype : data-type
            Data type of the storages.
        exec_info : dict
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : tuple[int]
            Storage default origin.
        rebuild : bool
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.
        """
        # store input arguments needed at run-time
        self._shape = shape
        self._nb = nb
        self._exec_info = exec_info

        # initialize the diffusivity
        gamma = smooth_coeff * np.ones((1, 1, shape[2]), dtype=dtype)

        # the diffusivity is monotonically increased towards the top of the model,
        # so to mimic the effect of a short-length wave absorber
        n = smooth_damp_depth
        if n > 0:
            pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
            gamma[:, :, :n] += (smooth_coeff_max - smooth_coeff) * pert

        # convert diffusivity to gt4py storage
        self._gamma = zeros(
            shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            mask=(True, True, True),
            managed_memory=managed_memory,
        )
        asarray = get_asarray_function(gt_powered, backend)
        self._gamma[...] = asarray(gamma)

        # initialize the underlying stencil
        if gt_powered:
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                name=self.__class__.__name__,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                **(backend_opts or {})
            )
            self._stencil_copy = gtscript.stencil(
                definition=stencil_copy_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @abc.abstractmethod
    def __call__(self, phi: taz_types.array_t, phi_out: taz_types.array_t) -> None:
        """
        Apply horizontal smoothing to a prognostic field.
        As this method is marked as abstract, its implementation is
        delegated to the derived classes.

        Parameters
        ----------
        phi : gt4py.storage.storage.Storage
            The 3-D field to filter.
        phi_out : gt4py.storage.storage.Storage
            The 3-D buffer into which the filtered field is written.
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
        gt_powered: bool = True,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        managed_memory: bool = False
    ) -> "HorizontalSmoothing":
        """
        Static method returning an instance of the derived class
        implementing the smoothing technique specified by `smooth_type`.

        Parameters
        ----------
        smooth_type : str
            String specifying the smoothing technique to implement. Either:

            * 'first_order', for first-order numerical smoothing;
            * 'second_order', for second-order numerical smoothing;
            * 'third_order', for third-order numerical smoothing.

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
        gt_powered : `bool`, optional
            `True` to harness GT4Py, `False` for a vanilla Numpy implementation.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.

        Return
        ------
        obj :
            Instance of the suitable derived class.
        """
        args = [
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
        ]

        if smooth_type == "first_order":
            assert not (shape[0] < 3 and shape[1] < 3)

            if shape[1] < 3:
                return FirstOrder1DX(*args)
            elif shape[0] < 3:
                return FirstOrder1DY(*args)
            else:
                return FirstOrder(*args)

            # return FirstOrder1DX(*args)
        elif smooth_type == "second_order":
            assert not (shape[0] < 5 and shape[1] < 5)

            if shape[1] < 5:
                return SecondOrder1DX(*args)
            elif shape[0] < 5:
                return SecondOrder1DY(*args)
            else:
                return SecondOrder(*args)

            # return SecondOrder1DX(*args)
        elif smooth_type == "third_order":
            assert not (shape[0] < 7 and shape[1] < 7)

            if shape[1] < 7:
                return ThirdOrder1DX(*args)
            elif shape[0] < 7:
                return ThirdOrder1DY(*args)
            else:
                return ThirdOrder(*args)
        else:
            raise ValueError(
                "Supported smoothing operators are ''first_order'', "
                "''second_order'', and ''third_order''."
            )

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


class FirstOrder(HorizontalSmoothing):
    """
    This class inherits :class:`~tasmania.HorizontalSmoothing`
    to apply a first-order horizontal digital filter to three-dimensional fields
    with at least three elements along each dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose dimensions
    match those specified at instantiation time. Hence, one should use (at least)
    one instance per field shape.
    """

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
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
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


class FirstOrder1DX(HorizontalSmoothing):
    """
    This class inherits	:class:`~tasmania.HorizontalSmoothing`
    to apply a first-order horizontal digital filter to three-dimensional fields
    with only one element along the second dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

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
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
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


class FirstOrder1DY(HorizontalSmoothing):
    """
    This class inherits :class:`~tasmania.HorizontalSmoothing`
    to apply a first-order horizontal digital filter to three-dimensional fields
    with only one element along the first direction.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

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
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
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


class SecondOrder(HorizontalSmoothing):
    """
    This class inherits	:class:`~tasmania.HorizontalSmoothing`
    to apply a second-order horizontal digital filter to three-dimensional fields
    with at least three elements along each dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

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
        nb = 2 if (nb is None or nb < 2) else nb
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
        if not self._gt_powered:
            phi_out[:nb, :] = phi[:nb, :]
            phi_out[-nb:, :] = phi[-nb:, :]
            phi_out[nb:-nb, :nb] = phi[nb:-nb, :nb]
            phi_out[nb:-nb, -nb:] = phi[nb:-nb, -nb:]
        else:
            self._stencil_copy(
                src=phi, dst=phi_out, origin=(0, 0, 0), domain=(nb, ny, nz)
            )
            self._stencil_copy(
                src=phi, dst=phi_out, origin=(nx - nb, 0, 0), domain=(nb, ny, nz)
            )
            self._stencil_copy(
                src=phi, dst=phi_out, origin=(nb, 0, 0), domain=(nx - 2 * nb, nb, nz)
            )
            self._stencil_copy(
                src=phi,
                dst=phi_out,
                origin=(nb, ny - nb, 0),
                domain=(nx - 2 * nb, nb, nz),
            )

    @staticmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        j = slice(origin[1], origin[1] + domain[1])
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.75 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.0625 * in_gamma[i, j, k] * (
            -in_phi[im2, j, k]
            + 4.0 * in_phi[im1, j, k]
            - in_phi[ip2, j, k]
            + 4.0 * in_phi[ip1, j, k]
            - in_phi[i, jm2, k]
            + 4.0 * in_phi[i, jm1, k]
            - in_phi[i, jp2, k]
            + 4.0 * in_phi[i, jp1, k]
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.75 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.0625 * in_gamma[0, 0, 0] * (
                -in_phi[-2, 0, 0]
                + 4.0 * in_phi[-1, 0, 0]
                - in_phi[+2, 0, 0]
                + 4.0 * in_phi[+1, 0, 0]
                - in_phi[0, -2, 0]
                + 4.0 * in_phi[0, -1, 0]
                - in_phi[0, +2, 0]
                + 4.0 * in_phi[0, +1, 0]
            )


class SecondOrder1DX(HorizontalSmoothing):
    """
    This class inherits	:class:`~tasmania.HorizontalSmoothing`
    to apply a second-order horizontal digital filter to three-dimensional fields
    with only one element along the second dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

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
        nb = 2 if (nb is None or nb < 2) else nb
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
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.375 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.0625 * in_gamma[i, j, k] * (
            -in_phi[im2, j, k]
            + 4.0 * in_phi[im1, j, k]
            - in_phi[ip2, j, k]
            + 4.0 * in_phi[ip1, j, k]
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.375 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.0625 * in_gamma[0, 0, 0] * (
                -in_phi[-2, 0, 0]
                + 4.0 * in_phi[-1, 0, 0]
                - in_phi[+2, 0, 0]
                + 4.0 * in_phi[+1, 0, 0]
            )


class SecondOrder1DY(HorizontalSmoothing):
    """
    This class inherits	:class:`~tasmania.HorizontalSmoothing`
    to apply a second-order horizontal digital filter to three-dimensional fields
    with only one element along the first dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

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
        nb = 2 if (nb is None or nb < 2) else nb
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
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.375 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.0625 * in_gamma[i, j, k] * (
            -in_phi[i, jm2, k]
            + 4.0 * in_phi[i, jm1, k]
            - in_phi[i, jp2, k]
            + 4.0 * in_phi[i, jp1, k]
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.375 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.0625 * in_gamma[0, 0, 0] * (
                -in_phi[0, -2, 0]
                + 4.0 * in_phi[0, -1, 0]
                - in_phi[0, +2, 0]
                + 4.0 * in_phi[0, +1, 0]
            )


class ThirdOrder(HorizontalSmoothing):
    """
    This class inherits :class:`~tasmania.HorizontalSmoothing`
    to apply a third-order horizontal digital filter to three-dimensional fields
    with at least three elements along each dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

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
        nb = 3 if (nb is None or nb < 3) else nb
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
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        im3 = slice(origin[0] - 3, origin[0] + domain[0] - 3)
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        ip3 = slice(origin[0] + 3, origin[0] + domain[0] + 3)
        j = slice(origin[1], origin[1] + domain[1])
        jm3 = slice(origin[1] - 3, origin[1] + domain[1] - 3)
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        jp3 = slice(origin[1] + 3, origin[1] + domain[1] + 3)
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.625 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.015625 * in_gamma[i, j, k] * (
            in_phi[im3, j, k]
            - 6.0 * in_phi[im2, j, k]
            + 15.0 * in_phi[im1, j, k]
            + in_phi[ip3, j, k]
            - 6.0 * in_phi[ip2, j, k]
            + 15.0 * in_phi[ip1, j, k]
            + in_phi[i, jm3, k]
            - 6.0 * in_phi[i, jm2, k]
            + 15.0 * in_phi[i, jm1, k]
            + in_phi[i, jp3, k]
            - 6.0 * in_phi[i, jp2, k]
            + 15.0 * in_phi[i, jp1, k]
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.625 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.015625 * in_gamma[0, 0, 0] * (
                in_phi[-3, 0, 0]
                - 6.0 * in_phi[-2, 0, 0]
                + 15.0 * in_phi[-1, 0, 0]
                + in_phi[+3, 0, 0]
                - 6.0 * in_phi[+2, 0, 0]
                + 15.0 * in_phi[+1, 0, 0]
                + in_phi[0, -3, 0]
                - 6.0 * in_phi[0, -2, 0]
                + 15.0 * in_phi[0, -1, 0]
                + in_phi[0, +3, 0]
                - 6.0 * in_phi[0, +2, 0]
                + 15.0 * in_phi[0, +1, 0]
            )


class ThirdOrder1DX(HorizontalSmoothing):
    """
    This class inherits :class:`~tasmania.HorizontalSmoothing`
    to apply a third-order horizontal digital filter to three-dimensional fields
    with only one element along the second dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

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
        nb = 3 if (nb is None or nb < 3) else nb
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
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        im3 = slice(origin[0] - 3, origin[0] + domain[0] - 3)
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        ip3 = slice(origin[0] + 3, origin[0] + domain[0] + 3)
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.3125 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.015625 * in_gamma[i, j, k] * (
            in_phi[im3, j, k]
            - 6.0 * in_phi[im2, j, k]
            + 15.0 * in_phi[im1, j, k]
            + in_phi[ip3, j, k]
            - 6.0 * in_phi[ip2, j, k]
            + 15.0 * in_phi[ip1, j, k]
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.3125 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.015625 * in_gamma[0, 0, 0] * (
                in_phi[-3, 0, 0]
                - 6.0 * in_phi[-2, 0, 0]
                + 15.0 * in_phi[-1, 0, 0]
                + in_phi[+3, 0, 0]
                - 6.0 * in_phi[+2, 0, 0]
                + 15.0 * in_phi[+1, 0, 0]
            )


class ThirdOrder1DY(HorizontalSmoothing):
    """
    This class inherits	:class:`~tasmania.HorizontalSmoothing`
    to apply a third-order horizontal digital filter to three-dimensional fields
    with only one element along the first dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

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
        nb = 3 if (nb is None or nb < 3) else nb
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
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        jm3 = slice(origin[1] - 3, origin[1] + domain[1] - 3)
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        jp3 = slice(origin[1] + 3, origin[1] + domain[1] + 3)
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = (1.0 - 0.3125 * in_gamma[i, j, k]) * in_phi[
            i, j, k
        ] + 0.015625 * in_gamma[i, j, k] * (
            +in_phi[i, jm3, k]
            - 6.0 * in_phi[i, jm2, k]
            + 15.0 * in_phi[i, jm1, k]
            + in_phi[i, jp3, k]
            - 6.0 * in_phi[i, jp2, k]
            + 15.0 * in_phi[i, jp1, k]
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = (1.0 - 0.3125 * in_gamma[0, 0, 0]) * in_phi[
                0, 0, 0
            ] + 0.015625 * in_gamma[0, 0, 0] * (
                in_phi[0, -3, 0]
                - 6.0 * in_phi[0, -2, 0]
                + 15.0 * in_phi[0, -1, 0]
                + in_phi[0, +3, 0]
                - 6.0 * in_phi[0, +2, 0]
                + 15.0 * in_phi[0, +1, 0]
            )
