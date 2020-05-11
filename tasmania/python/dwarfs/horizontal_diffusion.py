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
from tasmania.python.utils.storage_utils import zeros


class HorizontalDiffusion(abc.ABC):
    """
    Abstract base class whose derived classes calculates the
    tendency due to horizontal diffusion.
    """

    def __init__(
        self,
        shape: taz_types.triplet_int_t,
        dx: float,
        dy: float,
        diffusion_coeff: float,
        diffusion_coeff_max: float,
        diffusion_damp_depth: int,
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
        gt_powered : bool
            ``True`` to harness GT4Py, ``False`` for a vanilla Numpy implementation.
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
            ``True`` to trigger the stencils compilation at any class instantiation,
            ``False`` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory, ``False`` otherwise.
        """
        # store input arguments needed at run-time
        self._shape = shape
        self._nb = nb
        self._dx = dx
        self._dy = dy
        self._exec_info = exec_info

        # initialize the diffusivity
        gamma = zeros(
            (shape[0], shape[1], shape[2]),
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            mask=[True, True, True],
            managed_memory=managed_memory
            # (1, 1, shape[2]), backend, dtype, default_origin, mask=[False, False, True]
        )
        gamma[...] = diffusion_coeff
        self._gamma = gamma

        # the diffusivity is monotonically increased towards the top of the model,
        # so to mimic the effect of a short-length wave absorber
        n = diffusion_damp_depth
        if n > 0:
            pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
            gamma[:, :, :n] = (
                gamma[:, :, :n] + (diffusion_coeff_max - diffusion_coeff) * pert
            )

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
        else:
            self._stencil = self._stencil_numpy

    @abc.abstractmethod
    def __call__(
        self, phi: taz_types.gtstorage_t, phi_tnd: taz_types.gtstorage_t
    ) -> None:
        """
        Calculate the tendency.

        Parameters
        ----------
        phi : gt4py.storage.storage.Storage
            The 3-D prognostic field.
        phi_tnd : gt4py.storage.storage.Storage
            Buffer into which the calculated tendency is written.
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
    ) -> "HorizontalDiffusion":
        """
        Parameters
        ----------
        diffusion_type : str
            String specifying the diffusion technique to implement. Either:

            * 'second_order', for second-order computational diffusion;
            * 'fourth_order', for fourth-order computational diffusion.

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
        gt_powered : `bool`, optional
            ``True`` to harness GT4Py, ``False`` for a vanilla Numpy implementation.
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
            ``True`` to trigger the stencils compilation at any class instantiation,
            ``False`` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory, ``False`` otherwise.

        Return
        ------
        obj :
            Instance of the appropriate derived class.
        """
        args = [
            shape,
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
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

        if diffusion_type == "second_order":
            assert not (shape[0] < 3 and shape[1] < 3)

            if shape[1] < 3:
                return SecondOrder1DX(*args)
            elif shape[0] < 3:
                return SecondOrder1DY(*args)
            else:
                return SecondOrder(*args)
        elif diffusion_type == "fourth_order":
            assert not (shape[0] < 5 and shape[1] < 5)

            if shape[1] < 5:
                return FourthOrder1DX(*args)
            elif shape[0] < 5:
                return FourthOrder1DY(*args)
            else:
                return FourthOrder(*args)
        else:
            raise ValueError(
                "Supported diffusion operators are ''second_order'' "
                "and ''fourth_order''."
            )

    @staticmethod
    @abc.abstractmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
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
        *,
        dx: float,
        dy: float
    ) -> None:
        pass


class SecondOrder(HorizontalDiffusion):
    """
    This class inherits	:class:`tasmania.HorizontalDiffusion`
    to calculate the tendency due to second-order horizontal diffusion for any
    three-dimensional field	with at least three elements in each direction.

    Note
    ----
    An instance of this class should only be applied to fields whose dimensions
    match those specified at instantiation time. Hence, one should use (at least)
    one instance per field shape.
    """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
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
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
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
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        j = slice(origin[1], origin[1] + domain[1])
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)

        out_phi[i, j] = in_gamma[i, j] * (
            (in_phi[im1, j] - 2.0 * in_phi[i, j] + in_phi[ip1, j]) / (dx * dx)
            + (in_phi[i, jm1] - 2.0 * in_phi[i, j] + in_phi[i, jp1]) / (dy * dy)
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = in_gamma[0, 0, 0] * (
                (in_phi[-1, 0, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[1, 0, 0]) / (dx * dx)
                + (in_phi[0, -1, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[0, 1, 0])
                / (dy * dy)
            )


class SecondOrder1DX(HorizontalDiffusion):
    """
    This class inherits	:class:`tasmania.HorizontalDiffusion`
    to calculate the tendency due to second-order horizontal diffusion for any
    three-dimensional field	with only one element along the second dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
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
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
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
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs
    ) -> None:
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
    def _stencil_gt_defs(
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


class SecondOrder1DY(HorizontalDiffusion):
    """
    This class inherits	:class:`~tasmania.HorizontalDiffusion`
    to calculate the tendency due to second-order horizontal diffusion for any
    three-dimensional field	with only one element along the first dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
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
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
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
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs
    ) -> None:
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
    def _stencil_gt_defs(
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


class FourthOrder(HorizontalDiffusion):
    """
    This class inherits	:class:`~tasmania.HorizontalDiffusion`
    to calculate the tendency due to fourth-order horizontal diffusion for any
    three-dimensional field	with at least three elements in each direction.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
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
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
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
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs
    ) -> None:
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

        out_phi[i, j] = in_gamma[i, j] * (
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

    @staticmethod
    def _stencil_gt_defs(
        in_phi: gtscript.Field["dtype"],
        in_gamma: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dx: float,
        dy: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = in_gamma[0, 0, 0] * (
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


class FourthOrder1DX(HorizontalDiffusion):
    """
    This class inherits	:class:`~tasmania.HorizontalDiffusion`
    to calculate the tendency due to fourth-order horizontal diffusion for any
    three-dimensional field	with only one element along the second dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
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
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
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
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        ip2 = slice(origin[0] + 2, origin[0] + domain[0] + 2)
        ip1 = slice(origin[0] + 1, origin[0] + domain[0] + 1)
        im1 = slice(origin[0] - 1, origin[0] + domain[0] - 1)
        im2 = slice(origin[0] - 2, origin[0] + domain[0] - 2)
        j = slice(origin[1], origin[1] + domain[1])

        out_phi[i, j] = (
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

    @staticmethod
    def _stencil_gt_defs(
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
                * (
                    -in_phi[-2, 0, 0]
                    + 16.0 * in_phi[-1, 0, 0]
                    - 30.0 * in_phi[0, 0, 0]
                    + 16.0 * in_phi[1, 0, 0]
                    - in_phi[2, 0, 0]
                )
                / (12.0 * dx * dx)
            )


class FourthOrder1DY(HorizontalDiffusion):
    """
    This class inherits	:class:`~tasmania.HorizontalDiffusion`
    to calculate the tendency due to fourth-order horizontal diffusion for any
    three-dimensional field	with only one element along the first dimension.

    Note
    ----
    An instance of this class should only be applied to fields whose
    dimensions match those specified at instantiation time.
    Hence, one should use (at least) one instance per field shape.
    """

    def __init__(
        self,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
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
            dx,
            dy,
            diffusion_coeff,
            diffusion_coeff_max,
            diffusion_damp_depth,
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
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_numpy(
        in_phi: np.ndarray,
        in_gamma: np.ndarray,
        out_phi: np.ndarray,
        *,
        dx: float,
        dy: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        jp2 = slice(origin[1] + 2, origin[1] + domain[1] + 2)
        jp1 = slice(origin[1] + 1, origin[1] + domain[1] + 1)
        jm1 = slice(origin[1] - 1, origin[1] + domain[1] - 1)
        jm2 = slice(origin[1] - 2, origin[1] + domain[1] - 2)

        out_phi[i, j] = (
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

    @staticmethod
    def _stencil_gt_defs(
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
                * (
                    -in_phi[0, -2, 0]
                    + 16.0 * in_phi[0, -1, 0]
                    - 30.0 * in_phi[0, 0, 0]
                    + 16.0 * in_phi[0, 1, 0]
                    - in_phi[0, 2, 0]
                )
                / (12.0 * dy * dy)
            )
