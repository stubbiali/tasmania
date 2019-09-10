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
"""
This module contains:
    HorizontalDiffusion
    SecondOrder(HorizontalDiffusion)
    SecondOrder1D{X, Y}(HorizontalDiffusion)
    FourthOrder(HorizontalDiffusion)
    FourthOrder1D{X, Y}(HorizontalDiffusion)
"""
import abc
import math
import numpy as np

import gridtools as gt
from tasmania.python.utils.storage_utils import ones

try:
    from tasmania.conf import datatype
except ImportError:
    from numpy import float32 as datatype


class HorizontalDiffusion(abc.ABC):
    """
    Abstract base class whose derived classes calculates the
    tendency due to horizontal diffusion.
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
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        halo,
        rebuild,
    ):
        """
        Parameters
        ----------
        shape : tuple
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
            TODO
        backend_opts : dict
            TODO
        build_info : dict
            TODO
        dtype : numpy.dtype
            The data type for any :class:`numpy.ndarray` instantiated and
            used within this class.
        exec_info : dict
            TODO
        halo : tuple
            TODO
        rebuild : bool
            TODO
        """
        # store input arguments needed at run-time
        self._shape = shape
        self._nb = nb
        self._dx = dx
        self._dy = dy
        self._exec_info = exec_info

        # initialize the diffusivity
        gamma = diffusion_coeff * ones(
            (shape[0], shape[1], shape[2]), backend, dtype, halo, mask=[True, True, True]
            # (1, 1, shape[2]), backend, dtype, halo, mask=[False, False, True]
        )
        self._gamma = gamma

        # the diffusivity is monotonically increased towards the top of the model,
        # so to mimic the effect of a short-length wave absorber
        n = diffusion_damp_depth
        if n > 0:
            pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
            gamma[:, :, :n] += (diffusion_coeff_max - diffusion_coeff) * pert

        # initialize the underlying stencil
        decorator = gt.stencil(
            backend, backend_opts=backend_opts, build_info=build_info, rebuild=rebuild
        )
        self._stencil = decorator(self._stencil_defs)

    @abc.abstractmethod
    def __call__(self, phi, phi_tnd):
        """
        Calculate the tendency.

        Parameters
        ----------
        phi : gridtools.storage.Storage
            The 3-D prognostic field.
        phi_tnd : gridtools.storage.Storage
            Buffer where the calculated tendency will be written.
        """
        pass

    @staticmethod
    def factory(
        diffusion_type,
        shape,
        dx,
        dy,
        diffusion_coeff,
        diffusion_coeff_max,
        diffusion_damp_depth,
        nb=None,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        halo=None,
        rebuild=False
    ):
        """
        Parameters
        ----------
        diffusion_type : string
            String specifying the diffusion technique to implement. Either:

            * 'second_order', for second-order computational diffusion;
            * 'fourth_order', for fourth-order computational diffusion.

        shape : tuple
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
            TODO
        backend_opts : `dict`, optional
            TODO
        build_info : `dict`, optional
            TODO
        dtype : `numpy.dtype`, optional
            The data type for any :class:`numpy.ndarray` instantiated and
            used within this class.
        exec_info : `dict`, optional
            TODO
        halo : `tuple`, optional
            TODO
        rebuild : `bool`, optional
            TODO

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
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            halo,
            rebuild,
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
    def _stencil_defs(in_phi, in_gamma, out_phi, *, dx, dy):
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
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        halo,
        rebuild,
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
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            halo,
            rebuild,
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
            origin={"_all_": (nb, nb, 0)},
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_defs(
        in_phi: gt.storage.f64_sd,
        in_gamma: gt.storage.f64_k_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dx: float,
        dy: float
    ):
        out_phi = in_gamma[0, 0, 0] * (
            (in_phi[-1, 0, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[1, 0, 0]) / (dx * dx)
            + (in_phi[0, -1, 0] - 2.0 * in_phi[0, 0, 0] + in_phi[0, 1, 0]) / (dy * dy)
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
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        halo,
        rebuild,
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
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            halo,
            rebuild,
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
            origin={"_all_": (nb, 0, 0)},
            domain=(nx - 2 * nb, ny, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_defs(
        in_phi: gt.storage.f64_sd,
        in_gamma: gt.storage.f64_k_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dx: float,
        dy: float
    ):
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
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        halo,
        rebuild,
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
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            halo,
            rebuild,
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
            origin={"in_phi": (0, nb, 0), "in_gamma": (0, nb, 0), "out_phi": (0, nb, 0)},
            domain=(nx, ny - 2 * nb, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_defs(
        in_phi: gt.storage.f64_sd,
        in_gamma: gt.storage.f64_k_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dx: float,
        dy: float
    ):
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
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        halo,
        rebuild,
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
            backend_opts,
            build_info,
            dtype,
            exec_info,
            halo,
            rebuild,
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
            origin={"_all_": (nb, nb, 0)},
            domain=(nx - 2 * nb, ny - 2 * nb, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_defs(
        in_phi: gt.storage.f64_sd,
        in_gamma: gt.storage.f64_k_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dx: float,
        dy: float
    ):
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
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        halo,
        rebuild,
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
            backend_opts,
            build_info,
            dtype,
            exec_info,
            halo,
            rebuild,
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
            origin={"_all_": (nb, 0, 0)},
            domain=(nx - 2 * nb, ny, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_defs(
        in_phi: gt.storage.f64_sd,
        in_gamma: gt.storage.f64_k_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dx: float,
        dy: float
    ):
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
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        halo,
        rebuild,
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
            backend_opts,
            build_info,
            dtype,
            exec_info,
            halo,
            rebuild,
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
            origin={"_all_": (0, nb, 0)},
            domain=(nx, ny - 2 * nb, nz),
            exec_info=self._exec_info,
        )

    @staticmethod
    def _stencil_defs(
        in_phi: gt.storage.f64_sd,
        in_gamma: gt.storage.f64_k_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dx: float,
        dy: float
    ):
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
