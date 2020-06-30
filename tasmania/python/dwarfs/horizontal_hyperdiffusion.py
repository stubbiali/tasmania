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
from tasmania.python.utils.framework_utils import factorize
from tasmania.python.utils.storage_utils import zeros


def stage_laplacian_x_numpy(dx: float, phi: np.ndarray) -> np.ndarray:
    lap = (phi[:-2] - 2.0 * phi[1:-1] + phi[2:]) / (dx * dx)
    return lap


def stage_laplacian_y_numpy(dy: float, phi: np.ndarray) -> np.ndarray:
    lap = (phi[:, :-2] - 2.0 * phi[:, 1:-1] + phi[:, 2:]) / (dy * dy)
    return lap


def stage_laplacian_numpy(dx: float, dy: float, phi: np.ndarray) -> np.ndarray:
    lap = (phi[:-2, 1:-1] - 2.0 * phi[1:-1, 1:-1] + phi[2:, 1:-1]) / (dx * dx) + (
        phi[1:-1, :-2] - 2.0 * phi[1:-1, 1:-1] + phi[1:-1, 2:]
    ) / (dy * dy)
    return lap


@gtscript.function
def stage_laplacian_x(dx: float, phi: taz_types.gtfield_t) -> taz_types.gtfield_t:
    lap = (phi[-1, 0, 0] - 2.0 * phi[0, 0, 0] + phi[1, 0, 0]) / (dx * dx)
    return lap


@gtscript.function
def stage_laplacian_y(dy: float, phi: taz_types.gtfield_t) -> taz_types.gtfield_t:
    lap = (phi[0, -1, 0] - 2.0 * phi[0, 0, 0] + phi[0, 1, 0]) / (dy * dy)
    return lap


@gtscript.function
def stage_laplacian(
    dx: float, dy: float, phi: taz_types.gtfield_t
) -> taz_types.gtfield_t:
    lap_x = stage_laplacian_x(dx=dx, phi=phi)
    lap_y = stage_laplacian_y(dy=dy, phi=phi)
    lap = lap_x + lap_y
    return lap


class HorizontalHyperDiffusion(abc.ABC):
    """ Calculate the tendency due to horizontal hyper-diffusion. """

    registry = {}

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
            TODO
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

        # initialize the diffusion coefficient
        self._gamma = zeros(
            shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )

        # the diffusivity is monotonically increased towards the top of the model,
        # so to mimic the effect of a short-length wave absorber
        n = diffusion_damp_depth
        if True:  # if n > 0:
            pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype=dtype)) / n) ** 2
            pert = np.tile(pert[np.newaxis, np.newaxis, :], (shape[0], shape[1], 1))
            self._gamma[...] = diffusion_coeff
            self._gamma[:, :, :n] += (diffusion_coeff_max - diffusion_coeff) * pert

        if gt_powered:
            # initialize the underlying stencil
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                name=self.__class__.__name__,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                externals={
                    "stage_laplacian": stage_laplacian,
                    "stage_laplacian_x": stage_laplacian_x,
                    "stage_laplacian_y": stage_laplacian_y,
                },
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @abc.abstractmethod
    def __call__(
        self, phi: taz_types.gtstorage_t, phi_tnd: taz_types.gtstorage_t
    ) -> None:
        """ Calculate the tendency.

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
        gt_powered : `bool`, optional
            TODO
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
        args = (
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
        obj = factorize(diffusion_type, HorizontalHyperDiffusion, args)
        return obj

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
        domain: taz_types.triplet_int_t
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
