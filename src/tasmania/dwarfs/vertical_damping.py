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

from __future__ import annotations
import abc
import math
import numpy as np
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript
from sympl._core.factory import AbstractFactory

from tasmania.framework.base_components import GridComponent
from tasmania.framework.stencil import StencilFactory
from tasmania.framework.tag import stencil_definition
from tasmania.utils.utils import greater_or_equal_than as ge

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tasmania.domain.grid import Grid
    from tasmania.framework.options import BackendOptions, StorageOptions
    from tasmania.utils.typingx import NDArray, TimeDelta, TripletInt


class VerticalDamping(AbstractFactory, GridComponent, StencilFactory):
    """
    Abstract base class whose derived classes implement different
    vertical damping, i.e. wave absorbing, techniques.
    """

    def __init__(
        self,
        grid: Grid,
        damp_depth: int,
        damp_coeff_max: float,
        time_units: str,
        backend: str,
        backend_options: BackendOptions,
        storage_shape: Sequence[int],
        storage_options: StorageOptions,
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        damp_depth : int
            Number of vertical layers in the damping region.
        damp_coeff_max : float
            Maximum value for the damping coefficient.
        time_units : str
            Time units to be used throughout the class.
        backend : str
            The backend.
        backend_options : BackendOptions
            Backend-specific options.
        storage_options : StorageOptions
            Storage-related options.
        """
        super().__init__(grid)
        super(GridComponent, self).__init__(backend, backend_options, storage_options)

        # safety-guard checks
        assert damp_depth <= grid.nz, (
            f"The depth of the damping region ({damp_depth}) should be "
            f"smaller or equal than the number of main vertical levels "
            f"({grid.nz})."
        )

        # store input arguments needed at run-time
        self._damp_depth = damp_depth
        self._tunits = time_units
        storage_shape = self.get_storage_shape(
            storage_shape, max_shape=(grid.nx + 1, grid.ny + 1, grid.nz + 1)
        )
        self._shape = storage_shape

        # allocate the damping matrix
        self._rmat = self.zeros(shape=storage_shape)
        if damp_depth > 0:
            # fill the damping matrix
            z = (
                np.concatenate((grid.z.values, np.array([0])), axis=0)
                if storage_shape[2] == grid.nz + 1
                else grid.z.values
            )
            zt = grid.z_on_interface_levels.values[0]
            za = z[damp_depth - 1]
            r = ge(z, za) * damp_coeff_max * (1 - np.cos(math.pi * (z - za) / (zt - za)))
            self._rmat[...] = self.as_storage(data=r[np.newaxis, np.newaxis, :])

        # instantiate the underlying stencil
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil_damp = self.compile_stencil("damping")

    @abc.abstractmethod
    def __call__(
        self,
        dt: TimeDelta,
        field_now: NDArray,
        field_new: NDArray,
        field_ref: NDArray,
        field_out: NDArray,
    ) -> None:
        """Apply vertical damping to a generic field.

        As this method is marked as abstract, its implementation
        is delegated to the derived classes.

        Parameters
        ----------
        dt : datetime.timedelta
            The time step.
        field_now : array-like
            The field at the current time level.
        field_new : array-like
            The field at the next time level, on which the absorber will be
            applied.
        field_ref : array-like
            A reference value for the field.
        field_out : array-like
            Buffer into which the output vertically damped field is written.
        """

    @staticmethod
    @stencil_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="damping")
    @abc.abstractmethod
    def _damping_numpy(
        in_phi_now: np.ndarray,
        in_phi_new: np.ndarray,
        in_phi_ref: np.ndarray,
        in_rmat: np.ndarray,
        out_phi: np.ndarray,
        *,
        dt: float,
        origin: TripletInt,
        domain: TripletInt,
    ) -> None:
        pass

    @staticmethod
    @stencil_definition(backend="gt4py*", stencil="damping")
    @abc.abstractmethod
    def _damping_gt4py(
        in_phi_now: gtscript.Field["dtype"],
        in_phi_new: gtscript.Field["dtype"],
        in_phi_ref: gtscript.Field["dtype"],
        in_rmat: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dt: float,
    ) -> None:
        pass
