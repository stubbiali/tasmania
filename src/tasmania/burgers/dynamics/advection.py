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
import numpy as np
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript

from tasmania.framework.register import factorize
from tasmania.framework.stencil import StencilFactory
from tasmania.framework.tag import subroutine_definition

if TYPE_CHECKING:
    from tasmania.utils.typingx import GTField


class BurgersAdvection(StencilFactory, abc.ABC):
    """A discretizer for the 2-D Burgers advection flux."""

    registry = {}

    extent: int = 0

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="advection")
    @abc.abstractmethod
    def call_numpy(
        dx: float, dy: float, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the accelerations due to advection.

        Vanilla NumPy implementation.

        Parameters
        ----------
        dx : float
            x-grid spacing.
        dy : float
            y-grid spacing.
        u : numpy.ndarray
            u-velocity.
        v : numpy.ndarray
            v-velocity.

        Return
        ------
        adv_u_x : numpy.ndarray
            x-acceleration for u-velocity.
        adv_u_y : numpy.ndarray
            y-acceleration for u-velocity.
        adv_v_x : numpy.ndarray
            x-acceleration for v-velocity.
        adv_v_y : numpy.ndarray
            y-acceleration for v-velocity.
        """

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="advection")
    @gtscript.function
    @abc.abstractmethod
    def call_gt4py(
        dx: float, dy: float, u: GTField, v: GTField
    ) -> tuple[GTField, GTField, GTField, GTField]:
        """
        Compute the accelerations due to advection. GT4Py-based implementation.

        Parameters
        ----------
        dx : float
            x-grid spacing.
        dy : float
            y-grid spacing.
        u : gt4py.gtscript.Field
            u-velocity.
        v : gt4py.gtscript.Field
            v-velocity.

        Return
        ------
        adv_u_x : gt4py.gtscript.Field
            x-acceleration for u-velocity.
        adv_u_y : gt4py.gtscript.Field
            y-acceleration for u-velocity.
        adv_v_x : gt4py.gtscript.Field
            x-acceleration for v-velocity.
        adv_v_y : gt4py.gtscript.Field
            y-acceleration for v-velocity.
        """

    @staticmethod
    @subroutine_definition(backend="numba:cpu:stencil", stencil="advection")
    @abc.abstractmethod
    def call_numba_cpu(
        dx: float, dy: float, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def factory(flux_scheme: str, backend: str) -> BurgersAdvection:
        return factorize(flux_scheme, BurgersAdvection, (backend,))
