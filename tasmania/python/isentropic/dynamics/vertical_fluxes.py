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
import numpy as np
from typing import Any, Dict, Tuple

from sympl._core.factory import AbstractFactory

from gt4py import gtscript

from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.framework.tag import stencil_subroutine


class IsentropicMinimalVerticalFlux(AbstractFactory, StencilFactory):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the vertical numerical fluxes for the three-dimensional
    isentropic and *minimal* dynamical core. The conservative form of the
    governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: Dict[str, Any] = None

    def __init__(self, *, backend):
        super().__init__(backend)

    @staticmethod
    @stencil_subroutine(backend=("numpy", "cupy"), stencil="flux_dry")
    @abc.abstractmethod
    def flux_dry_numpy(
        dt: float,
        dz: float,
        w: np.ndarray,
        s: np.ndarray,
        su: np.ndarray,
        sv: np.ndarray,
    ) -> Tuple[np.ndarray]:
        pass

    @staticmethod
    @stencil_subroutine(backend=("numpy", "cupy"), stencil="flux_moist")
    @abc.abstractmethod
    def flux_moist_numpy(
        dt: float,
        dz: float,
        w: np.ndarray,
        sqv: np.ndarray,
        sqc: np.ndarray,
        sqr: np.ndarray,
    ) -> Tuple[np.ndarray]:
        pass

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="flux_dry")
    @gtscript.function
    @abc.abstractmethod
    def flux_dry_gt4py(
        dt: float,
        dz: float,
        w: gtscript.Field["dtype"],
        s: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
    ) -> "Tuple[gtscript.Field['dtype'], ...]":
        pass

    @staticmethod
    @stencil_subroutine(backend="gt4py*", stencil="flux_moist")
    @gtscript.function
    @abc.abstractmethod
    def flux_moist_gt4py(
        dt: float,
        dz: float,
        w: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"],
        sqc: gtscript.Field["dtype"],
        sqr: gtscript.Field["dtype"],
    ) -> "Tuple[gtscript.Field['dtype'], ...]":
        pass
