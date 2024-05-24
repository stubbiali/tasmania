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
from typing import TYPE_CHECKING

from gt4py.cartesian import gtscript
from sympl._core.factory import AbstractFactory

from tasmania.framework.stencil import StencilFactory
from tasmania.framework.tag import subroutine_definition

if TYPE_CHECKING:
    from typing import Any

    from tasmania.utils.typingx import NDArray


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
    externals: dict[str, Any] = None

    def __init__(self, *, backend):
        super().__init__(backend)

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy"), stencil="flux_dry")
    @abc.abstractmethod
    def flux_dry_numpy(
        dt: float,
        dz: float,
        w: NDArray,
        s: NDArray,
        su: NDArray,
        sv: NDArray,
    ) -> tuple[NDArray]:
        pass

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy"), stencil="flux_moist")
    @abc.abstractmethod
    def flux_moist_numpy(
        dt: float,
        dz: float,
        w: NDArray,
        sqv: NDArray,
        sqc: NDArray,
        sqr: NDArray,
    ) -> tuple[NDArray]:
        pass

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux_dry")
    @gtscript.function
    @abc.abstractmethod
    def flux_dry_gt4py(
        dt: float,
        dz: float,
        w: gtscript.Field["dtype"],
        s: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
    ) -> tuple[gtscript.Field["dtype"], ...]:
        pass

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux_moist")
    @gtscript.function
    @abc.abstractmethod
    def flux_moist_gt4py(
        dt: float,
        dz: float,
        w: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"],
        sqc: gtscript.Field["dtype"],
        sqr: gtscript.Field["dtype"],
    ) -> tuple[gtscript.Field["dtype"], ...]:
        pass
