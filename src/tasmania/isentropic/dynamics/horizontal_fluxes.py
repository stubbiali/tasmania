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
    from typing import Any, Optional

    from tasmania.utils.typingx import NDArray


class IsentropicHorizontalFlux(AbstractFactory, StencilFactory):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the horizontal numerical fluxes for the three-dimensional
    isentropic dynamical core. The conservative form of the governing
    equations is used.
    """

    # class attributes
    extent: Optional[int] = None
    order: Optional[int] = None
    external_names: Optional[list[str]] = None

    def __init__(self, *, backend: str) -> None:
        super().__init__(backend)

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="flux_dry")
    @abc.abstractmethod
    def flux_dry_numpy(
        dt: float,
        dx: float,
        dy: float,
        s: NDArray,
        u: NDArray,
        v: NDArray,
        su: NDArray,
        sv: NDArray,
        mtg: NDArray,
        s_tnd: Optional[NDArray] = None,
        su_tnd: Optional[NDArray] = None,
        sv_tnd: Optional[NDArray] = None,
        *,
        compute_density_fluxes: bool = True,
        compute_momentum_fluxes: bool = True,
        compute_water_species_fluxes: bool = True,
    ) -> list[NDArray]:
        pass

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="flux_moist")
    @abc.abstractmethod
    def flux_moist_numpy(
        dt: float,
        dx: float,
        dy: float,
        s: NDArray,
        u: NDArray,
        v: NDArray,
        sqv: NDArray,
        sqc: NDArray,
        sqr: NDArray,
        qv_tnd: Optional[NDArray] = None,
        qc_tnd: Optional[NDArray] = None,
        qr_tnd: Optional[NDArray] = None,
        *,
        compute_water_species_fluxes: bool = True,
    ) -> list[NDArray]:
        pass

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux_dry")
    @gtscript.function
    @abc.abstractmethod
    def flux_dry_gt4py(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
        mtg: gtscript.Field["dtype"],
        s_tnd: "Optional[gtscript.Field['dtype']]" = None,
        su_tnd: "Optional[gtscript.Field['dtype']]" = None,
        sv_tnd: "Optional[gtscript.Field['dtype']]" = None,
    ) -> "tuple[gtscript.Field['dtype'], ...]":
        pass

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux_moist")
    @gtscript.function
    @abc.abstractmethod
    def flux_moist_gt4py(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"],
        sqc: gtscript.Field["dtype"],
        sqr: gtscript.Field["dtype"],
        qv_tnd: Optional[gtscript.Field["dtype"]] = None,
        qc_tnd: Optional[gtscript.Field["dtype"]] = None,
        qr_tnd: Optional[gtscript.Field["dtype"]] = None,
    ) -> tuple[gtscript.Field["dtype"], ...]:
        pass


class IsentropicMinimalHorizontalFlux(AbstractFactory, StencilFactory):
    """
    Abstract base class whose derived classes implement different schemes
    to compute the horizontal numerical fluxes for the three-dimensional
    isentropic and *minimal* dynamical core. The conservative form of the
    governing equations is used.
    """

    # class attributes
    extent: int = None
    order: int = None
    externals: dict[str, Any] = None

    def __init__(self, *, backend: str) -> None:
        super().__init__(backend)

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="flux_dry")
    @abc.abstractmethod
    def flux_dry_numpy(
        dt: float,
        dx: float,
        dy: float,
        s: NDArray,
        u: NDArray,
        v: NDArray,
        su: NDArray,
        sv: NDArray,
        mtg: Optional[NDArray] = None,
        s_tnd: Optional[NDArray] = None,
        su_tnd: Optional[NDArray] = None,
        sv_tnd: Optional[NDArray] = None,
        *,
        compute_density_fluxes: bool = True,
        compute_momentum_fluxes: bool = True,
    ) -> list[NDArray]:
        pass

    @staticmethod
    @subroutine_definition(backend=("numpy", "cupy", "numba:cpu:numpy"), stencil="flux_moist")
    @abc.abstractmethod
    def flux_moist_numpy(
        dt: float,
        dx: float,
        dy: float,
        s: NDArray,
        u: NDArray,
        v: NDArray,
        sqv: NDArray,
        sqc: NDArray,
        sqr: NDArray,
        qv_tnd: Optional[NDArray] = None,
        qc_tnd: Optional[NDArray] = None,
        qr_tnd: Optional[NDArray] = None,
    ) -> list[NDArray]:
        pass

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux_dry")
    @gtscript.function
    @abc.abstractmethod
    def flux_dry_gt4py(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        su: gtscript.Field["dtype"],
        sv: gtscript.Field["dtype"],
        mtg: Optional[gtscript.Field["dtype"]] = None,
        s_tnd: Optional[gtscript.Field["dtype"]] = None,
        su_tnd: Optional[gtscript.Field["dtype"]] = None,
        sv_tnd: Optional[gtscript.Field["dtype"]] = None,
    ) -> tuple[gtscript.Field["dtype"], ...]:
        pass

    @staticmethod
    @subroutine_definition(backend="gt4py*", stencil="flux_moist")
    @gtscript.function
    @abc.abstractmethod
    def flux_moist_gt4py(
        dt: float,
        dx: float,
        dy: float,
        s: gtscript.Field["dtype"],
        u: gtscript.Field["dtype"],
        v: gtscript.Field["dtype"],
        sqv: gtscript.Field["dtype"],
        sqc: gtscript.Field["dtype"],
        sqr: gtscript.Field["dtype"],
        qv_tnd: Optional[gtscript.Field["dtype"]] = None,
        qc_tnd: Optional[gtscript.Field["dtype"]] = None,
        qr_tnd: Optional[gtscript.Field["dtype"]] = None,
    ) -> tuple[gtscript.Field["dtype"], ...]:
        pass
