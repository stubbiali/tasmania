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
import abc
from typing import Any, Dict, Mapping, Optional, Sequence, TYPE_CHECKING

from tasmania.python.utils.data import get_physical_constants

if TYPE_CHECKING:
    from sympl._core.typingx import DataArray

    from tasmania.python.domain.domain import Domain
    from tasmania.python.domain.grid import Grid
    from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
    from tasmania.python.utils.typingx import TripletInt


class PhysicalConstantsComponent(abc.ABC):
    default_physical_constants = {}

    def __init__(
        self: "PhysicalConstantsComponent",
        physical_constants: Mapping[str, "DataArray"],
    ) -> None:
        self.rpc = get_physical_constants(self.default_physical_constants, physical_constants)

    @property
    def raw_physical_constants(
        self: "PhysicalConstantsComponent",
    ) -> Dict[str, float]:
        return self.rpc.copy()

    @raw_physical_constants.setter
    def raw_physical_constants(self: "PhysicalConstantsComponent", value: Any) -> None:
        raise RuntimeError()


class GridComponent(abc.ABC):
    """A component built over a :class:`~tasmania.Grid`."""

    def __init__(self: "GridComponent", grid: "Grid") -> None:
        self._grid = grid

    @property
    def grid(self: "GridComponent") -> "Grid":
        """The underlying :class:`~tasmania.Grid`."""
        return self._grid

    def get_field_grid_shape(self, name: str) -> "TripletInt":
        if "at_uv_locations" in name:
            ni = self.grid.nx + 1
            nj = self.grid.ny + 1
        elif "at_u_locations" in name:
            ni = self.grid.nx + 1
            nj = self.grid.ny
        elif "at_v_locations" in name:
            ni = self.grid.nx
            nj = self.grid.ny + 1
        else:
            ni = self.grid.nx
            nj = self.grid.ny

        if "at_surface_level" in name:
            nk = 1
        elif "on_interface_levels" in name:
            nk = self.grid.nz + 1
        else:
            nk = self.grid.nz

        return ni, nj, nk

    def get_field_storage_shape(
        self, name: str, default_storage_shape: "TripletInt"
    ) -> "TripletInt":
        grid_shape = self.get_field_grid_shape(name)
        return self.get_shape(default_storage_shape, min_shape=grid_shape)

    def get_storage_shape(
        self,
        shape: Sequence[int],
        min_shape: Optional[Sequence[int]] = None,
        max_shape: Optional[Sequence[int]] = None,
    ) -> Sequence[int]:
        min_shape = min_shape or (self.grid.nx, self.grid.ny, self.grid.nz)
        return self.get_shape(shape, min_shape, max_shape)

    @staticmethod
    def get_shape(
        in_shape: Sequence[int],
        min_shape: Sequence[int],
        max_shape: Optional[Sequence[int]] = None,
    ) -> Sequence[int]:
        out_shape = in_shape or min_shape

        if max_shape is None:
            # error_msg = (
            #     f"storage shape must be larger or equal than "
            #     f"({', '.join(str(el) for el in min_shape)})."
            # )
            # assert all(
            #     tuple(
            #         out_shape[i] >= min_shape[i] for i in range(len(min_shape))
            #     )
            # ), error_msg
            out_shape = [a if a >= b else b for a, b in zip(out_shape, min_shape)]
        else:
            # error_msg = (
            #     f"storage shape must be between "
            #     f"({', '.join(str(el) for el in min_shape)}) and "
            #     f"({', '.join(str(el) for el in max_shape)})."
            # )
            # assert all(
            #     tuple(
            #         min_shape[i] <= out_shape[i] <= max_shape[i]
            #         for i in range(len(min_shape))
            #     )
            # ), error_msg
            out_shape = [
                a if c >= a >= b else (b if a < b else c)
                for a, b, c in zip(out_shape, min_shape, max_shape)
            ]

        return out_shape


class DomainComponent(GridComponent, abc.ABC):
    """A component built over a :class:`~tasmania.Domain`."""

    allowed_grid_types = ("numerical", "physical")

    def __init__(self: "DomainComponent", domain: "Domain", grid_type: str) -> None:
        assert grid_type in self.allowed_grid_types, (
            f"grid_type is {grid_type}, but either "
            f"({', '.join(self.allowed_grid_types)}) was expected."
        )
        super().__init__(domain.physical_grid if grid_type == "physical" else domain.numerical_grid)
        self._grid_type = grid_type
        self._hb = domain.horizontal_boundary

    @property
    def grid_type(self: "DomainComponent") -> str:
        """The grid type, either "physical" or "numerical"."""
        return self._grid_type

    @property
    def horizontal_boundary(self: "DomainComponent") -> "HorizontalBoundary":
        """The object handling the lateral boundary conditions."""
        return self._hb
