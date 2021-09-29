# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.subclasses.horizontal_boundaries.utils import (
    extend_axis,
    repeat_axis,
)
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.register import register


class Periodic(HorizontalBoundary):
    """Periodic boundary conditions."""

    def __init__(self, grid, nb, backend, storage_options):
        nx, ny = grid.nx, grid.ny
        assert nx > 1, (
            "Number of grid points along first dimension should be larger "
            "than 1."
        )
        assert ny > 1, (
            "Number of grid points along second dimension should be larger "
            "than 1."
        )
        assert nb <= nx / 2, "Number of boundary layers cannot exceed ny/2."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(
            grid, nb, backend=backend, storage_options=storage_options
        )

    @property
    def ni(self):
        return self.nx + 2 * self.nb

    @property
    def nj(self):
        return self.ny + 2 * self.nb

    def get_numerical_xaxis(self, dims=None):
        return extend_axis(self.physical_grid.x, self.nb, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return extend_axis(self.physical_grid.x_at_u_locations, self.nb, dims)

    def get_numerical_yaxis(self, dims=None):
        return extend_axis(self.physical_grid.y, self.nb, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return extend_axis(self.physical_grid.y_at_v_locations, self.nb, dims)

    def get_numerical_field(self, field, field_name=None):
        nx, ny, nb = self.nx, self.ny, self.nb
        field_name = field_name or ""
        mx = (
            nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else nx
        )
        mi = mx + 2 * nb
        my = (
            ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else ny
        )

        try:
            li, lj, lk = field.shape
            trg = self.zeros(shape=(li + 2 * nb, lj + 2 * nb, lk))
            src = field
        except ValueError:
            # resort to numpy for 2d arrays
            li, lj = field.shape
            trg = np.zeros(
                (li + 2 * nb, lj + 2 * nb), dtype=self.storage_options.dtype
            )
            src = to_numpy(field)

        trg[nb : mx + nb, nb : my + nb] = src[:mx, :my]
        trg[:nb, nb : my + nb] = trg[nx - 1 : nx - 1 + nb, nb : my + nb]
        trg[mx + nb : mx + 2 * nb, nb : my + nb] = (
            trg[nb + 1 : 2 * nb + 1, nb : my + nb]
            if mx == nx
            else trg[nb + 2 : 2 * nb + 2, nb : my + nb]
        )
        trg[:mi, :nb] = trg[:mi, ny - 1 : ny - 1 + nb]
        trg[:mi, my + nb : my + 2 * nb] = (
            trg[:mi, nb + 1 : 2 * nb + 1]
            if my == ny
            else trg[:mi, nb + 2 : 2 * nb + 2]
        )

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None
    ):
        nx, ny, nb = self.nx, self.ny, self.nb
        field_name = field_name or ""
        mx = (
            nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else nx
        )
        mi = mx + 2 * nb
        my = (
            ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else ny
        )

        field[:nb, nb : my + nb] = field[nx - 1 : nx - 1 + nb, nb : my + nb]
        field[mx + nb : mx + 2 * nb, nb : my + nb] = (
            field[nb + 1 : 2 * nb + 1, nb : my + nb]
            if mx == nx
            else field[nb + 2 : 2 * nb + 2, nb : my + nb]
        )
        field[:mi, :nb] = field[:mi, ny - 1 : ny - 1 + nb]
        field[:mi, my + nb : my + 2 * nb] = (
            field[:mi, nb + 1 : 2 * nb + 1]
            if my == ny
            else field[:mi, nb + 2 : 2 * nb + 2]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None
    ):
        field[0, :] = field[-2, :]
        field[-1, :] = field[1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None
    ):
        field[:, 0] = field[:, -2]
        field[:, -1] = field[:, 1]


class Periodic1DX(HorizontalBoundary):
    """Periodic boundary conditions for a physical grid with ``ny=1``."""

    def __init__(self, grid, nb, backend, storage_options):
        nx, ny = grid.nx, grid.ny
        assert nx > 1, (
            "Number of grid points along first dimension should be larger "
            "than 1."
        )
        assert (
            ny == 1
        ), "Number of grid points along second dimension must be 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed nx/2."

        super().__init__(
            grid, nb, backend=backend, storage_options=storage_options
        )

    @property
    def ni(self):
        return self.nx + 2 * self.nb

    @property
    def nj(self):
        return 2 * self.nb + 1

    def get_numerical_xaxis(self, dims=None):
        return extend_axis(self.physical_grid.x, self.nb, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return extend_axis(self.physical_grid.x_at_u_locations, self.nb, dims)

    def get_numerical_yaxis(self, dims=None):
        return repeat_axis(self.physical_grid.y, self.nb, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return repeat_axis(self.physical_grid.y_at_v_locations, self.nb, dims)

    def get_numerical_field(self, field, field_name=None):
        nx, nb = self.nx, self.nb
        field_name = field_name or ""
        mx = (
            nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else nx
        )
        mi = mx + 2 * nb
        my = (
            2
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else 1
        )

        try:
            li, lj, lk = field.shape
            trg = self.zeros(shape=(li + 2 * nb, lj + 2 * nb, lk))
            src = field
        except ValueError:
            # resort to numpy for 2d arrays
            li, lj = field.shape
            trg = np.zeros(
                (li + 2 * nb, lj + 2 * nb), dtype=self.storage_options.dtype
            )
            src = to_numpy(field)

        trg[nb : mx + nb, nb : my + nb] = src[:mx, :my]
        trg[:nb, nb : my + nb] = trg[nx - 1 : nx - 1 + nb, nb : my + nb]
        trg[mx + nb : mx + 2 * nb, nb : my + nb] = (
            trg[nb + 1 : 2 * nb + 1, nb : my + nb]
            if mx == nx
            else trg[nb + 2 : 2 * nb + 2, nb : my + nb]
        )
        trg[:mi, :nb] = trg[:mi, nb : nb + 1]
        trg[:mi, my + nb : my + 2 * nb] = (
            trg[:mi, nb : nb + 1] if my == 1 else trg[:mi, nb + 1 : nb + 2]
        )

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None
    ):
        nx, ny, nb = self.nx, self.ny, self.nb
        field_name = field_name or ""
        mx = (
            nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else nx
        )
        mi = mx + 2 * nb
        my = (
            2
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else 1
        )

        field[:nb, nb : my + nb] = field[nx - 1 : nx - 1 + nb, nb : my + nb]
        field[mx + nb : mx + 2 * nb, nb : my + nb] = (
            field[nb + 1 : 2 * nb + 1, nb : my + nb]
            if mx == nx
            else field[nb + 2 : 2 * nb + 2, nb : my + nb]
        )
        field[:mi, :nb] = field[:mi, nb : nb + 1]
        field[:mi, my + nb : my + 2 * nb] = (
            field[:mi, nb : nb + 1] if my == 1 else field[:mi, nb + 1 : nb + 2]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None
    ):
        field[0, :] = field[-2, :]
        field[-1, :] = field[1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None
    ):
        field[:, 0] = field[:, -2]
        field[:, -1] = field[:, 1]


class Periodic1DY(HorizontalBoundary):
    """Periodic boundary conditions for a physical grid with ``ny=1``."""

    def __init__(self, grid, nb, backend, storage_options):
        nx, ny = grid.nx, grid.ny
        assert (
            nx == 1
        ), "Number of grid points along first dimension must be 1."
        assert ny > 1, (
            "Number of grid points along second dimension should be larger "
            "than 1."
        )
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(
            grid, nb, backend=backend, storage_options=storage_options
        )

    @property
    def ni(self):
        return 2 * self.nb + 1

    @property
    def nj(self):
        return self.ny + 2 * self.nb

    def get_numerical_xaxis(self, dims=None):
        return repeat_axis(self.physical_grid.x, self.nb, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return repeat_axis(self.physical_grid.x_at_u_locations, self.nb, dims)

    def get_numerical_yaxis(self, dims=None):
        return extend_axis(self.physical_grid.y, self.nb, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return extend_axis(self.physical_grid.y_at_v_locations, self.nb, dims)

    def get_numerical_field(self, field, field_name=None):
        ny, nb = self.ny, self.nb
        field_name = field_name or ""
        mx = (
            2
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else 1
        )
        my = (
            ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else ny
        )
        mj = my + 2 * nb

        try:
            li, lj, lk = field.shape
            trg = self.zeros(shape=(li + 2 * nb, lj + 2 * nb, lk))
            src = field
        except ValueError:
            # resort to numpy for 2d arrays
            li, lj = field.shape
            trg = np.zeros(
                (li + 2 * nb, lj + 2 * nb), dtype=self.storage_options.dtype
            )
            src = to_numpy(field)

        trg[nb : mx + nb, nb : my + nb] = field[:mx, :my]
        trg[nb : mx + nb, :nb] = trg[nb : mx + nb, ny - 1 : ny - 1 + nb]
        trg[nb : mx + nb, my + nb : my + 2 * nb] = (
            trg[nb : mx + nb, nb + 1 : 2 * nb + 1]
            if my == ny
            else trg[nb : mx + nb, nb + 2 : 2 * nb + 2]
        )
        trg[:nb, :mj] = trg[nb : nb + 1, :mj]
        trg[mx + nb : mx + 2 * nb, :mj] = (
            trg[nb : nb + 1, :mj] if mx == 1 else trg[nb + 1 : nb + 2, :mj]
        )

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None
    ):
        ny, nb = self.ny, self.nb
        field_name = field_name or ""
        mx = (
            2
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else 1
        )
        my = (
            ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else ny
        )
        mj = my + 2 * nb

        field[nb : mx + nb, :nb] = field[nb : mx + nb, ny - 1 : ny - 1 + nb]
        field[nb : mx + nb, my + nb : my + 2 * nb] = (
            field[nb : mx + nb, nb + 1 : 2 * nb + 1]
            if my == ny
            else field[nb : mx + nb, nb + 2 : 2 * nb + 2]
        )
        field[:nb, :mj] = field[nb : nb + 1, :mj]
        field[mx + nb : mx + 2 * nb, :mj] = (
            field[nb : nb + 1, :mj] if mx == 1 else field[nb + 1 : nb + 2, :mj]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None
    ):
        field[0, :] = field[-2, :]
        field[-1, :] = field[1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None
    ):
        field[:, 0] = field[:, -2]
        field[:, -1] = field[:, 1]


@register(name="periodic", registry_class=HorizontalBoundary)
def dispatch(
    grid,
    nb,
    backend="numpy",
    backend_options=None,
    storage_shape=None,
    storage_options=None,
):
    """Instantiate the appropriate class based on the grid size."""
    if grid.nx == 1:
        return Periodic1DY(grid, nb, backend, storage_options)
    elif grid.ny == 1:
        return Periodic1DX(grid, nb, backend, storage_options)
    else:
        return Periodic(grid, nb, backend, storage_options)
