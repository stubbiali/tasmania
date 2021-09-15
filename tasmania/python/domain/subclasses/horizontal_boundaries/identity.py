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
import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.subclasses.horizontal_boundaries.utils import (
    change_dims,
    repeat_axis,
)
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.register import register


class Identity(HorizontalBoundary):
    """*Identity* boundary conditions."""

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
        return self.nx

    @property
    def nj(self):
        return self.ny

    def get_numerical_xaxis(self, dims=None):
        return change_dims(self.physical_grid.x, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return change_dims(self.physical_grid.x_at_u_locations, dims)

    def get_numerical_yaxis(self, dims=None):
        return change_dims(self.physical_grid.y, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return change_dims(self.physical_grid.y_at_v_locations, dims)

    def get_numerical_field(self, field, field_name=None):
        return field

    def get_physical_field(self, field, field_name=None):
        return field

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None
    ):
        pass

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None
    ):
        pass

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None
    ):
        pass


class Identity1DX(HorizontalBoundary):
    """*Identity* boundary conditions for a physical grid with ``ny=1``."""

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
        return self.nx

    @property
    def nj(self):
        return 2 * self.nb + 1

    def get_numerical_xaxis(self, dims=None):
        return change_dims(self.physical_grid.x, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return change_dims(self.physical_grid.x_at_u_locations, dims)

    def get_numerical_yaxis(self, dims=None):
        return repeat_axis(self.physical_grid.y, self.nb, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return repeat_axis(self.physical_grid.y_at_v_locations, self.nb, dims)

    def get_numerical_field(self, field, field_name=None):
        try:
            li, lj, lk = field.shape
            trg = self.zeros(shape=(li, lj + 2 * self.nb, lk))
            src = field
        except ValueError:
            # resort to numpy for 2d arrays
            li, lj = field.shape
            trg = np.zeros(
                (li, lj + 2 * self.nb), dtype=self.storage_options.dtype
            )
            src = to_numpy(field)

        trg[:, : self.nb + 1] = src[:, :1]
        trg[:, self.nb + 1 :] = src[:, -1:]

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[:, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None
    ):
        ny, nb = self.ny, self.nb
        field_name = field_name or ""
        my = (
            2
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else 1
        )
        field[:, :nb] = field[:, nb : nb + 1]
        field[:, my + nb : my + 2 * nb] = (
            field[:, nb : nb + 1] if my == 1 else field[:, nb + 1 : nb + 2]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None
    ):
        pass

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None
    ):
        pass


class Identity1DY(HorizontalBoundary):
    """*Identity* boundary conditions for a physical grid with ``nx=1``."""

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
        return self.ny

    def get_numerical_xaxis(self, dims=None):
        return repeat_axis(self.physical_grid.x, self.nb, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return repeat_axis(self.physical_grid.x_at_u_locations, self.nb, dims)

    def get_numerical_yaxis(self, dims=None):
        return change_dims(self.physical_grid.y, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return change_dims(self.physical_grid.y_at_v_locations, dims)

    def get_numerical_field(self, field, field_name=None):
        try:
            li, lj, lk = field.shape
            trg = self.zeros(shape=(li + 2 * self.nb, lj, lk))
            src = field
        except ValueError:
            # resort to numpy for 2d arrays
            li, lj = field.shape
            trg = np.zeros(
                (li + 2 * self.nb, lj), dtype=self.storage_options.dtype
            )
            src = to_numpy(field)

        trg[: self.nb + 1, :] = src[:1, :]
        trg[self.nb + 1 :, :] = src[-1:, :]

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, :]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None
    ):
        nx, nb = self.nx, self.nb
        field_name = field_name or ""
        mx = (
            2
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else 1
        )
        field[:nb, :] = field[nb : nb + 1, :]
        field[mx + nb : mx + 2 * nb, :] = (
            field[nb : nb + 1, :] if mx == 1 else field[nb + 1 : nb + 2, :]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None
    ):
        pass

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None
    ):
        pass


@register(name="identity", registry_class=HorizontalBoundary)
def dispatch(
    grid,
    nb,
    backend="numpy",
    backend_options=None,
    storage_shape=None,
    storage_options=None,
):
    """Instantiate appropriate class based on the grid size."""
    if grid.nx == 1:
        return Identity1DY(grid, nb, backend, storage_options)
    elif grid.ny == 1:
        return Identity1DX(grid, nb, backend, storage_options)
    else:
        return Identity(grid, nb, backend, storage_options)
